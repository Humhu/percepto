#!/usr/bin/env python

import numpy as np
import rospy
import poli
import optim
import broadcast
import infitu
from threading import Lock
from itertools import izip

from percepto_msgs.msg import RewardStamped, EpisodeBreak


class ParameterPolicyNode(object):
    """Continually improves a bandit policy with bandit policy gradient.
    """

    def __init__(self):

        # Parse parameter interface
        interface_info = rospy.get_param('~interface')
        self.interface = infitu.parse_interface(interface_info)
        output_dim = self.interface.num_parameters

        stream_name = rospy.get_param('~input_stream')
        self.stream_rx = broadcast.Receiver(stream_name)
        raw_input_dim = self.stream_rx.stream_feature_size

        # Parse policy
        self.policy_lock = Lock()
        self._active_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                         output_dim=output_dim,
                                                         info=rospy.get_param('~'))
        self.active_policy.B[:, -1] = -1

        policy_rate = rospy.get_param('~policy_rate')
        self._policy_timer = rospy.Timer(period=rospy.Duration(1.0 / policy_rate),
                                         callback=self.action_spin)

        # Parse action broadcaster
        if rospy.has_param('~action_broadcast'):
            tx_info = rospy.get_param('~action_broadcast')
            stream_name = tx_info['stream_name']
            self.action_tx = broadcast.Transmitter(stream_name=stream_name,
                                                   feature_size=output_dim,
                                                   description='Policy normalized output',
                                                   namespace='~actions')
        else:
            self.action_tx = None

        # Parse learner
        self.recorder = None
        if rospy.has_param('~learning'):
            # Create a duplicate policy to use for gradient estimation asynchronously
            # NOTE The learner wrapper should never process inputs/outputs, since it would
            # diverge from the active wrapper! We use only its policy member.
            self._learner_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                              output_dim=output_dim,
                                                              info=rospy.get_param('~'))
            # TODO HACK init
            self.learner_policy.B[:, -1] = -1

            learning_info = rospy.get_param('~learning')

            regularizer = None
            if 'regularization' in learning_info:
                reg_info = learning_info['regularization']
                regularizer = poli.parse_regularizer(policy=self.learner_policy,
                                                     spec=reg_info)

            grad_info = learning_info['gradient_estimation']
            self.grad_est = poli.parse_gradient_estimator(spec=grad_info,
                                                          policy=self.learner_policy,
                                                          regularizer=regularizer)

            # Parse optimizer
            optimizer_info = learning_info['optimizer']
            optimizer_info['mode'] = 'max'
            self.optimizer = optim.parse_optimizers(optimizer_info)

            # Parse evaluator
            recorder_info = learning_info['episode_recording']
            self.recorder = poli.EpisodeRecorder(**recorder_info)

            learn_rate = rospy.get_param('~learn_rate')
            self._learn_timer = rospy.Timer(period=rospy.Duration(1.0 / learn_rate),
                                            callback=self.learn_spin)
            self._reward_sub = rospy.Subscriber(learning_info['reward_topic'],
                                                RewardStamped,
                                                self.reward_callback)
            self._break_sub = rospy.Subscriber(learning_info['break_topic'],
                                               EpisodeBreak,
                                               self.break_callback)

    @property
    def active_policy(self):
        return self._active_wrapper.policy

    @property
    def learner_policy(self):
        return self._learner_wrapper.policy

    def action_spin(self, event):

        #now = event.current_real
        now, raw_state = self.stream_rx.read_stream(event.current_real,
                                                    mode='closest_before')
        if now is None or raw_state is None:
            rospy.logwarn('Could not read input stream at time %s',
                          str(event.current_real))
            return

        self.policy_lock.acquire()
        state = self._active_wrapper.process_input(raw_state)
        if state is None:
            rospy.loginfo('Input normalizer not ready yet')
            self.policy_lock.release()
            return

        action = self._active_wrapper.sample_action(state, proc_input=False)
        logprob = self.active_policy.logprob(state, action)
        action_mean = self.active_policy.mean
        action_cov = self.active_policy.cov
        self.policy_lock.release()

        self.interface.set_values(action)

        # Output printout
        msg = 'State: %s\nAction: %s\n' % (np.array_str(state),
                                           np.array_str(action))
        msg += 'Mean: %s\nCov:\n%s' % (np.array_str(action_mean),
                                       np.array_str(action_cov))
        rospy.loginfo(msg)

        if self.action_tx is not None:
            self.action_tx.publish(time=now, feats=action)

        if self.recorder is not None:
            self.recorder.report_state_action(time=now.to_sec(),
                                              state=state,
                                              action=action,
                                              logprob=logprob)

    def reward_callback(self, msg):
        self.recorder.report_reward(time=msg.header.stamp.to_sec(),
                                    reward=msg.reward)

    def break_callback(self, msg):
        self.recorder.report_episode_break(time=msg.break_time.to_sec())

    def learn_spin(self, event):

        rospy.loginfo('Processing timestep buffer...')
        for ep in self.recorder.process_episodes():
            if len(ep) == 0:
                continue
            rospy.loginfo(print_episode(ep))
            _, states, actions, logprobs, rewards = izip(*ep)
            # rewards = np.array(rewards) / len(ep)
            self.grad_est.report_episode(states=states,
                                         actions=actions,
                                         rewards=rewards,
                                         logprobs=logprobs)

        rospy.loginfo('Have %d episodes', self.grad_est.num_samples)
        rospy.loginfo('Beginning optimization...')
        theta, reward = self.optimizer.optimize(x_init=self.learner_policy.get_theta(),
                                                func=self.grad_est.estimate_reward_and_gradient)
        self.learner_policy.set_theta(theta)
        if reward is not None:
            rospy.loginfo('Estimated reward: %f', reward)
            rospy.loginfo('New A:\n%s', np.array_str(self.learner_policy.A))
            rospy.loginfo('New B:\n%s', np.array_str(self.learner_policy.B))

        self.policy_lock.acquire()
        self.active_policy.set_theta(theta)
        self.policy_lock.release()


def print_episode(ep):
    msg = 'Episode:\n'
    sum_r = 0
    for t, s, a, l, r in ep:
        msg += 'S: %s A:%s R: %f\n' % (np.array_str(s), np.array_str(a), r)
        sum_r += r
    #msg += 'Sum R: %f Avg R: %f' % (sum_r, sum_r / len(ep))
    return msg


if __name__ == '__main__':
    rospy.init_node('policy_gradient_node')

    node = ParameterPolicyNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
