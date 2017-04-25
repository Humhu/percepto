#!/usr/bin/env python
import cPickle as pickle
import numpy as np
import sys
import rospy
import poli
import optim
import broadcast
import infitu
from threading import Lock
from itertools import izip

from percepto_msgs.msg import RewardStamped, EpisodeBreak


class ParameterPPGENode(object):
    """Continually improves a policy with PPGE.
    """

    def __init__(self):

        # Parse parameter interface
        interface_info = rospy.get_param('~interface')
        self.interface = infitu.parse_interface(interface_info)
        output_dim = self.interface.num_parameters

        stream_name = rospy.get_param('~input_stream')
        self.stream_rx = broadcast.Receiver(stream_name)
        raw_input_dim = self.stream_rx.stream_feature_size

        # Parse deterministic policy
        self.policy_lock = Lock()
        self._policy_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                         output_dim=output_dim,
                                                         info=rospy.get_param('~deterministic_policy'))
        param_dim = len(self._policy_wrapper.policy.get_theta())

        self.ppge_lock = Lock()
        self._active_ppge = poli.parse_policy_wrapper(raw_input_dim=0,
                                                      output_dim=param_dim,
                                                      info=rospy.get_param('~ppge'))
        self._learner_ppge = poli.parse_policy_wrapper(raw_input_dim=0,
                                                       output_dim=param_dim,
                                                       info=rospy.get_param('~ppge'))

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
        learning_info = rospy.get_param('~learning')

        self.remove_traj_ll = float(learning_info['traj_remove_logprob'])

        grad_info = learning_info['gradient_estimation']
        self.grad_est = poli.parse_gradient_estimator(spec=grad_info,
                                                      policy=self._learner_ppge.policy)

        # Parse optimizer
        optimizer_info = learning_info['optimizer']
        optimizer_info['mode'] = 'max'
        self.optimizer = optim.parse_optimizers(optimizer_info)

        # Parse evaluator
        recorder_info = learning_info['episode_recording']
        self.recorder = poli.EpisodeRecorder(**recorder_info)

        learn_rate = rospy.get_param('~learn_rate')
        # self._learn_timer = rospy.Timer(period=rospy.Duration(1.0 / learn_rate),
        #                                callback=self.learn_spin)
        self._reward_sub = rospy.Subscriber(learning_info['reward_topic'],
                                            RewardStamped,
                                            self.reward_callback)
        self._break_sub = rospy.Subscriber(learning_info['break_topic'],
                                           EpisodeBreak,
                                           self.break_callback)

        self._max_eps = rospy.get_param('~max_episodes', float('inf'))
        self._eps_counter = 0

        self.actions = []
        self.sum_returns = []
        out_file = rospy.get_param('~out_file')
        self._out_file = open(out_file, 'w')

        self.sample_policy(rospy.Time.now().to_sec())

    def action_spin(self, event):

        now, raw_state = self.stream_rx.read_stream(event.current_real,
                                                    mode='closest_before')
        if now is None or raw_state is None:
            rospy.logwarn('Could not read input stream at time %s',
                          str(event.current_real))
            return

        with self.policy_lock:
            state = self._policy_wrapper.process_input(raw_state)
            if state is None:
                rospy.loginfo('Input normalizer not ready yet')
                return
            action = self._policy_wrapper.sample_action(
                state, proc_input=False)

        self.interface.set_values(action)

        # Output printout
        # msg = 'State: %s\nAction: %s\n' % (np.array_str(state),
        #                                    np.array_str(action))
        # rospy.loginfo(msg)

        if self.action_tx is not None:
            self.action_tx.publish(time=event.current_real, feats=action)

    def reward_callback(self, msg):
        self.recorder.report_reward(time=msg.header.stamp.to_sec(),
                                    reward=msg.reward)

    def break_callback(self, msg):
        self.sample_policy(msg.break_time.to_sec())

    def sample_policy(self, t):
        with self.ppge_lock:
            new_params = self._active_ppge.sample_action(state=np.empty(0))
            new_logprob = self._active_ppge.policy.logprob(state=np.empty(0),
                                                           action=new_params)

        with self.policy_lock:
            self._policy_wrapper.policy.set_theta(new_params)
            print 'New A: %s' % np.array_str(self._policy_wrapper.policy.A)
            self.actions.append(self._policy_wrapper.policy.A)

        # Report policy trial
        self.recorder.report_state_action(time=t,
                                          state=np.empty(0),
                                          action=new_params,
                                          logprob=new_logprob)
        self.recorder.report_episode_break(time=t)

    def learn_spin(self):

        for ep in self.recorder.process_episodes():
            if len(ep) == 0:
                continue
            rospy.loginfo(print_episode(ep))

            self._eps_counter += 1

            _, states, actions, logprobs, rewards = izip(*ep)
            # TODO Different reward normalization modes
            rewards = np.sum(rewards)
            self.sum_returns.append(rewards)

            self.grad_est.report_episode(states=states,
                                         actions=actions,
                                         rewards=[rewards],
                                         logprobs=logprobs)

        rospy.loginfo('Have %d episodes', self.grad_est.num_samples)

        init_gamma = self._learner_ppge.policy.get_theta()
        gamma, reward = self.optimizer.step(x_init=init_gamma,
                                            func=self.grad_est.estimate_reward_and_gradient)
        if np.any(init_gamma != gamma):
            self._learner_ppge.policy.set_theta(gamma)
            with self.policy_lock:
                self._active_ppge.policy.set_theta(gamma)

            self.grad_est.update_buffer()
            self.grad_est.remove_unlikely_trajectories(self.remove_traj_ll)

            rospy.loginfo('Estimated reward: %f', reward)
            rospy.loginfo('New mean:\n%s', np.array_str(self._learner_ppge.policy.mean))
            rospy.loginfo('New SD:\n%s', np.array_str(self._learner_ppge.policy.sds))

        if self._eps_counter >= self._max_eps:
            rospy.loginfo('Max episodes achieved.')
            pickle.dump((self.actions, self.sum_returns), self._out_file)
            sys.exit(0)


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

    node = ParameterPPGENode()
    try:
        while not rospy.is_shutdown():
            node.learn_spin()
            rospy.sleep(rospy.Duration(1.0))
    except rospy.ROSInterruptException:
        pass
