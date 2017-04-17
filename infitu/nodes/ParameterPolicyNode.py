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

import multiprocessing
import Queue

from percepto_msgs.msg import RewardStamped, EpisodeBreak


class ParameterLearner(object):
    def __init__(self, raw_input_dim, output_dim, sa_queue, param_queue):

        self.sa_queue = sa_queue
        self.param_queue = param_queue

        self._wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                  output_dim=output_dim,
                                                  info=rospy.get_param('~'))
        # TODO HACK init
        self.policy.B[:] = -1

        learning_info = rospy.get_param('~learning')
        self.traj_remove_logprob = float(
            learning_info['traj_remove_logprob'])

        regularizer = None
        if 'regularization' in learning_info:
            reg_info = learning_info['regularization']
            regularizer = poli.parse_regularizer(policy=self.policy,
                                                 spec=reg_info)

        grad_info = learning_info['gradient_estimation']
        self.grad_est = poli.parse_gradient_estimator(spec=grad_info,
                                                      policy=self.policy,
                                                      regularizer=regularizer)

        # Parse optimizer
        optimizer_info = learning_info['optimizer']
        optimizer_info['mode'] = 'max'
        self.optimizer = optim.parse_optimizers(optimizer_info)

        # Parse evaluator
        recorder_info = learning_info['episode_recording']
        self.recorder = poli.EpisodeRecorder(**recorder_info)

        self._max_eps = rospy.get_param('~max_episodes')
        self._eps_counter = 0

        self.sum_returns = []
        out_file = rospy.get_param('~out_file')
        self._out_file = open(out_file, 'w')

        learn_rate = rospy.get_param('~learn_rate')
        self.learn_rate = rospy.Rate(learn_rate)

    @property
    def policy(self):
        return self._wrapper.policy

    def learn_spin(self):
        self.learn_rate.sleep()

        try:
            while True:
                msg_type, data = self.sa_queue.get_nowait()
                if msg_type == 'sa':
                    stamp, state, action, logprob = data
                    self.recorder.report_state_action(time=stamp,
                                                    state=state,
                                                    action=action,
                                                    logprob=logprob)
                elif msg_type == 'r':
                    stamp, reward = data
                    self.recorder.report_reward(time=stamp,
                                                reward=reward)
                elif msg_type == 'break':
                    stamp = data
                    self.recorder.report_episode_break(time=stamp)
                else:
                    raise ValueError('Unknown message type: %s' % msg_type)

        except Queue.Empty:
            pass

        for ep in self.recorder.process_episodes():
            if len(ep) == 0:
                continue
            rospy.loginfo(print_episode(ep))

            self._eps_counter += 1

            _, states, actions, logprobs, rewards = izip(*ep)
            rewards = np.array(rewards) / len(ep)
            self.sum_returns.append(np.sum(rewards))

            self.grad_est.report_episode(states=states,
                                         actions=actions,
                                         rewards=rewards,
                                         logprobs=logprobs)

        rospy.loginfo('Have %d episodes', self.grad_est.num_samples)
        rospy.loginfo('Beginning optimization...')

        init_theta = self.policy.get_theta()
        theta, reward = self.optimizer.optimize(x_init=init_theta,
                                                func=self.grad_est.estimate_reward_and_gradient)
        if np.any(init_theta != theta):
            self.policy.set_theta(theta)
            self.param_queue.put(theta)

            self.grad_est.update_buffer()
            self.grad_est.remove_unlikely_trajectories(
                self.traj_remove_logprob)

            rospy.loginfo('Estimated reward: %f', reward)
            rospy.loginfo('New A:\n%s', np.array_str(self.policy.A))
            rospy.loginfo('New B:\n%s', np.array_str(self.policy.B))

        if self._eps_counter >= self._max_eps:
            AB = (self.policy.A, self.policy.B)
            pickle.dump((AB, self.sum_returns), self._out_file)
            rospy.signal_shutdown('Max episodes achieved.')


class ParameterPolicyNode(object):
    """Continually improves a bandit policy with bandit policy gradient.
    """

    def __init__(self, sa_queue, param_queue):

        # Parse parameter interface
        interface_info = rospy.get_param('~interface')
        self.interface = infitu.parse_interface(interface_info)
        output_dim = self.interface.num_parameters

        stream_name = rospy.get_param('~input_stream')
        self.stream_rx = broadcast.Receiver(stream_name)
        raw_input_dim = self.stream_rx.stream_feature_size

        # Parse policy
        self.policy_lock = Lock()
        self._wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                  output_dim=output_dim,
                                                  info=rospy.get_param('~'))
        self.policy.B[:] = -1

        policy_rate = rospy.get_param('~policy_rate')
        self._policy_timer = rospy.Timer(period=rospy.Duration(1.0 / policy_rate),
                                         callback=self.action_spin)

        learning_info = rospy.get_param('~learning')
        self._reward_sub = rospy.Subscriber(learning_info['reward_topic'],
                                            RewardStamped,
                                            self.reward_callback)
        self._break_sub = rospy.Subscriber(learning_info['break_topic'],
                                           EpisodeBreak,
                                           self.break_callback)

        self.sa_queue = sa_queue
        self.param_queue = param_queue

    @property
    def raw_input_dim(self):
        return self.stream_rx.stream_feature_size

    @property
    def output_dim(self):
        return self.interface.num_parameters

    @property
    def policy(self):
        return self._wrapper.policy

    def reward_callback(self, msg):
        # self.recorder.report_reward(time=msg.header.stamp.to_sec(),
                                    # reward=msg.reward)
        self.sa_queue.put(('r', (msg.header.stamp.to_sec(), msg.reward)))

    def break_callback(self, msg):
        self.sa_queue.put(('break', msg.break_time.to_sec()))
        #self.recorder.report_episode_break(time=msg.break_time.to_sec())

    def action_spin(self, event):

        try:
            theta = self.param_queue.get_nowait()
            self.policy.set_theta(theta)
        except Queue.Empty:
            pass

        #now = event.current_real
        now, raw_state = self.stream_rx.read_stream(event.current_real,
                                                    mode='closest_before')
        if now is None or raw_state is None:
            rospy.logwarn('Could not read input stream at time %s',
                          str(event.current_real))
            return

        self.policy_lock.acquire()
        state = self._wrapper.process_input(raw_state)
        if state is None:
            rospy.loginfo('Input normalizer not ready yet')
            self.policy_lock.release()
            return

        action = self._wrapper.sample_action(state, proc_input=False)
        logprob = self.policy.logprob(state, action)
        action_mean = self.policy.mean
        action_cov = self.policy.cov
        self.policy_lock.release()

        self.interface.set_values(action)

        self.sa_queue.put(('sa', (event.current_real.to_sec(), state, action, logprob)))

        # Output printout
        msg = 'State: %s\nAction: %s\n' % (np.array_str(state),
                                           np.array_str(action))
        msg += 'Mean: %s\nSD:\n%s' % (np.array_str(action_mean),
                                      np.array_str(np.sqrt(np.diag(action_cov))))
        # rospy.loginfo(msg)

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

    sa_queue = multiprocessing.Queue()
    param_queue = multiprocessing.Queue(maxsize=1)

    policy_node = ParameterPolicyNode(sa_queue=sa_queue,
                                      param_queue=param_queue)
    learner_node = ParameterLearner(raw_input_dim=policy_node.raw_input_dim,
                                    output_dim=policy_node.output_dim,
                                    sa_queue=sa_queue,
                                    param_queue=param_queue)

    def learn_spin():
        while not rospy.is_shutdown():
            learner_node.learn_spin()

    learner = multiprocessing.Process(target=learn_spin)
    learner.start()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    learner.join()