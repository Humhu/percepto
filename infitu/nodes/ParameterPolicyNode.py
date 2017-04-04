#!/usr/bin/env python

import numpy as np
import rospy
import poli
import optim
import broadcast
import infitu
from threading import Lock

from percepto_msgs.msg import RewardStamped


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
        self.policy_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                        output_dim=output_dim,
                                                        info=rospy.get_param('~'))

        # Parse learner
        learner_info = rospy.get_param('~learning/gradient_estimation')
        self.learner = poli.parse_gradient_estimator(learner_info,
                                                     policy=self.policy)

        # Parse optimizer
        optimizer_info = rospy.get_param('~learning/optimizer')
        optimizer_info['mode'] = 'max'
        self.optimizer = optim.parse_optimizers(optimizer_info)

        # Parse evaluator
        recorder_info = rospy.get_param('~learning/sar_recorder')
        reward_topic = recorder_info.pop('reward_topic')
        self.recorder = poli.SARRecorder(**recorder_info)

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

        policy_rate = rospy.get_param('~policy_rate')
        learn_rate = rospy.get_param('~learn_rate')
        self._policy_timer = rospy.Timer(period=rospy.Duration(1.0 / policy_rate),
                                         callback=self.action_spin)
        self._learn_timer = rospy.Timer(period=rospy.Duration(1.0 / learn_rate),
                                        callback=self.learn_spin)
        self._reward_sub = rospy.Subscriber(reward_topic,
                                            RewardStamped,
                                            self.reward_callback)

    def action_spin(self, event):

        now = rospy.Time.now()
        stamp, raw_state = self.stream_rx.read_stream(now,
                                                      mode='closest_before')
        if stamp is None or raw_state is None:
            rospy.logwarn('Could not read input stream at time %s', str(now))

        self.policy_lock.acquire()
        state = self.policy_wrapper.process_input(raw_state)
        if state is None:
            rospy.loginfo('Input normalizer not ready yet')
            self.policy_lock.release()
            return
        action = self.policy_wrapper.sample_action(state, proc_input=False)
        self.interface.set_values(action)

        # Output printout
        action_mean = self.policy.mean
        action_cov = self.policy.cov
        msg = 'State: %s\nAction: %s\n' % (
            np.array_str(state), np.array_str(action))
        msg += 'Mean: %s\nCov:\n%s' % (np.array_str(action_mean),
                                       np.array_str(action_cov))
        rospy.loginfo(msg)

        if self.action_tx is not None:
            self.action_tx.publish(time=now, feats=action)

        self.recorder.report_state_action(time=now.to_sec(),
                                          state=state,
                                          action=action)
        self.policy_lock.release()

    def reward_callback(self, msg):
        self.recorder.report_reward(time=msg.header.stamp.to_sec(),
                                    reward=msg.reward)

    def learn_spin(self, event):
        self.policy_lock.acquire()

        rospy.loginfo('Processing experience buffer...')
        sar_tuples = self.recorder.process_buffers()
        for t, s, a, r in sar_tuples:
            rospy.loginfo('Experience state: %s action: %s reward: %f',
                          np.array_str(s), np.array_str(a), r)
            self.learner.report_episode(states=[s], actions=[a], rewards=[r])

        rospy.loginfo('Have %d SAR samples', self.learner.num_samples)

        # NOTE Don't have to lock policy since ROS timer callbacks are
        # all single threaded?
        theta_init = self.policy.get_theta()
        theta = self.optimizer.step(x_init=theta_init,
                                    func=self.learner.estimate_reward_and_gradient)
        self.policy.set_theta(theta)
        rospy.loginfo('Parameter A:\n%s', np.array_str(self.policy.A))
        rospy.loginfo('Parameter B:\n%s', np.array_str(self.policy.B))
        self.policy_lock.release()

    @property
    def policy(self):
        return self.policy_wrapper.policy

if __name__ == '__main__':
    rospy.init_node('policy_gradient_node')

    node = ParameterPolicyNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
