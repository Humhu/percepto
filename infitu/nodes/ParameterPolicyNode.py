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

        # Parse policy
        self.policy_lock = Lock()
        policy_info = rospy.get_param('~policy')
        stream_name = policy_info.pop('input_stream')
        self.stream_rx = broadcast.Receiver(stream_name)
        # TODO Make homogeneous augmentation optional?
        policy_info['input_dim'] = self.stream_rx.stream_feature_size + 1
        policy_info['output_dim'] = self.interface.num_parameters
        self.policy = poli.parse_policy(policy_info)

        # Parse learner
        learner_info = rospy.get_param('~gradient_estimation')
        self.learner = poli.parse_learner(learner_info, self.policy)

        # Parse optimizer
        optimizer_info = rospy.get_param('~optimizer')
        optimizer_info['mode'] = 'max'
        self.optimizer = optim.parse_optimizers(optimizer_info)

        # Parse evaluator
        evaluator_info = rospy.get_param('~evaluator')
        reward_topic = evaluator_info.pop('reward_topic')
        self.evaluator = poli.DelayedBanditEvaluator(**evaluator_info)

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
        self.policy_lock.acquire()

        now = rospy.Time.now()
        stamp, state = self.stream_rx.read_stream(now, mode='closest_before')
        if stamp is None or state is None:
            rospy.logwarn('Could not read input stream at time %s', str(now))

        # Homogeneous coordinates
        state = np.hstack((state, 1))
        action = self.policy.sample_action(state)
        self.interface.set_values(action)

        self.evaluator.report_state_action(time=now.to_sec(),
                                           state=state,
                                           action=action)
        self.policy_lock.release()

    def reward_callback(self, msg):
        self.evaluator.report_reward(time=msg.header.stamp.to_sec(),
                                     reward=msg.reward)

    def learn_spin(self, event):
        self.policy_lock.acquire()
        
        rospy.loginfo('Processing experience buffer...')
        sar_tuples = self.evaluator.process_buffers()
        for s, a, r in sar_tuples:
            rospy.loginfo('Experience state: %s action: %s reward: %f',
                          np.array_str(s), np.array_str(a), r)
            self.learner.report_sample(state=s, action=a, reward=r)

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

if __name__ == '__main__':
    rospy.init_node('bandit_policy_gradient')

    node = ParameterPolicyNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
