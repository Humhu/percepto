#!/usr/bin/env python

import numpy as np
import rospy
from percepto_msgs.msg import RewardStamped
import broadcast
import paraset


class TestBanditProblem(object):
    """Provides a test parameter bandit problem.
    """
    def __init__(self):
        self.dim = rospy.get_param('~dim')
        self.params = []
        for i in range(self.dim):
            param = paraset.RuntimeParamGetter(param_type=float,
                                               name='param%d' % i,
                                               init_val=0,
                                               description='Test problem input %d' % i)
            self.params.append(param)

        self.err_cutoff = rospy.get_param('~err_cutoff')
        self.action_bias = np.random.uniform(-0.5, 0.5, self.dim)
        rospy.loginfo('Action bias: %s', np.array_str(self.action_bias))

        # Set up state generator and transmitter
        self.state = None
        self.state_tx = broadcast.Transmitter(stream_name='test_bandit_state',
                                              feature_size=self.dim,
                                              description='Test bandit problem context/state',
                                              mode='push',
                                              queue_size=10)
        self.__update_state(rospy.Time.now())

        self.rew_pub = rospy.Publisher('reward', RewardStamped, queue_size=10)
        update_rate = rospy.get_param('~update_rate')
        self.reward_timer = rospy.Timer(rospy.Duration(1.0 / update_rate),
                                        self.timer_callback)

    def timer_callback(self, event):
        # Compute the reward before updating state
        action = np.array([p.value for p in self.params])

        msg = RewardStamped()
        msg.header.stamp = event.current_real
        msg.reward = self.__compute_reward(action)
        self.rew_pub.publish(msg)

        self.__update_state(event.current_real)

    def __compute_reward(self, action):
        err = np.linalg.norm(self.state - (action - self.action_bias))
        if err > self.err_cutoff:
            upper = -self.err_cutoff
        else:
            upper = 0
        return np.random.uniform(low=-err, high=upper)

    def __update_state(self, time):
        self.state = np.random.uniform(low=-1, high=1, size=self.dim)
        self.state_tx.publish(time=time, feats=self.state)

if __name__ == '__main__':
    rospy.init_node('test_bandit_problem')
    prob = TestBanditProblem()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass