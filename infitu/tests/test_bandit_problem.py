#!/usr/bin/env python

import numpy as np
import rospy
from percepto_msgs.msg import RewardStamped
from percepto_msgs.srv import GetCritique, GetCritiqueResponse
import broadcast
import paraset


class TestBanditProblem(object):
    """Provides a test bandit problem with the GetCritique interface.
    """

    def __init__(self):
        self.dim = rospy.get_param('~dim')
        self.params = []

        self._critique_server = rospy.Service('~get_critique',
                                              GetCritique,
                                              self.critique_callback)

        self.err_cutoff = rospy.get_param('~err_cutoff')
        self.action_bias = np.random.uniform(-0.5, 0.5, self.dim)
        rospy.loginfo('Action bias: %s', np.array_str(self.action_bias))

        # Set up state generator and transmitter
        self.state = None
        self.state_tx = broadcast.Transmitter(stream_name='next_bandit_state',
                                              feature_size=self.dim,
                                              description='Test bandit problem context/state',
                                              mode='pull',
                                              queue_size=10)
        self.__update_state()

    def critique_callback(self, req):
        # Compute the reward before updating state
        action = np.array(req.input)

        res = GetCritiqueResponse()
        res.critique = self.__compute_reward(action)

        # NOTE Hopefully this is enough delay for the receiver to order correctly
        self.__update_state()
        return res

    def __compute_reward(self, action):
        err = np.linalg.norm(self.state - (action - self.action_bias))
        if err > self.err_cutoff:
            upper = -self.err_cutoff
        else:
            upper = 0
        reward = np.random.uniform(low=-err, high=upper)
        print 'State: %s Action: %s Reward: %f' % (np.array_str(self.state),
                                                   np.array_str(action),
                                                   reward)
        return reward

    def __update_state(self):
        self.state = np.random.uniform(low=-1, high=1, size=self.dim)
        print 'Next state: %s' % np.array_str(self.state)
        self.state_tx.publish(time=rospy.Time.now(), feats=self.state)

if __name__ == '__main__':
    rospy.init_node('test_bandit_problem')
    prob = TestBanditProblem()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
