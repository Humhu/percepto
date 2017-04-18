#!/usr/bin/env python

import rospy
from percepto_msgs.srv import SetParameters, SetParametersRequest
from percepto_msgs.msg import RewardStamped


class RewardMonitor(object):
    """Sets a set of safe parameters once a reward topic passes a threshold.
    """

    def __init__(self):
        param_topic = rospy.get_param('~parameter_topic')
        rospy.loginfo('Waiting for service %s...', param_topic)
        rospy.wait_for_service(param_topic)
        rospy.loginfo('Connected to %s', param_topic)
        self.set_param = rospy.ServiceProxy(name=param_topic,
                                            service_class=SetParameters,
                                            persistent=True)

        self.safe_params = rospy.get_param('~safety_parameters')
        self.min_reward = rospy.get_param('~min_reward_threshold')
        reward_topic = rospy.get_param('~reward_topic')
        self.reward_sub = rospy.Subscriber(name=reward_topic,
                                           data_class=RewardStamped,
                                           callback=self.reward_callback)

    def reward_callback(self, msg):
        if msg.reward < self.min_reward:
            rospy.logwarn('Reward %f less than threshold %f. Engaging safety...',
                          msg.reward, self.min_reward)
        try:
            req = 

if __name__ == '__main__':
    rospy.init_node('reward_monitor')
    rm = RewardMonitor()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
