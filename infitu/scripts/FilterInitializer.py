#!/usr/bin/env python

import rospy

from geometry_msgs.msg import TwistStamped

from fieldtrack.srv import ResetFilter, ResetFilterRequest
from argus_utils import wait_for_service
from threading import RLock, Condition


class FilterInitializer(object):
    """Resets filter state to ground truth
    """

    def __init__(self):

        self.lock = RLock()
        self.base_request = None

        self.two_dimensional = rospy.get_param('~two_dimensional', False)

        truth_dict = {'twist_stamped': TwistStamped}
        self.truth_mode = rospy.get_param('~truth_mode')
        if self.truth_mode not in truth_dict:
            raise ValueError('Unsupported truth mode: ' + self.truth_mode)
        self.truth_sub = rospy.Subscriber('truth',
                                          truth_dict[self.truth_mode],
                                          self.truth_callback)

        reset_topic = rospy.get_param('~reset_service')
        wait_for_service(reset_topic)
        self.reset_proxy = rospy.ServiceProxy(reset_topic,
                                              ResetFilter)

        self.reset_server = rospy.Service('~reset',
                                          ResetFilter,
                                          self.reset_callback)

    def __get_twist(self, twist):
        if self.two_dimensional:
            return [twist.linear.x, twist.linear.y, twist.angular.z]
        else:
            return [twist.linear.x, twist.linear.y, twist.linear.z,
                    twist.angular.x, twist.angular.y, twist.angular.z]

    def truth_callback(self, msg):
        with self.lock:
            if self.base_request is None:
                return

            rospy.loginfo('Ground truth received. Processing reset request...')
            if self.truth_mode == 'twist_stamped':
                self.base_request.state = self.__get_twist(msg.twist)
                self.base_request.filter_time = msg.header.stamp
            else:
                raise RuntimeError('Unsupported truth mode at runtime: %s' %
                                   self.truth_mode)

            try:
                self.reset_proxy(self.base_request)
            except rospy.ServiceException:
                rospy.logerr('Could not reset filter!')
            rospy.loginfo('Reset request completed!')
            self.base_request = None

    def reset_callback(self, req):
        with self.lock:
            self.base_request = req
            rospy.loginfo(
                'Reset request received. Waiting for ground truth message...')
        return []


if __name__ == '__main__':
    rospy.init_node('filter_initializer')
    fi = FilterInitializer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
