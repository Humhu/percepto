#!/usr/bin/env python

import numpy as np
import rospy
import infitu
from threading import Lock
from itertools import izip

from fieldtrack.srv import ResetFilter
from percepto_msgs.msg import EpisodeBreak
from nav_msgs.msg import Odometry


def twist_to_vec(msg):
    return np.array([msg.linear.x,
                     msg.linear.y,
                     msg.linear.z,
                     msg.angular.x,
                     msg.angluar.y,
                     msg.angular.z])


class StateEstimatorResetter(object):
    def __init__(self):

        self._odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

    def odom_callback(self, msg):
        vel = twist_to_vec(msg.twist.twist)
        vel_cov = 


if __name__ == '__main__':
    rospy.init_node('state_estimator_resetter')

    node = StateEstimatorResetter()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
