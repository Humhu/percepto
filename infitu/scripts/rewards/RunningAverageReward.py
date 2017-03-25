#!/usr/bin/env python

import rospy
import numpy as np
from math import sqrt
from percepto_msgs.msg import RewardStamped
from collections import deque

class RunningAveragePerformance:
    '''Outputs a running average window reward.'''

    def __init__( self ):
        rospy.init_node( 'running_average_performance' )

        self.max_window_len = round( rospy.get_param( '~window_length' ) )
        if self.max_window_len < 0:
            raise ValueError( 'window_length must be positive.' )

        self.reward_pub = rospy.Publisher( '~average', RewardStamped, queue_size=0 )
        self.odom_sub = rospy.Subscriber( 'reward', RewardStamped, self.RewardCallback )
        self.window = deque()
        self.acc = 0

    def RewardCallback( self, msg ):
        self.acc += msg.reward
        self.window.append( msg.reward )

        while len( self.window ) > self.max_window_len:
            self.acc = self.acc - self.window.popleft()

        out = RewardStamped()
        out.header.stamp = msg.header.stamp
        out.reward = self.acc / len( self.window )

        self.reward_pub.publish( out )

if __name__ == '__main__':
    try:
        rms = RunningAveragePerformance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



