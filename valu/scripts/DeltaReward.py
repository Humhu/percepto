#!/usr/bin/env python

import rospy
from percepto_msgs.msg import RewardStamped

class DeltaReward:
    '''Returns deltas of a reward signal.'''

    def __init__( self ):
        self.last_reward = None
        self.reward_sub = rospy.Subscriber( 'reward', RewardStamped, self.RewardCallback )
        self.delta_pub = rospy.Publisher( '~delta_reward', RewardStamped, queue_size=0 )

    def RewardCallback( self, msg ):
        if self.last_reward is None:
            self.last_reward = msg.reward
            return

        out = RewardStamped()
        out.header = msg.header
        out.reward = msg.reward - self.last_reward
        self.delta_pub.publish( out )

        self.last_reward = msg.reward

if __name__ == '__main__':
    rospy.init_node( 'delta_reward' )
    try:
        dr = DeltaReward()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass