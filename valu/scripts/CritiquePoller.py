#!/usr/bin/env python

import rospy
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from percepto_msgs.msg import RewardStamped

class CritiquePoller:

    def __init__( self ):

        service_path = rospy.get_param( '~critique_service' )
        rospy.wait_for_service( service_path, timeout=10.0 )
        self.get_critique = rospy.ServiceProxy( service_path, GetCritique, persistent = False )

        self.reward_pub = rospy.Publisher( '~critique', RewardStamped, queue_size=10 )

        poll_rate = rospy.get_param( '~poll_rate' )
        self.poll_offset = rospy.Duration( rospy.get_param( '~poll_offset' ) )
        self.timer = rospy.Timer( rospy.Duration( 1.0/poll_rate ), 
                                  self.TimerCallback )

    def TimerCallback( self, event ):
        msg = RewardStamped()
        query_time = event.current_expected - self.poll_offset
        msg.header.stamp = query_time
        
        req = GetCritiqueRequest()
        req.time = query_time
        try:
            res = self.get_critique( req )
            msg.reward = float( res.critique )
        except rospy.ServiceException as e:
            print 'Could not query critique: ' + str(e)
            return

        self.reward_pub.publish( msg )

if __name__ == '__main__':
    rospy.init_node( 'critique_poller' )
    
    try:
        cp = CritiquePoller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass