#!/usr/bin/env python

import rospy
import numpy as np
from scipy.stats import multivariate_normal
from argus_utils.MatrixUtils import MsgToMatrix
from argus_msgs.msg import FilterStepInfo
from percepto_msgs.msg import RewardStamped

class ObsLikelihoodReward:
    '''Publishes reward messages based on observation log-likelihoods.'''

    def __init__( self ):

        buff_size = rospy.get_param( '~buffer_size', 100 )
        self.reward_sub = rospy.Subscriber( 'info', FilterStepInfo, self.InfoCallback,
                                            queue_size=buff_size )
        self.reward_pub = rospy.Publisher( '~reward', RewardStamped, queue_size=buff_size )
        
    def InfoCallback( self, info_msg ):
        if info_msg.info_type != FilterStepInfo.UPDATE_STEP:
            return

        v = np.array( info_msg.update.prior_obs_error )
        V = MsgToMatrix( info_msg.update.obs_error_cov )

        reward_msg = RewardStamped()
        reward_msg.header = info_msg.header
        reward_msg.reward = multivariate_normal.logpdf( v, cov=V )

        self.reward_pub.publish( reward_msg );

if __name__ == '__main__':
    rospy.init_node( 'obs_likelihood_reward' )
    try:
        oll = ObsLikelihoodReward()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

