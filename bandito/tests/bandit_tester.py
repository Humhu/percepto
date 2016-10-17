"""
Provides an artificial problem to test bandit algorithms against.
"""

import numpy as np
import rospy, random
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse

def test_func( x ):
    x_norm = np.linalg.norm( x )
    return random.uniform( -x_norm, 0 )

class TestBanditProblem:

    def __init__( self ):
        self.query_server = rospy.Service( '~get_critique', 
                                           GetCritique, 
                                           self.critique_callback )

    def critique_callback( self, req ):
        res = GetCritiqueResponse()
        res.critique = test_func( req.input )
        return res

if __name__=='__main__':
    rospy.init_node('test_bandit_problem')
    try:
        tbp = TestBanditProblem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass