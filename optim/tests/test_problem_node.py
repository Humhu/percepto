#! /usr/bin/env python

"""Provides an artificial problem to test against.
"""
import numpy as np
import rospy, random
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse

def test_func(x):
    x_norm = np.linalg.norm(x)
    return random.uniform(-x_norm, 0)

class TestOptimizationProblem:
    """Simulates an optimization problem.
    """
    def __init__(self):
        query_time = rospy.get_param('~query_time')
        self.query_delay = rospy.Duration(query_time)
        self.query_server = rospy.Service('~get_critique',
                                          GetCritique,
                                          self.critique_callback)

    def critique_callback(self, req):
        res = GetCritiqueResponse()
        res.critique = test_func(req.input)
        rospy.sleep(self.query_delay)
        return res

if __name__=='__main__':
    rospy.init_node('test_problem_node')
    try:
        tbp = TestOptimizationProblem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass