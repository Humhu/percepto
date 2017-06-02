#! /usr/bin/env python

"""Provides an artificial problem to test multi-fidelity optimizers on.
"""
import numpy as np
import rospy
import random
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse


class MultiFidelityFunctions(object):
    def __init__(self, num_fidelities, noise=0.1, gamma=0.1):
        self.gamma = np.arange(num_fidelities)[::-1] * gamma
        self.theta_offsets = np.random.uniform(-self.gamma, self.gamma,
                                               size=num_fidelities)
        rospy.loginfo('Gammas: ' + str(self.gamma))
        rospy.loginfo('Offsets: ' + str(self.theta_offsets))
        self.noise = noise

    def __call__(self, fid, x):
        if round(fid) != fid:
            raise ValueError('Fidelity must be integer')
        if fid >= len(self.gamma):
            raise ValueError('Cannot query fidelity %d out of %d' %
                             (fid, len(self.gamma)))

        mean = -np.linalg.norm(x)
        noise = random.uniform(-self.noise, self.noise)
        bias = np.cos(mean + self.theta_offsets[fid]) * self.gamma[fid]
        return mean + noise + bias


class TestMultiFidelityOptimizationProblem:
    """Simulates an optimization problem.
    """

    def __init__(self):

        num_fids = rospy.get_param('~num_fidelities')
        noise = rospy.get_param('~noise', 0.1)
        gamma = rospy.get_param('~gamma', 0.1)
        self.problem = MultiFidelityFunctions(num_fidelities=num_fids,
                                              noise=noise,
                                              gamma=gamma)

        self.query_time = rospy.get_param('~query_time')
        self.query_server = rospy.Service('~get_critique',
                                          GetCritique,
                                          self.critique_callback)

    def critique_callback(self, req):
        res = GetCritiqueResponse()
        fid = req.input[0]
        x = req.input[1:]
        res.critique = self.problem(fid, x)
        rospy.sleep(rospy.Duration((fid + 1) * self.query_time))
        return res


if __name__ == '__main__':
    rospy.init_node('test_mf_problem_node')
    try:
        tbp = TestMultiFidelityOptimizationProblem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
