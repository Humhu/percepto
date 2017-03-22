#!/usr/bin/env python

import rospy
import poli.policies as pp
import poli.policy_gradient as ppg

class BanditPolicyGradientNode(object):
    """Continually improves a bandit policy with bandit policy gradient.
    """
    def __init__(self):

        # Parse policy
        policy_info = rospy.get_param('~policy')
        self.policy = pp.parse_policy(policy_info)

        # Parse learner
        learner_info = rospy.get_param('~policy_gradient')
        self.learner = ppg.parse_learner(learner_info, self.policy)

    def 
