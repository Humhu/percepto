#!/usr/bin/env python

import rospy
import numpy as np
from itertools import izip
from percepto_msgs.srv import GetCritique, GetCritiqueResponse
from argus_utils import wait_for_service


class MultiTrialEvaluator:

    def __init__(self):

        critique_topic = rospy.get_param('~critic_service')
        wait_for_service(critique_topic)
        self.critique_proxy = rospy.ServiceProxy(critique_topic, GetCritique)

        self.num_trials = rospy.get_param('~num_trials')
        self.critique_service = rospy.Service('~get_critique', GetCritique,
                                              self.CritiqueCallback)

    def CritiqueCallback(self, req):

        # Perform all the requisite service calls
        try:
            responses = [self.critique_proxy.call(req)
                         for i in range(self.num_trials)]
        except rospy.ServiceException:
            rospy.logerr('Error during critique query.')
            return None

        # Sort the responses
        critiques = []
        feedbacks = {}
        for res in responses:
            critiques.append(res.critique)
            for (fb_name, fb_val) in izip(res.feedback_names, res.feedback_values):
                if fb_name not in feedbacks:
                    feedbacks[fb_name] = []
                feedbacks[fb_name].append(fb_val)

        # Parse the responses, reporting means and variances
        res = GetCritiqueResponse()
        res.critique = np.mean(critiques)

        res.feedback_names.append('critique_var')
        res.feedback_values.append(np.var(critiques))

        for (fb_name, fb_vals) in feedbacks.iteritems():
            res.feedback_names.append(fb_name + '_mean')
            res.feedback_values.append(np.mean(fb_vals))
            res.feedback_names.append(fb_name + '_var')
            res.feedback_values.append(np.var(fb_vals))

        # Print results
        outstr = 'critique_mean: %f\n' % res.critique
        for (fb_name, fb_val) in izip(res.feedback_names, res.feedback_values):
            outstr += '%s: %f\n' % (fb_name, fb_val)
        rospy.loginfo(outstr)

        return res


if __name__ == '__main__':
    rospy.init_node('multi_trial_evaluator')
    try:
        mte = MultiTrialEvaluator()
        rospy.spin()
    except ROSInterruptException:
        pass
