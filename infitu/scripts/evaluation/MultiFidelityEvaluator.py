#!/usr/bin/env python

import rospy
import numpy as np
from itertools import izip

from percepto_msgs.srv import GetCritique, GetCritiqueResponse
from optim import CritiqueInterface
from argus_utils import wait_for_service


class MultiFidelityEvaluator:

    def __init__(self):
        topics = rospy.get_param('~topics')

        for i, topic in enumerate(topics):
            rospy.loginfo('Using %s for fidelity %d',
                          topic, i)
            wait_for_service(topic)
        self.evaluators = [CritiqueInterface(topic) for topic in topics]

        self.critique_server = rospy.Service('~get_critique',
                                             GetCritique,
                                             self.critique_callback)

    def critique_callback(self, req):
        fid = req.input[0]
        if fid != round(fid):
            rospy.logerr('Fidelity index must be integer')
            return None
        fid = int(round(fid))

        x = req.input[1:]
        if fid >= len(self.evaluators):
            rospy.logerr('Cannot evaluate fidelity %d out of %d',
                         fid, len(self.evaluators))
            return None

        req.input = x
        return self.evaluators[fid].raw_call(req)

if __name__ == '__main__':
    rospy.init_node('multi_fidelity_evaluator')
    try:
        epe = MultiFidelityEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
