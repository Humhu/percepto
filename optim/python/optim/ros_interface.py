"""Wrappers for rospy support
"""
import rospy
from itertools import izip
from percepto_msgs.srv import GetCritique, GetCritiqueRequest

class CritiqueInterface(object):
    """Provides an optimization problem interface by wrapping
    ROS service calls.

    Parameters
    ----------
    topic : string
        The GetCritique service topic name
    verbose : bool (default False)
        Whether to print each evaluation result
    """
    def __init__(self, topic, verbose=False):
        rospy.loginfo('Waiting for critique service: %s', topic)
        rospy.wait_for_service(topic)
        rospy.loginfo('Connected to service: %s', topic)
        self.proxy = rospy.ServiceProxy(topic, GetCritique)

        self.verbose = verbose

    def __call__(self, x):
        req = GetCritiqueRequest()
        req.input = x

        if self.verbose:
            rospy.loginfo('Evaluating %s...', str(x))

        try:
            res = self.proxy.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not evaluate: %s ' + str(x))
            raise RuntimeError('Could not evaluate')

        feedback = dict(izip(res.feedback_names, res.feedback_values))

        if self.verbose:
            msg = 'Evaluated: %s\n' % str(x)
            msg += 'Critique: %f\n' % res.critique
            msg += 'Feedback:\n'
            for name, value in feedback.iteritems():
                msg += '\t%s: %f\n' % (name, value)
            rospy.loginfo(msg)

        return res.critique, feedback