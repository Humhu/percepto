"""Wrappers for rospy support
"""
import rospy
from itertools import izip
from percepto_msgs.srv import GetCritique, GetCritiqueRequest

def _stringify_names(x, n):
    if n is None:
        return str(x)
    s = ''
    for xi, ni in zip(x, n):
        s += '\n\t%s: %f' % (ni, xi)
    return s

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

    def raw_call(self, x, n=None):
        req = GetCritiqueRequest()
        req.input = x
        if n is not None:
            req.names = n

        if self.verbose:
            rospy.loginfo('Evaluating %s...', _stringify_names(x, n))

        try:
            res = self.proxy.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not evaluate: %s ', str(x))
            raise RuntimeError('Could not evaluate')

        if self.verbose:
            msg = 'Evaluated: %s\n' % _stringify_names(x, n)
            msg += 'Critique: %f\n' % res.critique
            msg += 'Feedback:\n'
            for name, value in zip(res.feedback_names, res.feedback_values):
                msg += '\t%s: %f\n' % (name, value)
            rospy.loginfo(msg)

        return res

    def __call__(self, x, n=None):
        res = self.raw_call(x, n)
        feedback = dict(izip(res.feedback_names, res.feedback_values))
        return res.critique, feedback