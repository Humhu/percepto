"""Wrappers for rospy support
"""
import rospy
import time
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

    def __init__(self, topic, verbose=False, n_retries=5):
        rospy.loginfo('Waiting for critique service: %s', topic)
        rospy.wait_for_service(topic)
        rospy.loginfo('Connected to service: %s', topic)
        self.proxy = rospy.ServiceProxy(topic, GetCritique)
        self.topic = topic

        self.verbose = verbose
        self.n_retries = n_retries

    def raw_call(self, x, n=None):
        if isinstance(x, GetCritiqueRequest):
            req = x
            x = req.input
            n = req.names
        else:
            req = GetCritiqueRequest()
            req.input = x
            if n is not None:
                req.names = n

        if self.verbose:
            rospy.loginfo('Evaluating %s...', _stringify_names(x, n))

        succ = False
        counter = 0
        while not rospy.is_shutdown():
            try:
                if self.n_retries > 0 and counter > self.n_retries:
                    break
                counter += 1

                res = self.proxy.call(req)
                succ = True
                break
            except rospy.ServiceException:
                rospy.logwarn('Failed to evaluate: %s on %s, retrying...',
                              str(x), self.topic)
                time.sleep(1.0)

        if not succ:
            rospy.logerr('Could not evaluate: %s. All retries failed', str(x))
            return None

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
