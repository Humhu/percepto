#!/usr/bin/env python

import rospy
from threading import Lock
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from percepto_msgs.srv import SetParameters, SetParametersRequest
from infitu.srv import StartEvaluation, StartTeardown, SetRecording
from fieldtrack.srv import ResetFilter, ResetFilterRequest


def wait_for_service(srv):
    rospy.loginfo('Waiting for service %s', srv)
    rospy.wait_for_service(srv)
    rospy.loginfo('Service now available %s', srv)


class EmpiricalParameterEvaluator:

    def __init__(self):
        # Create parameter setter proxy
        setter_topic = rospy.get_param('~parameter_set_service')
        wait_for_service(setter_topic)
        self.setter_proxy = rospy.ServiceProxy(
            setter_topic, SetParameters, True)

        # Create evaluation trigger proxy
        evaluation_topic = rospy.get_param('~start_evaluation_service')
        wait_for_service(evaluation_topic)
        self.evaluation_proxy = rospy.ServiceProxy(
            evaluation_topic, StartEvaluation, True)

        # Check for evaluation teardown
        self.teardown_proxy = None
        if rospy.has_param('~start_teardown_service'):
            teardown_topic = rospy.get_param('~start_teardown_service')
            wait_for_service(teardown_topic)
            self.teardown_proxy = rospy.ServiceProxy(teardown_topic, StartTeardown, True)

        # Create filter reset proxy
        reset_topic = rospy.get_param('~reset_filter_service', None)
        self.reset_proxy = None
        if reset_topic is not None:
            wait_for_service(reset_topic)
            self.reset_proxy = rospy.ServiceProxy(reset_topic, ResetFilter, True)

        recording_topics = rospy.get_param('~recorders')
        self.recorders = {}
        for name, topic in recording_topics.iteritems():
            wait_for_service(topic)
            self.recorders[name] = rospy.ServiceProxy(
                topic, SetRecording, True)
        self.critique_record = rospy.get_param('~critique_record')
        if self.critique_record not in self.recorders:
            raise ValueError('Critique not a registered recorder!')

        # Create critique service
        self.evaluation_delay = rospy.Duration(rospy.get_param('~evaluation_delay', 0.0))
        self.critique_service = rospy.Service(
            '~get_critique', GetCritique, self.CritiqueCallback)

    def StartRecording(self):
        """Start evaluation recording. Returns success."""
        try:
            for recorder in self.recorders.itervalues():
                recorder.call(True)
        except rospy.ServiceException:
            rospy.logerr('Could not start recording.')
            return False
        return True

    def StopRecording(self):
        """Stop recording and return evaluation. Returns None if fails."""

        try:
            feedback = []
            for name, recorder in self.recorders.iteritems():
                res = recorder.call(False).evaluation
                if name == self.critique_record:
                    critique = res
                else:
                    feedback.append((name, res))
        except rospy.ServiceException:
            rospy.logerr('Could not stop recording.')
            return None

        return (critique, feedback)

    def SetParameters(self, inval):
        """Set the parameters to be evaluated. Returns success."""

        preq = SetParametersRequest()
        preq.parameters = inval
        try:
            self.setter_proxy.call(preq)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to %s', (str(inval),))
            return False
        return True

    # TODO Move this out to somewhere else?
    def ResetFilter(self):
        """Reset the state estimator. Returns success."""
        if self.reset_proxy is None:
            return True
            
        resreq = ResetFilterRequest()
        resreq.time_to_wait = 0
        resreq.filter_time = rospy.Time.now()
        try:
            self.reset_proxy.call(resreq)
        except rospy.ServiceException:
            rospy.logerr('Could not reset filter.')
            return False
        return True

    def RunEvaluation(self):
        try:
            self.evaluation_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not run evaluation.')
            return False
        return True

    def StartTeardown(self):
        try:
            if self.teardown_proxy is not None:
                self.teardown_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not teardown.')
            return False
        return True

    def CritiqueCallback(self, req):

        # Call parameter setter
        if not self.SetParameters(req.input):
            return None

        # Reset state estimator
        if not self.ResetFilter():
            return None

        # Wait before starting
        rospy.sleep(self.evaluation_delay)

        if not self.StartRecording():
            return None

        # Wait until evaluation is done
        if not self.RunEvaluation():
            return None

        # Get outcomes
        res = GetCritiqueResponse()
        res.critique, feedback = self.StopRecording()
        res.feedback_names = [f[0] for f in feedback]
        res.feedback_values = [f[1] for f in feedback]

        if not self.StartTeardown():
            return None

        return res


if __name__ == '__main__':
    rospy.init_node('empirical_parameter_evaluator')
    try:
        epe = EmpiricalParameterEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
