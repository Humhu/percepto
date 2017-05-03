#!/usr/bin/env python

import rospy
from threading import Lock
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from percepto_msgs.srv import SetParameters, SetParametersRequest
from infitu.srv import StartEvaluation, StartSetup, StartTeardown, SetRecording
from fieldtrack.srv import ResetFilter, ResetFilterRequest
from argus_utils import wait_for_service


class EmpiricalParameterEvaluator:

    def __init__(self):
        self.index_mode = rospy.get_param('~indexing_mode', 'as_is')
        if self.index_mode == 'as_is':
            pass  # TODO
        elif self.index_mode == 'block':
            # Parse block definitions
            block_info = rospy.get_param('~blocks')
            self.blocks = {}
            for name, params in block_info:
                if name in self.blocks:
                    raise ValueError('Block name %d repeated' % name)
                self.blocks[name] = params
        elif self.index_mode == 'individual':
            pass  # TODO

        self.verbose = rospy.get_param('~verbose', False)

        # Create parameter setter proxy
        setter_topic = rospy.get_param('~parameter_set_service')
        wait_for_service(setter_topic)
        self.setter_proxy = rospy.ServiceProxy(setter_topic,
                                               SetParameters,
                                               True)

        # Create evaluation trigger proxy
        self.mode = rospy.get_param('~evaluation_mode')
        if self.mode == 'service_call':
            evaluation_topic = rospy.get_param('~start_evaluation_service',
                                               None)
            wait_for_service(evaluation_topic)
            self.evaluation_proxy = rospy.ServiceProxy(evaluation_topic,
                                                       StartEvaluation,
                                                       True)
        elif self.mode == 'fixed_duration':
            self.evaluation_time = rospy.get_param('~evaluation_time')
        else:
            raise ValueError('Unknown mode %s' % self.mode)

        # Check for evaluation setup
        self.setup_proxy = None
        setup_topic = rospy.get_param('~start_setup_service', None)
        if setup_topic is not None:
            wait_for_service(setup_topic)
            self.setup_proxy = rospy.ServiceProxy(setup_topic,
                                                  StartSetup,
                                                  True)

        # Check for evaluation teardown
        self.teardown_proxy = None
        teardown_topic = rospy.get_param('~start_teardown_service', None)
        if teardown_topic is not None:
            wait_for_service(teardown_topic)
            self.teardown_proxy = rospy.ServiceProxy(teardown_topic,
                                                     StartTeardown,
                                                     True)

        # Create filter reset proxy
        self.reset_proxy = None
        reset_topic = rospy.get_param('~reset_filter_service', None)
        if reset_topic is not None:
            wait_for_service(reset_topic)
            self.reset_proxy = rospy.ServiceProxy(reset_topic,
                                                  ResetFilter,
                                                  True)

        recording_topics = rospy.get_param('~recorders')
        self.recorders = {}
        for name, topic in recording_topics.iteritems():
            wait_for_service(topic)
            self.recorders[name] = rospy.ServiceProxy(topic,
                                                      SetRecording,
                                                      True)
        self.critique_record = rospy.get_param('~critique_record')
        if self.critique_record not in self.recorders:
            raise ValueError('Critique not a registered recorder!')

        # Create critique service
        eval_delay = rospy.get_param('~evaluation_delay', 0.0)
        self.evaluation_delay = rospy.Duration(eval_delay)
        self.critique_service = rospy.Service('~get_critique',
                                              GetCritique,
                                              self.critique_callback)

    def start_recording(self):
        """Start evaluation recording. Returns success."""
        if self.verbose:
            rospy.loginfo('Starting recording...')

        try:
            for recorder in self.recorders.itervalues():
                recorder.call(True)
        except rospy.ServiceException:
            rospy.logerr('Could not start recording.')
            return False
        return True

    def stop_recording(self):
        """Stop recording and return evaluation. Returns None if fails."""
        if self.verbose:
            rospy.loginfo('Stopping recording...')

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

    def set_parameters(self, inval):
        """Set the parameters to be evaluated. Returns success."""
        if self.verbose:
            rospy.loginfo('Setting parameters...')

        preq = SetParametersRequest()
        preq.parameters = inval
        try:
            self.setter_proxy.call(preq)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to %s', (str(inval),))
            return False
        return True

    def reset_filter(self):
        """Reset the state estimator. Returns success."""
        if self.reset_proxy is None:
            return True

        if self.verbose:
            rospy.loginfo('Resetting filter...')

        resreq = ResetFilterRequest()
        resreq.time_to_wait = 0
        resreq.filter_time = rospy.Time.now()
        try:
            self.reset_proxy.call(resreq)
        except rospy.ServiceException:
            rospy.logerr('Could not reset filter.')
            return False
        return True

    def run_evaluation(self):
        if self.verbose:
            rospy.loginfo('Starting evaluation...')

        if self.mode == 'service_call':
            try:
                self.evaluation_proxy.call()
            except rospy.ServiceException:
                rospy.logerr('Could not run evaluation.')
                return False
            return True
        elif self.mode == 'fixed_duration':
            rospy.sleep(self.evaluation_time)
            return True

    def start_setup(self):
        if self.setup_proxy is None:
            return True

        if self.verbose:
            rospy.loginfo('Starting setup...')

        try:
            self.setup_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not setup.')
            return False
        return True

    def start_teardown(self):
        if self.teardown_proxy is None:
            return True

        if self.verbose:
            rospy.loginfo('Starting teardown...')

        try:
            self.teardown_proxy.call()
        except rospy.ServiceException:
            rospy.logerr('Could not teardown.')
            return False
        return True

    def critique_callback(self, req):

        if not self.start_setup():
            return None

        # Call parameter setter
        if not self.set_parameters(req.input):
            return None

        # Reset state estimator
        if not self.reset_filter():
            return None

        # Wait before starting
        # NOTE This catches instances when relying on sim tim
        if self.evaluation_delay.to_sec() > 0:
            rospy.sleep(self.evaluation_delay)

        if not self.start_recording():
            return None

        # Wait until evaluation is done
        if not self.run_evaluation():
            return None

        # Get outcomes
        res = GetCritiqueResponse()
        res.critique, feedback = self.stop_recording()
        res.feedback_names = [f[0] for f in feedback]
        res.feedback_values = [f[1] for f in feedback]

        if not self.start_teardown():
            return None

        return res


if __name__ == '__main__':
    rospy.init_node('empirical_parameter_evaluator')
    try:
        epe = EmpiricalParameterEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
