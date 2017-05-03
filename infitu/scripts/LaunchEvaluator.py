#!/usr/bin/env python

import subprocess32 as subprocess
import rospy

from infitu.srv import StartEvaluation, StartTeardown, StartSetup


class LaunchEvaluator(object):
    def __init__(self):
        self.setup_args = rospy.get_param('~setup_args')
        self.eval_args = rospy.get_param('~eval_args')

        self.setup_server = rospy.Service('~start_setup',
                                          StartSetup,
                                          self.setup_callback)
        self.eval_server = rospy.Service('~start_evaluation',
                                         StartEvaluation,
                                         self.eval_callback)
        self.teardown_server = rospy.Service('~start_teardown',
                                             StartTeardown,
                                             self.teardown_callback)

        self.setup_proc = None

    def setup_callback(self, req):
        args = self.setup_args
        rospy.loginfo('Setting up with: %s' % str(args))
        self.setup_proc = subprocess.Popen(args=args)
        rospy.loginfo('Done.')
        return []

    def eval_callback(self, req):
        if self.setup_proc is None:
            rospy.logerr('Not setup, cannot evaluation')
            return None

        args = self.eval_args
        rospy.loginfo('Evaluating with: %s' % str(args))
        proc = subprocess.Popen(args=args)
        proc.wait()
        return []

    def teardown_callback(self, req):
        if self.setup_proc is None:
            rospy.logerr('No setup to teardown!')
            return None
        rospy.loginfo('Terminating setup process...')
        self.setup_proc.terminate()
        self.setup_proc.wait()
        rospy.loginfo('Done.')
        return []

if __name__ == '__main__':
    rospy.init_node('launch_evaluator')
    le = LaunchEvaluator()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
1
