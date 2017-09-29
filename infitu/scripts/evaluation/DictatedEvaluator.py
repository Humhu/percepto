#!/usr/bin/env python

import cPickle as pickle
import rospy
import numpy as np
import os
import optim


class DictatedEvaluator:
    """Runs evaluations of inputs specified in a pickled dict
    """

    def __init__(self):
        interface_info = rospy.get_param('~interface')
        self.interface = optim.CritiqueInterface(**interface_info)

        self.pad_fidelity = rospy.get_param('~pad_fidelity', False)
        if self.pad_fidelity:
            self.fidelity = rospy.get_param('~fidelity')

        self.trials_per_input = rospy.get_param('~trials_per_input')

        spec_path = rospy.get_param('~in_path')
        spec_raw = pickle.load(open(spec_path, 'r'))
        self.test_inputs = spec_raw['test_inputs']

        for inp in self.test_inputs:
            rospy.loginfo('Testing input %s', str(inp))

    def gen_action(self, a):
        if not self.pad_fidelity:
            return a
        else:
            return np.hstack((self.fidelity, a))

    def execute(self):
        rounds = []

        for i, inp in enumerate(self.test_inputs):
            sub = []
            rospy.loginfo('Testing input %d/%d %s', i+1, len(self.test_inputs), str(inp))
            full_inp = self.gen_action(inp)
            for j in range(self.trials_per_input):
                rospy.loginfo('\tTrial %d/%d', j+1, self.trials_per_input)
                if rospy.is_shutdown():
                    return rounds
                
                sub.append(self.interface(full_inp))
            rounds.append((full_inp, sub))
        return rounds


if __name__ == '__main__':

    rospy.init_node('dictated_evaluator')

    out_path = rospy.get_param('~output_path')
    if os.path.isfile(out_path):
        rospy.logerr('Output path %s already exists!', out_path)
        sys.exit(-1)
    out_file = open(out_path, 'w')

    bopt = DictatedEvaluator()

    try:
        res = bopt.execute()
    except rospy.ROSInterruptException:
        pass

    pickle.dump(res, out_file)
