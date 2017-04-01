#!/usr/bin/env python

import cPickle as pickle
from itertools import izip
import numpy as np
import rospy
import optim.reward_models as orm
import optim.optimization as opt
import optim.arm_selectors as oas
import optim.ros_interface as opt_ros


class LipschitzTester(object):
    """Performs finite-differencing at random inputs for an optimization target.

    Interfaces with the target through the GetCritique service call.
    """

    def __init__(self):

        def farr(x, n):
            if hasattr(x, '__iter__'):
                out = np.array([float(xi) for xi in x])
                if len(out) != n:
                    raise ValueError('Expected %d but got %d values' % (n, len(out)))
            else:
                return np.full(n, float(x))

        if rospy.has_param('~random_seed'):
            np.random.seed(rospy.get_param('~random_seed'))

        dim = rospy.get_param('~dim')
        self.deltas = farr(rospy.get_param('~test_deltas'), dim)
        full_lower_bounds = farr(rospy.get_param('~lower_bounds'), dim)
        full_upper_bounds = farr(rospy.get_param('~upper_bounds'), dim)
        self.lower_bounds = full_lower_bounds
        self.upper_bounds = full_upper_bounds - self.deltas

        self.num_reps = rospy.get_param('~num_samples_per', 1)
        self.rounds = []
        self.num_tests = rospy.get_param('~num_tests')
        self.output_path = rospy.get_param('~output_path')

    def save(self):
        rospy.loginfo('Saving to %s...', self.output_path)
        out_file = open(self.output_path, 'wb')
        pickle.dump(self.rounds, out_file)

    def get_next_inputs(self):
        """Produces the next inputs to test.
        """
        center = np.random.uniform(self.lower_bounds, self.upper_bounds)
        out = [center] * self.num_reps
        for i, delta in enumerate(self.deltas):
            x = center.copy()
            x[i] += delta
            #out.append(x)
            out += [x] * self.num_reps
        return center, out

    def finished(self):
        """Returns whether the optimization is complete.
        """
        # TODO
        return len(self.rounds) >= self.num_tests

    def execute(self, interface):
        """Begin execution of the specified problem.
        """
        while not rospy.is_shutdown() and not self.finished():
            print 'Round %d/%d...' % (len(self.rounds) + 1, self.num_tests)

            x_center, x_tests = self.get_next_inputs()
            print 'Evaluating %d inputs centered at %s...' % (len(x_tests), str(x_center))
            test_outs = [interface(xi) for xi in x_tests]
            test_rewards, test_feedbacks = zip(*test_outs)
            self.rounds.append(zip(x_tests, test_rewards, test_feedbacks))
            self.save()

if __name__ == '__main__':
    rospy.init_node('lipschitz_tester')

    interface_info = rospy.get_param('~interface')
    interface = opt_ros.CritiqueInterface(**interface_info)

    # TODO implement progress saving and resuming
    optimizer = LipschitzTester()
    optimizer.execute(interface)
