#!/usr/bin/env python

import cPickle as pickle
from itertools import izip
import numpy as np
import rospy
import optim.reward_models as orm
import optim.optimization as opt
import optim.arm_selectors as oas
import optim.ros_interface as opt_ros

from abb_surrogate.srv import SetSpeed

class SpeedSweep(object):

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
        full_lower_bounds = farr(rospy.get_param('~lower_bounds'), dim)
        full_upper_bounds = farr(rospy.get_param('~upper_bounds'), dim)
        self.lower_bounds = full_lower_bounds
        self.upper_bounds = full_upper_bounds

        self.num_reps = rospy.get_param('~num_samples_per', 1)
        self.rounds = []
        self.num_tests = rospy.get_param('~num_tests')
        self.output_path = rospy.get_param('~output_path')

        speed_topic = rospy.get_param('~speed_topic')
        rospy.wait_for_service(speed_topic)
        self.speed_proxy = rospy.ServiceProxy(speed_topic, SetSpeed)
        speed_range = rospy.get_param('~speed_range')
        num_speeds = rospy.get_param('~num_speeds')
        self.speeds = np.linspace(speed_range[0], speed_range[1], num_speeds)

    def save(self):
        rospy.loginfo('Saving to %s...', self.output_path)
        out_file = open(self.output_path, 'wb')
        pickle.dump((self.speeds, self.rounds), out_file)

    def finished(self):
        """Returns whether the optimization is complete.
        """
        return len(self.rounds) >= self.num_tests

    def execute(self, interface):
        """Begin execution of the specified problem.
        """
        while not rospy.is_shutdown() and not self.finished():
            print 'Round %d/%d...' % (len(self.rounds) + 1, self.num_tests)

            x = np.random.uniform(self.lower_bounds, self.upper_bounds)
            print 'Testing params: %s' % np.array_str(x)
            round = []
            for speed in self.speeds:
                self.speed_proxy.call(speed)
                print 'Setting speed to %f' % speed
                test_outs = [interface(x) for i in range(self.num_reps)]
                round.append((x,test_outs))
            self.rounds.append(round)

            self.save()

if __name__ == '__main__':
    rospy.init_node('speed_sweep')

    interface_info = rospy.get_param('~interface')
    interface = opt_ros.CritiqueInterface(**interface_info)

    # TODO implement progress saving and resuming
    optimizer = SpeedSweep()
    optimizer.execute(interface)
