#!/usr/bin/env python

import sys
import pickle
import rospy
import numpy as np
import optim


class UniformSampler:

    def __init__(self):

         # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo('No random seed specified. Using default behavior.')
        else:
            rospy.loginfo('Initializing with random seed: ' + str(seed))
            np.random.seed(seed)

        def farr(x, n):
            if hasattr(x, '__iter__'):
                return np.array([float(xi) for xi in x])
            else:
                return np.full(n, float(x))

        self.num_samples_per = rospy.get_param('~num_repeats', 1)
        self.dim = rospy.get_param('~dim')
        input_lower = farr(rospy.get_param('~lower_bounds'), self.dim)
        input_upper = farr(rospy.get_param('~upper_bounds'), self.dim)

        fid = rospy.get_param('~fidelity', None)
        if fid is not None:
            input_lower = np.hstack((fid, input_lower))
            input_upper = np.hstack((fid, input_upper))
            self.dim += 1

        num_inputs_to_test = rospy.get_param('~num_samples')
        self.test_inputs = np.random.uniform(low=input_lower,
                                             high=input_upper,
                                             size=(num_inputs_to_test, self.dim))
        self.interface = None
        self.rounds = []

    def evaluate(self, inval):
        results = []
        for i in range(self.num_samples_per):
            rospy.loginfo('Sample %d/%d...', i + 1, self.num_samples_per)
            results.append(self.interface(inval))
        return results

    def execute(self):
        while len(self.rounds) < len(self.test_inputs):
            rospy.loginfo('Input %d/%d', len(self.rounds) +
                          1, len(self.test_inputs))
            x = self.test_inputs[len(self.rounds)]
            trial = self.evaluate(x)
            self.rounds.append((x, trial))


if __name__ == '__main__':
    rospy.init_node('uniform_sampler')

    out_path = rospy.get_param('~output_path')
    out_file = open(out_path, 'w')

    # See if we're resuming
    if rospy.has_param('~load_path'):
        data_path = rospy.get_param('~load_path')
        data_log = open(data_path, 'rb')
        rospy.loginfo('Found load data at %s...', data_path)
        sampler = pickle.load(data_log)
    else:
        rospy.loginfo('No resume data specified. Starting new sampler...')
        sampler = UniformSampler()

    interface_info = rospy.get_param('~interface')
    sampler.interface = optim.CritiqueInterface(**interface_info)

    res = 'incomplete'
    try:
        res = sampler.execute()
    except rospy.ROSInterruptException:
        pass

    # TODO save progress

    # Save output
    pickle.dump((res, sampler.rounds), out_file)
