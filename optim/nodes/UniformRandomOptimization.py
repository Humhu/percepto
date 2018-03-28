#!/usr/bin/env python

import sys
import os
import cPickle as pickle
import rospy
import numpy as np
import optim
import time

class UniformSampler:
    """Uniformly randomly samples an optimization problem.
    """

    def __init__(self):

        self.dim = rospy.get_param('~dim')
        def farr(x):
            if hasattr(x, '__iter__'):
                return np.array([float(xi) for xi in x])
            else:
                return np.full(self.dim, float(x))
        self.input_lower = farr(rospy.get_param('~lower_bounds'))
        self.input_upper = farr(rospy.get_param('~upper_bounds'))

        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo('No random seed specified. Using default behavior.')
        else:
            seed = int(seed)
            rospy.loginfo('Initializing with random seed: ' + str(seed))
            np.random.seed(seed)


        self.max_evals = float(rospy.get_param('~convergence/max_evaluations',
                                               float('inf')))
        self.max_time = float(rospy.get_param('~convergence/max_time',
                                              float('inf')))

        self.start_time = time.time()
        self.interface = None
        self.rounds = []

    def get_next_test(self):
        return np.random.uniform(low=self.input_lower,
                                 high=self.input_upper,
                                 size=self.dim)

    def finished(self):
        """Returns whether the optimization is complete
        """
        duration = time.time() - self.start_time
        hit_max_time = duration >= self.max_time
        hit_max_evals = len(self.rounds) >= self.max_evals

        if len(self.rounds) > 0:
            remaining_time = self.max_time - duration
            
            remaining_evals = self.max_evals - len(self.rounds)
            time_per = duration / len(self.rounds)
            remaining_eval_time = remaining_evals * time_per

            min_remaining = min((remaining_time, remaining_eval_time))
            rospy.loginfo('ETA: %f (s) at %f (s) per sample', min_remaining, time_per)

        return hit_max_evals or hit_max_time

    def execute(self):
        while not self.finished():
            inval = self.get_next_test()
            reward, feedback = self.interface(inval)
            self.rounds.append((inval, reward, feedback))


if __name__ == '__main__':
    rospy.init_node('uniform_sampler')

    out_path = rospy.get_param('~output_path')
    if os.path.isfile(out_path):
        rospy.logerr('Output path %s already exists!', out_path)
        sys.exit(-1)
    out_file = open(out_path, 'w')

    sampler = UniformSampler()
    interface_info = rospy.get_param('~interface')
    sampler.interface = optim.CritiqueInterface(**interface_info)

    res = 'incomplete'
    try:
        res = sampler.execute()
    except rospy.ROSInterruptException:
        pass

    # Save output
    pickle.dump((res, sampler.rounds), out_file)
