#!/usr/bin/env python

from itertools import izip
import numpy as np
import rospy
import optim.reward_models as orm
import optim.optimization as opt
import optim.arm_selectors as oas
import optim.ros_interface as opt_ros


class BayesianOptimizer(object):
    """Iteratively samples and updates a reward model (response surface) to optimize a function.

    Uses the GetCritique service to interface with optimization problems.
    """

    def __init__(self):

        # Parse reward model
        model_info = rospy.get_param('~reward_model')
        self.reward_model = orm.parse_reward_model(model_info)

        # Parse acquisition optimizer
        optimizer_info = rospy.get_param('~auxiliary_optimizer')
        self.aux_optimizer = opt.parse_optimizers(optimizer_info)

        def farr(x, n):
            if hasattr(x, '__iter__'):
                return np.array([float(xi) for xi in x])
            else:
                return np.full(n, float(x))

        dim = rospy.get_param('~dim')
        self.lower_bounds = farr(rospy.get_param('~lower_bounds'), dim)
        self.upper_bounds = farr(rospy.get_param('~upper_bounds'), dim)
        self.aux_optimizer.opts['bounds'] = [
            self.lower_bounds, self.upper_bounds]
        self.aux_x_init = 0.5 * (self.lower_bounds + self.upper_bounds)
        self.aux_x_init[np.logical_not(np.isfinite(self.aux_x_init))] = 0
        #self.aux_x_init = np.zeros(dim)

        # TODO Parse selection approach
        # TODO Support partial optimization
        self.acq_func = oas.UCBAcquisition(self.reward_model)

        self.rounds = []
        init_info = rospy.get_param('~initialization')
        self.num_init = init_info['num_samples']
        init_method = init_info['method']
        if init_method == 'uniform':
            self.init_sample_func = lambda: np.random.uniform(
                self.lower_bounds, self.upper_bounds)
        # TODO Support for latin hypercubes, other approaches?
        else:
            raise ValueError(
                'Invalid initial distribution method %s', init_method)

    @property
    def is_initialized(self):
        return len(self.rounds) >= self.num_init

    def pick_initial_sample(self):
        """Selects the next sample for initialization by sampling from
        a random distribution.
        """
        return self.init_sample_func()

    def pick_next_sample(self):
        """Selects the next sample to explore by optimizing the acquisition function.
        """
        x, acq = self.aux_optimizer.optimize(
            x0=self.aux_x_init, func=self.acq_func)
        rospy.loginfo('Next sample %s with acquisition value %f', str(x), acq)
        return x

    def finished(self):
        """Returns whether the optimization is complete.
        """
        # TODO
        return False

    def execute(self, interface):
        """Begin execution of the specified problem.
        """
        while not rospy.is_shutdown() and not self.finished():
            print 'Round %d...' % len(self.rounds)

            if not self.is_initialized:
                next_sample = self.pick_initial_sample()
                rospy.loginfo('Initializing with %s', str(next_sample))
            else:
                next_sample = self.pick_next_sample()
                pred_mean, pred_sd = self.reward_model.predict_reward(
                    next_sample)
                rospy.loginfo('Picked sample %s with predicted mean %f +- %f',
                              str(next_sample), pred_mean, pred_sd)

            reward, feedback = interface(next_sample)
            self.reward_model.report_sample(x=next_sample, reward=reward)
            self.rounds.append((next_sample, reward, feedback))


if __name__ == '__main__':
    rospy.init_node('bayesian_optimizer')

    interface_info = rospy.get_param('~interface')
    interface = opt_ros.CritiqueInterface(**interface_info)

    # TODO implement progress saving and resuming
    optimizer = BayesianOptimizer()
    optimizer.execute(interface)
