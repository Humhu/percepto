#!/usr/bin/env python

import math
from itertools import izip
import dill
import pickle

import numpy as np
import rospy
import optim
import matplotlib.pyplot as plt


class BayesianOptimizer(object):
    """Iteratively samples and updates a reward model (response surface) to optimize a function.

    Uses the GetCritique service to interface with optimization problems.
    """

    def __init__(self):

        # Parse reward model
        model_info = rospy.get_param('~reward_model')
        self.reward_model = optim.parse_reward_model(model_info)

        self.enable_sequential = rospy.get_param('~sequential_mode', False)
        if self.enable_sequential:
            rospy.loginfo('Enabling sequential optimization')
            self.iters_per_round = rospy.get_param('~iters_per_round')
            self.base_reward_model = self.reward_model
            self.reward_model = optim.PartialModelWrapper(
                base_model=self.base_reward_model)
            self.loop_counter = 0

        # Parse acquisition optimizer
        optimizer_info = rospy.get_param('~auxiliary_optimizer')
        self.aux_optimizer = optim.parse_optimizers(optimizer_info)

        def farr(x, n):
            if hasattr(x, '__iter__'):
                return np.array([float(xi) for xi in x])
            else:
                return np.full(n, float(x))

        self.input_dim = rospy.get_param('~dim')
        self.lower_bounds = farr(rospy.get_param(
            '~lower_bounds'), self.input_dim)
        self.upper_bounds = farr(rospy.get_param(
            '~upper_bounds'), self.input_dim)
        self.aux_optimizer.lower_bounds = self.lower_bounds
        self.aux_optimizer.upper_bounds = self.upper_bounds
        self.aux_x_init = 0.5 * (self.lower_bounds + self.upper_bounds)
        self.aux_x_init[np.logical_not(np.isfinite(self.aux_x_init))] = 0

        if self.enable_sequential:
            self.reward_model.active_inds = np.random.randint(low=0,
                                                              high=self.input_dim)
            self.reward_model.base_input = self.aux_x_init
            self.full_aux_init = self.aux_x_init
            self.aux_x_init = np.atleast_1d(
                self.aux_x_init[self.reward_model.active_inds])
            self.full_lower_bounds = self.lower_bounds
            self.full_upper_bounds = self.upper_bounds
            self.lower_bounds = self.full_lower_bounds[self.reward_model.active_inds]
            self.upper_bounds = self.full_upper_bounds[self.reward_model.active_inds]
            self.aux_optimizer.lower_bounds = self.lower_bounds
            self.aux_optimizer.upper_bounds = self.upper_bounds

        self.visualize = rospy.get_param('~visualize', False)
        if self.visualize:
            if self.input_dim != 1:
                rospy.logwarn(
                    'Visualization is only enabled for 1D reward model!')
                self.visualize = False
            else:
                plt.ion()

        # TODO Parse selection approach + parameters
        self.acq_func = optim.UCBAcquisition(self.reward_model)
        self.beta_base = rospy.get_param('~exploration_rate', 1.0)

        self.rounds = []
        init_info = rospy.get_param('~initialization')
        self.num_init = init_info['num_samples']
        init_method = init_info['method']

        # TODO Support for latin hypercubes, other approaches?
        if init_method == 'uniform':
            self.init_sample_func = lambda: np.random.uniform(self.lower_bounds,
                                                              self.upper_bounds)
        elif init_method == 'grid':
            if not self.enable_sequential:
                raise ValueError(
                    'Grid initialization only valid for sequential mode')
            self.grid = np.linspace(
                self.lower_bounds, self.upper_bounds, num=self.num_init)
            self.init_sample_func = lambda: self.grid[self.round_index]
        else:
            raise ValueError(
                'Invalid initial distribution method %s' % init_method)

        self.max_evals = rospy.get_param('~convergence/max_evaluations')

    def update_sequential_ind(self):
        current_optima = self.get_current_optima()

        next_inds = np.arange(self.input_dim)
        # TODO Robustify to allow for multiple active inds
        next_inds = next_inds[next_inds != self.reward_model.active_inds]
        self.reward_model.active_inds = np.random.choice(next_inds)
        self.reward_model.base_input = current_optima
        self.aux_x_init = np.atleast_1d(
            self.full_aux_init[self.reward_model.active_inds])

        self.lower_bounds = self.full_lower_bounds[self.reward_model.active_inds]
        self.upper_bounds = self.full_upper_bounds[self.reward_model.active_inds]
        self.aux_optimizer.lower_bounds = self.lower_bounds
        self.aux_optimizer.upper_bounds = self.upper_bounds

        rospy.loginfo('Next ind: %d', self.reward_model.active_inds)
        rospy.loginfo('Current optima: %s', np.array_str(
            self.reward_model.base_input))
        self.loop_counter += 1

    @property
    def round_index(self):
        if self.enable_sequential:
            return len(self.rounds) - self.loop_counter * (self.num_init + self.iters_per_round)
        else:
            return len(self.rounds)

    @property
    def is_initialized(self):
        return self.round_index >= self.num_init

    def pick_initial_sample(self):
        """Selects the next sample for initialization by sampling from
        a random distribution.
        """
        x = np.atleast_1d(self.init_sample_func())
        if self.enable_sequential:
            x = self.reward_model.generate_full_input(x)
        return x

    def get_current_optima(self):
        curr_beta = self.acq_func.exploration_rate
        self.acq_func.exploration_rate = 0
        x, acq = self.aux_optimizer.optimize(x_init=self.aux_x_init,
                                             func=self.acq_func)
        self.acq_func.exploration_rate = curr_beta
        if self.enable_sequential:
            x = self.reward_model.generate_full_input(x)
        return x

    def pick_next_sample(self):
        """Selects the next sample to explore by optimizing the acquisition function.
        """
        self.acq_func.exploration_rate = self.beta_base * \
            math.log(len(self.rounds) + 1)
        x, acq = self.aux_optimizer.optimize(x_init=self.aux_x_init,
                                             func=self.acq_func)
        if self.enable_sequential:
            x = self.reward_model.generate_full_input(x)
        rospy.loginfo('Next sample %s with beta %f and acquisition value %f',
                      str(x), self.acq_func.exploration_rate, acq)
        return x

    def finished(self):
        """Returns whether the optimization is complete.
        """
        return len(self.rounds) >= self.max_evals

    def visualize_rewards(self):
        plt.figure('Reward Visualization')
        plt.gca().clear()

        query_vals = np.linspace(
            start=self.lower_bounds, stop=self.upper_bounds, num=100)
        r_pred, r_std = izip(
            *[self.reward_model.predict(x, return_std=True) for x in query_vals])
        r_pred = np.asarray(r_pred)
        r_std = np.asarray(r_std)
        plt.plot(query_vals, r_pred, 'k-')
        plt.fill_between(query_vals, r_pred - r_std,
                         r_pred + r_std, alpha=0.5, color='b')
        x_past, r_past, _ = izip(*self.rounds)
        plt.plot(x_past, r_past, 'b.')
        plt.draw()

    def optimize_reward_model(self):
        if self.enable_sequential:
            self.base_reward_model.batch_optimize()
        else:
            self.reward_model.batch_optimize()

    def execute(self, interface):
        """Begin execution of the specified problem.
        """
        model_initialized = False
        while not rospy.is_shutdown() and not self.finished():
            rospy.loginfo('Round %d...', len(self.rounds))

            if not self.is_initialized:
                next_sample = self.pick_initial_sample()
                rospy.loginfo('Initializing with %s', str(next_sample))

            else:
                if not model_initialized:
                    self.optimize_reward_model()
                    model_initialized = True

                next_sample = self.pick_next_sample()
                pred_mean, pred_sd = self.reward_model.predict(
                    next_sample, return_std=True)
                rospy.loginfo('Picked sample %s with predicted mean %f +- %f',
                              str(next_sample), pred_mean, pred_sd)

            reward, feedback = interface(next_sample)
            self.reward_model.report_sample(x=next_sample, reward=reward)
            self.rounds.append((next_sample, reward, feedback))

            if self.enable_sequential and \
                    self.round_index >= self.num_init + self.iters_per_round:
                self.update_sequential_ind()
                model_initialized = False

            if self.visualize:
                self.visualize_rewards()


if __name__ == '__main__':
    rospy.init_node('bayesian_optimizer')

    interface_info = rospy.get_param('~interface')
    interface = optim.CritiqueInterface(**interface_info)

    out_path = rospy.get_param('~output_path')
    out_file = open(out_path, 'w')

    load_path = rospy.get_param('~load_path', None)
    if load_path is not None:
        optimizer = pickle.load(open(load_path))
    else:
        optimizer = BayesianOptimizer()

    prog_path = rospy.get_param('~progress_path', None)
    if prog_path is not None:
        prog_file = open(prog_path, 'w')
    else:
        prog_file = None
        rospy.logwarn(
            'No progress path specified. Will not save optimization progress!')

    try:
        optimizer.execute(interface)
    except rospy.ROSInterruptException:
        pass

    # Save progress
    if prog_file is not None:
        pickle.dump(optimizer, prog_file)

    # Save output
    x_best = optimizer.pick_next_sample()
    pickle.dump((x_best, optimizer.rounds), out_file)
