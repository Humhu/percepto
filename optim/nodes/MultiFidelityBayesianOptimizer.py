#!/usr/bin/env python

from itertools import izip
import dill
import pickle

import numpy as np
import rospy
import optim

from percepto_msgs.srv import RunOptimization, RunOptimizationResponse


class MultiFidelityBayesianOptimizer(object):
    """Iteratively samples and updates a reward model (response surface) to optimize a stochastic function.

    Uses the GetCritique service to interface with optimization problems.
    """

    def __init__(self):

        # Parse reward model
        model_info = rospy.get_param('~reward_model')
        self.reward_model = optim.parse_mf_reward_model(model_info)
        self.gammas = rospy.get_param('~fidelity_gammas')

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

        self.acq_func = optim.MultiFidelityUCBAcquisition(self.reward_model)

        alpha = rospy.get_param('~exploration_rate_alpha', 0.2)
        gamma = rospy.get_param('~exploration_rate_gamma', 2.0)
        self.exploration_rate = optim.UCBExplorationRate(dim=self.input_dim,
                                                         alpha=alpha,
                                                         gamma=gamma)

        self.model_initialized = False
        self.init_buffer = []
        self.rounds = []
        init_info = rospy.get_param('~initialization')
        self.num_init = init_info['num_samples']
        init_method = init_info['method']

        # TODO Support for latin hypercubes, other approaches?
        if init_method == 'uniform':
            self.init_sample_func = lambda: np.random.uniform(self.lower_bounds,
                                                              self.upper_bounds)
        else:
            raise ValueError(
                'Invalid initial distribution method %s' % init_method)

        self.max_evals = rospy.get_param('~convergence/max_evaluations')
        self.conv_check_iters = rospy.get_param('~convergence/eps_window', 5)
        self.conv_action_eps = rospy.get_param('~convergence/x_tol', 1E-3)

        self.opt_server = rospy.Service('~run_optimization', RunOptimization,
                                        self.opt_callback)

        self.interface = None

    def opt_callback(self, req):
        if not req.warm_start:
            self.reward_model.clear()

        res = RunOptimizationResponse()
        try:
            res.solution, res.objective = self.execute()
            res.success = True
        except:
            res.success = False
        return res

    @property
    def is_initializing(self):
        return len(self.init_buffer) < self.num_init

    def pick_initial_sample(self):
        """Selects the next sample for initialization by sampling from
        a random distribution.
        """
        x = np.atleast_1d(self.init_sample_func())
        return np.hstack((0, x))

    def get_current_optima(self):
        curr_beta = self.acq_func.exploration_rate
        self.acq_func.exploration_rate = 0
        x, acq = self.aux_optimizer.optimize(x_init=self.aux_x_init,
                                             func=self.acq_func)
        self.acq_func.exploration_rate = curr_beta
        return x, acq

    def pick_action(self):
        """Selects the next sample to explore by optimizing the acquisition function.
        """
        self.acq_func.exploration_rate = self.exploration_rate(
            len(self.rounds) + 1)

        fid, x = optim.pick_acquisition_mf(acq_func=self.acq_func,
                                           optimizer=self.aux_optimizer,
                                           gammas=self.gammas,
                                           x_init=self.aux_x_init)
        rospy.loginfo('Next sample (%d, %s) with beta %f',
                      fid, str(x), self.acq_func.exploration_rate)
        rmean, rsd = self.acq_func.predict_mf(fid=fid, x=x)
        return np.hstack((fid, x)), rmean, rsd

    def finished(self):
        """Returns whether the optimization is complete.
        """
        hit_max_evals = len(self.rounds) >= self.max_evals

        if len(self.rounds) < self.conv_check_iters:
            hit_conv = False
        else:
            last_rounds = self.rounds[-self.conv_check_iters:]
            _, _, a, r, f = zip(*last_rounds)
            a_sd = np.std(a, axis=0)
            hit_conv = (a_sd < self.conv_action_eps).all()

        return hit_max_evals or hit_conv

    def initialize_reward_model(self):
        if self.model_initialized:
            raise RuntimeError('Model already initialized!')

        for a, r, f in self.init_buffer:
            self.report_sample(action=a, reward=r, feedback=f)

        # TODO Do we need this?
        self.reward_model.batch_optimize()
        self.model_initialized = True

    def report_initialization(self, action, reward, feedback):
        self.init_buffer.append((action, reward, feedback))

        if not self.is_initializing:
            self.initialize_reward_model()

    def report_sample(self, action, reward, feedback):
        raw_context = None

        self.reward_model.report_sample(x=action, reward=reward)
        self.rounds.append((None, None, action, reward, feedback))

    def execute(self):
        """Begin execution of the specified problem.
        """
        while not rospy.is_shutdown() and not self.finished():

            if self.is_initializing:
                rospy.loginfo('Init %d/%d...',
                              len(self.init_buffer)+1, self.num_init)
            else:
                rospy.loginfo('Round %d/%d...',
                              len(self.rounds)+1, self.max_evals)

            # Decide what to do
            if self.is_initializing:
                action = self.pick_initial_sample()
                rospy.loginfo('Initializing with %s', str(action))
            else:
                action, rmean, rsd = self.pick_action()
                rospy.loginfo('Picked sample %s with predicted mean %f +- %f',
                              str(action), rmean, rsd)

            # Do it
            reward, feedback = self.interface(action)

            # Record the results
            if self.is_initializing:
                self.report_initialization(action=action,
                                           reward=reward,
                                           feedback=feedback)
            else:
                self.report_sample(action=action,
                                   reward=reward,
                                   feedback=feedback)

        return optimizer.get_current_optima()


if __name__ == '__main__':
    rospy.init_node('multi_fidelity_bayesian_optimizer')
    out_path = rospy.get_param('~output_path')
    out_file = open(out_path, 'w')

    load_path = rospy.get_param('~load_path', None)
    if load_path is not None:
        optimizer = pickle.load(open(load_path))
    else:
        optimizer = MultiFidelityBayesianOptimizer()

    prog_path = rospy.get_param('~progress_path', None)
    if prog_path is not None:
        prog_file = open(prog_path, 'w')
    else:
        prog_file = None
        rospy.logwarn(
            'No progress path specified. Will not save optimization progress!')

    interface_info = rospy.get_param('~interface')
    optimizer.interface = optim.CritiqueInterface(**interface_info)

    run_on_start = rospy.get_param('~run_on_start', False)
    res = 'incomplete'
    #try:
    if run_on_start:
        res = optimizer.execute()
    else:
        rospy.spin()
    #except rospy.ROSInterruptException:
    #    pass

    # Save progress
    if prog_file is not None:
        optimizer.interface = None
        pickle.dump(optimizer, prog_file)

    # Save output
    pickle.dump((res, optimizer.rounds), out_file)
