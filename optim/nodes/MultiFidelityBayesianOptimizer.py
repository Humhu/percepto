#!/usr/bin/env python

from itertools import izip
import dill
import pickle
import time
import math

import numpy as np
import scipy as sp
import rospy
import optim

from percepto_msgs.srv import RunOptimization, RunOptimizationResponse


def farr(x, n):
    if hasattr(x, '__iter__'):
        return np.array([float(xi) for xi in x])
    else:
        return np.full(n, float(x))


class MultiFidelityBayesianOptimizer(object):
    """Iteratively samples and updates a reward model (response surface) to optimize a stochastic function.

    Uses the GetCritique service to interface with optimization problems.
    """

    def __init__(self):

        # Parse reward model
        model_info = rospy.get_param('~reward_model')
        self.reward_model = optim.parse_mf_reward_model(model_info)
        self.gammas = farr(rospy.get_param('~fidelity_gammas'),
                           self.reward_model.num_fidelities)
        self.gamma_power = float(rospy.get_param('~gamma_power', 2.0))
        self.fidelity_costs = rospy.get_param('~fidelity_costs')

        # Parse acquisition optimizer
        optimizer_info = rospy.get_param('~auxiliary_optimizer')
        self.aux_optimizer = optim.parse_optimizers(optimizer_info)

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
        self.active_fidelities = []
        self.num_init = rospy.get_param('~initialization/num_samples', 0)
        self.init_variation = rospy.get_param('~initialization/min_variation', 0)
        init_method = rospy.get_param('~initialization/method', 'uniform')

        # TODO Support for latin hypercubes, other approaches?
        if init_method == 'uniform':
            self.init_sample_func = lambda: np.random.uniform(self.lower_bounds,
                                                              self.upper_bounds)
        else:
            raise ValueError('Invalid initial distribution method ' + init_method)

        self.max_evals = rospy.get_param('~convergence/max_evaluations', float('inf'))
        self.max_time = rospy.get_param('~convergence/max_time', float('inf'))
        self.conv_check_iters = rospy.get_param('~convergence/eps_window', 5)
        self.conv_action_eps = rospy.get_param('~convergence/x_tol', 0)

        self.opt_server = rospy.Service('~run_optimization', RunOptimization,
                                        self.opt_callback)

        self.interface = None
        self.duration = 0

    def opt_callback(self, req):
        if not req.warm_start:
            self.reward_model.clear()

        res = RunOptimizationResponse()
        try:
            # TODO
            self.execute()
            #res.solution, res.objective = self.execute()
            res.success = True
        except:
            res.success = False
        return res

    @property
    def done_initializing(self):
        enough_samples = len(self.init_buffer) >= self.num_init

        if enough_samples:
            _, _, init_r, _ = zip(*self.init_buffer)
            r = np.reshape(init_r, (-1, 1))
            variation = sp.spatial.distance.pdist(r)
            enough_variation = np.max(variation) > self.init_variation
        else:
            enough_variation = False

        return enough_samples and enough_variation

    def pick_initial_sample(self):
        """Selects the next sample for initialization by sampling from
        a random distribution.
        """
        x = np.atleast_1d(self.init_sample_func())
        return 0, x

    def pick_action(self):
        """Selects the next sample to explore by optimizing the acquisition function.
        """
        self.acq_func.exploration_rate = self.exploration_rate(
            len(self.rounds) + 1)

        fid, x = optim.pick_acquisition_mf(acq_func=self.acq_func,
                                           optimizer=self.aux_optimizer,
                                           gammas=self.gammas,
                                           x_init=self.aux_x_init)
        rmean, rsd = self.acq_func.predict_mf(fid=fid, x=x)
        rospy.loginfo('Next sample (%d, %s) with beta %f and predicted reward %f +- %f',
                      fid,
                      str(x), self.acq_func.exploration_rate,
                      rmean,
                      rsd)
        return fid, x

    def finished(self):
        """Returns whether the optimization is complete.
        """
        hit_max_evals = len(self.rounds) >= self.max_evals

        if len(self.rounds) < self.conv_check_iters:
            hit_conv = False
        else:
            last_rounds = self.rounds[-self.conv_check_iters:]
            i, a, r, f = zip(*last_rounds)
            a_sd = np.std(a, axis=0)
            hit_conv = (a_sd < self.conv_action_eps).all()

        hit_max_time = self.duration > self.max_time

        return hit_max_evals or hit_conv or hit_max_time

    def initialize_reward_model(self):
        if self.model_initialized:
            raise RuntimeError('Model already initialized!')

        for i, a, r, f in self.init_buffer:
            self.report_sample(fid=i, action=a, reward=r, feedback=f)

        self.model_initialized = True
        self.active_fidelities = [] # HACK sort of

    def report_initialization(self, fid, action, reward, feedback):
        self.init_buffer.append((fid, action, reward, feedback))

        if self.done_initializing:
            self.initialize_reward_model()

    def report_sample(self, fid, action, reward, feedback):
        self.reward_model.report_mf_sample(fid=fid,
                                           x=action,
                                           reward=reward)
        self.rounds.append((fid, action, reward, feedback))
        self.active_fidelities.append(fid)

        # Check fidelity gamma for all sub-fidelities
        if fid >= self.reward_model.num_fidelities - 1:
            return

        query_ratio = self.fidelity_costs[fid + 1] / self.fidelity_costs[fid]
        query_ratio = int(math.ceil(query_ratio))
        n_rounds = min((len(self.active_fidelities), query_ratio))
        prev_streak = np.sum(self.active_fidelities[-n_rounds:] == fid)
        if prev_streak >= query_ratio:
            self.gammas[fid] *= self.gamma_power
            rospy.loginfo('Queried fidelity %d for %d rounds in a row. Increasing gamma to %f',
                          fid, prev_streak, self.gammas[fid])

    def execute_action(self, fid, action):
        full_action = np.hstack((fid, action))

        start_t = time.time()
        ret = self.interface(full_action)
        end_t = time.time()
        eval_dur = end_t - start_t
        self.duration += eval_dur
        return ret

    def execute(self):
        """Begin execution of the specified problem.
        """
        while not rospy.is_shutdown() and not self.finished():

            if not self.done_initializing:
                rospy.loginfo('Init %d...',
                              len(self.init_buffer) + 1)

                fid, action = self.pick_initial_sample()
                reward, feedback = self.execute_action(fid=fid,
                                                       action=action)

                self.report_initialization(fid=fid,
                                           action=action,
                                           reward=reward,
                                           feedback=feedback)
            else:
                rospy.loginfo('Round %d, %f s elapsed',
                              len(self.rounds) + 1,
                              self.duration)

                fid, action = self.pick_action()
                reward, feedback = self.execute_action(fid=fid,
                                                       action=action)

                self.report_sample(fid=fid,
                                   action=action,
                                   reward=reward,
                                   feedback=feedback)

                # Check to see if we need to update biases
                if not self.reward_model.check_biases(fid=fid,
                                                      x=action,
                                                      reward=reward):
                    lower_r, lower_f = self.execute_action(fid=fid - 1,
                                                           action=action)
                    self.reward_model.update_bias(x=action,
                                                  fid_hi=fid,
                                                  reward_hi=reward,
                                                  fid_lo=fid - 1,
                                                  reward_lo=lower_r)
                    self.report_sample(fid=fid - 1,
                                       action=action,
                                       reward=lower_r,
                                       feedback=lower_f)

        return None


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
    # try:
    if run_on_start:
        res = optimizer.execute()
    else:
        rospy.spin()
    # except rospy.ROSInterruptException:
    #    pass

    # Save progress
    if prog_file is not None:
        optimizer.interface = None
        pickle.dump(optimizer, prog_file)

    # Save output
    pickle.dump((res, optimizer.rounds), out_file)
