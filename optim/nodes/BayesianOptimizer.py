#!/usr/bin/env python

import math
from itertools import izip
import dill
import pickle

import numpy as np
import rospy
import optim
import broadcast

import matplotlib.pyplot as plt
from percepto_msgs.srv import RunOptimization, RunOptimizationResponse


class BayesianOptimizer(object):
    """Iteratively samples and updates a reward model (response surface) to optimize a function.

    Uses the GetCritique service to interface with optimization problems.
    """

    def __init__(self):

        # Parse reward model
        model_info = rospy.get_param('~reward_model')
        self.reward_model = optim.parse_reward_model(model_info)

        # Parse input stream if available
        stream_name = rospy.get_param('~input_stream', None)
        self.use_context = (stream_name is not None)

        if self.use_context:
            self.stream_rx = broadcast.Receiver(stream_name)

            # TODO naive contextual mode?
            #self.full_reward_model = self.reward_model
            #self.reward_model = optim.PartialModelWrapper(self.full_reward_model)

            self.contexts = []
            self.acq_func = optim.ContextualUCBAcquisition(model=self.reward_model,
                                                           mode='empirical',
                                                           context=self.contexts)
        else:
            # TODO Parse selection approach + parameters
            self.stream_rx = None
            self.acq_func = optim.UCBAcquisition(self.reward_model)

        self.beta_base = rospy.get_param('~exploration_rate', 1.0)

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

        self.visualize = rospy.get_param('~visualize', False)
        if self.visualize:
            if self.input_dim != 1:
                rospy.logwarn(
                    'Visualization is only enabled for 1D reward model!')
                self.visualize = False
            else:
                plt.ion()

        self.round_index = 0
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

        self.opt_server = rospy.Service('~run_optimization', RunOptimization,
                                        self.opt_callback)

        self.interface = None

    def opt_callback(self, req):
        self.round_index = 0
        if not req.warm_start:
            self.reward_model.clear()

        res = RunOptimizationResponse()
        try:
            res.solution, res.objective = self.execute()
            res.success = True
        except:
            res.success = False
        return res

    def get_context(self):
        if not self.use_context:
            return None

        while not rospy.is_shutdown():
            context = self.stream_rx.read_stream(rospy.Time.now(),
                                                 mode='closest_before')[1]
            if context is None:
                rospy.logerr('Could not read context')
                rospy.sleep(1.0)
            else:
                return context

    @property
    def is_initialized(self):
        return self.round_index >= self.num_init

    def pick_initial_sample(self):
        """Selects the next sample for initialization by sampling from
        a random distribution.
        """
        x = np.atleast_1d(self.init_sample_func())
        return x

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
        self.acq_func.exploration_rate = self.beta_base * \
            math.log(self.round_index + 1)
        x, acq = self.aux_optimizer.optimize(x_init=self.aux_x_init,
                                             func=self.acq_func)
        rospy.loginfo('Next sample %s with beta %f and acquisition value %f',
                      str(x), self.acq_func.exploration_rate, acq)
        return x

    def finished(self):
        """Returns whether the optimization is complete.
        """
        return self.round_index >= self.max_evals

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
        if self.use_context:
            self.full_reward_model.batch_optimize()
        else:
            self.reward_model.batch_optimize()

    def report_sample(self, context, action, reward):
        if self.use_context:
            x = np.hstack((action, context))
            self.reward_model.report_sample(x=x, reward=reward)
        else:
            self.reward_model.report_sample(x=action, reward=reward)

    def execute(self):
        """Begin execution of the specified problem.
        """
        model_initialized = False
        while not rospy.is_shutdown() and not self.finished():
            rospy.loginfo('Round %d...', self.round_index)

            context = self.get_context()
            if self.stream_rx is not None:
                a_init = np.zeros(self.input_dim)
                self.reward_model.base_input = np.hstack((a_init, context))
                self.reward_model.active_inds = range(self.input_dim)

            if not self.is_initialized:
                action = self.pick_initial_sample()
                rospy.loginfo('Initializing with %s', str(action))

            else:
                if not model_initialized:
                    self.optimize_reward_model()
                    model_initialized = True

                action = self.pick_action()
                pred_mean, pred_sd = self.reward_model.predict(action,
                                                               return_std=True)
                rospy.loginfo('Picked sample %s with predicted mean %f +- %f',
                              str(action), pred_mean, pred_sd)

            reward, feedback = self.interface(action)
            self.report_sample(context=context, action=action, reward=reward)
            self.rounds.append((context, action, reward, feedback))
            self.round_index += 1

            if self.visualize:
                self.visualize_rewards()

        return optimizer.get_current_optima()


if __name__ == '__main__':
    rospy.init_node('bayesian_optimizer')

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

    interface_info = rospy.get_param('~interface')
    optimizer.interface = optim.CritiqueInterface(**interface_info)

    run_on_start = rospy.get_param('run_on_start', True)
    try:
        if run_on_start:
            res = optimizer.execute()
        else:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass

    # Save progress
    if prog_file is not None:
        optimizer.interface = None
        pickle.dump(optimizer, prog_file)

    # Save output
    pickle.dump((res, optimizer.rounds), out_file)
