#!/usr/bin/env python

import dill
import pickle
#import cPickle as pickle
import rospy
import sys
import cma
import numpy as np
import optim
from percepto_msgs.srv import RunOptimization, RunOptimizationResponse


class CMAOptimizer:
    """Covariance Matrix Adaptation (CMA) evolution strategy cma_optimizer. 

    Wraps a python CMA implementation to allow midway termination and resuming 
    for use with systems that may require downtime, ie. a robot that needs recharging.

    Interfaces with an optimization problem through the GetCritique service.

    ROS Parameters
    --------------
    #TODO Update
    ~initial_mean : N-length numeric array
        The CMA search initial mean. The dimensionality N of the optimization input 
        is inferred from this array.

    ~initial_std_dev : N-length numeric array or scalar (default 1.0)
        The CMA search initial standard deviations. If an array it must be N elements,
        one for each input dimension.

    ~output_path : string file path
        The file to pickle the execution trace and results to. If interrupted, 
        partial execution data is saved to output_path.partial

    ~resume_data_path: string file path (optional)
        If defined, execution will resume from the specified partial execution data
        dump file.

    ~random_seed : integer (optional, default system time)
        The random number generator seed. Defaults to system time if not specified.

    ~population_size : integer (optional, default 4+int(3*log(N)))
        The number of samples to collect at each iteration.

    ~diagonal_only : boolean, (optional, default true after 0*100*N/sqrt(popsize) iters)

    ~bounds : length 2 numeric array (optional, default [None,None])
        Lower and upper bounds for input dimensions.

    ~convergence/max_iters : numeric (default Inf)

    ~convergence/max_time : numeric (default Inf)

    ~convergence/output_change : numeric (default -Inf)

    ~convergence/input_change : numeric (default -Inf)
   """

    def __init__(self):

        # Specify either input dimension and assume zero mean, or full mean
        init_mean = rospy.get_param('~initial_mean', 0.0)
        if rospy.has_param('~input_dim'):
            in_dim = rospy.get_param('~input_dim')
            init_mean = init_mean * np.ones(in_dim)
        else:
            init_mean = np.array(init_mean)
        init_stds = rospy.get_param('~initial_std_dev', 1.0)

        cma_options = cma.CMAOptions()
        lower = float(rospy.get_param('~input_lower_bound', '-Inf'))
        upper = float(rospy.get_param('~input_upper_bound', 'Inf'))
        cma_options['bounds'] = [lower, upper]

        if rospy.has_param('~random_seed'):
            cma_options['seed'] = rospy.get_param('~random_seed')
        if rospy.has_param('~population_size'):
            cma_options['popsize'] = rospy.get_param('~population_size')

        diag_only = rospy.get_param('~diagonal_only', False)
        if diag_only:
            cma_options['CMA_diagonal'] = True

        # Set convergence criteria
        cma_options['maxfevals'] = float(rospy.get_param(
            '~convergence/max_evaluations', float('Inf')))
        cma_options['maxiter'] = float(rospy.get_param(
            '~convergence/max_iterations', float('Inf')))
        cma_options['tolfun'] = float(rospy.get_param(
            '~convergence/output_tolerance', -float('Inf')))
        cma_options['tolfunhist'] = float(rospy.get_param(
            '~convergence/output_history_tolerance', -float('Inf')))
        cma_options['tolx'] = float(rospy.get_param(
            '~convergence/input_tolerance', -float('Inf')))

        # TODO
        verbose = rospy.get_param('~verbose', False)
        cma_options['verb_log'] = 0
        if not verbose:
            cma_options['verbose'] = -9
        else:
            cma_options['verb_disp'] = 1
        #cma_options['verb_disp'] = 1
        # cma_options['verb_plot'] = 0 # NOTE Fails to pickle!

        self.prog_path = rospy.get_param('~progress_path', None)
        self.out_path = rospy.get_param('~output_path')

        def init_cma():
            return cma.CMAEvolutionStrategy(init_mean,
                                            init_stds,
                                            cma_options)
        self.create_cma = init_cma
        self.cma_optimizer = None

        # Initialize state
        self.rounds = []
        self.round_index = 0
        self.interface = None

        self.opt_server = rospy.Service('~run_optimization', RunOptimization,
                                        self.opt_callback)

    def opt_callback(self, req):
        self.round_index = 0
        if not req.warm_start:
            # NOTE Probably never want to keep the optimizer?
            self.cma_optimizer = None

        res = RunOptimizationResponse()
        try:
            res.solution, res.objective = self.execute()
            res.success = True
        except:
            res.success = False
        return res

    def execute(self):
        """Resumes execution from the current state.
        """
        if self.cma_optimizer is None:
            self.cma_optimizer = self.create_cma()

        while not rospy.is_shutdown() and not self.cma_optimizer.stop():

            current_inputs = self.cma_optimizer.ask()
            current_outputs = []
            current_feedbacks = {}
            # Evaluate all inputs requested by CMA
            for ind, inval in enumerate(current_inputs):
                rospy.loginfo('Iteration %d input %d/%d', self.round_index,
                              ind,
                              len(current_inputs))
                curr_outval, curr_feedback = self.interface(inval)
                current_outputs.append(curr_outval)
                for k, v in curr_feedback.iteritems():
                    if k not in current_feedbacks:
                        current_feedbacks[k] = []
                    current_feedbacks[k].append(v)

            # cma performs minimization, so we have to report the negated
            # rewards
            self.cma_optimizer.tell(current_inputs, -np.array(current_outputs))
            self.cma_optimizer.logger.add()
            self.cma_optimizer.disp()

            # Record iteration data
            self.rounds.append([current_inputs,
                                current_outputs,
                                current_feedbacks])
            self.round_index += 1

        return self.cma_optimizer.result()[0], self.cma_optimizer.result()[1]


if __name__ == '__main__':
    rospy.init_node('cma_optimizer')

    out_path = rospy.get_param('~output_path')
    out_file = open(out_path, 'w')

    load_path = rospy.get_param('~load_path', None)
    if load_path is not None:
        optimizer = pickle.load(open(load_path))
    else:
        optimizer = CMAOptimizer()

    prog_path = rospy.get_param('~progress_path', None)
    if prog_path is not None:
        prog_file = open(prog_path, 'w')
    else:
        prog_file = None
        rospy.logwarn(
            'No progress path specified. Will not save optimization progress!')

    # Create interface to optimization problem
    interface_info = rospy.get_param('~interface')
    optimizer.interface = optim.CritiqueInterface(**interface_info)

    run_on_start = rospy.get_param('~run_on_start', False)
    try:
        if run_on_start:
            rospy.loginfo('Running on start...')
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
