#!/usr/bin/env python

import rospy, sys, cma
import numpy as np
import cPickle as pickle
from threading import Lock
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from collections import namedtuple

def get_optional_param( self, ros_key, ref ):
    if rospy.has_param( ros_key ):
        ref = rospy.get_param( ros_key )

class CMAOptimizer:
    """Covariance Matrix Adaptation (CMA) evolution strategy cma_optimizer. 

    Wraps a python CMA implementation to allow midway termination and resuming 
    for use with systems that may require downtime, ie. a robot that needs recharging.
    
    Interfaces with an optimization problem through the GetCritique service.

    ROS Parameters
    --------------
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

    def __init__( self ):

        self.lock = Lock()

        # I/O initialization
        self.output_path = rospy.get_param( '~output_path' )
        self.output_log = open( self.output_path, 'wb' )
        if self.output_log is None:
            raise RuntimeError( 'Could not open output log at path: ' + self.output_path )

        # Specify either input dimension and assume zero mean, or full mean
        if rospy.has_param( '~input_dimension' ):
            in_dim = rospy.get_param( '~input_dimension' )
            init_mean = np.zeros( in_dim )
        else:
            init_mean = rospy.get_param( '~initial_mean' )
        init_stds = rospy.get_param( '~initial_std_dev', 1.0 )

        lower = float( rospy.get_param( '~input_lower_bound', '-Inf' ) )
        upper = float( rospy.get_param( '~input_upper_bound', 'Inf' ) )
        cma_options['bounds'] = [lower, upper]

        cma_options = cma.CMAOptions()
        get_optional_param( ros_key='~random_seed', ref=cma_options['seed'] )
        get_optional_param( ros_key='~population_size', ref=cma_options['popsize'] )
        
        diag_only = rospy.get_param( '~diagonal_only', False )
        if diag_only:
            cma_options['CMA_diagonal'] = True

        # Set convergence criteria
        cma_options['maxfevals'] = float( rospy.get_param( '~convergence/max_evaluations', float('Inf') ) )
        cma_options['maxiter'] = float( rospy.get_param( '~convergence/max_iterations', float('Inf') ) )
        cma_options['tolfun'] = float( rospy.get_param( '~convergence/output_change', -float('Inf') ) )
        cma_options['tolx'] = float( rospy.get_param( '~convergence/input_change', -float('Inf') ) )

        # TODO Make an option
        cma_options['verb_disp'] = 1
        cma_options['verb_plot'] = 1

        self.cma_optimizer = cma.CMAEvolutionStrategy( init_mean, init_stds, cma_options )
        
    def __del__( self ):
        if self.output_log is not None:
            self.output_log.close()

    def PauseExecution( self ):
        """Pause execution and save state to a file.
        """
        rospy.loginfo('Pausing execution and saving state to ' + self.output_path )
        with self.lock:
            pickle.dump( self.exec_state, self.output_log )

    def Resume( self, eval_cb ):
        """Resumes execution from the current state.
        """
        while not self.cma_optimizer.stop():

            if self.current_inputs is None:
                self.current_inputs = self.cma_optimizer.ask()
                self.current_outputs = []

            # Evaluate all inputs requested by CMA
            while len( self.current_outputs ) < len( self.current_inputs ):
                input_ind = len( self.current_outputs )
                curr_inval = self.current_inputs[ input_ind ]
                rospy.loginfo( 'Iteration %d input %d/%d', self.iter_counter,
                                                           input_ind,
                                                           len( self.current_inputs ) )

                with self.lock:
                    curr_outval = evaluate_input( proxy=eval_cb, inval=curr_inval )
                    self.current_outputs.append( curr_outval )

            with self.lock:
                # CMA feedback and visualization
                self.cma_optimizer.tell( inputs, outputs )
                self.cma_optimizer.logger.add()
                self.cma_optimizer.disp()
                cma.show()
                self.cma_optimizer.logger.plot()

                # Record iteration data
                self.rounds.append( [ self.iter_counter, 
                                      self.current_inputs, 
                                      self.current_outputs ] )

                # Reset state for next iteration
                self.current_inputs = None
                self.iter_counter += 1

        rospy.loginfo( 'Execution completed!' )
        pickle.dump( self.output_log, self.rounds )

def evaluate_input( proxy, inval ):
    """Query the optimization function.

    Parameters
    ----------
    proxy : rospy.ServiceProxy
        Service proxy to call the GetCritique service for evaluation.

    inval : numeric array
        Input values to evaluate.

    Return
    ------
    cost : numeric
        The cost of the input values
    """
    req = GetCritiqueRequest()
    req.input = inval

    with self.lock:
        try:
            res = proxy.call( req )
        except rospy.ServiceException:
            raise RuntimeError( 'Could not evaluate item: ' + PrintArray( inval ) )
    
    # Critique is a reward so we have to negate it to get a cost
    cost = -res.critique
    rospy.loginfo( 'Evaluated input: %s output: %f', 
                   np.array_str( inval, max_line_width=sys.maxint ),
                   cost )
    return cost

if __name__ == '__main__':
    rospy.init_node( 'cma_cma_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~resume_data_path' ):
        data_path = rospy.get_param( '~resume_data_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found resume data at %s...', data_path )
        cmaopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        cmaopt = CMAOptimizer()

    # Register callback so that the optimizer can save progress
    rospy.on_shutdown( cmaopt.PauseExecution )

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )

    cmaopt.Resume( critique_proxy )
