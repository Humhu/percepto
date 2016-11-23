#!/usr/bin/env python

import rospy, sys, cma
import numpy as np
#import cPickle as pickle
import pickle
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from collections import namedtuple

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

        # Specify either input dimension and assume zero mean, or full mean
        if rospy.has_param( '~input_dimension' ):
            in_dim = rospy.get_param( '~input_dimension' )
            init_mean = np.zeros( in_dim )
        else:
            init_mean = rospy.get_param( '~initial_mean' )
        init_stds = rospy.get_param( '~initial_std_dev', 1.0 )

        cma_options = cma.CMAOptions()
        lower = float( rospy.get_param( '~input_lower_bound', '-Inf' ) )
        upper = float( rospy.get_param( '~input_upper_bound', 'Inf' ) )
        cma_options['bounds'] = [lower, upper]

        if rospy.has_param( '~random_seed' ):
            cma_options['seed'] = rospy.get_param( '~random_seed' )
        if rospy.has_param( '~population_size' ):
            cma_options['popsize'] = rospy.get_param( '~population_size' )
        
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
        cma_options['verb_plot'] = 0 # NOTE Fails to pickle!

        self.prog_path = rospy.get_param( '~progress_path', None )

        self.cma_optimizer = cma.CMAEvolutionStrategy( init_mean, init_stds, cma_options )

        # Initialize state
        self.rounds = []
        self.iter_counter = 0

    def Save( self ):
        if self.prog_path is None:
            return
        rospy.loginfo( 'Saving progress at %s...', self.prog_path )
        out = open( self.prog_path, 'wb' )
        pickle.dump( self, out )
        out.close()

    def Resume( self, eval_cb, out_log ):
        """Resumes execution from the current state.
        """
        while not self.cma_optimizer.stop():

            current_inputs = self.cma_optimizer.ask()
            current_outputs = []
            current_feedbacks = []
            # Evaluate all inputs requested by CMA
            for ind,inval in enumerate(current_inputs):
                rospy.loginfo( 'Iteration %d input %d/%d', self.iter_counter,
                                                           ind,
                                                           len( current_inputs ) )
                (curr_outval, curr_feedback) = evaluate_input( proxy=eval_cb, inval=inval )
                current_outputs.append( curr_outval )
                current_feedbacks.append( curr_feedback )

            # CMA feedback and visualization
            self.cma_optimizer.tell( current_inputs, current_outputs )
            self.cma_optimizer.logger.add()
            self.cma_optimizer.disp()
            #cma.show()
            #self.cma_optimizer.logger.plot()

            # Record iteration data
            self.rounds.append( [ self.iter_counter, 
                                  current_inputs, 
                                  current_outputs,
                                  current_feedbacks ] )
            self.iter_counter += 1
            self.Save();

        rospy.loginfo( 'Execution completed!' )
        pickle.dump( self.rounds, out_log )
        out_log.close()

def evaluate_input( proxy, inval, num_retries=1 ):
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
    feedback : list
        List of feedback
    """
    req = GetCritiqueRequest()
    req.input = inval

    for i in range(num_retries+1):
        try:
            res = proxy.call( req )
            break
        except rospy.ServiceException:
            rospy.logerr( 'Could not evaluate item: ' + np.array_str( inval ) )
    
    # Critique is a reward so we have to negate it to get a cost
    cost = -res.critique
    rospy.loginfo( 'Evaluated input: %s\noutput: %f\n feedback: %s', 
                   np.array_str( inval, max_line_width=sys.maxint ),
                   cost,
                   str( res.feedback ) )
    return (cost, res.feedback)

if __name__ == '__main__':
    rospy.init_node( 'cma_cma_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        cmaopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        cmaopt = CMAOptimizer()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )

    # Open output file
    output_path = rospy.get_param( '~output_path' )
    output_log = open( output_path, 'wb' )
    rospy.loginfo( 'Opened output log at %s', output_path )
    
    # Register callback so that the optimizer can save progress
    cmaopt.Resume( eval_cb=critique_proxy, out_log=output_log )
