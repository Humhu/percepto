#!/usr/bin/env python

import dill
import cPickle as pickle
import rospy, sys, math
import numpy as np
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse
from collections import deque

# TODO Update
class CrossEntropyOptimizer:
    """Cross entropy numerical optimizer node.

    Interfaces with an optimization problem through the GetCritique service.
    """

    def __init__( self ):

        self.mode = rospy.get_param( '~mode' )
        if self.mode != 'minimization' and self.mode != 'maximization':
            raise ValueError( '~mode must be minimization or maximization' )

        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            np.random.seed( seed )

        # Specify either input dimension and assume zero mean, or full mean
        init_mean = rospy.get_param( '~initial_mean', 0.0 )
        init_stds = rospy.get_param( '~initial_std_dev', 1.0 )
        x_upper = rospy.get_param( '~input_upper_bound', float('Inf') )
        x_lower = rospy.get_param( '~input_lower_bound', -float('Inf') )
        if rospy.has_param( '~input_dimension' ):
            in_dim = rospy.get_param( '~input_dimension' )
            init_mean = (init_mean,) * in_dim
            init_stds = (init_stds,) * in_dim
            x_upper = (x_upper,) * in_dim
            x_lower = (x_lower,) * in_dim
        self.mean = np.array( init_mean )
        self.cov = np.diag( init_stds ) ** 2
        self.x_upper_lim = np.array( x_upper )
        self.x_lower_lim = np.array( x_lower )

        # Read optimization parameters
        self.population_size = rospy.get_param( '~population_size' )
        self.elite_size = rospy.get_param( '~elite_size' )
        self.inflation_scale = rospy.get_param( '~inflation_scale', 1.0 )
        self.inflation_offset = float( rospy.get_param( '~inflation_offset', 0.0 ) ) * \
                                np.identity( self.mean.shape[0] )
        self.inflation_decay_rate = float( rospy.get_param( '~inflation_decay_rate', 1.0 ) )



        self.diagonal_only = rospy.get_param( '~diagonal_only', True )
        self.elite_lifespan = rospy.get_param( '~elite_lifespan', 4 )
        
        # Read convergence criteria
        self.max_evals = rospy.get_param( '~convergence/max_evaluations', float('Inf') )
        self.max_iters = rospy.get_param( '~convergence/max_iterations', float('Inf') )
        self.x_tol = float(rospy.get_param( '~convergence/input_tolerance', -float('Inf') ) )
        self.y_tol = float(rospy.get_param( '~convergence/output_tolerance', -float('Inf') ) )
        self.y_tol_iters = rospy.get_param( '~convergence/output_tolerance_iterations', 10 )

        self.rounds = []

        # Initialize state
        self.iter_counter = 0
        self.eval_counter = 0
        self.y_hist = deque()

        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def save( self, status ):
        if self.prog_path is not None:
            rospy.loginfo( 'Saving progress at %s...', self.prog_path )
            prog = open( self.prog_path, 'wb' )
            pickle.dump( self, prog )
            prog.close()

        rospy.loginfo( 'Saving output at %s...', self.out_path )
        out = open( self.out_path, 'wb' )
        pickle.dump( (status, self.rounds), out )
        out.close()

    def is_done( self, x_vals, y_vals ):

        # Check evals and iters
        if self.eval_counter > self.max_evals:
            return {'max_evaluations' : self.max_evals}
        if self.iter_counter > self.max_iters:
            return {'max_iterations' : self.max_iters}

        # Check x tolerance
        if x_vals is not None:
            x_stds = np.std( x_vals, axis=0 )
            if np.all( x_stds < self.x_tol ):
                return {'input_tolerance' : self.x_tol}

        # Check y tolerance
        if y_vals is not None:
            y_mean = np.mean( y_vals )
            self.y_hist.append( y_mean )

            if len( self.y_hist ) >= self.y_tol_iters:
                y_range = max(self.y_hist) - min(self.y_hist)
                if y_range < self.y_tol:
                    return {'output_tolerance' : self.y_tol}

        return {}

    def ask( self ):
        """Brute force bounded Gaussian sampling.
        """
        samples = []
        while len(samples) < self.population_size:
            x = np.random.multivariate_normal( self.mean, self.cov )
            if np.all( x <= self.x_upper_lim ) and np.any( x >= self.x_lower_lim ):
                samples.append( x )
        return samples

    def execute( self, eval_cb ):

        last_inputs = None
        last_outputs = None
        while not rospy.is_shutdown() and not self.is_done( last_inputs, last_outputs ):

            # Execute round
            current_inputs = self.ask()
            current_outputs = []
            current_round = []
            for (ind,x) in enumerate(current_inputs):
                rospy.loginfo('Iteration %d input %d/%d', self.iter_counter,
                                                          ind,
                                                          len( current_inputs) )
                (reward, feedback) = eval_cb( x )
                current_round.append( (x,reward,feedback) )
                current_outputs.append(reward)
            self.rounds.append( current_round )

            # Get live elites from history
            if self.elite_lifespan > len( self.rounds ):
                generations = self.rounds
            else:
                generations = self.rounds[-self.elite_lifespan:]

            live_population = []
            for generation in self.rounds[-self.elite_lifespan:]:
                live_population += generation


            # Sort to find elites
            # Note: python sort is default ascending order 
            sort_key = lambda x : x[1]
            if self.mode == 'minimization':
                live_population.sort( key=sort_key, reverse=False )
            else: # maximization
                live_population.sort( key=sort_key, reverse=True )
            elite_inputs = [ sample[0] for sample in live_population[0:self.elite_size] ]

            elite_mean = np.mean( elite_inputs, axis=0 )
            elite_cov = np.cov( elite_inputs, rowvar=0 )

            # Apply offsets
            # Both the offset and the scale decay towards 0 and 1, respectively, over root t
            inflation_factor = math.exp( -self.iter_counter * self.inflation_decay_rate );
            scale = 1.0 + inflation_factor * ( self.inflation_scale - 1.0 )
            offset = inflation_factor * self.inflation_offset

            self.mean = elite_mean
            self.cov = elite_cov * scale + offset

            if self.diagonal_only:
                self.cov = np.diag( np.diag( self.cov ) )

            # Print updates
            new_stds = np.sqrt( np.diag( self.cov ) )
            rospy.loginfo( 'Iteration: %d Inflation factor: %f\nMean: %s\nSD:%s' % 
                           ( self.iter_counter,
                             inflation_factor,
                             np.array_str( self.mean, max_line_width=sys.maxint ),
                             np.array_str( new_stds, max_line_width=sys.maxint ) ) )

            # Bookkeeping for convergence checks
            self.iter_counter += 1
            self.eval_counter += len( current_inputs )
            last_inputs = current_inputs
            last_outputs = current_outputs

            self.save(status='In Progress')

        self.save(status=self.is_done( last_inputs, last_outputs ) )

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
    reward : numeric
        The reward of the input values
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
    
    reward = res.critique
    rospy.loginfo( 'Evaluated input: %s\noutput: %f\nfeedback: %s', 
                   np.array_str( inval, max_line_width=sys.maxint ),
                   reward,
                   str( res.feedback ) )
    return (reward, res.feedback)

if __name__ == '__main__':
    rospy.init_node( 'cross_entropy_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        ceopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        ceopt = CrossEntropyOptimizer()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input( critique_proxy, x )

    # Register callback so that the optimizer can save progress
    ceopt.execute( eval_cb=eval_cb )