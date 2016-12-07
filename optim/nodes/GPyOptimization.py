#!/usr/bin/env python

import dill
import pickle
import GPy, GPyOpt
import rospy, sys
import numpy as np
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse

class BayesianOptimizer:
    """Bayesian optimization (BO) optimizer.

    Uses the GPyOpt library: https://github.com/SheffieldML/GPyOpt

    Interfaces with an optimization problem through the GetCritique service.
    """

    def __init__( self ):

        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            np.random.seed( seed )

        input_dim = rospy.get_param( '~input_dimension' )
        lower = rospy.get_param( '~input_lower_bound' )
        upper = rospy.get_param( '~input_upper_bound' )
        if not np.iterable( lower ):
            lower = (lower,) * input_dim
        if not np.iterable( upper ):
            upper = (upper,) * input_dim

        # Construct bounds
        self.domain_dict = []
        for i in range( input_dim ):
            name = 'var_%d' % i
            bounds = (lower[i], upper[i])
            entry = { 'name' : name, 'type' : 'continuous', 'domain' : bounds }
            self.domain_dict.append( entry )

        self.model = rospy.get_param( '~optimizer/model', 'GP' )
        self.norm_y = rospy.get_param( '~optimizer/normalize_output', False )
        self.init_samples = rospy.get_param( '~optimizer/initial_samples', 10 )
        self.init_type = rospy.get_param( '~optimizer/initial_strategy', 'random' )
        self.exact = not rospy.get_param( '~optimizer/noiseless', False )
        self.acq_type = rospy.get_param( '~optimizer/acquisition_type', 'EI' )

        self.x_tol = rospy.get_param( '~convergence/input_tolerance', -float('inf') )
        self.max_evals = rospy.get_param( '~convergence/max_evaluations', float('inf') )
        self.evals = 0

        self.rounds = []
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )
        self.critique_topic = rospy.get_param( '~critic_service' )

    def initialize( self ):

        if len( self.rounds ) == 0:
            X = None
            Y = None
        else:
            X = []
            Y = []
            for r in self.rounds:
                X.append( r[0] )
                Y.append( r[1] )
            X = np.atleast_2d( X )
            Y = np.atleast_2d( Y )

            # Can't seem to make it initialize with less than 1...
            self.init_samples = 1

        self.optimizer = GPyOpt.methods.BayesianOptimization( self.objective_wrapper,
                                                              X = X,
                                                              Y = Y,
                                                              domain = self.domain_dict,
                                                              model = self.model,
                                                              normalize_y = self.norm_y,
                                                              initial_design_numdata = self.init_samples,
                                                              initial_design_type = self.init_type,
                                                              exact_feval = self.exact,
                                                              acquisition_type = self.acq_type )

    def execute( self ):
        # We track historical evals so resuming works correctly
        self.optimizer.run_optimization( max_iter = self.max_evals - self.evals, eps = self.x_tol )
        self.save( self.optimizer.x_opt )

    def objective_wrapper( self, xs ):
        ys = []
        for x in xs:
            ys.append( self.objective( x ) )
        return np.atleast_2d( ys )

    def objective( self, x ):

        req = GetCritiqueRequest()
        req.input = x

        proxy = rospy.ServiceProxy( self.critique_topic, GetCritique )
        try:
            res = proxy.call( req )
        except rospy.ServiceException:
            rospy.logerr( 'Could not evaluate: ' + np.array_str( x ) )
        
        rospy.loginfo( 'Evaluation %d\ninput: %s\noutput: %f\nfeedback: %s', 
                       self.evals,
                       np.array_str( x, max_line_width=sys.maxint ),
                       res.critique,
                       str( res.feedback ) )
        
        self.evals += 1
        self.rounds.append( (x, res.critique, res.feedback ) )
        self.save( 'in_progress' )
        # Critique is reward, so need to negate it
        return -res.critique

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

if __name__ == '__main__':

    rospy.init_node( 'bayesian_optimizer' )

    # See if we're resuming
    if rospy.has_param( '~load_path' ):
        data_path = rospy.get_param( '~load_path' )
        data_log = open( data_path, 'rb' )
        rospy.loginfo( 'Found load data at %s...', data_path )
        bopt = pickle.load( data_log )
    else:
        rospy.loginfo( 'No resume data specified. Starting new optimization...' )
        bopt = BayesianOptimizer()

    # Create interface to optimization problem
    critique_topic = rospy.get_param( '~critic_service' )
    rospy.loginfo( 'Waiting for service %s...', critique_topic )
    rospy.wait_for_service( critique_topic )
    rospy.loginfo( 'Connected to service %s.', critique_topic )

    bopt.initialize()
    bopt.execute()
