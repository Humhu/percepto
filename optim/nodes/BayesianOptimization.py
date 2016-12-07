#!/usr/bin/env python

import dill
import pickle
import GPy, GPyOpt
import rospy, sys, math
import numpy as np
from collections import deque
from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse

from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, RBF
from bandito.reward_models import GaussianProcessRewardModel as GPRewardModel
from bandito.arm_selectors import CMAOptimizerSelector
from bandito.arm_proposals import NullArmProposal
from bandito.bandits import BanditInterface

def farr( a ):
    return [ float(x) for x in a ]

class BayesianOptimizer:
    """Bayesian optimization (BO) optimizer.

    Uses a fork of the sklearn library: 

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

        # Reward model and bandit
        init_samples = rospy.get_param( '~optimizer/initial_samples', 30 )
        hyper_ll_delta = rospy.get_param( '~optimizer/hyperparam_refine_ll_delta', 3.0 )
        init_noise = rospy.get_param( '~optimizer/init_noise', 1.0 )
        noise_bounds = farr( rospy.get_param( '~optimizer/noise_bounds', (1e-3, 1e3) ) )
        init_scale = rospy.get_param( '~optimizer/init_scale', 1.0 )
        scale_bounds = farr( rospy.get_param( '~optimizer/scale_bounds', (1e-3, 1e3) ) )
        init_length = rospy.get_param( '~optimizer/init_kernel_length', 1.0 )
        length_bounds = farr( rospy.get_param( '~optimizer/kernel_length_bounds', (1e-3, 1e3) ) )
        nu = rospy.get_param( '~optimizer/kernel_roughness', 1.5 )
        if nu != 0.5 and nu != 1.5 and nu != 2.5 and nu != float('inf'):
            rospy.logwarn( 'Note: kernel_roughness not set to 0.5, 1.5, 2.5, or inf results ' +\
                           'in high computational cost!' )

        self.white = WhiteKernel( init_noise, noise_bounds )
        self.kernel_base = ConstantKernel( init_scale, scale_bounds ) * \
                           RBF( 1.0, (1e-3, 1e-1) )
        self.kernel_noisy = self.kernel_base  + self.white
        print self.kernel_noisy

        self.reward_model = GPRewardModel( kernel = self.kernel_noisy,
                                           kernel_noiseless = self.kernel_base,
                                           hyperparam_min_samples = init_samples,
                                           hyperparam_refine_ll_delta = hyper_ll_delta )

        input_dim = rospy.get_param( '~input_dimension' )
        input_lower = rospy.get_param( '~input_lower_bound' )
        input_upper = rospy.get_param( '~input_upper_bound' )

        self.init_beta = rospy.get_param( '~optimizer/init_beta', 1.0 )
        self.beta_scale = rospy.get_param( '~optimizer/beta_scale', 1.0 )
        self.arm_selector = CMAOptimizerSelector( reward_model = self.reward_model,
                                                  dim = input_dim,
                                                  bounds = [ input_lower, input_upper ],
                                                  popsize = 30,
                                                  verbose=-9 )

        self.arm_proposal = NullArmProposal()
        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

        # Convergence and state
        self.x_tol = rospy.get_param( '~convergence/input_tolerance', -float('inf') )
        self.max_evals = rospy.get_param( '~convergence/max_evaluations', float('inf') )
        self.evals = 0
        self.last_inputs = deque()

        self.rounds = []
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def is_done( self ):
        if self.evals >= self.max_evals:
            return {'max_evaluations' : self.evals}

        if len( self.rounds ) >= 2:
            delta_input = self.rounds[-1][0] - self.rounds[-2][0]
            if np.linalg.norm( delta_input ) < self.x_tol:
                return {'input_tolerance' : self.x_tol}
        return {}

    def compute_beta( self ):
        # TODO Different beta schedules
        return self.init_beta / math.sqrt( self.beta_scale * (self.evals + 1) )

    def execute( self, eval_cb ):
        while not rospy.is_shutdown() and not self.is_done():
            
            x = self.bandit.ask( beta = self.compute_beta() )
            rospy.loginfo( 'Evaluation %d', self.evals )
            (reward, feedback) = eval_cb( x )
            self.bandit.tell( x, reward )

            self.rounds.append( (x, reward, feedback ) )
            self.evals += 1
            self.save( 'in_progress' )

        self.save( self.is_done() )

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

def evaluate_input( proxy, inval):

    req = GetCritiqueRequest()
    req.input = inval

    try:
        res = proxy.call( req )
    except rospy.ServiceException:
        rospy.logerr( 'Could not evaluate item: ' + np.array_str( inval ) )
    
    rospy.loginfo( 'Evaluated input: %s\noutput: %f\nfeedback: %s', 
                   np.array_str( inval, max_line_width=sys.maxint ),
                   res.critique,
                   str( res.feedback ) )
    return (res.critique, res.feedback)

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
    critique_proxy = rospy.ServiceProxy( critique_topic, GetCritique, True )
    eval_cb = lambda x : evaluate_input( critique_proxy, x )

    bopt.execute( eval_cb )
