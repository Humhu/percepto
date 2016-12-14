#!/usr/bin/env python

import dill
import pickle
import rospy, sys, math
import numpy as np
from collections import deque
from itertools import izip

from percepto_msgs.srv import GetCritique, GetCritiqueRequest, GetCritiqueResponse

from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
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

        self.opt_mode = rospy.get_param( '~optimization_mode' )
        if self.opt_mode not in ['min', 'max']:
            raise ValueError( '~opt_mode must be min or max!' )

        self.init_mode = rospy.get_param( '~initialization_mode', 'mean' )
        if self.init_mode not in ['median', 'mean']:
            raise ValueError( '~init_mode must be mean or median!' )

        self.negative_rewards = rospy.get_param( '~negative_rewards', False )

        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            np.random.seed( seed )

        self.save_period = rospy.get_param( '~save_period', 1 )

        # Reward model and bandit
        init_samples = rospy.get_param( '~model/initial_samples', 30 )
        self.input_dim = rospy.get_param( '~input_dimension' )
        self.input_lower = rospy.get_param( '~input_lower_bound' )
        self.input_upper = rospy.get_param( '~input_upper_bound' )

        self.output_lower = float( rospy.get_param( '~output_lower_bound', '-inf' ) )
        self.output_upper = float( rospy.get_param( '~output_upper_bound', 'inf' ) )
        self.max_output_retries = rospy.get_param( '~max_output_retries', 10 )

        self.init_tests = np.random.uniform( low=self.input_lower, high=self.input_upper,
                                             size=(init_samples, self.input_dim ) )
        self.init_Y = []

        # Convergence and state
        self.init_beta = rospy.get_param( '~model/init_beta', 1.0 )
        self.beta_scale = rospy.get_param( '~model/beta_scale', 1.0 )
        self.x_tol = float( rospy.get_param( '~convergence/input_tolerance', -float('inf') ) )
        self.max_evals = float( rospy.get_param( '~convergence/max_evaluations', float('inf') ) )
        self.evals = 0
        self.last_inputs = deque()

        self.rounds = []
        self.bests = []
        self.bandit = None
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def is_done( self ):
        if self.evals >= self.max_evals:
            return {'max_evaluations' : self.evals}

        # TODO Something is up with np.linalg.norm...
        if len( self.rounds ) >= 2:
            delta_input = self.rounds[-1][0] - self.rounds[-2][0]
            if np.linalg.norm( delta_input ) < self.x_tol:
                return {'input_tolerance' : self.x_tol}
        return {}

    def compute_beta( self ):
        # TODO Different beta schedules
        return self.init_beta / math.sqrt( self.beta_scale * (self.evals + 1) )

    def evaluate( self, eval_cb, x ):
        for i in range( self.max_output_retries ):
            (reward, feedback) = eval_cb( x )
            if reward >= self.output_lower and reward <= self.output_upper:
                break
        return (reward, feedback)

    def predict_reward( self, x ):
        pred_y, pred_var = self.reward_model.query( x )
        pred_std = math.sqrt( pred_var )
        rospy.loginfo( 'raw mean: %f std: %f', pred_y, pred_std )

        pred_bound = np.array( [ pred_y - pred_std, pred_y + pred_std ] )
        if self.opt_model_logs:
            pred_y = math.exp( pred_y )
            pred_bound = np.exp( pred_bound )
        if self.negative_rewards:
            pred_y = -pred_y
            pred_bound = -pred_bound
        return pred_y, pred_bound

    def model_to_raw( self, y ):
        if self.opt_model_logs:
            y = math.exp( y )
        if self.negative_rewards:
            y = -y
        return y

    def raw_to_model( self, y ):
        if self.negative_rewards:
            y = -y 
        if self.opt_model_logs:
            y = math.log( y )
        return y

    def initialize( self, eval_cb ):
        # Run initial tests 
        while len( self.init_Y ) < len(self.init_tests):
            rospy.loginfo( 'Initial exploration %d/%d', 
                           len( self.init_Y ), 
                           len(self.init_tests) )
            x = self.init_tests[ len( self.init_Y ) ]
            (reward, feedback) = self.evaluate( eval_cb, x )
            self.init_Y.append( [reward] )
            
            self.rounds.append( (x, reward, feedback ) )
            self.save( 'initializing' )

        # Create optimizer
        hyper_ll_delta = rospy.get_param( '~model/hyperparam_refine_ll_delta', 3.0 )
        hyper_refine_retries = rospy.get_param( '~model/hyperparam_refine_retries', 1 )
        init_noise = rospy.get_param( '~model/init_noise', 1.0 )
        noise_bounds = farr( rospy.get_param( '~model/noise_bounds', (1e-3, 1e3) ) )
        init_scale = rospy.get_param( '~model/init_scale', 1.0 )
        scale_bounds = farr( rospy.get_param( '~model/scale_bounds', (1e-3, 1e3) ) )
        init_length = rospy.get_param( '~model/init_kernel_length', 1.0 )
        length_bounds = farr( rospy.get_param( '~model/kernel_length_bounds', (1e-3, 1e3) ) )
        nu = rospy.get_param( '~model/kernel_roughness', 1.5 )
        if nu != 0.5 and nu != 1.5 and nu != 2.5 and nu != float('inf'):
            rospy.logwarn( 'Note: kernel_roughness not set to 0.5, 1.5, 2.5, or inf results ' +\
                           'in high computational cost!' )

        self.white = WhiteKernel( init_noise, noise_bounds )
        self.kernel_base = ConstantKernel( init_scale, scale_bounds ) * \
                           Matern( init_length, length_bounds, nu )
        self.kernel_noisy = self.kernel_base  + self.white
        rospy.loginfo( 'Using kernel: %s', str(self.kernel_noisy) )

        self.opt_model_logs = rospy.get_param( '~model/model_log_reward', False )
        # prior_mean = float( rospy.get_param( '~model/prior_mean', 0 ) )
        # if self.negative_rewards:
        #     prior_mean = -prior_mean
        # if self.opt_model_logs:
        #     prior_mean = math.log( prior_mean )
        # print 'Initializing model with prior mean %f' % prior_mean
        
        normalize_y = rospy.get_param( '~model/normalize_output', False )
        self.reward_model = GPRewardModel( kernel = self.kernel_noisy,
                                           kernel_noiseless = self.kernel_base,
                                           hyperparam_min_samples = len( self.init_tests ),
                                           hyperparam_refine_ll_delta = hyper_ll_delta,
                                           hyperparam_refine_retries = hyper_refine_retries,
                                           normalize_y = normalize_y )

        raw_Y = self.init_Y
        self.init_Y = [ [self.raw_to_model( y[0] )] for y in self.init_Y ]
        if self.init_mode == 'mean':
            raw_mean = np.mean( raw_Y )
            model_mean = np.mean( self.init_Y )
        elif self.init_mode == 'median':
            raw_mean = np.median( raw_Y )
            model_mean = np.median( self.init_Y )
            
        rospy.loginfo( 'Initial reward %s raw: %f model: %f', 
                       self.init_mode, raw_mean, model_mean )
        self.reward_model.batch_initialize( np.atleast_2d( self.init_tests ), 
                                            np.atleast_2d( self.init_Y ) )

        acq_tol = float( rospy.get_param( '~model/acquisition_tolerance' ) )
        if self.opt_mode == 'min' and not self.negative_rewards:
            acq_mode = 'min'
        elif self.opt_mode == 'min' and self.negative_rewards:
            acq_mode = 'max'
        elif self.opt_mode == 'max' and not self.negative_rewards:
            acq_mode = 'max'
        elif self.opt_mode == 'max' and self.negative_rewards:
            acq_mode = 'min'
        else:
            raise RuntimeError( 'Logic error in determining acquisition mode!' )
        self.arm_selector = CMAOptimizerSelector( reward_model = self.reward_model,
                                                  dim = self.input_dim,
                                                  mode = acq_mode,
                                                  bounds = [ self.input_lower, self.input_upper ],
                                                  popsize = 30,
                                                  tolfun = acq_tol,
                                                  tolx = acq_tol,
                                                  verbose= -9 )

        self.arm_proposal = NullArmProposal()
        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

    def execute( self, eval_cb ):
        if len( self.init_Y ) < len( self.init_tests ):
            self.initialize( eval_cb )

        while not rospy.is_shutdown() and not self.is_done():
            beta = self.compute_beta()
            x = self.bandit.ask( beta = beta )

            # Report predictions
            pred_y, pred_bound = self.predict_reward( x )
            rospy.loginfo( 'Evaluation %d with beta %f and predicted value %f in %s', 
                           self.evals,
                           beta,
                           pred_y,
                           np.array_str(pred_bound) )

            # Perform evaluation and give feedback
            (reward, feedback) = self.evaluate( eval_cb, x )
            self.bandit.tell( x, self.raw_to_model( reward ) )

            self.rounds.append( (x, reward, feedback ) )
            self.evals += 1
            self.save( 'in_progress' )

        opt_x = self.bandit.ask( beta = 0 )
        opt_mean, opt_bound = self.predict_reward( opt_x )
        opt = (opt_x, opt_mean, opt_bound)
        crit = self.is_done()
        rospy.loginfo( 'Completed due to %s\noptimal x: %s\n pred y: %f in %s',
                       str(crit),
                       np.array_str(opt_x), 
                       opt_mean,
                       str(opt_bound) )
        self.save( opt )

    def save( self, status ):
        if self.evals % self.save_period != 0:
            return

        if self.prog_path is not None:
            rospy.loginfo( 'Saving progress at %s...', self.prog_path )
            prog = open( self.prog_path, 'wb' )
            pickle.dump( self, prog )
            prog.close()

        if self.bandit is not None:
            rospy.loginfo( 'Finding current best...' )
            opt_x = self.bandit.ask( beta = 0 )
            opt_mean, opt_bound = self.predict_reward( opt_x )
            self.bests.append( (self.evals, opt_x, opt_mean, opt_bound) )

        rospy.loginfo( 'Saving output at %s...', self.out_path )
        out = open( self.out_path, 'wb' )
        pickle.dump( (status, self.rounds, self.bests), out )
        out.close()

def evaluate_input( proxy, inval):

    req = GetCritiqueRequest()
    req.input = inval

    try:
        res = proxy.call( req )
    except rospy.ServiceException:
        rospy.logerr( 'Could not evaluate item: ' + np.array_str( inval ) )
        return None
    
    msg = 'Evaluated input: %s\n' % np.array_str( inval, max_line_width=sys.maxint )
    msg += 'Critique: %f\n' % res.critique
    msg += 'Feedback:\n'
    feedback = {}
    for (name,value) in izip( res.feedback_names, res.feedback_values ):
        msg += '\t%s: %f\n' % ( name, value )
        feedback[name] = value
    rospy.loginfo( msg )

    return (res.critique, feedback)

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
