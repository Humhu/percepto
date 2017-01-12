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
        if self.init_mode not in ['median', 'mean', 'max']:
            raise ValueError( '~init_mode must be mean or median or max!' )

        self.num_final_samples = rospy.get_param( '~final_samples', 10 )
        self.batch_period = rospy.get_param( '~batch_period', 10 )
        self.negative_rewards = rospy.get_param( '~negative_rewards', False )

        self.test_x = rospy.get_param( '~query_x', None )
        if self.test_x is not None:
            self.test_x = np.array( self.test_x )

        # Seed RNG if specified
        seed = rospy.get_param('~random_seed', None)
        if seed is None:
            rospy.loginfo( 'No random seed specified. Using default behavior.' )
        else:
            rospy.loginfo( 'Initializing with random seed: ' + str(seed) )
            np.random.seed( seed )

        self.save_period = rospy.get_param( '~save_period', 1 )

        # Reward model and bandit
        self.input_dim = rospy.get_param( '~input_dimension' )
        self.input_lower = rospy.get_param( '~input_lower_bound' )
        self.input_upper = rospy.get_param( '~input_upper_bound' )

        self.output_lower = float( rospy.get_param( '~output_lower_bound', '-inf' ) )
        self.output_upper = float( rospy.get_param( '~output_upper_bound', 'inf' ) )
        self.max_output_retries = rospy.get_param( '~max_output_retries', 10 )

        init_samples = rospy.get_param( '~model/initial_samples', 30 )
        self.init_x = np.random.uniform( low=self.input_lower, high=self.input_upper,
                                             size=(init_samples, self.input_dim ) )
        self.init_Y = []

        # Convergence and state
        self.init_beta = rospy.get_param( '~model/init_beta', 1.0 )
        self.beta_scale = rospy.get_param( '~model/beta_scale', 1.0 )

        self.x_tol = float( rospy.get_param( '~convergence/input_tolerance', -float('inf') ) )
        self.max_evals = float( rospy.get_param( '~convergence/max_evaluations', float('inf') ) )
        self.evals = 0
        self.last_inputs = deque()

        self.init_rounds = []
        self.test_rounds = []
        self.bests = []
        self.bandit = None
        self.prog_path = rospy.get_param( '~progress_path', None )
        self.out_path = rospy.get_param( '~output_path' )

    def is_done( self ):
        if self.evals >= self.max_evals:
            return {'max_evaluations' : self.evals}

        if len( self.test_rounds ) >= 2:
            delta_input = self.test_rounds[-1][0] - self.test_rounds[-2][0]
            if np.linalg.norm( delta_input ) < self.x_tol:
                return {'input_tolerance' : self.x_tol}
        return {}

    def compute_beta( self ):
        # TODO Different beta schedules
        #return self.init_beta / math.sqrt( self.beta_scale * (self.evals + 1) )
        t = self.evals + 1
        beta = self.init_beta * self.input_dim * math.log( self.beta_scale * t**2  )
        return math.sqrt( beta )

    def evaluate( self, eval_cb, x ):
        (reward, feedback) = eval_cb( x )
        return (reward, feedback)

    def predict_reward( self, x ):
        #if self.test_x is None:
        #    randx = np.random.uniform( low=self.input_lower, high=self.input_upper, size=x.shape )
        #else:
        #    randx = self.test_x
        #rpred_y, rpred_var = self.reward_model.query( randx ) 
        #rospy.loginfo( 'test x: %s\n mean: %f std: %f', np.array_str( randx ), rpred_y, rpred_var )

        raw_y, raw_var = self.reward_model.query( x )
        raw_std = math.sqrt( raw_var )
        pred_y = self.model_to_raw( raw_y )
        extrema = [ self.model_to_raw( raw_y - raw_std ), self.model_to_raw( raw_y + raw_std ) ]
        pred_lower = min( extrema )
        pred_upper = max( extrema )
        rospy.loginfo( 'raw mean: %f std: %f', raw_y, raw_std )
        return pred_y, (pred_lower, pred_upper)

    def model_to_raw( self, y ):
        """Convert a model reward to a raw reward.
        Order is: scale, log, negate
        """
        if self.normalize_scale:
            y = y * self.raw_scale
        if self.opt_model_logs:
            y = math.exp( y )
        if self.negative_rewards:
            y = -y
        return y

    def raw_to_model( self, y ):
        """Convert a raw reward to model reward.
        Order is: negate, log, scale
        """
        if math.isnan( y ):
            y = self.constraint_value
        if y < self.output_lower:
            y = self.output_lower
        if y > self.output_upper:
            y = self.output_upper
        if self.negative_rewards:
            y = -y 
        if self.opt_model_logs:
            y = math.log( y )
        if self.normalize_scale:
            y = y / self.raw_scale
        return y

    def initialize( self, eval_cb ):
        # Run initial tests 
        while len( self.init_Y ) < len(self.init_x):
            rospy.loginfo( 'Initial exploration %d/%d', 
                           len( self.init_Y )+1, 
                           len(self.init_x) )
            x = self.init_x[ len( self.init_Y ) ]
            (reward, feedback) = self.evaluate( eval_cb, x )
            self.init_Y.append( reward )
            self.init_rounds.append( (x, reward, feedback ) )
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
        #self.kernel_base = Matern( init_length, length_bounds, nu )
        self.kernel_noisy = self.kernel_base  + self.white
        rospy.loginfo( 'Using kernel: %s', str(self.kernel_noisy) )

        self.opt_model_logs = rospy.get_param( '~model/model_log_reward', False )
        self.normalize_scale = rospy.get_param( '~model/normalize_raw_scale', False )

        # Determine mean and scale
        raw_Y = [ y for y in self.init_Y if not np.isnan(y) ]
        self.constraint_value = min( raw_Y )
        self.raw_scale = 1
        unscaled_Y = [ self.raw_to_model( y ) for y in raw_Y ]
        self.raw_scale = ( max( unscaled_Y ) - min( unscaled_Y ) ) * 0.5
        rospy.loginfo( 'Constraint violations will be assigned raw value %f', self.constraint_value )
        rospy.loginfo( 'Raw value scale is %f', self.raw_scale )

        valid_Y = [ self.raw_to_model( y ) for y in raw_Y ]
        # NOTE Need init_Y to end up 2D for the GP
        self.init_Y = [ [self.raw_to_model( y )] for y in self.init_Y ]
        if self.init_mode == 'mean':
            raw_mean = np.mean( raw_Y )
            valid_mean = np.mean( valid_Y )
            all_mean = np.mean( self.init_Y )
        elif self.init_mode == 'median':
            raw_mean = np.median( raw_Y )
            valid_mean = np.median( raw_Y )
            all_mean = np.median( self.init_Y )
        elif self.init_mode == 'max':
            raw_mean = np.max( raw_Y )
            if self.negative_rewards:
                valid_mean = np.min( valid_Y )
                all_mean = np.min( self.init_Y )
            else:
                valid_mean = np.max( valid_Y )
                all_mean = np.max( self.init_Y )

            
        rospy.loginfo( 'Initial reward %s valid raw: %f valid model: %f all model: %f', 
                       self.init_mode, raw_mean, valid_mean, all_mean )
        
        self.reward_model = GPRewardModel( kernel = self.kernel_noisy,
                                           kernel_noiseless = self.kernel_base,
                                           hyperparam_min_samples = len( self.init_x ),
                                           hyperparam_refine_ll_delta = hyper_ll_delta,
                                           hyperparam_refine_retries = hyper_refine_retries,
                                           prior_mean = valid_mean )

        self.reward_model.batch_initialize( np.atleast_2d( self.init_x ), 
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
        popsize = rospy.get_param( '~cma_popsize', 100 )
        n_restarts = rospy.get_param( '~cma_restarts', 0 )
        self.arm_selector = CMAOptimizerSelector( reward_model = self.reward_model,
                                                  dim = self.input_dim,
                                                  mode = acq_mode,
                                                  bounds = [ self.input_lower, self.input_upper ],
                                                  popsize = popsize,
                                                  num_restarts = n_restarts,
                                                  tolfun = acq_tol,
                                                  tolx = acq_tol,
                                                  verbose= -9 )

        self.arm_proposal = NullArmProposal()
        self.bandit = BanditInterface( arm_proposal = self.arm_proposal,
                                       reward_model = self.reward_model,
                                       arm_selector = self.arm_selector )

    def execute( self, eval_cb ):
        if len( self.init_Y ) < len( self.init_x ):
            self.initialize( eval_cb )

        while not rospy.is_shutdown() and not self.is_done():
            beta = self.compute_beta()
            x = self.bandit.ask( beta = beta )

            # Report predictions
            pred_y, pred_bound = self.predict_reward( x )
            rospy.loginfo( 'Evaluation %d with beta %f predicted value %f in %s', 
                           self.evals+1, beta, pred_y, str(pred_bound) )

            # Perform evaluation and give feedback
            (reward, feedback) = self.evaluate( eval_cb, x )
            self.bandit.tell( x, self.raw_to_model( reward ) )

            # Report update
            pred_y, pred_bound = self.predict_reward( x )
            rospy.loginfo( 'After update: predicted value %f in %s', 
                           pred_y, str(pred_bound) )

            self.test_rounds.append( (x, reward, feedback ) )
            self.evals += 1
            self.save( 'in_progress' )

            if self.evals % self.batch_period == 0:
                self.init_x = [r[0] for r in self.init_rounds] + [ r[0] for r in self.test_rounds ]
                self.init_Y = [r[1] for r in self.init_rounds] + [ r[1] for r in self.test_rounds ]
                self.initialize( eval_cb )

        self.arm_selector.set_num_restarts( 100 )
        opt_x = self.bandit.ask( beta = 0 )
        opt_mean, opt_bound = self.predict_reward( opt_x )
        opt_samples = []
        for i in range( self.num_final_samples ):
            rospy.loginfo( 'Optima sample %d/%d', i+1, self.num_final_samples )
            opt_samples.append( self.evaluate( eval_cb, opt_x ) )
        opt = (opt_x, opt_mean, opt_bound, opt_samples)
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
        pickle.dump( (status, self.init_rounds, self.test_rounds, self.bests), out )
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
        # HACK to allow continuing
        bopt.max_evals = float( rospy.get_param( '~convergence/max_evaluations', float('inf') ) )
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
