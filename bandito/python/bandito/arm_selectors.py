"""
Various arm selection and arm increasing criteria functions.

Contents:
---------
ucb_v_criterion
ucb_air_criterion
siri_criterion
"""

import collections, math, abc, cma
import numpy as np
from bandito.reward_models import *
from bandito.arm_proposals import *

class ArmSelector:
    """
    Base class for all arm selection classes.

    Arm selectors accept a list of 
    """

    __metaclass__ = abc.ABCMeta

    def __init__( self ):
        return

    @abc.abstractmethod
    def select_arm( self, arms ):
        """
        Select an arm from a set of arms.

        Parameters:
        -----------
        arms: iterable of hashable
            The arms to select from.

        Returns:
        --------
        best: hashable
            The best arm from input arms.
        """
        return None

class RandomSelector(ArmSelector):
    """
    Uniformly randomly selects an arm.
    """

    def __init__( self ):
        return

    def select_arm( self, arms ):
        return random.choice( arms )

class CMAOptimizerSelector(ArmSelector):
    """
    Selects arms by optimizing an acquisition function.
    """

    def __init__( self, reward_model, dim, mode, **kwargs ):
        if mode != 'min' and mode != 'max':
            raise ValueError( 'mode must be min or max!' )
        self.mode = mode
        self.reward_model = reward_model
        self.dim = dim
        self.init_guess = np.zeros( dim )
        self.init_cov = 0.3 * np.identity( dim )
        self.args = kwargs

    def select_arm( self, arms, beta ):
        self.beta = beta
        es = cma.CMAEvolutionStrategy( self.init_guess, 1, self.args )
        es.optimize( self.criteria )
        return es.result()[0]

    def criteria( self, x ):
        u,v = self.reward_model.query( x )
        # crit is a lower bound estimate of the reward
        crit = u - self.beta * math.sqrt(v)
        if self.mode == 'min':
            return crit
        # Since cma is a minimizer, we have to negate to maximize
        else: # self.mode == 'max'
            return -crit

class UCBVSelector(ArmSelector):
    """
    Computes the Upper Confidence Bound Variance (UCB-V) criterion from a 
    history of rewards.

    This algorithm estimates an upper bound on the arm mean from the empirical mean 
    and variance. An exploration factor must be specified to combine these two quantities
    into a single criterion.
    
    For more details, refer to Audibert et al. 2007, "Tuning bandit algorithms
    in stochastic environments"

    Parameters:
    -----------
    reward_histories: EmpiricalRewardModel
        Reward model to use when selecting arms.
    exp_func: function
        Function that takes no arguments and returns a non-negative exploration factor
        when called.
    reward_scale: numeric (default 1.0)
        The size of the reward domain, ie. rewards on [0, b] have size b
    c: numeric default (1.0)
        Positive tuning constant that adjusts exploitation/exploration tradeoff. Smaller
        values of c result in more exploitation.
    """

    def __init__( self, reward_histories, exp_func, reward_scale=1.0, c=1.0 ):
        if not isinstance( reward_histories, EmpiricalRewardModel ):
            raise TypeError('reward_histories must be an EmpiricalRewardModel')
        self.histories = reward_histories

        self.exp_func = exp_func
        
        if reward_scale <= 0.0:
            raise ValueError( 'Reward scale must be positive.' )
        self.reward_scale = reward_scale
        
        if c <= 0.0:
            raise ValueError( 'Tuning factor c must be positive.' )
        self.c = c

    def select_arm( self, arms ):
        exp_factor = self.exp_func()
        if exp_factor < 0.0:
            raise ValueError( 'Exploration factor must be non-negative.' )
        
        print 'Exp factor: %f' % exp_factor
        best_crit = -float('inf')
        best_arm = None
        for arm in arms:
            (u,v) = self.histories.query( arm )
            s = self.histories.num_samples( arm )
            print 'Arm: %s Mean: %f Std: %f Pulls: %s' % (str(arm), u, math.sqrt(v), s)
            crit = self.criterion( exp_factor, u, v, s )
            if crit > best_crit:
                best_arm = arm
                best_crit = crit
        return best_arm

    def criterion( self, exp_factor, u, v, s):
        if s == 0:
            return float('inf')
        return u + math.sqrt( 2 * v * exp_factor / s ) + \
               self.c * 3.0 * self.reward_scale * exp_factor / s

    @staticmethod
    def default_exp_func( round_num ):
        """
        The exploration function recommended for Many-Armed Bandits with UCBV.
        """
        return math.log( math.log( round_num + 3 ) )

class UCBSelector(ArmSelector):
    """
    Computes the Upper Confidence Bound (UCB) criteria to select arms.

    Can operate on any reward model. When applied to the GP model, this is effectively
    the GP-UCB rule.

    Parameters:
    -----------
    beta_func: callable with no arguments
        Returns a positive numeric value that weights the exploitation/exploration tradeoff.
        Larger beta values increase exploration.
    reward_model: RewardModel derived
        The reward model to use.
    """

    def __init__( self, beta_func, reward_model ):
        if not callable( beta_func ):
            raise TypeError( 'beta_func must be callable.' )
        self.beta_func = beta_func

        if not isinstance( reward_model, RewardModel ):
            raise TypeError( 'reward_model must implement RewardModel.' )
        self.reward_model = reward_model

    def select_arm( self, arms ):
        beta = self.beta_func()
        best_crit = -float('inf')
        best_arm = None
        for arm in arms:
            (u,v) = self.reward_model.query( arm )
            crit = self.criterion( beta, u, v )
            print 'Arm: %s Mean: %f Std: %f Crit: %f' % (str(arm), u, math.sqrt(v), crit)
            if crit > best_crit:
                best_arm = arm
                best_crit = crit
        return best_arm

    def criterion( self, beta, u, v ):
        return u + math.sqrt(beta) * v

    @staticmethod
    def finite_beta_func( round_num, num_arms, uncertainty ):
        """
        Beta function for case of finite number of arms. Follows
        description in Srinivas et al., "Gaussian Process Optimization in the
        Bandit Setting: No Regret and Experiment Design," 2015, Theorem 1.

        Parameters:
        -----------
        round_num: integer
            The current round number, starting from 1
        num_arms: integer
            The number of arms for the GP, or analagously, the 'size' of the
            input space.
        uncertainty: numeric
            Float value between 0 and 1. The regret bound will be achieved with
            probability 1 - uncertainty

        Returns:
        --------
        criteria: numeric
            The selection criteria.
        """
        return 2 * math.log( num_arms * round_num * round_num * math.pi * math.pi \
                             / ( 6 * uncertainty ) )

    @staticmethod
    def bounded_beta_func( round_num, arm_bound, num_dims, uncertainty, a=1.0, b=1.0 ):
        """
        Beta function for case of multivariate continuous arms on a compact, bounded
        space. Follows description in Srinivas et al., "Gaussian Process Optimization 
        in the Bandit Setting: No Regret and Experiment Design," 2015, Theorem 2.

        Parameters:
        -----------
        round_num: integer
            The current round number, starting from 1
        arm_bound: numeric
            Bound on arms assuming each arm dimension is in range [0, arm_bound]
        num_dims: integer
            Number of elements in each arm
        uncertainty: numeric
            Float value between 0 and 1. The regret bound will be achieved with
            probability 1 - uncertainty
        a: numeric (default 1.0)
            Positive constant for tuning GP smoothness constraint
        b: numeric (default 1.0)
            Positive constant for tuning GP smoothness constraint

        Returns:
        --------
        criteria: numeric
            The selection criteria
        """
        round_sq = round_num * round_num
        return 2 * math.log( num_arms * round_sq * math.pi * math.pi \
                             / ( 3 * uncertainty ) ) \
             + 2 * num_dims * math.log( round_sq * num_dims * b * arm_bound \
                                        * math.sqrt( math.log( 4 * num_dims * a / uncertainty ) ) )

def ucb_air_criterion( num_arms, round_num, beta, max_attainable=False ):
    """
    Computes whether to add a new arm from the current number of arms,
    the round number, and problem properties using the Arm Introducing Rule (AIR).

    For more details, refer to Wang et al. 2008, "Algorithms for infinitely many
    armed bandits"

    Parameters:
    -----------
    num_arms: integer
        The current number of arms
    round_num: integer
        The current round number
    beta: numeric
        The problem hardness factor. Must be positive. Smaller corresponds to easier problems
        where it is more likely to sample a nearly optimal arm.
    max_attainable: boolean (default False)
        Whether the maximum reward is attainable, and correspondingly, whether arms with
        higher means will tend to have less variance.

    Returns:
    --------
    new_arm: boolean
        Whether or not a new arm should be added
    """
    
    # At minimum we should have 2 arms
    if num_arms < 2:
        return True

    if max_attainable or beta >= 1.0:
        return num_arms < round_num ** ( beta / (beta + 1) )
    else:
        return num_arms < round_num ** ( 0.5 * beta )

def siri_criterion( history, num_arms, beta, reward_scale, d ):
    """
    Computes the Simple Regret for Infinitely-many arms (SiRI) criterion from a 
    history of rewards.

    For more details, refer to Carpentier et al. 2015, "Simple regret for infinitely
    many armed bandits"

    Parameters:
    -----------
    history: iterable
        Iterable of previous rewards. Must not be empty.
    num_arms: numeric
        Number of arms
    beta: numeric
        The problem hardness constant
    reward_scale: numeric
        Bound on rewards, ie. rewards are in range [0, reward_scale]
    d: numeric
        Arm certainty. Final arm regret will be optimal with probability > (1 - d)

    Returns:
    --------
    """
    Kbeta = num_arms ** ( 2.0 / beta )
    C = reward_scale * 0.5
    emp_mean = np.mean( history )
    emp_var = np.var( history )
    log_term = 4 * C * math.log( Kbeta / d )

    return emp_mean + emp_var * math.sqrt( log_term ) + log_term
