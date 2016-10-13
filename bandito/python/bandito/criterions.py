"""
Various arm selection and arm increasing criteria functions.

Contents:
---------
ucb_v_criterion
ucb_air_criterion
siri_criterion
"""

import collections, math
import numpy as np

def ucb_v_criterion( history, exp_factor, reward_scale=1.0, c=1.0 ):
    """Computes the Upper Confidence Bound Variance (UCB-V) criterion from a 
    history of rewards.

    This algorithm estimates an upper bound on the arm mean from the empirical mean 
    and variance. An exploration factor must be specified to combine these two quantities
    into a single criterion.
    
    For more details, refer to Audibert et al. 2007, "Tuning bandit algorithms
    in stochastic environments"

    Parameters:
    -----------
    history: iterable
        Iterable of previous rewards. Must not be empty.
    exp_factor: numeric
        Non-negative exploration factor.
    reward_scale: numeric (default 1.0)
        The size of the reward domain, ie. rewards on [0, b] have size b
    exp_: numeric default (1.0)
        Positive tuning constant that adjusts 

    Returns:
    --------
    criterion: numeric
        The selection criterion. Larger criterion correspond to more desirable pulls.
    """
    if len(history) == 0:
        raise ValueError( 'History cannot be empty.' )
    if exp_factor < 0.0:
        raise ValueError( 'Exploration factor must be non-negative.' )
    if reward_scale <= 0.0:
        raise ValueError( 'Reward scale must be positive.' )
    if c <= 0.0:
        raise ValueError( 'Tuning factor c must be positive.' )

    emp_mean = np.mean( history )
    emp_var = np.var( history )
    s = len( history )
    return emp_mean + math.sqrt( 2 * emp_var * exp_factor / s ) + \
           c * 3.0 * reward_scale * exp_factor / s

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
