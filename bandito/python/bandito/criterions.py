"""
Various arm selection criteria calculation functions.

Contents:
---------
ucb_v_criterion
"""

import collections, math
import numpy as np

def ucbv_criterion( history, exp_factor ):
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
    """
    if len(history) == 0:
        raise ValueError( 'History cannot be empty.' )
    if exp_factor < 0.0:
        raise ValueError( 'Exploration factor must be non-negative.' )

    emp_mean = np.mean( history )
    emp_var = np.var( history )
    s = len( history )
    return emp_mean + math.sqrt( 2 * emp_var * exp_factor / s ) + 3.0 * exp_factor / s