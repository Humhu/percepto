# TODO Online mean/variance estimator for long-term operation

import collections, math
import numpy as np


class BanditInterface:
    """Interface to a generic bandit algorithm.

    A solver object that uses an ask/tell interface to request arm pulls and
    receive pull results. The behavior can be changed by specifying different
    arm selection criterion functions.

    Definitions:
    ------------
    criterion_func: 
        Parameters:
        -----------
            history: iterable 
                Iterable of previous rewards for this arm
            exp_factor: numeric
                Non-negative exploration factor.
        Returns:
        --------
            criterion: numeric
                How 'good' a choice this arm is to pull

    Parameters:
    -----------
    selection_func: criterion_func
        The arm pulling criterion to use.
    num_arms: numerical, optional (default: 0)
        The initial number of arms.
    init_rewards: iterable, or iterable of iterables, optional (default: [])
        Previously received rewards for the arms.

    Usage:
    ------
    Ask/tell interface usage example:

    ucb = Bandit_UCB_V( num_arms )
    for i in range( num_rounds ):
        to_pull = ucb.ask()
        reward = pull_arm( to_pull ) # pull_arm is implemented by user
        ucb.tell( to_pull, reward )

    Arms can be added on construction or sequentially.
    

    For more details on the algorithm itself, see Audibert et al. 2007, 
    "Tuning bandit algorithms in stochastic environments"
    """

    def __init__( self, selection_func, num_arms=0, init_rewards=[] ):
        
        if len(init_rewards) > 0 and len(init_rewards) != num_arms:
            raise ValueError( 'Number of arms must equal initial rewards' )

        if len(init_rewards) == 0:
            self.histories = [[[] for i in range(num_arms)]]
        else:
            if isinstance( init_rewards[0], collections.Iterable ):
                self.histories = init_rewards
            else:
                self.histories = [ [r] for r in init_rewards ]

        self.selection_func = selection_func

    def ask( self, eps ):
        """Request the next arm index to pull.

        Must specify the exploration parameter eps.
        """

        criteria = [ self.selection_func( hist, eps ) for hist in self.histories ]
        comp = lambda x : x[1]
        largest = max( enumerate(criteria), key=comp )
        return largest[0]

    def tell( self, arm, reward ):
        """Report reward received for an arm pull."""
        
        if arm > len(self.histories):
            raise ValueError( 'Arm index %d larger than number of arms %d' %
                              (arm, len(self.histories) ) )

        self.histories[arm].append( reward )
