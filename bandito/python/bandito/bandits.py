# TODO Online mean/variance estimator for long-term operation
# TODO Context interface

import collections, math
import numpy as np

class BanditInterface:
    """
    Interface to a generic bandit algorithm.

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
    
    arm_func: 
    
        Parameters:
        -----------
        num_arms: integer
            Current number of arms
        round_num: integer
            Current round number

        Returns:
        --------
        new_arm: boolean
            Whether or not a new arm is to be added

    Parameters:
    -----------
    selection_func: criterion_func
        The arm pulling criterion to use.
    arm_adding_func: arm_func
        Function that tells whether or not to add an arm
    num_arms: numerical, optional (default: 0)
        The initial number of arms.
    init_rewards: iterable, or iterable of iterables, optional (default: [])
        Previously received rewards for the arms.

    Usage:
    ------
    Ask/tell interface usage example:

    bandit = bandit_interface( selection_func=sfunc, num_arms=10 )
    for i in range( num_rounds ):
        to_pull = bandit.ask()
        reward = pull_arm( to_pull )
        bandit.tell( to_pull, reward )
    """

    def __init__( self, selection_func, arm_func=None, num_arms=0, init_rewards=[] ):
        
        if len(init_rewards) > 0 and len(init_rewards) != num_arms:
            raise ValueError( 'Number of arms must equal initial rewards' )

        if len(init_rewards) == 0:
            self.histories = [[] for i in range(num_arms)]
        else:
            if isinstance( init_rewards[0], collections.Iterable ):
                self.histories = init_rewards
            else:
                self.histories = [ [r] for r in init_rewards ]

        self.arm_func = arm_func
        self.selection_func = selection_func
        self.round_number = 0

        # TODO Compute round number from init_rewards?

    def ask( self, eps=None ):
        """Request the next arm index to pull.

        Must specify the exploration parameter eps.
        """
        # See if we need to add an arm
        if self.arm_func is not None and \
           self.arm_func( len(self.histories), self.round_number ):
            self.histories.append( [] )

        # First we must make sure every arm is pulled at least once
        for arm,hist in enumerate(self.histories):
            if len(hist) == 0:
                print( 'Arm %d does not have any pulls yet.' % arm )
                return arm

        # Compute exploration factor if none is given
        # By default we use a double-log exploration rate
        # TODO Move this outside somehow, since not all criterions use it!
        if eps is None:
            eps = math.log( math.log( self.round_number + 3 ) )

        criteria = [ self.selection_func( hist, eps ) for hist in self.histories ]

        print( 'Round %d exploration factor: %f' % (self.round_number, eps) )
        for arm,crit in enumerate(criteria):
            mean = np.mean( self.histories[arm] )
            std = np.std( self.histories[arm] )
            pulls= len( self.histories[arm] )
            print( 'Arm %d criteria: %f pulls: %d mean: %f std: %f' % (arm, crit, pulls, mean, std) )

        self.round_number += 1
        comp = lambda x : x[1]
        largest = max( enumerate(criteria), key=comp )
        return largest[0]

    def tell( self, arm, reward ):
        """Report reward received for an arm pull."""

        try:
            self.histories[arm].append( reward )
        except IndexError:
            raise IndexError( 'Arm index %d larger than number of arms %d' %
                              (arm, len(self.histories) ) )
