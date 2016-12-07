# TODO Online mean/variance estimator for long-term operation
# TODO Context interface

import collections, math
import numpy as np
from bandito.arm_proposals import ArmProposal
from bandito.reward_models import RewardModel
from bandito.arm_selectors import ArmSelector

class BanditInterface:
    """
    Interface to a generic bandit algorithm.

    A solver object that uses an ask/tell interface to request arm pulls and
    receive pull results.

    Parameters:
    -----------
    arm_proposal: ArmProposal derived
        An object that proposes arms to pull at each round.
    reward_model: RewardModel derived
        An object that estimates arm reward means and variances.
    arm_selector: ArmSelector derived
        An object that selects an arm to pull at each round.

    Usage:
    ------
    Ask/tell interface usage example:

    bandit = bandit_interface( selection_func=sfunc, num_arms=10 )
    for i in range( num_rounds ):
        to_pull = bandit.ask()
        reward = pull_arm( to_pull )
        bandit.tell( to_pull, reward )
    """

    def __init__( self, arm_proposal, reward_model, arm_selector ):

        if not isinstance( arm_proposal, ArmProposal ):
            raise TypeError( 'arm_proposal must implement ArmProposal.' )
        self.arm_proposal = arm_proposal

        if not isinstance( reward_model, RewardModel ):
            raise TypeError( 'reward_model must implement RewardModel.' )
        self.reward_model = reward_model

        if not isinstance( arm_selector, ArmSelector ):
            raise TypeError( 'arm_selector must implement ArmSelector.' )
        self.arm_selector = arm_selector

    def ask( self, **kwargs ):
        """
        Request the next arm index to pull.

        Returns:
        --------
        arm: hashable
            An arm from arm_proposal to pull.
        """
        arms = self.arm_proposal.propose_arms()
        return self.arm_selector.select_arm( arms, **kwargs )

    def tell( self, arm, reward ):
        """
        Report reward received for an arm pull.

        Parameters:
        -----------
        arm: hashable
            The arm pulled.
        reward: numeric
            The reward received from pulling the arm.
        """

        self.reward_model.report_sample( arm, reward )
