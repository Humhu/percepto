"""
Classes that propose arms to test.
"""

import abc, random
import numpy as np
from itertools import izip

class ArmProposal(object):
    """
    Base class for all arm proposal classes.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def propose_arms( self, num_arms ):
        """
        Propose a set of arms to try.

        Returns:
        --------
        arms: iterable of hashable
            A collection of arms.
        num_arms: integer
            The number of arms to propose
        """
        return

    def __add__(self, b):
        return JointArmProposal( self, b )

    def __radd__(self, b):
        return JointArmProposal( b, self )

class JointArmProposal(ArmProposal):
    """
    Combines proposals from two proposal classes.
    """

    def __init__( self, first, second, num_arms ):
        self.first = first
        self.second = second
        self.num_arms = num_arms

    def propose_arms( self, num_arms=None ):
        if num_arms is None:
            num_arms = self.num_arms
        first_arms = self.first.propose_arms( num_arms )
        second_arms = self.second.propose_arms( num_arms )
        return [ tuple(a) + tuple(b) for (a,b) 
                 in izip(first_arms, second_arms) ]

class DiscreteArmProposal(ArmProposal):
    """
    Proposes arms from a discrete set of arms.

    Parameters:
    -----------
    init_arms: iterable of hashable (default [])
        Initial arms to add to the set.
    """

    def __init__( self, init_arms=[] ):
        self.arms = []
        for arm in init_arms:
            self.add_arm( arm )

    def add_arm( self, arm ):
        self.arms.append( arm )

    def propose_arms( self, num_arms=None ):
        if num_arms is None:
            return self.arms
        return random.sample( population=self.arms, k=num_arms )

    def get_arm_ind( self, arm ):
        """
        Returns the arm index, if it is in the set.
        """
        return self.arms.index( arm )

class UniformArmProposal(ArmProposal):
    """
    Proposes arms from a uniform continuous distribution.

    Parameters:
    -----------
    bounds: list of 2D or list
        List of (lower,upper) bounds for each dimension.
    """

    def __init__( self, bounds, num_arms=10 ):
        self.bounds = np.atleast_2d(np.asarray(bounds))
        if self.bounds.shape[1] != 2:
            raise ValueError('Bounds must be list of (lower,upper) bounds.')
        for bound in self.bounds:
            if bound[0] > bound[1]:
                raise ValueError('Bound %s does not follow (lower,upper) convention'
                                 % str(bound))
        self.num_arms = num_arms

    def propose_arms( self, num_arms=None ):
        if num_arms is None:
            num_arms = self.num_arms
        return [ [ random.uniform(l,u) for (l,u) in self.bounds ] 
                 for i in range(num_arms) ]

