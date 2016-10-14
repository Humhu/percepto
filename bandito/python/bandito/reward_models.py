"""
Models of arm or arm/context rewards.
"""

import abc
import numpy as np
from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from gp_extras.kernels import HeteroscedasticKernel

class RewardModel(object):
    """
    Base class for all reward models.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def num_arms( self ):
        """
        The number of arms this model covers.
        """
        return 'Num arms'

    @abc.abstractmethod
    def report_sample( self, arm, reward ):
        """
        Reports the results of sampling an arm to update the mean and
        variance estimate.

        Parameters:
        -----------
        arm: hashable
            The arm value to lookup.
        reward: numeric
            The sampled arm reward.
        """
        return

    @abc.abstractmethod
    def query( self, arm ):
        """
        Return the mean and variance of the specified arm.

        Uses previous data reported with report_sample. By convention, 
        arms with no reported data will return the default mean and
        infinite variance.

        Parameters:
        -----------
        arm: hashable
            The arm value to lookup.

        Returns:
        --------
        mean: numeric
            The estimated arm reward mean.
        variance: numeric
            The estimated arm reward variance.
        """
        return

class EmpiricalRewardModel(RewardModel):
    """
    Models arm rewards with empirical quanities.

    Implemented as a dictionary that maps from arm hashables
    to histories of rewards. Accordingly, this model is best used
    with discrete arm inputs.

    Parameters:
    -----------
    default_mean: numeric (default 0)
        The mean estimate to return for arms with no data.
    default_var: numeric (default inf)
        The variance estimate to return for arms with no data.
    """

    def __init__( self, default_mean=0, default_var=float('inf') ):
        self.arms = {}
        self.default_mean = default_mean
        self.default_var = default_var

    @property
    def num_arms( self ):
        return len( self.arms )

    def report_sample( self, arm, reward ):
        try:
            self.arms[arm].append( reward )
        except KeyError:
            self.arms[arm] = deque()
            self.arms[arm].append( reward )

    def query( self, arm ):
        try:
            emp_mean = np.mean( self.arms[arm] )
            emp_var = np.var( self.arms[arm] )
            return (emp_mean, emp_var)
        except KeyError:
            return (self.default_mean, self.default_var)

class GaussianProcessRewardModel(RewardModel):
    """
    Models arm rewards with a Gaussian process regressor.

    Implemented with a modified version of scikit-learn's Gaussian Process 
    Regressor class. This class uses a zero mean prior and does not yet 
    support setting the prior.

    The GP is updated online as samples are added. As such, hyperparameters 
    for the GP are fit in batch after a threshold number of samples are collected.
    The hyperparameters are then refined afterwards as more samples are added
    until the number of samples passes an upper threshold, after which the
    hyperparameters are no longer updated. This helps avoid highly expensive
    refinement which has computational complexity of O(N^3) in number of samples.

    Parameters:
    -----------
    kernel: sklearn kernel
        A kernel object for the regressor
    hyperparam_min_samples: integer (default 100)
        The number of samples after which initial batch hyperparameter 
        fitting is performed.
    hyperparam_batch_retries: integer (default 20)
        The number of random restarts for the initial hyperparameter fit.
    hyperparam_refine_ll_delta: numeric (default 1.0)
        The hyperparameters are refined after the average GP marginal 
        log-likelihood decreases by this much since the last refinement. 
    hyperparam_max_samples: integer (default 1000)
        The number of samples after which hyperparameters are no longer
        refined.

    Other Keyword Parameters:
    -------------------
    Refer to sklearn.gaussian_process.GaussianProcessRegressor's __init__
    """

    def __init__( self, kernel, 
                  hyperparam_min_samples=100, 
                  hyperparam_batch_retries=20,
                  hyperparam_refine_ll_delta=1.0,
                  hyperparam_max_samples=1000,
                  **kwargs ):
        self.gp = GaussianProcessRegressor( kernel=kernel, **kwargs )
        self.hp_min_samples = hyperparam_min_samples
        self.hp_batch_retries = hyperparam_batch_retries
        self.hp_refine_ll_delta = hyperparam_refine_ll_delta
        self.hp_max_samples = hyperparam_max_samples
        self.hp_init = False
        self.last_ll = None

    # TODO Is this even a useful property to have?
    @property
    def num_arms( self ):
        return float('nan')

    def report_sample( self, arm, reward ):
        self.gp.add_data( arm, reward, incremental=True )
        
        num_samples = len( self.gp.training_y )
        if not self.hp_init and num_samples > self.hp_min_samples:
            self.gp.batch_update( num_restarts=self.hp_batch_retries )
            self.hp_init = True
            self.last_ll = self.gp.log_marginal_likelihood(self.gp.theta)

        if self.hp_init:
            current_ll = self.gp.log_marginal_likelihood(self.gp.theta)
            if current_ll < self.last_ll - self.hp_refine_ll_delta:
                self.gp.batch_update( num_restarts=1 )

    def query( self, arm ):
        # Required by the GP
        arm = np.atleast_2d( arm )
        if arm.shape[0] > 1 or len(arm.shape) > 2:
            raise ValueError( 'Input should be 1D' )
        pred_mean, pred_std = self.gp.predict( arm, return_std=True )
        pred_mean = pred_mean[0][0]
        pred_std = pred_std[0]
        return (pred_mean, pred_std*pred_std)

