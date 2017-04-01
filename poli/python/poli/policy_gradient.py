"""This module contains implementations of algorithms for computing likelihood ratio
policy gradients.
"""
import numpy as np
import poli.sampling as isamp
import scipy.linalg as spl
import scipy.signal as sps
import poli.importance_sampling as isamp
from itertools import izip

def estimate_fisher_matrix(gradients, curr_probs=0, past_probs=0,
                           inv_fisher_off=1E-9, diag=False):
    """Returns the Cholesky decomposition of the estimated inverse Fisher information matrix.

    Either computes outer products from gradients, or uses pre-computed outer products.

    Parameters
    ----------
    gradients         : iterable of numpy 1D or 2D array
        Gradients of log-prob for each trajectory, or gradient outer products
    curr_probs        : float or iterable of float (default 0)
        Current log-prob for each trajectory
    past_probs        : float or iterable of float (default 0)
        Generating log-prob for each trajectory
    inv_fisher_off    : float (default 1E-9)
        Offset to add to diagonal of inverse Fisher prior to decomposing
    diag              : bool (default False)
        Whether to use only the diagonal elements of the inverse fisher

    Returns
    -------
    fisher_chol       : Cholesky decomposition tuple
        Tuple to be used in spl.cho_solve
    """
    if len(gradients[0].shape) == 1:
        gradients = [np.outer(g, g) for g in gradients]

    inv_fisher = isamp.importance_sample(gradients,
                                         p_old=past_probs,
                                         p_new=curr_probs,
                                         log_prob=True,
                                         normalize=True)
    inv_fisher += inv_fisher_off * np.identity(inv_fisher.shape[0])
    if diag:
        inv_fisher = np.diag(np.diag(inv_fisher))
        
    fisher_chol = spl.cho_factor(inv_fisher)

    return fisher_chol

def reinforce_preprocess(rewards, gradients, curr_probs, prev_probs):
    """Processes trajectories according to the REINFORCE algorithm:

    Trajectory rewards are summed over the trajectory
    """
    traj_rewards = np.array([np.sum(r) for r in rewards])
    traj_grads =  np.array([np.sum(g, axis=0) for g in gradients])
    traj_cprobs = [np.sum(c) for c in curr_probs]
    traj_pprobs = [np.sum(p) for p in prev_probs]
    return traj_rewards, traj_grads, traj_cprobs, traj_pprobs

def reinforce_baseline(rewards, gradients, curr_probs, past_probs, fisher_chol,
                       est_reward=False, est_grad=False, grad_ops=None):
    """Computes the constant REINFORCE baseline for each trajectory.

    Parameters
    ----------
    rewards     : numpy 1D array
        The trajectory rewards
    gradients   : numpy 2D array
        Log-liklihood gradients for each quantity
    curr_probs  : iterable of float
        Current log-likelihood for each quantity
    past_probs  : iterable of float
        Generating log-likelihood for each quantity
    fisher_chol : Cholesky decomposition
        Cholesky decomposition of current Fisher information matrix
    est_reward  : boolean (default False)
        Whether to output a reward baseline
    est_grad    : boolean (default False)
        Whether to output a gradient baseline
    grad_ops    : numpy 3D array (default None)
        Gradient outer products. If not given, computed from gradients

    Returns
    -------
    baselines   : iterable
        Baselines to subtract from each quantity before importance sampling
    """
    rew_baselines = None
    grad_baselines = None

    if est_reward:
        rew_baseline_ests = (rewards * gradients.T).T
        rew_baseline_acc = isamp.importance_sample(rew_baseline_ests,
                                                   p_old=past_probs,
                                                   p_new=curr_probs,
                                                   log_prob=True,
                                                   normalize=True)
        rew_baseline = spl.cho_solve(fisher_chol, rew_baseline_acc)
        rew_baselines = np.dot(gradients, rew_baseline)
    if est_grad:
        if grad_ops is None:
            grad_ops = np.asarray([np.outer(g, g) for g in gradients])
        grad_baseline_ests = (rewards * grad_ops.T).T
        grad_baseline_acc = isamp.importance_sample(grad_baseline_ests,
                                                    p_old=past_probs,
                                                    p_new=curr_probs,
                                                    log_prob=True,
                                                    normalize=True)
        grad_baseline = spl.cho_solve(fisher_chol, grad_baseline_acc)
        grad_baselines = np.dot(gradients, grad_baseline)

    return rew_baselines, grad_baselines


def importance_policy_gradient(rewards, gradients, curr_probs, past_probs,
                               est_reward=True, est_grad=True, use_natural_grad=True,
                               inv_fisher_off=1E-9, inv_fisher_diag=False):
    """Compute a bandit policy expected rewards and gradient using importance
    sampling.

    Follows the description in Tang and Abbeel's "On a Connection between
    Importance Sampling and the Likelihood Ratio Policy Gradient."

    Parameters
    ----------
    rewards           : iterable of N floats
        The rewards received
    gradients         : iterable of N numpy 1D-arrays
        The policy gradients corresponding to each acquired reward
    curr_probs        : iterable of N floats
        The probabilities (or log-probabilities) of each action corresponding
        to the rewards for current parameters
    past_probs        : iterable of N floats
        The probabilities (or log-probabilities) of each action corresponding
        to the rewards when they were executed
    log_prob          : boolean
        Whether probabilities are log-probabilities or not
    est_reward        : boolean (default True)
        Whether to estimate the expected reward or not
    est_grad          : boolean (default True)
        Whether to estimate the gradient of the expected reward or not
    use_natural_grad  : boolean (default True)
        Whether to estimate the natural gradient
    inv_fisher_off : double (default 1E-9)
        Value to add to diagonal of inverse Fisher matrix for conditioning
    inv_fisher_diag   : boolean (default False)
        Whether to use only the diagonal of the inverse Fisher matrix

    Returns
    -------
    rew_val           : float if est_return is True, else None
        The estimated expected reward for this bandit
    grad_val          : numpy 1D-array if est_grad is True, else None
        The estimated policy gradient for this bandit
    """
    if not est_reward and not est_grad:
        return None, None

    curr_probs = np.asarray(curr_probs)
    past_probs = np.asarray(past_probs)
    rewards = np.asarray(rewards)
    gradients = np.asarray(gradients)

    # Estimate the policy inverse Fisher information matrix
    # This will be used for estimating the baseline and also in computing
    # the natural policy gradient
    grad_ops = np.array([np.outer(g, g) for g in gradients])
    fisher_cho = estimate_fisher_matrix(gradients=grad_ops,
                                        curr_probs=curr_probs,
                                        past_probs=past_probs,
                                        inv_fisher_off=inv_fisher_off,
                                        diag=inv_fisher_diag)

    rew_baselines, grad_baselines = reinforce_baseline(rewards=rewards,
                                                       gradients=gradients,
                                                       curr_probs=curr_probs,
                                                       past_probs=past_probs,
                                                       fisher_chol=fisher_cho,
                                                       est_reward=est_reward,
                                                       est_grad=est_grad,
                                                       grad_ops=grad_ops)

    j_grads = (rewards * gradients.T).T

    # Estimate the expected reward
    rew_val = None
    if est_reward:
        rew_ests = rewards - rew_baselines
        rew_val = isamp.importance_sample(rew_ests,
                                          p_old=past_probs,
                                          p_new=curr_probs,
                                          log_prob=True,
                                          normalize=True)

    # Estimate the policy gradient
    grad_val = None
    if est_grad:
        grad_ests = j_grads - grad_baselines
        grad_val = isamp.importance_sample(grad_ests,
                                           p_old=past_probs,
                                           p_new=curr_probs,
                                           log_prob=True,
                                           normalize=True)
        if use_natural_grad:
            grad_val = spl.cho_solve(fisher_cho, grad_val)

    import pdb
    pdb.set_trace()
    return rew_val, grad_val
