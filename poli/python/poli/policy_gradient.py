"""This module contains implementations of algorithms for computing likelihood ratio
policy gradients.
"""
import numpy as np
import poli.sampling as isamp
import scipy.linalg as spl
import scipy.signal as sps
from itertools import izip


def isample_fisher(gradients, p_tar, p_gen, offset=1E-9,
                   diag=False, **kwargs):
    """Compute the Cholesky decomposition of the Fisher information matrix
    computed from importance sampling gradients.

    Parameters
    ----------
    gradients         : iterable of iterable of float
        list of gradients in each trajectory
    p_tar        : iterable of iterable of float
        list of current log-probs for each action in each trajectory
    p_gen        : iterable of iterable of float
        list of generating log-probs for each action in each trajectory
    offset    : float (default 1E-9)
        Offset to add to diagonal of inverse Fisher prior to decomposing
    diag              : bool (default False)
        Whether to use only the diagonal elements of the inverse fisher
    min_weight       : float (default -inf)
        Minimum log-weight for importance sampling
    """

    if diag:
        grad_ops = [np.diag(g * g) for g in gradients]
    else:
        grad_ops = [np.outer(g, g) for g in gradients]

    # Fisher matrix is expectation over states of expected action log-prob outer prods
    # TODO Technically this importance sampling approach is wrong! Should use
    # mixture probs
    fisher = isamp.importance_sample(grad_ops,
                                     p_tar=p_tar,
                                     p_gen=p_gen,
                                     **kwargs)

    fisher += offset * np.identity(fisher.shape[0])
    if diag:
        fisher = np.diag(np.diag(fisher))
    fisher_chol = spl.cho_factor(fisher)
    return fisher_chol


def constant_isamp_baseline(rewards, gradients, r_grads, p_tar, p_gen,
                            fisher_chol=None, est_reward=False, est_grad=False,
                            fisher_diag=False, **kwargs):
    """Computes the optimal constant importance sampling baseline for each trajectory.

    Parameters
    ----------
    rewards     : numpy 1D array
        Rewards for each trajectory
    gradients   : numpy 2D array
        Log-liklihood gradients for each trajectory under the target distribution
    r_grads      : numpy 2D array
        Reward-gradient estimate for each trajectory
    p_tar       : iterable of float
        Target distribution log-prob for each trajectory
    p_gen  : iterable of float
        Generating log-prob for each trajectory
    fisher_chol : Cholesky decomposition
        Cholesky decomposition of current Fisher information matrix
    est_reward  : boolean (default False)
        Whether to output a reward baseline
    est_grad    : boolean (default False)
        Whether to output a gradient baseline

    Returns
    -------
    baselines   : iterable
        Baselines to subtract from each quantity before importance sampling
    """
    rew_baselines = None
    grad_baselines = None

    if fisher_chol is None:
        fisher_chol = isample_fisher(gradients,
                                     p_tar=p_tar,
                                     p_gen=p_gen,
                                     diag=fisher_diag,
                                     **kwargs)
    if est_reward:
        rew_baseline_ests = [r * g for r, g in izip(rewards, gradients)]
        rew_baseline_acc = isamp.importance_sample(rew_baseline_ests,
                                                   p_tar=p_tar,
                                                   p_gen=p_gen,
                                                   **kwargs)

        rew_baseline = spl.cho_solve(fisher_chol, rew_baseline_acc)
        rew_baselines = np.dot(gradients, rew_baseline)

    if est_grad:
        if fisher_diag:
            grad_base_ests = [np.diag(rp * g)
                              for rp, g in izip(r_grads, gradients)]
        else:
            grad_base_ests = [np.outer(rp, g)
                              for rp, g in izip(r_grads, gradients)]
        baseline_acc = isamp.importance_sample(grad_base_ests,
                                               p_tar=p_tar,
                                               p_gen=p_gen,
                                               **kwargs)
        grad_baseline = spl.cho_solve(fisher_chol, baseline_acc)
        grad_baselines = np.dot(gradients, grad_baseline)

    return rew_baselines, grad_baselines


def _importance_preprocess_gpomdp(rewards, gradients, p_tar, p_gen):
    """Computes various trajectory quantities using the GPOMDP formulation.
    """

    res = {'traj_r': [], 'traj_p_tar': [], 'traj_p_gen': [],
           'r_grads': [], 'state_act_p_tar': [],
           'state_act_p_gen': [], 'act_grads': [],
           'traj_grads': []}

    for rs, gs, ps, qs in izip(rewards, gradients, p_tar, p_gen):

        traj_p = np.sum(ps)
        traj_q = np.sum(qs)
        sum_grads = np.cumsum(gs, axis=0)
        r_grad = np.sum((np.asarray(rs) * sum_grads.T).T, axis=0)

        res['r_grads'].append(r_grad)
        res['traj_p_tar'].append(traj_p)
        res['traj_p_gen'].append(traj_q)
        res['traj_grads'].append(sum_grads[-1])
        res['traj_r'].append(np.sum(rs))

        # Used for estimating fisher
        res['act_grads'].extend(gs)
        res['state_act_p_tar'].extend(np.cumsum(ps))
        res['state_act_p_gen'].extend(np.cumsum(qs))

    return res


def _importance_preprocess_reinforce(rewards, gradients, p_tar, p_gen):
    """Computes various trajectory quantities using the REINFORCE formulation.
    """

    res = {'traj_r': [], 'traj_p_tar': [], 'traj_p_gen': [],
           'r_grads': [], 'state_act_p_tar': [],
           'state_act_p_gen': [], 'act_grads': [],
           'traj_grads': []}

    for rs, gs, ps, qs in izip(rewards, gradients, p_tar, p_gen):

        traj_p = np.sum(ps)
        traj_q = np.sum(qs)
        sum_grads = np.sum(gs, axis=0)
        traj_r = np.sum(rs)
        r_grad = sum_grads * traj_r

        res['r_grads'].append(r_grad)
        res['traj_p_tar'].append(traj_p)
        res['traj_p_gen'].append(traj_q)
        res['traj_grads'].append(sum_grads)
        res['traj_r'].append(np.sum(rs))

        # Used for estimating fisher
        res['act_grads'].extend(gs)
        res['state_act_p_tar'].extend(np.cumsum(ps))
        res['state_act_p_gen'].extend(np.cumsum(qs))

    return res


def importance_reinforce(rewards, gradients, p_tar, p_gen,
                         use_baseline=True, use_natural_grad=True,
                         fisher_diag=False, ret_diagnostics=False, **kwargs):
    res = _importance_preprocess_reinforce(rewards, gradients, p_tar, p_gen)
    return _importance_policy_gradient(res,
                                       use_baseline=use_baseline,
                                       use_natural_grad=use_natural_grad,
                                       fisher_diag=fisher_diag,
                                       ret_diagnostics=ret_diagnostics,
                                       **kwargs)


def importance_gpomdp(rewards, gradients, p_tar, p_gen,
                      use_baseline=True, use_natural_grad=True,
                      fisher_diag=False, ret_diagnostics=False, **kwargs):
    """Compute policy expected rewards and gradient using importance
    sampling.

    Follows the description in Tang and Abbeel's "On a Connection between
    Importance Sampling and the Likelihood Ratio Policy Gradient."

    Parameters
    ----------
    rewards           : iterable of N floats
        The rewards received
    gradients         : iterable of N numpy 1D-arrays
        The policy gradients corresponding to each acquired reward
    p_tar        : iterable of N floats
        The probabilities (or log-probabilities) of each action corresponding
        to the rewards for current parameters
    p_gen        : iterable of N floats
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
    fisher_diag   : boolean (default False)
        Whether to use only the diagonal of the Fisher matrix

    Returns
    -------
    rew_val           : float if est_return is True, else None
        The estimated expected reward for this bandit
    grad_val          : numpy 1D-array if est_grad is True, else None
        The estimated policy gradient for this bandit
    """

    res = _importance_preprocess_gpomdp(rewards=rewards,
                                        gradients=gradients,
                                        p_tar=p_tar,
                                        p_gen=p_gen)
    return _importance_policy_gradient(res,
                                       use_baseline=use_baseline,
                                       use_natural_grad=use_natural_grad,
                                       fisher_diag=fisher_diag,
                                       ret_diagnostics=ret_diagnostics,
                                       **kwargs)


def _importance_policy_gradient(res, use_baseline, use_natural_grad,
                                fisher_diag, ret_diagnostics=False, **kwargs):
    """Implementation of importance-sampling based policy gradient computation.
    """
    try:
        if use_baseline:
            rew_b, grad_b = constant_isamp_baseline(rewards=res['traj_r'],
                                                    gradients=res['traj_grads'],
                                                    r_grads=res['r_grads'],
                                                    p_tar=res['traj_p_tar'],
                                                    p_gen=res['traj_p_gen'],
                                                    est_reward=True,
                                                    est_grad=True,
                                                    fisher_diag=fisher_diag,
                                                    **kwargs)
        else:
            rew_b = np.zeros((1))
            grad_b = np.zeros((1))

        rew_val = isamp.importance_sample(res['traj_r'] - rew_b,
                                          p_tar=res['traj_p_tar'],
                                          p_gen=res['traj_p_gen'],
                                          **kwargs)

        # Estimate the policy gradient
        grad_val = isamp.importance_sample(res['r_grads'] - grad_b,
                                           p_tar=res['traj_p_tar'],
                                           p_gen=res['traj_p_gen'],
                                           **kwargs)

        if use_natural_grad:
            act_fisher_chol = isample_fisher(gradients=res['act_grads'],
                                             p_tar=res['state_act_p_tar'],
                                             p_gen=res['state_act_p_gen'],
                                             diag=fisher_diag,
                                             **kwargs)
            grad_val = spl.cho_solve(act_fisher_chol, grad_val)

        if ret_diagnostics:
            traj_mw, ess = isamp.importance_sample_ess(p_gen=res['traj_p_gen'],
                                                       p_tar=res['traj_p_tar'],
                                                       **kwargs)
            rew_var, rew_var_ess = isamp.importance_sample_var(x=res['traj_r'] - rew_b,
                                                               est=rew_val,
                                                               p_tar=res['traj_p_tar'],
                                                               p_gen=res['traj_p_gen'],
                                                               **kwargs)
            grad_var, grad_var_ess = isamp.importance_sample_var(x=res['r_grads'] - grad_b,
                                                                 est=grad_val,
                                                                 p_tar=res['traj_p_tar'],
                                                                 p_gen=res['traj_p_gen'],
                                                                 **kwargs)
            if use_natural_grad:
                grad_var_acc = spl.cho_solve(act_fisher_chol, grad_var)
                grad_var = spl.cho_solve(act_fisher_chol, grad_var_acc.T).T

            return rew_val, grad_val, ess, rew_var, grad_var

        else:
            return rew_val, grad_val

    except isamp.SamplingException:
        print 'Sampling exception: Could not estimate gradient'
        return None, None
