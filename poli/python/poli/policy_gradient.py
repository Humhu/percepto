"""This module contains implementations of algorithms for computing likelihood ratio
policy gradients.
"""
import numpy as np
import poli.sampling as isamp

def bandit_importance(rewards, gradients, curr_probs, past_probs, log_prob,
                      est_reward=True, est_grad=True, use_natural_grad=True,
                      inv_fisher_offset=1E-9, inv_fisher_diag=False):
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
    inv_fisher_offset : double (default 1E-9)
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
    grad_ops = np.asarray([np.outer(g, g) for g in gradients])
    inv_fisher = isamp.importance_sample(grad_ops,
                                         p_gen=past_probs,
                                         p_tar=curr_probs,
                                         log_prob=log_prob,
                                         normalize=True)
    if inv_fisher_diag:
        inv_fisher = np.diag(np.diag(inv_fisher))
    inv_fisher += inv_fisher_offset * np.identity(inv_fisher.shape[0])

    # Estimate the expected reward
    rew_val = None
    if est_reward:
        rew_baseline_ests = (rewards * gradients.T).T
        rew_baseline_acc = isamp.importance_sample(rew_baseline_ests,
                                                   p_gen=past_probs,
                                                   p_tar=curr_probs,
                                                   log_prob=log_prob,
                                                   normalize=True)
        rew_baseline = np.linalg.solve(inv_fisher, rew_baseline_acc)
        rew_ests = rewards - np.dot(gradients, rew_baseline)
        rew_val = isamp.importance_sample(rew_ests,
                                          p_gen=past_probs,
                                          p_tar=curr_probs,
                                          log_prob=log_prob,
                                          normalize=True)

    # Estimate the policy gradient
    grad_val = None
    if est_grad:
        grad_baseline_ests = (rewards * grad_ops.T).T
        grad_baseline_acc = isamp.importance_sample(grad_baseline_ests,
                                                    p_gen=past_probs,
                                                    p_tar=curr_probs,
                                                    log_prob=log_prob,
                                                    normalize=True)
        grad_baseline = np.linalg.solve(inv_fisher, grad_baseline_acc)
        grad_ests = (rewards * gradients.T).T - \
            np.dot(gradients, grad_baseline)
        grad_val = isamp.importance_sample(grad_ests,
                                           p_gen=past_probs,
                                           p_tar=curr_probs,
                                           log_prob=log_prob,
                                           normalize=True)
        if use_natural_grad:
            grad_val = np.linalg.solve(inv_fisher, grad_val)

    return rew_val, grad_val
