"""This module contains importance sampling functions.
"""

import numpy as np

def importance_sample(x, p_old, p_new, log_prob=False, normalize=False):
    """Perform standard importance sampling.

    Parameters
    ----------
    x         : iterable of sample values
    p_old     : iterable of generating distribution probabilities
    p_new     : iterable of target distribution probabilities
    log_prob  : whether the probabilities are actually log-probabilities
    normalize : whether to perform self-normalization
    """
    x = np.asarray(x)
    N = len(x)

    if not np.iterable(p_old):
        p_old = np.full(N, p_old)
    p_old = np.asarray(p_old)

    if not np.iterable(p_new):
        p_new = np.full(N, p_new)
    p_new = np.asarray(p_new)

    if log_prob:
        weights = np.exp(p_new - p_old)
    else:
        weights = p_new / p_old

    estimate = np.sum(weights * x.T, axis=-1).T
    if normalize:
        estimate = estimate / np.sum(weights)

    return estimate

# TODO: Deterministic Mixture MIS (DM-MIS)