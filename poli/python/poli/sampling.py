"""This module contains various sampling functions.
"""

import numpy as np

def importance_sample(x, p_gen, p_tar, log_prob=False, normalize=False):
    """Perform standard importance sampling.

    Parameters
    ----------
    x         : iterable of sample values
    p_gen     : iterable of generating distribution probabilities
    p_tar     : iterable of target distribution probabilities
    log_prob  : whether the probabilities are actually log-probabilities
    normalize : whether to perform self-normalization
    """
    x = np.asarray(x)
    N = len(x)

    if not np.iterable(p_gen):
        p_gen = np.full(N, p_gen)
    p_gen = np.asarray(p_gen)

    if not np.iterable(p_tar):
        p_tar = np.full(N, p_tar)
    p_tar = np.asarray(p_tar)

    if log_prob:
        weights = np.exp(p_tar - p_gen)
    else:
        weights = p_tar / p_gen

    estimate = np.sum(weights * x.T, axis=-1).T
    if normalize:
        estimate = estimate / np.sum(weights)

    return estimate

# TODO: Deterministic Mixture MIS (DM-MIS)