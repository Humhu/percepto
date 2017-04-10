"""This module contains various sampling functions.
"""

import numpy as np


class SamplingException(Exception):
    pass


def importance_sample_ess(p_gen, p_tar, log_weight_lim=float('inf'), normalize=False):
    # TODO Normalize not used?
    """Computes prior diagnostics for importance sampling.

    Parameters
    ----------
    x           : iterable of sample values
    p_gen       : iterable of generating distribution log-probabilities
    p_tar       : iterable of target distribution log-probabilities

    Returns
    -------
    mean_weight : Ideally near 1.0
    ess         : Effective sample size
    """
    N = len(p_gen)

    if not np.iterable(p_gen):
        p_gen = np.full(N, p_gen)
    else:
        p_gen = np.asarray(p_gen)

    if not np.iterable(p_tar):
        p_tar = np.full(N, p_tar)
    else:
        p_tar = np.asarray(p_tar)

    log_weights = p_tar - p_gen
    valid = np.logical_and(log_weights > -log_weight_lim,
                           log_weights < log_weight_lim)
    if not np.any(valid):
        return float('nan'), 0

    weights = np.exp(log_weights[valid])

    mean_weight = np.mean(weights)
    mean_sq_weight = np.mean(weights * weights)
    ess = len(weights) * mean_weight ** 2 / mean_sq_weight
    return mean_weight, ess


def importance_sample_var(x, est, p_gen, p_tar, log_weight_lim=float('inf'),
                          normalize=False):
    # TODO normalize not used?
    x = np.asarray(x)
    N = len(x)

    if not np.iterable(p_gen):
        p_gen = np.full(N, p_gen)
    else:
        p_gen = np.asarray(p_gen)

    if not np.iterable(p_tar):
        p_tar = np.full(N, p_tar)
    else:
        p_tar = np.asarray(p_tar)

    log_weights = p_tar - p_gen
    valid = np.logical_and(log_weights > -log_weight_lim,
                           log_weights < log_weight_lim)
    if not np.any(valid):
        return float('inf'), 0

    weights = np.exp(log_weights[valid])
    x = x[valid]

    deltas = x - est
    outers = np.asarray([np.outer(d, d) for d in deltas])
    norm_weights = weights / np.sum(weights)

    est_var = np.sum((norm_weights * norm_weights) * outers.T, axis=-1).T
    ess = np.sum(norm_weights ** 2) ** 2 / np.sum(norm_weights ** 4)
    return est_var, ess


def importance_sample(x, p_gen, p_tar, normalize=False, log_weight_lim=float('inf')):
    """Perform standard importance sampling.

    Parameters
    ----------
    x           : iterable of sample values
    p_gen       : iterable of generating distribution log-probabilities
    p_tar       : iterable of target distribution log-probabilities
    normalize   : whether to perform self-normalization
    min_logprob : minimum log probability to discard sample
    """
    x = np.asarray(x)
    N = len(x)

    if not np.iterable(p_gen):
        p_gen = np.full(N, p_gen)
    else:
        p_gen = np.asarray(p_gen)

    if not np.iterable(p_tar):
        p_tar = np.full(N, p_tar)
    else:
        p_tar = np.asarray(p_tar)

    log_weights = p_tar - p_gen
    valid = np.logical_and(log_weights > -log_weight_lim,
                           log_weights < log_weight_lim)
    if not np.any(valid):
        raise SamplingException()

    weights = np.exp(log_weights[valid])
    x = x[valid]
    estimate = np.sum(weights * x.T, axis=-1).T

    if normalize:
        est_mean = estimate / np.sum(weights)
    else:
        est_mean = estimate / len(weights)
    return est_mean

# TODO: Deterministic Mixture MIS (DM-MIS)
