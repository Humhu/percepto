"""This module contains various sampling functions.
"""

import numpy as np


class SamplingException(Exception):
    pass


def importance_sample_ess(p_gen, p_tar, min_weight=0):
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

    weights = np.exp(p_tar - p_gen)
    valid = weights > min_weight
    if not np.any(valid):
        return float('nan'), 0

    weights = weights[valid]
    N = len(weights)

    mean_weight = np.mean(weights)
    mean_sq_weight = np.mean(weights * weights)
    ess = N * mean_weight / mean_sq_weight
    return mean_weight, ess


def importance_sample(x, p_gen, p_tar, normalize=False, min_weight=0):
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

    logprobs = p_tar - p_gen

    weights = np.exp(logprobs)
    valid = weights > min_weight
    print '%d/%d valid weights' % (np.sum(valid), len(x))

    if not np.any(valid):
        raise SamplingException()

    weights = weights[valid]
    x = x[valid]

    estimate = np.sum(weights * x.T, axis=-1).T

    if normalize:
        est_mean = estimate / np.sum(weights)
    else:
        est_mean = estimate / len(weights)

    # More diagnostics
    # deltas = x - est_mean
    # outers = np.asarray([np.outer(d, d) for d in deltas])
    # norm_weights = weights / np.sum(weights)
    # est_var = np.sum((norm_weights * norm_weights) * outers.T, axis=-1).T
    # est_std = np.sqrt(np.diag(est_var))
    # eff_var_size = np.sum(norm_weights ** 2) ** 2 / np.sum(norm_weights ** 4)
    # print 'SD est %s sample size %f' % (np.array_str(est_std), eff_var_size)

    return est_mean

# TODO: Deterministic Mixture MIS (DM-MIS)
