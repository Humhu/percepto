"""This module contains various sampling functions.
"""

import numpy as np

def importance_sample(x, p_gen, p_tar, normalize=False, min_logprob=float('-inf')):
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

    weight_valid = logprobs > min_logprob
    gen_valid = p_gen > min_logprob
    #tar_valid = p_tar > min_logprob
    valid = np.logical_and(weight_valid, gen_valid)

    weights = np.exp(logprobs[valid])
    x = x[valid]

    estimate = np.sum(weights * x.T, axis=-1).T

    if normalize:
        return estimate / np.sum(weights)
    else:
        return estimate / len(weights)

# TODO: Deterministic Mixture MIS (DM-MIS)
