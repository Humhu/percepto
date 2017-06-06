"""Functions and classes for training and using additive Gaussian process models
"""

from GPy.kern import RBF, Add
from GPy.models import GPRegression
from itertools import combinations

def stringify_tuple(inds):
    s = ''
    for i in inds:
        s += '_%d' % i
    return s

def create_subkernels(input_dim, max_subset=1, base_kernel=RBF, **kwargs):
    """Creates a sum kernel composed of single-dimensional kernels.
    """
    kernels = []
    for s in range(max_subset):
        for inds in combinations(range(input_dim), s+1):
            kern = base_kernel(input_dim=len(inds),
                               active_dims=inds,
                               name='kernel%s' % stringify_tuple(inds),
                               **kwargs)
            kernels.append(kern)
    return Add(subkerns=kernels)
