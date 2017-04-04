"""This module contains classes for postprocessing policy outputs.
"""

import numpy as np

def parse_output_constraints(dim, spec):
    if 'type' not in spec:
        raise ValueError('Specification must include type!')

    norm_type = spec.pop('type')
    lookup = {'norm': NormConstrainer,
              'box': BoxConstrainer}
    if norm_type not in lookup:
        raise ValueError('Norm type %s not valid type: %s' %
                         (norm_type, str(lookup.keys())))
    return lookup[norm_type](dim=dim, **spec)


class NormConstrainer(object):
    """Constrains inputs to have a maximum norm.

    Parameters
    ----------
    type : string (L1, L2, Linf)
        What norm to use
    lim  : float
        A fixed norm constraint
    """

    def __init__(self, norm_type, mode, lim, dim=None):
        if norm_type == 'L1':
            self._norm = lambda x: np.linalg.norm(x, ord=1)
        elif norm_type == 'L2':
            self._norm = lambda x: np.linalg.norm(x, ord=2)
        elif norm_type == 'Linf':
            self._norm = lambda x: np.linalg.norm(x, ord=float('inf'))
        else:
            raise ValueError('Unknown norm type.')

        self.lim = lim

    def process(self, a):
        norm = self._norm(a)
        if norm > self.lim:
            a *= self.lim / norm
        return a


class BoxConstrainer(object):
    """Constrains inputs to a box.

    Parameters
    ----------
    lims : float, iterable of floats, or iterable of pairs of floats
    """

    def __init__(self, lims, dim=None):
        lims = np.array(lims)
        if len(lims.shape) == 0:
            if lims < 0:
                raise ValueError('Singular limit must be non-negative')
            if dim is None:
                raise ValueError('Must specify dim for singular limit')
            self._lower = np.full(dim, -lims)
            self._upper = np.full(dim, lims)

        elif len(lims.shape) == 1:
            if np.any(lims < 0):
                raise ValueError(
                    'List of singular limits must be non-negative')
            self._lower = -lims
            self._upper = lims

        elif len(lims.shape) == 2:
            if lims.shape[1] != 2:
                raise ValueError('Must give list of pairs of limits')
            if np.any(lims[0] > lims[1]):
                raise ValueError('Must give (min, max) limits')
            self._lower = lims[0]
            self._upper = lims[1]

        else:
            raise ValueError('Invalid limit shape')

    def process(self, a):
        a = np.max(np.vstack((a, self._lower)), axis=0)
        a = np.min(np.vstack((a, self._upper)), axis=0)
        return a
