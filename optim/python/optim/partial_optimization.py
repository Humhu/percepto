"""Classes and functions for optimizing subsets of variables (partial optimization)
"""

class PartialOptimizationWrapper(object):
    """Wraps an objective function to allow optimization with partial input specification.

    Parameters
    ----------
    func : callable
        The objective function to wrap
    """
    def __init__(self, func):
        self.func = func
        self.x_part = []
        self.part_inds = []

    def get_partial_input(self):
        """Return the current partial input specification.
        """
        return self.x_part

    def set_partial_input(self, x):
        """Specify the partial input for this wrapper.

        Parameters
        ----------
        x : iterable
            Iterable with 'None' in positions that are unspecified
        """
        self.x_part = x
        self.part_inds = [i for i, v in self.x_part if v is None]

    def generate_full_input(self, x):
        """Generates a fully-specified input by filling in the empty partial positions
        with the elements of x, in order.

        Parameters
        ----------
        x : iterable
            Elements to fill in the None positions of the partial input.

        Returns
        -------
        x_full : list
            List copy of fully-specified input
        """
        if len(x) != len(self.part_inds):
            raise ValueError('Expected %d inputs but got %d' % (len(x), len(self.part_inds)))
        x_full = list(self.x_part)
        x_full[self.part_inds] = x
        return x_full

    def __call__(self, x):
        """Returns the objective function evaluated with the partial input
        empty positions filled by x in order of occurrence.
        """
        return self.func(self.generate_full_input(x))