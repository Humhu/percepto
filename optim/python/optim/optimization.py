"""Wrappers around optimizers and utility functions for partially-specified optimization tasks."""
import cma

def parse_optimizers(spec):
    """Takes a specificaiton dictionary and returns an optimizer.
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')

    optimizer_type = spec.pop('type')

    lookup = {'cma' : CMAOptimizer}
    if optimizer_type not in lookup:
        raise ValueError('Optimizer type %s not valid type: %s' %(optimizer_type, str(lookup.keys())))
    return lookup[optimizer_type](**spec)

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

class CMAOptimizer(object):
    """A wrapper around the cma library's CMAEvolutionaryStrategy.
    """
    def __init__(self, mode='min', num_restarts=0, **kwargs):
        if mode == 'min':
            self.k = 1
        elif mode == 'max':
            self.k = -1
        else:
            raise ValueError('mode must be min or max!')

        self.num_restarts = num_restarts
        self.opts = cma.CMAOptions()
        for key, value in kwargs.iteritems():
            if key not in self.opts:
                raise ValueError('No option %s for CMA' % key)
            self.opts[key] = value

    @property
    def lower_bounds(self):
        return self.opts['bounds'][0]

    @lower_bounds.setter
    def lower_bounds(self, l):
        self.opts['bounds'][0] = l

    @property
    def upper_bounds(self):
        return self.opts['bounds'][1]

    @upper_bounds.setter
    def upper_bounds(self, u):
        self.opts['bounds'][1] = u

    def optimize(self, x0, func):
        """Optimize the specified function.
        """
        def obj(x):
            return self.k * func(x)

        best = cma.BestSolution()
        for i in range(self.num_restarts+1):
            # TODO Set the initial standard deviations
            es = cma.CMAEvolutionStrategy(x0, 0.5, self.opts)
            #es.optimize(obj)
            while not es.stop():
                queries = es.ask()
                es.tell(queries, func(queries))
            best.update(es.best)
        return best.x, best.f
