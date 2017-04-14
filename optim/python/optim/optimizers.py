"""Wrappers around optimizers and utility functions for partially-specified optimization tasks.
"""

import abc
import numpy as np
import cma


def parse_optimizers(spec):
    """Takes a specification dictionary and returns an optimizer.

    The spec should specify the optimizer type and any other constructor
    keyword arguments.

    Optimizers:
    -----------
    cma_es           :
        Covariance Matrix Adaptation Evolutionary Strategy, using the
        python cma library
    gradient_descent :
        Vanilla gradient descent algorithm
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')

    optimizer_type = spec.pop('type')

    lookup = {'cma_es': CMAOptimizer,
              'gradient_descent': GradientDescent}
    if optimizer_type not in lookup:
        raise ValueError('Optimizer type %s not valid type: %s' %
                         (optimizer_type, str(lookup.keys())))
    return lookup[optimizer_type](**spec)


class Optimizer(object):
    """The standard interface for optimizers.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, x_init, func):
        """Perform one round of optimization on the specified problem.

        Parameters
        ----------
        x_init : numpy ND-array
            The input to begin optimization at
        func   : function
            The optimization problem to optimize. Returns (objective, gradient)

        Returns
        -------
        x_next : numpy ND-array
            One-step optimized input
        """
        pass

    @abc.abstractmethod
    def optimize(self, x_init, func):
        """Optimize a problem until convergence.

        Parameters
        ----------
        x_init : numpy ND-array
            The input to begin optimization at
        func   : function
            The optimization problem to optimize. Returns (objective, gradient)

        Returns
        -------
        x_opt  : numpy ND-array
            The final optimized input
        y_opt  : float
            The final optimized objective
        """
        pass


class CMAOptimizer(Optimizer):
    """A wrapper around the cma library's CMAEvolutionaryStrategy.

    Parameters
    ----------
    mode
    num_restarts
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

    def step(self, x_init, func):
        raise RuntimeError('CMA optimizer does not have step mode')

    def optimize(self, x_init, func):
        """Optimize the specified function.
        """
        def obj(x):
            return self.k * func(x)[0]

        best = cma.BestSolution()
        for i in range(self.num_restarts + 1):
            # TODO Set the initial standard deviations
            es = cma.CMAEvolutionStrategy(x_init, 0.5, self.opts)
            # es.optimize(obj)
            while not es.stop():
                queries = es.ask()
                es.tell(queries, func(queries))
            best.update(es.best)
        return best.x, best.f


class GradientDescent(Optimizer):
    # TODO Add bounds
    """A simple gradient descent algorithm. Requires problems to return both the
    objective value and the gradient.

    Parameters
    ----------
    mode          : string (min or max)
        The optimization mode
    step_size     : float (default 0.1)
        The gradient step scale
    max_l2_norm   : float (default inf)
        The maximum allowable gradient L2 norm
    max_linf_norm : float (default inf)
        The maximum allowable gradient infinity norm
    max_iters     : integer (default 0)
        The maximum number of iterations for optimization. 0 means unlimited.
    x_tol         : float (default 1E-3)
        Minimum change in successive x L2 norm to converge
    grad_tol      : float (default 1E-3)
        Minimum gradient L2 norm to converge
    y_tol         : float (default 0)
        Minimum change in successive absolute objective value to converge
    """

    def __init__(self, mode='min', step_size=0.1, max_l2_norm=float('inf'),
                 max_linf_norm=float('inf'), max_iters=0, x_tol=1E-6,
                 grad_tol=1E-6, y_tol=0):
        if mode == 'min':
            self._k = -1
        elif mode == 'max':
            self._k = 1
        else:
            raise ValueError('mode must be min or max!')

        self._step_size = float(step_size)
        self._max_l2 = float(max_l2_norm)
        self._max_linf = float(max_linf_norm)
        self._max_iters = int(max_iters)
        self._x_tol = float(x_tol)
        self._grad_tol = float(grad_tol)
        self._y_tol = float(y_tol)

    def step(self, x_init, func):
        return self.__step(x_init, func)[0:2]

    def optimize(self, x_init, func):
        converged = False
        iter_count = 0
        x_curr = x_init
        y_curr = None
        while not converged:

            # NOTE This is actually y_curr, so we will be one step behind
            # when it comes to checking y_tol
            x_next, y_next, grad = self.__step(x_init=x_curr, func=func)
            if y_next is None or grad is None:
                return x_next, y_next

            # Check convergence
            iter_count += 1
            if self._max_iters != 0 and iter_count >= self._max_iters:
                converged = True

            if np.linalg.norm(x_next - x_curr, ord=2) < self._x_tol:
                converged = True
            elif np.linalg.norm(grad) < self._grad_tol:
                converged = True
            elif y_curr is not None and abs(y_next - y_curr) < self._y_tol:
                converged = True

            x_curr = x_next
            y_curr = y_next

        return x_curr, y_curr

    def __step(self, x_init, func):
        """Internal function for running one iteration of optimization.

        Returns
        -------
        x_next
        y_init
        grad
        """
        y_init, grad = func(x=x_init)
        if y_init is None or grad is None:
            return x_init, None, None

        step = grad * self._step_size * self._k

        # Check for infinity norm violations
        step_linf = np.linalg.norm(step, ord=float('inf'))
        if step_linf > self._max_linf:
            linf_scale = self._max_linf / step_linf
        else:
            linf_scale = 1

        # Check for L2 norm violations
        step_l2 = np.linalg.norm(step, ord=2)
        if step_l2 > self._max_l2:
            l2_scale = self._max_l2 / step_l2
        else:
            l2_scale = 1

        adj_scale = min(linf_scale, l2_scale)

        x_next = x_init + adj_scale * step
        return x_next, y_init, grad
