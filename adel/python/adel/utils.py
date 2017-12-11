"""Utility classes and functions
"""

import tensorflow as tf
import numpy as np
import bisect
import scipy.interpolate as spi
import scipy.integrate as spt


class Convergence(object):
    """Checks convergence over a moving window.
    """

    def __init__(self, min_n, max_iter, min_iter=0,
                 test_slope=True, max_slope=0.1,
                 test_bounds=True, max_bounds=1.0, bound_deltas=False,
                 test_stats=True, max_sd=3.0):
        self.min_n = min_n
        self._hist = []
        self.iter = 0
        self.min_iter = min_iter
        self.max_iter = max_iter

        self.test_slope = test_slope
        self.max_slope = max_slope
        self.test_bounds = test_bounds
        self.max_bounds = max_bounds
        self.bound_deltas = bound_deltas
        self.test_stats = test_stats
        self.max_sd = max_sd
        self.last_converged = False

    def _test_slope(self, hist):
        if not self.test_slope:
            return True

        y = hist - np.mean(hist)
        x = np.arange(len(y))
        slope = np.dot(x, y) / np.dot(x, x)
        return np.abs(slope) < self.max_slope

    def _test_bounds(self, hist):
        if self.bound_deltas:
            hist = np.diff(hist)
        return np.all(np.abs(hist) < self.max_bounds)

    def _test_stats(self, hist):
        dev = hist - np.mean(hist)
        sd = np.std(dev)
        return np.all(np.abs(dev) < self.max_sd * sd)

    def clear(self):
        self._hist = []
        self.iter = 0
        self.last_converged = False

    def check(self, f):
        self._hist.append(f)
        self.iter += 1

        if self.iter < self.min_iter:
            return False
        if len(self._hist) < self.min_n:
            return False
        if self.iter > self.max_iter:
            return True
        self._hist = self._hist[-self.min_n:]
        hist = np.array(self._hist)

        self.last_converged = self._test_slope(hist) \
            and self._test_bounds(hist) \
            and self._test_stats(hist)
        return self.last_converged


def optimizer_initializer(opt, var_list):
    """Creates a list of initializers for resetting a tensorflow
    optimizer
    """
    opt_vars = [opt.get_slot(var, name)
                for name in opt.get_slot_names()
                for var in var_list]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_vars.extend(list(opt._get_beta_accumulators()))
    return tf.variables_initializer(opt_vars)


class Integrator(object):
    """Interpolates and integrates an asynchronously-sampled 1D signal
    """

    def __init__(self):
        self.times = []
        self.vals = []

    def __len__(self):
        return len(self.times)

    def buffer(self, t, v):
        if len(self) > 0 and t < self.times[-1]:
            raise ValueError('Received non-increasing time buffer!')
        self.times.append(t)
        self.vals.append(v)

    def integrate(self, t0, tf, weights=None):
        """Integrates the data contained in the integrator from t0 to
        tf, inclusive. Applies weights to the data according to function
        weights(times - t0)
        """
        if t0 < self.times[0] or tf > self.times[-1]:
            return None

        interp = spi.interp1d(x=np.squeeze(self.times),
                              y=np.squeeze(self.vals), axis=0)

        # TODO Clean up using bisect
        istart = next(i for i, x in enumerate(self.times) if x > t0)
        ifinal = next((i for i, x in enumerate(self.times) if x > tf), -1)

        times = [t0] + self.times[istart:ifinal] + [tf]
        ref = [interp(t0)] + self.vals[istart:ifinal] + [interp(tf)]
        times = np.array(times)
        ref = np.asarray(ref)

        if weights is not None:
            ref = ref * weights(times - t0)

        return spt.trapz(y=ref, x=times)

    def trim(self, t0):
        """Remove all data covering times before t0
        """
        if len(self) == 0:
            return

        if self.times[0] > t0:
            return
        if self.times[-1] <= t0:
            self.times = self.times[-1:]
            self.vals = self.vals[-1:]
            return

        i = bisect.bisect_right(self.times, t0) - 1
        self.times = self.times[i:]
        self.vals = self.vals[i:]


class ChangepointSeries(object):
    """Maps discrete samples of a time series into regions of continuity

    Parameters
    ==========
    extend_end : Whether the last value should extend to +infinity
    """

    def __init__(self, extend_start=False, extend_end=False):
        # breaks denote start/end borders between segments (N + 1)
        self.segment_breaks = []
        # values denote value throughout segment (N)
        self.segment_values = []
        self.extend_start = extend_start
        self.extend_end = extend_end

    def __repr__(self):
        s = ''
        for i in range(len(self.segment_values)):
            s += '(%s, [%f,%f])' % (str(self.segment_values[i]),
                                    self.segment_breaks[i], self.segment_breaks[i + 1])
        return s

    def __len__(self):
        return len(self.segment_values)

    def earliest_time(self):
        if len(self) == 0:
            return None
        return self.segment_breaks[0]

    def buffer(self, t, v):

        # First segment starts and ends at t
        if len(self) == 0:
            self.segment_breaks.append(t)
            self.segment_breaks.append(t)
            self.segment_values.append(v)
            return

        if t < self.segment_breaks[-1]:
            raise ValueError('Received time %f less than latest time %f' %
                             (t, self.segment_breaks[-1]))

        self.segment_breaks[-1] = t
        # If same as last value, keep going
        if v == self.segment_values[-1]:
            pass
        # Else add new segment
        else:
            self.segment_breaks.append(t)
            self.segment_values.append(v)

    def in_range(self, t):
        if len(self) == 0:
            return False

        return (t >= self.segment_breaks[0] or self.extend_start) and \
            (t <= self.segment_breaks[-1] or self.extend_end)

    def __segment_ind(self, t):
        i = bisect.bisect_right(self.segment_breaks, t) - 1
        if self.extend_start:
            i = max(0, i)
        if self.extend_end:
            i = min(len(self.segment_values) - 1, i)
        return i

    def get_value(self, t):
        if not self.in_range(t):
            return None
        i = self.__segment_ind(t)
        return self.segment_values[i]

    def in_same_segment(self, ta, tb):
        """Checks to see if ta and tb are in the same segment. Returns False if
        ta or tb lay outside the current range.
        """
        if not self.in_range(ta) or not self.in_range(tb):
            return False

        ia = bisect.bisect(self.segment_breaks, ta)
        ib = bisect.bisect(self.segment_breaks, tb)
        return ia == ib

    def trim(self, t0):
        """Removes all segments fully before t0
        """
        if len(self) == 0:
            return

        if t0 < self.segment_breaks[0]:
            return

        if t0 >= self.segment_breaks[-1]:
            x = self.segment_breaks[-1]
            self.segment_breaks = [x, x]
            self.segment_values = self.segment_values[-1:]
            return

        i = self.__segment_ind(t0)
        self.segment_breaks = self.segment_breaks[i:]
        self.segment_values = self.segment_values[i:]


class EventSeries(object):
    """Maps discrete timed events
    """

    def __init__(self):
        # denotes event times
        self.event_times = []

    def __len__(self):
        return len(self.event_times)

    def earliest_time(self):
        if len(self) == 0:
            return None
        return self.event_times[0]

    def buffer(self, t):
        # First segment starts and ends at t
        self.event_times.append(t)

    def in_range(self, t):
        if len(self) == 0:
            return False
        return t >= self.event_times[0] and t <= self.event_times[-1]

    def count_events(self, ta, tb):
        """Counts the number of events between ta and tb, inclusive
        """
        ia = bisect.bisect_left(self.event_times, ta)
        ib = bisect.bisect_right(self.event_times, tb)
        return ib - ia

    def trim(self, t0):
        """Removes all segments fully before t0
        """
        if len(self) == 0:
            return

        if t0 < self.event_times[0]:
            return
        i = bisect.bisect_right(self.event_times, t0) - 1
        self.event_times = self.event_times[i:]
