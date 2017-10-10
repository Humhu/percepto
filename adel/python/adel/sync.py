"""Classes for synchronizing data into SAR tuples
"""

from collections import deque
import scipy.interpolate as spi
import scipy.integrate as spt


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

    def integrate(self, t0, tf):
        if t0 < self.times[0] or tf > self.times[-1]:
            raise ValueError('Requested (%f,%f) for limits (%f,%f)' %
                             (t0, tf, self.times[0], self.times[-1]))

        interp = spi.interp1d(x=self.times, y=self.vals)
        istart = next(i for i, x in enumerate(self.times) if x > t0)
        ifinal = next((i for i, x in enumerate(self.times) if x > tf), -1)

        times = [t0] + self.times[istart:ifinal] + [tf]
        ref = [interp(t0)] + self.vals[istart:ifinal] + [interp(tf)]
        return spt.trapz(y=ref, x=times)

    def trim(self, t0):
        """Remove all data before t0"""
        if self.times[0] > t0:
            return
        elif self.times[-1] < t0:
            self.times = []
            self.vals = []
            return
        istart = next((i for i, x in enumerate(self.times) if x >= t0), -1)
        self.times = self.times[istart:]
        self.vals = self.vals[istart:]


class TimeMap(object):
    """Provides convenience methods for finding items closest in temporal order
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

    def get_closest_before(self, t):
        if t < self.times[0]:
            raise ValueError('Cannot retrieve before %f for earliest %f' %
                             (t, self.times[0]))
        istart = next((i for i, x in enumerate(self.times) if x > t), 0)
        return self.times[istart - 1], self.vals[istart - 1]

    def get_closest_after(self, t):
        if t > self.times[-1]:
            raise ValueError('Cannot retrieve after %f for latest %f' %
                             (t, self.times[-1]))
        istart = next(i for i, x in enumerate(self.times) if x >= t)
        return self.times[istart], self.vals[istart]

    def get_closest(self, t, tol=float('inf')):
        bt, bv = self.get_closest_before(t)
        at, av = self.get_closest_after(t)
        db = t - bt
        da = at - t
        if db > tol and da > tol:
            return None, None
        elif db < da:
            return bt, bv
        else:
            return at, av

    def trim(self, t0):
        """Removes all data before t0
        """
        if self.times[0] > t0:
            return
        elif self.times[-1] < t0:
            self.times = []
            self.vals = []
            return
        istart = next((i for i, x in enumerate(self.times) if x >= t0), -1)
        self.times = self.times[istart:]
        self.vals = self.vals[istart:]


class SARSSynchronizer(object):
    """Synchronizes and duplicates data to form SAR tuples.

    Forms tuples of temporal length dt
    """

    def __init__(self, dt, lag, tol):
        if dt > lag:
            raise ValueError('dt %f must be less than lag %f' % (dt, lag))
        self.dt = dt
        self.lag = lag
        self.tol = tol

        self.state_map = TimeMap()
        self.action_map = TimeMap()
        self.reward_integrator = Integrator()

    def buffer_state(self, s, t):
        self.state_map.buffer(t=t, v=s)

    def buffer_action(self, a, t):
        self.action_map.buffer(t=t, v=a)

    def buffer_reward(self, r, t):
        self.reward_integrator.buffer(t=t, v=r)

    def process(self, now):

        sars = []
        if len(self.action_map) < 2:
            return sars
        
        # Ensure on first process that first action precedes others
        self.state_map.trim(self.action_map.times[0])
        self.reward_integrator.trim(self.action_map.times[0])

        for t in self.state_map.times:
            tn = t + self.dt
            if tn > now - self.lag:
                break

            a = self.action_map.get_closest_before(t)[1]
            an = self.action_map.get_closest_before(tn)[1]
            s = self.state_map.get_closest(t=t, tol=self.tol)[1]
            sn = self.state_map.get_closest(t=tn, tol=self.tol)[1]
            r = self.reward_integrator.integrate(t, tn)

            # If at an action change crossover
            if a != an:
                continue
            # If not within tolerance
            if s is None or sn is None:
                continue
            
            sars.append((s, a, r, sn))

        self.state_map.trim(t)
        # If earliest state is past second-earliest action, can trim earliest action
        if self.state_map.times[0] >= self.action_map.times[1]:
            self.action_map.trim(self.action_map.times[1])
        self.reward_integrator.trim(t)

        return sars