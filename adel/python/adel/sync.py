"""Classes for synchronizing data into SAR tuples
"""

from collections import deque
import scipy.interpolate as spi
import scipy.integrate as spt
from utils import Integrator, ChangepointSeries
from argus_utils import TimeSeries

class SARSSynchronizer(object):
    """Synchronizes and duplicates data to form SAR tuples.

    Forms tuples of temporal length dt. Relies on a processing lag to ensure that
    all messages are received. Specifically, assumes that no message will have
    delay greater than parameter lag seconds.
    """

    def __init__(self, dt, lag, tol):
        if dt > lag:
            raise ValueError('dt %f must be less than lag %f' % (dt, lag))
        self.dt = dt
        self.lag = lag
        self.tol = tol

        self.state_map = TimeSeries()
        self.action_map = ChangepointSeries()
        self.break_map = ChangepointSeries()
        self.reward_integrator = Integrator()

    def buffer_state(self, s, t):
        self.state_map.insert(time=t, val=s)

    def buffer_action(self, a, t):
        self.action_map.buffer(t=t, v=a)

    def buffer_reward(self, r, t):
        self.reward_integrator.buffer(t=t, v=r)

    def buffer_episode_active(self, t):
        self.break_map.buffer(t=t, v=True)

    def buffer_episode_terminate(self, t):
        self.break_map.buffer(t=t, v=False)

    def process(self, now):
        """Process the internal buffers up to now - lag, grouping data
        into SARS tuples and terminal SA tuples.

        Parameters
        ==========
        now : The current time when calling process, needed to lag correctly

        Returns
        =======
        sars      : list of SARS tuples for value training
        terminals : list of SA tuples resulting in episode end within dt
        """
        sars = []
        terminals = []

        if len(self.state_map) == 0 or len(self.action_map) == 0 or \
            len(self.break_map) == 0 or len(self.reward_integrator) == 0:
            return sars

        while len(self.state_map) > 0:
            # 0. Find earliest state and see if corresponding
            # state at t + dt exists close enough
            t, s_t = self.state_map.earliest_item()
            item_n = self.state_map.get_closest_either(t + self.dt)
            if item_n is None:
                self.state_map.remove_earliest()
                print 'Could not retrieve s_tn for tn=%f' % (t + self.dt)
                continue
            tn, s_tn = item_n
            if abs(tn - (t + self.dt)) > self.tol:
                self.state_map.remove_earliest()
                print 'Could not retrieve s_tn within tolerance for tn=%f' % tn
                continue

            # 1. If tn passes lag threshold, come back later
            # NOTE Should actually be now - self.lag - self.tol
            if tn > now - self.lag:
                break

            # 2. Make sure t is in an active episode
            ep_t = self.break_map.get_value(t)
            # NOTE tn cannot exceed break_map range if lag assumption is true            
            if ep_t is None or ep_t is False:
                self.state_map.remove_earliest()
                print 't=%f not active episode' % t
                continue

            # 3. Make sure t has a valid action
            a_t = self.action_map.get_value(t=t)
            # NOTE tn cannot exceed action_map range if lag assumption is true            
            if a_t is None:
                self.state_map.remove_earliest()
                print 't=%f does not have valid action' % t
                continue

            # 4. See if [t, tn] covers an episode termination
            # NOTE Since we know t is active, if t and tn are not in same
            # segment then tn must be inactive
            # NOTE tn cannot exceed break_map range if lag assumption is true            
            if not self.break_map.in_same_segment(t, tn):
                # 4.a If [t, tn] covers a termination, build termination tuple
                terminals.append((s_t, a_t))
                self.state_map.remove_earliest()
                print 'Terminated state t=%f' % t
                continue

            # 5. Make sure has the same action as t
            # NOTE tn cannot exceed action_map range if lag assumption is true
            if not self.action_map.in_same_segment(t, tn):
                self.state_map.remove_earliest()
                print 't=%f and tn=%f have different actions' % (t, tn)
                continue

            # 6. Integrate rewards
            # NOTE tn cannot exceed integrator range if lag assumption is true
            r_t = self.reward_integrator.integrate(t, tn)
            if r_t is None:
                self.state_map.remove_earliest()
                print 'integration failed'
                continue

            # 7. Cleared all checks, create SARS tuple
            sars.append((s_t, a_t, r_t, s_tn))
            self.state_map.remove_earliest()

        # We've processed all data at least preceding t, so we can trim to that point
        self.action_map.trim(t)
        self.break_map.trim(t)
        self.reward_integrator.trim(t)

        return sars, terminals
