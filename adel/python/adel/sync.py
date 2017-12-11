"""Classes for synchronizing data into SAR tuples
"""

import numpy as np
from collections import deque
import scipy.interpolate as spi
import scipy.integrate as spt
from utils import Integrator, ChangepointSeries
from argus_utils import TimeSeries


class SARSSynchronizer(object):
    """Synchronizes and duplicates data to form SARS tuples.

    Forms tuples of temporal length dt. Note that these tuples may overlap.

    Optionally can apply decaying weights when integrating rewards, useful for 
    using this class for state-action-value integration.
    """

    def __init__(self, dt, tol, gamma=0):
        self.dt = dt
        self.tol = tol

        self.state_map = TimeSeries()
        self.action_map = ChangepointSeries(extend_end=True)
        self.break_map = ChangepointSeries(extend_end=True)
        self.reward_integrator = Integrator()

        self.gamma = gamma

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

    @property
    def num_states_buffered(self):
        return len(self.state_map)

    @property
    def num_actions_buffered(self):
        return len(self.action_map)

    @property
    def num_rewards_buffered(self):
        return len(self.reward_integrator)

    def _decay_weights(self, t):
        w = np.exp(-self.gamma * t)
        w[w > self.dt] = 0
        return w

    def process(self, now):
        """Process the internal buffers up to now, grouping data
        into SARS tuples and terminal SA tuples. Should be called with a
        lag in now to ensure all messages are received.

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

        if len(self.state_map) == 0 or len(self.action_map) == 0 \
                or len(self.reward_integrator) == 0:
            return sars, terminals

        while len(self.state_map) > 0:
            # 0. Find earliest state and see if corresponding
            # state at t + dt exists close enough
            t, s_t = self.state_map.earliest_item()

            # 1. If tn passes time threshold, come back later
            if t + self.dt + self.tol > now:
                # print 'tn %f passes now %f' % (t + self.dt, now)
                break

            item_n = self.state_map.get_closest_either(t + self.dt)
            if item_n is None:
                self.state_map.remove_earliest()
                # print 'Could not retrieve s_tn for tn=%f' % (t + self.dt)
                continue

            tn, s_tn = item_n
            if abs(tn - (t + self.dt)) > self.tol:
                self.state_map.remove_earliest()
                # print 'Could not retrieve s_tn within tolerance tn_req=%f ret=%f' % (t + self.dt, tn)
                continue

            # 2. Make sure t is in an active episode
            ep_t = self.break_map.get_value(t)
            # NOTE tn cannot exceed break_map range if lag assumption is true
            if ep_t is None or ep_t is False:
                self.state_map.remove_earliest()
                # print 't=%f not active episode' % t
                continue

            # 3. Make sure t has a valid action
            a_t = self.action_map.get_value(t=t)
            if a_t is None:
                self.state_map.remove_earliest()
                # print 't=%f does not have valid action' % t
                continue

            # 4. See if [t, tn] covers an episode termination
            # NOTE Since we know t is active, if t and tn are not in same
            # segment then tn must be inactive
            if not self.break_map.in_same_segment(t, tn):
                # 4.a If [t, tn] covers a termination, build termination tuple
                terminals.append((s_t, a_t))
                self.state_map.remove_earliest()
                # print 'Terminated state t=%f, tn=%f' % (t, tn)
                continue

            # 5. Make sure has the same action as t
            if not self.action_map.in_same_segment(t, tn):
                self.state_map.remove_earliest()
                # print 't=%f and tn=%f have different actions' % (t, tn)
                continue

            # 6. Integrate rewards
            r_t = self.reward_integrator.integrate(t, tn, self._decay_weights)
            if r_t is None:
                self.state_map.remove_earliest()
                # print 'Integration failed for [%f,%f]' % (t, tn)
                continue

            # 7. Cleared all checks, create SARS tuple
            # print 'Adding tuple from [%f, %f] with r: %f' % (t, tn, r_t)
            sars.append((s_t, a_t, r_t, s_tn))
            self.state_map.remove_earliest()

        # We've processed all data at least preceding t, so we can trim to that
        # point
        self.action_map.trim(t)
        self.break_map.trim(t)
        self.reward_integrator.trim(t)

        return sars, terminals
