"""This module contains classes for time-delayed action evaluation.
"""

from threading import Lock
from collections import deque
import numpy as np
import scipy.integrate as spg
import scipy.interpolate as spi


class SARRecorder(object):
    """Buffers and returns integrated rewards between actions.

    Assumes that state-action tuples and rewards are reported in monotonically
    increasing temporal order, respectively.

    Accesses to the buffers are synchronized to allow simultaneous state-action and
    reward buffer accesses when appropriate.

    Parameters
    ----------
    integration_mode   : string (traps, simps)
    interpolation_mode : string (zero, linear, etc...)
        See scipy.interpolation.interp1d
    """

    def __init__(self, integration_mode, interpolation_mode='linear'):
        self._sa_buffer = deque()
        self._sa_lock = Lock()
        self._r_buffer = deque()
        self._r_lock = Lock()

        self._interp_mode = interpolation_mode
        if integration_mode == 'traps':
            self._integration_func = np.trapz
        elif integration_mode == 'simps':
            self._integration_func = spg.simps
        else:
            raise ValueError('Unknown integration mode: %s' % integration_mode)

    def report_state_action(self, time, state, action):
        """Buffers a timestamped state-action tuple.

        Parameters
        ----------
        time   : float
            The timestamp represented as a float
        state  : Any
        action : Any
        """
        self._sa_lock.acquire()
        if len(self._sa_buffer) > 0 and time <= self._sa_buffer[-1][0]:
            raise ValueError(
                'State-action tuples must be reported in temporal order')

        self._sa_buffer.append((time, state, action))
        self._sa_lock.release()

    def report_reward(self, time, reward):
        """Buffers a timestamped reward.

        Parameters
        ----------
        time   : float
            The timestamp represented as a float
        reward : Any
        """
        self._r_lock.acquire()
        if len(self._r_buffer) > 0 and time < self._r_buffer[-1][0]:
            raise ValueError('Rewards must be reported in temporal order')

        self._r_buffer.append((time, reward))
        self._r_lock.release()

    def process_buffers(self):
        """Processes the internal buffers to produce synchronized SAR tuples.

        Returns
        -------
        out : list of time-state-action-reward tuples
        """
        self._sa_lock.acquire()
        self._r_lock.acquire()

        out = []
        if len(self._r_buffer) < 2 or len(self._sa_buffer) < 1:
            self._sa_lock.release()
            self._r_lock.release()
            return out

        t_vals, r_vals = zip(*self._r_buffer)
        t_vals = np.array(t_vals)
        r_vals = np.array(r_vals)
        interp = spi.interp1d(x=t_vals, y=r_vals, kind=self._interp_mode)

        while len(self._sa_buffer) > 1:
            t, s, a = self._sa_buffer[0]
            tnext = self._sa_buffer[1][0]

            in_range = np.logical_and(t_vals > t, t_vals < tnext)
            ti_vals = t_vals[in_range]
            ri_vals = r_vals[in_range]
            try:
                if ti_vals[0] > t:
                    r = interp(t)
                    ti_vals = np.hstack((t, ti_vals))
                    ri_vals = np.hstack((r, ri_vals))
                if ti_vals[-1] < tnext:
                    rnext = interp(tnext)
                    ti_vals = np.hstack((ti_vals, tnext))
                    ri_vals = np.hstack((ri_vals, rnext))
            # interp throws ValueError when we go out of bounds
            except (IndexError, ValueError):
                break

            r_integrated = self._integration_func(y=ri_vals, x=ti_vals)
            self._sa_buffer.popleft()
            out.append((t, s, a, r))

        # Trim reward buffer down
        earliest_a_time = self._sa_buffer[0][0]
        while len(self._r_buffer) > 1 and self._r_buffer[1][0] < earliest_a_time:
            self._r_buffer.popleft()

        self._sa_lock.release()
        self._r_lock.release()
        return out


class EpisodeRecorder(object):
    """Converts timestamped SAR tuples into time-divided episodes.

    All calls are thread safe.
    """
    def __init__(self):
        self._sar_buffer = deque()
        self._ep_breaks = deque()
        self._sar_lock = Lock()
        self._ep_lock = Lock()

    def report_sar_tuples(self, sars):
        """Report timestamped SAR tuples.
        """
        self._sar_lock.acquire()
        self._sar_buffer.extend(sars)
        self._sar_lock.release()

    def report_episode_break(self, time):
        """Report a break between episodes.
        """
        self._ep_lock.acquire()
        self._ep_breaks.append(time)
        self._ep_lock.release()

    def retrieve_episodes(self):
        """Process internal buffers to produce episodes.
        """
        self._ep_lock.acquire()
        self._sar_lock.acquire()

        episodes = []
        while len(self._ep_breaks) > 0:
            next_break = self._ep_breaks[0]

            episode = []
            complete = False
            for t, s, a, r in self._sar_buffer:
                if t < next_break:
                    episode.append((t, s, a, r))
                else:
                    complete = True
                    break

            if not complete:
                break
            else:
                episodes.append(episode)
                self._ep_breaks.popleft()
                while self._sar_buffer[0][0] < next_break:
                    self._sar_buffer.popleft()

        self._ep_lock.release()
        self._sar_lock.release()
        return episodes
