"""This module contains classes for time-delayed action evaluation.
"""

from threading import Lock
from collections import deque
import numpy as np
import scipy.integrate as spg
import scipy.interpolate as spi
from itertools import izip


class IntegratorBuffer(object):
    def __init__(self, integration_mode='traps', interpolation_mode='linear'):

        # TODO Check interp mode
        self._interp_mode = interpolation_mode

        if integration_mode == 'traps':
            self._integration_func = np.trapz
        elif integration_mode == 'simps':
            self._integration_func = spg.simps
        else:
            raise ValueError('Unknown integration mode: %s' % integration_mode)

        self._buffer = deque()

    def append(self, t, x):
        if len(self._buffer) > 0 and t <= self._buffer[-1][0]:
            raise ValueError('Time must be strictly increasing')
        self._buffer.append((t, x))

    def clear_before(self, t, inclusive=False):
        if inclusive:
            def test(): return self._buffer[0][0] <= t
        else:
            def test(): return self._buffer[0][0] < t

        while test():
            self._buffer.popleft()

    def integrate(self, t_start, t_end):
        t_vals, x_vals = zip(*self._buffer)
        t_vals = np.array(t_vals)
        x_vals = np.array(x_vals)

        # TODO Argument to enable/disable extrapolation
        interp = spi.interp1d(x=t_vals, y=x_vals, kind=self._interp_mode,
                              fill_value=(x_vals[0], x_vals[-1]), bounds_error=False)
        in_range = np.logical_and(t_vals >= t_start, t_vals <= t_end)

        # If there are no values between start and end, we will interpolate
        if not np.any(in_range):
            ti_vals = []
            xi_vals = []
        else:
            ti_vals = t_vals[in_range]
            xi_vals = x_vals[in_range]

        # Catches case where there are no reward messages between start
        # and end, in addition to when first value is after start
        if len(ti_vals) == 0 or ti_vals[0] > t_start:
            x_start = interp(t_start)
            ti_vals = np.hstack((t_start, ti_vals))
            xi_vals = np.hstack((x_start, xi_vals))
        # Case when last value is before end
        if ti_vals[-1] < t_end:
            x_end = interp(t_end)
            ti_vals = np.hstack((ti_vals, t_end))
            xi_vals = np.hstack((xi_vals, x_end))

        return self._integration_func(y=xi_vals, x=ti_vals)


class ContinuousRecorder(object):
    """Merges asynchronous state-action and reward streams to produce
    state-action-reward (SAR) tuples.

    Assumes that state-action tuples and rewards are reported in monotonically
    increasing temporal order, respectively, and that the system evolves 
    continuously. For systems with discontinuities or episodic behavior, see
    EpisodicRecorder.

    Accesses to the buffers are synchronized to allow simultaneous state-action and
    reward buffer accesses when appropriate.

    Parameters
    ----------
    mixing_mode        : string (average, integrate)
        Whether to time-integrate or average rewards between state-actions.
    integration_mode   : string (traps, simps)
        The integration mode to use.
    interpolation_mode : string (zero, linear, etc...)
        See scipy.interpolation.interp1d
    """

    def __init__(self, mixing_mode='average', integration_mode='traps',
                 interpolation_mode='linear'):
        self._sa_buffer = deque()
        self._sa_lock = Lock()
        self._r_buffer = IntegratorBuffer(integration_mode=integration_mode,
                                          interpolation_mode=interpolation_mode)
        self._r_lock = Lock()

        if mixing_mode == 'average':
            self._use_integration = False
        elif mixing_mode == 'integrate':
            self._use_integration = True
        else:
            raise ValueError('Unknown mixing mode')

    def report_state_action(self, time, state, action, logprob):
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
            self._sa_lock.release()
            raise ValueError(
                'State-action tuples must be reported in temporal order')

        self._sa_buffer.append((time, state, action, logprob))
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
        # TODO Catch ValueError and release lock
        self._r_buffer.append(t=time, x=reward)
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

        # Look at the first and second action in the state-action buffer
        while len(self._sa_buffer) > 1:
            t, s, a, l = self._sa_buffer[0]
            t_next = self._sa_buffer[1][0]
            try:
                r_integrated = self._r_buffer.integrate(
                    t_start=t, t_end=t_next)
            except ValueError:
                break

            if not self._use_integration:
                r_integrated = r_integrated / (t_next - t)

            self._sa_buffer.popleft()
            out.append((t, s, a, l, r_integrated))

        # Trim reward buffer down
        earliest_a_time = self._sa_buffer[0][0]
        self._r_buffer.clear_before(t=earliest_a_time, inclusive=False)

        self._sa_lock.release()
        self._r_lock.release()
        return out


class EpisodeRecorder(object):
    """Merges asynchronous state-action and reward streams to produce
    lists of state-action-reward (SAR) tuples, or episodes.

    Assumes that state-action tuples and rewards are reported in monotonically
    increasing temporal order, respectively.

    Accesses to the buffers are synchronized to allow simultaneous state-action and
    reward buffer accesses when appropriate.

    Parameters
    ----------
    mixing_mode        : string (average, integrate)
        Whether to time-integrate or average rewards between state-actions.
    integration_mode   : string (traps, simps)
        The integration mode to use.
    interpolation_mode : string (zero, linear, etc...)
        See scipy.interpolation.interp1d
    """

    def __init__(self, mixing_mode='average', integration_mode='traps',
                 interpolation_mode='linear'):
        self._sa_buffer = deque()
        self._sa_lock = Lock()

        self._r_buffer = deque()
        self._r_lock = Lock()

        self._ep_breaks = deque()
        self._ep_lock = Lock()

        if mixing_mode == 'average':
            self._use_integration = False
        elif mixing_mode == 'integrate':
            self._use_integration = True

        self._integration_mode = integration_mode
        self._interpolation_mode = interpolation_mode

    def report_state_action(self, time, state, action, logprob):
        """Buffers a timestamped state-action tuple.
        """
        self._sa_lock.acquire()
        if len(self._sa_buffer) > 0 and time <= self._sa_buffer[-1][0]:
            self._sa_lock.release()
            raise ValueError('State-action times must be strictly increasing')
        self._sa_buffer.append((time, state, action, logprob))
        self._sa_lock.release()

    def report_reward(self, time, reward):
        """Buffers a timestamped reward.
        """
        self._r_lock.acquire()
        if len(self._r_buffer) > 0 and time <= self._r_buffer[-1][0]:
            self._r_lock.release()
            raise ValueError('Reward times must be strictly increasing')
        self._r_buffer.append((time, reward))
        self._r_lock.release()

    def report_episode_break(self, time):
        """Report a break between episodes.
        """
        self._ep_lock.acquire()
        self._ep_breaks.append(time)
        self._ep_lock.release()

    def process_episodes(self):
        """Process internal buffers to produce episodes.
        """
        self._ep_lock.acquire()
        self._sa_lock.acquire()
        self._r_lock.acquire()

        episodes = []
        while len(self._ep_breaks) > 0 and \
                len(self._sa_buffer) > 0 and \
                len(self._r_buffer) > 0:
            next_break = self._ep_breaks[0]

            # Make sure that we have data past the break time to ensure that
            # the full episode is captured
            if self._sa_buffer[-1][0] < next_break or self._r_buffer[-1][0] < next_break:
                break
            self._ep_breaks.popleft()

            sa_data = []
            while self._sa_buffer[0][0] < next_break:
                sa_data.append(self._sa_buffer.popleft())

            r_data = IntegratorBuffer(integration_mode=self._integration_mode,
                                      interpolation_mode=self._interpolation_mode)
            while self._r_buffer[0][0] < next_break:
                data = self._r_buffer.popleft()
                r_data.append(t=data[0], x=data[1])

            episode = []
            for i in range(len(sa_data)):
                t, s, a, l = sa_data[i]
                try:
                    t_next = sa_data[i + 1][0]
                except IndexError:
                    t_next = next_break

                r = r_data.integrate(t_start=t, t_end=t_next)
                if not self._use_integration:
                    r = r / (t_next - t)

                episode.append((t, s, a, l, r))

            episodes.append(episode)

        self._r_lock.release()
        self._sa_lock.release()
        self._ep_lock.release()
        return episodes
