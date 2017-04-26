"""Script for processing the output of a step response experiment to determine
system parameters.
"""

import sys
import numpy as np
import cPickle as pickle
import mdentropy as mde
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Please specify file to process'
        sys.exit(-1)

    path = sys.argv[1]
    print 'Processing result file: %s' % path
    data = pickle.load(open(path))

    # 1. Decompose data log
    pre_actions = np.array(data['pre_actions'])
    actions = np.array(data['actions'])
    reward_traces = data['reward_traces']

    # 1.1 Normalize reward trace values
    trace_times = [zip(*tr)[0] for tr in reward_traces]
    trace_durations = [tt[-1] - tt[0] for tt in trace_times]
    min_trace_dur = min(trace_durations)
    print 'Setting trace duration to %f' % min_trace_dur

    num_trace_times = 30
    norm_trace_times = np.linspace(start=0, stop=min_trace_dur,
                                         num=num_trace_times)

    def normalize_trace(tr):
        ts, rs = zip(*tr)
        ts = np.array(ts) - ts[0]
        interp = interp1d(x=ts, y=rs, kind='linear')
        return interp(norm_trace_times)
    print 'Normalizing reward traces...'
    norm_reward_traces = [normalize_trace(tr) for tr in reward_traces]
    norm_reward_traces = np.array(norm_reward_traces)

    # TODO Consider context as well

    # 2. Compute MI between pre/actions, reward values
    pre_act_mi = [mde.mutinf(n_bins=11, x=pre_actions, y=norm_reward_traces[:,i])
                  for i in range(num_trace_times)]
    post_act_mi = [mde.mutinf(n_bins=11, x=actions, y=norm_reward_traces[:,i])
                  for i in range(num_trace_times)]

    plt.ion()
    plt.figure()
    plt.title('Instantaneous reward MI vs. time')
    plt.plot(norm_trace_times, pre_act_mi, 'r.-', label='prev_act')
    plt.plot(norm_trace_times, post_act_mi, 'b.-', label='curr_act')
    plt.xlabel('Time after action, t (s)')
    plt.ylabel('Mutual Information')
    plt.legend(loc='best')

    # 3. Compute MI across action dimensions, reward values
