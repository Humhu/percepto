"""Classes for re-balancing data.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


class KernelDataResampler(object):
    """Resamples data using a kernel density estimate and importance sampling.
    """
    def __init__(self, use_replacement=True, kde_args=None):
        if kde_args is None:
            kde_args = {}
        self.kde = KernelDensity(*kde_args)

        self.data = []
        self.use_replacement = use_replacement

    def clear(self):
        self.data = []

    def report_sample(self, data):
        """Adds a sample to the resampler.
        """
        self.data.append(data)

    def sample_data(self, n_samples, pfunc=None):
        """Select data balanced according to a target distribution.
        """
        self.kde.fit(self.data)
        logq = self.kde.score_samples(self.data)

        if pfunc is None:
            logp = 0
        else:
            logp = [pfunc(x) for x in self.data]

        w = np.exp(logp - logq)
        w = w / np.sum(w)
        inds = range(len(self.data))
        picks = np.random.choice(inds,
                                 size=n_samples,
                                 replace=self.use_replacement,
                                 p=w)
        picked_data = [self.data[i] for i in picks]
        return picked_data, picks