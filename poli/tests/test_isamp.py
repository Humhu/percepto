import modprop
import numpy as np
import poli
import scipy.linalg as spl
from itertools import izip


class NormalDist(object):
    def __init__(self, mean, cov):
        self.x = modprop.ConstantModule(None)
        self.mean = modprop.ConstantModule(mean)
        self.cov = modprop.ConstantModule(cov)

        self.delx = modprop.DifferenceModule()
        modprop.link_ports(in_port=self.delx.left_port,
                           out_port=self.x.out_port)
        modprop.link_ports(in_port=self.delx.right_port,
                           out_port=self.mean.out_port)

        self.ll = modprop.LogLikelihoodModule()
        modprop.link_ports(in_port=self.ll.x_in,
                           out_port=self.delx.out_port)
        modprop.link_ports(in_port=self.ll.S_in,
                           out_port=self.cov.out_port)

        self.llSink = modprop.SinkModule()
        modprop.link_ports(in_port=self.llSink.in_port,
                           out_port=self.ll.ll_out)

    def sample(self):
        return np.random.multivariate_normal(mean=self.mean.value,
                                             cov=self.cov.value)

    def log_prob(self, x):
        self.invalidate()
        self.foreprop(x)
        return self.llSink.value

    def log_gradient(self, x):
        self.invalidate()
        self.foreprop(x)
        acc = modprop.AccumulatedBackprop(do_dx=np.identity(1))
        self.llSink.backprop_value = acc
        modprop.iterative_backprop(self.llSink)

        return np.hstack((self.mean.backprop_value[0],
                          self.cov.backprop_value[0]))

    def foreprop(self, x):
        self.x.value = x
        modprop.iterative_foreprop(self.x)
        modprop.iterative_foreprop(self.mean)
        modprop.iterative_foreprop(self.cov)

    def invalidate(self):
        modprop.iterative_invalidate(self.x)
        modprop.iterative_invalidate(self.mean)
        modprop.iterative_invalidate(self.cov)


import pdb

if __name__ == '__main__':
    dim = 1

    q_mean = np.random.rand(dim) - 0.5
    q_cov = np.diag(np.exp(np.random.rand(dim) - 0.5))
    q_dist = NormalDist(q_mean, q_cov)

    p_mean = np.random.rand(dim) - 0.5
    p_cov = np.diag(np.exp(np.random.rand(dim) - 0.5))
    p_dist = NormalDist(p_mean, p_cov)

    def estimate_q_mean(n_samples, use_baseline):
        samples = np.squeeze([q_dist.sample() for i in range(n_samples)])
        if use_baseline:
            grads = [q_dist.log_gradient(x) for x in samples]
            fisher_acc = np.mean([np.outer(g, g) for g in grads], axis=0)
            fisher = spl.cho_factor(fisher_acc)

            baseline = np.mean(
                [x * g for x, g in izip(samples, grads)], axis=0)
            baseline_vals = np.dot(grads, spl.cho_solve(fisher, baseline))
            samples = samples - baseline_vals

        return np.mean(samples)

    def estimate_p_mean(n_samples, use_baseline, min_weight=0):
        q_samples = np.squeeze([q_dist.sample() for i in range(n_samples)])
        qs = [q_dist.log_prob(x) for x in q_samples]
        ps = [p_dist.log_prob(x) for x in q_samples]

        if use_baseline:
            # Using access to samples from p
            # fisher_samples = [p_dist.sample() for i in range(n_samples)]
            # fisher_grads = [p_dist.log_gradient(x) for x in fisher_samples]
            # true_fisher_acc = np.mean([np.outer(g, g)
            #                            for g in fisher_grads], axis=0)
            # true_fisher = spl.cho_factor(true_fisher_acc)

            # true_baseline_acc = np.mean([x * g for x, g in izip(fisher_samples, fisher_grads)],
            #                             axis=0)

            # true_baseline = spl.cho_solve(true_fisher, true_baseline_acc)
            q_grads = [p_dist.log_gradient(x) for x in q_samples]
            # true_baseline_vals = np.dot(q_grads, true_baseline)

            est_fisher = poli.isample_fisher(q_grads,
                                             p_tar=ps,
                                             p_gen=qs,
                                             normalize=True)

            est_baseline_ests = [x * g for x, g in izip(q_samples, q_grads)]
            est_baseline_acc = poli.importance_sample(est_baseline_ests,
                                                      p_tar=ps,
                                                      p_gen=qs,
                                                      normalize=True,
                                                      min_weight=min_weight)
            est_baseline = spl.cho_solve(est_fisher, est_baseline_acc)
            est_baseline_vals = np.dot(q_grads, est_baseline)
            q_samples = q_samples - est_baseline_vals

        return poli.importance_sample(q_samples, p_gen=qs, p_tar=ps, normalize=True, min_weight=min_weight)

    n_samples = 30
    n_trials = 30

    # Estimating mean 1 using samples from mean 1
    #defa_estimates = [estimate_q_mean(n_samples, False) for i in range(n_trials)]
    #base_estimates = [estimate_q_mean(n_samples, True) for i in range(n_trials)]

    # Estimating mean 2 using samples from mean 1
    nobase_all_estimates = [estimate_p_mean(n_samples, False, 0) for i in range(n_trials)]
    nobase_filt_estimates = [estimate_p_mean(n_samples, False, 0.1) for i in range(n_trials)]
    base_all_estimates = [estimate_p_mean(n_samples, True, 0) for i in range(n_trials)]
    base_filt_estimates = [estimate_p_mean(n_samples, True, 0.1) for i in range(n_trials)]
    
