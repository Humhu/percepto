import numpy as np

import scipy.spatial as sspatial
from sklearn.neighbors import KRadiusNeighborsRegressor, NearestNeighbors

import optim
import poli

from GPy.models import GPRegression

import matplotlib.pyplot as plt
import matplotlib.cm as mcm


class NNR(object):
    def __init__(self, r=1.0, k=10, def_mean=0, def_sd=float('inf'), off=0.1):
        self.reg = KRadiusNeighborsRegressor(
            n_neighbors=k, radius=r, defval=def_mean, weights=self.comp_weights)
        # Regresses squared error
        self.err_reg = NearestNeighbors(n_neighbors=k, radius=r)
        self.off = off
        self.def_sd = float(def_sd)

    def comp_weights(self, dists):
        return 1.0 / (dists + self.off)

    def fit(self, X, y):
        self.reg.fit(X, y)
        errs = y - self.reg.predict(X)
        self.sse = errs * errs
        self.err_reg.fit(X)

    def predict(self, X, return_std=False):
        if not return_std:
            return self.reg.predict(X)
        else:
            mean_val = self.reg.predict(X)

            sds = []
            dists, inds = self.err_reg.radius_neighbors(
                X, return_distance=True)
            for d, i in zip(dists, inds):
                if len(d) < 2:
                    sds.append(self.def_sd)
                    continue
                else:
                    errs = self.sse[i]
                    weights = self.comp_weights(d)
                    sd = np.average(errs, axis=0, weights=weights) / len(d)
                    sds.append(sd)

            return mean_val, np.array(sds)


class TestModalProblem(object):
    def __init__(self, x_dim, a_dim, n_modes):
        self.centers = np.random.uniform(low=-1, high=1, size=(n_modes, x_dim))
        self.offsets = np.random.uniform(low=-1, high=1, size=(n_modes, a_dim))

        self.current_context = np.random.uniform(low=-1, high=1, size=x_dim)

    @property
    def context(self):
        return self.current_context

    def step(self, a):
        x = self.current_context.reshape(1, -1)
        dists = sspatial.distance.cdist(x, self.centers)
        # probs = np.exp( -dists )
        # probs = probs / np.sum(probs)
        # ind = np.random.choice(range(len(dists)), p=probs)
        ind = np.argmin(dists)

        noise = np.random.normal(scale=0.1)
        reward = -np.log(np.linalg.norm(self.offsets[ind] - a)) + noise
        reward = max(min(reward, 5), -5)

        self.current_context = np.random.uniform(low=-1, high=1, size=x_dim)
        return reward


if __name__ == '__main__':

    x_dim = 2
    a_dim = 1
    n_modes = 3

    problem = TestModalProblem(x_dim=x_dim, a_dim=a_dim, n_modes=n_modes)
    mp = poli.ModalPolicy()

    n_init = 30
    n_opt_iters = 5
    n_trials = 1000
    opt_per = 10
    rewards = []
    states_seen = []

    for i in range(n_init):
        s = problem.context
        a = np.random.uniform(-1, 1, a_dim)
        r = problem.step(a)
        states_seen.append(s)
        rewards.append(r)

        print 's: %s a: %s r: %f' % (np.array_str(s), np.array_str(a), r)

        mp.report_sample(s=s, a=a, r=r)

    def print_cluster():
        print 'clusters:\n'
        for cc, r in zip(mp.cluster_actions, mp.cluster_rewards):
            print '\tc: %s r: %f' % (np.array_str(cc), r)

    mp.initialize_modes(n_modes * 2)
    mp.optimize(n_opt_iters, beta=1)
    print_cluster()

    # plotting
    grid = np.linspace(-1, 1, 100)
    xx, yy = np.meshgrid(grid.flatten(), grid.flatten())
    X = np.vstack((xx.flatten(), yy.flatten())).T

    plt.ion()
    plt.figure()

    def update_plot():
        zz = mp.clusters.predict(X)
        Z = np.reshape(zz, xx.shape)
        plt.gca().clear()
        plt.gca().set_xlim((-1,1))
        plt.gca().set_ylim((-1,1))
        plt.pcolormesh(xx, yy, Z, cmap=mcm.PiYG)
        #plt.contourf(xx, yy, Z, np.unique(zz), antialiased=False)
        plt.plot(problem.centers[:,0], problem.centers[:,1], 'kx', markersize=5, mew=3)
        s = np.array(states_seen)
        plt.scatter(x=s[:,0], y=s[:,1], marker='.', color='k')
        plt.draw()

    for i in range(n_trials):
        s = problem.context
        a = mp.get_action(s)
        r = problem.step(a)
        rewards.append(r)
        states_seen.append(s)

        print 's: %s a: %s r: %f' % (np.array_str(s), np.array_str(a), r)

        # if i % opt_per == 0:
        mp.report_sample(s=s, a=a, r=r)
        mp.optimize(1, beta=1)
        print_cluster()
        update_plot()


    plt.figure('Reward trace')
    plt.title('Rewards vs time')
    plt.plot(rewards, 'r-')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')