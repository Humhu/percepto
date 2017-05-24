"""This policy contains classes for modal policies.
"""

import numpy as np
from sklearn.cluster import KMeans

#from sklearn.svm import SVC, SVR
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity

from poli import importance_sample

from optim import BFGSOptimizer
from optim import GaussianProcessRewardModel as GPRewardModel
from itertools import izip, product


class ModalPolicy(object):
    """Clusters the input space and returns local policies.
    """

    def __init__(self, optimizer=None, reward_model=None, mode_classifier=KNeighborsClassifier,
                 mode_args=None):
        if reward_model is None:
            self.reward_model = GPRewardModel()
        else:
            self.reward_model = reward_model
        self.reward_model_fitted = False

        self.mode_classifier = mode_classifier
        if mode_args is None:
            self.mode_args = {'weights': 'distance'}

        self.states = []
        self.actions = []
        self.rewards = []

        self.clusters = None
        self.clusters_init = False
        self.cluster_actions = []
        self.cluster_rewards = []
        self.active_clusters = []
        self.n_modes = 0

        self.sa_kde = KernelDensity()  # TODO

        if optimizer is None:
            self.optimizer = BFGSOptimizer(mode='max', num_restarts=3)
            self.optimizer.lower_bounds = -1
            self.optimizer.upper_bounds = 1  # TODO
        else:
            self.optimizer = optimizer

    def report_sample(self, s, a, r):
        x = np.hstack((s, a))
        # try:
        self.reward_model.report_sample(x, r)
        self.reward_model_fitted = True
        # except AttributeError:
            # self.reward_model_fitted = False

        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def get_action(self, s):
        s = np.asarray(s)
        if len(s.shape) < 2:
            s = s.reshape(1, -1)

        # TODO Support multiple queries?
        probs = self.clusters.predict_proba(s)
        ind = np.random.choice(self.active_clusters,
                               size=1,
                               p=np.atleast_1d(np.squeeze(probs)))
        a = [self.cluster_actions[i] for i in ind]

        return np.squeeze(a)

    def expected_reward(self, normalize=False):
        self.fit_reward_model()

        X = np.hstack((self.states, self.actions))
        r_pred, r_std = self.reward_model.predict(X, return_std=True)

        if normalize:
            logq = self.sa_kde.score_samples(X)
            logp = np.mean(logq)
            return importance_sample(x=r_pred, p_gen=logq, p_tar=logp,
                                     normalize=True, log_weight_lim=3)
        else:
            return np.mean(r_pred)

    def initialize_modes(self, n_modes, init_clusterer=None):
        if init_clusterer is None:
            self.clusters = KMeans(n_clusters=n_modes)
        else:
            self.clusters = init_clusterer

        self.n_modes = n_modes
        self.clusters.fit(X=self.states)
        self.cluster_actions = [None] * n_modes
        self.cluster_rewards = [None] * n_modes
        self.active_clusters = range(self.n_modes)
        self.optimize_mode_actions()

    def fit_reward_model(self):
        if self.reward_model_fitted:
            return

        # X = np.hstack((self.states, self.actions))
        # r = np.asarray(self.rewards).reshape(-1, 1)
        # self.reward_model.fit(X, r)
        # self.reward_model_fitted = True

    def optimize(self, n_iters, beta=1.0):
        for i in range(n_iters):
            self.optimize_mode_assignments()
            self.optimize_mode_actions(beta=beta)

    def optimize_mode_actions(self, beta=1.0):
        """Pick the best actions for the current mode assignments.
        """
        self.fit_reward_model()
        assignments = self.clusters.predict(X=self.states)
        present_modes = np.unique(assignments)
        probabilities = np.zeros((len(self.states), self.n_modes))

        try:
            probs = self.clusters.predict_proba(X=self.states)
            for i, k in zip(range(len(present_modes)), present_modes):
                probabilities[:, k] = probs[:, i]
        except AttributeError:
            for k in range(self.n_modes):
                probabilities[:, k] = assignments == k

        a_dim = len(self.actions[0])  # TODO HACK!
        states = np.asarray(self.states)

        for k in range(self.n_modes):

            print 'Optimizing action for mode %d...' % k

            # Start with the current mode action if exists, else init to zeros
            init_a = self.cluster_actions[k]
            if init_a is None:
                init_a = np.zeros(a_dim)

            potential_in = probabilities[:, k] > 0
            members = states[potential_in]
            member_probs = np.squeeze(probabilities[potential_in, k])

            if len(members) == 0:
                print 'Cluster %d empty!' % k
                self.cluster_actions[k] = np.random.uniform(-1, 1, a_dim)
                continue

            def obj(a):
                # TODO Settable criterion
                a = a.reshape(1, -1)
                A = np.tile(a, reps=(len(members), 1))
                X = np.hstack((members, A))
                r_pred, r_std = self.reward_model.predict(X, return_std=True)
                ucb = r_pred + beta * r_std
                return np.average(ucb, weights=member_probs)

            best_a, r_pred = self.optimizer.optimize(x_init=init_a, func=obj)
            self.cluster_actions[k] = best_a
            self.cluster_rewards[k] = r_pred

    def optimize_mode_assignments(self):
        """Pick the best assignments for states given the current actions
        for each mode.
        """

        # Predict rewards for all combinations of state, mode action
        N = len(self.states)
        s_dim = len(self.states[0])
        a_dim = len(self.actions[0])
        state_actions = np.empty((N * self.n_modes, s_dim + a_dim))
        state_actions[:, :s_dim] = np.repeat(self.states, self.n_modes, axis=0)
        state_actions[:, s_dim:] = np.tile(self.cluster_actions, reps=(N, 1))

        r_pred, r_std = self.reward_model.predict(state_actions,
                                                  return_std=True)
        # self.fit_reward_model()
        #r_pred = self.reward_model.predict(state_actions)

        r_pred = np.reshape(r_pred, (N, self.n_modes))

        new_assignments = np.argmax(r_pred, axis=1)

        self.active_clusters = np.unique(new_assignments)

        # NOTE Do we actually want to do this? 1 mode seems fine if state is irrelevant!
        # if len(self.active_clusters) == 1:
        # print 'Mode collapse! Re-initializing...'
        # self.initialize_modes(self.n_modes)
        # return

        # to_assign = []
        # for i in range(self.n_modes):
        #     if not np.any(new_assignments == i):
        #         print 'Randomly reassigning mode %d' % i
        #         to_assign.append(i)

        # ind = np.random.randint(0, len(new_assignments), size=len(to_assign))
        # for i, j in zip(ind, to_assign):
        #     new_assignments[i] = j

        # TODO Something with these new assignments?
        #self.clusters = SVC(decision_function_shape='ovr')
        #self.clusters.fit(X=self.states, y=new_assignments)

        if not self.clusters_init:
            self.clusters = self.mode_classifier(**self.mode_args)
            self.clusters_init = True
        self.clusters.fit(X=self.states,
                          y=new_assignments)
