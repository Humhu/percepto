#!/usr/bin/env python

import cPickle as pickle
from threading import Lock
import sys
import numpy as np
import rospy
import broadcast
import optim

from sklearn.neighbors import KernelDensity

from percepto_msgs.srv import SetParameters, SetParametersRequest
from percepto_msgs.msg import RewardStamped

from argus_utils import wait_for_service


class StepResponse:
    """Executes random parameter steps and records the reward trace afterward.

    Currently does not support synchronized operation.

    Outputs a pickle of a list of [theta_pre, theta_post, reward_list]

    Parameters
    ----------
    TODO!

    Topics
    ------
    TODO!
    """

    def __init__(self):

        output_path = rospy.get_param('~output_path')
        self.output_data = open(output_path, 'wb')
        if self.output_data is None:
            raise RuntimeError('Could not open: %s' % output_path)

        # Step parameters
        self.action_lower = rospy.get_param('~action_lower_bound')
        self.action_upper = rospy.get_param('~action_upper_bound')
        self.action_dim = int(rospy.get_param('~action_dim'))

        self.num_samples = int(rospy.get_param('~num_samples'))
        self.step_time = float(rospy.get_param('~step_time'))

        seed = rospy.get_param('~random_seed', None)
        if seed is not None:
            np.random.seed(int(seed))
            rospy.loginfo('Seeding RNG with: %d', seed)

        self.mode = rospy.get_param('~mode')
        if self.mode not in ['per_dimension', 'joint']:
            raise ValueError('Invalid operating mode')

        self.enable_adaptive_samp = rospy.get_param(
            '~enable_adaptive_sampling', True)
        if self.enable_adaptive_samp:
            self.adapt_init_samples = int(
                rospy.get_param('~adaptive_init_samples'))
            opt_info = rospy.get_param('~optimizer')
            self.optimizer = optim.parse_optimizers(opt_info)
            self.optimizer.lower_bounds = self.action_lower
            self.optimizer.upper_bounds = self.action_upper

            kde_info = rospy.get_param('~density_estimator', {})
            self.kde = KernelDensity(**kde_info)

        # Find optional input stream
        stream_name = rospy.get_param('~input_stream', None)
        self.stream_rx = None
        if stream_name is not None:
            self.stream_rx = broadcast.Receiver(stream_name)

        # Recording state
        self.mutex = Lock()
        self.current_rewards = []

        self.contexts = []
        self.pre_actions = []
        self.actions = []
        self.reward_traces = []

        # Create parameter setting service proxy
        set_topic = rospy.get_param('~set_parameters_service')
        wait_for_service(set_topic)
        self.set_service = rospy.ServiceProxy(set_topic,
                                              SetParameters,
                                              persistent=True)

       # Subscribe to reward topic
        self.reward_sub = rospy.Subscriber('reward',
                                           RewardStamped,
                                           self.reward_callback)

    @property
    def trial_num(self):
        return len(self.reward_traces)

    def fit_kde(self):
        if self.trial_num == 0:
            return

        x = np.asarray(self.contexts)
        a = np.asarray(self.actions)
        X = np.hstack((x, a))
        self.kde.fit(X)

    def get_context(self):
        if self.stream_rx is None:
            return None

        while not rospy.is_shutdown():
            context = self.stream_rx.read_stream(rospy.Time.now(),
                                                 mode='closest_before')[1]
            if context is None:
                rospy.logerr('Could not read context')
                rospy.sleep(1.0)
            else:
                return context

    def sample_action(self, context=None, ind=None):
        """Picks a new action and sets it. Not synchronized, so lock externally.
        """
        # if not dimensional
        if self.enable_adaptive_samp:
            action = self.sample_adaptive(ind=ind, context=context)
        else:
            action = self.sample_uniform(ind=ind)

        return action

    def sample_adaptive(self, context, ind=None):
        # Can't adaptive sample until we have at least one sample for
        # initialization
        if context is None:
            raise RuntimeError('Received None context for adaptive sample!')

        if self.trial_num < self.adapt_init_samples:
            rospy.loginfo('Samples %d less than init %d',
                          self.trial_num, self.adapt_init_samples)
            return self.sample_uniform(ind=ind)

        a_init = np.zeros(self.action_dim)
        X = np.hstack((context, a_init)).reshape(1, -1)
        x_dim = len(context)
        if ind is None:
            def obj(a):
                X[0][x_dim:] = a
                return self.kde.score_samples(X)
        else:
            def obj(a):
                X[0][x_dim + ind] = a
                return self.kde.score_samples(X)
        action, obj = self.optimizer.optimize(x_init=np.zeros(self.action_dim),
                                         func=obj)
        print 'Obj: %f' % obj
        return action

    def sample_uniform(self, ind=None):
        if ind is None:
            action = np.random.uniform(low=self.action_lower,
                                       high=self.action_upper,
                                       size=self.action_dim)
        else:
            action = np.zeros(self.action_dim)
            action[ind] = np.random.uniform(low=self.action_lower,
                                            high=self.action_upper,
                                            size=1)
        return action

    def set_action(self, action):

        req = SetParametersRequest()
        req.parameters = action
        try:
            self.set_service.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to %s',
                         str(req.parameters))
            return None
        return action

    def reward_callback(self, msg):
        with self.mutex:
            self.current_rewards.append((msg.header.stamp.to_sec(),
                                         msg.reward))

    def run_trial(self, context=None, ind=None):
        """Clears the reward buffer, steps the parameters, and waits
        for step_time.
        """
        with self.mutex:
            step_action = self.sample_action(context=context, ind=ind)
            self.current_rewards = []

        rospy.loginfo('Running trial with\n\tcontext: %s\n\taction: %s',
                      str(context), np.array_str(step_action))
        step_start_time = rospy.Time.now().to_sec()
        self.set_action(step_action)
        rospy.sleep(self.step_time)

        with self.mutex:
            reward_trace = [r for r in self.current_rewards
                            if r[0] > step_start_time]
        if len(reward_trace) == 0:
            rospy.logwarn('Empty reward trace!')

        rewards = zip(*reward_trace)[1]
        rospy.loginfo('Received reward mean: %f SD %f',
                      np.mean(rewards), np.std(rewards))

        return step_action, reward_trace

    def execute(self):
        if self.mode == 'per_dimension':
            # trials = self.execute_per_dim()
            raise RuntimeError('per dimension disabled')

        elif self.mode == 'joint':
            self.execute_joint()

        # Clean up
        rospy.loginfo('Trials complete! Saving data...')
        out = {'pre_actions': self.pre_actions,
               'actions': self.actions,
               'reward_traces': self.reward_traces}
        if self.stream_rx is not None:
            out['contexts'] = self.contexts

        pickle.dump(out, self.output_data)
        self.output_data.close()

    # def execute_per_dim(self):
    #     # Initialize
    #     with self.mutex:
    #         init_action = np.zeros(self.action_dim)
    #         self.set_action(init_action)
    #         pre_action = init_action

    #     rospy.sleep(self.step_time)

    #     for i in range(self.action_dim):
    #         dim_trials = []
    #         while len(dim_trials) < self.num_samples and not rospy.is_shutdown():
    #             rospy.loginfo('Running dimension %d/%d trial %d/%d',
    #                           len(dim_trials) + 1,
    #                           self.num_samples,
    #                           i + 1,
    #                           self.action_dim)
    #             trial = self.run_trial(ind=i)
    #             if trial is None:
    #                 rospy.sleep(self.step_time)
    #             else:
    #                 dim_trials.append(trial)
    #         trials.append(dim_trials)
    #     return trials

    def execute_joint(self):
        # Initialize
        with self.mutex:
            init_action = np.zeros(self.action_dim)
            self.set_action(init_action)
            pre_action = init_action

        while self.trial_num < self.num_samples and not rospy.is_shutdown():
            rospy.loginfo('Running trial %d/%d...',
                          self.trial_num + 1,
                          self.num_samples)

            context = self.get_context()
            if self.enable_adaptive_samp:
                self.fit_kde()

            action, rewards = self.run_trial(context=context)

            self.pre_actions.append(pre_action)
            self.actions.append(action)
            self.reward_traces.append(rewards)
            self.contexts.append(context)

            pre_action = action


if __name__ == '__main__':
    rospy.init_node('step_response_characterizer')
    cepo = StepResponse()
    try:
        cepo.execute()
    except rospy.ROSInterruptException:
        pass
    sys.exit(0)
