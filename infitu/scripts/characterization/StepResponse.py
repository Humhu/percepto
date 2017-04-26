#!/usr/bin/env python

import cPickle as pickle
from threading import Lock
import sys
import numpy as np
import rospy

from percepto_msgs.srv import SetParameters, SetParametersRequest
from percepto_msgs.msg import RewardStamped

from argus_utils import wait_for_service


class StepResponse:
    """Executes random parameter steps and records the reward trace afterward.

    Currently does not support synchronized operation. Outputs a pickle of a 
    list of [theta_pre, theta_post, reward_list]

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

        # Recording state
        self.mutex = Lock()
        self.traj_start_time = None
        self.current_action = None
        self.current_rewards = []

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

    def sample_action(self, action=None, ind=None):
        """Picks a new action and sets it. Not synchronized, so lock externally.
        """
        if action is None:
            if ind is None:
                action = np.random.uniform(low=self.action_lower,
                                           high=self.action_upper,
                                           size=self.action_dim)
            else:
                action = np.zeros(self.action_dim)
                action[ind] = np.random.uniform(low=self.action_lower,
                                                high=self.action_upper,
                                                size=1)
        self.current_action = action

        rospy.loginfo('Current action: %s', np.array_str(self.current_action))

        req = SetParametersRequest()
        req.parameters = self.current_action
        try:
            self.set_service.call(req)
        except rospy.ServiceException:
            rospy.logerr('Could not set parameters to %s',
                         str(req.parameters))
            return None
        return self.current_action

    def reward_callback(self, msg):
        with self.mutex:
            self.current_rewards.append((msg.header.stamp.to_sec(),
                                         msg.reward))

    def run_trial(self, ind=None):
        """Clears the reward buffer, steps the parameters, and waits
        for step_time. Returns a tuple of
        """

        with self.mutex:
            pre_action = self.current_action
            step_action = self.sample_action(ind=ind)
            step_time = rospy.Time.now().to_sec()
            if step_action is None:
                return None
            self.current_rewards = []

        rospy.sleep(self.step_time)

        with self.mutex:
            rewards = [r for r in self.current_rewards
                       if r[0] > step_time]

        return (pre_action, step_action, rewards)

    def execute(self):
        # Initialize
        with self.mutex:
            init_action = np.zeros(self.action_dim)
            self.sample_action(action=init_action)
        rospy.sleep(self.step_time)

        if self.mode == 'per_dimension':
            trials = self.execute_per_dim()

        elif self.mode == 'joint':
            trials = self.execute_joint()

        # Clean up
        rospy.loginfo('Trials complete! Saving data...')
        pickle.dump(trials, self.output_data)
        self.output_data.close()

    def execute_per_dim(self):
        trials = []
        for i in range(self.action_dim):
            dim_trials = []
            while len(dim_trials) < self.num_samples and not rospy.is_shutdown():
                rospy.loginfo('Running dimension %d/%d trial %d/%d',
                              len(dim_trials) + 1,
                              self.num_samples,
                              i + 1,
                              self.action_dim)
                trial = self.run_trial(ind=i)
                if trial is None:
                    rospy.sleep(self.step_time)
                else:
                    dim_trials.append(trial)
            trials.append(dim_trials)
        return trials

    def execute_joint(self):
        trials = []
        while len(trials) < self.num_samples and not rospy.is_shutdown():
            rospy.loginfo('Running trial %d/%d...',
                          len(trials) + 1,
                          self.num_samples)
            trial = self.run_trial()
            if trial is None:
                rospy.sleep(self.step_time)
            else:
                trials.append(trial)
        return trials


if __name__ == '__main__':
    rospy.init_node('step_response_characterizer')
    cepo = StepResponse()
    try:
        cepo.execute()
    except rospy.ROSInterruptException:
        pass
    sys.exit(0)
