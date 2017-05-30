#!/usr/bin/env python

import cPickle as pickle
from threading import Lock
import sys
import numpy as np
import rospy
import optim
import broadcast

from argus_utils import wait_for_service


class SynchronizedDimensionSample:
    """Sequentially uniformly randomly samples on individual dimensions and evaluates.
    """

    def __init__(self):

        output_path = rospy.get_param('~output_path')
        self.output_file = open(output_path, 'wb')
        if self.output_file is None:
            raise RuntimeError(
                'Could not open output data at path: ' + output_path)

        # Step parameters
        self.action_lower = rospy.get_param('~action_lower_bound')
        self.action_upper = rospy.get_param('~action_upper_bound')
        self.action_dim = rospy.get_param('~action_dim')

        self.num_samples = rospy.get_param('~num_samples_per_dim')

        stream_name = rospy.get_param('~input_stream', None)
        self.stream_rx = None
        if stream_name is not None:
            self.stream_rx = broadcast.Receiver(stream_name)

        interface_info = rospy.get_param('~interface')
        self.critic_interface = optim.CritiqueInterface(**interface_info)

    def sample_action(self, ind):
        """Picks a new action and sets it. Not synchronized, so lock externally.
        """
        if self.stream_rx is not None:
            state = self.stream_rx.read_stream(rospy.Time.now(),
                                               mode='closest_before')[0]
            if state is None:
                return None
        else:
            state = None

        action = np.zeros(self.action_dim)
        action[ind] = np.random.uniform(low=self.action_lower,
                                        high=self.action_upper,
                                        size=1)
        reward, feedback = self.critic_interface(action)

        return state, action, reward, feedback

    def execute(self):

        dim_states = []
        dim_actions = []
        dim_rewards = []
        dim_feedbacks = []
        for dim in range(self.action_dim):
            states = []
            actions = []
            rewards = []
            feedbacks = {}
            while len(states) < self.num_samples and not rospy.is_shutdown():
                rospy.loginfo('Running trial %d/%d for dimension %d/%d...',
                              len(actions) + 1,
                              self.num_samples,
                              dim + 1,
                              self.action_dim)
                ret = self.sample_action(dim)
                if ret is None:
                    rospy.sleep(1.0)
                    continue
                s, a, r, f = ret

                for k, v in f.iteritems():
                    if k not in feedbacks:
                        feedbacks[k] = []
                    feedbacks[k].append(v)

                states.append(s)
                actions.append(a)
                rewards.append(r)
            dim_states.append(states)
            dim_actions.append(actions)
            dim_rewards.append(rewards)
            dim_feedbacks.append(feedbacks)

        # Clean up
        rospy.loginfo('Trials complete! Saving data...')
        data = (dim_states, dim_actions, dim_rewards, dim_feedbacks)
        pickle.dump(data, self.output_file)
        self.output_file.close()


if __name__ == '__main__':
    rospy.init_node('synchronized_dimension_sample')
    cepo = SynchronizedDimensionSample()
    cepo.execute()
    sys.exit(0)
