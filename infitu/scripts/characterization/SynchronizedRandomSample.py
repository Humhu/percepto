#!/usr/bin/env python

import cPickle as pickle
from threading import Lock
import sys
import numpy as np
import rospy
import optim
import broadcast
import time

from argus_utils import wait_for_service


class SynchronizedRandomSample:
    """Uniformly randomly samples and evaluates.
    """

    def __init__(self):

        output_path = rospy.get_param('~output_path')
        self.output_data = open(output_path, 'wb')
        if self.output_data is None:
            raise RuntimeError(
                'Could not open output data at path: ' + output_path)

        # Step parameters
        self.action_lower = rospy.get_param('~action_lower_bound')
        self.action_upper = rospy.get_param('~action_upper_bound')
        self.action_dim = rospy.get_param('~action_dim')

        self.num_samples = rospy.get_param('~num_samples')

        stream_name = rospy.get_param('~input_stream', None)
        self.stream_rx = None
        if stream_name is not None:
            self.stream_rx = broadcast.Receiver(stream_name)

        interface_info = rospy.get_param('~interface')
        self.critic_interface = optim.CritiqueInterface(**interface_info)

    def sample_action(self):
        """Picks a new action and sets it. Not synchronized, so lock externally.
        """
        if self.stream_rx is not None:
            state = self.stream_rx.read_stream(rospy.Time.now(),
                                               mode='closest_before')[0]
            if state is None:
                return None
        else:
            state = None

        action = np.random.uniform(low=self.action_lower,
                                   high=self.action_upper,
                                   size=self.action_dim)
        reward, feedback = self.critic_interface(action)

        return state, action, reward, feedback

    def execute(self):

        states = []
        actions = []
        rewards = []
        feedbacks = {}
        while len(states) < self.num_samples and not rospy.is_shutdown():
            rospy.loginfo('Running trial %d/%d...',
                          len(actions) + 1,
                          self.num_samples)
            ret = self.sample_action()
            if ret is None:
                time.sleep(1.0)
                continue
            s, a, r, f = ret

            for k, v in f.iteritems():
                if k not in feedbacks:
                    feedbacks[k] = []
                feedbacks[k].append(v)

            states.append(s)
            actions.append(a)
            rewards.append(r)

        # Clean up
        rospy.loginfo('Trials complete! Saving data...')
        pickle.dump((states, actions, rewards, feedbacks), self.output_data)
        self.output_data.close()


if __name__ == '__main__':
    rospy.init_node('step_response_characterizer')
    cepo = SynchronizedRandomSample()
    cepo.execute()
    sys.exit(0)
