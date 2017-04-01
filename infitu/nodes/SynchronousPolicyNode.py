#!/usr/bin/env python

import numpy as np
import rospy
import poli
import optim
import broadcast
import infitu
from threading import Lock


class SynchronousPolicyNode(object):
    """Interfaces with an optimization problem with the GetCritique interface
    """

    def __init__(self):

        # Parse policy
        self.policy_lock = Lock()
        policy_info = rospy.get_param('~policy')
        stream_name = policy_info.pop('input_stream')
        output_dim = policy_info.pop('action_dim')
        self.stream_rx = broadcast.Receiver(stream_name)

        self.project_actions = policy_info.pop('project_actions')
        raw_input_dim = self.stream_rx.stream_feature_size
        input_dim = raw_input_dim + 1

        self.normalizer = None
        self.augmenter = None
        if rospy.has_param('~input_preprocessing'):

            if rospy.has_param('~input_preprocessing/augmentation'):
                aug_info = rospy.get_param('~input_preprocessing/augmentation')
                self.augmenter = poli.FeaturePolynomialAugmenter(dim=raw_input_dim,
                                                                 **aug_info)
                raw_input_dim = self.augmenter.dim
                input_dim = raw_input_dim + 1

            if rospy.has_param('~input_preprocessing/normalization'):
                norm_info = rospy.get_param(
                    '~input_preprocessing/normalization')
                self.normalizer = poli.OnlineFeatureNormalizer(dim=raw_input_dim,
                                                               **norm_info)

        # TODO Make homogeneous augmentation optional?
        policy_info['input_dim'] = input_dim
        policy_info['output_dim'] = output_dim
        self.policy = poli.parse_policy(policy_info)

        # Parse learner
        learner_info = rospy.get_param('~gradient_estimation')
        self.learner = poli.parse_learner(learner_info, self.policy)

        # Parse optimizer
        optimizer_info = rospy.get_param('~optimizer')
        optimizer_info['mode'] = 'max'
        self.optimizer = optim.parse_optimizers(optimizer_info)

        # Critic interface
        critique_topic = rospy.get_param('~critique_topic')
        self.critic_interface = optim.CritiqueInterface(topic=critique_topic)

        policy_rate = rospy.get_param('~policy_rate')

        learn_rate = rospy.get_param('~learn_rate')
        self._policy_rate = rospy.Rate(policy_rate)
        self._learn_timer = rospy.Timer(period=rospy.Duration(1.0 / learn_rate),
                                        callback=self.learn_spin)

    def __preprocess_input(self, v):
        if self.augmenter is not None:
            v = self.augmenter.process(v)
        if self.normalizer is not None:
            v = self.normalizer.process(v)
            if v is None:
                return None

        v = np.hstack((v, 1))
        return v

    @property
    def default_input(self):
        v = np.zeros(self.policy.input_dim)
        v[-1] = 1
        return v

    def action_spin(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            stamp, raw_state = self.stream_rx.read_stream(now,
                                                          mode='closest_before')

            if stamp is None or raw_state is None:
                rospy.logwarn('Could not read input stream at time %s',
                              str(now))
                rospy.sleep(rospy.Duration(1.0))
                continue

            state = self.__preprocess_input(raw_state)
            use_default = state is None
            if use_default:
                rospy.loginfo('Input normalizer not ready yet. Using default input.')
                state = self.default_input

            self.policy_lock.acquire()
            action = self.policy.sample_action(state)
            self.policy_lock.release()
            if self.project_actions:
                action[action > 1] = 1
                action[action < -1] = -1

            action_mean = self.policy.mean
            action_cov = self.policy.cov

            msg = 'Executing:\n'
            msg += '\tState: %s\n' % np.array_str(state)
            msg += '\tAction: %s\n' % np.array_str(action)
            msg += '\tMean: %s\n' % np.array_str(action_mean)
            msg += '\tVar:%s' % np.array_str(np.diag(action_cov))
            rospy.loginfo(msg)
            reward, feedback = self.critic_interface(action)
            rospy.loginfo('Received reward: %f' % reward)

            if not use_default:
                self.policy_lock.acquire()
                self.learner.report_sample(
                    state=state, action=action, reward=reward)
                self.policy_lock.release()

            self._policy_rate.sleep()

    def learn_spin(self, event):

        rospy.loginfo('Have %d SAR samples', self.learner.num_samples)

        self.policy_lock.acquire()
        theta_init = self.policy.get_theta()
        theta = self.optimizer.step(x_init=theta_init,
                                    func=self.learner.compute_objective_and_gradient)
        self.policy.set_theta(theta)
        rospy.loginfo('Parameter A:\n%s', np.array_str(self.policy.A))
        rospy.loginfo('Parameter B:\n%s', np.array_str(self.policy.B))
        self.policy_lock.release()


if __name__ == '__main__':
    rospy.init_node('synchronous_policy_node')

    node = SynchronousPolicyNode()
    try:
        node.action_spin()
    except rospy.ROSInterruptException:
        pass
