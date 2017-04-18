#!/usr/bin/env python

import cPickle as pickle
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
        stream_name = rospy.get_param('~input_stream')
        output_dim = rospy.get_param('~action_dim')
        self.stream_rx = broadcast.Receiver(stream_name)
        raw_input_dim = self.stream_rx.stream_feature_size

        # Parse policy
        self._active_lock = Lock()
        self._active_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                         output_dim=output_dim,
                                                         info=rospy.get_param('~'))
        # Create a duplicate policy to use for gradient estimation asynchronously
        # NOTE The learner wrapper should never process inputs/outputs, since it would
        # diverge from the active wrapper! We use only its policy member.
        self._learn_lock = Lock()
        self._learner_wrapper = poli.parse_policy_wrapper(raw_input_dim=raw_input_dim,
                                                          output_dim=output_dim,
                                                          info=rospy.get_param('~'))
        #self.active_policy.B[:, -1] = -1
        #self.learner_policy.B[:, -1] = -1
        self.active_policy.B[:] = -1
        self.learner_policy.B[:] = -1

        # Parse learner
        learner_info = rospy.get_param('~learning/gradient_estimation')
        self.learner = poli.parse_gradient_estimator(learner_info,
                                                     policy=self.learner_policy)

        # Parse optimizer
        optimizer_info = rospy.get_param('~learning/optimizer')
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

        self._max_eps = rospy.get_param('~max_episodes')
        self._eps_counter = 0

        self.sar_tuples = []
        self.feedbacks = {}
        out_file = rospy.get_param('~out_file')
        self._out_file = open(out_file, 'w')

    @property
    def active_policy(self):
        return self._active_wrapper.policy

    @property
    def learner_policy(self):
        return self._learner_wrapper.policy

    @property
    def default_input(self):
        v = np.zeros(self.active_policy.input_dim)
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

            self._active_lock.acquire()
            state = self._active_wrapper.process_input(raw_state)
            use_default = state is None
            if use_default:
                rospy.loginfo(
                    'Input normalizer not ready yet. Using default input.')
                state = self.default_input

            action = self._active_wrapper.sample_action(state=state,
                                                        proc_input=False)
            action_logprob = self.active_policy.logprob(state, action)
            action_mean = self.active_policy.mean
            action_cov = self.active_policy.cov
            self._active_lock.release()

            msg = 'Executing:\n'
            msg += '\tState: %s\n' % np.array_str(state)
            msg += '\tAction: %s\n' % np.array_str(action)
            msg += '\tMean: %s\n' % np.array_str(action_mean)
            msg += '\tVar:%s' % np.array_str(np.diag(action_cov))
            rospy.loginfo(msg)
            reward, feedback = self.critic_interface(action)
            rospy.loginfo('Received reward: %f' % reward)

            if not use_default:
                self.sar_tuples.append((state, action, reward))
                for k, v in feedback.iteritems():
                    if k not in self.feedbacks:
                        self.feedbacks[k] = []
                    self.feedbacks[k].append(v)
                self._eps_counter += 1

                # TODO Generalize between sequential, bandit problems?
                self._learn_lock.acquire()
                self.learner.report_episode(states=[state],
                                            actions=[action],
                                            rewards=[reward],
                                            logprobs=[action_logprob])
                self._learn_lock.release()
            self._policy_rate.sleep()

    def learn_spin(self, event):

        rospy.loginfo('Have %d SAR samples', self.learner.num_samples)

        self._learn_lock.acquire()
        init_theta = self.learner_policy.get_theta()

        theta, obj = self.optimizer.step(x_init=init_theta,
                                         func=self.learner.estimate_reward_and_gradient)
        self._learn_lock.release()

        if np.any(init_theta != theta):
            self._active_lock.acquire()
            self.active_policy.set_theta(theta)
            self._active_lock.release()

            self._learn_lock.acquire()
            self.learner_policy.set_theta(theta)
            self.learner.update_buffer()
            self.learner.remove_unlikely_trajectories()
            self._learn_lock.release()

            rospy.loginfo('Predicted reward: %f' % obj)
            rospy.loginfo('Parameter A:\n%s',
                          np.array_str(self.learner_policy.A))
            rospy.loginfo('Parameter B:\n%s',
                          np.array_str(self.learner_policy.B))

        if self._eps_counter >= self._max_eps:
            AB = (self.learner_policy.A, self.learner_policy.B)
            pickle.dump((AB, self.sar_tuples, self.feedbacks), self._out_file)
            rospy.loginfo('Max episodes achieved. Shutting down...')
            rospy.signal_shutdown('Max episodes achieved.')


if __name__ == '__main__':
    rospy.init_node('synchronous_policy_node')

    node = SynchronousPolicyNode()
    try:
        node.action_spin()
    except rospy.ROSInterruptException:
        pass
