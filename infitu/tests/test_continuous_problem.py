#!/usr/bin/env python

import math
import numpy as np
import rospy
from percepto_msgs.msg import RewardStamped, EpisodeBreak
from percepto_msgs.srv import GetCritique, GetCritiqueResponse
import broadcast
import paraset

from threading import Lock
import matplotlib.pyplot as plt


def box_constrain(a, lim):
    a[a > lim] = lim
    a[a < -lim] = -lim
    return a


class CartSystem(object):
    """Simulates a simple 2D first-order cart system
    """

    def __init__(self, dt, max_acc, max_steps):
        self._counter = 0
        self._dt = dt
        self._max_vel = 1.0
        self._max_acc = max_acc
        self._max_steps = max_steps
        self._state_lock = Lock()

        self._fig = plt.figure()
        self._ax = plt.axes()

        self._x = None
        self._xdot = None

        self.reset()

    def reset(self):
        self._counter = 0
        self._x = np.zeros(2)  # np.random.uniform(-1, 1, size=2)
        self._xdot = np.random.uniform(-0.5, 0.5, size=2)
        return self.__reward()

    def step(self, acc):
        acc = box_constrain(acc, self._max_acc)

        self._state_lock.acquire()
        self._x += self._dt * self._xdot + 0.5 * self._dt ** 2 * acc
        self._x = box_constrain(self._x, 1)

        self._xdot += self._dt * acc
        self._xdot = box_constrain(self._xdot, 1)

        self._state_lock.release()
        self._counter += 1

        return self.__reward()

    def __reward(self):
        err = np.dot(self._x, self._x)
        return math.exp(-err)

    def done(self):
        max_time = self._counter > self._max_steps
        hit_lims = np.any(np.abs(self._x) == 1.0)
        return max_time or hit_lims

    def update_plot(self):
        self._state_lock.acquire()
        x = self._x
        xdot = self._xdot
        self._state_lock.release()

        self._ax.clear()
        self._ax.set_xlim([-1, 1])
        self._ax.set_ylim([-1, 1])
        plt.plot(x[0], x[1], axes=self._ax,
                 marker='o', markersize=10)
        plt.arrow(x[0], x[1], xdot[0], xdot[1])
        plt.title('Step %d/%d' % (self._counter, self._max_steps))
        plt.draw()

    @property
    def pos_dim(self):
        return 2

    @property
    def state_dim(self):
        return 4

    @property
    def state(self):
        return np.hstack((self._x, self._xdot))

    @property
    def position(self):
        return self._x

    @property
    def velocity(self):
        return self._xdot


class TestContinuousProblem(object):
    """Provides a test continuous problem with RuntimeParameters.
    """

    def __init__(self):

        dt = rospy.get_param('~cart_dt')
        max_acc = rospy.get_param('~cart_max_acc')
        max_steps = rospy.get_param('~cart_max_steps')
        self.cart = CartSystem(dt=dt, max_acc=max_acc, max_steps=max_steps)

        # Initialize cart parameters
        self.parameters = []
        for i in range(self.cart.pos_dim):
            param = paraset.RuntimeParamGetter(param_type=float, name='acc_%d' % i,
                                               init_val=0, description='cart acceleration')
            self.parameters.append(param)

        update_rate = rospy.get_param('~tick_rate')
        self.timer = rospy.Timer(period=rospy.Duration(1.0 / update_rate),
                                 callback=self.timer_callback)

        self.last_acc = np.zeros(2)
        self.rew_pub = rospy.Publisher('~reward', RewardStamped, queue_size=10)
        self.break_pub = rospy.Publisher(
            '~breaks', EpisodeBreak, queue_size=10)

        # Set up state generator and transmitter
        self.state_tx = broadcast.Transmitter(stream_name='cart_state',
                                              feature_size=self.cart.state_dim,
                                              description='Test continuous problem state',
                                              mode='push',
                                              queue_size=10)

        reward = self.cart.reset()
        self.__publish_cart(time=rospy.Time.now(), reward=reward)

    def timer_callback(self, event):
        if self.cart.done():
            msg = EpisodeBreak()
            msg.break_time = event.current_real
            self.break_pub.publish(msg)
            reward = self.cart.reset()
        else:
            reward = self.cart.step(self.__get_acc())

        self.cart.update_plot()
        self.__publish_cart(time=event.current_real, reward=reward)

    def __get_acc(self):
        return np.array([p.value for p in self.parameters])

    def __publish_cart(self, time, reward):
        msg = RewardStamped()
        msg.reward = reward
        msg.header.stamp = time
        self.rew_pub.publish(msg)

        self.state_tx.publish(time=time, feats=self.cart.state)


if __name__ == '__main__':
    rospy.init_node('test_continuous_problem')
    prob = TestContinuousProblem()
    try:
        plt.show(block=True)
    except rospy.ROSInterruptException:
        pass
