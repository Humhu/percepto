#!/usr/bin/env python

import rospy
import numpy as np
import infitu


class TestInterfaceSetter(object):
    def __init__(self):
        self._interface = infitu.NumericParameterInterface()
        self._interface.add_parameter(name='param1',
                                      base_topic='/test_getter',
                                      limits=(-5, 5))
        self._interface.add_parameter(name='param0',
                                      base_topic='/test_getter',
                                      limits=(-0.5, 0.1))

        self._timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

    def timer_callback(self, event):
        v = np.random.uniform(low=-1, high=1, size=2)
        rospy.loginfo('Setting: %s', np.array_str(v))
        self._interface.set_values(v)

if __name__ == '__main__':
    rospy.init_node('test_interface_setter')
    tis = TestInterfaceSetter()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass