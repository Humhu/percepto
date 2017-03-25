#!/usr/bin/env python

import rospy
import numpy as np
import infitu
import paraset


class TestInterfaceGetter(object):
    def __init__(self):
        self._param0 = paraset.RuntimeParamGetter(param_type=float,
                                                  name='param0',
                                                  init_val=0,
                                                  description='Test parameter 0')
        self._param1 = paraset.RuntimeParamGetter(param_type=float,
                                                  name='param1',
                                                  init_val=0,
                                                  description='Test parameter 1')

        self._timer = rospy.Timer(rospy.Duration(2.0), self.timer_callback)

    def timer_callback(self, event):
        v = [self._param0.value, self._param1.value]
        rospy.loginfo('Params: %s', str(v))


if __name__ == '__main__':
    rospy.init_node('test_interface_getter')
    tis = TestInterfaceGetter()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
