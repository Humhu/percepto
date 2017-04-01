#!/usr/bin/env python

import rospy
import infitu

from percepto_msgs.srv import SetParameters


class ContinuousParameterInterface(object):
    def __init__(self):
        interface_info = rospy.get_param('~interface')
        self.interface = infitu.parse_interface(interface_info)

        self.set_param = rospy.Service('~set_parameters',
                                       SetParameters,
                                       self.param_callback)

    def param_callback(self, req):
        try:
            self.interface.set_values(req.parameters)
            return []
        except ValueError as e:
            rospy.logerr('Could not set params: %s', str(e))
            return None


if __name__ == '__main__':
    rospy.init_node('continuous_parameter_interface')
    cpi = ContinuousParameterInterface()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
