#!/usr/bin/env python

import rospy
import infitu

from percepto_msgs.srv import SetParameters
from percepto_msgs.srv import GetParameters, GetParametersResponse


class ContinuousParameterInterface(object):
    def __init__(self):
        interface_info = rospy.get_param('~interface')
        self.interface = infitu.parse_interface(interface_info)

        self.set_param = rospy.Service('~set_parameters',
                                       SetParameters,
                                       self.set_param_callback)
        self.get_param = rospy.Service('~get_parameters',
                                       GetParameters,
                                       self.get_param_callback)

    def set_param_callback(self, req):
        try:
            self.interface.set_values(req.parameters)
            return []
        except ValueError as e:
            rospy.logerr('Could not set params: %s', str(e))
            return None

    def get_param_callback(self, req):
        req = GetParametersResponse()
        req.parameters = self.interface.get_values()
        return req


if __name__ == '__main__':
    rospy.init_node('continuous_parameter_interface')
    cpi = ContinuousParameterInterface()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
