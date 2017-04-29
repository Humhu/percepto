#!/usr/bin/env python

import rospy
import infitu

from percepto_msgs.srv import SetParameters
from percepto_msgs.srv import GetParameters, GetParametersResponse


class NumericParamInterface(object):
    def __init__(self):
        interface_info = rospy.get_param('~')
        self.interface = infitu.parse_interface(interface_info)

        self.set_norm_param = rospy.Service('~set_normalized_parameters',
                                            SetParameters,
                                            lambda req: self.set_param_callback(req, True))

        self.set_raw_param = rospy.Service('~set_raw_parameters',
                                           SetParameters,
                                           lambda req: self.set_param_callback(req, False))

        self.get_param_nor = rospy.Service('~get_normalized_parameters',
                                           GetParameters,
                                           lambda req: self.get_param_callback(req, True))
        self.get_param_raw = rospy.Service('~get_raw_parameters',
                                           GetParameters,
                                           lambda req: self.get_param_callback(req, False))

    def set_param_callback(self, req, normalized):
        try:
            if len(req.names) == 0:
                names = None
            else:
                names = req.names

            self.interface.set_values(v=req.parameters,
                                      names=names,
                                      normalized=normalized)
            return []
        except ValueError as e:
            rospy.logerr('Could not set params: %s', str(e))
            return None

    def get_param_callback(self, req, normalized):
        req = GetParametersResponse()
        req.parameters = self.interface.get_values(normalized=normalized)
        req.names = self.interface.parameter_names
        return req


if __name__ == '__main__':
    rospy.init_node('continuous_parameter_interface')
    cpi = NumericParamInterface()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
