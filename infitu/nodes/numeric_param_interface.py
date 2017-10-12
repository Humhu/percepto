#!/usr/bin/env python

import rospy
import infitu
from threading import Lock

from percepto_msgs.srv import SetParameters
from percepto_msgs.srv import GetParameters, GetParametersResponse
from broadcast import Transmitter


class NumericParamInterface(object):
    def __init__(self):
        self.mutex = Lock()

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

        raw_stream_name = rospy.get_param('~raw_stream_name',
                                          'parameter_configuration_raw')
        self.raw_tx = Transmitter(stream_name=raw_stream_name,
                                  feature_size=self.interface.num_parameters,
                                  description='Raw parameter values: %s' % str(
                                      self.interface.parameter_names),
                                  mode='push')
        norm_stream_name = rospy.get_param('~normalized_stream_name',
                                           'parameter_configuration_normalized')
        self.norm_tx = Transmitter(stream_name=norm_stream_name,
                                   feature_size=self.interface.num_parameters,
                                   description='Normalized parameter values: %s' % str(
                                       self.interface.parameter_names),
                                   mode='push')

    def set_param_callback(self, req, normalized):
        with self.mutex:
            try:
                now = rospy.Time.now()
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

        raw_vals, norm_vals = self.interface.map_values(v=req.parameters,
                                                        names=names,
                                                        normalized=normalized)
        self.raw_tx.publish(time=now, feats=raw_vals)
        self.norm_tx.publish(time=now, feats=norm_vals)

    def get_param_callback(self, req, normalized):
        with self.mutex:
            req = GetParametersResponse()
            req.parameters = self.interface.get_values(normalized=normalized)
            req.names = self.interface.parameter_names
            return req


if __name__ == '__main__':
    rospy.init_node('continuous_parameter_interface')
    npi = NumericParamInterface()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
