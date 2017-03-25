"""Contains classes that provide wrappers around sets of runtime parameters.
"""

import rospy
import numpy as np
import paraset
from itertools import izip


def parse_interface(info):
    """Parses a dictionary to produce a NumericParameterInterface.
    """
    interface = NumericParameterInterface()
    for name, v in info.iteritems():
        base_topic = v['base_namespace']
        limits = (v['lower_limit'], v['upper_limit'])
        interface.add_parameter(name=name,
                                base_topic=base_topic,
                                limits=limits)
    return interface

class NumericParameterInterface(object):
    """Wraps and normalizes a set of numeric runtime parameters,
    allowing them to be set with a vector input.

    Parameters are sorted alphabetically by name and scaled to correspond
    to an input range of -1 to 1.
    """

    def __init__(self):
        self._setters = []

    def add_parameter(self, name, base_topic, limits):
        """Adds another parameter to this interface.

        The internal list of parameters will be sorted after adding.
        """
        setter = paraset.RuntimeParamSetter(param_type=float,
                                            name=name,
                                            base_topic=base_topic)
        scale = (limits[1] - limits[0]) / 2.0
        offset = (limits[1] + limits[0]) / 2.0
        if scale < 0:
            raise ValueError('Limits must be [low, high]')

        def unscale_func(x): return x * scale + offset

        self._setters.append((name, setter, unscale_func))
        self._setters.sort(key=lambda x: x[0])

    def set_values(self, v):
        if len(v) != len(self._setters):
            raise ValueError('Expected %d elements, got %d' %
                             (len(self._setters), len(v)))

        out = 'Setting values:'
        for vi, setinfo in izip(v, self._setters):
            name, setter, scale_func = setinfo
            vscaled = scale_func(vi)

            out += '\n\t%s: %f (%f)' % (name, vi, vscaled)
            setter.set_value(vscaled)
        rospy.loginfo(out)

    @property
    def num_parameters(self):
        """The number of parameters wrapped in this interface.
        """
        return len(self._setters)
