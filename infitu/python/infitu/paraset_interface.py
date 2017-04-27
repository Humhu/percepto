"""Contains classes that provide wrappers around sets of runtime parameters.
"""

import rospy
import numpy as np
import paraset
from itertools import izip


def parse_interface(info):
    """Parses a dictionary to produce a NumericParameterInterface.
    """
    if 'verbose' in info:
        verbose = bool(info.pop('verbose'))
    else:
        verbose = False

    interface = NumericParameterInterface(verbose=verbose)

    if 'param_order' in info:
        param_order = info.pop('param_order')
    else:
        param_order = info.keys()

    ordered_items = [None] * len(info)
    for name, v in info.iteritems():
        try:
            ind = param_order.index(name)
        except ValueError:
            rospy.logerr('Could not find parameter %s in ordering %s',
                         name, str(param_order))
        ordered_items[ind] = (name, v)

    for name, v in ordered_items:
        param_name = v['param_name']
        base_topic = v['base_namespace']
        limits = (v['lower_limit'], v['upper_limit'])
        interface.add_parameter(name=name,
                                param_name=param_name,
                                base_topic=base_topic,
                                limits=limits)
    return interface


class NumericParameterInterface(object):
    """Wraps and normalizes a set of numeric runtime parameters,
    allowing them to be set with a vector input.

    Parameters are sorted alphabetically by name and scaled to correspond
    to an input range of -1 to 1.
    """

    def __init__(self, verbose=False):
        self._setters = []
        self._verbose = verbose

    def add_parameter(self, name, param_name, base_topic, limits):
        """Adds another parameter to this interface.

        The internal list of parameters will be sorted after adding.
        """
        if self._verbose:
            rospy.loginfo('Adding parameter %s' % name)

        setter = paraset.RuntimeParamSetter(param_type=float,
                                            name=param_name,
                                            base_topic=base_topic)
        scale = (limits[1] - limits[0]) / 2.0
        offset = (limits[1] + limits[0]) / 2.0
        if scale < 0:
            raise ValueError('Limits must be [low, high]')

        def unscale_func(x):
            x = min(max(x, -1), 1)
            return x * scale + offset
        def scale_func(x):
            return (x - offset) / scale

        self._setters.append((name, param_name, setter, unscale_func, scale_func))
        self._setters.sort(key=lambda x: x[0])

    def set_values(self, v, names=None):
        """Sets all parameters according to vector input v.

        Optionally, assign the parameters with corresponding ordered names.

        Elements of v outside of -1, 1 will be truncated
        """
        if len(v) != len(self._setters):
            raise ValueError('Expected %d elements, got %d' %
                             (len(self._setters), len(v)))

        if names is None:
            names = self.parameter_names

        out = 'Setting values:'
        resout = 'Actual values:'
        warnouts = []
        ref_names = self.parameter_names
        for vi, v_name in izip(v, names):
            try:
                ind = ref_names.index(v_name)
            except ValueError:
                rospy.logerr('Parameter %s not in interface', v_name)

            name, _, setter, unscale_func, scale_func = self._setters[ind]
            v_raw = unscale_func(vi)

            out += '\n\t%s: %f (%f)' % (name, vi, v_raw)
            actual_raw = setter.set_value(v_raw)
            actual = scale_func(actual_raw)
            resout += '\n\t%s: %f (%f)' % (name, actual, actual_raw)

            if vi != actual:
                warnout.append( 'Set param %s to %f but got actual %f' % (name, v_raw, actual_raw) )

        if self._verbose:
            rospy.loginfo(out)
            rospy.loginfo(resout)
            for w in warnouts:
                rospy.logwarn(w)

    def get_values(self, normalized=True):
        return [s[4](s[2].get_value()) for s in self._setters]

    @property
    def num_parameters(self):
        """The number of parameters wrapped in this interface.
        """
        return len(self._setters)

    @property
    def parameter_names(self):
        """An ordered list of the parameter names
        """
        return [s[0] for s in self._setters]
