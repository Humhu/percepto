"""Contains classes and modules that interface with dynamic reconfigure
"""

import rospy
import dynamic_reconfigure.client as drc
from infitu.utils import *
from itertools import izip


def parse_reconfig_interface(info):
    if 'verbose' in info:
        verbose = bool(info.pop('verbose'))
    else:
        verbose = False
    node_name = info.pop('node_name')

    interface = NumericDynaReconfigInterface(node_name=node_name,
                                             verbose=verbose)
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
        limits = (v['lower_limit'], v['upper_limit'])
        interface.add_parameter(name=name,
                                param_name=param_name,
                                limits=limits)
    return interface


class NumericDynaReconfigInterface(object):
    """Provides a wrapper around a set of numeric dynamic reconfiguration
    parameters.
    """

    def __init__(self, node_name, verbose=False):
        self.client = drc.Client(node_name)
        self.names = []
        self.param_names = []
        self.normalizers = []
        self._verbose = verbose

    def add_parameter(self, name, param_name, limits):
        if name in self.normalizers:
            raise ValueError('Param %s already registered!' % name)

        raw_descriptions = self.client.get_parameter_descriptions()
        desc_names = [r['name'] for r in raw_descriptions]
        try:
            ind = desc_names.index(param_name)
        except ValueError:
            raise ValueError('Param %s not a valid parameter' % name)
        desc = raw_descriptions[ind]
        if desc['type'] not in ['int', 'double']:
            raise ValueError('Parameter %s has non-numeric type %s' %
                             (desc['name'], desc['type']))

        if limits[0] < desc['min']:
            raise ValueError('Requested minimum %f for %s less than reconfig min %f',
                             limits[0], name, desc['min'])
        if limits[1] > desc['max']:
            raise ValueError('Requested maximum %f for %s less than reconfig max %f',
                             limits[1], name, desc['max'])

        n = ParameterNormalizer(min_val=limits[0],
                                max_val=limits[1],
                                enable_rounding=(desc['type'] == 'int'))
        self.names.append(name)
        self.param_names.append(param_name)
        self.normalizers.append(n)

    def process_element(self, v, name, normalized):
        try:
            ind = self.names.index(name)
        except ValueError:
            raise ValueError('Param %s not registered!' % name)

        param_name = self.param_names[ind]
        normalizer = self.normalizers[ind]

        vpre = v
        v = normalizer.check_bounds(v, normalized)

        if normalized:
            vpre_raw = normalizer.unnormalize(vpre)
            v_norm = v
            v_raw = normalizer.unnormalize(v)
        else:
            vpre_raw = vpre
            v_raw = v
            v_norm = normalizer.normalize(v)

        if vpre_raw != v_raw:
            rospy.logwarn('Bounding requested %s raw %f to (%f, %f)',
                          name, vpre_raw, normalizer.min_val, normalizer.max_val)

        return param_name, v_raw, v_norm, normalizer

    def set_values(self, v, names=None, normalized=True):
        """Sets all parameters according to vector input v.

        Optionally, assign the parameters with corresponding ordered names.
        """
        if names is None:
            names = self.names

        if len(v) != len(names):
            raise ValueError('Expected %d elements, got %d' %
                             (len(names), len(v)))

        # Set config
        config = self.client.get_configuration()
        procced = []
        out = 'Setting values:'
        for vi, v_name in zip(v, names):
            proc = self.process_element(v=vi,
                                        name=v_name,
                                        normalized=normalized)
            procced.append(proc)
            param_name, v_raw, v_norm, _ = proc
            config[param_name] = v_raw
            out += '\n\t%s: %f (%f)' % (v_name, v_raw, v_norm)

        if self._verbose:
            rospy.loginfo(out)

        # Check actual
        actual_config = self.client.update_configuration(config)
        resout = 'Actual values:'
        for vi, v_name, proc in zip(v, names, procced):
            param_name, v_raw, v_norm, normalizer = proc

            actual_raw = actual_config[param_name]
            actual = normalizer.normalize(actual_raw)
            resout += '\n\t%s: %f (%f)' % (v_name, actual_raw, actual)

            if v_raw != actual_raw:
                rospy.logwarn('Set param %s to %f but got actual %f',
                              v_name, v_raw, actual_raw)
        if self._verbose:
            rospy.loginfo(resout)

    def get_values(self, normalized=True):
        config = self.client.get_configuration()

        if not normalized:
            return [config[n] for n in self.names]
        else:
            return [nz.normalize(config[n]) for n, nz in izip(self.names, self.normalizers)]

    @property
    def num_parameters(self):
        return len(self.names)

    @property
    def parameter_names(self):
        return list(self.names)
