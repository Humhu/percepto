"""Contains classes for combining interfaces together into a single interface.
"""

from paraset_interface import parse_paraset_interface
from dyna_reconfig_interface import parse_reconfig_interface

import numpy as np
import rospy


def parse_meta_interface(info):
    """Parses a dictionary to produce a MetaInterface
    """
    if 'verbose' in info:
        verbose = info.pop('verbose')
    else:
        verbose = False

    interface = MetaInterface(verbose)

    if 'interface_order' in info:
        interface_order = info.pop('interface_order')
    else:
        interface_order = info.keys()

    ordered_items = [None] * len(info)
    for name, v in info.iteritems():
        try:
            ind = interface_order.index(name)
        except ValueError:
            rospy.logerr('Could not find interface %s in ordering %s',
                         name, str(interface_order))
        ordered_items[ind] = (name, v)

    for name, v in ordered_items:
        interface_type = v.pop('type')
        if interface_type == 'paraset':
            rospy.loginfo('Parsing paraset subinterface %s', name)
            subint = parse_paraset_interface(v)
        elif interface_type == 'dynamic_reconfigure':
            rospy.loginfo('Parsing dynamic_reconfigure subinterface %s', name)
            subint = parse_reconfig_interface(v)
        else:
            raise ValueError('Unknown interface type: %s' % interface_type)
        interface.add_interface(subint)

    return interface

# TODO Base interface interface class?
class MetaInterface(object):
    def __init__(self, verbose=False):
        self.interfaces = []
        self._verbose = verbose

    def add_interface(self, inter):
        self.interfaces.append(inter)

    def set_values(self, v, names=None, normalized=True):
        if names is None:
            names = self.parameter_names

        if len(v) != len(names):
            raise ValueError('Expected %d elements, got %d' %
                             (len(names), len(v)))

        if self._verbose:
            rospy.loginfo('Setting constituent interfaces...')

        ind = 0
        for i in self.interfaces:
            bdim = i.num_parameters
            subnames = names[ind:ind + bdim]
            subv = v[ind:ind + bdim]
            i.set_values(v=subv, names=subnames, normalized=normalized)

            ind += bdim

        if self._verbose:
            rospy.loginfo('Setting complete.')

    def get_values(self, normalized=False):
        out = []
        for i in self.interfaces:
            out.extend(i.get_values(normalized=normalized))
        return out

    @property
    def num_parameters(self):
        return np.sum([i.num_parameters for i in self.interfaces])

    @property
    def parameter_names(self):
        names = []
        for i in self.interfaces:
            names.extend(i.parameter_names)
        return names
