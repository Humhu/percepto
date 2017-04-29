"""Contains top-level parsers for all interfaces.
"""

from meta_interface import parse_meta_interface
from paraset_interface import parse_paraset_interface
from dyna_reconfig_interface import parse_reconfig_interface

# TODO Generalize/move guts of meta parser to here

def parse_interface(info):
    interface_type = info.pop('type')
    if interface_type == 'meta':
        return parse_meta_interface(info)
    elif interface_type == 'paraset':
        return parse_paraset_interface(info)
    elif interface_type == 'dynamic_reconfigure':
        return parse_reconfig_interface(info)
    else:
        raise ValueError('Unknown interface type: %s' % interface_type)