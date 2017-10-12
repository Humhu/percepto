"""Contains classes that provide wrappers around sets of runtime parameters.
"""

from itertools import izip
import rospy
import paraset
from infitu.utils import *


def parse_paraset_interface(info):
    """Parses a dictionary to produce a NumericParameterInterface.
    """
    verbose = False
    if 'verbose' in info:
        verbose = bool(info.pop('verbose'))

    persistent = True
    if 'persistent' in info:
        persistent = bool(info.pop('persistent'))

    n_retries = 1
    if 'n_retries' in info:
        n_retries = int(info.pop('n_retries'))

    interface = NumericParameterInterface(verbose=verbose,
                                          persistent=persistent)

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
        param_names = v['param_name']
        if isinstance(param_names, str):
            param_names = [param_names]
        base_topics = v['base_namespace']
        if isinstance(base_topics, str):
            base_topics = [base_topics]
        normalizer = parse_normalizer(v)

        interface.add_parameter(name=name,
                                param_names=param_names,
                                base_topics=base_topics,
                                normalizer=normalizer)
    return interface


class NumericParameterInterface(object):
    """Wraps and normalizes a set of numeric runtime parameters,
    allowing them to be set with a vector input.

    Parameters are sorted alphabetically by name and scaled to correspond
    to an input range of -1 to 1.
    """

    def __init__(self, verbose=False, persistent=True, n_retries=5):
        self.names = []
        self.pnames = []
        self.setters = []
        self.normalizers = []
        self._verbose = verbose
        self._persistent = persistent
        self._n_retries = n_retries

    def add_parameter(self, name, param_names, base_topics, normalizer):
        """Adds another parameter to this interface.
        """
        if self._verbose:
            rospy.loginfo('Adding parameter %s' % name)

        setters = []
        full_names = []
        for pname, btop in zip(param_names, base_topics):
            setters.append(paraset.RuntimeParamSetter(param_type=float,
                                                      name=pname,
                                                      base_topic=btop,
                                                      persistent=self._persistent,
                                                      n_retries=self._n_retries))
            full_names.append(btop + '/' + pname)

        self.names.append(name)
        self.pnames.append(full_names)
        self.setters.append(setters)
        self.normalizers.append(normalizer)

    def map_values(self, v, names=None, normalized=True):
        if names is None:
            names = self.parameter_names

        if len(v) != len(names):
            raise ValueError('Expected %d elements, got %d' %
                             (len(names), len(v)))

        out = [self.process_element(vi, ni, normalized) for vi, ni in zip(v, names)]
        return zip(*out)[0:2]

    def process_element(self, v, name, normalized):
        try:
            ind = self.names.index(name)
        except ValueError:
            raise ValueError('Param %s not registered!' % name)

        pnames = self.pnames[ind]
        setters = self.setters[ind]
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

        return v_raw, v_norm, pnames, setters, normalizer

    def set_values(self, v, names=None, normalized=True):
        """Sets all parameters according to vector input v.

        Optionally, assign the parameters with corresponding ordered names.

        Elements of v outside of -1, 1 will be truncated
        """

        if names is None:
            names = self.parameter_names

        if len(v) != len(names):
            raise ValueError('Expected %d elements, got %d' %
                             (len(names), len(v)))

        out = 'Setting values:'
        resout = 'Actual values:'
        warnouts = []
        outvals = []
        for vi, v_name in zip(v, names):

            proc = self.process_element(v=vi,
                                        name=v_name,
                                        normalized=normalized)
            v_raw, v_norm, pnames, setters, normalizer = proc

            for pname, setter in zip(pnames, setters):
                out += '\n\t%s (%s): %f (%f)' % (v_name, pname, v_raw, v_norm)
                actual_raw = setter.set_value(v_raw)
                actual = normalizer.normalize(actual_raw)
                resout += '\n\t%s (%s): %f (%f)' % (v_name,
                                                    pname, actual_raw, actual)
                if v_raw != actual_raw:
                    warnouts.append('Set param %s to %f but got actual %f' % (
                        pname, v_raw, actual_raw))

        if self._verbose:
            rospy.loginfo(out)
            rospy.loginfo(resout)
            for w in warnouts:
                rospy.logwarn(w)

    def get_values(self, normalized=True):
        # TODO Handle multi-setters properly
        if normalized:
            return [nn.normalize(ss[0].get_value())
                    for nn, ss in izip(self.normalizers, self.setters)]
        else:
            return [s[0].get_value() for s in self.setters]

    @property
    def num_parameters(self):
        """The number of parameters wrapped in this interface.
        """
        return len(self.names)

    @property
    def parameter_names(self):
        """An ordered list of the parameter names
        """
        return list(self.names)
