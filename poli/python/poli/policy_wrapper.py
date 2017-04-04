"""This module contains convenience classes for managing policies and input state_procs
together, as well as functions for constructing them from spec dicts."""

import numpy as np
import poli.policies as pp
import poli.feature_preprocessing as pfpp
import poli.output_constraints as poc


def parse_policy_wrapper(raw_input_dim, output_dim, info):

    # Parse input preprocessing
    input_dim = raw_input_dim
    normalizer = None
    augmenter = None
    enable_homogeneous = False
    if 'input_processing' in info:
        ipp_info = info['input_processing']

        if 'normalization' in ipp_info:
            norm_info = ipp_info['normalization']
            normalizer = pfpp.OnlineFeatureNormalizer(dim=input_dim, **norm_info)

        if 'augmentation' in ipp_info:
            aug_info = ipp_info['augmentation']
            augmenter = pfpp.parse_feature_augmenter(dim=input_dim, spec=aug_info)
            input_dim = augmenter.dim

        if 'enable_homogeneous' in ipp_info:
            enable_homogeneous = bool(ipp_info['enable_homogeneous'])
            if enable_homogeneous:
                input_dim += 1

    def state_proc(x):
        if normalizer is not None:
            x = normalizer.process(x)
            if x is None:
                return None
        if augmenter is not None:
            x = augmenter.process(x)
        if enable_homogeneous:
            x = np.hstack((x, 1))
        return x

    constrainer = None
    if 'output_processing' in info:
        out_info = info['output_processing']
        constrainer = poc.parse_output_constraints(dim=output_dim, spec=out_info)

    def action_proc(x):
        if constrainer is not None:
            x = constrainer.process(x)
        return x

    # Parse policy
    policy_info = info.pop('policy')
    policy_info['input_dim'] = input_dim
    policy_info['output_dim'] = output_dim
    policy = pp.parse_policy(policy_info)

    return PolicyWrapper(policy=policy,
                         state_proc=state_proc,
                         action_proc=action_proc)


class PolicyWrapper(object):
    """Convenience class that combines multiple objects.
    """

    def __init__(self, policy, state_proc=None, action_proc=None):
        self._policy = policy
        self._state_proc = state_proc
        self._action_proc = action_proc

    def process_input(self, raw_state):
        if self._state_proc is None:
            return raw_state
        else:
            return self._state_proc(raw_state)

    def process_output(self, output):
        if self._action_proc is None:
            return output
        else:
            return self._action_proc(output)

    def sample_action(self, state, proc_input=True, proc_output=True):
        if proc_input:
            state = self.process_input(state)
            if state is None:
                return None

        action = self._policy.sample_action(state)
        if proc_output:
            action = self.process_output(action)
        return action

    @property
    def policy(self):
        return self._policy
