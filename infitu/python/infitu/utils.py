"""Utilities for parameter scaling
"""

class ParameterNormalizer(object):
    def __init__(self, min_val, max_val, enable_rounding):
        self.offset = 0.5 * (min_val + max_val)
        self.scale = 0.5 * (max_val - min_val)
        if self.scale < 0:
            raise ValueError('max_val must be > min_val')
        self.min_val = min_val
        self.max_val = max_val
        self.enable_rounding = enable_rounding

    def check_bounds(self, x, normalized):
        if normalized:
            x = max(min(x, 1), -1)
        else:
            x = max(min(x, self.max_val), self.min_val)
        return x

    def normalize(self, x):
        if self.enable_rounding:
            x = int(round(x))
        return (x - self.offset) / self.scale

    def unnormalize(self, x):
        x = (x * self.scale) + self.offset
        if self.enable_rounding:
            x = int(round(x))
        return x
