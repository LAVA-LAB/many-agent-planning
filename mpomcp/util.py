from __future__ import annotations
import numpy as np

def stable_normalizer(x, t=1.0):
    ''' Normalise with respect to temperature parameter t. '''
    x = np.array(x)
    x = (x / x.max())**t
    return np.abs(x / x.sum())