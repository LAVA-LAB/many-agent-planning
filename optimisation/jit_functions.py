from __future__ import annotations

import math
import random
import itertools
import numpy as np

from numba import jit, int32, float32, njit, prange

@jit(float32(float32, int32, int32, float32), nopython=True, cache=True)
def UCB(qnode_value : float, root_visits : int, child_visits : int, c : float) -> float:
    return qnode_value + c * math.sqrt(math.log(root_visits + 1) / (child_visits + 1))
