"""
Shared visualisation helpers for battleship / plate active learning.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def interpolate_curve(history: List[Dict], key: str, max_steps: int) -> np.ndarray:
    """Forward-fill a sparse episode history into a fixed-length array."""
    v = np.zeros(max_steps)
    prev, ptr = 0.0, 0
    for entry in history:
        idx = entry["step"] - 1
        while ptr <= idx and ptr < max_steps:
            v[ptr] = prev
            ptr += 1
        if idx < max_steps:
            v[idx] = entry[key]
            prev = entry[key]
    while ptr < max_steps:
        v[ptr] = prev
        ptr += 1
    return v
