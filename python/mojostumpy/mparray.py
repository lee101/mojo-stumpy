"""The ndarray subclass returned by matrix-profile functions."""

from __future__ import annotations

import numpy as np


class mparray(np.ndarray):
    def __new__(cls, input_array, m: int, k: int, excl_zone_denom: int = 4):
        obj = np.asarray(input_array).view(cls)
        obj._m = m
        obj._k = k
        obj._excl_zone_denom = excl_zone_denom
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._m = getattr(obj, "_m", None)
        self._k = getattr(obj, "_k", None)
        self._excl_zone_denom = getattr(obj, "_excl_zone_denom", None)

    @property
    def P_(self):
        values = self[:, : self._k].astype(np.float64)
        return values.ravel() if self._k == 1 else values

    @property
    def I_(self):
        values = self[:, self._k : 2 * self._k].astype(np.int64)
        return values.ravel() if self._k == 1 else values

    @property
    def left_I_(self):
        return self[:, 2 * self._k].astype(np.int64).ravel()

    @property
    def right_I_(self):
        return self[:, 2 * self._k + 1].astype(np.int64).ravel()

