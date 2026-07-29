"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "stumpy.mojo")
LIBRARY = os.environ.get("MOJOSTUMPY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-stumpy.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mst_profile": (
        [
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
            F,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
        ],
        None,
    ),
    "mst_distance_profile": (
        [I, I, I, I, I, I, I, I, F, F, I, I, F, I],
        None,
    ),
    "mst_normalize_products": (
        [I, I, I, I, I, F, F, I],
        None,
    ),
}


class BuildError(RuntimeError):
    pass


def build() -> str:
    if os.environ.get("MOJOSTUMPY_LIB"):
        if not os.path.exists(LIBRARY):
            raise BuildError(f"MOJOSTUMPY_LIB does not exist: {LIBRARY}")
        return LIBRARY
    if os.path.exists(LIBRARY) and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE):
        return LIBRARY
    pixi = shutil.which("pixi")
    if pixi:
        command = [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "build",
        ]
    else:
        command = ["bash", os.path.join(ROOT, "build", "build.sh")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIBRARY):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIBRARY


_LIB: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIB, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIB


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)
