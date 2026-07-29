"""Benchmarks against upstream STUMPY on identical contiguous arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojostumpy as mst  # noqa: E402
import stumpy  # noqa: E402


def time_best(function, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


CASES = []


def case(name):
    def decorate(function):
        CASES.append((name, function))
        return function

    return decorate


@case("stump self-join (5,000, m=64)")
def _():
    values = np.random.default_rng(1).normal(size=5_000)
    return lambda: mst.stump(values, 64), lambda: stumpy.stump(values, 64)


@case("stump AB-join (3,500 x 4,000, m=64)")
def _():
    rng = np.random.default_rng(2)
    first = rng.normal(size=3_500)
    second = rng.normal(size=4_000)
    return (
        lambda: mst.stump(first, 64, second, ignore_trivial=False),
        lambda: stumpy.stump(first, 64, second, ignore_trivial=False),
    )


@case("stump top-3 self-join (3,000, m=64)")
def _():
    values = np.random.default_rng(3).normal(size=3_000)
    return (
        lambda: mst.stump(values, 64, k=3),
        lambda: stumpy.stump(values, 64, k=3),
    )


@case("aamp p=2 self-join (5,000, m=64)")
def _():
    values = np.random.default_rng(4).normal(size=5_000)
    return lambda: mst.aamp(values, 64), lambda: stumpy.aamp(values, 64)


@case("mass distance profile (1,000,000, m=128)")
def _():
    rng = np.random.default_rng(5)
    query = rng.normal(size=128)
    values = rng.normal(size=1_000_000)
    return lambda: mst.mass(query, values), lambda: stumpy.mass(query, values)


@case("mpdist (1,500 x 1,700, m=48)")
def _():
    rng = np.random.default_rng(6)
    first = rng.normal(size=1_500)
    second = rng.normal(size=1_700)
    return (
        lambda: mst.mpdist(first, second, 48),
        lambda: stumpy.mpdist(first, second, 48),
    )


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"STUMPY {stumpy.__version__}"
    )
    print()
    print("| benchmark | mojo-stumpy | STUMPY | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, make_case in CASES:
        ours, upstream = make_case()
        ours_result = ours()
        upstream_result = upstream()
        if hasattr(ours_result, "P_"):
            assert np.allclose(ours_result.P_, upstream_result.P_, atol=1e-8)
        else:
            assert np.allclose(ours_result, upstream_result, atol=1e-8)
        ours_time = time_best(ours)
        upstream_time = time_best(upstream)
        ratio = upstream_time / ours_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
