# mojo-stumpy

`mojo-stumpy` is a standalone Mojo port of the compute-heavy core of
[STUMPY](https://github.com/stumpy-dev/stumpy). It computes exact one-dimensional
matrix profiles and exposes a Python API with the same covered function names,
signatures, return layout, and `mparray` convenience properties as STUMPY 1.14.

This is useful when exact CPU matrix-profile computation dominates a workload.
It is not a fork of STUMPY and does not import STUMPY at runtime. STUMPY is an
environment dependency only so that the tests and benchmarks can compare both
implementations directly.

## Coverage

The covered API is:

- `stump`: normalized self-joins and AB-joins, including top-k profiles,
  exclusion zones, left/right indices, non-finite windows, constant windows,
  custom constant flags, and `normalize=False`
- `aamp`: non-normalized self-joins and AB-joins for arbitrary positive p-norms
- `mass`: normalized and non-normalized distance profiles, precomputed rolling
  statistics, non-finite windows, custom constant flags, and `query_idx`
- `match`: exclusion-zone query matching over the Mojo distance profile
- `mpdist` and `aampdist`: matrix-profile distance
- `mparray`: STUMPY-compatible `P_`, `I_`, `left_I_`, and `right_I_` properties

Not covered are streaming and approximate profiles (`stumpi`, `scrump`),
multidimensional profiles (`mstump`), distributed or GPU entry points, and
higher-level motif, discord, segmentation, chain, and consensus algorithms.
Inputs in the covered subset are one-dimensional real arrays that can be
represented safely as `float64`. Complex values and integers outside the exact
`float64` integer range are rejected instead of being silently narrowed.

## Install

Install the pinned Mojo toolchain, Python, NumPy, pytest, and upstream STUMPY:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-stumpy.so`. All development commands run
through Pixi:

```bash
pixi run test
pixi run bench
```

## Usage

This example runs from the repository after `pixi install` and `pixi run build`:

```python
import numpy as np
import mojostumpy as stumpy

T = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
mp = stumpy.stump(T, m=3)

print(mp.P_)
# [0.11633857 2.69407392 3.00009263 2.69407392 0.11633857]
print(mp.I_)
# [4 3 0 1 0]

query = np.array([-11.1, 23.4, 79.5, 1001.0])
print(stumpy.mass(query, T))
# [3.18792463e+00 1.11297393e-03 3.23874018e+00 3.34470195e+00]
```

Save it as a script and run it with `pixi run python script.py`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, NumPy 2.4.6, and STUMPY 1.14.1. Each row uses the
best of three warmed runs on identical contiguous arrays. The Pixi task holds a
machine-wide lock for the whole benchmark.

| benchmark | mojo-stumpy | STUMPY | result |
| --- | ---: | ---: | ---: |
| stump self-join (5,000, m=64) | 44.42 ms | 3569.42 ms | 80.35x faster |
| stump AB-join (3,500 x 4,000, m=64) | 45.95 ms | 3667.54 ms | 79.82x faster |
| stump top-3 self-join (3,000, m=64) | 24.93 ms | 3140.14 ms | 125.95x faster |
| aamp p=2 self-join (5,000, m=64) | 36.14 ms | 3625.20 ms | 100.32x faster |
| mass distance profile (1,000,000, m=128) | 217.72 ms | 230.43 ms | 1.06x faster |
| mpdist (1,500 x 1,700, m=48) | 30.80 ms | 7209.67 ms | 234.10x faster |

Large, finite normalized MASS workloads use an FFT correlation followed by a
parallel SIMD normalization pass in Mojo. Smaller or irregular workloads stay
on the direct SIMD kernel to avoid FFT and thread-launch overhead. The exact
matrix-profile kernels walk distance-matrix diagonals with constant-time
recurrence updates.

There is no GPU path. These kernels stream through input arrays with low
arithmetic intensity, so this port focuses on CPU execution.

## How it works

NumPy owns all memory. The Python layer converts inputs to C-contiguous
`float64` arrays, allocates outputs and per-worker scratch arrays, then passes
their integer addresses through `ctypes`. The Mojo C ABI reconstructs
`UnsafePointer` values from those addresses; Mojo never owns or frees a Python
buffer.

The matrix-profile kernel traverses diagonals of the implicit distance matrix.
Sliding dot products update normalized distances in constant time, while
sliding powered differences do the same for AAMP. Independent workers keep
private top-k, left, and right profiles to avoid write races, followed by a
deterministic reduction. Arrays are row-major, and top-k scratch storage has
shape `(workers, subsequences, k)`.

MASS uses SIMD-vectorized direct dot products for small or irregular inputs and
FFT correlation for large, finite normalized inputs. Rolling finite, constant,
mean, and standard-deviation metadata is prepared in linear time before
entering Mojo.
