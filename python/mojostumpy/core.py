"""STUMPY-compatible matrix-profile API backed by Mojo."""

from __future__ import annotations

import math
import operator
import os
import warnings
from collections.abc import Callable

import numpy as np

from ._lib import addr, lib
from .mparray import mparray

EXCL_ZONE_DENOM = 4
MASS_FFT_WORK_THRESHOLD = 200_000_000


def _one_dimensional(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.ravel()
    if array.ndim != 1:
        raise ValueError(f"{name} is {array.ndim}-dimensional and must be 1-dimensional")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real values, found {array.dtype}")
    try:
        converted = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be convertible to float64") from error
    if np.issubdtype(array.dtype, np.integer):
        if array.size and np.any(np.abs(array.astype(object)) > 2**53):
            raise ValueError(f"{name} contains integers that cannot be represented as float64")
    elif np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > 8:
        equal = np.equal(array, converted) | (np.isnan(array) & np.isnan(converted))
        if not equal.all():
            raise ValueError(f"{name} contains values that cannot be represented as float64")
    return converted


def _integer(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _validate_window(m: int, n_a: int, n_b: int | None = None) -> int:
    m = _integer(m, "m")
    limit = n_a if n_b is None else min(n_a, n_b)
    if m < 3:
        raise ValueError("All window sizes must be greater than or equal to 3")
    if m > limit:
        raise ValueError(f"The window size must be less than or equal to {limit}")
    return m


def _window_flags(
    array: np.ndarray,
    m: int,
    finite: np.ndarray | None = None,
    all_finite: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if finite is None:
        finite = np.isfinite(array)
    if all_finite is None:
        all_finite = bool(finite.all())
    length = array.size - m + 1
    if all_finite:
        valid = np.ones(length, dtype=np.int64)
    else:
        bad_prefix = np.empty(array.size + 1, dtype=np.int64)
        bad_prefix[0] = 0
        np.cumsum(~finite, dtype=np.int64, out=bad_prefix[1:])
        valid = (bad_prefix[m:] == bad_prefix[:-m]).astype(np.int64)
    if m == 1:
        constant = valid.copy()
    else:
        changes = (array[1:] != array[:-1]) | ~finite[1:] | ~finite[:-1]
        if bool(changes.all()):
            constant = np.zeros(length, dtype=np.int64)
        else:
            change_prefix = np.empty(changes.size + 1, dtype=np.int64)
            change_prefix[0] = 0
            np.cumsum(changes, dtype=np.int64, out=change_prefix[1:])
            constant = (
                (change_prefix[m - 1 :] == change_prefix[: -(m - 1)])
                & valid.astype(bool)
            ).astype(np.int64)
    return valid, constant


def _constant_flags(
    original: np.ndarray,
    m: int,
    valid: np.ndarray,
    specification,
    default: np.ndarray,
) -> np.ndarray:
    if specification is None:
        return default
    values = specification(original, m) if callable(specification) else specification
    values = np.asarray(values)
    if values.shape != valid.shape:
        raise ValueError(
            f"constant-subsequence flags must have shape {valid.shape}, found {values.shape}"
        )
    if values.dtype != np.bool_:
        raise ValueError("constant-subsequence flags must have boolean dtype")
    return np.ascontiguousarray(values & valid.astype(bool), dtype=np.int64)


def _normalized_data(
    value,
    m: int,
    name: str,
    constant_specification=None,
    means=None,
    stds=None,
):
    original = _one_dimensional(value, name)
    finite = np.isfinite(original)
    all_finite = bool(finite.all())
    valid, default_constant = _window_flags(original, m, finite, all_finite)
    constants = _constant_flags(
        original, m, valid, constant_specification, default_constant
    )
    center = (
        float(np.mean(original if all_finite else original[finite]))
        if all_finite or finite.any()
        else 0.0
    )
    clean = original.copy()
    clean[~finite] = center
    clean -= center
    if means is None or stds is None:
        prefix = np.empty(clean.size + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(clean, out=prefix[1:])
        square_prefix = np.empty(clean.size + 1, dtype=np.float64)
        square_prefix[0] = 0.0
        np.square(clean, out=square_prefix[1:])
        np.cumsum(square_prefix[1:], out=square_prefix[1:])
        rolling_mean = np.subtract(prefix[m:], prefix[:-m])
        rolling_mean /= m
        rolling_std = np.subtract(square_prefix[m:], square_prefix[:-m])
        rolling_std /= m
        np.multiply(rolling_mean, rolling_mean, out=prefix[: rolling_mean.size])
        rolling_std -= prefix[: rolling_mean.size]
        np.maximum(rolling_std, 0.0, out=rolling_std)
        np.sqrt(rolling_std, out=rolling_std)
    else:
        rolling_mean = np.ascontiguousarray(means, dtype=np.float64) - center
        rolling_std = np.ascontiguousarray(stds, dtype=np.float64)
        expected = clean.size - m + 1
        if rolling_mean.shape != (expected,) or rolling_std.shape != (expected,):
            raise ValueError(f"M_T and Σ_T must have shape ({expected},)")
    return (
        np.ascontiguousarray(clean),
        np.ascontiguousarray(rolling_mean),
        np.ascontiguousarray(rolling_std),
        valid,
        constants,
    )


def _raw_data(value, m: int, name: str, finite_flags=None):
    original = _one_dimensional(value, name)
    finite = np.isfinite(original)
    valid, constants = _window_flags(original, m, finite, bool(finite.all()))
    if finite_flags is not None:
        supplied = np.asarray(finite_flags, dtype=bool)
        if supplied.shape != valid.shape:
            raise ValueError(f"T_subseq_isfinite must have shape {valid.shape}")
        valid = np.ascontiguousarray(supplied, dtype=np.int64)
    clean = original.copy()
    clean[~finite] = 0.0
    return clean, valid, constants


def _workers(diagonal_count: int) -> int:
    return max(1, min(diagonal_count, os.cpu_count() or 1, 16))


def _next_fast_length(target: int) -> int:
    candidate = target
    while True:
        remainder = candidate
        for factor in (2, 3, 5, 7, 11):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def _fft_sliding_dot_product(query: np.ndarray, series: np.ndarray) -> np.ndarray:
    transform_length = _next_fast_length(series.size + query.size - 1)
    series_spectrum = np.fft.rfft(series, transform_length)
    query_spectrum = np.fft.rfft(query, transform_length)
    np.conjugate(query_spectrum, out=query_spectrum)
    series_spectrum *= query_spectrum
    del query_spectrum
    products = np.fft.irfft(series_spectrum, transform_length)
    return products[: series.size - query.size + 1]


def _profile(
    T_A,
    m: int,
    T_B,
    ignore_trivial: bool,
    normalized: bool,
    p: float,
    k: int,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
) -> mparray:
    original_a = _one_dimensional(T_A, "T_A")
    supplied_b = T_B is not None
    original_b = original_a if T_B is None else _one_dimensional(T_B, "T_B")
    m = _validate_window(m, original_a.size, original_b.size)
    k = _integer(k, "k")
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    p = float(p)
    if not math.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and greater than 0")

    same = (
        original_a.shape == original_b.shape
        and np.array_equal(original_a, original_b, equal_nan=True)
    )
    self_join = (not supplied_b) or (same and bool(ignore_trivial))
    if supplied_b and bool(ignore_trivial) and not same:
        warnings.warn(
            "A-B join detected; ignore_trivial has been set to False",
            UserWarning,
            stacklevel=2,
        )
        self_join = False
    if not supplied_b and not ignore_trivial:
        warnings.warn(
            "A self-join was detected; ignore_trivial has been set to True",
            UserWarning,
            stacklevel=2,
        )

    if normalized:
        a, means_a, stds_a, valid_a, constants_a = _normalized_data(
            original_a, m, "T_A", T_A_subseq_isconstant
        )
        b, means_b, stds_b, valid_b, constants_b = _normalized_data(
            original_b,
            m,
            "T_B",
            T_A_subseq_isconstant
            if not supplied_b
            else T_B_subseq_isconstant,
        )
    else:
        a, valid_a, constants_a = _raw_data(original_a, m, "T_A")
        b, valid_b, constants_b = _raw_data(original_b, m, "T_B")
        means_a = np.zeros(original_a.size - m + 1)
        stds_a = np.ones(original_a.size - m + 1)
        means_b = np.zeros(original_b.size - m + 1)
        stds_b = np.ones(original_b.size - m + 1)

    length_a = a.size - m + 1
    length_b = b.size - m + 1
    exclusion = math.ceil(m / EXCL_ZONE_DENOM)
    diagonal_count = (
        max(0, length_a - exclusion - 1)
        if self_join
        else length_a + length_b - 1
    )
    workers = _workers(diagonal_count)
    profiles = np.full((workers, length_a, k), np.inf, dtype=np.float64)
    indices = np.full((workers, length_a, k), -1, dtype=np.int64)
    left_profiles = np.full((workers, length_a), np.inf, dtype=np.float64)
    right_profiles = np.full((workers, length_a), np.inf, dtype=np.float64)
    left_indices = np.full((workers, length_a), -1, dtype=np.int64)
    right_indices = np.full((workers, length_a), -1, dtype=np.int64)

    lib().mst_profile(
        addr(a),
        addr(b),
        a.size,
        b.size,
        m,
        addr(means_a),
        addr(stds_a),
        addr(means_b),
        addr(stds_b),
        addr(valid_a),
        addr(valid_b),
        addr(constants_a),
        addr(constants_b),
        k,
        int(self_join),
        int(normalized),
        float(p),
        workers,
        addr(profiles),
        addr(indices),
        addr(left_profiles),
        addr(right_profiles),
        addr(left_indices),
        addr(right_indices),
    )

    result = np.empty((length_a, 2 * k + 2), dtype=object)
    result[:, :k] = profiles[0]
    result[:, k : 2 * k] = indices[0]
    result[:, 2 * k] = left_indices[0]
    result[:, 2 * k + 1] = right_indices[0]
    return mparray(result, m, k, EXCL_ZONE_DENOM)


def stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """Compute an exact z-normalized matrix profile."""
    return _profile(
        T_A,
        m,
        T_B,
        ignore_trivial,
        bool(normalize),
        p,
        k,
        T_A_subseq_isconstant,
        T_B_subseq_isconstant,
    )


def aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1):
    """Compute an exact non-normalized matrix profile."""
    return _profile(T_A, m, T_B, ignore_trivial, False, p, k)


def mass(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    normalize=True,
    p=2.0,
    T_subseq_isfinite=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
    query_idx=None,
):
    """Compute the distance profile between query ``Q`` and every window in ``T``."""
    query = _one_dimensional(Q, "Q")
    series = _one_dimensional(T, "T")
    m = _validate_window(query.size, series.size)
    length = series.size - m + 1
    result = np.full(length, np.inf, dtype=np.float64)
    if not np.isfinite(query).all():
        result.fill(np.inf)
        return result
    p = float(p)
    if not math.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and greater than 0")

    if normalize:
        (
            clean_series,
            means,
            stds,
            valid,
            constants,
        ) = _normalized_data(
            series,
            m,
            "T",
            T_subseq_isconstant,
            M_T,
            Σ_T,
        )
        (
            clean_query,
            query_means,
            query_stds,
            _,
            query_constants,
        ) = _normalized_data(query, m, "Q", Q_subseq_isconstant)
        query_mean = float(query_means[0])
        query_std = float(query_stds[0])
        query_constant = int(query_constants[0])
    else:
        clean_series, valid, constants = _raw_data(
            series, m, "T", T_subseq_isfinite
        )
        clean_query, _, query_constants = _raw_data(query, m, "Q")
        means = np.zeros(length)
        stds = np.ones(length)
        query_mean = 0.0
        query_std = 1.0
        query_constant = int(query_constants[0])

    use_fft = (
        normalize
        and length * m >= MASS_FFT_WORK_THRESHOLD
        and query_constant == 0
        and bool(valid.all())
        and not bool(constants.any())
    )
    if use_fft:
        products = _fft_sliding_dot_product(clean_query, clean_series)
        lib().mst_normalize_products(
            addr(products),
            addr(means),
            addr(stds),
            length,
            m,
            query_mean,
            query_std,
            addr(result),
        )
    else:
        lib().mst_distance_profile(
            addr(clean_query),
            addr(clean_series),
            m,
            series.size,
            addr(means),
            addr(stds),
            addr(valid),
            addr(constants),
            query_mean,
            query_std,
            query_constant,
            int(bool(normalize)),
            float(p),
            addr(result),
        )
    if query_idx is not None:
        query_idx = _integer(query_idx, "query_idx")
        if 0 <= query_idx < result.size:
            result[query_idx] = 0.0
    return result


def match(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    max_distance=None,
    max_matches=None,
    atol=1e-8,
    query_idx=None,
    normalize=True,
    p=2.0,
    T_subseq_isfinite=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    """Find non-overlapping matches of ``Q`` in ``T``, sorted by distance."""
    query = _one_dimensional(Q, "Q")
    if not np.isfinite(query).all():
        raise ValueError("Q contains illegal values (NaN or inf)")
    distances = mass(
        query,
        T,
        M_T,
        Σ_T,
        normalize,
        p,
        T_subseq_isfinite,
        T_subseq_isconstant,
        Q_subseq_isconstant,
        query_idx,
    )
    if max_distance is None:
        finite = distances[np.isfinite(distances)]
        max_distance = max(
            float(np.mean(finite) - 2.0 * np.std(finite)), float(np.min(finite))
        )
    elif callable(max_distance):
        max_distance = max_distance(distances.copy())
    max_matches = (
        math.inf if max_matches is None else _integer(max_matches, "max_matches")
    )
    exclusion = math.ceil(query.size / EXCL_ZONE_DENOM)
    candidates = distances.copy()
    found: list[list[float | int]] = []
    candidate = int(query_idx) if query_idx is not None else int(np.argmin(candidates))
    while (
        len(found) < max_matches
        and np.isfinite(candidates[candidate])
        and candidates[candidate] <= float(max_distance) + atol
    ):
        found.append([distances[candidate], candidate])
        start = max(0, candidate - exclusion)
        stop = min(candidates.size, candidate + exclusion + 1)
        candidates[start:stop] = np.inf
        candidate = int(np.argmin(candidates))
    return np.asarray(found, dtype=object).reshape((-1, 2))


def _mpdist_profiles(T_A, T_B, m, normalize: bool, p: float) -> np.ndarray:
    first = _profile(T_A, m, T_B, False, normalize, p, 1).P_
    second = _profile(T_B, m, T_A, False, normalize, p, 1).P_
    return np.concatenate((first, second))


def mpdist(
    T_A,
    T_B,
    m,
    percentage=0.05,
    k=None,
    normalize=True,
    p=2.0,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
):
    """Compute the z-normalized matrix-profile distance between two series."""
    if T_A_subseq_isconstant is not None or T_B_subseq_isconstant is not None:
        first = _profile(
            T_A,
            m,
            T_B,
            False,
            bool(normalize),
            p,
            1,
            T_A_subseq_isconstant,
            T_B_subseq_isconstant,
        ).P_
        second = _profile(
            T_B,
            m,
            T_A,
            False,
            bool(normalize),
            p,
            1,
            T_B_subseq_isconstant,
            T_A_subseq_isconstant,
        ).P_
        profiles = np.concatenate((first, second))
    else:
        profiles = _mpdist_profiles(T_A, T_B, m, bool(normalize), p)
    if k is None:
        percentage = float(np.clip(percentage, 0.0, 1.0))
        k = min(math.ceil(percentage * (len(T_A) + len(T_B))), profiles.size - 1)
    else:
        k = min(_integer(k, "k"), profiles.size - 1)
    return float(np.partition(profiles, k)[k])


def aampdist(T_A, T_B, m, percentage=0.05, k=None, p=2.0):
    """Compute the non-normalized matrix-profile distance between two series."""
    return mpdist(T_A, T_B, m, percentage, k, normalize=False, p=p)
