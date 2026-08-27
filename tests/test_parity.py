"""Numerical and behavioral parity with STUMPY 1.14."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import stumpy
from stumpy import core as stumpy_core

import mojostumpy as mst
from mojostumpy import core as mst_core
from mojostumpy._lib import addr

RNG = np.random.default_rng(2026)


def assert_profile_equal(ours, theirs, *, atol=1e-10, indices=True):
    assert ours.shape == theirs.shape
    assert np.allclose(ours.P_, theirs.P_, atol=atol, rtol=1e-10, equal_nan=True)
    if indices:
        assert np.array_equal(ours.I_, theirs.I_)
        assert np.array_equal(ours.left_I_, theirs.left_I_)
        assert np.array_equal(ours.right_I_, theirs.right_I_)


def test_published_stump_vector():
    values = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    ours = mst.stump(values, m=3)
    theirs = stumpy.stump(values, m=3)
    assert_profile_equal(ours, theirs)
    assert ours.I_.tolist() == [4, 3, 0, 1, 0]
    assert ours.left_I_.tolist() == [-1, -1, 0, 1, 0]
    assert ours.right_I_.tolist() == [4, 3, 4, -1, -1]


@pytest.mark.parametrize("n,m", [(40, 5), (67, 9), (100, 12)])
def test_stump_random_self_join(n, m):
    values = RNG.normal(size=n)
    assert_profile_equal(mst.stump(values, m), stumpy.stump(values, m))


def test_stump_top_k():
    values = RNG.normal(size=73)
    ours = mst.stump(values, 8, k=4)
    theirs = stumpy.stump(values, 8, k=4)
    assert ours.P_.shape == theirs.P_.shape == (66, 4)
    assert_profile_equal(ours, theirs)


def test_stump_ab_join():
    first = RNG.normal(size=57)
    second = RNG.normal(size=71)
    ours = mst.stump(first, 7, second, ignore_trivial=False, k=3)
    theirs = stumpy.stump(first, 7, second, ignore_trivial=False, k=3)
    assert_profile_equal(ours, theirs)
    assert np.all(ours.left_I_ == -1)
    assert np.all(ours.right_I_ == -1)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_aamp_self_join(p):
    values = RNG.normal(size=65)
    ours = mst.aamp(values, 9, p=p, k=2)
    theirs = stumpy.aamp(values, 9, p=p, k=2)
    assert_profile_equal(ours, theirs, atol=3e-8)


def test_aamp_ab_join():
    first = RNG.normal(size=48)
    second = RNG.normal(size=52)
    assert_profile_equal(
        mst.aamp(first, 6, second, ignore_trivial=False),
        stumpy.aamp(first, 6, second, ignore_trivial=False),
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_nonfinite_and_constant_windows(normalize):
    values = np.r_[np.ones(8), np.arange(10.0), np.nan, np.ones(8) * 2.0]
    ours = mst.stump(values, 4, normalize=normalize)
    theirs = stumpy.stump(values, 4, normalize=normalize)
    assert np.allclose(ours.P_, theirs.P_, atol=1e-10, equal_nan=True)
    assert np.array_equal(np.isinf(ours.P_), np.isinf(theirs.P_))


def test_custom_constant_subsequence_flags():
    values = RNG.normal(size=55)
    flags = np.zeros(values.size - 7 + 1, dtype=bool)
    flags[[3, 18, 33]] = True
    ours = mst.stump(values, 7, T_A_subseq_isconstant=flags)
    theirs = stumpy.stump(values, 7, T_A_subseq_isconstant=flags)
    assert_profile_equal(ours, theirs, indices=False)
    for row in np.flatnonzero(flags):
        assert flags[ours.I_[row]]


def test_custom_constant_flags_for_ab_join():
    first = RNG.normal(size=41)
    second = RNG.normal(size=47)
    first_flags = np.zeros(first.size - 6 + 1, dtype=bool)
    second_flags = np.zeros(second.size - 6 + 1, dtype=bool)
    first_flags[[2, 20]] = True
    second_flags[[4, 25]] = True
    ours = mst.stump(
        first,
        6,
        second,
        ignore_trivial=False,
        T_A_subseq_isconstant=first_flags,
        T_B_subseq_isconstant=second_flags,
    )
    theirs = stumpy.stump(
        first,
        6,
        second,
        ignore_trivial=False,
        T_A_subseq_isconstant=first_flags,
        T_B_subseq_isconstant=second_flags,
    )
    assert_profile_equal(ours, theirs, indices=False)


def test_mass_published_vector():
    query = np.array([-11.1, 23.4, 79.5, 1001.0])
    series = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    assert np.allclose(mst.mass(query, series), stumpy.mass(query, series), atol=1e-11)


@pytest.mark.parametrize(
    "normalize,p", [(True, 2.0), (False, 1.0), (False, 2.0), (False, 3.0)]
)
def test_mass_random_distance_profile(normalize, p):
    query = RNG.normal(size=13)
    series = RNG.normal(size=93)
    ours = mst.mass(query, series, normalize=normalize, p=p)
    theirs = stumpy.mass(query, series, normalize=normalize, p=p)
    assert np.allclose(ours, theirs, atol=3e-8, rtol=1e-10)


def test_mass_with_precomputed_statistics():
    query = RNG.normal(size=11)
    series = RNG.normal(size=80)
    means, stds = stumpy_core.compute_mean_std(series, query.size)
    ours = mst.mass(query, series, M_T=means, Σ_T=stds)
    theirs = stumpy.mass(query, series, M_T=means, Σ_T=stds)
    assert np.allclose(ours, theirs, atol=1e-10)


def test_mass_with_custom_constant_flags():
    query = RNG.normal(size=9)
    series = RNG.normal(size=60)
    series_flags = np.zeros(series.size - query.size + 1, dtype=bool)
    series_flags[[5, 21]] = True
    query_flags = np.ones(1, dtype=bool)
    ours = mst.mass(
        query,
        series,
        T_subseq_isconstant=series_flags,
        Q_subseq_isconstant=query_flags,
    )
    theirs = stumpy.mass(
        query,
        series,
        T_subseq_isconstant=series_flags,
        Q_subseq_isconstant=query_flags,
    )
    assert np.allclose(ours, theirs, atol=1e-10)


def test_non_normalized_mass_with_custom_finite_flags():
    query = RNG.normal(size=8)
    series = RNG.normal(size=50)
    finite_flags = np.ones(series.size - query.size + 1, dtype=bool)
    finite_flags[[3, 17]] = False
    ours = mst.mass(
        query,
        series,
        normalize=False,
        T_subseq_isfinite=finite_flags,
    )
    theirs = stumpy.mass(
        query,
        series,
        normalize=False,
        T_subseq_isfinite=finite_flags,
    )
    assert np.allclose(ours, theirs, atol=1e-10)


def test_mass_nonfinite_windows_and_query_index():
    series = RNG.normal(size=70)
    series[12] = np.inf
    query = series[30:39].copy()
    ours = mst.mass(query, series, query_idx=30)
    theirs = stumpy.mass(query, series, query_idx=30)
    assert np.allclose(ours, theirs, atol=1e-10, equal_nan=True)
    assert ours[30] == 0.0
    assert np.array_equal(np.isinf(ours), np.isinf(theirs))


@pytest.mark.parametrize("n", [139, 16_421])
def test_mass_fft_serial_parallel_and_simd_tail(monkeypatch, n):
    query = RNG.normal(size=17)
    series = RNG.normal(size=n)
    monkeypatch.setattr(mst_core, "MASS_FFT_WORK_THRESHOLD", 0)
    ours = mst.mass(query, series)
    theirs = stumpy.mass(query, series)
    assert (ours.size % 2) != 0
    assert np.allclose(ours, theirs, atol=1e-10, rtol=1e-10)


def test_mass_all_finite_nonconstant_fast_path():
    rng = np.random.default_rng(101)
    series = rng.normal(size=205)
    query = rng.normal(size=19)
    valid, constant = mst_core._window_flags(series, query.size)
    assert np.all(valid == 1)
    assert np.all(constant == 0)
    assert np.allclose(
        mst.mass(query, series),
        stumpy.mass(query, series),
        atol=1e-10,
        rtol=1e-10,
    )


def test_mass_direct_parallel_simd_tail(monkeypatch):
    rng = np.random.default_rng(102)
    query = rng.normal(size=17)
    series = rng.normal(size=16_421)
    monkeypatch.setattr(mst_core, "MASS_FFT_WORK_THRESHOLD", float("inf"))
    ours = mst.mass(query, series)
    theirs = stumpy.mass(query, series)
    assert ours.size >= 16_384
    assert (query.size % 2) != 0
    assert np.allclose(ours, theirs, atol=1e-10, rtol=1e-10)


def test_match_parity():
    series = RNG.normal(size=100)
    query = series[20:31].copy()
    ours = mst.match(query, series, query_idx=20, max_matches=5)
    theirs = stumpy.match(query, series, query_idx=20, max_matches=5)
    assert np.allclose(ours[:, 0].astype(float), theirs[:, 0].astype(float), atol=1e-10)
    assert np.array_equal(ours[:, 1].astype(int), theirs[:, 1].astype(int))


@pytest.mark.parametrize("normalize", [True, False])
def test_mpdist_parity(normalize):
    first = RNG.normal(size=50)
    second = RNG.normal(size=65)
    ours = mst.mpdist(first, second, 8, percentage=0.1, normalize=normalize)
    theirs = stumpy.mpdist(first, second, 8, percentage=0.1, normalize=normalize)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_aampdist_parity():
    first = RNG.normal(size=45)
    second = RNG.normal(size=62)
    assert mst.aampdist(first, second, 7, k=4, p=1.0) == pytest.approx(
        stumpy.aampdist(first, second, 7, k=4, p=1.0), abs=1e-10
    )


def test_mparray_properties_and_slice():
    profile = mst.stump(RNG.normal(size=50), 6, k=2)
    assert isinstance(profile, mst.mparray)
    assert profile.P_.dtype == np.float64
    assert profile.I_.dtype == np.int64
    assert isinstance(profile[:5], mst.mparray)
    assert profile[:5].P_.shape == (5, 2)


@pytest.mark.parametrize("name", ["stump", "aamp", "mass", "match", "mpdist", "aampdist"])
def test_covered_signatures_match_upstream(name):
    assert inspect.signature(getattr(mst, name)) == inspect.signature(getattr(stumpy, name))


def test_argument_validation():
    with pytest.raises(ValueError):
        mst.stump(np.arange(10.0), 2)
    with pytest.raises(ValueError):
        mst.stump(np.arange(10.0), 4, k=0)
    with pytest.raises(ValueError):
        mst.mass(np.arange(12.0).reshape(3, 4), np.arange(20.0))
    with pytest.raises(TypeError):
        mst.stump(np.arange(10.0), 3.5)
    with pytest.raises(TypeError):
        mst.stump(np.arange(10.0), 3, k=1.5)
    with pytest.raises(TypeError):
        mst.stump(np.arange(10.0, dtype=np.complex128), 3)
    with pytest.raises(ValueError):
        mst.stump(np.array([2**60, 2**60 + 1, 2**60 + 2]), 3)
    with pytest.raises(ValueError):
        mst.aamp(np.arange(10.0), 3, p=np.nan)


def test_ffi_buffer_preconditions():
    assert addr(np.ones(3, dtype=np.float64)) != 0
    assert addr(np.ones(3, dtype=np.int64)) != 0
    with pytest.raises(TypeError):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        addr(np.ones(6, dtype=np.float64)[::2])
    with pytest.raises(ValueError):
        addr(np.empty(0, dtype=np.float64))
