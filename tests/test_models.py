import json
import pickle

import numpy as np
import pytest

from SL_GPsim import (
    Broadband, GaussianPeak, Independent, LogPeaks, Shared, SpectralGrid, SpectralWarning,
    model_from_parameters, multitaper_csd, simulate_from_csd, simulate_like, spectral_bootstrap,
)


def test_times_in_seconds_and_legacy_count():
    grid = SpectralGrid(fs=128, duration=8)
    from_count = SpectralGrid(sampling_rate=128, n_samples=1024)
    assert grid.n_samples == 1024 and grid.fs == 128 and grid.duration == 8
    assert grid.times[0] == 0 and grid.times[-1] == 8 - 1/128
    np.testing.assert_array_equal(grid.frequencies, from_count.frequencies)
    with pytest.warns(SpectralWarning, match="using 13 samples"):
        rounded = SpectralGrid(fs=10, duration=1.25)
    assert rounded.duration == 1.3
    assert SpectralGrid(fs=100, duration=.29).n_samples == 29


@pytest.mark.parametrize("kwargs", [
    dict(fs=16, duration=.01), dict(fs=10, duration=0), dict(fs=10, duration=np.nan),
    dict(fs=10, duration=-1), dict(fs=10, duration=1, n_samples=10),
    dict(fs=10, sampling_rate=10, duration=1), dict(fs=10), dict(duration=1),
])
def test_invalid_recording_settings(kwargs):
    with pytest.raises(ValueError):
        SpectralGrid(**kwargs)


def test_scalar_irregular_and_dense_evaluations_are_independent_of_recording():
    bg = Broadband(scale=8, exponent=2, knee_hz=2)
    peak = GaussianPeak(center_hz=12, sigma_hz=1.5, height=.7)
    freq = np.array([12, 0, 200, 11.8, 2, 12])
    expected_bg = 8 / (4 + freq**2)
    expected_peak = .7*np.exp(-.5*((freq-12)/1.5)**2)
    model = bg + peak
    np.testing.assert_allclose(model(freq), expected_bg + expected_peak)
    assert bg(0) == 2 and bg(2) == 1 and peak(12) == .7
    assert model(np.linspace(0, 300, 17301)).shape == (17301,)
    assert model([]).shape == (0,)
    with pytest.raises(ValueError, match="frequencies"):
        model([-1])
    with pytest.raises(ValueError, match="frequencies"):
        model([1+1j])


@pytest.mark.parametrize("n_fft", [128, 129])
def test_multiplicative_form_is_weighted_once_after_composition(n_fft):
    grid = SpectralGrid(fs=64, duration=1, n_fft=n_fft)
    bg = Broadband(scale=8, exponent=2, knee=4)
    peak = GaussianPeak(center_hz=12, sigma_hz=3, height=.8)
    model = LogPeaks(bg, peak)
    expected = (8 / (4 + grid.frequencies**2)
                * 10**(.8*np.exp(-.5*((grid.frequencies-12)/3)**2)))
    np.testing.assert_allclose(model(grid.frequencies), expected)
    expected[0] /= 2
    if n_fft % 2 == 0:
        expected[-1] /= 2
    np.testing.assert_allclose(grid.sample(model), expected)
    for raw in (grid.sample(model), model):
        np.testing.assert_array_equal(simulate_from_csd(raw, fs=64, duration=1, n_fft=n_fft, random_state=17),
                                      grid.simulate(model, random_state=17))


def test_ordinary_callable_keeps_preweighted_simulator_convention():
    grid = SpectralGrid(fs=64, duration=1)
    m = Broadband(scale=8, exponent=2, knee=4)
    prepared = grid.sample(m)
    np.testing.assert_array_equal(grid.simulate(m, random_state=14),
                                  simulate_from_csd(lambda f: prepared, 64, 64, random_state=14))


def test_shared_sources_use_physical_delays_and_only_project_fft_nyquist():
    bg = Broadband(scale=8, exponent=2, knee=4)
    delays = [0., 1/128]
    s = Shared(bg, gains=[1, .5], delays_seconds=delays)
    delays[1] = 0
    assert s.delays_seconds[1] == 1/128
    f = [32, 3.4, 32]
    expected = .5 * bg(f) * np.exp(-2j*np.pi*np.array(f)/128)
    np.testing.assert_allclose(s(f)[:, 1, 0], expected)
    np.testing.assert_array_equal(s(32), s([32])[0])
    for fs in [64, 256]:
        g = SpectralGrid(fs=fs, duration=2)
        if fs == 64:
            with pytest.warns(SpectralWarning, match="Fractional relative delay"):
                prepared = g.sample(s)
        else:
            prepared = g.sample(s)
        n = np.flatnonzero(g.frequencies == 32)[0]
        if fs == 64:
            assert prepared[n, 1, 0].imag == 0
            assert prepared[n, 1, 0].real == pytest.approx(expected[0].real/2)
        else:
            np.testing.assert_allclose(prepared[n, 1, 0], expected[0])
    assert s([]).shape == (0, 2, 2)


def test_export_original_parameters_and_restore_all_nested_forms():
    bg = Broadband(scale=np.float64(8), exponent=1.6, knee=2)
    alpha = GaussianPeak(center_hz=10, sigma_hz=1, height=.5)
    beta = GaussianPeak(center_hz=20, sigma_hz=2, height=.7)
    model = Independent(bg + alpha, LogPeaks(bg, alpha)) + Shared(beta, gains=[1, -.7], delays_seconds=[0, .016])
    payload = json.loads(json.dumps(model.parameters, allow_nan=False))
    original = payload["left"]["channels"][0]["left"]
    assert original == dict(form="Broadband", scale=8, exponent=1.6, knee=2, knee_hz=None)
    f = [0., .2, 33.7, 18]
    for restored in (model_from_parameters(payload), pickle.loads(pickle.dumps(model))):
        np.testing.assert_array_equal(restored(f), model(f))
        assert restored.parameters == model.parameters
    payload["right"]["gains"][0] = 19
    assert model.parameters["right"]["gains"][0] == 1


def test_model_pilots_and_rate_keyword_in_measured_pipeline():
    fs, duration = 32, 2
    model = Broadband(scale=2, exponent=1.6, knee=3)
    grid = SpectralGrid(fs=fs, duration=duration)
    x = simulate_from_csd(model, fs=fs, duration=duration, random_state=11)
    np.testing.assert_array_equal(simulate_like(x, fs=fs, random_state=12),
                                  simulate_like(x, fs, random_state=12))
    for a,b in zip(multitaper_csd(x, fs=fs), multitaper_csd(x, fs)):
        np.testing.assert_array_equal(a, b)
    kwargs = dict(n_resamples=9, random_state=33)
    actual = spectral_bootstrap(x, fs=fs, pilot_csd=model, **kwargs)
    raw = spectral_bootstrap(x, fs, pilot_csd=grid.sample(model), **kwargs)
    np.testing.assert_array_equal(actual.draws, raw.draws)


def test_validation_on_composition_and_recording_band():
    bg = Broadband(scale=8, exponent=2, knee=4)
    peak = GaussianPeak(center_hz=40, sigma_hz=1, height=.7)
    assert peak(40) == .7
    with pytest.raises(ValueError, match="Nyquist"):
        SpectralGrid(fs=64, duration=2).simulate(bg + peak)
    with pytest.raises(ValueError, match="matching channel"):
        (Independent(bg) + bg)([0, 1])
    with pytest.raises(ValueError, match="univariate"):
        Shared(Independent(bg), gains=[.5])([0, 1])
    with pytest.raises(ValueError, match="unknown spectral form"):
        model_from_parameters({"form": "missing"})
