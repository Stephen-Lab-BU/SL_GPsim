import warnings

import numpy as np
import pytest

from SL_GPsim import SpectralGrid, SpectralWarning, multitaper_csd, simulate_from_csd, simulate_like


@pytest.mark.parametrize("m", [64, 65])
def test_white_density_integrates_to_target_variance(m):
    fs, n = 32, 32
    grid = SpectralGrid(fs, n, n_fft=m)
    psd = grid.density(lambda f: 2 / fs)
    assert psd[0] == 1 / fs
    assert psd[-1] == (1 / fs if m % 2 == 0 else 2 / fs)
    assert psd.sum() * fs/m == pytest.approx(1)
    np.testing.assert_array_equal(grid.frequencies, np.fft.rfftfreq(m, 1/fs))
    np.testing.assert_array_equal(grid.simulate(psd, random_state=7),
                                  simulate_from_csd(psd, fs, n, n_fft=m, random_state=7))


def test_knee_conventions_and_arithmetic():
    g = SpectralGrid(128, 1024)
    by_hz = g.broadband(scale=8, knee_hz=2, exponent=2)
    by_k = g.broadband(scale=8, knee=4, exponent=2)
    np.testing.assert_array_equal(by_hz, by_k)
    assert by_hz[g.frequencies == 2].item() == (2 * by_hz[0]) / 2
    beta = g.peak(center_hz=20, sigma_hz=1.8, height=.7)
    assert beta[g.frequencies == 20].item() == .7
    assert (by_k + beta)[0] == pytest.approx(by_k[0])


@pytest.mark.parametrize("m", [128, 129])
def test_three_channel_sources_match_independent_transfer_matrix(m):
    g = SpectralGrid(128, 64, n_fft=m)
    beta = g.peak(center_hz=20, sigma_hz=2, height=.7)
    noise = g.density(lambda f: .03)
    gains = np.array([1., .7, -.4])
    delay = np.array([0, 2, -3])
    csd = g.independent(noise, noise*2, noise*0) + g.shared(beta, gains=gains, delays_samples=delay)
    h = gains[None, :] * np.exp(-2j*np.pi*g.frequencies[:, None]*delay/128)
    expected = beta[:, None, None] * h[:, :, None] * h[:, None, :].conj()
    expected[:, 0, 0] += noise
    expected[:, 1, 1] += noise*2
    np.testing.assert_allclose(csd, expected, atol=1e-15)
    assert g.simulate(csd, random_state=2).shape == (64, 3)


def test_raw_endpoint_validation():
    g = SpectralGrid(128, 64)
    s = g.independent(g.density(lambda f: 1), g.density(lambda f: 1))
    s[-1, 0, 1], s[-1, 1, 0] = 0.2j, -0.2j
    with pytest.raises(ValueError, match=r"real at Nyquist \(64 Hz\)"):
        g.simulate(s)
    s[-1, 0, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite spectral density at Nyquist"):
        g.simulate(s)
    with pytest.raises(ValueError, match="nonfinite spectral density at DC"):
        g.simulate(np.full(len(s), np.inf))


def test_fractional_delay_at_even_nyquist_and_odd_last_bin():
    gains, delays = np.array([1., .7, .5]), np.array([0., .5, 1.25])
    g = SpectralGrid(64, 32)
    white = g.density(lambda f: 1)
    with pytest.warns(SpectralWarning, match="Fractional relative delay at Nyquist") as caught:
        s = g.shared(white, gains=gains, delays_samples=delays)
    assert len(caught) == 1
    expected = white[-1] * np.outer(gains, gains) * np.cos(np.pi*(delays[:, None]-delays))
    np.testing.assert_allclose(s[-1], expected, atol=1e-15)
    assert np.linalg.eigvalsh(s[-1]).min() >= -1e-14
    assert not s[-1].imag.any()
    assert g.simulate(s, random_state=3).shape == (32, 3)
    odd = SpectralGrid(64, 32, n_fft=65)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpectralWarning)
        s = odd.shared(odd.density(lambda f: 1), gains=gains, delays_samples=delays)
        g.shared(white, gains=[1, -.4], delays_samples=[.5, 1.5])
        g.shared(white, gains=[1, 0], delays_samples=[0, .5])
        white[-1] = 0
        g.shared(white, gains=gains, delays_samples=delays)
    np.testing.assert_allclose(s[-1, 1, 0], .7*np.exp(-2j*np.pi*odd.frequencies[-1]*.5/64))
    assert odd.simulate(s, random_state=3).shape == (32, 3)


@pytest.mark.parametrize("center,width,message", [(1, 1, "DC"), (31, 1, "Nyquist"),
                                                    (16, .1, "Increase n_fft")])
def test_peak_warnings(center, width, message):
    g = SpectralGrid(64, 64)
    with pytest.warns(SpectralWarning, match=message):
        psd = g.peak(center_hz=center, sigma_hz=width, height=1)
    assert np.isfinite(psd).all()


@pytest.mark.parametrize("knee_hz,message", [(0.1, "below the first"), (50, "above Nyquist")])
def test_unresolved_knee(knee_hz, message):
    g = SpectralGrid(64, 64)
    with pytest.warns(SpectralWarning, match=message):
        g.broadband(scale=1, exponent=2, knee_hz=knee_hz)


def test_invalid_component_inputs_and_grid():
    g = SpectralGrid(64, 64)
    with pytest.raises(ValueError, match="n_fft"):
        SpectralGrid(64, 64, n_fft=64)
    with pytest.raises(ValueError, match="exactly one"):
        g.broadband(scale=1, exponent=1, knee=2, knee_hz=2)
    with pytest.raises(ValueError, match="at DC"):
        g.broadband(scale=1, exponent=1, knee=0)
    with pytest.raises(ValueError, match="at DC"):
        g.density(lambda f: 1/f)
    with pytest.raises(ValueError, match="Nyquist"):
        g.peak(center_hz=33, height=1, sigma_hz=2)
    with pytest.raises(ValueError, match="real"):
        g.density(lambda f: 1+1j)
    with pytest.raises(TypeError, match="formula"):
        g.density(np.ones(65))
    p = g.density(lambda f: 1)
    with pytest.raises(ValueError, match="length"):
        g.independent(p, p[1:])
    with pytest.raises(ValueError, match="per channel"):
        g.shared(p, gains=[1, .7], delays_samples=[0])
    with pytest.raises(ValueError, match="real"):
        g.shared(p, gains=[1j])
    with pytest.raises(ValueError, match="finite"):
        g.shared(p, gains=[np.nan])


@pytest.mark.parametrize("channels,m", [(1, 128), (3, 131)])
def test_simulate_like_preserves_estimator_scaling_and_settings(channels, m):
    fs, n = 64, 64
    x = np.random.default_rng(3).normal(size=(n, channels)) + 2
    if channels == 1:
        x = x[:, 0]
    settings = dict(n_fft=m, n_tapers=2, time_halfbandwidth=1.5, demean=False)
    _, s = multitaper_csd(x, fs, **settings)
    expected = simulate_from_csd(s, fs, n, n_fft=m, random_state=42)
    np.testing.assert_array_equal(simulate_like(x, fs, random_state=42, **settings), expected)
