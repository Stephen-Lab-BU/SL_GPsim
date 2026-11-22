import numpy as np
import pytest
from scipy.signal import csd as scipy_csd, windows

from SL_GPsim import multitaper_csd, simulate_from_csd, spectral_bootstrap


@pytest.mark.parametrize("n,m", [(24, 48), (25, 51)])
def test_delayed_channels_and_scaling(n, m):
    # x is unit white noise. y[t] = 0.7*x[t-2] + independent noise of variance 1.51.
    fs = 80.0
    f = np.fft.rfftfreq(m, 1 / fs)
    s = np.empty((len(f), 2, 2), complex)
    s[:, 0, 0], s[:, 1, 1] = 1 / fs, 2 / fs
    s[:, 1, 0] = 0.7 * np.exp(-2j * np.pi * f * 2 / fs) / fs
    s[:, 0, 1] = s[:, 1, 0].conj()
    s[1:(m + 1) // 2] *= 2
    rng = np.random.default_rng(4)
    x, y = np.stack([simulate_from_csd(s, fs, n, n_fft=m, random_state=rng)
                     for _ in range(700)]).transpose(2, 0, 1)
    np.testing.assert_allclose([np.mean(x*x), np.mean(y*y)], [1, 2], atol=0.06)
    assert np.mean(x[:, :-2] * y[:, 2:]) == pytest.approx(.7, abs=.04)
    assert abs(np.mean(x[:, 2:] * y[:, :-2])) < .04
    assert abs(np.mean(x * y)) < .04


@pytest.mark.parametrize("parity", [0, 1])
def test_dpss_matches_independent_cross_periodograms(parity):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(33, 2))
    m, fs = 66 + parity, 120
    freq, s = multitaper_csd(x, fs, m)
    manual = []
    for taper in windows.dpss(33, 2, 3):
        f, p = scipy_csd(x[:, 1], x[:, 0], window=taper, fs=fs, nperseg=len(x),
                         noverlap=0, nfft=m, detrend="constant", scaling="density")
        manual.append(p)
    np.testing.assert_array_equal(freq, f)
    np.testing.assert_allclose(s[:, 0, 1], np.mean(manual, axis=0), atol=1e-16)
    np.testing.assert_allclose(s, s.conj().swapaxes(1, 2), atol=1e-16)


def test_fitted_embedding_reproduces_linear_cross_covariance_at_every_lag():
    # Independent time-domain oracle for the finite covariance in P&C Section 3.
    n, fs = 13, 90.
    x = np.random.default_rng(14).normal(size=(n, 2))
    _, s = multitaper_csd(x, fs, 2*n, demean=False)
    s[1:-1] /= 2
    lag = fs * np.fft.irfft(s, n=2*n, axis=0)
    h = windows.dpss(n, 2, 3)
    for i in range(n):
        pairs = [(x[i:] * w[i:, None]).T @ (x[:n-i] * w[:n-i, None]) for w in h]
        np.testing.assert_allclose(lag[i], np.mean(pairs, axis=0), atol=1e-14)
    np.testing.assert_allclose(lag[n], 0, atol=1e-14)


def test_no_record_length_wraparound():
    # Sample-length circular synthesis would give covariance .8 at lag N-1.
    n = 12
    rho = .8
    def ar1(f):
        s = (1 - rho**2) / abs(1 - rho * np.exp(-2j*np.pi*f))**2
        s[1:-1] *= 2
        return s
    rng = np.random.default_rng(14)
    x = np.array([simulate_from_csd(ar1, 1, n, n_fft=128, random_state=rng)[:, 0]
                  for _ in range(2000)])
    assert np.mean(x[:, 0] * x[:, -1]) == pytest.approx(rho**(n-1), abs=.065)
    assert .2 < x.mean(axis=1).std() < .85


def test_rank_deficient_and_zero_channels():
    v = np.array([1., 2., 0.])
    s = np.broadcast_to(np.outer(v, v), (19, 3, 3))
    x = simulate_from_csd(s, 10, 18, random_state=5)
    np.testing.assert_allclose(x[:, 1], 2*x[:, 0], atol=1e-11)
    np.testing.assert_array_equal(x[:, 2], 0)
    np.testing.assert_array_equal(x, simulate_from_csd(s, 10, 18, random_state=5))


def test_endpoint_only_densities():
    n, fs = 24, 48
    for index in (0, -1):
        s = np.zeros(n + 1)
        s[index] = 1.
        rng = np.random.default_rng(80)
        rows = np.array([simulate_from_csd(s, fs, n, random_state=rng)[:, 0]
                         for _ in range(800)])
        sign = 1 if index == 0 else (-1.)**np.arange(n)
        np.testing.assert_allclose(rows, rows[:, [0]] * sign * np.ones(n), atol=1e-12)
        assert rows[:, 0].var() == pytest.approx(fs/(2*n), abs=.13)


@pytest.mark.parametrize("matrix,error", [
    ([[1, 2], [2, 1]], "positive semidefinite"),
    ([[1e-30, 2e-15], [2e-15, 1]], "positive semidefinite"),
    ([[1, .3j], [0, 1]], "Hermitian"),
    ([[1, .3j], [-.3j, 1]], "real at DC"),
    ([[1, 0], [0, -1]], "nonnegative"),
    ([[0, .1], [.1, 1]], "zero-power"),
    ([[np.nan]], "nonfinite"),
])
def test_invalid_spectral_matrix(matrix, error):
    s = np.tile(matrix, (17, 1, 1))
    with pytest.raises(ValueError, match=error):
        simulate_from_csd(s, 100, 16)


def test_scalar_and_single_channel_input_identical():
    fs, n = 18., 19
    s = np.arange(n+1, dtype=float)
    kwargs = dict(sampling_rate=fs, n_samples=n, random_state=6)
    np.testing.assert_array_equal(simulate_from_csd(s, **kwargs),
                                  simulate_from_csd(s[:, None, None], **kwargs))


def test_bootstrap_recovers_dependence_between_analyzed_frequency_bins():
    # For tapered white noise, adjacent periodogram bins share information.
    n, fs = 64, 64
    def hann(x, fs, size):
        f, p = scipy_csd(x[:, 0], x[:, 0], fs, window="hann", nperseg=len(x),
                         noverlap=0, nfft=size, detrend=False)
        return f, p
    def selected(f, s):
        return s[[10, 11, 21], 0, 0].real
    white = np.full(n+1, 2/fs)
    white[[0, -1]] /= 2
    x = np.random.default_rng(2).normal(size=n)
    r = spectral_bootstrap(x, fs, selected, pilot_csd=white, estimator=hann,
                           n_resamples=1500, demean=False, random_state=11)
    corr = np.corrcoef(r.draws.T)
    assert corr[0, 1] == pytest.approx(4/9, abs=.09)
    assert abs(corr[0, 2]) < .09
    np.testing.assert_allclose(r.draws.mean(axis=0), 2/fs, rtol=.08)


def test_bootstrap_refits_paired_and_reproducibly_with_custom_estimator():
    n = 27
    x = np.random.default_rng(1).normal(size=(n, 2))
    calls = []
    def estimator(data, fs, size):
        calls.append((data.shape, size))
        return multitaper_csd(data, fs, size, demean=False)
    def pair(f, s):
        a = s[2:6, 0, 1].imag.mean()
        return np.array([a, 3*a])
    kw = dict(n_resamples=25, random_state=2, estimator=estimator)
    a = spectral_bootstrap(x, 10, pair, **kw)
    assert [m for _, m in calls] == [n, 2*n] + [n]*25
    assert all(shape == x.shape for shape, _ in calls)
    b = spectral_bootstrap(x, 10, pair, **kw)
    np.testing.assert_array_equal(a.draws, b.draws)
    np.testing.assert_allclose(a.draws[:, 1], 3*a.draws[:, 0])
    np.testing.assert_allclose(a.estimate, pair(*multitaper_csd(x, 10)))
    np.testing.assert_allclose([a.lower, a.upper], np.quantile(a.draws, [.025, .975], axis=0))


def test_default_intervals_and_missing_peaks():
    x = np.random.default_rng(5).normal(size=32)
    res = spectral_bootstrap(x, 100, n_resamples=12, random_state=5)
    assert res.draws.shape == (12, 17, 1)
    np.testing.assert_allclose(res.estimate, multitaper_csd(x, 100)[1][:, :, 0].real)
    def missing(f, s):
        return [s[0, 0, 0].real, np.nan]
    with pytest.raises(ValueError, match="nonfinite"):
        spectral_bootstrap(x, 100, missing, n_resamples=9)
    r = spectral_bootstrap(x, 100, missing, n_resamples=12, allow_missing=True, random_state=6)
    np.testing.assert_array_equal(r.n_valid, [12, 0])
    assert np.isfinite(r.lower[0]) and np.isnan(r.upper[1])


def test_input_validation_and_visible_fit_failure():
    x = np.arange(20.)
    with pytest.raises(ValueError, match="n_fft"):
        simulate_from_csd(x, 100, 20, n_fft=20)
    with pytest.raises(ValueError, match="shape"):
        simulate_from_csd(x, 100, 20)
    with pytest.raises(ValueError, match="same channels"):
        spectral_bootstrap(x, 100, pilot_csd=np.zeros((21, 2, 2)))
    with pytest.raises(ValueError, match="real"):
        spectral_bootstrap(1j*x, 100)
    with pytest.raises(ValueError, match="rfft grid"):
        spectral_bootstrap(x, 100, estimator=lambda x, fs, m: (np.arange(m), np.ones(m)))
    calls = 0
    def fail(f, s):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("fit did not converge")
        return 1.
    with pytest.raises(RuntimeError, match="replicate 1.*fit did not converge"):
        spectral_bootstrap(x, 100, fail, n_resamples=4, random_state=2)


def test_single_precision_rank_one_input_and_no_mutation():
    rng = np.random.default_rng(42)
    z = (rng.normal(size=(25, 3)) + 1j*rng.normal(size=(25, 3))).astype("complex64")
    z[[0, -1]] = z[[0, -1]].real
    s = z[:, :, None] * z[:, None, :].conj()
    original = s.copy()
    x = simulate_from_csd(s, 80, 24, random_state=7)
    assert x.shape == (24, 3) and np.isfinite(x).all()
    np.testing.assert_array_equal(s, original)


@pytest.mark.parametrize("m", [16, 17])
def test_entire_cropped_covariance_equals_time_domain_moving_average(m):
    """Drive draw() with each Gaussian basis vector to recover its exact covariance."""
    from SL_GPsim.cross_spectral import _EmbeddedGaussian

    fs, n = 80., 8
    # h[channel, innovation, lag] describes a causal two-source MA(2).
    h = np.array([[[1., .4, 0.], [0., .2, 0.]],
                  [[0., 0., .8], [1.1, -.3, 0.]]])
    f = np.fft.rfftfreq(m, 1/fs)
    phase = np.exp(-2j*np.pi*f[:, None]*np.arange(3)/fs)
    H = np.einsum("fL,csL->fcs", phase, h)
    spectrum = H @ H.conj().swapaxes(1, 2) / fs
    spectrum[1:(m+1)//2] *= 2
    model = _EmbeddedGaussian(spectrum, fs, n, m)

    # x[t] = sum_l h[l] u[t-l]. Include two innovations before time zero.
    A = np.zeros((n, 2, n+2, 2))
    for t in range(n):
        for lag in range(3):
            A[t, :, t+2-lag, :] = h[:, :, lag]
    A = A.reshape(n*2, -1)
    expected = A @ A.T

    # The generator is linear in independent N(0,1) coordinates. Cov[x] is
    # the sum of outer products of its responses to the coordinate vectors.
    shape = (2, len(f), 2)  # real/imaginary noise, frequencies, channels
    class BasisNoise:
        def __init__(self, coordinates):
            self.coordinates = iter(coordinates)

        def standard_normal(self, size):
            block = next(self.coordinates)
            assert block.shape == size
            return block

    responses = []
    for e in np.eye(np.prod(shape)).reshape((-1,) + shape):
        responses.append(model.draw(BasisNoise(e)).ravel())
    responses = np.stack(responses)
    np.testing.assert_allclose(responses.T @ responses, expected, rtol=0, atol=2e-13)


def test_bootstrap_multitaper_bandpower_moments_against_gaussian_quadratic_form():
    """Exact finite-N oracle for the mean and variance after tapering and centering."""
    n, fs, k = 32, 32., 3
    draws = 4000
    f0, f1 = 4, 8
    freq = np.fft.rfftfreq(n, 1/fs)
    band = (freq >= f0) & (freq <= f1)
    # Band integration is x.T @ Q @ x. Build Q in time using a cosine sum.
    t = np.arange(n)
    dt = t[:, None] - t[None, :]
    cosine = np.cos(2*np.pi*freq[band, None, None]*dt/fs).sum(axis=0)
    tapers = windows.dpss(n, 2, Kmax=k)
    center = np.eye(n) - np.ones((n, n))/n
    Q = center @ (2/(k*n) * cosine * (tapers.T @ tapers)) @ center
    # For unit-variance Gaussian white noise, E[x.T Q x]=tr(Q) and Var=2tr(Q²).
    target_mean = np.trace(Q)
    target_var = 2*np.trace(Q @ Q)
    kappa4 = 48*np.trace(np.linalg.matrix_power(Q, 4))
    variance_mcse = np.sqrt(kappa4/draws + 2*target_var**2/(draws-1))

    def power(f, csd):
        return csd[(f >= f0) & (f <= f1), 0, 0].real.sum() * fs/n

    pilot = np.full(n+1, 2/fs)
    pilot[[0, -1]] /= 2
    data = np.random.default_rng(17).normal(size=n)
    result = spectral_bootstrap(data, fs, power, pilot_csd=pilot, n_resamples=draws,
                                n_tapers=k, time_halfbandwidth=2, random_state=73)
    assert abs(result.draws.mean()-target_mean) < 5*np.sqrt(target_var/draws)
    assert abs(result.draws.var(ddof=1)-target_var) < 5*variance_mcse
