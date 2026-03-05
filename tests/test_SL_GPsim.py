import numpy as np
import pytest

from SL_GPsim import CombinedSimulator, ParametricDecomposition, spectrum, spectrogram


def test_simulator_determinism():
    """With a fixed random_state the simulator should produce identical draws."""
    # Additive model
    sim1 = CombinedSimulator(
        sampling_rate=200,
        duration=1.0,
        aperiodic_exponent=1.5,
        aperiodic_offset=0.5,
        knee=None,
        peaks=[{"freq": 10.0, "amplitude": 5.0, "sigma": 1.0}],
        average_firing_rate=0.0,
        random_state=42,
        mode="additive",
    )
    sim2 = CombinedSimulator(
        sampling_rate=200,
        duration=1.0,
        aperiodic_exponent=1.5,
        aperiodic_offset=0.5,
        knee=None,
        peaks=[{"freq": 10.0, "amplitude": 5.0, "sigma": 1.0}],
        average_firing_rate=0.0,
        random_state=42,
        mode="additive",
    )
    td1 = sim1.simulate()
    td2 = sim2.simulate()
    # Combined, broadband and rhythmic should be identical
    assert np.allclose(td1.combined_signal, td2.combined_signal)
    assert np.allclose(td1.broadband_signal, td2.broadband_signal)
    assert np.allclose(td1.rhythmic_signal, td2.rhythmic_signal)
    # Multiplicative model
    sim3 = CombinedSimulator(
        sampling_rate=200,
        duration=1.0,
        aperiodic_exponent=1.2,
        aperiodic_offset=0.7,
        knee=5.0,
        peaks=[{"freq": 15.0, "amplitude": 2.0, "sigma": 1.5}],
        average_firing_rate=0.0,
        random_state=99,
        mode="multiplicative",
    )
    sim4 = CombinedSimulator(
        sampling_rate=200,
        duration=1.0,
        aperiodic_exponent=1.2,
        aperiodic_offset=0.7,
        knee=5.0,
        peaks=[{"freq": 15.0, "amplitude": 2.0, "sigma": 1.5}],
        average_firing_rate=0.0,
        random_state=99,
        mode="multiplicative",
    )
    td3 = sim3.simulate()
    td4 = sim4.simulate()
    assert np.allclose(td3.combined_signal, td4.combined_signal)
    assert np.allclose(td3.broadband_signal, td4.broadband_signal)
    assert np.allclose(td3.rhythmic_signal, td4.rhythmic_signal)


def test_frequency_grid_and_decomposition_invariants():
    """Check n_fft sizing and frequency grid/DC handling."""
    fs = 1000.0
    n_samples = 1000
    # CombinedSimulator default target_df=0.01 → required ~ fs/df = 100000
    cs = CombinedSimulator(
        sampling_rate=fs,
        n_samples=n_samples,
        aperiodic_exponent=1.0,
        aperiodic_offset=1.0,
        knee=None,
        peaks=[],
        mode="additive",
    )
    # expected n_fft is next power of two ≥ max(fs/target_df, n_samples)
    required = max(int(np.ceil(fs / 0.01)), n_samples)
    expected_n_fft = 2 ** int(np.ceil(np.log2(required)))
    assert cs.n_fft == expected_n_fft
    # ParametricDecomposition frequency axis properties
    dec = ParametricDecomposition(
        sampling_rate=fs,
        n_fft=1024,
        aperiodic_exponent=1.5,
        aperiodic_offset=0.0,
        knee=None,
        peaks=[{"freq": 20.0, "amplitude": 10.0, "sigma": 3.0}],
    )
    fd = dec.compute()
    # Length matches n_fft
    assert len(fd.frequencies) == 1024
    # Unshifted ordering: first element is DC (0), second positive, last negative
    assert fd.frequencies[0] == pytest.approx(0.0)
    assert fd.frequencies[1] > 0
    assert fd.frequencies[-1] < 0
    # DC bins of components should be exactly zero
    zero_idx = np.argmin(np.abs(fd.frequencies))
    assert fd.broadband_spectrum[zero_idx] == 0.0
    assert fd.rhythmic_spectrum[zero_idx] == 0.0


def test_additive_theoretical_correctness():
    """In additive mode the combined spectrum should equal broadband+rhythmic."""
    res = spectrum(
        sampling_rate=200,
        duration=2.0,
        aperiodic_exponent=1.0,
        aperiodic_offset=0.0,
        knee=None,
        peaks=[{"freq": 25.0, "amplitude": 10.0, "sigma": 2.0}],
        direct_estimate=False,
        random_state=0,
        mode="additive",
    )
    fd = res.frequency_domain
    # Combined spectrum approximately equals sum of broadband and rhythmic components
    combined = fd.combined_spectrum
    broadband = fd.broadband_spectrum
    rhythmic = fd.rhythmic_spectrum
    assert broadband is not None and rhythmic is not None
    assert np.allclose(combined, broadband + rhythmic)
    # Peak should stand out at f0 relative to baseline
    freqs = fd.frequencies
    pos = freqs > 0
    freqs_pos = freqs[pos]
    comb_pos = combined[pos]
    f0 = 25.0
    idx0 = np.argmin(np.abs(freqs_pos - f0))
    # Compare to spectrum at f0 + 3*sigma
    idx_off = np.argmin(np.abs(freqs_pos - (f0 + 3 * 2.0)))
    assert comb_pos[idx0] > comb_pos[idx_off]


def test_multiplicative_theoretical_correctness():
    """In multiplicative mode the theoretical PSD should match baseline*10**G."""
    # Use a simple configuration
    fs = 300.0
    aexp = 1.5
    aoff = 0.3
    peaks = [{"freq": 30.0, "amplitude": 1.0, "sigma": 3.0}]
    res = spectrum(
        sampling_rate=fs,
        duration=1.0,
        aperiodic_exponent=aexp,
        aperiodic_offset=aoff,
        knee=None,
        peaks=peaks,
        direct_estimate=False,
        random_state=1,
        mode="multiplicative",
    )
    fd = res.frequency_domain
    freqs = fd.frequencies
    # Compute expected baseline and bump following the implementation in spectrum()
    kappa = 0.0
    # Note: the implementation uses f**exponent (not |f|**exponent); this can
    # produce NaNs for negative frequencies when exponent is noninteger.
    with np.errstate(divide="ignore", invalid="ignore"):
        L = aoff - np.log10(kappa + np.power(freqs, aexp))
    # Set DC bin to -inf to zero it out after exponentiation
    if freqs.size and freqs[0] == 0.0:
        L[0] = -np.inf
    G = np.zeros_like(freqs)
    for pk in peaks:
        c = pk["freq"]
        a_log = pk["amplitude"]
        sigma = pk["sigma"]
        # Only positive bump is added in the implementation
        G += a_log * np.exp(-0.5 * ((freqs - c) / sigma) ** 2)
    P_expected = np.power(10.0, L + G)
    # Force DC bin to zero
    if freqs.size and freqs[0] == 0.0:
        P_expected[0] = 0.0
    # Compare ignoring NaNs (NaNs arise from negative frequencies)**
    mask = ~np.isnan(fd.combined_spectrum)
    assert np.allclose(fd.combined_spectrum[mask], P_expected[mask])
    # broadband and rhythmic components should be None or zero
    assert fd.broadband_spectrum is None or np.all(fd.broadband_spectrum == 0)
    assert fd.rhythmic_spectrum is None or np.all(fd.rhythmic_spectrum == 0)


def test_empirical_vs_theoretical_agreement():
    """Empirical multitaper PSD should correlate with theoretical PSD (log space)."""
    pytest.importorskip("spectral_connectivity")
    res = spectrum(
        sampling_rate=200,
        duration=2.0,
        aperiodic_exponent=1.2,
        aperiodic_offset=0.5,
        knee=None,
        peaks=[{"freq": 15.0, "amplitude": 5.0, "sigma": 2.0}],
        direct_estimate=True,
        random_state=2,
        mode="additive",
    )
    fd = res.frequency_domain
    theo = fd.combined_spectrum
    emp = fd.empirical_spectrum
    freqs = fd.frequencies
    # Restrict to positive frequencies excluding DC and Nyquist
    pos = (freqs > 1e-6) & (freqs < 0.5 * 200)
    theo_log = np.log10(theo[pos])
    emp_log = np.log10(emp[pos])
    # Compute correlation coefficient
    corr = np.corrcoef(theo_log, emp_log)[0, 1]
    assert corr > 0.7
    # Median absolute log error should be reasonable
    mae = np.median(np.abs(theo_log - emp_log))
    assert mae < 0.8


def test_broadband_slope_sanity():
    """Slope of log10(PSD) vs log10(f) should match the negative exponent."""
    fs = 300.0
    exponent = 2.0
    dec = ParametricDecomposition(
        sampling_rate=fs,
        n_fft=4096,
        aperiodic_exponent=exponent,
        aperiodic_offset=0.0,
        knee=None,
        peaks=[],
    )
    fd = dec.compute()
    freqs = fd.frequencies
    pos = (freqs > 1.0) & (freqs < 0.4 * fs)
    f_pos = freqs[pos]
    psd_pos = fd.combined_spectrum[pos]
    # Avoid zeros
    mask = psd_pos > 0
    f_pos = f_pos[mask]
    psd_pos = psd_pos[mask]
    logf = np.log10(f_pos)
    logp = np.log10(psd_pos)
    # Fit slope via least squares
    x = logf - logf.mean()
    y = logp - logp.mean()
    slope = (x * y).sum() / (x * x).sum()
    assert np.isclose(slope, -exponent, atol=0.2)


def test_invalid_mode_raises():
    """Supplying an unknown mode should raise a ValueError."""
    with pytest.raises(ValueError):
        CombinedSimulator(
            sampling_rate=200,
            duration=1.0,
            aperiodic_exponent=1.0,
            aperiodic_offset=1.0,
            knee=None,
            peaks=[],
            mode="invalid",
        )


def _n_windows_for_duration(duration: float, window: float, step: float) -> int:
    """
    Old intent: total duration in seconds.
    New API: total length is T = (n_windows - 1)*step + window.
    """
    n = int(round((duration - window) / step)) + 1
    if n < 1:
        raise ValueError("duration must be >= window_duration for this helper.")
    return n


def test_spectrogram_no_boundary_blow_up():
    """Check that overlap-add produces smooth boundaries between windows."""
    fs = 100.0
    duration = 2.0
    window = 0.5
    step = 0.25

    n_windows = _n_windows_for_duration(duration, window, step)

    res = spectrogram(
        sampling_rate=fs,
        n_windows=n_windows,
        window_duration=window,
        step_duration=step,
        aperiodic_exponent=1.0,
        aperiodic_offset=0.0,
        knee=None,
        peaks=[{"freq": 5.0, "amplitude": 2.0, "sigma": 1.0}],
        average_firing_rate=0.0,
        random_state=123,
        mode="additive",
        direct_estimate=False,
    )

    td = res.time_domain
    signal = td.combined_signal

    # Sanity: stitched length matches implied duration
    implied_duration = (n_windows - 1) * step + window
    assert len(signal) == int(round(implied_duration * fs))

    diff = np.abs(np.diff(signal))

    n_window = int(round(window * fs))
    n_step = int(round(step * fs))
    starts = np.arange(0, n_windows * n_step, n_step, dtype=int)

    boundary_idx = starts[1:]  # start of each window after the first
    boundary_diffs = np.abs(signal[boundary_idx] - signal[boundary_idx - 1])

    median_diff = np.median(diff)
    assert np.median(boundary_diffs) < 5 * median_diff


def test_spectrogram_time_varying_rhythm():
    """A time-varying peak amplitude should be reflected in the theoretical PSD across windows."""
    fs = 100.0
    duration = 3.0
    window = 0.5
    step = 0.25

    n_windows = _n_windows_for_duration(duration, window, step)

    f0 = 10.0
    base_amp = 2.0

    def peaks_fn(t):
        amp = base_amp * (1.0 + 0.5 * np.sin(2 * np.pi * t / duration))
        return [{"freq": f0, "amplitude": amp, "sigma": 1.0}]

    res = spectrogram(
        sampling_rate=fs,
        n_windows=n_windows,
        window_duration=window,
        step_duration=step,
        aperiodic_exponent=1.0,
        aperiodic_offset=0.0,
        knee=None,
        peaks_fn=peaks_fn,
        average_firing_rate=0.0,
        random_state=7,
        mode="additive",
        direct_estimate=False,
    )

    window_times = res.window_times
    theoretical = res.theoretical["combined"]
    broadband = res.theoretical["broadband"]
    freqs = res.frequencies

    pos = freqs > 0
    freqs_pos = freqs[pos]
    idx_f0 = int(np.argmin(np.abs(freqs_pos - f0)))

    peak_powers = theoretical[:, pos][:, idx_f0]
    peak_baseline = broadband[:, pos][:, idx_f0]
    peak_excess = peak_powers - peak_baseline

    expected_amp = base_amp * (1.0 + 0.5 * np.sin(2 * np.pi * window_times / duration))

    corr = np.corrcoef(expected_amp, peak_excess)[0, 1]
    assert corr > 0.8


def test_spectrogram_params_list_length():
    """params_per_window length should equal number of windows and include resolved values."""
    fs = 50.0
    duration = 1.0
    window = 0.4
    step = 0.2

    n_windows = _n_windows_for_duration(duration, window, step)

    exponents = [1.0, 1.5, 2.0]
    res = spectrogram(
        sampling_rate=fs,
        n_windows=n_windows,
        window_duration=window,
        step_duration=step,
        aperiodic_exponent_fn=exponents,
        aperiodic_offset=0.0,
        knee=None,
        peaks=[],
        random_state=0,
        mode="additive",
        direct_estimate=False,
    )

    assert len(res.window_times) == len(res.params_per_window)

    used = [p["aperiodic_exponent"] for p in res.params_per_window]
    expected = exponents + [exponents[-1]] * (len(res.window_times) - len(exponents))
    assert np.allclose(used, expected)