"""
Spectrogram simulation module for SL_GPsim.

This module defines a helper to simulate a series of windowed signals whose
spectral parameters may change over time.  It returns both the stitched
time‑domain signal (constructed via overlap‑add of tapered windows) and
per‑window theoretical and optional empirical power spectral densities (PSDs).

The API mirrors the high‑level ``spectrum`` function but introduces
``window_duration`` and ``step_duration`` to specify the spectrogram windows
and hop size.  Parameters may be passed as constants or as functions of
time (callables) to enable slow evolution of the 1/f exponent, offset,
knee and spectral peaks.  Alternatively, per‑window lists/arrays of
parameter values may be provided.  See ``spectrogram`` docstring for
details.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Union

from .simulation import CombinedSimulator
from .decomposition import ParametricDecomposition
from .time_domain import TimeDomainData
from .frequency_domain import FrequencyDomainData


@dataclass
class SpectrogramResult:
    """Container for spectrogram simulation outputs.

    Attributes
    ----------
    time_domain : TimeDomainData
        The stitched time‑domain signal including broadband and rhythmic parts.
    window_times : np.ndarray
        Array of center times (seconds) for each window.
    frequencies : np.ndarray
        The frequency grid (Hz) used for theoretical PSDs (common to all windows).
    theoretical : dict
        Dictionary with keys ``'combined'``, ``'broadband'`` and
        ``'rhythmic'`` whose values are arrays of shape ``(n_windows, n_freqs)``.
        For multiplicative mode the broadband and rhythmic entries will be zero
        arrays (since the decomposition into components is ill‑defined).
    empirical : np.ndarray or None
        Optional array of shape ``(n_windows, n_freqs_emp)`` containing the
        multitaper PSD for each window when ``direct_estimate=True``.  ``None``
        when ``direct_estimate=False``.
    frequencies_empirical : np.ndarray or None
        Frequency grid (Hz) for the empirical multitaper spectra.  ``None`` when
        ``direct_estimate=False``.
    params_per_window : list of dict
        A list containing the parameter values used for each window.
    windowed_time : np.ndarray
        The sample times (seconds) within a single window of length
        ``window_duration``.  Shape is ``(n_window,)``.
    windowed_combined : np.ndarray
        Raw combined signals for each window before stitching.  Shape
        ``(n_windows, n_window)``.
    windowed_broadband : np.ndarray
        Raw broadband signals for each window before stitching.  Shape
        ``(n_windows, n_window)``.
    windowed_rhythmic : np.ndarray
        Raw rhythmic signals for each window before stitching.  Shape
        ``(n_windows, n_window)``.
    """

    time_domain: TimeDomainData
    window_times: np.ndarray
    frequencies: np.ndarray
    theoretical: Dict[str, np.ndarray]
    empirical: Optional[np.ndarray]
    frequencies_empirical: Optional[np.ndarray]
    params_per_window: List[Dict[str, Any]]
    windowed_time: np.ndarray
    windowed_combined: np.ndarray
    windowed_broadband: np.ndarray
    windowed_rhythmic: np.ndarray


def _get_value_for_window(spec, idx: int, t: float, default):
    """Resolve a potentially time‑varying parameter for a specific window.

    Parameters
    ----------
    spec : any
        The parameter specification.  If callable it will be invoked with
        the window centre time ``t``.  If it is a list or numpy array, the
        element at position ``idx`` will be returned.  Otherwise the
        specification itself is returned.
    idx : int
        Window index used for indexing lists/arrays.
    t : float
        Window centre time passed to callables.
    default : any
        Value to return when spec is ``None``.

    Returns
    -------
    any
        The resolved value for this window.
    """
    if spec is None:
        return default
    # Callable functions take precedence.  Call with signature
    # (idx, t) if the callable accepts two parameters, otherwise
    # call with a single argument ``t``.
    if callable(spec):
        try:
            import inspect  # local import to avoid global import overhead
            sig = inspect.signature(spec)
            if len(sig.parameters) >= 2:
                return spec(idx, t)
            else:
                return spec(t)
        except Exception:
            # Fallback: call with t
            try:
                return spec(t)
            except Exception:
                return default
    # Indexable containers (list/array)
    if isinstance(spec, (list, np.ndarray)):
        if len(spec) == 0:
            return default
        # Bound the index to the range
        j = idx if idx < len(spec) else len(spec) - 1
        return spec[j]
    # Otherwise use as constant
    return spec


def spectrogram(
    sampling_rate: float,
    n_windows: int,
    window_duration: float,
    step_duration: Optional[float] = None,
    *,
    aperiodic_exponent: float = 1.0,
    aperiodic_offset: float = 1.0,
    knee: Optional[float] = None,
    peaks: Optional[List[Dict[str, Any]]] = None,
    average_firing_rate: float = 0.0,
    random_state: Optional[int] = None,
    mode: str = "additive",
    direct_estimate: bool = False,
    aperiodic_exponent_fn: Optional[Union[Callable[[int, float], float], List[float], np.ndarray]] = None,
    aperiodic_offset_fn: Optional[Union[Callable[[int, float], float], List[float], np.ndarray]] = None,
    knee_fn: Optional[Union[Callable[[int, float], Optional[float]], List[Optional[float]], np.ndarray]] = None,
    peaks_fn: Optional[Union[Callable[[int, float], List[Dict[str, Any]]], List[List[Dict[str, Any]]], np.ndarray]] = None,
    **mt_kwargs: Any,
) -> SpectrogramResult:
    """Simulate a piecewise‑stationary spectrogram with optional time‑varying
    spectral parameters.

    This function generates ``n_windows`` stationary segments (windows) of
    length ``window_duration`` seconds.  For each window a new
    :class:`CombinedSimulator` instance is created with parameters that may
    vary across windows according to user‑supplied callables or lists.  The
    resulting time‑domain windowed signals are concatenated (or overlap‑added
    when ``step_duration`` is less than ``window_duration``) to form a single
    continuous signal.  Window boundaries are handled via simple overlap‑add
    with rectangular weights; no Hann taper is used so that the underlying
    stationary character of each window is preserved.  When overlap is used,
    a weight envelope is accumulated and the final stitched signal is divided
    by this envelope to avoid amplitude modulation.

    The function returns the stitched time‑domain signal, per‑window
    theoretical power spectra and, optionally, multitaper empirical spectra
    computed via :mod:`spectral_connectivity`.  Additionally, the raw
    per‑window signals and the empirical frequency grid are returned for
    accurate comparison of theoretical and empirical spectra.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.
    n_windows : int
        Number of stationary windows to simulate.  Must be >= 1.
    window_duration : float
        Length of each window in seconds.  Must be > 0.
    step_duration : float, optional
        Hop size between successive windows in seconds.  When ``None`` (the
        default) non‑overlapping concatenation is used, i.e. ``step_duration``
        is set equal to ``window_duration``.  When provided, must be > 0,
        <= ``window_duration`` and such that the implied number of samples
        divides evenly into the total length of the simulated signal.
    aperiodic_exponent : float, optional
        Baseline 1/f exponent used when no time‑varying function is
        supplied.  Ignored when ``aperiodic_exponent_fn`` or an array is
        provided.
    aperiodic_offset : float, optional
        Baseline log10 offset for the broadband 1/f component.  Ignored when
        ``aperiodic_offset_fn`` or an array is provided.
    knee : float or None, optional
        Knee parameter of the aperiodic component.  Interpreted as zero
        when ``None``.  Ignored when ``knee_fn`` or an array is provided.
    peaks : list of dict or None, optional
        List of Gaussian peak dictionaries with keys ``'freq'``,
        ``'amplitude'`` and ``'sigma'``.  Used when ``peaks_fn`` is not
        supplied.  For multiplicative mode the amplitudes are interpreted
        as log10 heights.
    average_firing_rate : float, optional
        Constant offset added to the time‑domain signal.
    random_state : int or None, optional
        Seed for reproducibility.  When provided, each window receives a
        different but deterministic seed derived from this value.
    mode : {"additive", "multiplicative"}, optional
        Model type for constructing the broadband and rhythmic components.
    direct_estimate : bool, optional
        If ``True`` and the optional dependency ``spectral_connectivity``
        is installed, compute multitaper PSDs for each window.  Raises an
        ImportError if the dependency is absent.
    aperiodic_exponent_fn, aperiodic_offset_fn, knee_fn, peaks_fn : callable
        or array or list, optional
        Functions mapping window centre time (seconds) to the corresponding
        parameter value for that window.  Alternatively, arrays/lists of
        length equal to the number of windows may be provided.  If None,
        the corresponding baseline value is used.

    Returns
    -------
    SpectrogramResult
        A dataclass containing the stitched time‑domain data, per‑window
        theoretical and optional empirical spectra, the empirical frequency grid,
        raw per‑window signals and the parameters used for each window.
    """
    # Validate input lengths and convert durations to sample counts
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1.")
    if window_duration <= 0:
        raise ValueError("window_duration must be positive.")
    # Determine step_duration: default to non‑overlap
    step_duration = window_duration if step_duration is None else step_duration
    if step_duration <= 0:
        raise ValueError("step_duration must be positive.")
    if step_duration > window_duration:
        raise ValueError("step_duration must be <= window_duration.")
    if mode not in ("additive", "multiplicative"):
        raise ValueError(f"Unknown mode: {mode!r}")

    # Convert durations to integer sample counts
    n_window = int(round(window_duration * sampling_rate))
    n_step = int(round(step_duration * sampling_rate))
    if n_window < 1 or n_step < 1:
        raise ValueError("window_duration and step_duration are too short for the given sampling_rate.")
    if n_step > n_window:
        raise ValueError("step_duration must be <= window_duration when converted to samples.")

    # Determine total number of samples in the stitched signal
    # The final window begins at (n_windows-1) * n_step and has length n_window
    n_total = (n_windows - 1) * n_step + n_window

    # Construct start indices to ensure exactly n_windows windows
    starts = np.arange(0, n_windows * n_step, n_step, dtype=int)

    # Check that last window fits within n_total
    if starts[-1] + n_window > n_total:
        raise ValueError("The combination of n_windows and step_duration does not allow windows to fit; adjust durations or n_windows.")

    # Prepare deterministic seeds for each window
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    seeds = rng.randint(0, 2**32 - 1, size=n_windows)

    # Initialise CombinedSimulator once to get the FFT length; ensure n_fft large enough
    dummy_simulator = CombinedSimulator(
        sampling_rate=sampling_rate,
        n_samples=n_window,
        aperiodic_exponent=aperiodic_exponent,
        aperiodic_offset=aperiodic_offset,
        knee=knee,
        peaks=peaks,
        average_firing_rate=average_firing_rate,
        random_state=random_state,
        mode=mode,
    )
    n_fft = dummy_simulator.n_fft
    # Frequency grid for PSDs
    freq_grid = np.fft.fftfreq(n_fft, d=1.0 / sampling_rate)

    # Allocate arrays for theoretical PSDs
    the_combined = np.zeros((n_windows, n_fft), dtype=float)
    the_broadband = np.zeros((n_windows, n_fft), dtype=float)
    the_rhythmic = np.zeros((n_windows, n_fft), dtype=float)
    # Prepare empirical storage lazily; frequency grid for empirical PSDs will be
    # determined when the first PSD is computed.
    empirical_list: List[np.ndarray] | None = [] if direct_estimate else None
    frequencies_empirical: Optional[np.ndarray] = None

    # Allocate overlap‑add buffers for the time‑domain signals
    full_combined = np.zeros(n_total, dtype=float)
    full_broadband = np.zeros(n_total, dtype=float)
    full_rhythmic = np.zeros(n_total, dtype=float)
    full_weight = np.zeros(n_total, dtype=float)

    # Allocate arrays to store raw window signals
    windowed_time = np.arange(n_window) / sampling_rate
    windowed_combined = np.zeros((n_windows, n_window), dtype=float)
    windowed_broadband = np.zeros((n_windows, n_window), dtype=float)
    windowed_rhythmic = np.zeros((n_windows, n_window), dtype=float)

    # Use rectangular weights (no taper) for stitching.  A weight envelope is
    # accumulated in ``full_weight`` and later used to normalise the stitched
    # signal.  This preserves the stationarity of each window and avoids
    # spectral leakage from Hann tapers.
    taper = np.ones(n_window, dtype=float)

    # Collect per‑window parameter dictionaries
    params_per_window: List[Dict[str, Any]] = []

    # Iterate through windows
    for idx, start in enumerate(starts):
        # Compute window centre time
        t_center = (start + n_window / 2.0) / sampling_rate
        # Resolve parameters for this window
        expo = _get_value_for_window(aperiodic_exponent_fn, idx, t_center, aperiodic_exponent)
        offset = _get_value_for_window(aperiodic_offset_fn, idx, t_center, aperiodic_offset)
        knee_val = _get_value_for_window(knee_fn, idx, t_center, knee)
        peaks_val = _get_value_for_window(peaks_fn, idx, t_center, peaks)

        # Instantiate simulator for this window
        window_sim = CombinedSimulator(
            sampling_rate=sampling_rate,
            n_samples=n_window,
            aperiodic_exponent=float(expo),
            aperiodic_offset=float(offset),
            knee=knee_val,
            peaks=peaks_val,
            average_firing_rate=average_firing_rate,
            random_state=int(seeds[idx]),
            mode=mode,
        )
        td = window_sim.simulate()

        # Store raw signals for this window
        windowed_combined[idx, :] = td.combined_signal
        windowed_broadband[idx, :] = td.broadband_signal
        windowed_rhythmic[idx, :] = td.rhythmic_signal

        # Overlap‑add with rectangular weights
        s0, s1 = start, start + n_window
        full_combined[s0:s1] += td.combined_signal * taper
        full_broadband[s0:s1] += td.broadband_signal * taper
        full_rhythmic[s0:s1] += td.rhythmic_signal * taper
        full_weight[s0:s1] += taper

        # Theoretical PSD for this window
        if mode == "additive":
            dec = ParametricDecomposition(
                sampling_rate=sampling_rate,
                n_fft=n_fft,
                aperiodic_exponent=float(expo),
                aperiodic_offset=float(offset),
                knee=knee_val,
                peaks=peaks_val,
            )
            fd = dec.compute()
            the_combined[idx, :] = fd.combined_spectrum
            the_broadband[idx, :] = fd.broadband_spectrum
            the_rhythmic[idx, :] = fd.rhythmic_spectrum
        else:
            # Multiplicative: baseline * 10**G; broadband/rhythmic parts undefined
            f = freq_grid
            kappa = 0.0 if knee_val is None else float(knee_val)
            with np.errstate(divide="ignore"):
                L = float(offset) - np.log10(kappa + np.abs(f) ** float(expo))
            if f.size and f[0] == 0.0:
                L[0] = -np.inf
            G = np.zeros_like(f)
            for pk in (peaks_val or []):
                c = float(pk.get("freq"))
                a_log = float(pk.get("amplitude"))
                sigma_val = float(pk.get("sigma"))
                # two‑sided bumps
                G += a_log * np.exp(-0.5 * ((f - c) / sigma_val) ** 2)
                G += a_log * np.exp(-0.5 * ((f + c) / sigma_val) ** 2)
            P = np.power(10.0, L + G)
            if f.size and f[0] == 0.0:
                P[0] = 0.0
            the_combined[idx, :] = P
            the_broadband[idx, :] = 0.0
            the_rhythmic[idx, :] = 0.0

        # Empirical PSD (optional)
        if direct_estimate:
            try:
                from spectral_connectivity import Multitaper, Connectivity
            except ImportError:
                raise ImportError(
                    "Install 'spectral_connectivity' for direct_estimate=True in spectrogram."
                )
            # Prepare input for multitaper: shape (time, trials, channels)
            sig = td.combined_signal[:, np.newaxis, np.newaxis]
            # Gather multitaper parameters from kwargs or use defaults
            mt_settings = {
                "time_halfbandwidth_product": 2,
                "n_tapers": 3,
                "n_fft_samples": n_window,
            }
            mt_settings.update(mt_kwargs)
            m = Multitaper(
                time_series=sig,
                sampling_frequency=sampling_rate,
                **mt_settings,
            )
            c = Connectivity.from_multitaper(m)
            power = c.power().squeeze()
            # Flatten potential shape (freqs, trials, channels) to (freqs,)
            if power.ndim > 1:
                power = power[..., 0, 0]
            freqs_emp = c.frequencies
            # On first window, allocate empirical array and store frequency grid
            if frequencies_empirical is None:
                frequencies_empirical = freqs_emp.copy()
                # Initialise empirical_list now that we know frequency length
                empirical_list = []
            # If subsequent windows have different length, interpolate to first grid
            if not np.array_equal(freqs_emp, frequencies_empirical):
                power_interp = np.interp(frequencies_empirical, freqs_emp, power)
                empirical_list.append(power_interp)
            else:
                empirical_list.append(power)

        # Save parameters used for this window
        params_per_window.append({
            "aperiodic_exponent": float(expo),
            "aperiodic_offset": float(offset),
            "knee": knee_val,
            "peaks": [] if peaks_val is None else list(peaks_val),
        })

    # Normalise by the weight envelope to stitch windows seamlessly
    nonzero = full_weight > 0
    full_combined[nonzero] /= full_weight[nonzero]
    full_broadband[nonzero] /= full_weight[nonzero]
    full_rhythmic[nonzero] /= full_weight[nonzero]

    # Construct TimeDomainData for the stitched signal
    time = np.arange(n_total) / sampling_rate
    time_domain = TimeDomainData(
        time=time,
        combined_signal=full_combined,
        broadband_signal=full_broadband,
        rhythmic_signal=full_rhythmic,
    )

    theoretical = {
        "combined": the_combined,
        "broadband": the_broadband,
        "rhythmic": the_rhythmic,
    }

    # Convert empirical_list to array if needed
    empirical_array: Optional[np.ndarray] = None
    if direct_estimate:
        if frequencies_empirical is None:
            # No spectra were computed; this can happen if n_window == 0 or
            # direct_estimate is True but spectral computation failed.
            empirical_array = None
        else:
            empirical_array = np.vstack(empirical_list)

    return SpectrogramResult(
        time_domain=time_domain,
        window_times=(starts + n_window / 2.0) / sampling_rate,
        frequencies=freq_grid,
        theoretical=theoretical,
        empirical=empirical_array,
        frequencies_empirical=frequencies_empirical,
        params_per_window=params_per_window,
        windowed_time=windowed_time,
        windowed_combined=windowed_combined,
        windowed_broadband=windowed_broadband,
        windowed_rhythmic=windowed_rhythmic,
    )