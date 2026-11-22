"""Gaussian time-domain bootstrap for spectral estimates and decomposition fits."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cross_spectral import (
    RandomState, SpectralModel, _EmbeddedGaussian, _integer, _matrix_shape,
    _rate, _record, _tapered_csd, _tapers,
)

Estimator = Callable[[NDArray, float, int], tuple[ArrayLike, ArrayLike]]
Statistic = Callable[[NDArray, NDArray], ArrayLike]


@dataclass
class SpectralBootstrapResult:
    """Point estimate, paired draws and marginal percentile bounds in statistic units."""

    frequencies: NDArray
    estimate: NDArray
    draws: NDArray
    lower: NDArray
    upper: NDArray
    n_valid: NDArray
    confidence_level: float


def spectral_bootstrap(
    data: ArrayLike, sampling_rate: float | None = None, statistic: Statistic | None = None, *,
    fs: float | None = None,
    n_resamples: int = 999, confidence_level: float = 0.95,
    pilot_csd: SpectralModel | None = None, n_fft_embedding: int | None = None,
    estimator: Estimator | None = None, time_halfbandwidth: float = 2.0,
    n_tapers: int = 3, demean: bool = True, allow_missing: bool = False,
    random_state: RandomState = None,
) -> SpectralBootstrapResult:
    """Re-estimate and refit each same-length stationary Gaussian record.

    data: (time,) or (time, channels). By default each analyzed record is
    centered; with demean=False the model assumes a known zero process mean.
    statistic(freqs, csd) receives the full one-sided matrix (F,C,C) and
    returns a fixed-shape real scalar/array. Default: auto-power (F,C).
    Rows of draws correspond to independent records; all entries in a row
    come from the same record. Bounds are pointwise percentile intervals.

    The default generating density is the equal-weight DPSS estimate of
    the observed record evaluated directly on the embedding FFT grid
    (default 2*N). pilot_csd overrides it with a full density on that grid,
    or a callable taking the grid in Hz. SpectrumModel pilots use the same automatic endpoint
    weighting as simulate_from_csd.

    A custom estimator(data_2d, fs, n_fft) returns (rfft frequencies, CSD)
    in the same units. It is called at N for analysis and at the embedding
    length for the pilot unless pilot_csd is given. The default DPSS settings
    apply only to the built-in estimator. Restrict fit bands in statistic.

    By default a failed or nonfinite fit stops the run. To represent a missing
    peak, return NaN in its fixed slot and set allow_missing=True. Its interval
    is conditional on detection, with n_valid giving the denominator. Fewer
    than two finite draws produce NaN bounds. Exceptions always propagate.
    """
    x = _record(data)
    n, channels = x.shape
    fs = _rate(sampling_rate, fs)
    b = _integer(n_resamples, "n_resamples", 2)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    m = 2 * n if n_fft_embedding is None else _integer(n_fft_embedding, "n_fft_embedding", 2 * n)
    rng = np.random.default_rng(random_state)
    if estimator is None:
        tapers = _tapers(n, time_halfbandwidth, n_tapers)

    def estimate(record: NDArray, size: int) -> tuple[NDArray, NDArray]:
        freqs = np.fft.rfftfreq(size, 1 / fs)
        if estimator is None:
            return freqs, _tapered_csd(record, fs, size, tapers, demean)
        if demean:
            record = record - record.mean(axis=0)
        grid, raw = estimator(record, fs, size)
        grid = np.asarray(grid)
        if grid.shape != freqs.shape or not np.allclose(grid, freqs, rtol=1e-10, atol=0):
            raise ValueError("estimator must return the requested rfft grid in Hz")
        csd = _matrix_shape(raw, len(freqs))
        if csd.shape[1] != channels or not np.isfinite(csd).all():
            raise ValueError("estimator returned invalid channel count or nonfinite CSD")
        return freqs, csd

    def measure(record: NDArray) -> NDArray:
        freqs, csd = estimate(record, n)
        value = (csd.diagonal(axis1=1, axis2=2).real if statistic is None
                 else np.asarray(statistic(freqs, csd)))
        if np.iscomplexobj(value):
            raise ValueError("statistic must be real; return real/imaginary parts separately")
        value = np.asarray(value, dtype=float)
        if np.isinf(value).any() or (not allow_missing and np.isnan(value).any()):
            raise ValueError("statistic returned a nonfinite value")
        return value

    observed = measure(x)
    if pilot_csd is None:
        _, pilot_csd = estimate(x, m)
    model = _EmbeddedGaussian(pilot_csd, fs, n, m)
    if model.factor.shape[1] != channels:
        raise ValueError("pilot_csd must have the same channels as data")
    draws = np.empty((b, *observed.shape))
    for i in range(b):
        try:
            value = measure(model.draw(rng))
        except Exception as exc:
            raise RuntimeError(f"bootstrap replicate {i + 1} of {b} failed: {exc}") from exc
        if value.shape != observed.shape:
            raise ValueError(f"statistic changed shape in bootstrap replicate {i + 1}")
        draws[i] = value

    alpha = (1 - confidence_level) / 2
    valid = np.isfinite(draws).sum(axis=0)
    if allow_missing:
        flat = draws.reshape(b, -1)
        usable = valid.ravel() >= 2
        bounds = np.full((2, flat.shape[1]), np.nan)
        if usable.any():
            bounds[:, usable] = np.nanquantile(flat[:, usable], [alpha, 1 - alpha], axis=0)
        lower, upper = bounds.reshape((2, *observed.shape))
    else:
        lower, upper = np.quantile(draws, [alpha, 1 - alpha], axis=0)

    return SpectralBootstrapResult(np.fft.rfftfreq(n, 1 / fs), observed, draws,
                                   lower, upper, valid, confidence_level)
