"""Callable generating spectra. Frequencies are in Hz and delays in seconds."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SpectralWarning(UserWarning):
    """A valid spectral construction needs attention on this sampling grid."""


class _FFT(NamedTuple):
    fs: float
    n_fft: int


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


class SpectrumModel:
    """A callable smooth one-sided PSD or CSD, independent of a recording grid."""

    def __call__(self, frequencies: ArrayLike) -> NDArray:
        """Evaluate at any nonnegative Hz value or vector, in input order."""
        f = np.asarray(frequencies)
        if f.ndim > 1 or np.iscomplexobj(f):
            raise ValueError("frequencies must be a real scalar or vector in Hz")
        f = np.asarray(f, dtype=float)
        if not np.isfinite(f).all() or (f < 0).any():
            raise ValueError("frequencies must be finite and nonnegative in Hz")
        value = self._evaluate(np.atleast_1d(f), None)
        return value[0] if f.ndim == 0 else value

    def _evaluate(self, f: NDArray, fft: _FFT | None) -> NDArray:
        raise NotImplementedError

    def _sample(self, f: NDArray, fs: float, n_fft: int) -> NDArray:
        """Weight continuous one-sided limits once, after evaluating the full expression."""
        value = self._evaluate(f, _FFT(fs, n_fft))
        if not np.isfinite(value).all():
            raise ValueError("model produced nonfinite density on the simulation grid")
        value[0] *= 0.5
        if n_fft % 2 == 0:
            value[-1] *= 0.5
        return value

    def __add__(self, other: SpectrumModel) -> SpectrumModel:
        if not isinstance(other, SpectrumModel):
            return NotImplemented
        return _Sum(self, other)

    @property
    def parameters(self) -> dict:
        """Return a fresh JSON-compatible description, including all nested forms."""
        def export(value):
            if isinstance(value, SpectrumModel):
                return value.parameters
            if isinstance(value, tuple):
                return [export(item) for item in value]
            return value

        return {"form": type(self).__name__.lstrip("_"),
                **{f.name: export(getattr(self, f.name)) for f in fields(self)}}


@dataclass(frozen=True)
class Broadband(SpectrumModel):
    """scale / (knee + f**exponent), where knee = knee_hz**exponent."""

    scale: float
    exponent: float
    knee: float | None = None
    knee_hz: float | None = None

    def __post_init__(self):
        if (self.knee is None) == (self.knee_hz is None):
            raise ValueError("give exactly one of knee or knee_hz")
        for key in ("scale", "exponent", "knee", "knee_hz"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, _number(value, key))
        if self.scale < 0 or self.exponent <= 0:
            raise ValueError("scale must be nonnegative and exponent positive")
        given = self.knee if self.knee is not None else self.knee_hz
        if given <= 0:
            raise ValueError("use a positive knee for a finite broadband density at DC")
        if not np.isfinite(self._coefficient) or self._coefficient == 0:
            raise ValueError("knee_hz**exponent is outside the representable range")

    @property
    def _coefficient(self):
        if self.knee is not None:
            return self.knee
        with np.errstate(over="ignore", under="ignore"):
            return np.power(self.knee_hz, self.exponent)

    def _evaluate(self, f, fft):
        k = self._coefficient
        if fft and self.scale:
            log_knee = np.log(k) / self.exponent
            if log_knee < np.log(fft.fs / fft.n_fft) - 1e-12:
                warnings.warn("The knee lies below the first positive embedding frequency. "
                              "Increase n_fft to sample the turnover.", SpectralWarning, stacklevel=3)
            if log_knee > np.log(fft.fs / 2) + 1e-12:
                warnings.warn("The knee lies above Nyquist. Check knee units or increase "
                              "the sampling rate to include the turnover.", SpectralWarning, stacklevel=3)
        with np.errstate(over="ignore"):
            return self.scale / (k + f**self.exponent)


@dataclass(frozen=True)
class GaussianPeak(SpectrumModel):
    """Gaussian evaluated at nonnegative frequencies, with standard deviation in Hz."""

    center_hz: float
    sigma_hz: float
    height: float

    def __post_init__(self):
        for key in ("center_hz", "sigma_hz", "height"):
            object.__setattr__(self, key, _number(getattr(self, key), key))
        if self.center_hz < 0 or self.sigma_hz <= 0 or self.height < 0:
            raise ValueError("center_hz and height must be nonnegative and sigma_hz positive")

    def _evaluate(self, f, fft):
        if fft:
            nyquist = fft.fs / 2
            if self.center_hz > nyquist:
                raise ValueError(f"center_hz must lie between DC (0 Hz) and Nyquist ({nyquist:g} Hz)")
            edges = []
            if self.center_hz - 3 * self.sigma_hz < 0:
                edges.append("DC (0 Hz)")
            if self.center_hz + 3 * self.sigma_hz > nyquist:
                edges.append(f"Nyquist ({nyquist:g} Hz)")
            if self.height and edges:
                warnings.warn(f"The peak's ±3 SD range crosses {' and '.join(edges)}. "
                              "Check the center, width, and sampling rate.", SpectralWarning, stacklevel=3)
            if self.height and self.sigma_hz < fft.fs / fft.n_fft:
                warnings.warn("Peak SD is below one embedding frequency bin. Increase n_fft "
                              "to sample the generating peak more finely.", SpectralWarning, stacklevel=3)
        with np.errstate(over="ignore"):
            return self.height * np.exp(-0.5 * ((f - self.center_hz) / self.sigma_hz)**2)


def _require_model(value):
    if not isinstance(value, SpectrumModel):
        raise TypeError("expected a callable SL_GPsim spectrum model")


def _univariate(model, f, fft):
    value = model._evaluate(f, fft)
    if value.ndim != 1:
        raise ValueError("a source or private channel must have a univariate PSD")
    return value


@dataclass(frozen=True)
class _Sum(SpectrumModel):
    left: SpectrumModel
    right: SpectrumModel

    def _evaluate(self, f, fft):
        a, b = self.left._evaluate(f, fft), self.right._evaluate(f, fft)
        if a.shape != b.shape:
            raise ValueError("add spectra with matching channel dimensions")
        return a + b


@dataclass(frozen=True)
class LogPeaks(SpectrumModel):
    """Multiplicative model: background * 10**peaks; peak heights are in log10 units."""

    background: SpectrumModel
    peaks: SpectrumModel

    def __post_init__(self):
        _require_model(self.background)
        _require_model(self.peaks)

    def _evaluate(self, f, fft):
        background = _univariate(self.background, f, fft)
        return background * 10**_univariate(self.peaks, f, fft)


def _diagonal(spectra):
    c = len(spectra)
    result = np.zeros((len(spectra[0]), c, c), dtype=complex)
    result[:, np.arange(c), np.arange(c)] = np.column_stack(spectra)
    return result


@dataclass(frozen=True, init=False)
class Independent(SpectrumModel):
    """One private PSD per channel, in positional channel order."""

    channels: tuple[SpectrumModel, ...]

    def __init__(self, *channels: SpectrumModel):
        if not channels:
            raise ValueError("supply one private PSD per channel")
        for ch in channels:
            _require_model(ch)
        object.__setattr__(self, "channels", channels)

    def _evaluate(self, f, fft):
        return _diagonal([_univariate(ch, f, fft) for ch in self.channels])


def _mixing_vectors(gains, delays):
    gain = np.asarray(gains)
    delay = np.zeros(gain.shape) if delays is None else np.asarray(delays)
    if gain.ndim != 1 or not gain.size or delay.shape != gain.shape:
        raise ValueError("give one gain and, optionally, one delay per channel")
    if np.iscomplexobj(gain) or np.iscomplexobj(delay):
        raise ValueError("gains and delays must be real")
    gain, delay = np.asarray(gain, dtype=float), np.asarray(delay, dtype=float)
    if not np.isfinite(gain).all() or not np.isfinite(delay).all():
        raise ValueError("gains and delays must be finite")
    return gain, delay


def _shared_csd(source, f, gain, delay_seconds, fft):
    c = len(gain)
    result = np.empty((len(f), c, c), dtype=complex)
    fractional = False
    for i in range(c):
        result[:, i, i] = gain[i]**2 * source
        for j in range(i):
            lag = delay_seconds[i] - delay_seconds[j]
            cross = gain[i] * gain[j] * source * np.exp(-2j * np.pi * f * lag)
            if fft and fft.n_fft % 2 == 0:
                lag_samples = lag * fft.fs
                if source[-1] and gain[i] and gain[j] and abs(lag_samples - np.rint(lag_samples)) > 1e-12:
                    fractional = True
                cross[-1] = cross[-1].real
            result[:, i, j] = cross
            result[:, j, i] = cross.conj()
    if fractional:
        warnings.warn("Fractional relative delay at Nyquist uses real cosine cross-power "
                      "and can reduce coherence in that bin. Use an odd n_fft to avoid "
                      "a Nyquist bin.", SpectralWarning, stacklevel=3)
    return result


@dataclass(frozen=True, init=False)
class Shared(SpectrumModel):
    """One source shared across channels; positive delay means the channel lags."""

    source: SpectrumModel
    gains: tuple[float, ...]
    delays_seconds: tuple[float, ...]

    def __init__(self, source: SpectrumModel, *, gains: ArrayLike, delays_seconds: ArrayLike | None = None):
        _require_model(source)
        g, d = _mixing_vectors(gains, delays_seconds)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "gains", tuple(g.tolist()))
        object.__setattr__(self, "delays_seconds", tuple(d.tolist()))

    def _evaluate(self, f, fft):
        return _shared_csd(_univariate(self.source, f, fft), f, self.gains, self.delays_seconds, fft)


def model_from_parameters(parameters: dict) -> SpectrumModel:
    """Restore a built-in model from its JSON-compatible parameters."""
    config = dict(parameters)
    form = config.pop("form", None)
    if form == "Broadband":
        return Broadband(**config)
    if form == "GaussianPeak":
        return GaussianPeak(**config)
    if form == "Independent":
        return Independent(*(model_from_parameters(ch) for ch in config["channels"]))
    if form == "Shared":
        return Shared(model_from_parameters(config.pop("source")), **config)
    if form == "Sum":
        return model_from_parameters(config["left"]) + model_from_parameters(config["right"])
    if form == "LogPeaks":
        return LogPeaks(model_from_parameters(config["background"]), model_from_parameters(config["peaks"]))
    raise ValueError(f"unknown spectral form {form!r}")
