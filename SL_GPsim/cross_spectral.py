"""Stationary, real Gaussian records from matrix-valued spectral densities."""
from __future__ import annotations

from collections.abc import Callable
from operator import index
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import dpss

from .models import SpectralWarning, SpectrumModel

SpectralModel = ArrayLike | Callable[[NDArray], ArrayLike]
RandomState = int | np.random.Generator | None


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _rate(sampling_rate: float | None, fs: float | None = None) -> float:
    if (sampling_rate is None) == (fs is None):
        raise ValueError("supply exactly one sampling rate in Hz, using fs or sampling_rate")
    value = sampling_rate if fs is None else fs
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) or np.iscomplexobj(value):
        raise ValueError("sampling_rate must be finite and positive")
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    return rate


def _record_length(fs: float, n_samples: int | None, duration: float | None) -> int:
    if (n_samples is None) == (duration is None):
        raise ValueError("give exactly one of duration (seconds) or n_samples")
    if n_samples is not None:
        return _integer(n_samples, "n_samples", 1)
    if isinstance(duration, (bool, np.bool_)) or np.ndim(duration) or np.iscomplexobj(duration):
        raise ValueError("duration must be a finite positive number of seconds")
    count = float(duration) * fs
    if not np.isfinite(count) or count <= 0:
        raise ValueError("duration must give a finite positive sample count at this fs")
    n = int(np.floor(count + 0.5))
    if n < 1:
        raise ValueError("duration is too short to produce one sample at this fs")
    if abs(n - count) > 4 * np.spacing(count):
        warnings.warn(f"duration × fs is {count:g} samples; using {n} samples "
                      f"({n/fs:g} seconds).", SpectralWarning, stacklevel=3)
    return n


def _record(data: ArrayLike) -> NDArray:
    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) == 0 or np.iscomplexobj(x):
        raise ValueError("data must be real with shape (time,) or (time, channels)")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("data must be finite")
    return x


def _matrix_shape(csd: ArrayLike, bins: int) -> NDArray:
    s = np.asarray(csd)
    if s.ndim == 1:
        s = s[:, None, None]
    if s.ndim != 3 or s.shape[0] != bins or s.shape[1] != s.shape[2] or s.shape[1] == 0:
        raise ValueError(f"expected a PSD of length {bins} or a CSD of shape ({bins}, C, C) "
                         "on the full rFFT grid, including DC and Nyquist for even n_fft")
    return s


def _tapers(n: int, time_halfbandwidth: float, n_tapers: int) -> NDArray:
    k = _integer(n_tapers, "n_tapers", 1)
    if k > n or not 0 < time_halfbandwidth < n / 2:
        raise ValueError("DPSS requires n_tapers <= N and 0 < time_halfbandwidth < N/2")
    return dpss(n, time_halfbandwidth, Kmax=k, norm=2)


def _tapered_csd(x: NDArray, fs: float, n_fft: int, tapers: NDArray, demean: bool) -> NDArray:
    if demean:
        x = x - x.mean(axis=0)
    shape = (n_fft // 2 + 1, x.shape[1], x.shape[1])
    csd = np.zeros(shape, dtype=complex)
    for taper in tapers:
        z = np.fft.rfft(x * taper[:, None], n=n_fft, axis=0)
        csd += z[:, :, None] * z[:, None, :].conj()
    csd /= fs * len(tapers)
    csd[1:(n_fft + 1) // 2] *= 2
    return csd


def multitaper_csd(
    data: ArrayLike, sampling_rate: float | None = None, n_fft: int | None = None, *,
    fs: float | None = None,
    time_halfbandwidth: float = 2.0, n_tapers: int = 3, demean: bool = True,
) -> tuple[NDArray, NDArray]:
    """Return Hz and one-sided equal-weight DPSS density, shape (frequency, C, C).

    Time is the first axis. S[i,j] uses X_i conj(X_j), i.e. the conjugate of
    scipy.signal.csd(x_i, x_j). Subtract the record mean before tapering by
    default. A longer FFT evaluates the same tapered estimate on a finer grid.
    """
    x = _record(data)
    fs = _rate(sampling_rate, fs)
    n = len(x)
    m = n if n_fft is None else _integer(n_fft, "n_fft", n)
    tapers = _tapers(n, time_halfbandwidth, n_tapers)
    return np.fft.rfftfreq(m, 1 / fs), _tapered_csd(x, fs, m, tapers, demean)


class _EmbeddedGaussian:
    """Prepare once, then reuse the spectral factor for independent draws."""

    def __init__(self, csd: SpectralModel, fs: float, n: int, n_fft: int | None):
        self.n = n
        self.m = 2 * n if n_fft is None else _integer(n_fft, "n_fft", 2 * n)
        f = np.fft.rfftfreq(self.m, 1 / fs)
        if isinstance(csd, SpectrumModel):
            raw = csd._sample(f, fs, self.m)
        else:
            raw = csd(f) if callable(csd) else csd
        s = _matrix_shape(raw, len(f))
        self.factor = np.empty(s.shape, dtype=complex)
        eps = np.finfo(float).eps
        if np.issubdtype(s.dtype, np.inexact):
            eps = max(eps, np.finfo(s.real.dtype).eps)
        tol = max(64 * np.finfo(float).eps, 4 * eps) * s.shape[1]
        endpoints = (0, self.m // 2) if self.m % 2 == 0 else (0,)
        # Bound temporary memory for multichannel records.
        batch = max(1, 262144 // s.shape[1] ** 2)
        for start in range(0, len(f), batch):
            stop = min(len(f), start + batch)
            block = np.array(s[start:stop], dtype=complex, copy=True)
            valid = np.isfinite(block).all(axis=(1, 2))
            if not valid.all():
                i = start + np.flatnonzero(~valid)[0]
                if i == 0:
                    raise ValueError("nonfinite spectral density at DC (0 Hz); supply a finite "
                                     "low-frequency model, such as a broadband with a positive knee")
                name = "Nyquist" if i in endpoints else f"bin {i}"
                raise ValueError(f"nonfinite spectral density at {name} ({f[i]:g} Hz)")
            diag = block.diagonal(axis1=1, axis2=2).real
            if (diag < 0).any():
                raise ValueError("auto-spectral densities must be nonnegative")
            # Use pairwise channel units, so a high-power channel cannot hide
            # an invalid low-power cross-spectrum.
            scale = np.sqrt(diag)
            pair_scale = scale[:, :, None] * scale[:, None, :]
            for endpoint in endpoints:
                if start <= endpoint < stop:
                    j = endpoint - start
                    if np.any(abs(block[j].imag) > tol * pair_scale[j]):
                        name = "DC" if endpoint == 0 else "Nyquist"
                        raise ValueError(f"CSD must be real at {name} ({f[endpoint]:g} Hz); "
                                         "supply a real symmetric (Hermitian) matrix at that bin")
                    block[j] = block[j].real
            if np.any(abs(block - block.conj().swapaxes(1, 2)) > tol * pair_scale):
                raise ValueError("CSD must be Hermitian at each frequency")
            if np.any((pair_scale == 0) & (block != 0)):
                raise ValueError("cross-power with a zero-power channel must be zero")
            corr = np.divide(block, pair_scale, out=np.zeros_like(block), where=pair_scale > 0)
            vals, vectors = np.linalg.eigh(corr)
            # At self-conjugate bins the factor itself must be real. A complex
            # Hermitian solver is free to choose arbitrary eigenvector phases.
            for endpoint in endpoints:
                if start <= endpoint < stop:
                    j = endpoint - start
                    vals[j], vectors[j] = np.linalg.eigh(corr[j].real)
            if np.any(vals[:, 0] < -tol * np.maximum(1, vals[:, -1])):
                raise ValueError(f"CSD must be positive semidefinite in bins [{start}, {stop})")
            # L L* = S. An eigenfactor also works for exactly coherent channels.
            self.factor[start:stop] = scale[:, :, None] * vectors * np.sqrt(vals.clip(0))[:, None, :]
        # Convert folded one-sided density to two-sided density, then to DFT units.
        self.factor[1:(self.m + 1) // 2] /= np.sqrt(2)
        self.factor *= np.sqrt(fs * self.m)

    def draw(self, rng: np.random.Generator) -> NDArray:
        bins, channels, _ = self.factor.shape
        z = (rng.standard_normal((bins, channels)) +
             1j * rng.standard_normal((bins, channels))) / np.sqrt(2)
        z[0] = np.sqrt(2) * z[0].real
        if self.m % 2 == 0:
            z[-1] = np.sqrt(2) * z[-1].real
        fft = np.einsum("fij,fj->fi", self.factor, z)
        return np.fft.irfft(fft, n=self.m, axis=0)[:self.n].copy()


def simulate_from_csd(
    csd: SpectralModel, sampling_rate: float | None = None, n_samples: int | None = None, *,
    fs: float | None = None, duration: float | None = None,
    n_fft: int | None = None, random_state: RandomState = None,
) -> NDArray:
    """Draw an (n_samples, channels) zero-mean stationary Gaussian record.

    Use fs in Hz and duration in seconds, or supply an exact n_samples. Built-in
    SpectrumModel objects are evaluated and endpoint-weighted automatically.
    For arrays and ordinary callables, supply a one-sided linear density per Hz on rfftfreq(n_fft, 1/fs), as a
    (frequency, channel, channel) array, a 1-D PSD, or a callable taking Hz.
    Interior densities include both spectral sides; DC and an even-grid
    Nyquist do not. S_ij = E[X_i conj(X_j)] in density units. The default
    embedding FFT is 2*n_samples. Set n_fft explicitly for a denser grid.
    The finite draw has the covariance implied by that grid; for an arbitrary
    continuous spectrum this is a quadrature approximation.
    """
    fs = _rate(sampling_rate, fs)
    n = _record_length(fs, n_samples, duration)
    model = _EmbeddedGaussian(csd, fs, n, n_fft)
    return model.draw(np.random.default_rng(random_state))
