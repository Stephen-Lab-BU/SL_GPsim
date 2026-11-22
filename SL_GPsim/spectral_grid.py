"""Build one-sided spectra in linear power per Hz on the simulation FFT grid."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cross_spectral import (
    RandomState, _integer, _rate, _record, _record_length, multitaper_csd, simulate_from_csd,
)
from .models import (
    Broadband, GaussianPeak, SpectralWarning, SpectrumModel, _FFT, _diagonal,
    _mixing_vectors, _shared_csd,
)


@dataclass(frozen=True, eq=False, init=False)
class SpectralGrid:
    """Recording and frequency grids specified by fs (Hz) and duration (seconds).

    The embedding length is at least twice the record length. sample(model)
    returns the endpoint-weighted array for the simulator. Univariate components
    add in linear power. A sum of independent and shared contributions is a
    CSD with axes (frequency, channel, channel).
    """

    sampling_rate: float
    n_samples: int
    n_fft: int
    frequencies: NDArray = field(repr=False)

    def __init__(
        self, sampling_rate: float | None = None, n_samples: int | None = None, *,
        fs: float | None = None, duration: float | None = None, n_fft: int | None = None,
    ):
        fs = _rate(sampling_rate, fs)
        n = _record_length(fs, n_samples, duration)
        length = 2 * n if n_fft is None else _integer(n_fft, "n_fft", 2 * n)
        f = np.fft.rfftfreq(length, 1 / fs)
        f.setflags(write=False)
        object.__setattr__(self, "sampling_rate", fs)
        object.__setattr__(self, "n_samples", n)
        object.__setattr__(self, "n_fft", length)
        object.__setattr__(self, "frequencies", f)

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz."""
        return self.sampling_rate

    @property
    def duration(self) -> float:
        """Realized recording duration in seconds."""
        return self.n_samples / self.fs

    @property
    def times(self) -> NDArray:
        """Recording timestamps in seconds, starting at zero."""
        return np.arange(self.n_samples) / self.fs

    def sample(self, model: SpectrumModel) -> NDArray:
        """Return the exact one-sided PSD or CSD array used for this simulation."""
        if not isinstance(model, SpectrumModel):
            raise TypeError("sample expects a spectrum model; use density for a custom PSD formula")
        return model._sample(self.frequencies, self.fs, self.n_fft)

    def _check_psd(self, values: ArrayLike) -> NDArray:
        a = np.asarray(values)
        if a.shape != self.frequencies.shape:
            raise ValueError(f"expected a PSD of length {len(self.frequencies)} on grid.frequencies")
        if np.iscomplexobj(a) and np.any(a.imag != 0):
            raise ValueError("a channel PSD must be real; use a CSD matrix for cross-channel phase")
        a = np.asarray(a.real, dtype=float)
        invalid = np.flatnonzero(~np.isfinite(a) | (a < 0))
        if invalid.size:
            j = invalid[0]
            if j == 0:
                raise ValueError("PSD at DC (0 Hz) must be finite and nonnegative; "
                                 "give the low-frequency model a finite limit, e.g. a positive knee")
            raise ValueError(f"PSD must be finite and nonnegative at {self.frequencies[j]:g} Hz")
        return a

    def density(self, model: Callable[[NDArray], ArrayLike]) -> NDArray:
        """Evaluate a smooth one-sided PSD formula and weight self-conjugate bins.

        The formula gives the *interior* one-sided density continued to the
        boundaries. A scalar return value makes a white spectrum. To simulate
        an already folded estimate or explicit DC/Nyquist line, pass the
        prepared array directly to simulate().
        """
        if isinstance(model, SpectrumModel):
            return self.sample(model)
        if not callable(model):
            raise TypeError("density takes a formula; pass an already weighted array directly to simulate")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = np.asarray(model(self.frequencies))
        if values.ndim == 0:
            values = np.broadcast_to(values, self.frequencies.shape)
        psd = self._check_psd(values).copy()
        psd[0] *= 0.5
        if self.n_fft % 2 == 0:
            psd[-1] *= 0.5
        return psd

    def broadband(
        self, *, scale: float, exponent: float, knee_hz: float | None = None,
        knee: float | None = None,
    ) -> NDArray:
        """Return scale / (knee + f**exponent), with knee = knee_hz**exponent.

        Supply one knee parameter. At knee_hz the smooth density equals half
        the low-frequency plateau. Scale is the linear numerator.
        """
        return self.sample(Broadband(scale, exponent, knee=knee, knee_hz=knee_hz))

    def peak(self, *, center_hz: float, sigma_hz: float, height: float) -> NDArray:
        """Positive-frequency Gaussian with a linear PSD height and SD in Hz."""
        return self.sample(GaussianPeak(center_hz, sigma_hz, height))

    def independent(self, *channel_psds: ArrayLike) -> NDArray:
        """Build a diagonal CSD from one already weighted private PSD per channel."""
        if not channel_psds:
            raise ValueError("supply one private PSD per channel")
        return _diagonal([self._check_psd(values) for values in channel_psds])

    def shared(
        self, source_psd: ArrayLike, *, gains: ArrayLike, delays_samples: ArrayLike | None = None,
    ) -> NDArray:
        """Mix one latent source into channels with real amplitude gains and delays.

        Positive delay means a later signal. At even-grid Nyquist the symmetric
        real part of the cross-spectrum preserves auto-power and defines a valid
        real matrix for fractional relative delays.
        """
        source = self._check_psd(source_psd)
        gain, delays = _mixing_vectors(gains, delays_samples)
        return _shared_csd(source, self.frequencies, gain, delays / self.fs, _FFT(self.fs, self.n_fft))

    def simulate(self, spectrum: SpectrumModel | ArrayLike, *, random_state: RandomState = None) -> NDArray:
        """Draw a real (n_samples, channels) record from a callable model or prepared PSD/CSD."""
        return simulate_from_csd(spectrum, self.sampling_rate, self.n_samples,
                                 n_fft=self.n_fft, random_state=random_state)


def simulate_like(
    data: ArrayLike, sampling_rate: float | None = None, *, fs: float | None = None, n_fft: int | None = None,
    time_halfbandwidth: float = 2.0, n_tapers: int = 3, demean: bool = True,
    random_state: RandomState = None,
) -> NDArray:
    """Draw a zero-population-mean record from the observed full multitaper CSD.

    Input time is the first axis. The output always has shape (time, channels).
    This is a single plug-in Gaussian replicate on the bootstrap embedding.
    """
    x = _record(data)
    grid = SpectralGrid(sampling_rate, len(x), fs=fs, n_fft=n_fft)
    _, csd = multitaper_csd(x, grid.sampling_rate, n_fft=grid.n_fft,
                            time_halfbandwidth=time_halfbandwidth, n_tapers=n_tapers, demean=demean)
    return grid.simulate(csd, random_state=random_state)
