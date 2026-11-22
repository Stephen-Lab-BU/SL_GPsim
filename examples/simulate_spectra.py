"""Callable ground truth and eight-second Gaussian records with one or two channels."""
import json

import numpy as np
from SL_GPsim import Broadband, GaussianPeak, Independent, Shared, SpectralGrid, multitaper_csd


def figure_models():
    background = Broadband(scale=8, knee=2, exponent=1.6)
    beta = GaussianPeak(center_hz=20, sigma_hz=1.8, height=0.70)
    single = background + beta

    private = (Broadband(scale=5.4, knee=4, exponent=1.4)
               + GaussianPeak(center_hz=10, sigma_hz=1.1, height=0.44))
    # The common beta source reaches channel 2 15.625 ms later.
    pair = (Independent(background, private)
            + Shared(beta, gains=[1, 0.7], delays_seconds=[0, 0.015625]))
    return background, beta, single, pair


def figure_records():
    grid = SpectralGrid(fs=128, duration=8)
    background, beta, single, pair = figure_models()
    x = grid.simulate(single, random_state=7)
    xy = grid.simulate(pair, random_state=112)
    return (grid.frequencies, grid.sample(background), grid.sample(beta),
            grid.sample(single), grid.sample(pair), x, xy)


if __name__ == "__main__":
    _, _, single, pair = figure_models()
    f_plot = np.linspace(0, 64, 10_001)
    print("Smooth PSD and CSD shapes:", single(f_plot).shape, pair(f_plot).shape)
    print("Univariate generating parameters:", json.dumps(single.parameters))
    f, _, _, psd, csd, x, xy = figure_records()
    print("Simulation grid and target shapes:", f.shape, psd.shape, csd.shape)
    freqs, estimate = multitaper_csd(xy, fs=128)
    print("Time series and observed CSD shapes:", x.shape, xy.shape, estimate.shape)
