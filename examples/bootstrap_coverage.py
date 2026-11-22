"""Repeated-record coverage for a specified decomposition, with and without a rhythm.

Run with an installed checkout: python examples/bootstrap_coverage.py
Use --fit specparam to study the fitted exponent with data-driven peak selection
(specparam 2.0.0rc7). The default template fit estimates exponent and log-peak
height with known peak center and width. Reports empirical coverage, Wilson
intervals for Monte Carlo uncertainty, mean interval width and point-estimate
bias. The template's signed log-peak height is unconstrained.
"""
import argparse

import numpy as np
from scipy.stats import binomtest

from SL_GPsim import simulate_from_csd, spectral_bootstrap


def experiment(records, draws, seed, fit_method):
    fs, n = 128.0, 512
    rng = np.random.default_rng(seed)
    analysis_f = np.fft.rfftfreq(n, 1 / fs)
    band = (analysis_f >= 4) & (analysis_f <= 40)
    f = analysis_f[band]
    peak = np.exp(-0.5 * ((f - 12.0) / 2.0) ** 2)
    design = np.column_stack([np.ones(len(f)), -np.log10(f), peak])
    fit = np.linalg.pinv(design)

    def parameters(freqs, csd):
        return (fit @ np.log10(csd[band, 0, 0].real))[1:]

    if fit_method == "specparam":
        from specparam import SpectralModel

        def specparam_exponent(freqs, csd):
            model = SpectralModel(aperiodic_mode="fixed", max_n_peaks=3, verbose=False)
            model.fit(freqs[band], csd[band, 0, 0].real)
            return np.array([model.get_params("aperiodic", "exponent")])

        statistic = specparam_exponent
        names = ("exponent",)
    else:
        statistic = parameters
        names = ("exponent", "log10_peak_height")

    for height in (0.0, 0.8):
        truth = np.array([1.6, height])[:len(names)]
        grid = np.fft.rfftfreq(2 * n, 1 / fs)
        peak = np.exp(-0.5 * ((grid - 12.0) / 2.0) ** 2)
        density = np.maximum(grid, 1.0) ** -truth[0] * 10 ** (height * peak)
        density[1:-1] *= 2
        hit = np.zeros(len(names), dtype=int)
        width = np.zeros(len(names))
        bias = np.zeros(len(names))
        for _ in range(records):
            x = simulate_from_csd(density, fs, n, random_state=rng)
            result = spectral_bootstrap(x, fs, statistic, n_resamples=draws, random_state=rng)
            hit += (result.lower <= truth) & (truth <= result.upper)
            width += result.upper - result.lower
            bias += result.estimate - truth
        for j, name in enumerate(names):
            mc = binomtest(hit[j], records).proportion_ci(method="wilson")
            print(f"{fit_method} height={height:g} {name}: covered {hit[j]}/{records} "
                  f"({hit[j]/records:.1%}, MC 95% {mc.low:.1%}–{mc.high:.1%}); "
                  f"mean width {width[j]/records:.3f}, bias {bias[j]/records:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--draws", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--fit", choices=("template", "specparam"), default="template")
    args = parser.parse_args()
    experiment(args.records, args.draws, args.seed, args.fit)
