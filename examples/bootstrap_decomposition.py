"""Refit specparam and SL_specdecomp on identical realizations of a synthetic EEG window.

Optional dependencies: specparam 2.0.0rc7 and SL_specdecomp with fast Gamma MAP.
Run from an installed checkout: python examples/bootstrap_decomposition.py --draws 99
"""
import argparse
from contextlib import redirect_stdout
import io

import numpy as np

from SL_GPsim import Broadband, GaussianPeak, spectral_bootstrap, simulate_from_csd


def fit_both(freqs, csd):
    from specparam import SpectralModel
    from SL_specdecomp import Decompose

    band = (freqs >= 3) & (freqs <= 45)
    f, p = freqs[band], csd[band, 0, 0].real
    sp = SpectralModel(aperiodic_mode="knee", max_n_peaks=1, verbose=False)
    sp.fit(f, p)
    spec = np.full(7, np.nan)
    if sp.results.has_model:
        offset, k, exponent = sp.results.params.aperiodic.get_params("fit")
        knee_hz = k**(1/exponent) if k > 0 and exponent > 0 else np.nan
        spec[:4] = [offset, exponent, k, knee_hz]
        peaks = np.asarray(sp.get_params("periodic"))
        if peaks.size and np.isfinite(peaks).any():
            peaks = peaks.reshape(-1, 3)
            beta = peaks[(peaks[:,0] >= 15) & (peaks[:,0] <= 25)]
            if len(beta):
                spec[4:] = beta[0]
    with redirect_stdout(io.StringIO()):
        sl = Decompose(f, p, mode="additive", n_rhythms=1, rhythm_bands=[(15, 25)],
                       k_tapers=3, sample_kwargs={"inference_method": "fast_map"})
    fit = sl.idata.posterior
    if not fit.attrs["optimizer_success"]:
        raise RuntimeError(fit.attrs["optimizer_message"])
    vals = [fit[k].item() for k in ("b_0", "chi_0", "knee_hz_0", "center_0", "A_lin_0", "sigma_0")]
    return np.r_[spec, vals]


def main(draws):
    fs = 128
    model = (Broadband(scale=8, knee=2, exponent=1.6)
             + GaussianPeak(center_hz=20, sigma_hz=1.8, height=0.70))
    x = simulate_from_csd(model, fs=fs, duration=8, random_state=7)
    boot = spectral_bootstrap(x, fs, fit_both, n_resamples=draws, random_state=8,
                              allow_missing=True)
    labels = ("specparam offset", "specparam exponent", "specparam raw knee",
              "specparam knee (Hz)", "specparam beta center (Hz)", "specparam beta height (log10)",
              "specparam beta bandwidth (Hz)", "SL offset", "SL exponent", "SL knee (Hz)",
              "SL beta center (Hz)", "SL beta height (linear density)", "SL beta SD (Hz)")
    for i, label in enumerate(labels):
        print(f"{label}: {boot.estimate[i]:.4g} [{boot.lower[i]:.4g}, "
              f"{boot.upper[i]:.4g}], {boot.n_valid[i]}/{draws} finite")
    # Each method has its own parameterization. Examine parameter covariance
    # within method. For specparam, use only draws that detected a beta peak.
    for name, pair in (("specparam", boot.draws[:, [1, 5]]), ("SL", boot.draws[:, [8, 11]])):
        valid = pair[np.isfinite(pair).all(axis=1)]
        if len(valid) > 1 and (valid.std(axis=0) > 0).all():
            print(f"{name} exponent/height correlation: {np.corrcoef(valid.T)[0, 1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=99)
    main(parser.parse_args().draws)
