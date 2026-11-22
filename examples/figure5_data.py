"""Figure 5 rhythm-height experiment, record-wise bootstrap and posterior comparison.

Run from the checkout in the optional decomposition environment. Outputs are keyed
by amplitude and record; the first published seed in each condition is the
preselected recording for the within-record comparison.
"""
import argparse
import csv
from contextlib import redirect_stdout
from functools import lru_cache
import io
from pathlib import Path

import numpy as np
from SL_GPsim import multitaper_csd, spectral_bootstrap, spectrum

OUT = Path(__file__).parent / "out" / "figure5"
AMPS = (0.5, 1., 2., 4.)
FS, SECONDS, N_RECORDS = 1000, 30, 30
METHODS = ("specparam", "Gamma multiplicative", "Gamma additive")
MODES = ("FOOOF_spectrum", "additive")
FIELDS = ("slope40_60", "log10_numerator", "exponent", "knee_raw", "knee_hz",
          "center_hz", "height_native", "sigma_hz", "bb_at_8", "bb_at_50")
VARS = (("b", "chi", "knee_hz", "center_0", "a_log_0", "sigma_0"),
        ("b_0", "chi_0", "knee_hz_0", "center_0", "A_lin_0", "sigma_0"))
PRIORS = {"knee_hz_bounds": (1., 200.), "slope_mu": -2., "slope_sigma": 1.,
          "slope_bounds": (-5., -.5), "sigma_mu": 3., "sigma_sigma": 2.,
          "sigma_bounds": (.5, 12.), "a_log_sigma": 1.}

def truth(f, amplitude):
    f = np.asarray(f)
    bb = 10**.5 / (60 + f**2)
    rhythm = amplitude * (np.exp(-.5 * ((f - 8) / 3)**2) + np.exp(-.5 * ((f + 8) / 3)**2))
    return bb, rhythm

@lru_cache(maxsize=1)
def seeds():
    rng = np.random.default_rng(np.random.SeedSequence(12345).spawn(4)[0])
    return rng.integers(0, np.iinfo(np.uint32).max, size=(4, 30), dtype=np.uint32)

def record(level, rep):
    return np.asarray(spectrum(sampling_rate=FS, duration=SECONDS, aperiodic_exponent=2.,
        aperiodic_offset=.5, knee=60., peaks=[dict(freq=8., sigma=3., amplitude=AMPS[level])],
        random_state=int(seeds()[level, rep]), mode="additive").time_domain.combined_signal)

def fit_grid(f, csd):
    # The paper plots the positive half of a two-sided density. Convert our
    # one-sided DPSS estimate after thinning. DC and Nyquist are outside fit band.
    f = f[::4]
    keep = (f >= 1) & (f <= 200)
    return f[keep], csd[::4, 0, 0].real[keep] / 2

def broadband(f, b, exponent, k):
    return 10 ** np.asarray(b)[..., None] / (np.asarray(k)[..., None] +
                                              np.asarray(f) ** np.asarray(exponent)[..., None])

def slope(f, curve):
    mask = (f >= 40) & (f <= 60)
    x = np.log10(f[mask])
    dx = x - x.mean()
    return np.log10(curve[..., mask]) @ dx / (dx @ dx)

def metrics(f, b, exp, k, knee, center, height, sigma):
    sub = f[(f >= 40) & (f <= 60)]
    s = slope(sub, broadband(sub, b, exp, k))
    powers = broadband(np.array([8., 50.]), b, exp, k)
    return np.stack((s, b, exp, k, knee, center, height, sigma, powers[..., 0], powers[..., 1]), axis=-1)

def spec_fit(f, psd, *, background_only=False):
    from specparam import SpectralModel
    sp = SpectralModel(aperiodic_mode="knee", peak_width_limits=[1., 30.],
        max_n_peaks=0 if background_only else 1, min_peak_height=0., peak_threshold=2., verbose=False)
    sp.fit(f, psd)
    if not sp.results.has_model:
        return np.full(len(FIELDS), np.nan)
    b, k, exp = sp.results.params.aperiodic.get_params("fit")
    peak = np.asarray(sp.results.params.periodic.get_params("fit")).reshape(-1, 3)
    center, height, sd = peak[0] if len(peak) else (np.nan,)*3
    knee = k ** (1 / exp) if k > 0 and exp > 0 else np.nan
    return metrics(f, b, exp, k, knee, center, height, sd)

def sl_fit(f, psd, model, **kwargs):
    from SL_specdecomp import Decompose
    with redirect_stdout(io.StringIO()):
        return Decompose(f, psd, mode=MODES[model], rhythm_bands=[(1., 20.)],
            k_tapers=3., n_rhythms=1, priors=PRIORS, sample_kwargs=kwargs)

def sl_metrics(f, post, model):
    b, exp, knee, center, height, sd = [post[v].values for v in VARS[model]]
    return metrics(f, b, exp, knee**exp, knee, center, height, sd)

def fit_all(f, csd):
    grid, psd = fit_grid(f, csd)
    result = [spec_fit(grid, psd)]
    for m in range(2):
        post = sl_fit(grid, psd, m, inference_method="fast_map").idata.posterior
        if post.attrs["optimizer_success"]:
            result.append(sl_metrics(grid, post, m).reshape(-1))
        else:
            result.append(np.full(len(FIELDS), np.nan))
    return np.asarray(result)

def estimate(level, rep):
    x = record(level, rep)
    f, csd = multitaper_csd(x, fs=FS, time_halfbandwidth=2, n_tapers=3)
    return x, f, csd

def compare_estimators():
    from spectral_connectivity import Multitaper, Connectivity
    x, f, csd = estimate(0, 0)
    mt = Multitaper(x[:, None, None], sampling_frequency=FS,
                    time_halfbandwidth_product=2, n_tapers=3, time_window_duration=30,
                    time_window_step=30, start_time=0)
    c = Connectivity.from_multitaper(mt)
    grid, p = fit_grid(f, csd)
    actual = np.interp(grid, c.frequencies, c.power().squeeze())
    return float(np.max(abs(p / actual - 1)))

def run(level, kind, args):
    if kind == "population":
        blocks = []
        for rep in range(N_RECORDS):
            _, f, csd = estimate(level, rep)
            blocks.append(fit_all(f, csd))
        np.savez_compressed(OUT / f"records_{level}.npz", fits=blocks, seeds=seeds()[level])
        return
    x, f, csd = estimate(level, args.rep)
    grid, p = fit_grid(f, csd)
    if kind in ("bootstrap", "bootstrap_spec"):
        statistic = fit_all if kind == "bootstrap" else lambda f, c: spec_fit(*fit_grid(f, c))
        prefix = "boot" if kind == "bootstrap" else "spec"
        res = spectral_bootstrap(x, fs=FS, statistic=statistic, n_resamples=args.draws,
            time_halfbandwidth=2, n_tapers=3, allow_missing=True,
            random_state=np.random.SeedSequence([5119, level, args.rep]))
        np.savez_compressed(OUT / f"{prefix}_{level}_{args.rep}.npz", observed=res.estimate,
            draws=res.draws, lower=res.lower, upper=res.upper, valid=res.n_valid, f=grid, psd=p)
    else:
        import arviz as az
        model = 0 if kind == "posterior_mul" else 1
        seed = int(np.random.SeedSequence([4435, level, args.rep, model]).generate_state(1)[0])
        fit = sl_fit(grid, p, model, inference_method="sample", nuts_sampler="pymc", chains=4,
            cores=1, draws=args.posterior_draws, tune=args.tune, target_accept=.95,
            random_seed=seed, progressbar=False)
        idata = fit.idata
        diag = az.summary(idata, var_names=list(VARS[model]), kind="diagnostics", round_to=6)
        dest = OUT / f"post_{level}_{args.rep}_{model}"
        diag.to_csv(dest.with_suffix(".csv"))
        np.savez_compressed(dest.with_suffix(".npz"), draws=sl_metrics(grid, idata.posterior, model),
            divergences=idata.sample_stats.diverging.values, bfmi=az.bfmi(idata),
            depth_hits=int(idata.sample_stats.reached_max_treedepth.sum()), seed=seed)

def joint_log_fit(f, p, initial):
    """Control: minimize log-power residuals jointly in the positive-knee model."""
    from scipy.optimize import least_squares
    def resid(par):
        b, exp, k, cf, h, sigma = par
        return b - np.log10(k + f**exp) + h * np.exp(-.5 * ((f-cf)/sigma)**2) - np.log10(p)
    bounds = ([-np.inf, .01, 0, 1, 0, .5], [np.inf, 6, np.inf, 200, np.inf, 15])
    starts = ([initial[1], initial[2], max(0., initial[3]), *initial[5:8]],
              [.5, 2., 60., 8., np.log10(max(p)/max(truth(f, 0)[0])), 3.])
    opt = min((least_squares(resid, q, bounds=bounds, ftol=1e-12, gtol=1e-12, xtol=1e-12)
               for q in starts), key=lambda q: q.cost)
    if not opt.success:
        raise RuntimeError(opt.message)
    b, exp, k, cf, height, sigma = opt.x
    return metrics(f, b, exp, k, k**(1/exp), cf, height, sigma)

def controls():
    f = np.fft.rfftfreq(SECONDS * FS, 1 / FS)[::4]
    f = f[(f >= 1) & (f <= 200)]
    rows = []
    for a in AMPS:
        bb, rh = truth(f, a)
        # Multiplicative control has the same total power at the 8 Hz center.
        bb8, rh8 = truth(8., a)
        log_peak = np.log10(1 + rh8 / bb8) * np.exp(-.5 * ((f-8.)/3.)**2)
        for label, p, bg_only in (("additive truth", bb+rh, False),
            ("multiplicative truth", bb * 10**log_peak, False), ("known broadband", bb, True)):
            fit = spec_fit(f, p, background_only=bg_only)
            rows.append([a, label, *fit])
            if not bg_only:
                rows.append([a, label + " (joint fit)", *joint_log_fit(f, p, fit)])
    with (OUT / "population_controls.csv").open("w") as s:
        w = csv.writer(s); w.writerow(["amplitude", "control", *FIELDS]); w.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("population", "bootstrap", "bootstrap_spec", "posterior_mul", "posterior_add", "controls"))
    parser.add_argument("--level", type=int, choices=range(4), default=0)
    parser.add_argument("--rep", type=int, choices=range(30), default=0)
    parser.add_argument("--draws", type=int, default=399)
    parser.add_argument("--posterior-draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=800)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.kind == "controls":
        print("Maximum relative PSD difference versus paper estimator:", compare_estimators())
        controls()
    else:
        run(args.level, args.kind, args)
