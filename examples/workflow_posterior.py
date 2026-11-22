"""Sample the SL_specdecomp posterior on the same observed PSD used for MAP."""
from pathlib import Path
import json
import numpy as np
import arviz as az
from SL_specdecomp import Decompose

OUT = Path(__file__).resolve().parent / "out"
VARS = ["b_0", "chi_0", "knee_hz_0", "center_0", "A_lin_0", "sigma_0"]


def main():
    data = np.load(OUT / "spectra.npz")
    f, psd = data["fa"], data["estimate"]
    band = (f >= 3) & (f <= 45)
    result = Decompose(f[band], psd[band], mode="additive", n_rhythms=1,
                       rhythm_bands=[(15, 25)], k_tapers=3,
                       sample_kwargs=dict(inference_method="sample", nuts_sampler="pymc",
                                          chains=4, cores=2, tune=1500, draws=1500,
                                          random_seed=918, target_accept=0.95, progressbar=False))
    draws = result.idata.posterior[VARS]
    diag = az.summary(draws, var_names=VARS, kind="diagnostics", round_to=5)
    diag.to_csv(OUT / "nuts_diagnostics.csv")
    stats = result.idata.sample_stats
    audit = dict(chains=4, draws_per_chain=1500, warmup_per_chain=1500, seed=918,
                 divergences=int(stats.diverging.sum()), max_rhat=float(diag.r_hat.max()),
                 min_ess_bulk=float(diag.ess_bulk.min()), min_ess_tail=float(diag.ess_tail.min()),
                 bfmi_by_chain=az.bfmi(result.idata).tolist())
    if "reached_max_treedepth" in stats:
        audit["max_tree_depth_hits"] = int(stats.reached_max_treedepth.sum())
    samples = np.stack([draws[k].values for k in VARS], axis=-1)
    assert samples.shape == (4,1500,6) and np.isfinite(samples).all()
    np.savez_compressed(OUT / "posterior.npz", draws=samples, variables=VARS,
                        f_fit=f[band], psd_fit=psd[band], divergences=stats.diverging.values)
    (OUT / "nuts_run.json").write_text(json.dumps(audit, indent=2))
    print(diag, audit, flush=True)


if __name__ == "__main__":
    main()
