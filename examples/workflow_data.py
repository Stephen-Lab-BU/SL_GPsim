"""Generate shared input data for the two SL_GPsim workflow figures."""
from pathlib import Path
import json
import csv
import importlib.metadata

import numpy as np
from SL_GPsim import multitaper_csd, spectral_bootstrap
from simulate_spectra import figure_records
from bootstrap_decomposition import fit_both

OUT = Path(__file__).resolve().parent / "out"
FS, N, B = 128, 1024, 999
methods = ["specparam"] * 7 + ["SL_specdecomp MAP"] * 6
names = ["offset", "exponent", "knee_k", "knee_Hz", "center_Hz", "peak_height_log10", "bandwidth_Hz",
         "b_0", "chi_0", "knee_Hz", "center_Hz", "A_linear", "sigma_Hz"]


def generate():
    OUT.mkdir(exist_ok=True)
    f, background, beta, true, matrix, x, x2 = figure_records()
    cross = matrix[:,1,0]
    fa, estimate = multitaper_csd(x, FS, time_halfbandwidth=2, n_tapers=3)
    fm, sm = multitaper_csd(x2, FS, time_halfbandwidth=2, n_tapers=3)
    assert np.min(np.linalg.eigvalsh(matrix)) >= 0
    assert np.allclose(np.angle(cross[np.argmin(abs(f-20))], deg=True), -112.5)
    assert np.array_equal(fm, fa)
    np.savez_compressed(OUT/"spectra.npz", f=f, fa=fa, background=background, beta=beta, true=true,
                        x=x, estimate=estimate[:,0,0].real, matrix=matrix, x2=x2, sm=sm)
    boot = spectral_bootstrap(x, FS, fit_both, n_resamples=B, random_state=8,
                              time_halfbandwidth=2, n_tapers=3, allow_missing=True)
    assert boot.draws.shape == (B, len(names))
    np.testing.assert_array_equal(boot.estimate, fit_both(fa, estimate))
    np.testing.assert_allclose(np.nanquantile(boot.draws, [.025,.975],axis=0), [boot.lower,boot.upper])
    np.savez_compressed(OUT/"parameters.npz", draws=boot.draws, observed=boot.estimate,
                        lo=boot.lower, hi=boot.upper, valid=boot.n_valid)
    with (OUT/"intervals.csv").open("w") as stream:
        w=csv.writer(stream); w.writerow(["method","parameter","observed","lower_2.5pct","upper_97.5pct","valid","attempted"])
        w.writerows(zip(methods,names,boot.estimate,boot.lower,boot.upper,boot.n_valid,[B]*len(names),strict=True))
    print((OUT/"intervals.csv").read_text())
    print("SL knee at 1-Hz lower bound:", np.isclose(boot.draws[:,9],1,rtol=1e-6).sum(), "/", B)
    (OUT/"versions.json").write_text(json.dumps({p:importlib.metadata.version(p) for p in
        ("specparam","SL_specdecomp","SL_GPsim","numpy","scipy","matplotlib","pymc","arviz")},indent=2))


if __name__ == "__main__":
    generate()
