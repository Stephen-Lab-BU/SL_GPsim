"""Write exact intervals and Monte Carlo diagnostics alongside the figures."""
from pathlib import Path
import csv
import numpy as np
from workflow_data import names, methods

OUT = Path(__file__).resolve().parent/"out"
b = np.load(OUT/"parameters.npz")
p = np.load(OUT/"posterior.npz")
s = np.load(OUT/"spectra.npz")
f = s["fa"]; mask = (f >= 3) & (f <= 45)
np.testing.assert_array_equal(p["f_fit"], f[mask])
np.testing.assert_array_equal(p["psd_fit"], s["estimate"][mask])
post = p["draws"].reshape(-1,6)
assert len(names) == len(methods) == b["draws"].shape[1] == 13
with (OUT/"intervals.csv").open("w") as stream:
    w = csv.writer(stream)
    w.writerow(["method","parameter","observed","lower_2.5pct","upper_97.5pct","valid","attempted"])
    w.writerows(zip(methods,names,b["observed"],b["lo"],b["hi"],b["valid"],
                    [len(b["draws"])]*len(names),strict=True))
with (OUT/"posterior_comparison.csv").open("w") as stream:
    w = csv.writer(stream)
    w.writerow(["SL_parameter","MAP","bootstrap_CI_lo","bootstrap_CI_hi","posterior_median","credible_lo","credible_hi"])
    for i, k in enumerate(p["variables"]):
        lo,med,hi = np.quantile(post[:,i],[.025,.5,.975])
        j = 7+i
        w.writerow([k,b["observed"][j],b["lo"][j],b["hi"][j],med,lo,hi])
