"""Analytic and paired-resampling regression checks for the Figure 5 estimand."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from figure5_data import AMPS, broadband, fit_grid, metrics, seeds, slope, truth
from figure5_figures import contrasts


def test_broadband_slope_is_invariant_to_amplitude_and_scale():
    f = np.arange(8, 1501) * (4 / 30)
    expected = np.polyfit(np.log10(f[(f >= 40) & (f <= 60)]),
                          np.log10(truth(f[(f >= 40) & (f <= 60)], 0)[0]), 1)[0]
    assert -1.952 < expected < -1.950
    for amp in AMPS:
        bg, _ = truth(f, amp)
        np.testing.assert_allclose(slope(f, bg), expected, atol=1e-12)
    params = metrics(f, np.array([.5, 2.5]), np.array([2., 2.]), np.array([60.,60.]),
                     *[np.ones(2) for _ in range(4)])
    np.testing.assert_allclose(params[:, 0], expected, atol=1e-12)
    np.testing.assert_allclose(broadband(f, .5, 2, 60), bg)


def test_seed_order_and_paper_psd_convention():
    rng = np.random.default_rng(np.random.SeedSequence(12345).spawn(4)[0])
    scalar_stream = [rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32) for _ in range(120)]
    np.testing.assert_array_equal(seeds().ravel(), scalar_stream)
    f = np.fft.rfftfreq(30000, .001)
    g, x = fit_grid(f, np.full((len(f),1,1), 2.))
    np.testing.assert_allclose(np.diff(g), 4 / 30)
    assert 1 <= g[0] < 1.14 and g[-1] == 200
    np.testing.assert_allclose(x, 1.)


def test_mean_contrast_preserves_method_pairing():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(4,30))
    records = np.stack((base, base+9, np.zeros_like(base)), axis=2)[...,None]
    d = contrasts(records, n=200)
    np.testing.assert_allclose(d[:,0], d[:,1], atol=1e-12)
    np.testing.assert_array_equal(d[:,2], 0)
    assert d[:,0].std() > 0
