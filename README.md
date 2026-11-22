# SL_GPsim

Simulate stationary Gaussian time series from a power spectrum or a complex cross-spectral matrix. Estimate a multitaper spectrum from the recording, or bootstrap a spectral decomposition by generating and refitting whole records.

The new interface is on `feature/cross-spectral-bootstrap` (package version 0.2.0). The `SL_specdecomp-paper` tag identifies the earlier simulator at commit `f20946a`. To reproduce that version, check out the tag and use its own environment file.

## Install with mamba

From a terminal with mamba available, run

```bash
git clone --branch feature/cross-spectral-bootstrap https://github.com/Stephen-Lab-BU/SL_GPsim.git
cd SL_GPsim
mamba env create -f environment.yml
mamba activate sl-gpsim-bootstrap
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

The environment specifies Python and direct scientific dependencies by version, uses conda-forge, and installs spectral-connectivity at a fixed PyPI version. It includes a Jupyter kernel. See the validation section for the mathematical checks. The specparam and SL_specdecomp fitters are optional.

For an exact base environment, choose the explicit lock for your platform
from `locks/`. For example, on Linux x86-64 use

```bash
mamba create -n sl-gpsim-locked --file locks/linux-64.txt
mamba activate sl-gpsim-locked
python -m pip install --only-binary=:all: --no-deps --require-hashes -r locks/pip-wheels.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

The explicit files pin conda package builds by SHA-256. The separate wheel
lock pins spectral-connectivity. Linux was built and tested. The macOS locks
were cross-solved for macOS 14; execution was tested on Linux only.

## Simulate a rhythm over a broadband knee

```python
import numpy as np
from SL_GPsim import Broadband, GaussianPeak, SpectralGrid, multitaper_csd

grid = SpectralGrid(fs=128, duration=8)                      # Hz, seconds
background = Broadband(scale=8, exponent=1.6, knee=2)
beta = GaussianPeak(center_hz=20, sigma_hz=1.8, height=0.70)
single = background + beta

x = grid.simulate(single, random_state=7)                    # (1024, 1)
t = grid.times                                               # seconds
f, observed = multitaper_csd(x, fs=grid.fs)                  # (513, 1, 1)
truth_for_plot = single(np.linspace(0, 64, 10_001))          # arbitrary frequencies
simulation_target = grid.sample(single)                      # on grid.frequencies
```

The broadband model is `scale / (knee + f**exponent)`. In this example the raw coefficient 2 gives a half-power knee of 1.54 Hz. Use `knee_hz` to specify that frequency directly. The additive Gaussian height is a one-sided linear density per Hz and its width is a standard deviation in Hz. For multiplicative log-Gaussian peaks, use `LogPeaks(background, peaks)`, which evaluates `background(f) * 10**peaks(f)`.

Each component and the combined model accept a scalar or vector of nonnegative frequencies. Save `single.parameters` as JSON and pass the decoded dictionary to `model_from_parameters()` to reconstruct it. The sampling grid also accepts `n_samples` for an exact count. Specifying a fractional number of samples via duration rounds to the nearest sample (half up) and emits a warning. Use `grid.duration` for the realized duration.

## Simulate channels with a shared rhythm

```python
from SL_GPsim import Independent, Shared, simulate_like

private = (Broadband(scale=5.4, exponent=1.4, knee=4)
           + GaussianPeak(center_hz=10, sigma_hz=1.1, height=0.44))
pair = (Independent(background, private)
        + Shared(beta, gains=[1, 0.7], delays_seconds=[0, 0.015625]))
xy = grid.simulate(pair, random_state=112)                   # (1024, 2)
S20 = pair(20)                                               # 2 × 2 CSD at 20 Hz
freqs, estimate = multitaper_csd(xy, fs=grid.fs)

# Starting from a measured time-by-channel recording takes one call.
replicate = simulate_like(xy, fs=grid.fs, random_state=42)
```

Each call to `Shared` supplies one independent latent source. A gain of 0.7 contributes 0.49 times the source power. Here channel 2 lags by 15.625 ms, so the phase of `S20[1, 0]` is −112.5°. A [runnable example](examples/simulate_spectra.py) produces both records and their ground truth.

## Bootstrap a statistic or decomposition

```python
from SL_GPsim import spectral_bootstrap

def beta_band_power(f, csd):
    band = (f >= 15) & (f < 25)
    auto = csd.diagonal(axis1=1, axis2=2).real
    return auto[band].sum(axis=0) * (f[1] - f[0])

result = spectral_bootstrap(xy, fs=128, statistic=beta_band_power,
                            n_resamples=999, random_state=8)
print(result.estimate, result.lower, result.upper)           # one entry per channel
print(np.cov(result.draws.T))                                # paired across channels
```

The default pilot is the full multitaper CSD of the observed record evaluated on the embedding grid. Each replicate generates a time series of the original length, reapplies the estimator, and calls the statistic on its frequency grid and CSD. The defaults use equal-weight DPSS with NW=2 and K=3. Pass a fitted generating model as `pilot_csd`, or pass a custom `estimator(data, fs, n_fft)` that returns frequencies and a one-sided CSD. Apply fit-band masks in the statistic.

The returned limits are marginal percentile confidence intervals conditional on the pilot and stationary Gaussian assumption. The pilot includes estimator smoothing and leakage, and re-estimation can smooth a second time. A K-taper pilot has rank at most K in each bin. For intermittent fitted peaks, return NaN for a missing component and set `allow_missing=True`. Report `result.n_valid` with each conditional interval. A thrown fit error identifies the failing replicate.

## Endpoint and phase conventions

- Analytic models give smooth one-sided densities at arbitrary frequencies. Simulation weights DC and even-grid Nyquist by one half. On an odd embedding the final positive bin receives full weight. The grid warns about undersampled peaks, knees outside the grid range and Gaussian peaks crossing the frequency boundaries.
- Raw inputs to `simulate_from_csd`, including ordinary callbacks, must supply already weighted one-sided densities on the full rFFT grid. A multitaper CSD computed with `n_fft=grid.n_fft` can be passed directly. Invalid DC or Nyquist values raise a diagnostic. For a custom smooth univariate formula, `grid.density(lambda f: ...)` handles weighting.
- CSD matrices have axes (frequency, channel, channel) and use `S[i,j] = X_i * conj(X_j)` in density units. This corresponds to SciPy's `csd(x_j, x_i)`. Every matrix must be Hermitian positive semidefinite. DC and even-grid Nyquist must be real. For a fractional relative delay at even-grid Nyquist the shared-source helper warns and uses the real cosine cross-power, which can reduce coherence at that bin.
- The default embedding is twice the record length. With the fitted DPSS pilot it reproduces the taper-weighted linear covariance for observed lags. For a continuous generating function, sampling the frequency grid approximates its covariance integral. A denser embedding can improve this approximation. The direct estimate's resolving power depends on record length and taper bandwidth.

## Mathematical validation

Run `python -m pytest -q` from the checkout. The suite includes the following independent references.

| Quantity | Reference and check |
| --- | --- |
| Full finite-record covariance | A two-channel moving-average process with known time-domain coefficients. The exact covariance of the implemented Gaussian draw matches the time-domain matrix for every sample pair on even and odd embeddings. |
| Cross-phase and lag | White-noise channels with a known two-sample delay. Simulated lagged cross-products recover the direction and magnitude. |
| Multitaper density | Directly averaged SciPy tapered cross-periodograms for odd and even FFTs, including phase and physical units. |
| Fitted-pilot embedding | Direct lagged products of tapered time series for every observed lag and channel pair. |
| Bootstrap band power | For unit Gaussian white noise the band integral is a quadratic form. Bootstrap mean and variance are compared to its analytic moments with a tolerance in Monte Carlo standard errors. |
| Edges and parameter handling | DC-only and Nyquist-only processes, silent and perfectly coherent channels, invalid matrices, conditional missing-peak intervals, and model export/import. |

To measure repeated-record coverage for the supplied broadband benchmark, run `python examples/bootstrap_coverage.py --records 200 --draws 199`. This benchmark fits a free exponent with a fixed no-knee background form and known peak center and width. It reports coverage, a Monte Carlo interval, mean width, and bias for each scenario. Coverage of a particular knee or peak estimator requires a study under its own generating and fitting conditions.

## Optional decomposition and posterior examples

Install Git and the pinned example dependencies in the active environment.

```bash
python -m pip install -r examples/requirements-decomposition.txt
python examples/bootstrap_decomposition.py --draws 99
```

The paired example fits specparam in knee mode and SL_specdecomp using additive Gamma MAP (K=3) to the same 3–45 Hz estimates. It reports offsets, exponents, knees, beta centers, heights, widths, finite-fit counts, and exponent–height correlations. Specparam's positive half-power knee and detected beta parameters have their own denominators. Its height is in log10 power, while the SL height is an additive density.

The four figure scripts reproduce the synthetic ground truth, time series, direct multitaper estimates, paired bootstrap intervals and original-record SL posterior. Run them in this order. Outputs are written to `examples/out/`.

```bash
python examples/workflow_data.py        # 999 paired refits
python examples/workflow_posterior.py   # four NUTS chains
python examples/workflow_summary.py     # intervals and posterior comparison CSVs
python examples/workflow_figures.py     # four-page PDF and separate figures
```

The bootstrap describes estimates across regenerated records. The posterior conditions on the original estimated spectrum, likelihood, and priors. The NUTS workflow saves convergence diagnostics with the draws.

## Earlier examples

The original `spectrum()` and `spectrogram()` interfaces are available. The [simulation notebook](examples/SL_GPsim_Examples.ipynb) and [multichannel propofol notebook](examples/propofol_32ch_sim.ipynb) accompany the paper-era API.

## Method references

- [Percival & Constantine (2006)](https://doi.org/10.1007/s11222-006-5198-0)
  derive the finite-record embedding for nonparametric spectral estimates.
- [Dai & Guo (2004)](https://doi.org/10.1093/biomet/91.3.629) generate
  multivariate records using a factor of the spectral matrix. Here the pilot
  is DPSS or user supplied; a spectral eigenfactor handles singular matrices.
- [Kessy, Lewin & Strimmer (2018)](https://doi.org/10.1080/00031305.2016.1277159)
  describe the freedom in whitening transforms. Forward simulation uses
  coloring: any factor satisfying `L @ L.conj().T == S` gives the required
  Gaussian covariance, including when an inverse does not exist.

### Reproduce the Figure 5 rhythm-amplitude experiment

This example regenerates 30 independent 30 s recordings at 1,000 Hz for each
rhythm amplitude (0.5, 1, 2, 4). It uses the one-channel generator and seed
stream from the [paper analysis](https://github.com/Stephen-Lab-BU/Bloniasz_Stephen_Estimation/blob/6b740a44ba4f336d44f313e2fd37cc9063a78017/Scripts/AnalysisScripts/06_supp_figure_height_grid_benchmark.py).
The additive 8 Hz peak has a 3 Hz standard deviation. The broadband is
`10**0.5 / (60 + f**2)`. Its OLS log–log slope over 40–60 Hz is −1.951
at every amplitude.

The example applies the new time-domain bootstrap to these recordings. It
fits specparam (knee mode, at most one peak), Gamma-multiplicative
SL_specdecomp and Gamma-additive SL_specdecomp on the **same** 1–200 Hz
band and 0.1333 Hz grid. The equal-weight multitaper estimator uses NW=2
and K=3. Fits use the positive half of the two-sided density, as in the
paper. The bootstrap pilot itself retains the full one-sided density.
Both SL estimators use the pinned package's data-scaled prior defaults with
the paper's slope, knee and width bounds. The published analysis used
posterior-mean broadband curves, an SL fitting grid extending to 500 Hz
and additional prior overrides. These are new fits of the published design.

With the optional decomposition environment active, run the seven-page figure
set from the repository root. Commands below reproduce the included draw counts.
The per-record loops can run independently. Outputs go to `examples/out/figure5`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python examples/figure5_data.py controls
for level in 0 1 2 3; do
  python examples/figure5_data.py population --level "$level"
  python examples/figure5_data.py bootstrap --level "$level"
  python examples/figure5_data.py posterior_add --level "$level"
  python examples/figure5_data.py posterior_mul --level "$level"
  for rep in $(seq 1 29); do
    python examples/figure5_data.py bootstrap_spec --level "$level" --rep "$rep"
  done
done
python examples/figure5_figures.py
```

- **Across recordings.** Each raster dot is an observed fit. The high-minus-low
  contrast resamples 30 independent recordings per endpoint, with the same
  resampled records used for all three methods.
- **Within one recording.** Record 1 was selected in advance at each amplitude.
  Each of its 399 Gaussian bootstrap time series is retapered and refitted by
  all three methods. Violin halves compare the empirical distribution of MAP
  refits with posterior draws from SL fitted to that same recording. All 120
  specparam fits also have record-specific nominal 95% percentile intervals.
  Nonpositive raw specparam knees have no positive half-power frequency.
  The tables include finite counts and retain the raw coefficient.
- **Mechanism controls.** Fit the exact additive spectrum, a matched
  multiplicative spectrum and the isolated broadband. A joint log-power
  least-squares fit tests the influence of specparam's fitting procedure while
  keeping the log-Gaussian peak family. Direct joint fitting constrains the knee
  coefficient to be nonnegative and the exponent to 0.01–6. Specparam uses its
  native bounds. Inspect the control estimates before generalizing.

The bootstrap assumes a stationary Gaussian record conditional on the
specified pilot spectrum. Its interval describes variation in a refitted
statistic. A posterior credible interval conditions on the observed spectrum,
the Gamma likelihood and the SL priors. Both can miss a generating component
when the fitted mean is misspecified. The CSV tables report fit denominators
and MCMC diagnostics alongside the figures.
