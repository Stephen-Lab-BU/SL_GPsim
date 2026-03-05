# SL_GPsim Package

A Python package to simulate Gaussian time-domain signals from theoretical power spectra (1/f + peaks), and optionally estimate PSDs empirically with [spectral_connectivity](https://github.com/Eden-Kramer-Lab/spectral_connectivity).

## Installation

```bash
# 1) Ensure you have conda installed (e.g., Miniconda or Anaconda).
#    Then install mamba into the 'base' environment:
conda install mamba -n base -c conda-forge

# 2) Clone your repository:
git clone https://github.com/Stephen-Lab-BU/SL_GPsim
cd SL_GPsim

# 3) Create and activate your environment with mamba:
#    (Assumes environment.yml is in this repo)
mamba env create -f environment.yml
mamba activate SL_GPsim

# 4) (Optional) Install your local package in editable mode
#    If environment.yml doesn't already do so:
python -m pip install -e .

# or if you want to install from GitHub directly:
# python -m pip install git+https://github.com/Stephen-Lab-BU/SL_GPsim

```

## Basic Usage
#### 1. Simple Simulation

```python
from SL_GPsim import spectrum

# Simulate 2 seconds at 500 Hz, with a 1/f exponent=2.0, offset=1.0
# plus a peak at 10 Hz (amplitude=50, sigma=2).
res = spectrum(
    sampling_rate=500,
    duration=2.0,
    aperiodic_exponent=2.0,
    aperiodic_offset=1.0,
    knee=None,
    peaks=[{'freq':10, 'amplitude':50.0, 'sigma':2.0}],
    average_firing_rate=0.0,
    random_state=42,
    direct_estimate=False,  # skip empirical PSD
    plot=True
)

# Access the time-domain data
time_data = res.time_domain
print("Time-domain samples:", len(time_data))
print("Mean amplitude:", time_data.combined_signal.mean())

# Access the frequency-domain data
freq_data = res.frequency_domain
print("Number of freq bins:", len(freq_data))
```

#### 2. Empirical PSD Estimation

```python
res_emp = spectrum(
    sampling_rate=500,
    duration=10.0,
    aperiodic_exponent=1.5,
    aperiodic_offset=1.0,
    knee=10.0,
    peaks=[{'freq':12, 'amplitude':10.0, 'sigma':3.0}],
    average_firing_rate=0.0,
    random_state=0,
    direct_estimate=True,   # requires spectral_connectivity
    plot=True
)
# This will show both the theoretical and the empirically estimated PSD.

#### 3. Piecewise‑Stationary Spectrogram Simulation

```python
from SL_GPsim import spectrogram

# Simulate a spectrogram consisting of 5 stationary windows, each 2 seconds long,
# sampled at 500 Hz.  Here the broadband exponent drifts linearly across
# windows and the rhythmic peak amplitude oscillates with window index.

def exp_fn(i, t):
    # linearly increase exponent from 1.0 to 2.0 across windows
    return 1.0 + i * 0.25

def peaks_fn(i, t):
    # peak amplitude varies sinusoidally with index
    amp = 10.0 + 5.0 * np.sin(2 * np.pi * i / 4)
    return [{'freq':10, 'amplitude':amp, 'sigma':2.0}]

res_spec = spectrogram(
    sampling_rate=500,
    n_windows=5,
    window_duration=2.0,
    step_duration=None,  # non‑overlapping windows
    aperiodic_exponent_fn=exp_fn,
    peaks_fn=peaks_fn,
    mode="additive",
    random_state=42,
    direct_estimate=True,  # requires spectral_connectivity
)

# Access stitched time‑domain signal
td = res_spec.time_domain
print("Stitched samples:", len(td))

# Access theoretical spectra (per window)
theo = res_spec.theoretical['combined']
print("Shape of theoretical spectra:", theo.shape)  # (n_windows, n_freqs)

# Access empirical multitaper spectra (per window)
emp = res_spec.empirical
freqs_emp = res_spec.frequencies_empirical
print("Empirical spectrogram shape:", emp.shape)

```

In this example, `spectrogram` returns a dataclass containing the stitched
signal, per‑window theoretical and empirical spectra, the empirical
frequency grid, and the raw windowed signals.  Empirical PSDs are
computed using the multitaper method from the
`spectral_connectivity` package, and the frequency grid for the empirical
spectra may differ from the theoretical grid.  The raw windowed signals
are provided so that users can perform their own spectral analyses or
plot per‑window signals.
```