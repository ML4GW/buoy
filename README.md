# buoy
![PyPI - Version](https://img.shields.io/pypi/v/ml4gw-buoy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ml4gw-buoy) 
![GitHub License](https://img.shields.io/github/license/ML4GW/buoy)
![Test status](https://github.com/ML4GW/buoy/actions/workflows/unit-tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/ML4GW/buoy/branch/main/graph/badge.svg)](https://codecov.io/gh/ML4GW/buoy)

**buoy** deploys trained [Aframe](https://github.com/ML4GW/aframe) and [AMPLFI](https://github.com/ML4GW/amplfi) models on gravitational wave events, producing detection statistics and parameter estimation outputs.

- **Aframe** is a neural network that scans strain data and assigns a detection statistic at each time step. buoy runs it over a segment of data surrounding an event and reports the integrated detection statistic alongside the raw network output.
- **AMPLFI** is a normalizing flow that performs rapid Bayesian parameter estimation. Given a coalescence time (inferred from Aframe or provided directly), it generates posterior samples for intrinsic and extrinsic parameters and produces a sky localization map.

Model weights (~320 MB total) are downloaded automatically from [HuggingFace](https://huggingface.co/ML4GW) on first use and cached locally.

**Documentation:** https://ml4gw.github.io/buoy/

---

## Installation

```bash
pip install ml4gw-buoy
```

A virtual environment is recommended:

```bash
conda create -n buoy python=3.11
conda activate buoy
pip install ml4gw-buoy
```

### Analyzing unreleased data

Open data (O1–O4a) is fetched automatically. For events from data not yet publicly released, frame-discovery dependencies are required. A pre-built container with those dependencies is available:

```bash
apptainer pull buoy.sif docker://ghcr.io/ml4gw/buoy/buoy:latest
```

---

## Supported event types

| Format | Example | Source |
|--------|---------|--------|
| GWTC catalog event | `GW150914` | [GWOSC](https://gw-openscience.org) |
| GraceDB event | `G363842` | [GraceDB](https://gracedb.ligo.org) (requires LIGO credentials) |
| GraceDB superevent | `S200213t` | [GraceDB](https://gracedb.ligo.org) (requires LIGO credentials) |
| GPS time | `1187008882.4` | User-supplied |

When using a GPS time, buoy defaults to fetching data for H1, L1, and V1. Use `--ifos` to restrict the detector set.

---

## Usage

Exactly one event source must be supplied: explicit event name(s) via `--events`, an observing run via `--observing_runs`, or a GPS window via `--gps_start`/`--gps_end`.

### Single event

```bash
buoy --events GW150914 --outdir ./results
```

### Multiple events

```bash
buoy --events '["GW190521", "GW190828_063405", "S200213t"]' --outdir ./results
```

### All public events from an observing run

```bash
buoy --observing_runs '["O3a"]' --outdir ./results
```

### All public events in a GPS time window

```bash
buoy --gps_start 1126051217 --gps_end 1137254417 --outdir ./results
```

### GPS time event

```bash
buoy --events 1187008882.4 --outdir ./results --ifos '["H1", "L1"]'
```

### Parallel processing

Use `--max_workers` to process multiple events concurrently. Data fetching for the next event overlaps with model inference for the current one:

```bash
buoy --observing_runs '["O3a"]' --outdir ./results --max_workers 4
```

### Config file

All arguments can be stored in a YAML config file:

```yaml
# config.yaml
events:
  - GW190521
  - GW190814
outdir: ./results
samples_per_event: 10000
device: cuda
```

```bash
buoy --config config.yaml
```

---

## Output structure

```
<outdir>/
└── <event>/
    ├── data/
    │   ├── <event>.hdf5          # Raw strain data
    │   ├── aframe_outputs.hdf5     # Aframe times, detection statistics, integrated outputs
    │   ├── posterior_samples.dat # AMPLFI posterior samples
    │   ├── whitened_data.npy     # Whitened strain used for plotting
    │   └── amplfi_<HL|HLV>.fits  # Skymap in FITS format
    └── plots/
        ├── aframe_response.png   # Detection statistic vs. time with whitened strain
        ├── H1_qtransform.png     # Q-transform for H1
        ├── L1_qtransform.png     # Q-transform for L1
        ├── skymap_<HL|HLV>.png   # Mollweide sky localization map
        └── corner_plot_<HL|HLV>.png  # Corner plot of posterior samples
```

---

## CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--events` | — | Event name(s) or GPS time(s) to analyze |
| `--observing_runs` | — | GWOSC run label(s) (e.g. `["O3a", "O3b"]`); fetches all public events |
| `--gps_start` | — | Start of GPS window; fetches all public events in `[gps_start, gps_end]` |
| `--gps_end` | — | End of GPS window; must be paired with `--gps_start` |
| `--outdir` | *(required)* | Directory to write results |
| `--samples_per_event` | `20000` | Number of AMPLFI posterior samples |
| `--nside` | `64` | HEALPix resolution for the skymap |
| `--min_samples_per_pix` | `5` | Minimum samples per pixel for distance ansatz |
| `--use_distance` | `true` | Include distance in the 3D skymap |
| `--aframe_weights` | HuggingFace | Path to Aframe TorchScript weights (`.pt`) |
| `--amplfi_hl_weights` | HuggingFace | Path to AMPLFI HL checkpoint (`.ckpt`) |
| `--amplfi_hlv_weights` | HuggingFace | Path to AMPLFI HLV checkpoint (`.ckpt`) |
| `--aframe_config` | HuggingFace | Path to config of Aframe model config (`.yaml`) |
| `--amplfi_hl_config` | HuggingFace | Path to config of AMPLFI HL model config (`.yaml`) |
| `--amplfi_hlv_config` | HuggingFace | Path to config of AMPLFI HLV model config (`.yaml`) |
| `--aframe_revision` | default branch | HuggingFace revision for Aframe weights |
| `--amplfi_revision` | default branch | HuggingFace revision for AMPLFI weights |
| `--use_true_tc_for_amplfi` | `false` | Use catalog/GraceDB time instead of Aframe-inferred time |
| `--ifos` | `["H1","L1","V1"]` | Detectors to use for GPS time events |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | `None` | Random seed for AMPLFI reproducibility |
| `--verbose` | `false` | Enable DEBUG-level logging |
| `--run_aframe` | `true` | Run Aframe inference; if false, load saved outputs |
| `--run_amplfi` | `true` | Run AMPLFI inference; if false, skip PE |
| `--generate_plots` | `true` | Generate output plots |
| `--force` | `false` | Reprocess events even if outputs already exist |
| `--max_workers` | `1` | Number of events to process concurrently |
| `--corner_parameters` | see below | Parameters to include in the corner plot |
| `--to_html` | `false` | Generate an HTML summary page |
| `--config` | — | Path to a YAML config file |

Default corner plot parameters: `chirp_mass`, `mass_ratio`, `distance`, `mass_1`, `mass_2`.
