# Usage

## Supported event types

| Format | Example | Source |
|--------|---------|--------|
| GWTC catalog event | `GW150914` | [GWOSC](https://gw-openscience.org) |
| GraceDB event | `G363842` | [GraceDB](https://gracedb.ligo.org) (requires LIGO credentials) |
| GraceDB superevent | `S200213t` | [GraceDB](https://gracedb.ligo.org) (requires LIGO credentials) |
| GPS time | `1187008882.4` | User-supplied |

When using a GPS time, buoy defaults to fetching data for H1, L1, and V1.
Use `--ifos` to restrict the detector set.

## CLI

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

### GPS time

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

## Python API

```python
from buoy.main import main

# Single event
main(events="GW150914", outdir="./results", device="cuda")

# All public events from a run, processed in parallel
main(observing_runs=["O3a"], outdir="./results", device="cuda", max_workers=4)
```

Models can also be instantiated directly for custom workflows:

```python
from buoy import Aframe, Amplfi

aframe = Aframe(device="cuda")
amplfi = Amplfi(config="path/to/config.yaml", load_weights=False)
print(aframe.sample_rate, amplfi.kernel_length)
```

## Output structure

```
<outdir>/
└── <event>/
    ├── data/
    │   ├── <event>.hdf5            # Raw strain data
    │   ├── aframe_outputs.hdf5     # Aframe times, detection statistics
    │   ├── posterior_samples.dat   # AMPLFI posterior samples
    │   ├── whitened_data.npy       # Whitened strain used for plotting
    │   └── amplfi_<HL|HLV>.fits   # Skymap in FITS format
    └── plots/
        ├── aframe_response.png     # Detection statistic vs. time
        ├── H1_qtransform.png       # Q-transform for H1
        ├── L1_qtransform.png       # Q-transform for L1
        ├── skymap_<HL|HLV>.png     # Mollweide sky localization map
        └── corner_plot_<HL|HLV>.png  # Corner plot of posterior samples
```

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
| `--aframe_config` | HuggingFace | Path to Aframe model config (`.yaml`) |
| `--amplfi_hl_config` | HuggingFace | Path to AMPLFI HL model config (`.yaml`) |
| `--amplfi_hlv_config` | HuggingFace | Path to AMPLFI HLV model config (`.yaml`) |
| `--aframe_revision` | default branch | HuggingFace revision for Aframe weights |
| `--amplfi_revision` | default branch | HuggingFace revision for AMPLFI weights |
| `--model_cache_dir` | `~/.cache/huggingface` | Local HuggingFace cache directory |
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
