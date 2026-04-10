# Installation

```bash
pip install ml4gw-buoy
```

A virtual environment is recommended:

```bash
conda create -n buoy python=3.11
conda activate buoy
pip install ml4gw-buoy
```

## Analyzing unreleased data

Open data (O1–O4a) is fetched automatically. For events from data not yet
publicly released, frame-discovery dependencies are required. A pre-built
container with those dependencies is available:

```bash
apptainer pull buoy.sif docker://ghcr.io/ml4gw/buoy/buoy:latest
```
