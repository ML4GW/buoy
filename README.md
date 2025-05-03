# Installation

This library is `pip` installable with

```bash
pip install ml4gw-buoy
```

It is recommended that you install `buoy` in a virtual environment such as `conda`.

# Usage

The function of this library is to run trained [Aframe](https://github.com/ML4GW/aframe) and [AMPLFI](https://github.com/ML4GW/amplfi) models over a gravitaional wave event reported by the LIGO-Virgo-KAGRA collaboration during their third observing run, O3.

Note: the trained models will be downloaded from [HuggingFace](https://huggingface.co/ML4GW) and require about 320 MB of space. 

To product model outputs, first identify an event of interest (e.g., from [GWTC-3](https://arxiv.org/pdf/2111.03606)), and then run

```bash
buoy --events <EVENT_NAME> --outdir <OUTPUT_DIRECTORY>
```

The output directory is structured as follows will contain a directory matching the name of the event.
Inside, there will be a `data` directory containing data created during the analysis, and a `plots`
directory containing Aframe's response to the event as well as a skymap and corner plot from AMPLFI.

Multiple events can be specified at once, e.g.:

```bash
buoy --events '["GW190828_063405", "GW190521"]' --outdir <OUTPUT_DIRECTORY>
```

About 10 MB of space is required for each event.
