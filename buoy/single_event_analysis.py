import logging
from pathlib import Path
import sys
from typing import List, Optional
import warnings

import gwosc
import h5py
import jsonargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import io
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from ligo.skymap.tool.ligo_skymap_plot import main as ligo_skymap_plot
from ml4gw.transforms import SpectralDensity, Whiten

from amplfi.train.architectures.flows import FlowArchitecture
from amplfi.train.data.utils.utils import ParameterSampler
from amplfi.utils.result import AmplfiResult
from architectures import Architecture
from online.main import get_time_offset, load_amplfi
from online.utils.pe import postprocess_samples, run_amplfi
from utils.preprocessing import BackgroundSnapshotter, BatchWhitener

plt.rcParams.update(
    {
        "font.size": 16,
        "figure.dpi": 250,
    }
)

"""
TODO:
- Allow specifying event string or GPS time
- Allow for list of events/times
- Better handle times when an event is not present,
or Aframe does not react strongly to an event
- Add options to save out more of the data
- Remove hardcoded data times
"""


def get_data(
    event: str,
    sample_rate: float,
    datadir: Path,
):
    event_time = gwosc.datasets.event_gps(event)
    offset = event_time % 1
    start = event_time - 96 - offset
    end = event_time + 32 - offset
    ifos = sorted(gwosc.datasets.event_detectors(event))

    if ifos not in [["H1", "L1"], ["H1", "L1", "V1"]]:
        raise ValueError(
            f"Event {event} does not have the required detectors. "
            f"Expected ['H1', 'L1'] or ['H1', 'L1', 'V1'], got {ifos}"
        )

    datafile = datadir / f"{event}.hdf5"
    if not datafile.exists():
        logging.info(
            "Fetching open data from GWOSC between GPS times "
            f"{start} and {end} for {ifos}"
        )

        ts_dict = TimeSeriesDict()
        for ifo in ifos:
            ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end)
        ts_dict = ts_dict.resample(sample_rate)

        logging.info(f"Saving data to file {datafile}")

        with h5py.File(datafile, "w") as f:
            f.attrs["tc"] = event_time
            f.attrs["t0"] = start
            for ifo in ifos:
                f.create_dataset(ifo, data=ts_dict[ifo].value)

        t0 = start
        data = np.stack([ts_dict[ifo].value for ifo in ifos])[None]

    else:
        logging.info(f"Loading {ifos} data from file for event {event}")
        with h5py.File(datafile, "r") as f:
            data = np.stack([f[ifo][:] for ifo in ifos])[None]
            event_time = f.attrs["tc"]
            t0 = f.attrs["t0"]

    return torch.Tensor(data).double(), ifos, t0, event_time


def run_aframe(
    data: torch.Tensor,
    t0: float,
    aframe: Architecture,
    whitener: BatchWhitener,
    snapshotter: BackgroundSnapshotter,
    inference_sampling_rate: float,
    integration_window_length: float,
    batch_size: int,
    device: str = "cpu",
):
    """
    Run the aframe model over the data
    """
    step_size = int(batch_size * whitener.stride_size)

    # Iterate through the data, making predictions
    ys, batches = [], []
    start = 0
    state = torch.zeros((1, 2, snapshotter.state_size)).to(device)
    while start < (data.shape[-1] - step_size):
        stop = start + step_size
        x = data[:, :, start:stop]
        with torch.no_grad():
            x, state = snapshotter(x, state)
            batch = whitener(x)
            y_hat = aframe(batch)[:, 0].cpu().numpy()

        batches.append(batch.cpu().numpy())
        ys.append(y_hat)
        start += step_size
    batches = np.concatenate(batches)
    ys = np.concatenate(ys)

    times = np.arange(
        t0, t0 + len(ys) / inference_sampling_rate, 1 / inference_sampling_rate
    )
    window_size = int(integration_window_length * inference_sampling_rate) + 1
    window = np.ones((window_size,)) / window_size
    integrated = np.convolve(ys, window, mode="full")
    integrated = integrated[: -window_size + 1]

    return times, ys, integrated


def get_amplfi_data(
    data: torch.Tensor,
    sample_rate: float,
    t0: float,
    tc: float,
    amplfi_kernel_length: float,
    event_position: float,
    amplfi_psd_length: float,
    amplfi_fduration: float,
):
    """
    Slice the data to get the PSD window and kernel for amplfi
    """
    window_start = tc - t0 - event_position - amplfi_fduration / 2
    window_start = int(sample_rate * window_start)
    window_length = int((amplfi_kernel_length + amplfi_fduration) * sample_rate)
    window_end = window_start + window_length

    psd_start = window_start - int(amplfi_psd_length * sample_rate)

    psd_data = data[0, :, psd_start:window_start]
    window = data[0, :, window_start:window_end]

    return psd_data, window


def plot_aframe_response(
    times: np.ndarray,
    ys: np.ndarray,
    integrated: np.ndarray,
    whitened: np.ndarray,
    whitened_times: np.ndarray,
    t0: float,
    tc: float,
    event_time: float,
    plotdir: Path,
):
    """
    Plot raw and integrated output alongside the whitened strain
    """

    # Shift the times to be relative to the event time
    times -= event_time
    whitened_times -= event_time
    t0 -= event_time
    tc -= event_time

    plt.figure(figsize=(12, 8))
    plt.plot(whitened_times, whitened[0, 0], label="H1", alpha=0.3)
    plt.plot(whitened_times, whitened[0, 1], label="L1", alpha=0.3)
    plt.xlabel("Time from event (s)")
    plt.axvline(tc, color="tab:red", linestyle="--", label="Predicted time")
    plt.axvline(0, color="k", linestyle="--", label="Event time")
    plt.ylabel("Whitened strain")
    plt.legend(loc="upper left")
    plt.grid()
    plt.twinx()

    plt.plot(times, ys, color="tab:gray", label="Raw", lw=2)
    plt.plot(times, integrated, color="k", label="Integrated", lw=2)
    plt.ylabel("Detection statistic")
    plt.legend(loc="upper right")
    plt.xlim(t0 + 94, t0 + 102)
    plt.grid()
    plt.savefig(plotdir / "aframe_response.png", bbox_inches="tight")


def plot_amplfi_result(
    result: AmplfiResult,
    nside: int,
    ifos: List[str],
    datadir: Path,
    plotdir: Path,
):
    """
    Plot the skymap and corner plot from amplfi
    """

    suffix = "".join([ifo[0] for ifo in ifos])

    skymap = result.to_skymap(nside, use_distance=False)
    fits_skymap = io.fits.table_to_hdu(skymap)
    fits_fname = datadir / f"amplfi_{suffix}.fits"
    fits_skymap.writeto(fits_fname, overwrite=True)
    plot_fname = plotdir / f"skymap_{suffix}.png"

    ligo_skymap_plot(
        [
            str(fits_fname),
            "--annotate",
            "--contour",
            "50",
            "90",
            "-o",
            str(plot_fname),
        ]
    )
    plt.close()

    corner_fname = plotdir / f"corner_plot_{suffix}.png"
    result.plot_corner(
        parameters=["chirp_mass", "mass_ratio", "distance"],
        filename=corner_fname,
    )
    plt.close()


def main(
    model_dir: Path,
    amplfi_hl_architecture: FlowArchitecture,
    amplfi_hlv_architecture: FlowArchitecture,
    amplfi_parameter_sampler: ParameterSampler,
    event: str,
    outdir: Path,
    inference_params: List[str],
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    amplfi_psd_length: float,
    aframe_right_pad: float,
    amplfi_kernel_length: float,
    event_position: float,
    fduration: float,
    amplfi_fduration: float,
    integration_window_length: float,
    batch_size: int,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    amplfi_highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    samples_per_event: int = 20000,
    nside: int = 32,
    device: str = "cpu",
):
    if device == "cpu":
        warnings.warn(
            "Device is set to 'cpu'. This will take about "
            "15 minutes to run. If a GPU is available, it "
            "is recommended to set device to 'cuda'.",
            stacklevel=2,
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Device is set to 'cuda', but no GPU is available. "
            "Please set device to 'cpu' or move to a node with "
            "a GPU."
        )

    outdir = outdir / event
    datadir = outdir / "data"
    plotdir = outdir / "plots"
    datadir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    aframe_weights = model_dir / "aframe.pt"
    amplfi_hl_weights = model_dir / "amplfi-hl.ckpt"
    amplfi_hlv_weights = model_dir / "amplfi-hlv.ckpt"

    logging.info("Setting up preprocessing modules")
    # Create objects for whitening and
    # for generating snapshots of data
    whitener = BatchWhitener(
        kernel_length,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        fduration,
        fftlength,
        highpass=highpass,
    ).to(device)
    snapshotter = BackgroundSnapshotter(
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ).to(device)

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    amplfi_whitener = Whiten(
        fduration=amplfi_fduration,
        sample_rate=sample_rate,
        highpass=amplfi_highpass,
        lowpass=lowpass,
    ).to(device)

    logging.info("Fetching or loading data")
    data, ifos, t0, event_time = get_data(
        event=event,
        sample_rate=sample_rate,
        datadir=datadir,
    )
    data = data.to(device)

    # Compute whitened data for plotting later
    # Use the first psd_length seconds of data
    # to calculate the PSD and whiten the rest
    idx = int(sample_rate * psd_length)
    psd = spectral_density(data[..., :idx])
    whitened = amplfi_whitener(data[..., idx:], psd).cpu().numpy()
    whitened_start = t0 + psd_length + amplfi_fduration / 2
    whitened_end = t0 + data.shape[-1] / sample_rate - amplfi_fduration / 2
    whitened_times = np.arange(whitened_start, whitened_end, 1 / sample_rate)

    logging.info("Loading Aframe")

    # Load the trained models
    aframe = torch.jit.load(aframe_weights)
    aframe = aframe.to(device)

    if ifos == ["H1", "L1"]:
        logging.info("Loading AMPLFI HL model")
        amplfi, scaler = load_amplfi(
            amplfi_hl_architecture, amplfi_hl_weights, len(inference_params)
        )
        amplfi = amplfi.to(device)
        scaler = scaler.to(device)
    else:
        logging.info("Loading AMPLFI HLV model")
        amplfi, scaler = load_amplfi(
            amplfi_hlv_architecture, amplfi_hlv_weights, len(inference_params)
        )
        amplfi = amplfi.to(device)
        scaler = scaler.to(device)

    # Calculate offset between integration peak and
    # the time of the event
    time_offset = get_time_offset(
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        integration_window_length=integration_window_length,
        kernel_length=kernel_length,
        aframe_right_pad=aframe_right_pad,
    )

    logging.info("Running Aframe")

    times, ys, integrated = run_aframe(
        data=data[:, :2],
        t0=t0,
        aframe=aframe,
        whitener=whitener,
        snapshotter=snapshotter,
        inference_sampling_rate=inference_sampling_rate,
        integration_window_length=integration_window_length,
        batch_size=batch_size,
        device=device,
    )
    tc = times[np.argmax(integrated)] + time_offset

    logging.info("Plotting Aframe response")
    plot_aframe_response(
        times=times,
        ys=ys,
        integrated=integrated,
        whitened=whitened,
        whitened_times=whitened_times,
        t0=t0,
        tc=tc,
        event_time=event_time,
        plotdir=plotdir,
    )

    amplfi_psd_data, amplfi_window = get_amplfi_data(
        data=data,
        sample_rate=sample_rate,
        t0=t0,
        tc=tc,
        amplfi_kernel_length=amplfi_kernel_length,
        event_position=event_position,
        amplfi_psd_length=amplfi_psd_length,
        amplfi_fduration=amplfi_fduration,
    )

    logging.info("Running AMPLFI model")
    samples, _, _, _ = run_amplfi(
        amplfi_window[: len(ifos)],
        amplfi_psd_data[: len(ifos)],
        samples_per_event,
        spectral_density,
        amplfi_whitener,
        amplfi,
        scaler,
        device=device,
    )
    result = postprocess_samples(
        samples.cpu(),
        tc,
        inference_params,
        amplfi_parameter_sampler,
    )
    logging.info("Plotting AMPLFI result")
    plot_amplfi_result(
        result=result,
        nside=nside,
        ifos=ifos,
        datadir=datadir,
        plotdir=plotdir,
    )


def cli():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        stream=sys.stdout,
    )

    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    parser.add_argument("--config", action="config")

    parser.link_arguments(
        "inference_params",
        "amplfi_hl_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )

    parser.link_arguments(
        "inference_params",
        "amplfi_hlv_architecture.init_args.num_params",
        compute_fn=lambda x: len(x),
        apply_on="parse",
    )
    args = parser.parse_args()
    args.pop("config")
    args = parser.instantiate_classes(args)

    main(**vars(args))


if __name__ == "__main__":
    cli()
