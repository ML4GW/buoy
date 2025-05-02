import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Union
import warnings

import numpy as np
import torch
from ml4gw.transforms import SpectralDensity, Whiten

from buoy.utils.data import get_data, slice_amplfi_data
from buoy.utils.detection import get_time_offset, run_aframe
from buoy.utils.pe import load_amplfi, postprocess_samples, run_amplfi
from buoy.utils.plotting import plot_aframe_response, plot_amplfi_result
from buoy.utils.preprocessing import BackgroundSnapshotter, BatchWhitener

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from amplfi.train.data.utils.utils import ParameterSampler


def main(
    model_dir: Path,
    amplfi_hl_architecture: "FlowArchitecture",
    amplfi_hlv_architecture: "FlowArchitecture",
    amplfi_parameter_sampler: "ParameterSampler",
    events: Union[str, List[str]],
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
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        warnings.warn(
            "Device is set to 'cpu'. This will take about "
            "15 minutes to run with default settings. "
            "If a GPU is available, set '--device cuda'. ",
            stacklevel=2,
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Device is set to 'cuda', but no GPU is available. "
            "Please set device to 'cpu' or move to a node with "
            "a GPU."
        )

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

    logging.info("Loading Aframe")

    # Load the trained models
    aframe = torch.jit.load(aframe_weights)
    aframe = aframe.to(device)

    logging.info("Loading AMPLFI HL model")
    amplfi_hl, scaler_hl = load_amplfi(
        amplfi_hl_architecture, amplfi_hl_weights, len(inference_params)
    )
    amplfi_hl = amplfi_hl.to(device)
    scaler_hl = scaler_hl.to(device)

    logging.info("Loading AMPLFI HLV model")
    amplfi_hlv, scaler_hlv = load_amplfi(
        amplfi_hlv_architecture, amplfi_hlv_weights, len(inference_params)
    )
    amplfi_hlv = amplfi_hlv.to(device)
    scaler_hlv = scaler_hlv.to(device)

    if isinstance(events, str):
        events = [events]
    for event in events:
        eventdir = outdir / event
        datadir = eventdir / "data"
        plotdir = eventdir / "plots"
        datadir.mkdir(parents=True, exist_ok=True)
        plotdir.mkdir(parents=True, exist_ok=True)

        logging.info("Fetching or loading data")
        data, ifos, t0, event_time = get_data(
            event=event,
            sample_rate=sample_rate,
            datadir=datadir,
        )
        data = torch.Tensor(data).double()
        data = data.to(device)

        # Compute whitened data for plotting later
        # Use the first psd_length seconds of data
        # to calculate the PSD and whiten the rest
        idx = int(sample_rate * psd_length)
        psd = spectral_density(data[..., :idx])
        whitened = amplfi_whitener(data[..., idx:], psd).cpu().numpy()
        whitened = np.squeeze(whitened)
        whitened_start = t0 + psd_length + amplfi_fduration / 2
        whitened_end = t0 + data.shape[-1] / sample_rate - amplfi_fduration / 2
        whitened_times = np.arange(
            whitened_start, whitened_end, 1 / sample_rate
        )
        whitened_data = np.concatenate([whitened_times[None], whitened])
        np.save(datadir / "whitened_data.npy", whitened_data)

        # Calculate offset between integration peak and
        # the time of the event
        time_offset = get_time_offset(
            inference_sampling_rate=inference_sampling_rate,
            fduration=fduration,
            integration_window_length=integration_window_length,
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

        amplfi_psd_data, amplfi_window = slice_amplfi_data(
            data=data,
            sample_rate=sample_rate,
            t0=t0,
            tc=tc,
            amplfi_kernel_length=amplfi_kernel_length,
            event_position=event_position,
            amplfi_psd_length=amplfi_psd_length,
            amplfi_fduration=amplfi_fduration,
        )

        if len(ifos) == 2:
            amplfi = amplfi_hl
            scaler = scaler_hl
        else:
            amplfi = amplfi_hlv
            scaler = scaler_hlv
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
        result.save_posterior_samples(
            filename=datadir / "posterior_samples.dat"
        )
        logging.info("Plotting AMPLFI result")
        plot_amplfi_result(
            result=result,
            nside=nside,
            ifos=ifos,
            datadir=datadir,
            plotdir=plotdir,
        )
