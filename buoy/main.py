import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Union
import warnings

import numpy as np
import torch

from buoy.models.aframe import Aframe
from buoy.models.amplfi import Amplfi
from buoy.utils.data import get_data
from buoy.utils.plotting import plot_aframe_response, plot_amplfi_result

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from amplfi.train.data.utils.utils import ParameterSampler


def main(
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
    aframe_weights: Optional[Path] = None,
    amplfi_hl_weights: Optional[Path] = None,
    amplfi_hlv_weights: Optional[Path] = None,
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

    logging.info("Settinp up models")

    aframe = Aframe(device=device)

    amplfi_hl = Amplfi(
        model_weights="amplfi-hl.ckpt",
        config="amplfi-hl-config.yaml",
        device=device,
    )

    amplfi_hlv = Amplfi(
        model_weights="amplfi-hlv.ckpt",
        config="amplfi-hlv-config.yaml",
        device=device,
    )

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
        psd = amplfi_hl.spectral_density(data[..., :idx])
        whitened = amplfi_hl.whitener(data[..., idx:], psd).cpu().numpy()
        whitened = np.squeeze(whitened)
        whitened_start = t0 + psd_length + amplfi_fduration / 2
        whitened_end = t0 + data.shape[-1] / sample_rate - amplfi_fduration / 2
        whitened_times = np.arange(
            whitened_start, whitened_end, 1 / sample_rate
        )
        whitened_data = np.concatenate([whitened_times[None], whitened])
        np.save(datadir / "whitened_data.npy", whitened_data)

        logging.info("Running Aframe")

        times, ys, integrated = aframe(data[:, :2], t0)
        tc = times[np.argmax(integrated)] + aframe.get_time_offset()

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

        logging.info("Running AMPLFI model")
        amplfi = amplfi_hl if len(data) == 2 else amplfi_hlv
        result = amplfi(
            data=data,
            t0=t0,
            tc=tc,
            samples_per_event=samples_per_event,
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
