import logging
from pathlib import Path
from typing import List, Optional, Union
import warnings

import numpy as np
import torch

from .models import Aframe, Amplfi
from .utils.data import get_data
from .utils.html import generate_html
from .utils.plotting import plot_aframe_response, plot_amplfi_result


def main(
    events: Union[str, List[str]],
    outdir: Path,
    samples_per_event: int = 20000,
    nside: int = 32,
    aframe_weights: Optional[Path] = None,
    amplfi_hl_weights: Optional[Path] = None,
    amplfi_hlv_weights: Optional[Path] = None,
    aframe_config: Optional[Path] = None,
    amplfi_hl_config: Optional[Path] = None,
    amplfi_hlv_config: Optional[Path] = None,
    use_true_tc_for_amplfi: bool = False,
    device: Optional[str] = None,
    to_html: bool = False,
):
    """
    Main function to run Aframe and AMPLFI on the given events
    and produce output plots.

    Args:
        events:
            Gravitational wave event name(s) to process.
        outdir:
            Output directory to save results.
        samples_per_event:
            Number of samples for AMPLFI to generate for each event.
        nside:
            Healpix resolution for AMPLFI skymap
        aframe_weights:
            Path to Aframe model weights. Can be a local path
            or in the ML4GW/aframe Hugging Face repository.
        amplfi_hl_weights:
            Path to AMPLFI HL model weights. Can be a local path
            or in the ML4GW/amplfi Hugging Face repository.
        amplfi_hlv_weights:
            Path to AMPLFI HLV model weights. Can be a local path
            or in the ML4GW/amplfi Hugging Face repository.
        aframe_config:
            Path to Aframe config file. Can be a local path
            or in the ML4GW/aframe Hugging Face repository.
        amplfi_hl_config:
            Path to AMPLFI HL config file. Can be a local path
            or in the ML4GW/amplfi Hugging Face repository.
        amplfi_hlv_config:
            Path to AMPLFI HLV config file. Can be a local path
            or in the ML4GW/amplfi Hugging Face repository.
        use_true_tc_for_amplfi:
            If True, use the true time of coalescence for AMPLFI.
            Else, use the merger time inferred from Aframe.
        device:
            Device to run the models on ("cpu" or "cuda").
        to_html:
            If True, generate an HTML summary page.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        warnings.warn(
            "Device is set to 'cpu'. This will take about "
            "15 minutes to run with default settings. "
            "If a GPU is available, set '--device cuda'. ",
            stacklevel=2,
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"Device is set to {device}, but no GPU is available. "
            "Please set device to 'cpu' or move to a node with "
            "a GPU."
        )

    logging.info("Setting up models")

    aframe = Aframe(
        model_weights=aframe_weights or "aframe.pt",
        config=aframe_config or "aframe_config.yaml",
        device=device,
    )

    amplfi_hl = Amplfi(
        model_weights=amplfi_hl_weights or "amplfi-hl.ckpt",
        config=amplfi_hl_config or "amplfi-hl-config.yaml",
        device=device,
    )

    amplfi_hlv = Amplfi(
        model_weights=amplfi_hlv_weights or "amplfi-hlv.ckpt",
        config=amplfi_hlv_config or "amplfi-hlv-config.yaml",
        device=device,
    )

    # TODO: should we check that the sample rate for each model is the same?

    if not isinstance(events, list):
        events = [events]
    for event in events:
        datadir = outdir / event / "data"
        plotdir = outdir / event / "plots"
        datadir.mkdir(parents=True, exist_ok=True)
        plotdir.mkdir(parents=True, exist_ok=True)

        logging.info("Fetching or loading data")
        data, ifos, t0, event_time = get_data(
            event=event,
            sample_rate=aframe.sample_rate,
            datadir=datadir,
        )
        data = torch.Tensor(data).double()
        data = data.to(device)

        logging.info("Running Aframe")

        times, ys, integrated = aframe(data[:, :2], t0)
        if use_true_tc_for_amplfi:
            tc = event_time
        else:
            tc = times[np.argmax(integrated)] + aframe.get_time_offset()

        logging.info("Running AMPLFI model")
        amplfi = amplfi_hl if data.shape[1] == 2 else amplfi_hlv
        result = amplfi(
            data=data,
            t0=t0,
            tc=tc,
            samples_per_event=samples_per_event,
        )

        # Compute whitened data for plotting later
        # Use the first psd_length seconds of data
        # to calculate the PSD and whiten the rest
        idx = int(amplfi.sample_rate * amplfi.psd_length)
        psd = amplfi.spectral_density(data[..., :idx])
        whitened = amplfi.whitener(data[..., idx:], psd).cpu().numpy()
        whitened = np.squeeze(whitened)
        whitened_start = t0 + amplfi.psd_length + amplfi.fduration / 2
        whitened_end = (
            t0 + data.shape[-1] / amplfi.sample_rate - amplfi.fduration / 2
        )
        whitened_times = np.arange(
            whitened_start, whitened_end, 1 / amplfi.sample_rate
        )
        whitened_data = np.concatenate([whitened_times[None], whitened])
        np.save(datadir / "whitened_data.npy", whitened_data)

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

        if to_html:
            logging.info("Generating HTML page")
            generate_html(
                plotdir=plotdir,
                output_file=outdir / event / "summary.html",
                label=event + " Event Summary",
            )
