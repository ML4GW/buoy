import logging
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

from .models import Aframe, Amplfi
from .utils.data import get_data, get_events_for_runs
from .utils.html import generate_html
from .utils.plotting import (
    plot_aframe_response,
    plot_amplfi_result,
    q_plots,
)


def _resolve_device(device: str | None) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "Device is set to 'cpu'. This will take about "
            "15 minutes to run with default settings. "
            "If a GPU is available, set '--device cuda'. ",
            stacklevel=3,
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            f"Device is set to {device}, but no GPU is available. "
            "Please set device to 'cpu' or move to a node with "
            "a GPU."
        )
    return device


def _resolve_events(
    events: str | list[str] | None,
    observing_runs: list[str] | None,
    gps_start: float | None,
    gps_end: float | None,
) -> list[str]:
    if (gps_start is None) != (gps_end is None):
        raise ValueError("gps_start and gps_end must be provided together.")

    gps_segment = (gps_start, gps_end) if gps_start is not None else None

    if events is not None and (
        observing_runs is not None or gps_segment is not None
    ):
        logging.warning(
            "Both 'events' and run/GPS times were provided. "
            "Only 'events' will be used."
        )

    if events is not None:
        return [events] if not isinstance(events, list) else events

    if observing_runs is None and gps_segment is None:
        raise ValueError(
            "One of 'events', 'observing_runs', or "
            "'gps_start'/'gps_end' must be provided."
        )
    if observing_runs is not None and gps_segment is not None:
        raise ValueError(
            "observing_runs and gps_start/gps_end are mutually exclusive."
        )
    logging.info(
        "Fetching public GW events from GWOSC"
        + (f" for runs: {observing_runs}" if observing_runs else "")
        + (f" between GPS {gps_start} and {gps_end}" if gps_segment else "")
    )
    resolved = get_events_for_runs(observing_runs, gps_segment)
    logging.info(f"Found {len(resolved)} public GW event(s) to analyze")
    return resolved


def main(
    outdir: Path,
    events: str | list[str] | None = None,
    observing_runs: list[str] | None = None,
    gps_start: float | None = None,
    gps_end: float | None = None,
    samples_per_event: int = 20000,
    nside: int = 64,
    min_samples_per_pix: int = 5,
    use_distance: bool = True,
    aframe_weights: Path | None = None,
    amplfi_hl_weights: Path | None = None,
    amplfi_hlv_weights: Path | None = None,
    aframe_config: Path | None = None,
    amplfi_hl_config: Path | None = None,
    amplfi_hlv_config: Path | None = None,
    aframe_revision: str | None = None,
    amplfi_revision: str | None = None,
    model_cache_dir: Path | None = None,
    use_true_tc_for_amplfi: bool = False,
    ifos: list[str] | None = None,
    device: str | None = None,
    to_html: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    run_aframe: bool = True,
    run_amplfi: bool = True,
    generate_plots: bool = True,
    force: bool = False,
    max_workers: int = 1,
    corner_parameters: list[str] | None = None,
):
    """
    Main function to run Aframe and AMPLFI on the given events
    and produce output plots.

    Exactly one event source must be supplied: either explicit event
    names via ``events``, a set of observing runs via ``observing_runs``,
    or a GPS time window via ``gps_start`` and ``gps_end``.

    Args:
        outdir:
            Output directory to save results.
        events:
            Gravitational wave event name(s) to process. Accepts known
            event names (e.g. GW150914), GraceDB events (e.g. G363842),
            GraceDB superevents (e.g. S200213t), or GPS times
            (e.g. 1187008882.4). Mutually exclusive with
            ``observing_runs`` and ``gps_start``/``gps_end``.
        observing_runs:
            GWOSC observing run label(s) (e.g. ["O3a", "O3b"]). All
            public GW events from those runs are fetched automatically.
            Valid run names: O1, O2, O3a, O3b, O4a. Mutually exclusive
            with ``events`` and ``gps_start``/``gps_end``.
        gps_start:
            Start of a GPS time window. All public GW events with
            GPS times in [gps_start, gps_end] will be analyzed. Must be
            provided together with ``gps_end``. Mutually exclusive with
            ``events`` and ``observing_runs``.
        gps_end:
            End of a GPS time window. Must be provided together with
            ``gps_start``.
        samples_per_event:
            Number of samples for AMPLFI to generate for each event.
        nside:
            Healpix resolution for AMPLFI skymap
        min_samples_per_pix:
            Minimum number of samples per healpix pixel
            required to estimate parameters of the distance
            ansatz
        use_distance:
            If true, use distance samples to create a 3D skymap
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
        aframe_revision:
            HuggingFace repository revision (branch, tag, or commit
            hash) for Aframe model weights and config. If None, uses
            the default branch.
        amplfi_revision:
            HuggingFace repository revision (branch, tag, or commit
            hash) for AMPLFI model weights and config. If None, uses
            the default branch.
        model_cache_dir:
            Local directory to use as the HuggingFace download cache
            for model weights and configs. If None, uses the default
            HuggingFace cache location (~/.cache/huggingface).
        use_true_tc_for_amplfi:
            If True, use the true time of coalescence for AMPLFI.
            Else, use the merger time inferred from Aframe.
        ifos:
            List of detectors to use when fetching data for a GPS time
            event. Ignored for named events (GW/G/S), which look up
            detectors automatically. Defaults to ["H1", "L1", "V1"].
        device:
            Device to run the models on ("cpu" or "cuda").
        to_html:
            If True, generate an HTML summary page.
        seed:
            Random seed for reproducibility of AMPLFI results.
        verbose:
            If True, log at the DEBUG level. Else, log at INFO level.
        run_aframe:
            If True, run Aframe inference and save outputs. If False,
            load previously saved Aframe outputs if available.
        run_amplfi:
            If True, run AMPLFI inference and save posterior samples.
            If False, skip parameter estimation and PE plots.
        generate_plots:
            If True, generate all output plots. If False, skip plotting.
        force:
            If True, reprocess events even if output files already exist.
            If False, skip events whose inference outputs are all present.
        max_workers:
            Number of events to process concurrently. Default 1
            (sequential). Values > 1 use a thread pool so that data
            fetching for the next event overlaps with model inference
            for the current one. On a single GPU, values above the
            number of GPUs will not further improve GPU utilisation
            but may increase memory pressure.
        corner_parameters:
            List of parameter names to include in the AMPLFI corner plot.
            Defaults to ["chirp_mass", "mass_ratio", "distance",
            "mass_1", "mass_2"].
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )
    logging.getLogger("bilby").setLevel(logging.WARNING)
    logging.getLogger("gwdatafind").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    events = _resolve_events(events, observing_runs, gps_start, gps_end)

    if seed is not None:
        torch.manual_seed(seed)

    device = _resolve_device(device)

    logging.info("Setting up models")

    aframe = Aframe(
        model_weights=aframe_weights or "aframe.pt",
        config=aframe_config or "aframe_config.yaml",
        device=device,
        revision=aframe_revision,
        load_weights=run_aframe,
        cache_dir=model_cache_dir,
    )

    amplfi_hl = Amplfi(
        model_weights=amplfi_hl_weights or "amplfi-hl.ckpt",
        config=amplfi_hl_config or "amplfi-hl-config.yaml",
        device=device,
        revision=amplfi_revision,
        load_weights=run_amplfi,
        cache_dir=model_cache_dir,
    )

    amplfi_hlv = Amplfi(
        model_weights=amplfi_hlv_weights or "amplfi-hlv.ckpt",
        config=amplfi_hlv_config or "amplfi-hlv-config.yaml",
        device=device,
        revision=amplfi_revision,
        load_weights=run_amplfi,
        cache_dir=model_cache_dir,
    )

    if not (
        aframe.sample_rate == amplfi_hl.sample_rate == amplfi_hlv.sample_rate
    ):
        raise ValueError(
            f"Sample rate mismatch: Aframe={aframe.sample_rate}, "
            f"AMPLFI-HL={amplfi_hl.sample_rate}, "
            f"AMPLFI-HLV={amplfi_hlv.sample_rate}. All models must match."
        )

    default_ifos = ifos

    def _process_event(event: str) -> None:
        log = logging.getLogger(event)
        datadir = outdir / str(event) / "data"
        plotdir = outdir / str(event) / "plots"
        datadir.mkdir(parents=True, exist_ok=True)
        plotdir.mkdir(parents=True, exist_ok=True)

        aframe_output_file = datadir / "aframe_outputs.hdf5"
        amplfi_output_file = datadir / "posterior_samples.dat"

        if not force:
            aframe_done = aframe_output_file.exists()
            amplfi_done = not run_amplfi or amplfi_output_file.exists()
            if aframe_done and amplfi_done:
                log.info(
                    f"Skipping {event}: outputs already present. "
                    "Use --force to reprocess."
                )
                return

        log.info("Fetching or loading data")
        data, ifos, t0, event_time = get_data(
            event=event,
            sample_rate=aframe.sample_rate,
            psd_length=aframe.psd_length,
            datadir=datadir,
            ifos=default_ifos,
        )
        data = torch.Tensor(data).double()
        data = data.to(device)

        amplfi_model = amplfi_hl if data.shape[1] == 2 else amplfi_hlv

        if run_aframe:
            log.info("Running Aframe")
            times, ys, timing_integrated, signif_integrated = aframe(
                data[:, :2], t0
            )
            predicted_tc = (
                times[np.argmax(timing_integrated)] + aframe.time_offset
            )
            log.info("Saving Aframe outputs")
            with h5py.File(aframe_output_file, "w") as f:
                f.create_dataset("times", data=times)
                f.create_dataset("ys", data=ys)
                f.create_dataset("timing_integrated", data=timing_integrated)
                f.create_dataset("signif_integrated", data=signif_integrated)
                f.attrs["predicted_tc"] = predicted_tc
        else:
            log.info("Loading saved Aframe outputs")
            if not aframe_output_file.exists():
                raise FileNotFoundError(
                    f"Aframe output file {aframe_output_file} not found. "
                    "Run with run_aframe=True first."
                )
            with h5py.File(aframe_output_file, "r") as f:
                times = f["times"][:]
                ys = f["ys"][:]
                timing_integrated = f["timing_integrated"][:]
                signif_integrated = f["signif_integrated"][:]
                predicted_tc = f.attrs["predicted_tc"]

        tc = event_time if use_true_tc_for_amplfi else predicted_tc

        if run_amplfi:
            log.info("Running AMPLFI model")
            result = amplfi_model(
                data=data,
                t0=t0,
                tc=tc,
                samples_per_event=samples_per_event,
            )

            # Compute whitened data for plotting later
            # Use the first psd_length seconds of data
            # to calculate the PSD and whiten the rest
            idx = int(amplfi_model.sample_rate * amplfi_model.psd_length)
            psd = amplfi_model.spectral_density(data[..., :idx])
            whitened = (
                amplfi_model.whitener(data[..., idx:], psd).cpu().numpy()
            )
            whitened = np.squeeze(whitened)
            whitened_start = (
                t0 + amplfi_model.psd_length + amplfi_model.fduration / 2
            )
            whitened_end = (
                t0
                + data.shape[-1] / amplfi_model.sample_rate
                - amplfi_model.fduration / 2
            )
            whitened_times = np.arange(
                whitened_start, whitened_end, 1 / amplfi_model.sample_rate
            )
            whitened_data = np.concatenate([whitened_times[None], whitened])
            np.save(datadir / "whitened_data.npy", whitened_data)

            result.save_posterior_samples(filename=amplfi_output_file)

        if generate_plots:
            log.info("Creating Q-plots")
            q_plots(
                data=data.squeeze().cpu().numpy(),
                t0=t0,
                plotdir=plotdir,
                gpstime=event_time,
                sample_rate=amplfi_model.sample_rate,
                amplfi_highpass=amplfi_model.highpass,
            )

            if run_aframe:
                whitened_data_file = datadir / "whitened_data.npy"
                if whitened_data_file.exists():
                    whitened_arr = np.load(whitened_data_file)
                    whitened_times = whitened_arr[0]
                    whitened = whitened_arr[1:]
                else:
                    log.warning(
                        "Whitened data not found, plotting only "
                        "network output. Run with run_amplfi=True "
                        "to generate whitened data."
                    )
                    whitened = None
                    whitened_times = None

                log.info("Plotting Aframe response")
                plot_aframe_response(
                    times=times,
                    ys=ys,
                    timing_integrated=timing_integrated,
                    signif_integrated=signif_integrated,
                    whitened=whitened,
                    whitened_times=whitened_times,
                    t0=t0,
                    tc=tc,
                    event_time=event_time,
                    plotdir=plotdir,
                )

            if run_amplfi:
                log.info("Plotting AMPLFI result")
                plot_amplfi_result(
                    result=result,
                    nside=nside,
                    min_samples_per_pix=min_samples_per_pix,
                    use_distance=use_distance,
                    ifos=ifos,
                    datadir=datadir,
                    plotdir=plotdir,
                    corner_parameters=corner_parameters,
                )

        if to_html:
            log.info("Generating HTML page")
            generate_html(
                plotdir=plotdir,
                output_file=outdir / str(event) / "summary.html",
                label=str(event) + " Event Summary",
            )

    if max_workers == 1:
        for event in events:
            _process_event(event)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_event, event): event for event in events
            }
            for future in as_completed(futures):
                event = futures[future]
                if exc := future.exception():
                    logging.error(f"Event {event} failed: {exc}", exc_info=exc)
