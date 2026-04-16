import logging
import warnings
from pathlib import Path

import gwosc
import gwosc.datasets
import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from ligo.gracedb.rest import GraceDb

STRAIN_CHANNELS = {
    "H1": "H1:GDS-CALIB_STRAIN_CLEAN",
    "L1": "L1:GDS-CALIB_STRAIN_CLEAN",
    "V1": "V1:Hrec_hoft_16384Hz",
}


def get_local_or_hf(
    filename: str | Path,
    repo_id: str,
    descriptor: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """
    Determine whether a given file exists locally or in a HuggingFace
    repository. If the file exists locally, return the filename.
    If it does not exist locally, attempt to download it from the
    HuggingFace repository. If the file is not found in either
    location, raise a ValueError.

    Args:
        filename: The name of the file to load.
        repo_id: The HuggingFace repository ID.
        descriptor: A description of the file for logging.
        revision:
            The HuggingFace repository revision (branch, tag, or commit
            hash) to download from. If None, uses the default branch.
        cache_dir:
            Local directory to use as the HuggingFace download cache.
            If None, uses the default HuggingFace cache location.

    Returns:
        The path to the file.
    """
    if Path(filename).exists():
        logging.info(f"Loading {descriptor} from {filename}")
        return str(filename)
    else:
        try:
            logging.info(
                f"Downloading {descriptor} from HuggingFace "
                "or loading from cache"
            )
            return hf_hub_download(
                repo_id=repo_id,
                filename=str(filename),
                revision=revision,
                cache_dir=cache_dir,
            )
        except EntryNotFoundError as e:
            raise ValueError(
                f"{descriptor} {filename} not found locally or in "
                f"HuggingFace repository {repo_id}. Please check the name."
            ) from e


def slice_amplfi_data(
    data: torch.Tensor,
    sample_rate: float,
    t0: float,
    tc: float,
    kernel_length: float,
    event_position: float,
    psd_length: float,
    fduration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice a strain tensor into the PSD background and inference window
    required by AMPLFI.

    Args:
        data: Strain tensor of shape ``(1, num_channels, num_samples)``.
        sample_rate: Sampling rate in Hz.
        t0: GPS start time of ``data``.
        tc: Coalescence time (GPS).
        kernel_length: Length of the inference kernel in seconds.
        event_position:
            Position of the coalescence within the kernel in seconds
            (range ``[0, kernel_length]``).
        psd_length: Length of the PSD background segment in seconds.
        fduration: Whitening filter duration in seconds.

    Returns:
        A ``(psd_data, window)`` tuple where ``psd_data`` has shape
        ``(num_channels, psd_length * sample_rate)`` and ``window`` has
        shape ``(num_channels, (kernel_length + fduration) * sample_rate)``.

    Raises:
        ValueError: If the requested window extends outside the bounds of
            ``data``.
    """
    window_start = tc - t0 - event_position - fduration / 2
    window_start = int(sample_rate * window_start)
    window_length = int((kernel_length + fduration) * sample_rate)
    window_end = window_start + window_length

    if window_start < 0:
        raise ValueError(
            "The start of the AMPLFI window before the start of the data. "
            "This may be due to the event time being too close to "
            "the start of the data."
        )
    if window_end > data.shape[-1]:
        raise ValueError(
            "The end of the AMPLFI window is after the end of the data. "
            "This may be due to the event time being too close to "
            "the end of the data."
        )

    psd_start = window_start - int(psd_length * sample_rate)

    if psd_start < 0:
        raise ValueError(
            "The start of the PSD window before the start of the data. "
            "This may be due to the event time being too close to "
            "the start of the data."
        )

    psd_data = data[0, :, psd_start:window_start]
    window = data[0, :, window_start:window_end]

    return psd_data, window


def get_events_for_runs(
    observing_runs: list[str] | None = None,
    gps_segment: tuple[float, float] | None = None,
) -> list[str]:
    """Return public GW event base names filtered by run or GPS range.

    Exactly one of observing_runs or gps_segment must be provided.

    Args:
        observing_runs: GWOSC run labels (e.g. ["O3a", "O3b"]). Valid run
            names include O1, O2, O3a, O3b, O4a.
        gps_segment: (start, end) GPS time tuple to filter events by time
            window. Mutually exclusive with observing_runs.
    """
    if observing_runs is None and gps_segment is None:
        raise ValueError(
            "At least one of observing_runs or gps_segment must be provided. "
            "To fetch all public events use get_all_public_events()."
        )
    if observing_runs is not None and gps_segment is not None:
        raise ValueError(
            "observing_runs and gps_segment are mutually exclusive."
        )

    if gps_segment is not None:
        raw = gwosc.datasets.find_datasets(type="event", segment=gps_segment)
        context = f" in GPS range {gps_segment}"
    else:
        raw = []
        for run in observing_runs:
            seg = gwosc.datasets.run_segment(run)
            raw.extend(gwosc.datasets.find_datasets(type="event", segment=seg))
        context = f" for observing runs: {observing_runs}"

    bases = set()
    for name in raw:
        base = name.split("-v")[0] if "-v" in name else name
        if base.startswith("GW"):
            bases.add(base)
    if not bases:
        raise ValueError(f"No public GW events found{context}")
    return sorted(bases)


def get_data(
    event: str | float,
    sample_rate: float,
    psd_length: float,
    datadir: Path,
    ifos: list[str] | None = None,
) -> tuple[np.ndarray, list[str], float, float]:
    event = str(event)
    if event.startswith("GW"):
        event_time = gwosc.datasets.event_gps(event)
        ifos = sorted(gwosc.datasets.event_detectors(event))
    elif event.startswith("G"):
        client = GraceDb()
        response = client.event(event).json()
        event_time = response["gpstime"]
        ifos = response["instruments"].split(",")
    elif event.startswith("S"):
        client = GraceDb()
        response = client.superevent(event).json()
        event_time = response["preferred_event_data"]["gpstime"]
        ifos = response["preferred_event_data"]["instruments"].split(",")
    else:
        try:
            event_time = float(event)
        except ValueError as e:
            raise ValueError(
                f"Event {event} is not a valid event name. "
                "Should be a valid GPS time, a known gravitational wave "
                "event name (e.g. GW123456), or a GraceDB event or superevent "
                "(e.g. G123456 or S123456)."
            ) from e
        ifos = ifos or ["H1", "L1", "V1"]

    # Make sure things start at an integer time for consistency.
    # Take data from psd_length * (-1.5, 0.5) around the event
    # time to make sure there's enough for analysis. This isn't
    # totally robust, but should be good for most use cases.
    offset = event_time % 1
    start = event_time - 1.5 * psd_length - offset
    end = event_time + 0.5 * psd_length - offset

    datafile = datadir / f"{event}.hdf5"
    if not datafile.exists():
        logging.info(
            "Fetching data from between GPS times "
            f"{start} and {end} for {ifos}"
        )

        ts_dict = TimeSeriesDict()
        for ifo in ifos:
            if start < 1389456018:
                ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end)
            else:
                ts_dict[ifo] = TimeSeries.get(STRAIN_CHANNELS[ifo], start, end)

            span = ts_dict[ifo].span
            if span.end - span.start < 128:
                ts_dict.pop(ifo)
                warnings.warn(
                    f"Detector {ifo} did not have sufficient data surrounding "
                    "the event time, removing it from the dataset",
                    stacklevel=2,
                )

        ifos = list(ts_dict.keys())
        logging.info(f"Fetched data for detectors {ifos}")
        if ifos not in [["H1", "L1"], ["H1", "L1", "V1"]]:
            raise ValueError(
                f"Event {event} does not have the required detectors. "
                f"Expected ['H1', 'L1'] or ['H1', 'L1', 'V1'], got {ifos}"
            )
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
        logging.info(f"Loading data from file for event {event}")
        with h5py.File(datafile, "r") as f:
            ifos = list(f.keys())
            data = np.stack([f[ifo][:] for ifo in ifos])[None]
            event_time = f.attrs["tc"]
            t0 = f.attrs["t0"]
        logging.info(f"Loaded data for detectors {ifos}")

    return data, ifos, t0, event_time
