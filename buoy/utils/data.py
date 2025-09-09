import logging
import warnings
from pathlib import Path

import gwosc
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
    filename: Path,
    repo_id: str,
    descriptor: str,
):
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

    Returns:
        The path to the file.
    """
    if Path(filename).exists():
        logging.info(f"Loading {descriptor} from {filename}")
        return filename
    else:
        try:
            logging.info(
                f"Downloading {descriptor} from HuggingFace "
                "or loading from cache"
            )
            return hf_hub_download(repo_id=repo_id, filename=str(filename))
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
):
    """
    Slice the data to get the PSD window and kernel for amplfi
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


def get_data(
    event: str,
    sample_rate: float,
    psd_length: float,
    datadir: Path,
):
    if event.startswith("GW"):
        event_time = gwosc.datasets.event_gps(event)
        ifos = sorted(gwosc.datasets.event_detectors(event))
    else:
        client = GraceDb()
        if event.startswith("G"):
            response = client.event(event).json()
            event_time = response["gpstime"]
            ifos = response["instruments"].split(",")
        elif event.startswith("S"):
            response = client.superevent(event).json()
            event_time = response["preferred_event_data"]["gpstime"]
            ifos = response["preferred_event_data"]["instruments"].split(",")
        else:
            raise ValueError(
                f"Event {event} is not a valid event name. "
                "Should be a valid GPS time, a known gravitational wave "
                "event name (e.g. GW123456), or a GraceDB event or superevent "
                "(e.g. G123456 or S123456)."
            )

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
            if start < 1269363618:
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
