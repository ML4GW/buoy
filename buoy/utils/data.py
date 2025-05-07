import logging
from pathlib import Path

import gwosc
import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


def get_local_or_hf(
    filename: str,
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
            return hf_hub_download(repo_id=repo_id, filename=filename)
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

    return data, ifos, t0, event_time
