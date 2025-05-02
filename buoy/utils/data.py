import logging
from pathlib import Path

import gwosc
import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict


def slice_amplfi_data(
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
    window_length = int(
        (amplfi_kernel_length + amplfi_fduration) * sample_rate
    )
    window_end = window_start + window_length

    psd_start = window_start - int(amplfi_psd_length * sample_rate)

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
