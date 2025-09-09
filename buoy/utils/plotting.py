import warnings
from pathlib import Path
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy import io
from gwpy.timeseries import TimeSeries
from ligo.skymap.tool.ligo_skymap_plot import main as ligo_skymap_plot

if TYPE_CHECKING:
    from amplfi.utils.result import AmplfiResult

plt.rcParams.update(
    {
        "font.size": 16,
        "figure.dpi": 250,
    }
)

IFOS = ["H1", "L1", "V1"]


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
    plt.plot(whitened_times, whitened[0], label="H1", alpha=0.3)
    plt.plot(whitened_times, whitened[1], label="L1", alpha=0.3)
    plt.xlabel("Time from event (s)")
    plt.axvline(tc, color="tab:red", linestyle="--", label="Predicted time")
    plt.axvline(0, color="k", linestyle="--", label="Event time")
    plt.ylabel("Whitened strain")
    plt.legend(loc="upper left")
    plt.grid()
    plt.twinx()

    plt.plot(times, ys, color="tab:gray", label="Network output", lw=2)
    plt.plot(times, integrated, color="k", label="Integrated output", lw=2)
    plt.ylabel("Detection statistic")
    plt.legend(loc="upper right")
    plt.xlim(t0 + 94, t0 + 102)
    plt.grid()
    plt.title(f"Maximum detection statistic: {max(integrated):.2f}")
    plt.savefig(plotdir / "aframe_response.png", bbox_inches="tight")
    plt.close()


def plot_amplfi_result(
    result: "AmplfiResult",
    nside: int,
    min_samples_per_pix: int,
    use_distance: bool,
    ifos: List[str],
    datadir: Path,
    plotdir: Path,
):
    """
    Plot the skymap and corner plot from amplfi
    """

    suffix = "".join([ifo[0] for ifo in ifos])

    skymap = result.to_skymap(
        use_distance=use_distance,
        adaptive=True,
        min_samples_per_pix_dist=min_samples_per_pix,
        metadata={"INSTRUME": ",".join(ifos)},
    )
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
        parameters=[
            "chirp_mass",
            "mass_ratio",
            "distance",
            "mass_1",
            "mass_2",
        ],
        filename=corner_fname,
    )
    plt.close()


def q_plots(
    data: np.ndarray,
    t0: float,
    plotdir: Path,
    gpstime: float,
    sample_rate: float,
    amplfi_highpass: float,
) -> None:
    """
    Create Q-plots of the whitened AMPLFI data

    Args:
        data: Array containing strain data
        t0: Starting gpstime of the strain data
        plotdir: Directory to save the plots
        gpstime: GPS time of the event
        sample_rate: Sample rate of the data
        amplfi_highpass: Highpass value used for AMPLFI
    """
    for i, ifo in enumerate(IFOS[: len(data)]):
        ts = TimeSeries(data[i], sample_rate=sample_rate, t0=t0)
        try:
            q_transform = ts.q_transform(
                whiten=True,
                gps=gpstime,
                logf=True,
                outseg=(gpstime - 1.5, gpstime + 0.5),
                frange=(amplfi_highpass, np.inf),
            )
            qplot = q_transform.plot(epoch=gpstime)
        except ValueError as e:
            warnings.warn(
                f"Failed to create Q-plot for {ifo} due to error {e}",
                stacklevel=2,
            )
            continue
        qplot.colorbar(
            clim=(0, np.max(q_transform.value)), label="Normalized energy"
        )
        ax = qplot.gca()
        ax.set_yscale("log")
        qplot.savefig(plotdir / f"{ifo}_qtransform.png", dpi=150)
        plt.close()
