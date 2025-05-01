from pathlib import Path
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy import io
from ligo.skymap.tool.ligo_skymap_plot import main as ligo_skymap_plot

if TYPE_CHECKING:
    from amplfi.utils.result import AmplfiResult

plt.rcParams.update(
    {
        "font.size": 16,
        "figure.dpi": 250,
    }
)


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

    plt.plot(times, ys, color="tab:gray", label="Raw", lw=2)
    plt.plot(times, integrated, color="k", label="Integrated", lw=2)
    plt.ylabel("Detection statistic")
    plt.legend(loc="upper right")
    plt.xlim(t0 + 94, t0 + 102)
    plt.grid()
    plt.savefig(plotdir / "aframe_response.png", bbox_inches="tight")


def plot_amplfi_result(
    result: "AmplfiResult",
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
