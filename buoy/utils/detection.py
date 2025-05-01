from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from buoy.utils.preprocessing import BatchWhitener, BackgroundSnapshotter


def get_time_offset(
    inference_sampling_rate: float,
    fduration: float,
    integration_window_length: float,
    aframe_right_pad: float,
):
    time_offset = (
        # end of the first kernel in batch
        1 / inference_sampling_rate
        # account for whitening padding
        - fduration / 2
        # distance coalescence time lies away from right edge
        - aframe_right_pad
        # account for time to build peak
        - integration_window_length
    )

    return time_offset


def run_aframe(
    data: torch.Tensor,
    t0: float,
    aframe: torch.nn.Module,
    whitener: "BatchWhitener",
    snapshotter: "BackgroundSnapshotter",
    inference_sampling_rate: float,
    integration_window_length: float,
    batch_size: int,
    device: str = "cpu",
):
    """
    Run the aframe model over the data
    """
    step_size = int(batch_size * whitener.stride_size)

    # Iterate through the data, making predictions
    ys, batches = [], []
    start = 0
    state = torch.zeros((1, 2, snapshotter.state_size)).to(device)
    while start < (data.shape[-1] - step_size):
        stop = start + step_size
        x = data[:, :, start:stop]
        with torch.no_grad():
            x, state = snapshotter(x, state)
            batch = whitener(x)
            y_hat = aframe(batch)[:, 0].cpu().numpy()

        batches.append(batch.cpu().numpy())
        ys.append(y_hat)
        start += step_size
    batches = np.concatenate(batches)
    ys = np.concatenate(ys)

    times = np.arange(
        t0, t0 + len(ys) / inference_sampling_rate, 1 / inference_sampling_rate
    )
    window_size = int(integration_window_length * inference_sampling_rate) + 1
    window = np.ones((window_size,)) / window_size
    integrated = np.convolve(ys, window, mode="full")
    integrated = integrated[: -window_size + 1]

    return times, ys, integrated
