import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from jsonargparse import ArgumentParser

from buoy.utils.data import get_local_or_hf
from buoy.utils.preprocessing import BackgroundSnapshotter, BatchWhitener

REPO_ID = "ML4GW/aframe"

# TODO: Allow specification of a cache directory
# TODO: When we have multiple model versions, provide
# a way to specify which one to use


@dataclass
class AframeConfig:
    sample_rate: float
    kernel_length: float
    psd_length: float
    fduration: float
    highpass: float
    fftlength: float
    inference_sampling_rate: float
    batch_size: int
    aframe_right_pad: float
    integration_window_length: float
    lowpass: Optional[float] = None


class Aframe(AframeConfig):
    def __init__(
        self,
        model_weights: Optional[str] = "aframe.pt",
        config: Optional[str] = "aframe_config.yaml",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logging.debug(f"Using device: {self.device}")

        model_weights = get_local_or_hf(
            filename=model_weights,
            repo_id=REPO_ID,
            descriptor="Aframe model weights",
        )
        self.model = torch.jit.load(model_weights).to(self.device)

        config = get_local_or_hf(
            filename=config,
            repo_id=REPO_ID,
            descriptor="Aframe model config",
        )

        parser = ArgumentParser()
        parser.add_class_arguments(AframeConfig)
        args = parser.parse_path(config)

        super().__init__(**vars(args))
        self.setup()

    def setup(self):
        self.whitener = BatchWhitener(
            kernel_length=self.kernel_length,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
            batch_size=self.batch_size,
            fduration=self.fduration,
            fftlength=self.fftlength,
            highpass=self.highpass,
            lowpass=self.lowpass,
        ).to(self.device)
        self.snapshotter = BackgroundSnapshotter(
            psd_length=self.psd_length,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            sample_rate=self.sample_rate,
            inference_sampling_rate=self.inference_sampling_rate,
        ).to(self.device)

    def get_time_offset(self):
        """
        Estimate the time offset between the peak of the integrated
        outputs and the merger time of the signal
        """

        time_offset = (
            # end of the first kernel in batch
            1 / self.inference_sampling_rate
            # account for whitening padding
            - self.fduration / 2
            # distance coalescence time lies away from right edge
            - self.aframe_right_pad
            # account for time to build peak
            - self.integration_window_length
        )

        return time_offset

    @property
    def minimum_data_size(self) -> int:
        """
        The minimum length of data, in samples, required
        for the model to run with its current configuration
        """
        fsize = int(self.fduration * self.sample_rate)
        psd_size = int(self.psd_length * self.sample_rate)
        total_size = (
            psd_size
            + fsize
            + self.whitener.kernel_size
            + (self.batch_size - 1) * self.whitener.stride_size
        )
        return total_size

    def __call__(
        self,
        data: torch.Tensor,
        t0: float,
    ):
        """
        Run the aframe model over the data
        """
        if data.shape[-1] < self.minimum_data_size:
            raise ValueError(
                f"Data size {data.shape[-1]} is less than the minimum "
                f"size of {self.minimum_data_size}"
            )

        step_size = int(self.batch_size * self.whitener.stride_size)
        state = torch.zeros(
            (1, 2, self.snapshotter.state_size), device=self.device
        )

        # Ensure data is on the correct device
        data = data.to(self.device)

        # Iterate through the data, making predictions
        ys, batches = [], []
        start = 0
        with torch.no_grad():
            for start in range(0, data.shape[-1] - step_size, step_size):
                stop = start + step_size
                x = data[:, :, start:stop]

                # Forward through snapshotter and whitener
                x, state = self.snapshotter(x, state)
                batch = self.whitener(x)

                # Run model inference
                y_hat = self.model(batch).detach().cpu()[:, 0]
                ys.append(y_hat)
                batches.append(batch.detach().cpu())

        ys = torch.cat(ys).numpy()
        batches = torch.cat(batches).numpy()

        tf = t0 + len(ys) / self.inference_sampling_rate
        times = np.arange(t0, tf, 1 / self.inference_sampling_rate)

        window_size = (
            int(self.integration_window_length * self.inference_sampling_rate)
            + 1
        )
        window = np.ones(window_size) / window_size
        integrated = np.convolve(ys, window, mode="full")
        integrated = integrated[: -window_size + 1]

        return times, ys, integrated
