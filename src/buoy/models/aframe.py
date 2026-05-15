import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jsonargparse import ArgumentParser

from buoy.models.base import BuoyModel
from buoy.utils.data import get_local_or_hf
from buoy.utils.preprocessing import BackgroundSnapshotter, BatchWhitener

REPO_ID = "ML4GW/aframe"


@dataclass
class AframeConfig:
    sample_rate: float
    kernel_length: float
    psd_length: float
    fduration: float
    highpass: float
    fftlength: float
    inference_sampling_rate: float
    offline_sampling_rate: float
    batch_size: int
    aframe_right_pad: float
    integration_window_length: float
    lowpass: float | None = None


class Aframe(AframeConfig, BuoyModel):
    """
    Aframe neural network model for gravitational wave detection.

    Wraps a trained TorchScript model and its associated preprocessing
    pipeline. Config attributes (``sample_rate``, ``psd_length``, etc.)
    are always loaded; neural network weights are only loaded when
    ``load_weights=True``.
    """

    def __init__(
        self,
        model_weights: str | Path = "aframe.pt",
        config: str | Path = "aframe_config.yaml",
        device: str | None = None,
        revision: str | None = None,
        load_weights: bool = True,
        cache_dir: str | Path | None = None,
    ):
        """
        Args:
            model_weights:
                Path to TorchScript weights (`.pt`) or filename in the
                ``ML4GW/aframe`` HuggingFace repository.
            config:
                Path to YAML config or filename in the ``ML4GW/aframe``
                HuggingFace repository.
            device:
                Device to run inference on (``"cpu"`` or ``"cuda"``).
                If ``None``, uses CUDA when available, otherwise CPU.
            revision:
                HuggingFace repository revision (branch, tag, or commit
                hash). If ``None``, uses the default branch.
            load_weights:
                If ``True``, download and load the neural network weights.
                Set to ``False`` to load only the config (e.g. to inspect
                hyperparameters without allocating GPU memory).
            cache_dir:
                Local directory for HuggingFace download cache. If
                ``None``, uses the default ``~/.cache/huggingface``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logging.debug(f"Using device: {self.device}")

        if load_weights:
            weights_path = get_local_or_hf(
                filename=model_weights,
                repo_id=REPO_ID,
                descriptor="Aframe model weights",
                revision=revision,
                cache_dir=cache_dir,
            )
            self.model = torch.jit.load(weights_path).to(self.device)

        config = get_local_or_hf(
            filename=config,
            repo_id=REPO_ID,
            descriptor="Aframe model config",
            revision=revision,
            cache_dir=cache_dir,
        )

        parser = ArgumentParser()
        parser.add_class_arguments(AframeConfig)
        args = parser.parse_path(config)

        super().__init__(**vars(args))
        self.configure_preprocessing()
        self.online_offline_stride = int(
            self.inference_sampling_rate / self.offline_sampling_rate
        )

    def configure_preprocessing(self) -> None:
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

    @property
    def time_offset(self) -> float:
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the aframe model over the data
        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Aframe model weights were not loaded. "
                "Re-initialize with load_weights=True."
            )
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

        online_window_size = (
            int(self.integration_window_length * self.inference_sampling_rate)
            + 1
        )
        online_window = np.ones(online_window_size) / online_window_size
        timing_integrated = np.convolve(ys, online_window, mode="full")
        timing_integrated = timing_integrated[: -online_window_size + 1]

        offline_window_size = (
            int(self.integration_window_length * self.offline_sampling_rate)
            + 1
        )
        offline_window = np.ones(offline_window_size) / offline_window_size
        signif_integrated = np.convolve(
            ys[:: self.online_offline_stride], offline_window, mode="full"
        )
        signif_integrated = signif_integrated[: -offline_window_size + 1]

        return times, ys, timing_integrated, signif_integrated
