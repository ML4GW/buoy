import logging
from dataclasses import dataclass
from typing import Optional

import torch
from jsonargparse import ArgumentParser

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
        cfg = parser.parse_path(config)
        args = parser.instantiate_classes(cfg)

        super().__init__(**vars(args))

    def setup(
        self,
        sample_rate: float,
        kernel_length: float,
        psd_length: float,
        fduration: float,
        highpass: float,
        fftlength: float,
        inference_sampling_rate: float,
        batch_size: int,
        aframe_right_pad: float,
        integration_window_length: float,
        lowpass: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.fduration = fduration
        self.highpass = highpass
        self.inference_sampling_rate = inference_sampling_rate
        self.batch_size = batch_size
        self.aframe_right_pad = aframe_right_pad
        self.integration_window_length = integration_window_length
        self.lowpass = lowpass

        self.whitener = BatchWhitener(
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
            lowpass=lowpass,
        ).to(self.device)
        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to(self.device)
