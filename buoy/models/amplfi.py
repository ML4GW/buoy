import logging
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch
from jsonargparse import ArgumentParser
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

from buoy.utils.data import get_local_or_hf, slice_amplfi_data
from buoy.utils.pe import postprocess_samples, run_amplfi

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from amplfi.train.data.utils.utils import ParameterSampler

REPO_ID = "ML4GW/amplfi"


@dataclass
class AmplfiConfig:
    architecture: "FlowArchitecture"
    parameter_sampler: "ParameterSampler"
    sample_rate: float
    kernel_length: float
    inference_params: List[str]
    event_position: float
    psd_length: float
    fduration: float
    fftlength: float
    highpass: float
    lowpass: Optional[float] = None


class Amplfi(AmplfiConfig):
    def __init__(
        self,
        model_weights: str,
        config: str,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logging.debug(f"Using device: {self.device}")

        model_weights = get_local_or_hf(
            filename=model_weights,
            repo_id=REPO_ID,
            descriptor="AMPLFI model weights",
        )
        config = get_local_or_hf(
            filename=config,
            repo_id=REPO_ID,
            descriptor="AMPLFI model config",
        )

        parser = ArgumentParser()
        parser.add_class_arguments(AmplfiConfig)
        parser.link_arguments(
            "inference_params",
            "architecture.init_args.num_params",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )
        args = parser.parse_path(config)
        args = parser.instantiate_classes(args)

        super().__init__(**vars(args))

        model, scaler = self.load_model(
            args.architecture,
            model_weights,
            len(args.inference_params),
        )
        self.model = model.to(self.device)
        self.scaler = scaler.to(self.device)

        self.setup()

    def load_model(
        self,
        model: "FlowArchitecture",
        model_weights: str,
        num_params: int,
    ):
        checkpoint = torch.load(
            model_weights, map_location="cpu", weights_only=False
        )
        arch_weights = {
            k[6:]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(arch_weights)
        model.eval()
        scaler_weights = {
            k[len("scaler.") :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("scaler.")
        }
        scaler = ChannelWiseScaler(num_params)
        scaler.load_state_dict(scaler_weights)
        return model, scaler

    def setup(self):
        self.spectral_density = SpectralDensity(
            sample_rate=self.sample_rate,
            fftlength=self.fftlength,
            average="median",
        ).to(self.device)
        self.whitener = Whiten(
            fduration=self.fduration,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
            lowpass=self.lowpass,
        ).to(self.device)

    @property
    def minimum_data_size(self) -> int:
        """Minimum data size required for the model to run."""
        return int(
            (self.kernel_length + self.psd_length + self.fduration)
            * self.sample_rate
        )

    def __call__(
        self,
        data: torch.Tensor,
        t0: float,
        tc: float,
        samples_per_event: int,
    ):
        if data.shape[-1] < self.minimum_data_size:
            raise ValueError(
                f"Data size {data.shape[-1]} is less than the minimum "
                f"size of {self.minimum_data_size}"
            )

        psd_data, window = slice_amplfi_data(
            data=data,
            sample_rate=self.sample_rate,
            t0=t0,
            tc=tc,
            kernel_length=self.kernel_length,
            event_position=self.event_position,
            psd_length=self.psd_length,
            fduration=self.fduration,
        )

        samples = run_amplfi(
            amplfi_strain=window,
            amplfi_psd_strain=psd_data,
            samples_per_event=samples_per_event,
            spectral_density=self.spectral_density,
            amplfi_whitener=self.whitener,
            amplfi=self.model,
            std_scaler=self.scaler,
            device=self.device,
        )
        samples = samples.cpu()
        result = postprocess_samples(
            samples=samples,
            event_time=tc,
            inference_params=self.inference_params,
            parameter_sampler=self.parameter_sampler,
        )
        return result
