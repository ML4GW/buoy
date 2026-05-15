import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jsonargparse import ArgumentParser
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten

from buoy.models.base import BuoyModel
from buoy.utils.data import get_local_or_hf, slice_amplfi_data
from buoy.utils.pe import postprocess_samples, run_amplfi

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from amplfi.train.prior import AmplfiPrior
    from amplfi.utils.result import AmplfiResult

REPO_ID = "ML4GW/amplfi"


@dataclass
class AmplfiConfig:
    architecture: "FlowArchitecture"
    parameter_sampler: "AmplfiPrior"
    sample_rate: float
    kernel_length: float
    inference_params: list[str]
    event_position: float
    psd_length: float
    fduration: float
    fftlength: float
    highpass: float
    lowpass: float | None = None


class Amplfi(AmplfiConfig, BuoyModel):
    """
    AMPLFI normalizing-flow model for rapid gravitational wave parameter
    estimation.

    Wraps a trained Lightning checkpoint and its associated preprocessing
    pipeline. Config attributes are always loaded; the flow weights are
    only loaded when ``load_weights=True``.
    """

    def __init__(
        self,
        model_weights: str | Path = "amplfi-hlv.ckpt",
        config: str | Path = "amplfi-hlv-config.yaml",
        device: str | None = None,
        revision: str | None = None,
        load_weights: bool = True,
        cache_dir: str | Path | None = None,
    ):
        """
        Args:
            model_weights:
                Path to Lightning checkpoint (`.ckpt`) or filename in the
                ``ML4GW/amplfi`` HuggingFace repository.
            config:
                Path to YAML config or filename in the ``ML4GW/amplfi``
                HuggingFace repository.
            device:
                Device to run inference on (``"cpu"`` or ``"cuda"``).
                If ``None``, uses CUDA when available, otherwise CPU.
            revision:
                HuggingFace repository revision (branch, tag, or commit
                hash). If ``None``, uses the default branch.
            load_weights:
                If ``True``, download and load the flow weights. Set to
                ``False`` to load only the config (e.g. to inspect
                hyperparameters without allocating GPU memory).
            cache_dir:
                Local directory for HuggingFace download cache. If
                ``None``, uses the default ``~/.cache/huggingface``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logging.debug(f"Using device: {self.device}")

        config = get_local_or_hf(
            filename=config,
            repo_id=REPO_ID,
            descriptor="AMPLFI model config",
            revision=revision,
            cache_dir=cache_dir,
        )

        parser = ArgumentParser()
        parser.add_class_arguments(
            AmplfiConfig, fail_untyped=False, sub_configs=True
        )
        parser.link_arguments(
            "inference_params",
            "architecture.init_args.num_params",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )
        args = parser.parse_path(config)
        args = parser.instantiate_classes(args)

        super().__init__(**vars(args))

        if load_weights:
            weights_path = get_local_or_hf(
                filename=model_weights,
                repo_id=REPO_ID,
                descriptor="AMPLFI model weights",
                revision=revision,
                cache_dir=cache_dir,
            )
            model, scaler = self.load_model(
                args.architecture,
                weights_path,
                len(args.inference_params),
            )
            self.model = model.to(self.device)
            self.scaler = scaler.to(self.device)

        self.configure_preprocessing()

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

    def configure_preprocessing(self) -> None:
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
    ) -> "AmplfiResult":
        if not hasattr(self, "model"):
            raise RuntimeError(
                "AMPLFI model weights were not loaded. "
                "Re-initialize with load_weights=True."
            )
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
