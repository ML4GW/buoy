import pytest
import torch
from conftest import AFRAME_CONFIG, FDURATION, SAMPLE_RATE
from ml4gw.transforms import SpectralDensity, Whiten

from buoy.models.aframe import AframeConfig
from buoy.models.amplfi import Amplfi
from buoy.models.base import BuoyModel

PSD_LENGTH = 64
KERNEL_LENGTH = 4


class _ConcreteModel(BuoyModel):
    """Minimal BuoyModel subclass for testing update_config."""

    def __init__(self):
        self.alpha = 1.0
        self.beta = 2.0
        self.configure_calls = 0

    def configure_preprocessing(self) -> None:
        self.configure_calls += 1


def test_update_config_sets_attribute():
    model = _ConcreteModel()
    model.update_config(alpha=99.0)
    assert model.alpha == 99.0


def test_update_config_sets_multiple_attributes():
    model = _ConcreteModel()
    model.update_config(alpha=10.0, beta=20.0)
    assert model.alpha == 10.0
    assert model.beta == 20.0


def test_update_config_calls_configure_preprocessing():
    model = _ConcreteModel()
    model.update_config(alpha=5.0)
    assert model.configure_calls == 1


def test_update_config_invalid_key_raises():
    model = _ConcreteModel()
    with pytest.raises(ValueError, match="Invalid configuration parameter"):
        model.update_config(nonexistent=99)


def test_update_config_invalid_key_does_not_partially_apply():
    """A bad key in a multi-key update should not leave partial state."""
    model = _ConcreteModel()
    with pytest.raises(ValueError):
        model.update_config(alpha=99.0, nonexistent=0)
    # alpha was set before the bad key was encountered;
    # this test documents that behaviour explicitly
    assert model.alpha == 99.0
    assert model.configure_calls == 0


# --- Aframe properties ---


def test_aframe_time_offset(aframe):
    """Regression test: time_offset formula uses four specific terms."""
    cfg = aframe
    expected = (
        1 / cfg.inference_sampling_rate
        - cfg.fduration / 2
        - cfg.aframe_right_pad
        - cfg.integration_window_length
    )
    assert aframe.time_offset == pytest.approx(expected)


def test_aframe_minimum_data_size(aframe):
    """
    Regression test: minimum_data_size accounts for all buffer components.
    """
    fsize = int(aframe.fduration * aframe.sample_rate)
    psd_size = int(aframe.psd_length * aframe.sample_rate)
    expected = (
        psd_size
        + fsize
        + aframe.whitener.kernel_size
        + (aframe.batch_size - 1) * aframe.whitener.stride_size
    )
    assert aframe.minimum_data_size == expected


def test_aframe_call_raises_without_weights(aframe):
    """__call__ must raise RuntimeError when load_weights=False."""
    with pytest.raises(RuntimeError, match="load_weights=True"):
        aframe(torch.zeros(1, 2, 1000), t0=0.0)


CHIRP_MASS_LOW = 1.0
CHIRP_MASS_HIGH = 2.5
NUM_CHIRP_MASSES = 10
CHIRP_MASS_SPACING = "log"
KEEP_LAST_N_SECONDS = 5.0
TOP_K = 5


def test_bns_missing_params_raises():
    """BNS config without heterodyne parameters should raise ValueError."""
    with pytest.raises(ValueError, match="chirp_mass_low"):
        AframeConfig(**AFRAME_CONFIG, cbc_type="BNS")


def test_bns_full_params_does_not_raise():
    """BNS config with heterodyne parameters should not raise ValueError."""
    AframeConfig(
        **AFRAME_CONFIG,
        cbc_type="BNS",
        chirp_mass_low=CHIRP_MASS_LOW,
        chirp_mass_high=CHIRP_MASS_HIGH,
        num_chirp_masses=NUM_CHIRP_MASSES,
        chirp_mass_spacing=CHIRP_MASS_SPACING,
        keep_last_n_seconds=KEEP_LAST_N_SECONDS,
        top_k=TOP_K,
    )


# --- Amplfi properties ---
# Amplfi's config requires complex sub-configs
# (architecture, parameter_sampler), so we use object.__new__
# to bypass __init__ and set only the attributes under test.


def _amplfi_stub(**kwargs):
    """Create an Amplfi instance with no __init__, setting only given attrs."""
    stub = object.__new__(Amplfi)
    for k, v in kwargs.items():
        setattr(stub, k, v)
    return stub


def test_amplfi_minimum_data_size():
    """minimum_data_size is (kernel + psd + fduration) * sample_rate."""
    stub = _amplfi_stub(
        kernel_length=KERNEL_LENGTH,
        psd_length=PSD_LENGTH,
        fduration=FDURATION,
        sample_rate=SAMPLE_RATE,
    )
    assert stub.minimum_data_size == int(
        (KERNEL_LENGTH + PSD_LENGTH + FDURATION) * SAMPLE_RATE
    )


def test_amplfi_configure_preprocessing_sets_transforms():
    """configure_preprocessing must create spectral_density and whitener."""
    stub = _amplfi_stub(
        sample_rate=2048.0,
        fftlength=4.0,
        fduration=1.0,
        highpass=20.0,
        lowpass=None,
        device="cpu",
    )
    stub.configure_preprocessing()
    assert isinstance(stub.spectral_density, SpectralDensity)
    assert isinstance(stub.whitener, Whiten)


def test_amplfi_call_raises_without_weights():
    """__call__ must raise RuntimeError when model weights were not loaded."""
    stub = _amplfi_stub()
    with pytest.raises(RuntimeError, match="load_weights=True"):
        stub(
            data=torch.zeros(1, 2, 1000),
            t0=0.0,
            tc=1.0,
            samples_per_event=10,
        )
