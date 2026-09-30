import pytest
import yaml

from buoy.models.aframe import Aframe

# Shared constants used across multiple test files
SAMPLE_RATE = 2048.0
FDURATION = 1.0
FFTLENGTH = 2.0
HIGHPASS = 20.0
NUM_CHANNELS = 2

AFRAME_CONFIG = {
    "sample_rate": SAMPLE_RATE,
    "kernel_length": 4.0,
    "psd_length": 8.0,
    "fduration": FDURATION,
    "highpass": HIGHPASS,
    "fftlength": FFTLENGTH,
    "inference_sampling_rate": 32.0,
    "offline_sampling_rate": 4.0,
    "batch_size": 8,
    "aframe_right_pad": 0.25,
    "integration_window_length": 0.5,
}


@pytest.fixture
def aframe(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(AFRAME_CONFIG))
    return Aframe(config=cfg, load_weights=False, device="cpu")
