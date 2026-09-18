import pytest
import torch
import torch.nn.functional as F
from conftest import FDURATION, FFTLENGTH, HIGHPASS, NUM_CHANNELS, SAMPLE_RATE
from ml4gw.transforms import Heterodyne

from buoy.utils.augmentation import HeterodyneAugmentor
from buoy.utils.preprocessing import (
    BackgroundSnapshotter,
    BatchWhitener,
    PsdEstimator,
)

INFERENCE_SAMPLING_RATE = 32
KERNEL_LENGTH = 1.0
PSD_LENGTH = 8.0
BATCH_SIZE = 8


def make_batch_whitener(**kwargs):
    defaults = {
        "kernel_length": KERNEL_LENGTH,
        "sample_rate": SAMPLE_RATE,
        "inference_sampling_rate": INFERENCE_SAMPLING_RATE,
        "batch_size": BATCH_SIZE,
        "fduration": FDURATION,
        "fftlength": FFTLENGTH,
        "highpass": HIGHPASS,
    }
    return BatchWhitener(**{**defaults, **kwargs})


def whitener_input(whitener):
    """Minimum-sized input for a BatchWhitener forward pass."""
    fsize = int(FDURATION * SAMPLE_RATE)
    kernel = (
        (BATCH_SIZE - 1) * whitener.stride_size + whitener.kernel_size + fsize
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    return torch.randn(NUM_CHANNELS, kernel + min_background)


# --- BackgroundSnapshotter ---


def test_snapshotter_output_shapes():
    snapshotter = BackgroundSnapshotter(
        psd_length=PSD_LENGTH,
        kernel_length=KERNEL_LENGTH,
        fduration=FDURATION,
        sample_rate=SAMPLE_RATE,
        inference_sampling_rate=INFERENCE_SAMPLING_RATE,
    )
    stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    update = torch.randn(1, NUM_CHANNELS, stride)
    snapshot = torch.zeros(1, NUM_CHANNELS, snapshotter.state_size)

    x, new_snapshot = snapshotter(update, snapshot)

    assert x.shape == (1, NUM_CHANNELS, snapshotter.state_size + stride)
    assert new_snapshot.shape == snapshot.shape


def test_snapshotter_state_is_tail_of_output():
    snapshotter = BackgroundSnapshotter(
        psd_length=PSD_LENGTH,
        kernel_length=KERNEL_LENGTH,
        fduration=FDURATION,
        sample_rate=SAMPLE_RATE,
        inference_sampling_rate=INFERENCE_SAMPLING_RATE,
    )
    stride = int(SAMPLE_RATE / INFERENCE_SAMPLING_RATE)
    update = torch.randn(1, NUM_CHANNELS, stride)
    snapshot = torch.randn(1, NUM_CHANNELS, snapshotter.state_size)

    x, new_snapshot = snapshotter(update, snapshot)

    assert torch.allclose(new_snapshot, x[:, :, -snapshotter.state_size :])


# --- PsdEstimator ---


def test_psd_estimator_output_shapes():
    estimator = PsdEstimator(
        length=KERNEL_LENGTH,
        sample_rate=SAMPLE_RATE,
        fftlength=FFTLENGTH,
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    x = torch.randn(NUM_CHANNELS, estimator.size + min_background)

    data, psd = estimator(x)

    assert data.shape[-1] == estimator.size
    assert psd.shape[0] == NUM_CHANNELS


def test_psd_estimator_returns_tail_as_data():
    estimator = PsdEstimator(
        length=KERNEL_LENGTH,
        sample_rate=SAMPLE_RATE,
        fftlength=FFTLENGTH,
    )
    min_background = int(FFTLENGTH * SAMPLE_RATE)
    x = torch.randn(NUM_CHANNELS, estimator.size + min_background)

    data, _ = estimator(x)

    assert torch.allclose(data, x[..., -estimator.size :])


# --- BatchWhitener ---


def test_batch_whitener_returns_tensor_with_correct_shape():
    whitener = make_batch_whitener()
    x = whitener_input(whitener)

    result = whitener(x)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (BATCH_SIZE, NUM_CHANNELS, whitener.kernel_size)


def test_batch_whitener_return_whitened_returns_tuple():
    whitener = make_batch_whitener(return_whitened=True)
    x = whitener_input(whitener)

    batches, whitened = whitener(x)

    assert isinstance(batches, torch.Tensor)
    assert isinstance(whitened, torch.Tensor)
    assert batches.shape == (BATCH_SIZE, NUM_CHANNELS, whitener.kernel_size)


def test_batch_whitener_raises_on_wrong_ndim():
    whitener = make_batch_whitener()
    with pytest.raises(ValueError, match="2 or 3 dimensional"):
        whitener(torch.zeros(4))
    with pytest.raises(ValueError, match="2 or 3 dimensional"):
        whitener(torch.zeros(1, 2, 3, 4))


def test_with_heterodyne_augmentor_top_k():

    augmentor = HeterodyneAugmentor(
        sample_rate=SAMPLE_RATE,
        kernel_length=KERNEL_LENGTH,
        chirp_mass_low=1.0,
        chirp_mass_high=2.5,
        num_chirp_masses=10,
        chirp_mass_spacing="log",
        keep_last_n_seconds=1.5,
        top_k=5,
    )

    whitener = make_batch_whitener()
    whitener_aug = make_batch_whitener(augmentor=augmentor)

    x = whitener_input(whitener)

    heterodyne = Heterodyne(
        sample_rate=SAMPLE_RATE,
        kernel_length=KERNEL_LENGTH,
        chirp_mass=augmentor.chirp_mass_grid,
        return_type="time",
    )

    kernels = whitener_aug(x)

    kernels_heterodyned = heterodyne(whitener(x))
    _B, _C, _M, _T = kernels_heterodyned.shape
    avgpool = F.avg_pool1d(
        torch.abs(kernels_heterodyned.reshape(_B, _C * _M, _T)),
        31,
        stride=1,
        padding=15,
    ).reshape(_B, _C, _M, _T)
    avgpool_snr = torch.sqrt(
        (avgpool[..., -int(1.5 * SAMPLE_RATE) :] ** 2).sum(dim=1)
    )
    vals = torch.max(avgpool_snr, dim=-1).values
    idx = torch.topk(vals, k=5, dim=-1).indices
    idx_expand = idx[:, None, :, None].expand(-1, _C, -1, _T)
    kernels_heterodyned = torch.gather(
        kernels_heterodyned, dim=2, index=idx_expand
    )
    kernels_heterodyned = kernels_heterodyned.reshape(_B, _C * 5, _T)
    kernels_heterodyned = kernels_heterodyned[..., -int(1.5 * SAMPLE_RATE) :]
    kernels = kernels_heterodyned

    kernels_aug = whitener_aug(x)
    assert torch.allclose(kernels_aug, kernels)
