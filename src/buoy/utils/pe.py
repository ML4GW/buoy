import logging
from typing import TYPE_CHECKING

import astropy.units as u
import lal
import numpy as np
import pandas as pd
import torch
from amplfi.utils.result import AmplfiResult
from astropy import cosmology
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from amplfi.train.architectures.flows import FlowArchitecture
    from amplfi.train.prior import AmplfiPrior
    from ml4gw.transforms import SpectralDensity, Whiten


def get_redshifts(distances, num_pts=10000):
    """
    Compute redshift using the Planck18 cosmology. Implementation
    taken from https://git.ligo.org/emfollow/em-properties/em-bright/-/blob/main/ligo/em_bright/em_bright.py

    This function accepts distance values in Mpc and computes
    redshifts by interpolating the distance-redshift relation.
    This process is much faster compared to astropy.cosmology
    APIs with lesser than a percent difference.
    """
    func = cosmology.Planck18.luminosity_distance  # ty: ignore[unresolved-attribute]
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    z_min = cosmology.z_at_value(func=func, fval=min_dist * u.Mpc)  # ty: ignore[unresolved-attribute]
    z_max = cosmology.z_at_value(func=func, fval=max_dist * u.Mpc)  # ty: ignore[unresolved-attribute]
    z_steps = np.linspace(
        z_min - (0.1 * z_min), z_max + (0.1 * z_max), num_pts
    )
    lum_dists = cosmology.Planck18.luminosity_distance(z_steps)  # ty: ignore[unresolved-attribute]
    s = interp1d(lum_dists, z_steps)
    redshifts = s(distances)
    return redshifts


def filter_samples(samples, parameter_sampler, inference_params):
    net_mask = torch.ones((samples.shape[0],), dtype=torch.bool)
    priors = parameter_sampler.priors
    for i, param in enumerate(inference_params):
        prior = priors[param]
        curr_samples = samples[:, i]
        log_probs = prior.log_prob(curr_samples)
        mask = log_probs == float("-inf")

        logging.debug(
            f"Removed {mask.sum()}/{len(mask)} samples for parameter "
            f"{param} outside of prior range"
        )

        net_mask &= ~mask

    logging.debug(
        f"Removed {(~net_mask).sum()}/{len(net_mask)} total samples "
        f"outside of prior range"
    )
    samples = samples[net_mask]
    return samples


def run_amplfi(
    amplfi_strain,
    amplfi_psd_strain,
    samples_per_event: int,
    spectral_density: "SpectralDensity",
    amplfi_whitener: "Whiten",
    amplfi: "FlowArchitecture",
    std_scaler: "ChannelWiseScaler",
    device: str | torch.device,
):
    # get pe data from the buffer and whiten it
    amplfi_psd_strain = amplfi_psd_strain.to(device)
    amplfi_strain = amplfi_strain.to(device)[None]
    psd = spectral_density(amplfi_psd_strain)[None]
    whitened = amplfi_whitener(amplfi_strain, psd)

    # construct and bandpass asd
    freqs = torch.fft.rfftfreq(
        whitened.shape[-1], d=1 / amplfi_whitener.sample_rate
    )
    num_freqs = len(freqs)
    psd = torch.nn.functional.interpolate(
        psd, size=(num_freqs,), mode="linear"
    )

    mask = freqs > amplfi_whitener.highpass
    if amplfi_whitener.lowpass is not None:
        mask *= freqs < amplfi_whitener.lowpass

    psd = psd[:, :, mask]
    asds = torch.sqrt(psd)

    # sample from the model and descale back to physical units
    logging.debug("Starting sampling")
    samples = amplfi.sample(samples_per_event, context=(whitened, asds))  # ty: ignore[invalid-argument-type]
    samples = samples.squeeze(1)
    logging.debug("Descaling samples")
    samples = samples.transpose(1, 0)
    descaled_samples = std_scaler(samples, reverse=True)
    descaled_samples = descaled_samples.transpose(1, 0)
    logging.debug("Finished AMPLFI")

    return descaled_samples


def postprocess_samples(
    samples: torch.Tensor,
    event_time: float,
    inference_params: list[str],
    parameter_sampler: "AmplfiPrior",
) -> AmplfiResult:
    """
    Process samples into a bilby Result object
    that can be used for all downstream tasks
    """
    samples = filter_samples(samples, parameter_sampler, inference_params)

    phi_idx = inference_params.index("phi")
    dec_idx = inference_params.index("dec")
    ra = torch.remainder(
        lal.GreenwichMeanSiderealTime(event_time) + samples[..., phi_idx],  # ty: ignore[unresolved-attribute]
        torch.as_tensor(2 * torch.pi),
    )
    dec = samples[..., dec_idx]

    # build bilby posterior object for
    # parameters we want to keep
    posterior_params = [
        "chirp_mass",
        "mass_ratio",
        "distance",
        "inclination",
    ]

    posterior = {}
    for param in posterior_params:
        idx = inference_params.index(param)
        posterior[param] = samples.T[idx].flatten()

    # add source frame chirp mass information
    z_vals = get_redshifts(posterior["distance"].numpy())
    posterior["chirp_mass_source"] = posterior["chirp_mass"] / (1 + z_vals)

    posterior["ra"] = ra
    posterior["dec"] = dec
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        posterior["chirp_mass"], posterior["mass_ratio"]
    )
    posterior["mass_1"] = mass_1
    posterior["mass_2"] = mass_2
    posterior = pd.DataFrame(posterior)

    result = AmplfiResult(
        label=f"{event_time}",
        posterior=posterior,
        search_parameter_keys=inference_params,
    )
    return result
