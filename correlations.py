import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd

from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned
from jaxdem.analysis.kernels import isf_self_isotropic_kernel, msd_kernel, unwrap_angles_2d, msad_kernel_2d, isf_angular_kernel_2d

def get_pseudo_log_bins_from_steps(steps, dt):
    """
    Get pseudo-logarithmic time-bins from a list of save steps.
    Return the bins and the time axis.
    Used for 1-time correlation functions.
    """
    T = steps.shape[0]
    bins = LagBinsPseudoLog(
        T,
        dt_min=1,
        dt_max=int(steps[-1] - steps[0]),
        timestep=steps,
    )
    t = bins.values() * dt
    return bins, t

def translational_correlations(pos, diam, bins, name_suffix=''):
    """
    Calculate correlations for translation motion:
    MSD: < r(t+dt) \dot r(t) > (dt)
    ISF: < exp( i 2 \pi ( r(t+dt) - r(t) ) / diam ) > (dt)
    """
    corrs = {}

    msd_res = evaluate_binned(msd_kernel, {"pos": pos}, bins)
    corrs.update({f'msd{name_suffix}': np.asarray(msd_res.mean)})

    k = 2.0 * jnp.pi / diam  # wave vector for the particle diameter
    isf_res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos}, bins, kernel_kwargs={"k": k})
    corrs.update({f'isf{name_suffix}': np.asarray(isf_res.mean)})

    return corrs

def rotational_correlations_2d(q_w, q_xyz, n, bins, name_suffix=''):
    """
    Calculate correlations for rotational motion - ONLY WORKS IN 2D:
    MSAD: < \theta(t+dt) \theta(t) > (dt)
    AISF: < cos( n (\theta(t+dt) - \theta(t)) ) > (dt)
    n is the symmetry count for the rotations
    """
    corrs = {}

    theta = unwrap_angles_2d(q_w, q_xyz)
    theta_0 = 2 * np.pi / (2 * np.pi / n)  # angular period for the particle symmetry angle

    msad_res = evaluate_binned(msad_kernel_2d, {"theta": theta}, bins)
    corrs.update({f'msad{name_suffix}': np.asarray(msad_res.mean)})

    aisf_res = evaluate_binned(isf_angular_kernel_2d,{"theta": theta}, bins, kernel_kwargs={"theta_0": theta_0})
    corrs.update({f'aisf{name_suffix}': np.asarray(aisf_res.mean)})

    return corrs