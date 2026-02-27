import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd
import os

from dataclasses import dataclass

from .correlations import get_pseudo_log_bins_from_steps, translational_correlations

@dataclass
class JobConfig:
    seed: int  # random seed
    can_rotate: bool = True  # whether or not the particles can rotate (NEEDED FOR CLUMPS)
    subtract_drift: bool = True  # whether or not to subtract drift from particle velocities
    delta_phi: float = 1e-2  # amount to compress each step
    target_temperature: float = 1e-5  # target temperature to maintain during compression
    n_steps: int = 10_000_000  # number of steps to run NVE dynamics for
    min_save_decade: int = 1_000  # minimum save frequency
    reset_save_decade: int = 100_000  # reset the logarithmic saving scheme each of these decades

def run_1(state, system, output_dir, config, save_strides = None, compress = True, save_fn = None, save_all = False):
    """
    Compress a system of spheres by a small increment while controlling the temperature with a rescaling thermostat.
    Allow the system to relax under the thermostat.
    Remove the thermostat and run NVE dynamics for 10x longer than the preliminary protocol.
    Save the data and calculate correlation functions.
    IMPORTANT: translational correlations assume 1:1.4 bidispersity
    """
    if save_all and save_fn is not None:
        raise ValueError('Got incompatible arguments: save_fn passed but save_all is True')

    if save_strides is None:
        save_steps = jnp.asarray(jd.utils.make_save_steps_pseudolog(
            num_steps=config.n_steps,
            reset_save_decade=config.reset_save_decade,
            min_save_decade=config.min_save_decade,
            decade=10,
            include_step0=True,
        ))
        save_strides = save_steps[1:] - save_steps[:-1]

    # compress and maintain temperature
    print('Running NVT...')
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=config.n_steps // 20,
        rescale_every=100,
        temperature_target=config.target_temperature,
        packing_fraction_delta=config.delta_phi * (compress),
        can_rotate=config.can_rotate,
        subtract_drift=config.subtract_drift,
    )
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=config.n_steps // 20,
        rescale_every=100,
        temperature_target=config.target_temperature,
        packing_fraction_delta=0.0,
        can_rotate=config.can_rotate,
        subtract_drift=config.subtract_drift,
    )
    print('Done')

    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(output_dir, f'phi-{phi:.6f}')

    with jd.CheckpointWriter(directory=os.path.join(run_root, 'init')) as writer:
        writer.save(state, system)

    if not save_all and save_fn is None:
        def save_fn(st, sy):
            perm = jnp.empty_like(st.unique_id)
            perm = perm.at[st.unique_id].set(jnp.arange(st.unique_id.shape[0], dtype=st.unique_id.dtype))
            return dict(
                step_count=sy.step_count,
                pos=st.pos_c[perm],
                vel=st.vel[perm],
                pe=jd.utils.thermal.compute_potential_energy(st, sy),
                ke=jd.utils.thermal.compute_translational_kinetic_energy(st),
            )

    print('Running NVE...')
    rollout_kwargs = dict(strides=save_strides)
    if save_fn is not None:
        rollout_kwargs['save_fn'] = save_fn

    state, system, logged = jd.System.trajectory_rollout(
        state, system, **rollout_kwargs,
    )
    print("Done")

    if save_all:
        traj_state, traj_system = logged

        def sort_frame(uid, pos, vel):
            perm = jnp.empty_like(uid)
            perm = perm.at[uid].set(jnp.arange(uid.shape[0], dtype=uid.dtype))
            return pos[perm], vel[perm]

        pos, vel = jax.vmap(sort_frame)(
            traj_state.unique_id, traj_state.pos_c, traj_state.vel,
        )

        data = dict(
            step_count=traj_system.step_count,
            pos=pos,
            vel=vel,
            pe=jax.vmap(jd.utils.thermal.compute_potential_energy)(traj_state, traj_system),
            ke=jax.vmap(jd.utils.thermal.compute_translational_kinetic_energy)(traj_state),
        )

        jd.utils.h5.save(traj_state, os.path.join(run_root, 'traj_state.h5'))
        jd.utils.h5.save(traj_system, os.path.join(run_root, 'traj_system.h5'))
    else:
        data = logged

    np.savez(os.path.join(run_root, 'traj.npz'), **data)

    # correlation functions
    bins, t = get_pseudo_log_bins_from_steps(data['step_count'], system.dt)
    corrs = {'t': t}
    rad_sorted = state.rad[jnp.argsort(state.unique_id)]
    unique_radii = jnp.unique(rad_sorted)
    for rad_val, name in zip([jnp.min(unique_radii), jnp.max(unique_radii)], ['small', 'large']):
        mask = rad_sorted == rad_val
        diam = float(2.0 * rad_val)
        corrs.update(translational_correlations(data['pos'][:, mask], diam, bins, name_suffix=f'_{name}'))

    np.savez(os.path.join(run_root, 'corrs.npz'), **corrs)

    with jd.CheckpointWriter(directory=os.path.join(run_root, 'final')) as writer:
        writer.save(state, system)
    return state, system, jnp.mean(data['pe']), run_root