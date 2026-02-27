import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd
import os

from dataclasses import dataclass

from correlations import get_pseudo_log_bins_from_steps, translational_correlations

@dataclass
class JobConfig:
    seed: int  # random seed
    can_rotate: bool = False  # whether or not the particles can rotate (NEEDED FOR CLUMPS)
    subtract_drift: bool = True  # whether or not to subtract drift from particle velocities
    delta_phi: float = 1e-2  # amount to compress each step
    target_temperature: float = 1e-5  # target temperature to maintain during compression
    n_steps: int = 10_000_000  # number of steps to run NVE dynamics for
    min_save_decade: int = 1_000  # minimum save frequency
    reset_save_decade: int = 100_000  # reset the logarithmic saving scheme each of these decades

def run_1(state, system, output_dir, config, save_strides = None, compress = True, save_fn = None, save_all = False):
    """
    Compress a system of deformable particles by a small increment while controlling the temperature with a rescaling thermostat.
    Allow the system to relax under the thermostat.
    Remove the thermostat and run NVE dynamics for 10x longer than the preliminary protocol.
    Save the data and calculate correlation functions.
    IMPORTANT: rotational correlations assume 2D dynamics only!
    IMPORTANT: translational correlations assume 1:1.4 bidispersity
    """
    if save_all and save_fn is not None:
        raise ValueError('Got incompatible arguments: save_fn passed but save_all is True')
    
    if save_strides is None:  # define default save strides
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
        temperature_target=config.target_temperature,  # maintain temperature
        packing_fraction_delta=config.delta_phi * (compress),  # compress
        can_rotate=config.can_rotate,
        subtract_drift=config.subtract_drift,
    )
    # maintain temperature
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=config.n_steps // 20,
        rescale_every=100,
        temperature_target=config.target_temperature,  # maintain temperature
        packing_fraction_delta=0.0,  # maintain density
        can_rotate=config.can_rotate,
        subtract_drift=config.subtract_drift,
    )
    print('Done')

    # create the directories
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(output_dir, f'phi-{phi:.6f}')
    
    # save initial data
    with jd.CheckpointWriter(directory=os.path.join(run_root, 'init')) as writer:
        writer.save(state, system)

    # define default save fn
    if not save_all and save_fn is None:
        def save_fn(st, sy):
            perm = jnp.empty_like(st.unique_id)
            perm = perm.at[st.unique_id].set(jnp.arange(st.unique_id.shape[0], dtype=st.unique_id.dtype))
            return dict(
                step_count=sy.step_count,
                pos=st.pos[perm],
                vel=st.vel[perm],
                pe=jd.utils.thermal.compute_potential_energy(st, sy),
                ke=jd.utils.thermal.compute_translational_kinetic_energy(st),
            )

    # run dynamics
    print('Running NVE...')
    rollout_kwargs = dict(strides=save_strides)
    if save_fn is not None:
        rollout_kwargs['save_fn'] = save_fn

    state, system, logged = jd.System.trajectory_rollout(
        state, system, **rollout_kwargs,
    )
    print("Done")

    # save the trajectory data and extract COM data for corrs, if needed
    if save_all:
        traj_state, traj_system = logged

        # sort the trajectory data before extracting
        def sort_frame(uid, arr):
            perm = jnp.empty_like(uid)
            perm = perm.at[uid].set(jnp.arange(uid.shape[0], dtype=uid.dtype))
            return arr[perm]

        data = dict(
            step_count=traj_system.step_count,
            pos=jax.vmap(sort_frame)(traj_state.unique_id, traj_state.pos),
            vel=jax.vmap(sort_frame)(traj_state.unique_id, traj_state.vel),
            pe=jax.vmap(jd.utils.thermal.compute_potential_energy)(traj_state, traj_system),
            ke=jax.vmap(jd.utils.thermal.compute_translational_kinetic_energy)(traj_state),
        )

        jd.utils.h5.save(traj_state, os.path.join(run_root, 'traj_state.h5'))
        jd.utils.h5.save(traj_system, os.path.join(run_root, 'traj_system.h5'))
    else:
        data = logged  # already a dict from save_fn
    
    # Reconstruct COM positions and velocities from per-frame component trajectories
    bond_id_sorted = state.bond_id[jnp.argsort(state.unique_id)]  # sort bond_id since we sort pos and vel
    N_dps = int(jnp.max(bond_id_sorted)) + 1
    data['pos_dp'] = jax.vmap(lambda p: compute_com(p, bond_id_sorted, N_dps))(data['pos'])
    data['vel_dp'] = jax.vmap(lambda v: compute_com(v, bond_id_sorted, N_dps))(data['vel'])
        
    np.savez(
        os.path.join(run_root, 'traj.npz'),
        **data,
    )

    # calculate the correlation functions for each size
    bins, t = get_pseudo_log_bins_from_steps(data['step_count'], system.dt)
    corrs = {'t': t}
    _, nv = jnp.unique(bond_id_sorted, return_counts=True)  # vertices per DP
    for _nv, name, diam in zip([min(nv), max(nv)], ['small', 'large'], [1.0, 1.4]):
        mask = nv == _nv
        vertex_mask = mask[bond_id_sorted]
        corrs.update(translational_correlations(data['pos_dp'][:, mask], diam, bins, name_suffix=f'_{name}'))
        corrs.update(translational_correlations(data['pos'][:, vertex_mask], diam, bins, name_suffix=f'_vertex_{name}'))
    np.savez(
        os.path.join(run_root, 'corrs.npz'),
        **corrs,
    )

    # save the final data
    with jd.CheckpointWriter(directory=os.path.join(run_root, 'final')) as writer:
        writer.save(state, system)
    return state, system, jnp.mean(data['pe']), run_root

def compute_com(arr, bond_id, N_dps):
    """Compute center of mass from component array (positions or velocities)."""
    total = jax.ops.segment_sum(arr, bond_id, num_segments=N_dps)
    counts = jax.ops.segment_sum(
        jnp.ones(arr.shape[0], dtype=arr.dtype),
        bond_id,
        num_segments=N_dps,
    )
    return total / jnp.maximum(counts[:, None], 1.0)