"""
Various functions for calculating the change in specific heat around jamming.
The core idea is to start with a jammed configuration, decrease the density slightly below jamming,
initialize several copies of the system at various, closely spaced temperatures, and run NVE dynamics.
The slope of the total energy vs temperature across all temperature initializations is used to directly
measure the specific heat.
The protocol is then repeated across various offsets from jamming.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class JobConfig:
    seed: int  # random seed
    n_steps: int = 100_000  # run dynamics for this long
    save_stride: int = 100  # save the configurations every save_stride steps
    dt: float = 1e-2  # timestep
    e_int: float = 1.0  # interaction scale
    can_rotate: bool = True  # whether or not the particles can rotate (NEEDED FOR CLUMPS)
    subtract_drift: bool = True  # whether or not to subtract drift from particle velocities
    delta_phi_min: float = 1e-4  # smallest density offset from jamming
    n_phi_steps: int = 20  # number of densities to measure
    temp_min: float = 1e-15  # minimum temperature
    temp_max: float = 2e-15  # maximum temperature
    n_temperature_steps: int = 20  # number of temperatures to simultaneously probe

def run_1(state, system, output_dir, config):
    """
    Start with initial jammed system, remove rattlers, and use protocol 1 for measuring the specific
    heat across various INDEPENDENT density trials.
    """
    state, system = system.collider.compute_force(state, system)  # force the neighbor list to  update

    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    temperatures = jnp.linspace(config.temp_min, config.temp_max, config.n_temperature_steps)
    delta_phis = - jnp.logspace(jnp.log10(config.delta_phi_min), jnp.log10(phi / 2), config.n_phi_steps)

    state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_clump_rattler_ids(state, system)
    base_state = jd.utils.contacts.remove_rattlers_from_state(state, rattler_ids)
    base_system = _copy_system(base_state, system, config)

    _run_for_densities_1(base_state, base_system, output_dir, config, temperatures, delta_phis)

def run_2(state, system, output_dir, config):
    """
    Start with initial jammed system, initialize at the minimum temperature, run NVE dynamics to
    relax initial configuration, then use protocol 1 for measuring the specific heat across
    various INDEPENDENT density trials.
    """
    state, system = system.collider.compute_force(state, system)  # force the neighbor list to  update

    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    temperatures = jnp.linspace(config.temp_min, config.temp_max, config.n_temperature_steps)
    delta_phis = - jnp.logspace(jnp.log10(config.delta_phi_min), jnp.log10(phi / 2), config.n_phi_steps)

    state = jd.utils.thermal.set_temperature(state, config.temp_min, config.can_rotate, config.subtract_drift, config.seed)
    base_state, system = system.step(state, system, n=1_000_000)
    base_system = _copy_system(base_state, system, config)

    _run_for_densities_1(base_state, base_system, output_dir, config, temperatures, delta_phis)

def _run_for_densities_1(base_state, base_system, output_dir, config, temperatures, delta_phis):
    """
    Take an input state and system (base).
    Create n_temperature_steps copies of the state and system, intialize each at the temperatures of interest.
    Scale all copies to a given packing fraction and run NVE dynamics in parallel.
    Repeat for n_phi_steps packing fractions, using the input state and system as the initial condition for each step.
    IMPORTANT: each step in density is independent.
    """
    N_clumps = int(jnp.max(base_state.clump_id) + 1)
    phi = jd.utils.packingUtils.compute_packing_fraction(base_state, base_system)

    def save_fn(st, sy):
        return (
            jd.utils.thermal.compute_potential_energy(st, sy),
            jd.utils.thermal.compute_translational_kinetic_energy(st),
            jd.utils.thermal.compute_rotational_kinetic_energy(st),
            jd.utils.thermal.compute_temperature(st, config.can_rotate, config.subtract_drift),
        )

    for delta_phi in tqdm(delta_phis):
        interm_state, interm_system = jd.utils.packingUtils.scale_to_packing_fraction(base_state, base_system, phi + delta_phi)
        state = jd.State.stack([interm_state for _ in range(temperatures.size)])
        state = jax.vmap(
            lambda st, temp: jd.utils.thermal.set_temperature(
                st,
                temp,
                config.can_rotate,
                config.subtract_drift,
                config.seed,
            )
        )(state, temperatures)
        system = jd.System.stack([interm_system for _ in range(temperatures.size)])

        state, system, logged = system.trajectory_rollout(
            state, system,
            n=config.n_steps // config.save_stride,
            stride=config.save_stride,
            save_fn=save_fn,
        )

        pe, ke, ke_r, temp = logged

        np.savez(
            os.path.join(output_dir, f'thermal_{delta_phi}.npz'),
            pe=pe,
            ke=ke,
            ke_r=ke_r,
            temp=temp,
            target_temp=temperatures,
            delta_phi=delta_phi,
            N=N_clumps,
        )

def _copy_system(state, system, config):
    """
    Create a new system object for a modified state, giving an existing system.
    """
    mats = [jd.Material.create("elastic", young=config.e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    return jd.System.create(
        state_shape=state.shape,
        dt=config.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=2.0 * jnp.max(state.rad),
            skin=0.05,
            safety_factor=5.0,
        ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=system.domain.box_size,
        ),
    )
