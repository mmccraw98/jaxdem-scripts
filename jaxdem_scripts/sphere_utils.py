import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

def create_spheres(N, dim, phi, mass, dt, e_int, force_model='spring'):
    """
    Create a bidisperse system of spheres in any dimension
    Accomodates spring and wca potentials
    """
    seed = np.random.randint(0, 1e9)
    rad = jd.utils.dispersity.get_polydisperse_radii(N)
    pos, box_size = random_sphere_configuration(rad, phi, dim)
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(pos.shape[0]) * mass,
    )
    if force_model == 'spring':
        mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    elif force_model == 'wca_shifted':
        mean_sigma = 2.0 * jnp.mean(rad)
        eps_wca = e_int * mean_sigma**2 / 456.0
        mats = [jd.Material.create("lj", epsilon=eps_wca, density=1.0)]
    else:
        raise ValueError(f'force_model {force_model} unknown')
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type=force_model,
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=2.0 * jnp.max(state.rad),
            skin=0.05,
            safety_factor=2.0,
        ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system