import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd
from typing import Any, Optional

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration


def _build_mat_table(
    *,
    force_model: str,
    e_int: float,
    rad,
    mat_table=None,
    material_type: Optional[str] = None,
    material_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: Optional[dict[str, Any]] = None,
):
    if mat_table is not None:
        return mat_table

    if material_type is None:
        if force_model == 'spring':
            material_type = "elastic"
            material_kwargs = (
                dict(young=e_int, poisson=0.5, density=1.0)
                if material_kwargs is None
                else material_kwargs
            )
        elif force_model == 'wca_shifted':
            material_type = "lj"
            if material_kwargs is None:
                mean_sigma = 2.0 * jnp.mean(rad)
                eps_wca = e_int * mean_sigma**2 / 456.0
                material_kwargs = dict(epsilon=eps_wca, density=1.0)
        else:
            raise ValueError(f'force_model {force_model} unknown')

    material = jd.Material.create(material_type, **dict(material_kwargs or {}))
    matcher = jd.MaterialMatchmaker.create(matcher_type, **dict(matcher_kwargs or {}))
    return jd.MaterialTable.from_materials([material], matcher=matcher)

def create_spheres(
    N,
    dim,
    phi,
    mass,
    dt,
    e_int,
    force_model='spring',
    mat_table=None,
    material_type: Optional[str] = None,
    material_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Create a bidisperse system of spheres in any dimension
    Accomodates spring and wca potentials
    """
    rad = jd.utils.dispersity.get_polydisperse_radii(N)
    pos, box_size = random_sphere_configuration(rad, phi, dim)
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(pos.shape[0]) * mass,
    )
    mat_table = _build_mat_table(
        force_model=force_model,
        e_int=e_int,
        rad=rad,
        mat_table=mat_table,
        material_type=material_type,
        material_kwargs=material_kwargs,
        matcher_type=matcher_type,
        matcher_kwargs=matcher_kwargs,
    )
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