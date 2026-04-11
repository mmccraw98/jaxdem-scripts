import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import minimize_scalar, brentq
import jax.numpy as jnp
import jaxdem as jd
from typing import Any, Optional

from jaxdem.utils import generate_ga_clump_state, generate_ga_deformable_state


def _build_mat_table(
    *,
    mat_table=None,
    material_type: str = "elastic",
    material_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: Optional[dict[str, Any]] = None,
):
    if mat_table is not None:
        return mat_table

    material = jd.Material.create(material_type, **dict(material_kwargs or {}))
    matcher = jd.MaterialMatchmaker.create(matcher_type, **dict(matcher_kwargs or {}))
    return jd.MaterialTable.from_materials([material], matcher=matcher)


def _normalize_bonded_force_model_type(
    bonded_force_model_type: Optional[str],
    tau_s,
) -> str:
    if bonded_force_model_type is None:
        return (
            "PlasticDeformableParticleModel"
            if tau_s is not None
            else "DeformableParticleModel"
        )

    aliases = {
        "elastic": "DeformableParticleModel",
        "deformable": "DeformableParticleModel",
        "plastic": "PlasticDeformableParticleModel",
    }
    return aliases.get(bonded_force_model_type, bonded_force_model_type)


def _coerce_ga_dp_container(
    state,
    container,
    *,
    bonded_force_model_type: Optional[str] = None,
    bonded_force_model_kwargs: Optional[dict[str, Any]] = None,
    tau_s=None,
):
    bonded_force_model_type = _normalize_bonded_force_model_type(
        bonded_force_model_type, tau_s
    )
    bonded_force_model_kwargs = dict(bonded_force_model_kwargs or {})
    tau_s_kw = bonded_force_model_kwargs.pop("tau_s", tau_s)

    if bonded_force_model_type == "DeformableParticleModel":
        if tau_s_kw is not None:
            raise ValueError(
                "Got `tau_s`, but `bonded_force_model_type` resolves to "
                "`DeformableParticleModel`. Use the plastic model instead."
            )
        if bonded_force_model_kwargs:
            unexpected = ", ".join(sorted(bonded_force_model_kwargs))
            raise ValueError(
                f"Unsupported bonded_force_model_kwargs for elastic GA DPs: {unexpected}"
            )
        return container

    if bonded_force_model_type != "PlasticDeformableParticleModel":
        raise ValueError(
            "Unsupported bonded_force_model_type "
            f"`{bonded_force_model_type}`. Expected one of "
            "`DeformableParticleModel`, `PlasticDeformableParticleModel`, "
            "`elastic`, `deformable`, or `plastic`."
        )

    if tau_s_kw is None:
        raise ValueError(
            "`PlasticDeformableParticleModel` requires `tau_s` either directly "
            "or via bonded_force_model_kwargs."
        )
    if bonded_force_model_kwargs:
        unexpected = ", ".join(sorted(bonded_force_model_kwargs))
        raise ValueError(
            f"Unsupported bonded_force_model_kwargs for plastic GA DPs: {unexpected}"
        )

    return jd.BondedForceModel.create(
        "PlasticDeformableParticleModel",
        vertices=state.pos,
        elements=container.elements,
        edges=container.edges,
        element_adjacency=container.element_adjacency,
        element_adjacency_edges=container.element_adjacency_edges,
        elements_id=container.elements_id,
        initial_body_contents=container.initial_body_contents,
        initial_element_measures=container.initial_element_measures,
        initial_edge_lengths=container.initial_edge_lengths,
        initial_bendings=container.initial_bendings,
        em=container.em,
        ec=container.ec,
        eb=container.eb,
        el=container.el,
        gamma=container.gamma,
        tau_s=tau_s_kw,
        w_b=container.w_b,
    )

def create_bidisperse_ga_clumps_2d(
    N_clumps,
    mu_eff,
    min_nv,
    phi,
    aspect_ratio,
    clump_mass,
    dt,
    e_int,
    body_type='solid',
    size_ratios=(1.0, 1.4),
    count_ratios=(0.5, 0.5),
    mat_table=None,
    material_type: str = "elastic",
    material_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Create a bidisperse system of 2D GA clump particles given a desired friction coefficient
    and number of vertices in the small particles.
    Accomodates various aspect ratios.
    """
    dim = 2
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N_clumps, size_ratios=size_ratios, count_ratios=count_ratios)
    asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(mu_eff, min(particle_radii), min_nv)
    max_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(mu_eff, asperity_radius, max(particle_radii))
    vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv
    state, box_size = generate_ga_clump_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        body_type=body_type,
        aspect_ratio=aspect_ratio,
        use_uniform_mesh=True,
        mass=clump_mass,
        seed=np.random.randint(0, 1e9),
    )
    mat_table = _build_mat_table(
        mat_table=mat_table,
        material_type=material_type,
        material_kwargs=(
            dict(young=e_int, poisson=0.5, density=1.0)
            if material_kwargs is None
            else material_kwargs
        ),
        matcher_type=matcher_type,
        matcher_kwargs=matcher_kwargs,
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
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
            box_size=box_size,
        ),
    )
    return state, system

def create_bidisperse_ga_dps_2d(
    N_dps,
    mu_eff,
    min_nv,
    phi,
    aspect_ratio,
    dp_mass,
    dt,
    e_int,
    e_m,
    e_c,
    e_b,
    e_l,
    e_gamma,
    tau_s=None,
    size_ratios=(1.0, 1.4),
    count_ratios=(0.5, 0.5),
    mat_table=None,
    material_type: str = "elastic",
    material_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: Optional[dict[str, Any]] = None,
    bonded_force_model_type: Optional[str] = None,
    bonded_force_model_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Create a bidisperse system of 2D GA DP particles given a desired friction coefficient
    and number of vertices in the small particles.
    Accomodates various aspect ratios.
    e_m: measure elasticity (triangle area in 3d, edge length in 2d)
    e_c: content elasticity (enclosed volume in 3d, enclosed area in 2d)
    e_b: bending elasticity
    e_l: length elasticity (not normalized)
    e_gamma: surface/line tension
    tau_s: perimeter relaxation timescale
    """
    dim = 2
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N_dps, size_ratios=size_ratios, count_ratios=count_ratios)
    asperity_radius = get_closest_vertex_radius_for_mu_eff_2d(mu_eff, min(particle_radii), min_nv)
    max_nv, _, _ = find_num_vertices_for_target_mu_eff_2d(mu_eff, asperity_radius, max(particle_radii))
    vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv
    state, dp_container, box_size = generate_ga_deformable_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        seed=np.random.randint(0, 1e9),
        use_uniform_mesh=True,
        mass=dp_mass,
        aspect_ratio=aspect_ratio,
        mesh_type="ico",
        em=e_m,
        ec=e_c,
        eb=e_b,
        el=e_l,
        gamma=e_gamma,
        random_orientations=True,
    )
    dp_container = _coerce_ga_dp_container(
        state,
        dp_container,
        bonded_force_model_type=bonded_force_model_type,
        bonded_force_model_kwargs=bonded_force_model_kwargs,
        tau_s=tau_s,
    )
    mat_table = _build_mat_table(
        mat_table=mat_table,
        material_type=material_type,
        material_kwargs=(
            dict(young=e_int, poisson=0.5, density=1.0)
            if material_kwargs is None
            else material_kwargs
        ),
        matcher_type=matcher_type,
        matcher_kwargs=matcher_kwargs,
    )
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
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
            box_size=box_size,
        ),
        bonded_force_model=dp_container,
    )
    return state, system

def calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices):
    """
    Calculate the effective friction coefficient in 2D.
    """
    return 1 / np.sqrt(((2 * vertex_radius) / ((outer_radius - vertex_radius) * np.sin(np.pi / num_vertices))) ** 2 - 1)

def find_num_vertices_for_target_mu_eff_2d(
    target_mu_eff: float,
    vertex_radius: float,
    outer_radius: float,
    num_vertices_min: int = 3,
    num_vertices_max: int = 100):
    """
    Solve for the number of vertices that gives a desired effective friction coefficient in 2D.
    """
    best_nv = None
    best_mu = np.nan
    best_err = np.inf
    for nv in range(int(num_vertices_min), int(num_vertices_max) + 1):
        try:
            mu = float(calc_mu_eff_2d(vertex_radius, outer_radius, nv))
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError, TypeError):
            continue
        if not np.isfinite(mu):
            continue
        err = abs(mu - target_mu_eff)
        if err < best_err:
            best_nv, best_mu, best_err = nv, mu, err
    return best_nv, best_mu, best_err

def get_closest_vertex_radius_for_mu_eff_2d(mu_eff, outer_radius, num_vertices):
    """
    Solve for the vertex diameter that gives a desired effective friction coefficient in 2D.
    """
    # Calculate mathematically valid bounds
    sin_term = np.sin(np.pi / num_vertices)
    min_vertex_radius = outer_radius * sin_term / (2 + sin_term) + 1e-12
    max_vertex_radius = outer_radius - 1e-12
    
    # Check if target mu_eff is achievable
    max_mu_eff = calc_mu_eff_2d(min_vertex_radius, outer_radius, num_vertices)
    min_mu_eff = calc_mu_eff_2d(max_vertex_radius, outer_radius, num_vertices)
    
    if mu_eff > max_mu_eff or mu_eff < min_mu_eff:
        # Target mu_eff is outside achievable range
        return np.nan
    try:
        # Use root finding since we want calc_mu_eff(vertex_radius) = mu_eff
        def objective(vertex_radius):
            return calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices) - mu_eff
        
        # Brent's method is robust for this monotonic function
        result = brentq(objective, min_vertex_radius, max_vertex_radius, xtol=1e-12)
        return result
        
    except (ValueError, RuntimeError, ZeroDivisionError):
        # Fallback to bounded scalar minimization if root finding fails
        def obj_squared(vertex_radius):
            try:
                return (calc_mu_eff_2d(vertex_radius, outer_radius, num_vertices) - mu_eff) ** 2
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.inf
        
        result = minimize_scalar(obj_squared, bounds=(min_vertex_radius, max_vertex_radius), method='bounded')
        return result.x if result.success else np.nan
