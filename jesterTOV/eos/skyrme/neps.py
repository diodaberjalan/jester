"""Helpers for deriving nuclear empirical parameters from Skyrme inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jesterTOV import utils

SKYRME_INPUT_KEYS: tuple[str, ...] = (
    "t2",
    "t4",
    "x0",
    "x1",
    "x4",
    "alph",
    "beta",
    "gamma",
    "kfsat",
    "av",
    "J",
    "meffs",
    "meffv",
    "Kinf",
    "eNMhd",
)

SKYRME_RESULT_PRIMARY_KEYS: tuple[str, ...] = (
    "E_sat",
    "P_sat",
    "K_sat",
    "Q_sat",
    "E_sym",
    "L_sym",
    "K_sym",
    "Q_sym",
)

SKYRME_RESULT_ALIAS_KEYS: tuple[str, ...] = (
    "esm0",
    "psm0",
    "Ksat0",
    "Qsat0",
    "J0",
    "L0",
    "lsymt0",
    "ksymt20",
    "Ksym20",
    "Qsym20",
)

SKYRME_RESULT_NEP_KEYS: tuple[str, ...] = (
    *SKYRME_RESULT_PRIMARY_KEYS,
    *SKYRME_RESULT_ALIAS_KEYS,
)


def _nsat_from_kfsat(kfsat: jax.Array) -> jax.Array:
    return 2.0 * kfsat**3 / (3.0 * jnp.pi**2)


def _solve_skyrme_system(
    params: Mapping[str, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Solve for Skyrme force parameters from INM inputs.

    This mirrors :meth:`Skyrme_EOS_model.solve_skyrme_system` but stays pure so it
    can be reused by HDF5 backfill code without constructing a full EOS object.
    """

    hbm = 20.73553000
    hmt = 2.0 / (1.0 / hbm + 1.0 / hbm)

    kf = params["kfsat"]
    rhosat = _nsat_from_kfsat(kf)
    rho_hd = jnp.array(1.0)
    kfn_hd = (3.0 * jnp.pi**2) ** (1.0 / 3.0)

    r3 = rhosat ** (params["alph"] + 1.0)
    r4 = rhosat ** (params["beta"] + 1.0)
    r5 = rhosat ** (params["gamma"] + 1.0)

    r_hd3 = rho_hd ** (params["alph"] + 1.0)
    r_hd4 = rho_hd ** (params["beta"] + 1.0)
    r_hd5 = rho_hd ** (params["gamma"] + 1.0)

    a1 = jnp.array(
        [
            [
                3.0 * rhosat / 8.0,
                3.0 * rhosat * kf**2 / 80.0,
                r3 / 16.0,
                3.0 * r5 * kf**2 / 80.0,
            ],
            [
                9.0 * rhosat / 8.0,
                3.0 * rhosat * kf**2 / 16.0,
                3.0 * (params["alph"] + 1.0) * r3 / 16.0,
                3.0 * kf**2 * r5 * (5.0 + 3.0 * params["gamma"]) / 80.0,
            ],
            [0.0, rhosat / (16.0 * hmt), 0.0, r5 / (16.0 * hmt)],
            [
                9.0 * rhosat / 4.0,
                3.0 * rhosat * kf**2 / 4.0,
                3.0 * (params["alph"] + 1.0) * (3.0 * params["alph"] + 2.0) * r3 / 16.0,
                3.0
                * (3.0 * params["gamma"] + 5.0)
                * (3.0 * params["gamma"] + 4.0)
                * r5
                * kf**2
                / 80.0,
            ],
        ]
    )
    b1 = jnp.array(
        [
            params["av"] - 0.6 * hmt * kf**2 - 9.0 * params["t4"] * kf**2 * r4 / 80.0,
            -1.2 * hmt * kf**2
            - 9.0 * params["t4"] * kf**2 * r4 * (5.0 + 3.0 * params["beta"]) / 80.0,
            1.0 / params["meffs"] - 1.0 - 3.0 * params["t4"] * r4 / (16.0 * hmt),
            params["Kinf"]
            - 6.0 * hmt * kf**2 / 5.0
            - 9.0
            * params["t4"]
            * kf**2
            * r4
            * (3.0 * params["beta"] + 5.0)
            * (3.0 * params["beta"] + 4.0)
            / 80.0,
        ]
    )
    res1 = jax.scipy.linalg.solve(a1, b1)
    t0_sol, term_t1_x2, t3_sol, term_t5_x5 = res1

    a2 = jnp.array(
        [
            [
                -(params["x1"] + 1.25) * rhosat * kf**2 / 8.0,
                -t3_sol * r3 / 24.0,
                -3.0 * kf**2 * r5 / 32.0,
            ],
            [
                rhosat * (params["x1"] + 1.25) / (8.0 * hmt),
                0.0,
                3.0 * r5 / (32.0 * hmt),
            ],
            [
                -3.0 * (params["x1"] + 1.25) * rho_hd * kfn_hd**2 / 40.0,
                -t3_sol * r_hd3 / 24.0,
                -9.0 * r_hd5 * kfn_hd**2 / 160.0,
            ],
        ]
    )
    b2 = jnp.array(
        [
            params["J"]
            - hmt * kf**2 / 3.0
            + t0_sol * rhosat * (2.0 * params["x0"] + 1.0) / 8.0
            + params["t4"] * kf**2 * r4 * params["x4"] / 8.0
            - 5.0 * term_t5_x5 * r5 * kf**2 / 96.0
            - (5.0 * term_t1_x2 - 9.0 * params["t2"]) * rhosat * kf**2 / 96.0
            + t3_sol * r3 / 48.0,
            1.0 / params["meffv"]
            - 1.0
            - (
                term_t5_x5 * r5 / 4.0
                + params["t4"] * (2.0 + params["x4"]) * r4
                + rhosat * (3.0 * params["t2"] + term_t1_x2) / 4.0
            )
            / (8.0 * hmt),
            params["eNMhd"]
            - 3.0 * hmt * kfn_hd**2 / 5.0
            - 3.0 * params["t4"] * (1.0 - params["x4"]) * r_hd4 * kfn_hd**2 / 40.0
            - 0.25 * t0_sol * (1.0 - params["x0"]) * rho_hd
            - t3_sol * r_hd3 / 24.0
            - 9.0
            * kfn_hd**2
            * (term_t5_x5 * r_hd5 + rho_hd * (term_t1_x2 - params["t2"]))
            / 160.0,
        ]
    )
    res2 = jax.scipy.linalg.solve(a2, b2)
    t1_sol, x3_sol, t5_sol = res2

    x2_sol = (term_t1_x2 / params["t2"] - 3.0 * t1_sol / params["t2"] - 5.0) / 4.0
    x5_sol = (term_t5_x5 / t5_sol - 5.0) / 4.0

    t_final = jnp.array([t0_sol, t1_sol, params["t2"], t3_sol, params["t4"], t5_sol])
    x_final = jnp.array(
        [params["x0"], params["x1"], x2_sol, x3_sol, params["x4"], x5_sol]
    )
    return t_final, x_final


def _energy_density_from_force(
    t: jax.Array,
    x: jax.Array,
    alph: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    ron: jax.Array,
    rop: jax.Array,
) -> jax.Array:
    """Pure Skyrme energy-density functional matching ``Skyrme_EOS_model.eDenSky``."""

    t0, t1, t2, t3, t4, t5 = t
    x0, x1, x2, x3, x4, x5 = x
    t2p = 1.0

    ro = ron + rop
    tau_factor = 0.6 * (3.0 * jnp.pi**2) ** (2.0 / 3.0)
    taun = tau_factor * ron ** (5.0 / 3.0)
    taup = tau_factor * rop ** (5.0 / 3.0)

    e_kin = 0.5 * utils.hbarc**2 * (taun / utils.m_n + taup / utils.m_p)

    ro2 = ro**2
    r_sq_sum = rop**2 + ron**2
    tausum = taun + taup
    tau_mix = rop * taup + ron * taun

    e0 = 0.25 * t0 * ((2.0 + x0) * ro2 - (2.0 * x0 + 1.0) * r_sq_sum)
    e3 = t3 * ro**alph * ((2.0 + x3) * ro2 - (2.0 * x3 + 1.0) * r_sq_sum) / 24.0

    c_eff_1 = t1 * (2.0 + x1) + t2 * (2.0 * t2p + x2)
    c_eff_2 = t2 * (2.0 * x2 + t2p) - t1 * (2.0 * x1 + 1.0)
    e_eff = 0.125 * (c_eff_1 * ro * tausum + c_eff_2 * tau_mix)

    c_4_1 = 1.0 + 0.5 * x4
    c_4_2 = 0.5 + x4
    e4 = 0.25 * t4 * ro**beta * (c_4_1 * ro * tausum - c_4_2 * tau_mix)

    c_5_1 = 1.0 + 0.5 * x5
    c_5_2 = 0.5 + x5
    e5 = 0.25 * t5 * ro**gamma * (c_5_1 * ro * tausum + c_5_2 * tau_mix)

    return e_kin + e0 + e3 + e_eff + e4 + e5


def _energy_per_particle(
    n_baryon: jax.Array,
    delta: jax.Array,
    t: jax.Array,
    x: jax.Array,
    alph: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
) -> jax.Array:
    ron = n_baryon * (1.0 + delta) / 2.0
    rop = n_baryon * (1.0 - delta) / 2.0
    return _energy_density_from_force(t, x, alph, beta, gamma, ron, rop) / n_baryon


def _compute_primary_neps_from_force(
    t: jax.Array,
    x: jax.Array,
    nsat: jax.Array,
    alph: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
) -> dict[str, jax.Array]:
    def e_sm(xvar: jax.Array) -> jax.Array:
        density = nsat * (1.0 + 3.0 * xvar)
        return _energy_per_particle(density, 0.0, t, x, alph, beta, gamma)

    def symmetry_energy(xvar: jax.Array) -> jax.Array:
        density = nsat * (1.0 + 3.0 * xvar)

        def symmetric_expansion(delta: jax.Array) -> jax.Array:
            return _energy_per_particle(density, delta, t, x, alph, beta, gamma)

        return 0.5 * jax.grad(jax.grad(symmetric_expansion))(0.0)

    e_sat = e_sm(0.0)
    p_sat = jax.grad(e_sm)(0.0)
    k_sat = jax.grad(jax.grad(e_sm))(0.0)
    q_sat = jax.grad(jax.grad(jax.grad(e_sm)))(0.0)

    e_sym = symmetry_energy(0.0)
    l_sym = jax.grad(symmetry_energy)(0.0)
    k_sym = jax.grad(jax.grad(symmetry_energy))(0.0)
    q_sym = jax.grad(jax.grad(jax.grad(symmetry_energy)))(0.0)

    return {
        "E_sat": e_sat,
        "P_sat": p_sat,
        "K_sat": k_sat,
        "Q_sat": q_sat,
        "E_sym": e_sym,
        "L_sym": l_sym,
        "K_sym": k_sym,
        "Q_sym": q_sym,
    }


def _compute_legacy_aliases(primary: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "esm0": primary["E_sat"],
        "psm0": primary["P_sat"],
        "Ksat0": primary["K_sat"],
        "Qsat0": primary["Q_sat"],
        "J0": primary["E_sym"],
        "L0": primary["L_sym"],
        "lsymt0": primary["L_sym"],
        "ksymt20": primary["K_sym"],
        "Ksym20": primary["K_sym"],
        "Qsym20": primary["Q_sym"],
    }


def _with_aliases(
    primary: Mapping[str, jax.Array],
    legacy: Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
    derived = dict(primary)
    derived.update(legacy)
    return derived


def compute_skyrme_neps_from_force(
    t: jax.Array | np.ndarray,
    x: jax.Array | np.ndarray,
    *,
    nsat: float,
    alph: float,
    beta: float,
    gamma: float,
) -> dict[str, float]:
    """Compute Skyrme-derived NEPs from direct force parameters."""

    primary = _compute_primary_neps_from_force(
        jnp.asarray(t),
        jnp.asarray(x),
        jnp.asarray(nsat),
        jnp.asarray(alph),
        jnp.asarray(beta),
        jnp.asarray(gamma),
    )
    legacy = _compute_legacy_aliases(primary)
    return {key: float(value) for key, value in _with_aliases(primary, legacy).items()}


def compute_skyrme_neps_from_params(
    params: Mapping[str, Any],
    *,
    nsat: float | None = None,
) -> dict[str, float]:
    """Compute Skyrme-derived NEPs from Jester's Skyrme input parametrization."""

    scalar_params = {key: jnp.asarray(params[key]) for key in SKYRME_INPUT_KEYS}
    t, x = _solve_skyrme_system(scalar_params)
    sat_density = (
        jnp.asarray(nsat)
        if nsat is not None
        else _nsat_from_kfsat(scalar_params["kfsat"])
    )
    primary = _compute_primary_neps_from_force(
        t,
        x,
        sat_density,
        scalar_params["alph"],
        scalar_params["beta"],
        scalar_params["gamma"],
    )
    legacy = _compute_legacy_aliases(primary)
    return {key: float(value) for key, value in _with_aliases(primary, legacy).items()}


def has_skyrme_parameters(
    posterior: Mapping[str, Any],
    fixed_params: Mapping[str, Any] | None = None,
) -> bool:
    """Return ``True`` when a result carries a complete Skyrme parameter set."""

    available = set(posterior)
    if fixed_params:
        available.update(fixed_params)
    return all(key in available for key in SKYRME_INPUT_KEYS)


def _infer_sample_count(posterior: Mapping[str, Any]) -> int:
    for value in posterior.values():
        if isinstance(value, dict):
            continue
        arr = np.asarray(value)
        if arr.ndim > 0:
            return int(arr.shape[0])
    return 1


def _posterior_param_arrays(
    posterior: Mapping[str, Any],
    fixed_params: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    n_samples = _infer_sample_count(posterior)
    combined: dict[str, np.ndarray] = {}
    fixed_params = fixed_params or {}

    for key in SKYRME_INPUT_KEYS:
        if key in posterior:
            arr = np.asarray(posterior[key])
        elif key in fixed_params:
            arr = np.asarray(fixed_params[key])
        else:
            raise KeyError(f"Missing Skyrme input parameter {key}")

        if arr.ndim == 0:
            arr = np.full((n_samples,), float(arr))
        combined[key] = arr

    return combined


def compute_skyrme_neps(
    posterior: Mapping[str, Any],
    fixed_params: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized HDF5-oriented Skyrme NEP calculation."""

    params = _posterior_param_arrays(posterior, fixed_params)
    params_jax = {key: jnp.asarray(value) for key, value in params.items()}

    def per_sample(sample: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
        t, x = _solve_skyrme_system(sample)
        primary = _compute_primary_neps_from_force(
            t,
            x,
            _nsat_from_kfsat(sample["kfsat"]),
            sample["alph"],
            sample["beta"],
            sample["gamma"],
        )
        legacy = _compute_legacy_aliases(primary)
        return _with_aliases(primary, legacy)

    computed = jax.vmap(per_sample)(params_jax)
    return {key: np.asarray(value) for key, value in computed.items()}
