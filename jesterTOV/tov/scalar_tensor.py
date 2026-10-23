"""
Scalar-Tensor TOV equation solver.

This module implements TOV equations for scalar-tensor theories of gravity,
where the gravitational interaction includes both a metric tensor and a scalar field.

**Units:** All calculations are performed in geometric units where G = c = 1.

**Reference:** G. Creci et al Phys.Rev.D 111 (2025) 8, 089901 (erratum)

# FIXME: Need to fully integrate the TOV solver: see docs/developer_guide/adding_new_tov.md
"""

import functools
from typing import Literal, cast
import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_batching import custom_vmap
from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri5,
    Tsit5,
    Bosh3,
    SaveAt,
    PIDController,
    Event,
    RecursiveCheckpointAdjoint,
)

from jesterTOV import utils
from jesterTOV.logging_config import get_logger
from jesterTOV.tov._modax.compat import _SurfaceIntegration, _is_differentiating

try:
    from jesterTOV.tov._modax import integrator as _modax_integrator
except ImportError:  # Optional GPU compiler dependencies are absent.
    _modax_integrator = None
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution
from jesterTOV.tov.scalar_tensor_utils import (
    _LocalRecoveryResult,
    _correct_background_local,
    _recover_background,
    _select_probe_pair as _select_probe_pair,
    build_exterior_basis,
    build_exterior_basis_autodiff,
    coeff_solver,
    compute_tidal_deformabilities,
    predict_phi0_last,
)

logger = get_logger("jester")


def _tov_ode_iter(h, y, eos):
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]

    # scalar-tensor parameters
    beta_ST = eos["beta_ST"]

    r, m, nu, psi, phi = y

    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # scalar coupling function
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    alpha_phi = beta_ST * phi

    # Regularization parameter
    EPS = 1e-25

    # Modified dpdr to avoid division by zero
    dpdr = -(e + p) * (
        (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m + EPS))  # Regularize denominator
        + 0.5 * r * jnp.power(psi, 2)
        + alpha_phi * psi
    )

    # Safe division for drdh (handles dpdr ≈ 0)
    safe_dpdr = jnp.where(
        jnp.abs(dpdr) < EPS,
        jnp.copysign(EPS, dpdr),
        dpdr,  # Preserve sign
    )
    drdh = (e + p) / safe_dpdr  # Numerically stable division

    # Remaining equations with regularized denominators
    dmdh = (
        4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 2) * e
        + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)
    ) * drdh

    dnudh = (
        2
        * (m + 4.0 * jnp.pi * jnp.power(A_phi, 4) * jnp.power(r, 3) * p)
        / (r * (r - 2.0 * m + EPS))  # Regularized
        + r * jnp.power(psi, 2)
    ) * drdh

    dpsidh = (
        (
            4.0
            * jnp.pi
            * jnp.power(A_phi, 4)
            * r
            / (r - 2.0 * m + EPS)  # Regularized
            * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        )
        - (2.0 * (r - m) / (r * (r - 2.0 * m + EPS)) * psi)  # Regularized
    ) * drdh

    dphidh = psi * drdh
    return drdh, dmdh, dnudh, dpsidh, dphidh


def _tov_ode_iter_tidal(h, y, eos):
    # EOS quantities
    ps = eos["p"]
    hs = eos["h"]
    es = eos["e"]
    dloge_dlogps = eos["dloge_dlogp"]
    beta_ST = eos["beta_ST"]  # scalar-tensor parameter

    r, m, nu, psi, phi, H0, H0_prime, delta_phi, delta_phi_prime = y
    EPS = 1e-25  # small value to avoid zero division error

    # Interpolate EOS
    e = utils.interp_in_logspace(h, hs, es)
    p = utils.interp_in_logspace(h, hs, ps)
    dedp = e / p * jnp.interp(h, hs, dloge_dlogps)

    # Scalar field terms
    A_phi = jnp.exp(0.5 * beta_ST * jnp.power(phi, 2))
    alpha_phi = beta_ST * phi
    A_phi4 = jnp.power(A_phi, 4)
    four_pi_Aphi4 = 4.0 * jnp.pi * A_phi4
    r2 = r * r
    r3 = r2 * r

    # Core equations
    denom_non_tidal = r - 2.0 * m + EPS
    dpdr = -(e + p) * (
        (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal)
        + 0.5 * r * jnp.power(psi, 2)
        + alpha_phi * psi
    )

    safe_dpdr = jnp.where(jnp.abs(dpdr) < EPS, jnp.copysign(EPS, dpdr), dpdr)
    drdh = (e + p) / safe_dpdr

    dmdh = (four_pi_Aphi4 * r2 * e + 0.5 * r * (r - 2.0 * m) * jnp.power(psi, 2)) * drdh

    dnudh = (
        2 * (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) + r * jnp.power(psi, 2)
    ) * drdh

    dpsidh = (
        four_pi_Aphi4
        * r
        * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        / denom_non_tidal
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh

    dphidh = psi * drdh

    # Tidal deformabilities (l=2)
    denom_pert = r - 2.0 * m + EPS

    F1 = (4.0 * jnp.pi * jnp.power(r, 3) * A_phi4 * (p - e) + 2.0 * (r - m)) / (
        r * denom_pert
    )

    F0_num = (
        4.0
        * jnp.pi
        * jnp.power(r, 3)
        * p
        * A_phi4
        * (r * (dedp + 9.0) - 2.0 * m * (dedp + 13.0))
        + 4.0 * jnp.pi * jnp.power(r, 3) * e * A_phi4 * (dedp + 5.0) * (r - 2.0 * m)
        - 4.0
        * jnp.power(r, 2)
        * (r - 2.0 * m)
        * jnp.power(psi, 2)
        * (4.0 * jnp.pi * jnp.power(r, 3) * p * A_phi4 + m)
        - 64.0
        * jnp.power(jnp.pi, 2)
        * jnp.power(r, 6)
        * jnp.power(p, 2)
        * jnp.power(A_phi4, 2)
        - 6.0 * r * (r - 2.0 * m)  # l(l+1) = 6 for l=2
        - jnp.power(r, 4) * jnp.power(r - 2.0 * m, 2) * jnp.power(psi, 4)
        - 4.0 * jnp.power(m, 2)
    )
    F0 = F0_num / (jnp.power(r, 2) * jnp.power(r - 2.0 * m, 2))

    Fs_num = (
        4.0
        * jnp.power(r, 2)
        * (
            2.0
            * jnp.pi
            * A_phi4
            * (-alpha_phi * ((dedp - 9.0) * p + (dedp - 1.0) * e) + 4.0 * r * p * psi)
            + (r - 2.0 * m) * jnp.power(psi, 3)
        )
        + 8.0 * m * psi
    )
    Fs = Fs_num / (r * (r - 2.0 * m))

    # Coefficients for dphi equation
    G1 = F1
    G0 = (
        4.0
        * jnp.pi
        * r
        * A_phi4
        / (r - 2.0 * m)
        * (
            jnp.power(alpha_phi, 2) * ((dedp + 9.0) * p + (dedp - 7.0) * e)
            + (e - 3.0 * p) * (-beta_ST)
        )
        - 6.0 / (r * (r - 2.0 * m))
        - 4.0 * jnp.power(psi, 2)
    )
    Gs = Fs / 4.0

    # Perturbation derivatives
    dH0dh = H0_prime * drdh
    dH0_primedh = (-F1 * H0_prime - F0 * H0 + Fs * delta_phi) * drdh
    ddelta_phidh = delta_phi_prime * drdh
    ddelta_phi_primedh = (-G1 * delta_phi_prime - G0 * delta_phi + Gs * H0) * drdh

    return (
        drdh,
        dmdh,
        dnudh,
        dpsidh,
        dphidh,
        dH0dh,
        dH0_primedh,
        ddelta_phidh,
        ddelta_phi_primedh,
    )


def _diffrax_surface(
    y0,
    h0,
    dh,
    eos_dict,
    accurate,
    mass_limit,
    kind: Literal["background", "tidal", "no_tidal"],
) -> _SurfaceIntegration:
    """Existing Diffrax integration settings, factored for backend dispatch."""
    if kind == "background":

        def mass_event(t, y, args, **kwargs):
            return y[1] > mass_limit

        solution = diffeqsolve(
            ODETerm(_tov_ode_iter),
            Bosh3(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos_dict,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(
                rtol=jnp.where(accurate, 1e-6, 1e-5),
                atol=jnp.where(accurate, 1e-7, 1e-6),
            ),
            event=Event(mass_event),
            adjoint=RecursiveCheckpointAdjoint(),
            max_steps=500,
            throw=False,
        )
    elif kind == "tidal":

        def mass_event(t, y, args, **kwargs):
            return y[1] > mass_limit

        solution = diffeqsolve(
            ODETerm(_tov_ode_iter_tidal),
            Dopri5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos_dict,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            event=Event(mass_event),
            max_steps=1000,
            throw=False,
        )
    else:
        solution = diffeqsolve(
            ODETerm(_tov_ode_iter),
            Tsit5(scan_kind="bounded"),
            t0=h0,
            t1=0,
            dt0=dh,
            y0=y0,
            args=eos_dict,
            saveat=SaveAt(t1=True),
            stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
            throw=False,
        )
    return _SurfaceIntegration(
        jnp.stack(tuple(value[-1] for value in solution.ys)),
        (solution.result._value == 0) & (solution.ts[-1] == 0),
    )


def _integrate_surface(y0, h0, dh, eos_dict, accurate, mass_limit, kind):
    """Use Diffrax for AD/missing compiler dependencies and Modax on CUDA."""
    if _modax_integrator is None or _is_differentiating(y0, h0, eos_dict):
        return _diffrax_surface(y0, h0, dh, eos_dict, accurate, mass_limit, kind)

    # This is resolved while tracing the solver. CPU-only processes retain
    # Diffrax and never lower the CUDA custom-call primitive.
    if jax.default_backend() != "gpu":
        return _diffrax_surface(y0, h0, dh, eos_dict, accurate, mass_limit, kind)
    return _modax_integrator.surface(y0, h0, dh, eos_dict, accurate, mass_limit, kind)


def _background_forward(params, accurate, h0, dh, r0, m0, eos_dict, phi_inf_target):
    """Shared background integration for fast shooting and queued recovery."""
    phi0_trial = params[0]
    M_limit = 7.0 * utils.solar_mass_in_meter
    is_nan = jnp.isnan(phi0_trial)
    m0_eff = jnp.where(is_nan, M_limit + 10.0, m0)
    t0_eff = jnp.where(is_nan, 0.0, h0)
    y0 = (r0, m0_eff, 0.0, 0.0, phi0_trial)

    sol_iter = _integrate_surface(
        y0, t0_eff, dh, eos_dict, accurate, M_limit, "background"
    )
    R = sol_iter.state[0]
    M_s = sol_iter.state[1]
    psi_s = sol_iter.state[3]
    phi_s = sol_iter.state[4]
    nu_s_prime = 2 * M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)
    front = (
        2 * psi_s / jnp.sqrt(jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2) + 1e-25)
    )
    inside_tanh = jnp.sqrt(jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2)) / (
        nu_s_prime + 2 / R + 1e-25
    )
    inside_tanh_safe = jnp.clip(inside_tanh, -0.999999, 0.999999)
    phi_inf = phi_s + front * jnp.arctanh(inside_tanh_safe)
    residual = jnp.array([phi_inf - phi_inf_target])
    surface_ok = sol_iter.reached_surface
    surface_ok &= (
        jnp.all(
            jnp.isfinite(
                jnp.array(
                    [
                        residual[0],
                        R,
                        M_s,
                        inside_tanh,
                    ]
                )
            )
        )
        & (R > 0)
        & (M_s > 0)
    )
    return residual, (R, M_s), surface_ok


@custom_vmap
def _deferred_recovery(active, previous, forward_args):
    """Scalar path; batching compacts flagged lanes into a GPU-only queue."""
    # Real TOV arguments have six entries; a seventh optionally carries
    # three scalar recovery hints through the existing queue pytree.
    hinted = len(forward_args) == 7
    background_args = forward_args[:6] if hinted else forward_args
    hint = forward_args[6] if hinted else None

    def recover(_):
        def forward(params):
            return _background_forward(params, False, *background_args)

        return _recover_background(forward, active, hint)

    return lax.cond(active, recover, lambda _: previous, operand=None)


@_deferred_recovery.def_vmap
def _deferred_recovery_batch(axis_size, in_batched, active, previous, forward_args):
    active, previous, forward_args = _broadcast_recovery_inputs(
        axis_size, in_batched, active, previous, forward_args
    )

    @custom_vmap
    def across_eos(active, previous, forward_args):
        return _run_recovery_queue(active, previous, forward_args, width=8)

    @across_eos.def_vmap
    def batch_eos(eos_count, in_batched, active, previous, forward_args):
        data = _broadcast_recovery_inputs(
            eos_count, in_batched, active, previous, forward_args
        )
        flat = jax.tree.map(
            lambda x: x.reshape((eos_count * axis_size,) + x.shape[2:]), data
        )
        outputs = _run_recovery_queue(*flat, width=eos_count * 8)
        outputs = jax.tree.map(
            lambda x: x.reshape((eos_count, axis_size) + x.shape[1:]), outputs
        )
        return outputs, jax.tree.map(lambda _: True, outputs)

    outputs = across_eos(active, previous, forward_args)
    return outputs, jax.tree.map(lambda _: True, outputs)


def _broadcast_recovery_inputs(axis_size, in_batched, active, previous, forward_args):
    """Materialize unmapped arguments before compacting the mapped axes."""
    return jax.tree.map(
        lambda x, batched: (
            x if batched else jnp.broadcast_to(x, (axis_size,) + jnp.shape(x))
        ),
        (active, previous, forward_args),
        tuple(in_batched),
    )


def _run_recovery_queue(active, previous, forward_args, *, width: int):
    """Drain one compact queue with a static number of concurrent stars."""
    # Shapes remain static under jit. Only the queue count is dynamic: no host
    # synchronization, dynamic allocation, or recompilation by failure count.
    axis_size = active.shape[0]
    count = jnp.sum(active, dtype=jnp.int32)
    indices = jnp.nonzero(active, size=axis_size, fill_value=0)[0]
    width = min(width, axis_size)
    # Padding permits a fixed-width final chunk without processing a real lane
    # twice. Out-of-range slots are inactive and never scattered back.
    indices = jnp.pad(indices, (0, width))

    def condition(state):
        offset, _ = state
        return offset < count

    def step(state):
        offset, outputs = state
        ids = lax.dynamic_slice_in_dim(indices, offset, width)
        enabled = offset + jnp.arange(width) < count
        args = jax.tree.map(lambda x: x[ids], forward_args)
        old = jax.tree.map(lambda x: x[ids], previous)
        recovered = jax.vmap(_deferred_recovery.fun)(enabled, old, args)
        # Drop padding rather than overwriting the valid result at index zero.
        targets = jnp.where(enabled, ids, axis_size)
        outputs = jax.tree.map(
            lambda x, y: x.at[targets].set(y, mode="drop"), outputs, recovered
        )
        return offset + width, outputs

    _, outputs = lax.while_loop(condition, step, (jnp.int32(0), previous))
    return outputs


@functools.partial(jax.jit, static_argnames=["max_iterations", "calculate_tidal"])
def _compiled_tov_solve_impl(
    pc,
    beta_ST,
    phi_inf_target,
    phi0,
    ns,
    ps,
    hs,
    es,
    dloge_dlogps,
    max_iterations=15,
    calculate_tidal=True,
):

    # Calculate log_ps_nsat
    ns_arr = ns
    log_ps_arr = jnp.log10(ps)
    sort_idx = jnp.argsort(ns_arr)
    ns_arr_sorted = ns_arr[sort_idx]
    log_ps_arr_sorted = log_ps_arr[sort_idx]

    nsat_standard = 0.16
    fixed_nsat_grid = jnp.logspace(jnp.log10(0.01), jnp.log10(15.0), 20)
    fixed_ns_grid = fixed_nsat_grid * nsat_standard
    query_ns_grid = jnp.where(
        jnp.max(ns_arr_sorted) > 1e10,
        fixed_ns_grid * utils.fm_inv3_to_geometric,
        fixed_ns_grid,
    )

    fixed_log_ps_nsat = jnp.interp(query_ns_grid, ns_arr_sorted, log_ps_arr_sorted)

    log_pc = jnp.log10(pc)
    hc = utils.interp_in_logspace(pc, ps, hs)

    phi0_guess = predict_phi0_last(beta_ST, log_pc, fixed_log_ps_nsat)
    is_valid = (phi0_guess > 0) & (phi0_guess < 1)
    phi0_guess = jnp.where(is_valid, phi0_guess, 0.5)

    eos_dict = {
        "p": ps,
        "h": hs,
        "e": es,
        "dloge_dlogp": dloge_dlogps,
        "beta_ST": beta_ST,
        "phi_c": phi0_guess,
        "phi_inf_target": phi_inf_target,
    }

    ec = utils.interp_in_logspace(hc, hs, es)
    dedp_c = ec / pc * jnp.interp(hc, hs, dloge_dlogps)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    dh = -1e-3 * hc
    h0 = hc + dh
    r0 = jnp.sqrt(3.0 * (-dh) / 2.0 / jnp.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * jnp.pi * ec * jnp.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    psi0 = 0.0

    H0_center = jnp.power(r0, 2)
    H0_prime_center = 2.0 * r0
    delta_phi_center = jnp.power(r0, 2)
    delta_phi_prime_center = 2.0 * r0

    nu0 = 0.0
    tol = 1e-4

    def forward_solver(params, accurate=False):
        return _background_forward(
            params, accurate, h0, dh, r0, m0, eos_dict, phi_inf_target
        )

    def run_iteration(phi0_init):
        return _correct_background_local(
            forward_solver,
            phi0_init,
            max_iterations,
            tol,
            return_skip_recovery=True,
            return_recovery_hint=True,
        )

    # Directly evaluate the single phi0 guess
    # Static opt-in flags above select the ten-entry metadata return.
    state = cast(_LocalRecoveryResult, run_iteration(phi0_guess))
    (
        i_final,
        _,
        R_final,
        M_inf_final,
        phi_inf_final,
        prev_x_final,
        _,
        surface_ok,
        skip_recovery,
        recovery_hint,
    ) = state
    # prev_x is the phi actually integrated; the other phi is the next proposal.
    phi0_final = prev_x_final[0]
    background_ok = surface_ok & (i_final <= max_iterations)
    background_ok &= jnp.abs(phi_inf_final) < tol
    background_ok &= jnp.isfinite(phi0_final)
    # Only the last evaluated trial decides eligibility: an intermediate cutoff
    # may still be followed by a valid trial during the fast iterations.
    hit_mass_cutoff = M_inf_final > 7.0 * utils.solar_mass_in_meter
    needs_recovery = ~background_ok & ~hit_mass_cutoff & ~skip_recovery
    phi0_final, R_final, M_inf_final, phi_inf_final, background_ok = _deferred_recovery(
        needs_recovery,
        (phi0_final, R_final, M_inf_final, phi_inf_final, background_ok),
        (h0, dh, r0, m0, eos_dict, phi_inf_target, recovery_hint),
    )
    # Failed lanes must also skip the final ODEs when cond is batched by vmap.
    phi0_final = jnp.where(background_ok, phi0_final, jnp.nan)
    returnNAN = ~background_ok

    def compute_success_branch(_):

        def compute_tidal(_):
            y0_batched = (
                jnp.array([r0, r0]),
                jnp.array([m0, m0]),
                jnp.array([nu0, nu0]),
                jnp.array([psi0, psi0]),
                jnp.array([phi0_final, phi0_final]),
                jnp.array([0.0, H0_center]),
                jnp.array([0.0, H0_prime_center]),
                jnp.array([delta_phi_center, 0.0]),
                jnp.array([delta_phi_prime_center, 0.0]),
            )

            def solve_single(y0):
                M_limit = 7.0 * utils.solar_mass_in_meter
                is_nan = jnp.isnan(y0[4])
                m0_eff = jnp.where(is_nan, M_limit + 10.0, y0[1])
                t0_eff = jnp.where(is_nan, 0.0, h0)
                y0_eff = (
                    y0[0],
                    m0_eff,
                    y0[2],
                    y0[3],
                    y0[4],
                    y0[5],
                    y0[6],
                    y0[7],
                    y0[8],
                )

                return _integrate_surface(
                    y0_eff, t0_eff, dh, eos_dict, False, M_limit, "tidal"
                )

            sol_batched = jax.vmap(solve_single)(y0_batched)
            tidal_ok = jnp.all(sol_batched.reached_surface)
            tidal_ok &= jnp.all(jnp.isfinite(sol_batched.state))

            M_s_final = sol_batched.state[0, 1]
            psi_s_final = sol_batched.state[0, 3]
            phi_s_final = sol_batched.state[0, 4]

            H0_surface_1, H0_surface_2 = sol_batched.state[:, 5]
            H0_prime_surface_1, H0_prime_surface_2 = sol_batched.state[:, 6]
            delta_phi_surface_1, delta_phi_surface_2 = sol_batched.state[:, 7]
            delta_phi_prime_surface_1, delta_phi_prime_surface_2 = sol_batched.state[
                :, 8
            ]

            nu_s_prime = 2 * M_s_final / (
                R_final * (R_final - 2 * M_s_final)
            ) + R_final * jnp.power(psi_s_final, 2)
            q = 2 * psi_s_final / (nu_s_prime + 1e-25)

            exterior_basis_matrix = build_exterior_basis(M_inf_final, q, R_final)
            exterior_basis_matrix_prime = build_exterior_basis_autodiff(
                M_inf_final, q, R_final
            )

            interior_sol = (
                H0_surface_2,
                H0_prime_surface_2,
                delta_phi_surface_2,
                delta_phi_prime_surface_2,
            )

            mat1_p0 = jnp.array(exterior_basis_matrix[0])
            mat1_p1 = jnp.array(exterior_basis_matrix[1])
            mat1_prime_p0 = jnp.array(exterior_basis_matrix_prime[0])
            mat1_prime_p1 = jnp.array(exterior_basis_matrix_prime[1])
            mat1_p0 = mat1_p0.at[1].set(-H0_surface_1)
            mat1_p1 = mat1_p1.at[1].set(-delta_phi_surface_1)
            mat1_prime_p0 = mat1_prime_p0.at[1].set(-H0_prime_surface_1)
            mat1_prime_p1 = mat1_prime_p1.at[1].set(-delta_phi_prime_surface_1)
            exterior_basis_matrix_1 = (mat1_p0, mat1_p1)
            exterior_basis_matrix_prime_1 = (mat1_prime_p0, mat1_prime_p1)

            coeffs_1 = coeff_solver(
                interior_sol, exterior_basis_matrix_1, exterior_basis_matrix_prime_1
            )
            cQT1, c2, cQS1, cES = coeffs_1

            mat2_part0 = jnp.array(exterior_basis_matrix[0])
            mat2_part1 = jnp.array(exterior_basis_matrix[1])
            mat2_prime_part0 = jnp.array(exterior_basis_matrix_prime[0])
            mat2_prime_part1 = jnp.array(exterior_basis_matrix_prime[1])
            mat2_part0 = mat2_part0.at[3].set(-H0_surface_1)
            mat2_part1 = mat2_part1.at[3].set(-delta_phi_surface_1)
            mat2_prime_part0 = mat2_prime_part0.at[3].set(-H0_prime_surface_1)
            mat2_prime_part1 = mat2_prime_part1.at[3].set(-delta_phi_prime_surface_1)
            exterior_basis_matrix_2 = (mat2_part0, mat2_part1)
            exterior_basis_matrix_prime_2 = (mat2_prime_part0, mat2_prime_part1)

            coeffs_2 = coeff_solver(
                interior_sol, exterior_basis_matrix_2, exterior_basis_matrix_prime_2
            )
            cQT2, cET, cQS2, c2 = coeffs_2

            coeffs = cQT1, cQT2, cET, cQS1, cQS2, cES
            lambda_T, lambda_S, lambda_ST1, lambda_ST2 = compute_tidal_deformabilities(
                coeffs
            )

            A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
            A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s_final, 2))
            R_jordan = A_phi_s * R_final
            M_inf_jordan = (1 / A_phi_inf) * (
                M_inf_final + (beta_ST * phi_inf_target * (-q * M_inf_final))
            )

            phi_inf_target_safe = jnp.where(
                jnp.abs(phi_inf_target) < 1e-25, 1e-25, phi_inf_target
            )
            beta_ST_safe = jnp.where(jnp.abs(beta_ST) < 1e-25, 1e-25, beta_ST)

            Lambda_T_J = lambda_T * jnp.power(M_inf_final, -5.0)
            Lambda_S_J = (
                (
                    jnp.exp(2 * beta_ST * jnp.power(phi_inf_target, 2))
                    / (4 * jnp.power(beta_ST_safe * phi_inf_target_safe, 2))
                )
                * lambda_S
                * jnp.power(M_inf_final, -5.0)
            )
            Lambda_ST1_J = (
                (
                    -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2))
                    / (2 * beta_ST_safe * phi_inf_target_safe)
                )
                * lambda_ST1
                * jnp.power(M_inf_final, -5.0)
            )
            Lambda_ST2_J = (
                (
                    -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2))
                    / (2 * beta_ST_safe * phi_inf_target_safe)
                )
                * lambda_ST2
                * jnp.power(M_inf_final, -5.0)
            )

            values = (
                M_inf_jordan,
                R_jordan,
                Lambda_T_J,
                Lambda_S_J,
                Lambda_ST1_J,
                Lambda_ST2_J,
                q,
            )
            tidal_ok &= jnp.all(jnp.isfinite(jnp.array(values)))
            tidal_ok &= (M_inf_jordan > 0) & (R_jordan > 0)
            return tuple(jnp.where(tidal_ok, value, jnp.nan) for value in values)

        def compute_no_tidal(_):
            M_limit = 20.0 * utils.solar_mass_in_meter
            is_nan = jnp.isnan(phi0_final)
            m0_eff = jnp.where(is_nan, M_limit + 10.0, m0)
            t0_eff = jnp.where(is_nan, 0.0, h0)
            y0 = (r0, m0_eff, nu0, psi0, phi0_final)

            sol = _integrate_surface(
                y0, t0_eff, dh, eos_dict, False, M_limit, "no_tidal"
            )

            M_s_final = sol.state[1]
            psi_s_final = sol.state[3]
            phi_s_final = sol.state[4]

            nu_s_prime = 2 * M_s_final / (
                R_final * (R_final - 2 * M_s_final)
            ) + R_final * jnp.power(psi_s_final, 2)
            q = 2 * psi_s_final / (nu_s_prime + 1e-25)

            A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
            A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s_final, 2))
            R_jordan = A_phi_s * R_final
            M_inf_jordan = (1 / A_phi_inf) * (
                M_inf_final + (beta_ST * phi_inf_target * (-q * M_inf_final))
            )

            values = (M_inf_jordan, R_jordan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, q)
            surface_ok = sol.reached_surface
            surface_ok &= (
                jnp.all(
                    jnp.isfinite(
                        jnp.array(
                            [
                                M_inf_jordan,
                                R_jordan,
                                q,
                            ]
                        )
                    )
                )
                & (M_inf_jordan > 0)
                & (R_jordan > 0)
            )
            return tuple(jnp.where(surface_ok, value, jnp.nan) for value in values)

        (
            M_inf_jordan,
            R_jordan,
            Lambda_T_J,
            Lambda_S_J,
            Lambda_ST1_J,
            Lambda_ST2_J,
            q,
        ) = lax.cond(calculate_tidal, compute_tidal, compute_no_tidal, operand=None)

        return (
            M_inf_jordan,
            R_jordan,
            Lambda_T_J,
            Lambda_S_J,
            Lambda_ST1_J,
            Lambda_ST2_J,
            q,
        )

    def compute_nan_branch(_):
        return (jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan)

    return lax.cond(returnNAN, compute_nan_branch, compute_success_branch, operand=None)


def _compiled_tov_solve(
    pc,
    beta_ST,
    phi_inf_target,
    phi0,
    ns,
    ps,
    hs,
    es,
    dloge_dlogps,
    max_iterations=15,
    calculate_tidal=True,
):
    """Dispatch a scalar-tensor solve while preserving its AD contract.

    The complete shooting/recovery workflow does not support JAX automatic
    differentiation in the pre-Modax Diffrax implementation either. Reject it
    before a native forward-only primitive can be staged; isolated background
    integrations continue to select the differentiable Diffrax path.
    """
    if _is_differentiating(
        pc, beta_ST, phi_inf_target, phi0, ns, ps, hs, es, dloge_dlogps
    ):
        raise NotImplementedError(
            "Full scalar-tensor shooting does not support JAX automatic "
            "differentiation; "
            "differentiate an isolated background integration, which uses Diffrax."
        )
    return _compiled_tov_solve_impl(
        pc,
        beta_ST,
        phi_inf_target,
        phi0,
        ns,
        ps,
        hs,
        es,
        dloge_dlogps,
        max_iterations=max_iterations,
        calculate_tidal=calculate_tidal,
    )


class ScalarTensorTOVSolver(TOVSolverBase):
    r"""
    Scalar-tensor theory TOV solver.

    Solves modified TOV equations that include scalar field coupling.
    The solution requires iterative solving to match boundary conditions
    at the star surface and spatial infinity.

    Uses five damped updates followed by safeguarded secant updates within
    15 local evaluations, including at most two temporary accurate trials.
    A worsening negative residual can trigger four additional probes once.
    Unresolved points use GPU-queued grid/bracket recovery, reusing a valid
    negative lower endpoint when an upward search encounters a failed trial.
    Recovery allows 17 boundary refinements and 40 root bisections. Background
    Diffrax fallback tolerances and its 500-step ceiling are unchanged. The
    optional CUDA Modax backend uses tighter validated tolerances. Any final
    integration failure invalidates the family.

    Implements the scalar-tensor TOV equations with tidal deformability
    following Creci et al. (2023) Phys.Rev.D 111 (2025) 8, 089901 (erratum).
    """

    def __init__(self, calculate_tidal: bool = True):
        self.calculate_tidal = calculate_tidal
        if _modax_integrator is None:
            logger.info(
                "Scalar-tensor ODE backend: diffrax "
                "(Modax optional dependencies unavailable)"
            )
        elif jax.default_backend() == "gpu":
            logger.info(
                "Scalar-tensor ODE backend: modax "
                "(CUDA forward integrations; Diffrax for AD)"
            )
        else:
            logger.info("Scalar-tensor ODE backend: diffrax (non-CUDA backend)")

    def solve(
        self, eos_data: EOSData, pc: float, tov_params: dict[str, float]
    ) -> TOVSolution:
        beta_ST = tov_params.get("beta_ST", 0.0)
        phi_inf_target = tov_params.get("phi_inf_tgt", 1e-3)
        phi0 = tov_params.get("phi_c", 1.0)
        max_iterations = 15

        (
            M_inf_jordan,
            R_jordan,
            Lambda_T_J,
            Lambda_S_J,
            Lambda_ST1_J,
            Lambda_ST2_J,
            q,
        ) = _compiled_tov_solve(
            pc,
            beta_ST,
            phi_inf_target,
            phi0,
            eos_data.ns,
            eos_data.ps,
            eos_data.hs,
            eos_data.es,
            eos_data.dloge_dlogps,
            max_iterations=max_iterations,
            calculate_tidal=self.calculate_tidal,
        )

        extra = {
            "lambda_S": Lambda_S_J,
            "lambda_ST1": Lambda_ST1_J,
            "lambda_ST2": Lambda_ST2_J,
            "q": q,
        }

        return TOVSolution(
            M=M_inf_jordan,
            R=R_jordan,
            k2=3.0
            / 2.0
            * Lambda_T_J
            * jnp.power(M_inf_jordan, 5.0)
            / jnp.power(R_jordan, 5.0),  # Rescaled from Lambda_T
            extra=extra,
        )  # type: ignore[arg-type]

    def get_required_parameters(self) -> list[str]:
        return ["beta_ST", "phi_inf_tgt", "phi_c"]

    def _create_family_data(
        self,
        pcs,
        masses,
        radii,
        k2s,
        ndat,
        extra=None,
        stability_branch: Literal["default", "first"] = "default",
    ):
        """Reject the whole ST family after any final raw or processed failure."""
        valid = jnp.all(jnp.isfinite(masses) & (masses > 0))
        valid &= jnp.all(jnp.isfinite(radii) & (radii > 0))
        if self.calculate_tidal:
            valid &= jnp.all(jnp.isfinite(k2s))
            if extra is not None:
                valid &= jnp.all(
                    jnp.stack(
                        [jnp.all(jnp.isfinite(value)) for value in extra.values()]
                    )
                )
        family = super()._create_family_data(
            pcs, masses, radii, k2s, ndat, extra, stability_branch
        )
        valid &= jnp.all(jnp.isfinite(family.masses) & (family.masses > 0))
        valid &= jnp.all(jnp.isfinite(family.radii) & (family.radii > 0))
        if self.calculate_tidal:
            valid &= jnp.all(jnp.isfinite(family.lambdas))
            if family.extra is not None:
                valid &= jnp.all(
                    jnp.stack(
                        [
                            jnp.all(jnp.isfinite(value))
                            for value in family.extra.values()
                        ]
                    )
                )
        return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), family)

    def get_family_extra_keys(self) -> list[str]:
        """Return scalar--tensor observables produced for each family point."""
        return ["lambda_S", "lambda_ST1", "lambda_ST2", "q"]
