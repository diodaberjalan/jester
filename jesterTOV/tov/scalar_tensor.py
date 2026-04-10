"""
Scalar-Tensor TOV equation solver.

This module implements TOV equations for scalar-tensor theories of gravity,
where the gravitational interaction includes both a metric tensor and a scalar field.

**Units:** All calculations are performed in geometric units where G = c = 1.

**Reference:** G. Creci et al Phys.Rev.D 111 (2025) 8, 089901 (erratum)

# FIXME: Need to fully integrate the TOV solver: see docs/developer_guide/adding_new_tov.md
"""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from diffrax import (
    diffeqsolve,
    ODETerm,
    Dopri8,
    Bosh3,
    SaveAt,
    PIDController,
    Event,
)

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution
from jesterTOV.tov.scalar_tensor_utils import (
    build_exterior_basis,
    build_exterior_basis_autodiff,
    coeff_solver,
    compute_tidal_deformabilities,
)


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
        jnp.abs(dpdr) < EPS, jnp.copysign(EPS, dpdr), dpdr  # Preserve sign
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
        / denom_non_tidal
        * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh

    dphidh = psi * drdh

    # Tidal deformabilities (l=2)
    comp = m / r
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


@functools.partial(jax.jit, static_argnames=["max_iterations", "calculate_tidal"])
def _compiled_tov_solve(
    pc,
    beta_ST,
    phi_inf_target,
    phi0,
    ps,
    hs,
    es,
    dloge_dlogps,
    max_iterations=100,
    calculate_tidal=True,
):
    eos_dict = {
        "p": ps,
        "h": hs,
        "e": es,
        "dloge_dlogp": dloge_dlogps,
        "beta_ST": beta_ST,
        "phi_c": phi0,
        "phi_inf_target": phi_inf_target,
    }

    hc = utils.interp_in_logspace(pc, ps, hs)
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
    psi0 = jnp.array(0.0, dtype=r0.dtype)
    nu0 = jnp.array(0.0, dtype=r0.dtype)

    H0_center = jnp.power(r0, 2.0)
    H0_prime_center = 2.0 * r0
    delta_phi_center = jnp.power(r0, 2.0)
    delta_phi_prime_center = 2.0 * r0

    damping = 0.5
    tol = 1e-5

    def run_iteration(phi0_init):
        init_state = (
            jnp.array(0, dtype=jnp.int32),
            phi0_init,
            jnp.array(0.0, dtype=r0.dtype),
            phi0_init,
            jnp.array(1e9, dtype=r0.dtype),
            r0,
            m0,
            jnp.array(False, dtype=jnp.bool_),
        )

        def forward_solver_bosh3(phi0_trial, skip_flag):
            y0 = (r0, m0, nu0, psi0, phi0_trial)
            M_limit = 20.0 * utils.solar_mass_in_meter

            t0_eval = jnp.where(skip_flag, 0.0, h0)

            def mass_event(t, y, args, **kwargs):
                return y[1] > M_limit

            sol_iter = diffeqsolve(
                ODETerm(_tov_ode_iter),
                Bosh3(scan_kind="bounded"),
                t0=t0_eval,
                t1=0.0,
                dt0=dh,
                y0=y0,
                args=eos_dict,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-4),
                event=Event(mass_event),
                throw=False,
                max_steps=256,
            )
            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            nu_s_prime = 2.0 * M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2.0)
            front = (
                2.0
                * psi_s
                / jnp.sqrt(jnp.power(nu_s_prime, 2.0) + 4.0 * jnp.power(psi_s, 2.0))
            )
            inside_tanh = jnp.sqrt(
                jnp.power(nu_s_prime, 2.0) + 4.0 * jnp.power(psi_s, 2.0)
            ) / (nu_s_prime + 2.0 / R)
            phi_inf = phi_s + front * jnp.arctanh(inside_tanh)

            return (phi_inf - phi_inf_target), (R, M_s), sol_iter.stats["num_steps"]

        def forward_solver_dopri8(phi0_trial, skip_flag):
            y0 = (r0, m0, nu0, psi0, phi0_trial)
            M_limit = 20.0 * utils.solar_mass_in_meter

            t0_eval = jnp.where(skip_flag, 0.0, h0)

            def mass_event(t, y, args, **kwargs):
                return y[1] > M_limit

            sol_iter = diffeqsolve(
                ODETerm(_tov_ode_iter),
                Dopri8(scan_kind="bounded"),
                t0=t0_eval,
                t1=0.0,
                dt0=dh,
                y0=y0,
                args=eos_dict,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                event=Event(mass_event),
                throw=False,
                max_steps=1024,
            )
            R = sol_iter.ys[0][-1]
            M_s = sol_iter.ys[1][-1]
            nu_s = sol_iter.ys[2][-1]
            psi_s = sol_iter.ys[3][-1]
            phi_s = sol_iter.ys[4][-1]

            nu_s_prime = 2.0 * M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2.0)
            front = (
                2.0
                * psi_s
                / jnp.sqrt(jnp.power(nu_s_prime, 2.0) + 4.0 * jnp.power(psi_s, 2.0))
            )
            inside_tanh = jnp.sqrt(
                jnp.power(nu_s_prime, 2.0) + 4.0 * jnp.power(psi_s, 2.0)
            ) / (nu_s_prime + 2.0 / R)
            phi_inf = phi_s + front * jnp.arctanh(inside_tanh)

            return (phi_inf - phi_inf_target), (R, M_s), sol_iter.stats["num_steps"]

        def step_func(state, _):
            i, curr_x, curr_F, prev_x, prev_F, R_prev, M_prev, done = state

            def compute_active():
                use_bosh3 = (i < 2) | (jnp.abs(curr_F) > 0.05)

                def run_solver(x_guess):
                    return lax.cond(
                        use_bosh3,
                        lambda _: forward_solver_bosh3(x_guess, done),
                        lambda _: forward_solver_dopri8(x_guess, done),
                        None,
                    )

                def damped_step():
                    step = -damping * curr_F
                    x_proposed = curr_x + step
                    x_next = jnp.where(x_proposed <= 0.0, curr_x * 0.5, x_proposed)

                    new_F, (R, M), steps = run_solver(x_next)
                    return x_next, new_F, curr_x, curr_F, R, M, steps, use_bosh3

                def illinois_step():
                    dx = curr_x - prev_x
                    dF = curr_F - prev_F
                    J = dF / (dx + 1e-15)

                    raw_step = -1.0 * curr_F / (J + 1e-15)
                    x_proposed = curr_x + jnp.clip(raw_step, -1e6, 1e6)

                    x_next = jnp.where(x_proposed <= 0.0, curr_x * 0.1, x_proposed)

                    new_F, (R, M), steps = run_solver(x_next)

                    same_sign = (new_F * curr_F) > 0.0
                    next_prev_x = jnp.where(same_sign, prev_x, curr_x)
                    next_prev_F = jnp.where(same_sign, prev_F * 0.5, curr_F)

                    return (
                        x_next,
                        new_F,
                        next_prev_x,
                        next_prev_F,
                        R,
                        M,
                        steps,
                        use_bosh3,
                    )

                return lax.cond(i < 2, damped_step, illinois_step)

            def skip_active():
                return (
                    curr_x,
                    curr_F,
                    prev_x,
                    prev_F,
                    R_prev,
                    M_prev,
                    jnp.array(0),
                    jnp.array(False),
                )

            (
                next_x,
                next_F,
                next_prev_x,
                next_prev_F,
                next_R,
                next_M,
                next_steps,
                is_bosh3,
            ) = lax.cond(done, skip_active, compute_active)

            new_done = done | (jnp.abs(next_F) < tol)

            new_state = (
                i + 1,
                next_x,
                next_F,
                next_prev_x,
                next_prev_F,
                next_R,
                next_M,
                new_done,
            )
            # Returning None below prevents JAX from accumulating a history tensor stack.
            return new_state, None

        final_state, _ = lax.scan(step_func, init_state, None, length=max_iterations)
        _, _, _, _, _, R_final, M_inf_final, is_done = final_state

        phi0_final = final_state[1]

        too_big_mass = (M_inf_final / utils.solar_mass_in_meter) > 20.0
        returnNAN = too_big_mass | (~is_done)

        def nan_branch(_):
            return (jnp.nan,) * 15

        def compute_branch_tidal(_):
            y0_batched = (
                jnp.array([r0, r0]),
                jnp.array([m0, m0]),
                jnp.array([nu0, nu0]),
                jnp.array([psi0, psi0]),
                jnp.array([final_state[1], final_state[1]]),
                jnp.array([0.0, H0_center]),
                jnp.array([0.0, H0_prime_center]),
                jnp.array([delta_phi_center, 0.0]),
                jnp.array([delta_phi_prime_center, 0.0]),
            )

            def solve_single(y0):
                return diffeqsolve(
                    ODETerm(_tov_ode_iter_tidal),
                    Dopri8(scan_kind="bounded"),
                    t0=h0,
                    t1=0.0,
                    dt0=dh,
                    y0=y0,
                    args=eos_dict,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                    throw=False,
                    max_steps=2048,
                )

            sol_batched = jax.vmap(solve_single)(y0_batched)

            R_s = sol_batched.ys[0][0, -1]
            M_s_final = sol_batched.ys[1][0, -1]
            nu_s = sol_batched.ys[2][0, -1]
            psi_s = sol_batched.ys[3][0, -1]
            phi_s = sol_batched.ys[4][0, -1]

            H0_surface_1, H0_surface_2 = sol_batched.ys[5][:, -1]
            H0_prime_surface_1, H0_prime_surface_2 = sol_batched.ys[6][:, -1]
            delta_phi_surface_1, delta_phi_surface_2 = sol_batched.ys[7][:, -1]
            delta_phi_prime_surface_1, delta_phi_prime_surface_2 = sol_batched.ys[8][
                :, -1
            ]

            return (
                R_final,
                M_inf_final,
                nu_s,
                psi_s,
                phi_s,
                M_s_final,
                H0_surface_1,
                H0_prime_surface_1,
                delta_phi_surface_1,
                delta_phi_prime_surface_1,
                H0_surface_2,
                H0_prime_surface_2,
                delta_phi_surface_2,
                delta_phi_prime_surface_2,
                phi0_final,
            )

        def compute_branch_no_tidal(_):
            y0 = (r0, m0, nu0, psi0, final_state[1])
            sol = diffeqsolve(
                ODETerm(_tov_ode_iter),
                Dopri8(scan_kind="bounded"),
                t0=h0,
                t1=0.0,
                dt0=dh,
                y0=y0,
                args=eos_dict,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                throw=False,
                max_steps=2048,
            )
            M_s_final = sol.ys[1][-1]
            nu_s = sol.ys[2][-1]
            psi_s = sol.ys[3][-1]
            phi_s = sol.ys[4][-1]

            return (
                R_final,
                M_inf_final,
                nu_s,
                psi_s,
                phi_s,
                M_s_final,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                phi0_final,
            )

        def branch_fun(_):
            return lax.cond(
                calculate_tidal, compute_branch_tidal, compute_branch_no_tidal, None
            )

        return lax.cond(returnNAN, nan_branch, branch_fun, None)

    (
        R,
        M_inf,
        nu_s,
        psi_s,
        phi_s,
        M_s,
        H0_surface_1,
        H0_prime_surface_1,
        delta_phi_surface_1,
        delta_phi_prime_surface_1,
        H0_surface_2,
        H0_prime_surface_2,
        delta_phi_surface_2,
        delta_phi_prime_surface_2,
        phi0_final,
    ) = run_iteration(phi0)

    nu_s_prime = 2.0 * M_s / (R * (R - 2.0 * M_s)) + R * psi_s * psi_s
    q = 2.0 * psi_s / nu_s_prime

    exterior_basis_matrix = build_exterior_basis(M_inf, q, R)
    exterior_basis_matrix_prime = build_exterior_basis_autodiff(M_inf, q, R)

    interior_sol = (
        H0_surface_2,
        H0_prime_surface_2,
        delta_phi_surface_2,
        delta_phi_prime_surface_2,
    )

    mat1_p0 = jnp.array(exterior_basis_matrix[0]).at[1].set(-H0_surface_1)
    mat1_p1 = jnp.array(exterior_basis_matrix[1]).at[1].set(-delta_phi_surface_1)
    mat1_prime_p0 = (
        jnp.array(exterior_basis_matrix_prime[0]).at[1].set(-H0_prime_surface_1)
    )
    mat1_prime_p1 = (
        jnp.array(exterior_basis_matrix_prime[1]).at[1].set(-delta_phi_prime_surface_1)
    )

    coeffs_1 = coeff_solver(
        interior_sol, (mat1_p0, mat1_p1), (mat1_prime_p0, mat1_prime_p1)
    )
    cQT1, c2, cQS1, cES = coeffs_1

    mat2_part0 = jnp.array(exterior_basis_matrix[0]).at[3].set(-H0_surface_1)
    mat2_part1 = jnp.array(exterior_basis_matrix[1]).at[3].set(-delta_phi_surface_1)
    mat2_prime_part0 = (
        jnp.array(exterior_basis_matrix_prime[0]).at[3].set(-H0_prime_surface_1)
    )
    mat2_prime_part1 = (
        jnp.array(exterior_basis_matrix_prime[1]).at[3].set(-delta_phi_prime_surface_1)
    )

    coeffs_2 = coeff_solver(
        interior_sol, (mat2_part0, mat2_part1), (mat2_prime_part0, mat2_prime_part1)
    )
    cQT2, cET, cQS2, c2 = coeffs_2

    coeffs = cQT1, cQT2, cET, cQS1, cQS2, cES
    lambda_T, lambda_S, lambda_ST1, lambda_ST2 = compute_tidal_deformabilities(coeffs)

    A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2.0))
    A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s, 2.0))
    R_jordan = A_phi_s * R
    M_inf_jordan = (1.0 / A_phi_inf) * (
        M_inf + (beta_ST * phi_inf_target * (-q * M_inf))
    )

    Lambda_T_J = lambda_T * jnp.power(M_inf, -5.0)
    Lambda_S_J = (
        (
            jnp.exp(2.0 * beta_ST * jnp.power(phi_inf_target, 2.0))
            / (4.0 * beta_ST * beta_ST * phi_inf_target * phi_inf_target)
        )
        * lambda_S
        * jnp.power(M_inf, -5.0)
    )
    Lambda_ST1_J = (
        (
            -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2.0))
            / (2.0 * beta_ST * phi_inf_target)
        )
        * lambda_ST1
        * jnp.power(M_inf, -5.0)
    )
    Lambda_ST2_J = (
        (
            -jnp.exp(beta_ST * jnp.power(phi_inf_target, 2.0))
            / (2.0 * beta_ST * phi_inf_target)
        )
        * lambda_ST2
        * jnp.power(M_inf, -5.0)
    )

    return (
        M_inf_jordan,
        R_jordan,
        Lambda_T_J,
        Lambda_S_J,
        Lambda_ST1_J,
        Lambda_ST2_J,
        q,
        phi0_final,
    )


class ScalarTensorTOVSolver(TOVSolverBase):
    r"""
    Scalar-tensor theory TOV solver.

    Solves modified TOV equations that include scalar field coupling.
    The solution requires iterative solving to match boundary conditions
    at the star surface and spatial infinity.

    Implements the scalar-tensor TOV equations with tidal deformability
    following Creci et al. (2023) Phys.Rev.D 111 (2025) 8, 089901 (erratum).
    """

    def __init__(self, calculate_tidal: bool = True):
        self.calculate_tidal = calculate_tidal

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
            phi0_final,
        ) = _compiled_tov_solve(
            pc,
            beta_ST,
            phi_inf_target,
            phi0,
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
            "phi0_final": phi0_final,
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
