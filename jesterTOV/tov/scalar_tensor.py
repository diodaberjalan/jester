"""
Scalar-Tensor TOV equation solver.

This module implements TOV equations for scalar-tensor theories of gravity,
where the gravitational interaction includes both a metric tensor and a scalar field.

**Units:** All calculations are performed in geometric units where G = c = 1.

**Reference:** G. Creci et al Phys.Rev.D 111 (2025) 8, 089901 (erratum)

# FIXME: Need to fully integrate the TOV solver: see docs/developer_guide/adding_new_tov.md
"""

import functools
from typing import Literal, NamedTuple, cast
import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_batching import custom_vmap
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, Bosh3, SaveAt, PIDController, Event, RecursiveCheckpointAdjoint

from jesterTOV import utils
from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, TOVSolution
from jesterTOV.tov.scalar_tensor_utils import (
    build_exterior_basis,
    build_exterior_basis_autodiff,
    coeff_solver,
    compute_tidal_deformabilities,
    predict_phi0_last,
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
            * (
                -alpha_phi
                * (
                    (dedp - 9.0) * p + (dedp - 1.0) * e
                )  
                + 4.0 * r * p * psi
            )
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
            + (e - 3.0 * p)
            * (-beta_ST)  
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


class _ProbePair(NamedTuple):
    next_phi: jax.Array
    next_residual: jax.Array
    previous_phi: jax.Array
    previous_residual: jax.Array
    has_positive: jax.Array
    has_negative: jax.Array
    has_valid: jax.Array


def _select_probe_pair(residuals: jax.Array, valid: jax.Array) -> _ProbePair:
    """Return the smallest positive residual, largest-field negative, and valid fallback."""
    fields = jnp.array([1e-3, 1e-2, 1e-1, 5e-1], dtype=residuals.dtype)
    valid = valid & jnp.isfinite(residuals)
    positive = valid & (residuals > 0)
    negative = valid & (residuals < 0)
    positive_index = jnp.argmin(jnp.where(positive, residuals, jnp.inf))
    negative_index = jnp.argmax(jnp.where(negative, fields, -jnp.inf))
    have_positive = jnp.any(positive)
    have_negative = jnp.any(negative)
    have_valid = jnp.any(valid)
    largest_valid = jnp.max(jnp.where(valid, fields, -jnp.inf))
    return _ProbePair(
        jnp.where(have_positive, fields[positive_index],
                  jnp.where(have_valid, largest_valid, jnp.nan)),
        jnp.where(have_positive, residuals[positive_index], jnp.nan),
        jnp.where(have_negative, fields[negative_index], jnp.nan),
        jnp.where(have_negative, residuals[negative_index], jnp.nan),
        have_positive,
        have_negative,
        have_valid,
    )


_RecoveryHint = tuple[jax.Array, jax.Array, jax.Array]
_LocalRecoveryResult = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array, _RecoveryHint,
]


class _BackgroundRootState(NamedTuple):
    i: jax.Array
    trial: jax.Array
    radius: jax.Array
    mass: jax.Array
    residual: jax.Array
    phi: jax.Array
    valid: jax.Array
    done: jax.Array
    negative_phi: jax.Array
    negative_F: jax.Array
    have_negative: jax.Array
    positive_phi: jax.Array
    positive_F: jax.Array
    have_positive: jax.Array
    anchor: jax.Array
    anchor_F: jax.Array
    have_anchor: jax.Array
    failed_lower: jax.Array
    failed_upper: jax.Array
    orientation: jax.Array
    side: jax.Array
    widths: jax.Array
    bests: jax.Array
    history_count: jax.Array
    best: jax.Array
    phase: jax.Array
    accuracy_used: jax.Array
    other_phi: jax.Array
    accurate_phi: jax.Array
    accurate_F: jax.Array
    probe_used: jax.Array
    probe_pending: jax.Array
    force_secant: jax.Array
    previous_phi: jax.Array
    previous_F: jax.Array
    previous_valid: jax.Array
    probe_valid: jax.Array
    midpoint_pending: jax.Array
    local_seed: jax.Array
    local_start_i: jax.Array
    damping_budget: jax.Array
    seek_positive: jax.Array
    seek_eligible: jax.Array
    largest_negative_phi: jax.Array
    largest_negative_F: jax.Array
    skip_recovery: jax.Array


def _correct_background_local(forward_solver, phi0_init, max_iterations, tol,
                              return_skip_recovery=False, return_recovery_hint=False):
    """Correct an NN seed with damped updates, then safeguarded secant steps.

    The first five updates use the residual directly; later updates use the
    latest two surface-valid ordinary evaluations. Retain the local search and
    Illinois bracket as safeguards for unusable proposals or failed trials.
    Only surface-valid ordinary residuals enter the secant history or bracket.
    A contracting bracket with stagnant residuals triggers at most two tighter
    evaluations, then ordinary accuracy resumes using a corrected proposal.
    An increasing field with a worsening negative residual triggers four
    additional probes once. Restart from a signed pair's log midpoint, or
    damp once from the largest valid field if no positive probe exists.
    Optional fixed-shape metadata lets recovery reuse a negative lower bound.
    Return the last actual trial for the caller's cutoff/recovery decision.
    """
    initial = _BackgroundRootState(
        i=jnp.int32(0),
        trial=phi0_init,
        radius=0.0,
        mass=0.0,
        residual=1e9,
        phi=phi0_init,
        valid=jnp.array(False),
        done=jnp.array(False),
        negative_phi=0.0,
        negative_F=-1.0,
        have_negative=jnp.array(False),
        positive_phi=jnp.inf,
        positive_F=1.0,
        have_positive=jnp.array(False),
        anchor=phi0_init,
        anchor_F=1e9,
        have_anchor=jnp.array(False),
        failed_lower=0.0,
        failed_upper=jnp.inf,
        orientation=1.0,
        side=jnp.int32(0),
        widths=jnp.full((4,), jnp.inf),
        bests=jnp.full((4,), jnp.inf),
        history_count=jnp.int32(0),
        best=jnp.inf,
        phase=jnp.int32(0),
        accuracy_used=jnp.array(False),
        other_phi=phi0_init,
        accurate_phi=phi0_init,
        accurate_F=0.0,
        probe_used=jnp.array(False),
        probe_pending=jnp.array(False),
        force_secant=jnp.array(False),
        previous_phi=phi0_init,
        previous_F=jnp.array(jnp.nan),
        previous_valid=jnp.array(False),
        probe_valid=jnp.zeros((4,), dtype=bool),
        midpoint_pending=jnp.array(False),
        local_seed=phi0_init,
        local_start_i=jnp.int32(0),
        damping_budget=jnp.int32(5),
        seek_positive=jnp.array(False),
        seek_eligible=jnp.array(False),
        largest_negative_phi=jnp.array(0.0),
        largest_negative_F=jnp.array(jnp.nan),
        skip_recovery=jnp.array(False),
    )

    def condition(state):
        return (state.i < max_iterations) & ~state.done & ~state.probe_pending

    def step(state):
        active = (state.i < max_iterations) & ~state.done & ~state.probe_pending
        trial = state.trial
        accurate = state.phase > 0
        # Completed lanes must skip physical work even in a vmapped while loop.
        residual, (radius, mass), valid = forward_solver(
            jnp.array([jnp.where(active, trial, jnp.nan)]), accurate=accurate
        )
        value = residual[0]
        valid = valid & jnp.isfinite(value) & jnp.isfinite(trial) & (trial > 0)
        at_cutoff = jnp.asarray(mass) > 7.0 * utils.solar_mass_in_meter
        finished = (valid | at_cutoff) & (jnp.abs(value) < tol)
        ordinary_valid = ~accurate & valid
        use_opposite = state.midpoint_pending & ordinary_valid
        secant_anchor = jnp.where(
            use_opposite,
            jnp.where(value < 0, state.positive_phi, state.negative_phi),
            state.anchor,
        )
        secant_anchor_F = jnp.where(
            use_opposite,
            jnp.where(value < 0, state.positive_F, state.negative_F),
            state.anchor_F,
        )
        negative = ordinary_valid & (value < 0)
        advance_negative = state.seek_positive & negative
        advance_negative &= trial >= state.largest_negative_phi
        largest_negative_phi = jnp.where(advance_negative, trial, state.largest_negative_phi)
        largest_negative_F = jnp.where(advance_negative, value, state.largest_negative_F)
        positive = ordinary_valid & (value >= 0)

        # Endpoints are named by residual sign, NOT by position: either slope
        # direction is allowed. Illinois weights prevent endpoint stagnation.
        nx = jnp.where(negative, trial, state.negative_phi)
        px = jnp.where(positive, trial, state.positive_phi)
        nf = jnp.where(
            negative,
            value,
            jnp.where(
                positive & (state.side == 1), 0.5 * state.negative_F, state.negative_F
            ),
        )
        pf = jnp.where(
            positive,
            value,
            jnp.where(
                negative & (state.side == -1), 0.5 * state.positive_F, state.positive_F
            ),
        )
        have_negative = state.have_negative | negative
        have_positive = state.have_positive | positive
        bracketed = have_negative & have_positive

        # Failed endpoints guide domain refinement, but supply no root residual.
        failed = ~accurate & ~valid
        upper = jnp.where(
            failed & (~state.have_anchor | (trial > secant_anchor)),
            jnp.minimum(state.failed_upper, trial),
            state.failed_upper,
        )
        lower = jnp.where(
            failed & state.have_anchor & (trial < secant_anchor),
            jnp.maximum(state.failed_lower, trial),
            state.failed_lower,
        )
        upper = jnp.where(ordinary_valid & (trial >= upper), jnp.inf, upper)
        lower = jnp.where(ordinary_valid & (trial <= lower), 0.0, lower)
        anchor = jnp.where(ordinary_valid, trial, secant_anchor)
        anchor_F = jnp.where(ordinary_valid, value, secant_anchor_F)
        have_anchor = state.have_anchor | ordinary_valid

        # Learn search direction on a valid unbracketed branch. With an
        # unresolved failed boundary, refine that local domain first instead
        # of following a spurious derivative away from the NN neighborhood.
        dx = trial - secant_anchor
        slope = (value - secant_anchor_F) / jnp.where(jnp.abs(dx) > 1e-14, dx, 1.0)
        learn_direction = ordinary_valid & state.have_anchor & ~bracketed
        learn_direction &= ~jnp.isfinite(upper) & (lower == 0.0)
        learn_direction &= (
            (jnp.abs(dx) > 1e-14) & jnp.isfinite(slope) & (jnp.abs(slope) > 1e-12)
        )
        orientation = jnp.where(learn_direction, jnp.sign(slope), state.orientation)
        orientation = jnp.where(state.seek_positive, 1.0, orientation)
        direction = jnp.where(have_negative, orientation, -orientation)
        base = jnp.where(have_negative, nx, px)
        base = jnp.where(state.seek_positive, largest_negative_phi, base)
        width = jnp.maximum(1e-6, 0.02 * state.local_seed) * jnp.exp2(state.i - state.local_start_i)
        boundary = jnp.where(direction > 0, upper, lower)
        bounded = jnp.where(direction > 0, jnp.isfinite(upper), lower > 0)
        expanded = jnp.maximum(0.5 * base, base + direction * width)
        search = jnp.where(bounded, 0.5 * (base + boundary), expanded)
        restoration = state.local_seed * (1.0 - jnp.minimum(0.75, 0.02 * jnp.exp2(state.i - state.local_start_i)))
        search = jnp.where(have_anchor, search, restoration)
        denominator = pf - nf
        interpolation = nx - nf * (px - nx) / jnp.where(
            denominator != 0, denominator, 1.0
        )
        inside = jnp.isfinite(interpolation) & (interpolation > jnp.minimum(nx, px))
        inside &= interpolation < jnp.maximum(nx, px)
        interpolation = jnp.where(inside, interpolation, 0.5 * (nx + px))
        safeguarded_trial = jnp.where(bracketed, interpolation, search)

        # Retain the old fast path without its unchecked residuals or repeated
        # half-field fallback. The anchor contains the last ordinary VALID
        # evaluation, so failed/accurate trials cannot contaminate the slope.
        damped_trial = trial - 0.5 * value
        secant_slope = (value - secant_anchor_F) / (dx + 1e-12)
        secant_step = jnp.clip(-0.8 * value / (secant_slope + 1e-12), -1e6, 1e6)
        secant_trial = trial + secant_step
        use_damping = (state.i - state.local_start_i < state.damping_budget) & ~state.force_secant
        proposed_trial = jnp.where(use_damping, damped_trial, secant_trial)
        usable = ordinary_valid & (use_damping | state.have_anchor)
        # Residual-only damping can run away on a negative-slope branch.
        # Keep useful small updates; safeguard steps larger than the field.
        runaway_damping = use_damping & (cast(jax.Array, orientation) < 0)
        runaway_damping &= jnp.abs(proposed_trial - trial) > jnp.abs(trial)
        usable &= ~runaway_damping
        usable &= jnp.isfinite(proposed_trial) & (proposed_trial > 0)
        usable &= use_damping | (
            jnp.isfinite(secant_slope) & (jnp.abs(dx) > 1e-14)
        )
        usable &= (proposed_trial > lower) & (proposed_trial < upper)
        upward_safeguard = state.seek_positive & negative
        upward_safeguard &= ~jnp.isfinite(proposed_trial) | (proposed_trial <= largest_negative_phi)
        usable &= ~upward_safeguard
        ordinary_trial = jnp.where(usable, proposed_trial, safeguarded_trial)

        # Three bracket updates: >=8x contraction but <2x residual improvement.
        best = jnp.where(
            ordinary_valid, jnp.minimum(state.best, jnp.abs(value)), state.best
        )
        record = ordinary_valid & bracketed & ~finished
        widths = cast(jax.Array, jnp.where(
            record,
            jnp.concatenate((state.widths[1:], jnp.array([jnp.abs(px - nx)]))),
            state.widths,
        ))
        bests = cast(jax.Array, jnp.where(
            record, jnp.concatenate((state.bests[1:], jnp.array([best]))), state.bests
        ))
        history_count = state.history_count + record.astype(jnp.int32)
        trigger = record & (history_count >= 4) & ~state.accuracy_used
        trigger &= (widths[-1] <= widths[0] / 8.0) & (bests[-1] >= bests[0] / 2.0)
        trigger &= state.i + 1 < max_iterations

        # Accurate samples NEVER update the ordinary bracket above. Two samples
        # estimate a direction correction; the next solve is ordinary again.
        fine_dx = trial - state.accurate_phi
        fine_slope = (value - state.accurate_F) / jnp.where(
            jnp.abs(fine_dx) > 1e-16, fine_dx, 1.0
        )
        slope_ok = (
            (jnp.abs(fine_dx) > 1e-16)
            & jnp.isfinite(fine_slope)
            & (jnp.abs(fine_slope) > 1e-12)
        )
        trust = 0.5 * jnp.maximum(jnp.abs(trial), 1e-4)
        correction = jnp.clip(
            -value / jnp.where(slope_ok, fine_slope, 1.0), -trust, trust
        )
        corrected = trial + correction
        corrected = jnp.where(
            valid & slope_ok & jnp.isfinite(corrected) & (corrected > 0),
            corrected,
            ordinary_trial,
        )
        second_accurate = (state.phase == 1) & valid
        next_trial = jnp.where(
            second_accurate,
            state.other_phi,
            jnp.where(state.phase == 2, corrected, ordinary_trial),
        )
        next_trial = jnp.where(trigger, trial, next_trial)
        phase = jnp.where(trigger, 1, jnp.where(second_accurate, 2, 0)).astype(
            jnp.int32
        )
        request_probe = ordinary_valid & state.previous_valid & ~state.probe_used
        request_probe &= (trial > state.previous_phi) & (value < state.previous_F)
        request_probe &= state.previous_F < 0
        request_probe &= ~finished & (state.i + 1 < max_iterations)
        positive_seen = state.seek_positive & ordinary_valid & (value > 0)
        crossing = positive_seen & ~finished & (state.i + 1 < max_iterations)
        crossing_midpoint = jnp.exp(0.5 * (jnp.log(largest_negative_phi) + jnp.log(trial)))
        next_trial = jnp.where(crossing, crossing_midpoint, next_trial)
        phase = jnp.where(crossing, jnp.int32(0), phase)
        # An unchanged failed NON-cutoff trial cannot add information. Return
        # it unresolved to recovery; never add an early mass-based rejection.
        stalled_failure = failed & ~at_cutoff & (next_trial == trial)
        updated = state._replace(
            i=state.i + 1,
            trial=next_trial,
            radius=radius,
            mass=mass,
            residual=value,
            phi=trial,
            valid=valid,
            done=finished | stalled_failure,
            negative_phi=jnp.where(crossing, largest_negative_phi, nx),
            negative_F=jnp.where(crossing, largest_negative_F, nf),
            have_negative=have_negative,
            positive_phi=jnp.where(crossing, trial, px),
            positive_F=jnp.where(crossing, value, pf),
            have_positive=have_positive,
            anchor=anchor,
            anchor_F=anchor_F,
            have_anchor=have_anchor,
            failed_lower=lower,
            failed_upper=upper,
            orientation=orientation,
            side=jnp.where(negative, -1, jnp.where(positive, 1, state.side)),
            widths=jnp.where(crossing, jnp.full((4,), jnp.inf), widths),
            bests=jnp.where(crossing, jnp.full((4,), jnp.inf), bests),
            history_count=jnp.where(crossing, jnp.int32(0), history_count),
            best=best,
            phase=phase,
            accuracy_used=state.accuracy_used | trigger,
            other_phi=jnp.where(trigger, jnp.where(value < 0, px, nx), state.other_phi),
            accurate_phi=jnp.where(accurate, trial, state.accurate_phi),
            accurate_F=jnp.where(accurate, value, state.accurate_F),
            midpoint_pending=crossing,
            force_secant=state.force_secant | crossing,
            # Remember a failed upward trial so recovery can reuse the
            # largest valid negative endpoint; local iterations may continue.
            skip_recovery=state.skip_recovery | (state.seek_positive & failed),
            seek_positive=state.seek_positive & ~positive_seen,
            largest_negative_phi=largest_negative_phi,
            largest_negative_F=largest_negative_F,
            probe_used=state.probe_used | request_probe,
            probe_pending=request_probe,
            previous_phi=trial,
            previous_F=value,
            previous_valid=ordinary_valid,
        )
        return jax.tree.map(
            lambda new, old: jnp.where(active, new, old), updated, state
        )

    # Pause each triggered lane once. Never put four extra integrations inside
    # every local iteration: one fixed probe axis is evaluated between phases.
    paused = lax.while_loop(condition, step, initial)
    probe_fields = jnp.array([1e-3, 1e-2, 1e-1, 5e-1])
    probe_forward = forward_solver

    def evaluate_probes(_):
        def one(field):
            field = jnp.where(paused.probe_pending, field, jnp.nan)
            output = probe_forward(jnp.array([field]), accurate=False)
            residual = output[0][0]
            valid = output[2] & jnp.isfinite(residual)
            return residual, valid
        return jax.vmap(one)(probe_fields)

    probe_F, probe_valid = lax.cond(
        paused.probe_pending,
        evaluate_probes,
        lambda _: (jnp.full((4,), jnp.nan), jnp.zeros((4,), dtype=bool)),
        operand=None,
    )
    pair = _select_probe_pair(probe_F, probe_valid)
    pending = paused.probe_pending
    restart_pair = pending & pair.has_positive
    switch_trial = pending & pair.has_valid
    # A negative local point always exists at this trigger. Keep the most
    # recent one only when the four probes provide no valid negative residual.
    previous_phi = cast(jax.Array, jnp.where(pair.has_negative, pair.previous_phi, paused.anchor))
    previous_F = cast(jax.Array, jnp.where(pair.has_negative, pair.previous_residual, paused.anchor_F))
    # One bisection in log(field): the geometric, not arithmetic, midpoint.
    log_midpoint = jnp.exp(0.5 * (jnp.log(previous_phi) + jnp.log(pair.next_phi)))
    selected_trial = jnp.where(pair.has_positive, log_midpoint, pair.next_phi)
    resumed = paused._replace(
        trial=jnp.where(switch_trial, selected_trial, paused.trial),
        probe_pending=jnp.array(False),
        phase=jnp.where(switch_trial, jnp.int32(0), paused.phase),
        force_secant=paused.force_secant | switch_trial,
        midpoint_pending=restart_pair,
        anchor=jnp.where(restart_pair, previous_phi, paused.anchor),
        anchor_F=jnp.where(restart_pair, previous_F, paused.anchor_F),
        have_anchor=paused.have_anchor | restart_pair,
        negative_phi=jnp.where(restart_pair, previous_phi, paused.negative_phi),
        negative_F=jnp.where(restart_pair, previous_F, paused.negative_F),
        have_negative=paused.have_negative | restart_pair,
        positive_phi=jnp.where(restart_pair, pair.next_phi, paused.positive_phi),
        positive_F=jnp.where(restart_pair, pair.next_residual, paused.positive_F),
        have_positive=paused.have_positive | restart_pair,
        side=jnp.where(restart_pair, jnp.int32(-1), paused.side),
        widths=jnp.where(restart_pair, jnp.full((4,), jnp.inf), paused.widths),
        bests=jnp.where(restart_pair, jnp.full((4,), jnp.inf), paused.bests),
        history_count=jnp.where(restart_pair, jnp.int32(0), paused.history_count),
        best=jnp.where(restart_pair, jnp.minimum(jnp.abs(previous_F),
                                               jnp.abs(pair.next_residual)), paused.best),
        skip_recovery=paused.skip_recovery | (pending & ~pair.has_valid),
        probe_valid=probe_valid,
    )
    # Negative-only probes provide no bracket. Treat their largest valid field
    # as a new initialization: discard old NN/root history, make exactly one
    # residual-damped update, then use the existing secant. The ordinary global
    # counter and one-time probe metadata remain unchanged.
    negative_restart = pending & pair.has_valid & ~pair.has_positive
    valid_pairs = jnp.triu(probe_valid[:, None] & probe_valid[None, :], k=1)
    decreasing = probe_F[None, :] < probe_F[:, None]
    all_negative = jnp.all(~probe_valid | (probe_F < 0))
    eligible = negative_restart & all_negative & jnp.any(valid_pairs)
    eligible &= jnp.all(~valid_pairs | decreasing)
    largest_index = jnp.argmax(jnp.where(probe_valid, probe_fields, -jnp.inf))
    reset_names = (
        "radius", "mass", "residual", "valid", "done",
        "negative_phi", "negative_F", "have_negative",
        "positive_phi", "positive_F", "have_positive",
        "anchor_F", "have_anchor", "failed_lower", "failed_upper",
        "orientation", "side", "widths", "bests", "history_count", "best",
        "phase", "accuracy_used", "accurate_F",
        "previous_F", "previous_valid", "force_secant", "midpoint_pending",
    )
    reset_values = {name: getattr(initial, name) for name in reset_names}
    fresh = resumed._replace(
        **reset_values,
        trial=selected_trial,
        phi=selected_trial,
        anchor=selected_trial,
        other_phi=selected_trial,
        accurate_phi=selected_trial,
        previous_phi=selected_trial,
        local_seed=selected_trial,
        local_start_i=paused.i,
        damping_budget=jnp.int32(1),
        seek_positive=eligible,
        seek_eligible=eligible,
        largest_negative_phi=selected_trial,
        largest_negative_F=probe_F[largest_index],
    )
    resumed = jax.tree.map(
        lambda reset, old: jnp.where(negative_restart, reset, old), fresh, resumed
    )
    final = lax.while_loop(condition, step, resumed)
    all_probes_failed = final.probe_used & ~jnp.any(final.probe_valid)
    bounded_recovery = final.skip_recovery & final.seek_eligible & ~all_probes_failed
    bounded_recovery &= jnp.isfinite(final.largest_negative_phi) & (final.largest_negative_phi > 0)
    bounded_recovery &= jnp.isfinite(final.largest_negative_F) & (final.largest_negative_F < 0)
    # This bounds the upward crossing being tracked, not every possible root.
    recovery_hint = (final.largest_negative_phi, final.largest_negative_F, bounded_recovery)
    skip_recovery = final.skip_recovery & ~bounded_recovery
    return (
        final.i,
        final.trial,
        final.radius,
        final.mass,
        final.residual,
        jnp.array([final.phi]),
        jnp.array([final.residual]),
        final.valid,
        *((skip_recovery,) if return_skip_recovery else ()),
        *((recovery_hint,) if return_recovery_hint else ()),
    )


def _recover_background(forward_solver, active, recovery_hint: _RecoveryHint | None = None):
    """Recover an unresolved shooting root using the tested grid and bisection.

    ``forward_solver`` returns residual, (radius, mass), and surface validity.
    Inactive lanes use the existing NaN sentinel: no recovery ODE steps under
    vmap, even when another lane needs the fallback. Never accept a cutoff
    endpoint as a stellar surface. Reuse an optional valid negative endpoint,
    stop at the first adjacent sign bracket, and otherwise refine the first
    eligible failed boundary at most 17 times before ordinary bisection.
    """
    grid = jnp.concatenate((
        jnp.array([0.0]), jnp.geomspace(1e-6, 0.02, 28),
        jnp.linspace(0.02, 0.4, 32)[1:], jnp.array([0.5, 0.65, 0.8]),
    ))

    # Optional metadata is a fixed three-scalar pytree. The Python branch
    # depends only on argument structure, never on traced numerical values.
    use_lower = jnp.array(False)
    lower_F = jnp.array(jnp.nan)
    if recovery_hint is not None:
        lower, lower_F, requested = recovery_hint
        use_lower = active & requested & jnp.isfinite(lower) & (lower > 0)
        use_lower &= jnp.isfinite(lower_F) & (lower_F < 0)
        lower = cast(jax.Array, jnp.where(use_lower, lower, 0.0))
        keep = grid > lower
        count = jnp.sum(keep, dtype=jnp.int32)
        ids = jnp.nonzero(keep, size=62, fill_value=0)[0]
        tail = jnp.where(jnp.arange(62) < count, grid[ids], jnp.nan)
        narrowed = jnp.concatenate((jnp.array([lower]), tail))
        grid = jnp.where(use_lower, narrowed, grid)

    def evaluate(phi, enabled):
        residual, (radius, mass), valid = forward_solver(
            jnp.array([jnp.where(active & enabled, phi, jnp.nan)])
        )
        return jnp.array([residual[0], radius, mass]), valid

    def grid_point(index, phi, enabled):
        reuse = use_lower & (index == 0)
        value, good = evaluate(phi, enabled & ~reuse)
        # Only the residual is used for initial bracket selection; actual
        # output radius/mass always come from a later bisection evaluation.
        known = jnp.array([lower_F, 0.0, 0.0])
        return jnp.where(reuse, known, value), jnp.where(reuse, True, good)

    tile_size = 16
    # One NaN pad makes each final tile statically sixteen elements. It never
    # contributes a value or bracket to the original 63-point grid.
    scan_grid = jnp.concatenate((grid, jnp.array([jnp.nan])))

    def scan_cond(state):
        index, _, _, found = state
        base = jnp.minimum(index, jnp.int32(48))
        tile = lax.dynamic_slice(scan_grid, (base,), (tile_size,))
        return active & (index < 63) & ~found & jnp.any(jnp.isfinite(tile))

    def scan_step(state):
        index, values, valid, found = state
        # ``vmap`` executes a mapped while body for completed lanes while any
        # lane remains live.  Keep those lane states unchanged, rather than
        # allowing a stale final tile to overwrite prior values.
        base = jnp.minimum(index, jnp.int32(48))
        indices = base + jnp.arange(tile_size, dtype=jnp.int32)
        tile = lax.dynamic_slice(scan_grid, (base,), (tile_size,))
        running = active & ~found & (index < 63)
        enabled = running & (indices < 63) & jnp.isfinite(tile)
        phi = jnp.where(enabled, tile, jnp.nan)
        value, good = jax.vmap(grid_point)(indices, phi, enabled)
        good = cast(jax.Array, good)
        previous_value = values[jnp.maximum(base - 1, 0)]
        previous_valid = valid[jnp.maximum(base - 1, 0)]
        prior_values = jnp.concatenate((previous_value[None], value[:-1]))
        prior_valid = jnp.concatenate((previous_valid[None], good[:-1]))
        hit = enabled & (indices > 0) & prior_valid & good
        hit &= prior_values[:, 0] * value[:, 0] <= 0
        new_state = (
            index + tile_size,
            values.at[indices].set(value),
            valid.at[indices].set(good),
            found | jnp.any(hit),
        )
        return jax.tree.map(
            lambda updated, original: jnp.where(running, updated, original),
            new_state, state,
        )

    _, values, valid, _ = lax.while_loop(
        scan_cond, scan_step,
        (jnp.int32(0), jnp.full((63, 3), jnp.nan),
         jnp.zeros(63, dtype=bool), jnp.array(False)),
    )
    residuals = values[:, 0]
    signs = valid[:-1] & valid[1:] & (residuals[:-1] * residuals[1:] <= 0)
    have_sign = jnp.any(signs)
    si = jnp.argmax(signs)
    boundary = valid[:-1] & ~valid[1:] & jnp.isfinite(grid[1:]) & (residuals[:-1] < 0)
    have_boundary = jnp.any(boundary) & ~have_sign
    bi = jnp.argmax(boundary)

    def boundary_cond(state):
        k, _, _, _, found = state
        return (k < 17) & have_boundary & ~found

    def boundary_step(state):
        k, lo, hi, flo, found = state
        enabled = have_boundary & ~found
        mid = (lo + hi) / 2
        value, good = evaluate(mid, enabled)
        hit = enabled & good & (value[0] >= 0)
        move = enabled & good & ~hit
        return (k + 1, jnp.where(move, mid, lo),
                jnp.where(hit | (enabled & ~good), mid, hi),
                jnp.where(move, value[0], flo), found | hit)

    _, lo, hi, flo, found = lax.while_loop(
        boundary_cond, boundary_step,
        (jnp.int32(0), grid[bi], grid[bi + 1], residuals[bi], jnp.array(False)),
    )
    lo = jnp.where(have_sign, grid[si], lo)
    hi = jnp.where(have_sign, grid[si + 1], hi)
    flo = jnp.where(have_sign, residuals[si], cast(jax.Array, flo))
    have = have_sign | found

    def bisect_cond(state):
        k, _, _, _, done, bad, _, _ = state
        return (k < 40) & have & ~done & ~bad

    def bisect_step(state):
        k, lo, hi, flo, done, bad, root, best = state
        enabled = have & ~done & ~bad
        mid = (lo + hi) / 2
        value, good = evaluate(mid, enabled)
        right = enabled & good & (flo * value[0] > 0)
        return (k + 1, jnp.where(right, mid, lo),
                jnp.where(enabled & good & ~right, mid, hi),
                jnp.where(right, value[0], flo),
                done | (enabled & good & (jnp.abs(value[0]) < 1e-4)),
                bad | (enabled & ~good), jnp.where(enabled, mid, root),
                jnp.where(enabled, value, best))

    _, _, _, _, done, bad, phi, best = lax.while_loop(
        bisect_cond, bisect_step,
        (jnp.int32(0), lo, hi, flo, jnp.array(False), jnp.array(False),
         jnp.array(jnp.nan), jnp.full(3, jnp.nan)),
    )
    return phi, best[1], best[2], best[0], have & done & ~bad


def _background_forward(params, accurate, h0, dh, r0, m0, eos_dict, phi_inf_target):
    """Shared background integration for fast shooting and queued recovery."""
    phi0_trial = params[0]
    M_limit = 7.0 * utils.solar_mass_in_meter
    is_nan = jnp.isnan(phi0_trial)
    m0_eff = jnp.where(is_nan, M_limit + 10.0, m0)
    t0_eff = jnp.where(is_nan, 0.0, h0)
    y0 = (r0, m0_eff, 0.0, 0.0, phi0_trial)

    def mass_event(t, y, args, **kwargs):
        return y[1] > M_limit

    sol_iter = diffeqsolve(
        ODETerm(_tov_ode_iter),
        Bosh3(scan_kind="bounded"),
        t0=t0_eff,
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
    R = sol_iter.ys[0][-1]
    M_s = sol_iter.ys[1][-1]
    psi_s = sol_iter.ys[3][-1]
    phi_s = sol_iter.ys[4][-1]
    nu_s_prime = 2 * M_s / (R * (R - 2.0 * M_s)) + R * jnp.power(psi_s, 2)
    front = (
        2 * psi_s / jnp.sqrt(jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2) + 1e-25)
    )
    inside_tanh = jnp.sqrt(
        jnp.power(nu_s_prime, 2) + 4 * jnp.power(psi_s, 2)
    ) / (nu_s_prime + 2 / R + 1e-25)
    inside_tanh_safe = jnp.clip(inside_tanh, -0.999999, 0.999999)
    phi_inf = phi_s + front * jnp.arctanh(inside_tanh_safe)
    residual = jnp.array([phi_inf - phi_inf_target])
    surface_ok = (sol_iter.result._value == 0) & (sol_iter.ts[-1] == 0)
    surface_ok &= jnp.all(jnp.isfinite(jnp.array([
        residual[0], R, M_s, inside_tanh,
    ]))) & (R > 0) & (M_s > 0)
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
        lambda x, batched: x if batched else jnp.broadcast_to(x, (axis_size,) + jnp.shape(x)),
        (active, previous, forward_args), tuple(in_batched),
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
def _compiled_tov_solve(
    pc, beta_ST, phi_inf_target, phi0, ns, ps, hs, es, dloge_dlogps, max_iterations=15, calculate_tidal=True
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
    query_ns_grid = jnp.where(jnp.max(ns_arr_sorted) > 1e10, fixed_ns_grid * utils.fm_inv3_to_geometric, fixed_ns_grid)
    
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
            forward_solver, phi0_init, max_iterations, tol,
            return_skip_recovery=True, return_recovery_hint=True
        )

    # Directly evaluate the single phi0 guess
    # Static opt-in flags above select the ten-entry metadata return.
    state = cast(_LocalRecoveryResult, run_iteration(phi0_guess))
    i_final, _, R_final, M_inf_final, phi_inf_final, prev_x_final, _, surface_ok, skip_recovery, recovery_hint = state
    # prev_x is the phi actually integrated; the other phi is the next proposal.
    phi0_final = prev_x_final[0]
    background_ok = surface_ok & (i_final <= max_iterations)
    background_ok &= jnp.abs(phi_inf_final) < tol
    background_ok &= jnp.isfinite(phi0_final)
    # Only the last evaluated trial decides eligibility: an intermediate cutoff
    # may still be followed by a valid trial during the fast iterations.
    hit_mass_cutoff = M_inf_final > 7.0 * utils.solar_mass_in_meter
    needs_recovery = ~background_ok & ~hit_mass_cutoff & ~skip_recovery
    (
        phi0_final, R_final, M_inf_final, phi_inf_final, background_ok
    ) = _deferred_recovery(
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
                y0_eff = (y0[0], m0_eff, y0[2], y0[3], y0[4], y0[5], y0[6], y0[7], y0[8])

                def tidal_mass_event(t, y, args, **kwargs):
                    return y[1] > M_limit

                return diffeqsolve(
                    ODETerm(_tov_ode_iter_tidal),
                    Dopri5(scan_kind="bounded"),
                    t0=t0_eff,
                    t1=0,
                    dt0=dh,
                    y0=y0_eff,
                    args=eos_dict,
                    saveat=SaveAt(t1=True),
                    stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                    event=Event(tidal_mass_event),
                    max_steps=1000,
                    throw=False,
                )

            sol_batched = jax.vmap(solve_single)(y0_batched)
            tidal_ok = jnp.all(sol_batched.result._value == 0)
            tidal_ok &= jnp.all(sol_batched.ts[:, -1] == 0)
            tidal_ok &= jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(value)) for value in sol_batched.ys
            ]))

            M_s_final = sol_batched.ys[1][0, -1]
            psi_s_final = sol_batched.ys[3][0, -1]
            phi_s_final = sol_batched.ys[4][0, -1]

            H0_surface_1, H0_surface_2 = sol_batched.ys[5][:, -1]
            H0_prime_surface_1, H0_prime_surface_2 = sol_batched.ys[6][:, -1]
            delta_phi_surface_1, delta_phi_surface_2 = sol_batched.ys[7][:, -1]
            delta_phi_prime_surface_1, delta_phi_prime_surface_2 = sol_batched.ys[8][:, -1]

            nu_s_prime = 2 * M_s_final / (R_final * (R_final - 2 * M_s_final)) + R_final * jnp.power(psi_s_final, 2)
            q = 2 * psi_s_final / (nu_s_prime + 1e-25)

            exterior_basis_matrix = build_exterior_basis(M_inf_final, q, R_final)
            exterior_basis_matrix_prime = build_exterior_basis_autodiff(M_inf_final, q, R_final)

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
            lambda_T, lambda_S, lambda_ST1, lambda_ST2 = compute_tidal_deformabilities(coeffs)

            A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
            A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s_final, 2))
            R_jordan = A_phi_s * R_final
            M_inf_jordan = (1 / A_phi_inf) * (M_inf_final + (beta_ST * phi_inf_target * (-q * M_inf_final)))

            phi_inf_target_safe = jnp.where(jnp.abs(phi_inf_target) < 1e-25, 1e-25, phi_inf_target)
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
                q
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
            
            sol = diffeqsolve(
                ODETerm(_tov_ode_iter),
                Tsit5(scan_kind="bounded"),
                t0=t0_eff,
                t1=0,
                dt0=dh,
                y0=y0,
                args=eos_dict,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                throw=False,
            )
            
            M_s_final = sol.ys[1][-1]
            psi_s_final = sol.ys[3][-1]
            phi_s_final = sol.ys[4][-1]

            nu_s_prime = 2 * M_s_final / (R_final * (R_final - 2 * M_s_final)) + R_final * jnp.power(psi_s_final, 2)
            q = 2 * psi_s_final / (nu_s_prime + 1e-25)

            A_phi_inf = jnp.exp(0.5 * beta_ST * jnp.power(phi_inf_target, 2))
            A_phi_s = jnp.exp(0.5 * beta_ST * jnp.power(phi_s_final, 2))
            R_jordan = A_phi_s * R_final
            M_inf_jordan = (1 / A_phi_inf) * (M_inf_final + (beta_ST * phi_inf_target * (-q * M_inf_final)))

            values = (
                M_inf_jordan,
                R_jordan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                jnp.nan,
                q
            )
            surface_ok = (sol.result._value == 0) & (sol.ts[-1] == 0)
            surface_ok &= jnp.all(jnp.isfinite(jnp.array([
                M_inf_jordan, R_jordan, q,
            ]))) & (M_inf_jordan > 0) & (R_jordan > 0)
            return tuple(jnp.where(surface_ok, value, jnp.nan) for value in values)

        (
            M_inf_jordan, R_jordan, Lambda_T_J, Lambda_S_J, Lambda_ST1_J, Lambda_ST2_J, q
        ) = lax.cond(calculate_tidal, compute_tidal, compute_no_tidal, operand=None)

        return (
            M_inf_jordan, R_jordan, Lambda_T_J, Lambda_S_J, Lambda_ST1_J, Lambda_ST2_J, q
        )

    def compute_nan_branch(_):
        return (
            jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan
        )

    return lax.cond(returnNAN, compute_nan_branch, compute_success_branch, operand=None)


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
    ODE tolerances and the 500-step ceiling are unchanged. Any final integration
    failure invalidates the family.

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
            q
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
            "q": q
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
                valid &= jnp.all(jnp.stack([
                    jnp.all(jnp.isfinite(value)) for value in extra.values()
                ]))
        family = super()._create_family_data(
            pcs, masses, radii, k2s, ndat, extra, stability_branch
        )
        valid &= jnp.all(jnp.isfinite(family.masses) & (family.masses > 0))
        valid &= jnp.all(jnp.isfinite(family.radii) & (family.radii > 0))
        if self.calculate_tidal:
            valid &= jnp.all(jnp.isfinite(family.lambdas))
            if family.extra is not None:
                valid &= jnp.all(jnp.stack([
                    jnp.all(jnp.isfinite(value)) for value in family.extra.values()
                ]))
        return jax.tree.map(lambda value: jnp.where(valid, value, jnp.nan), family)

    def get_family_extra_keys(self) -> list[str]:
        """Return scalar--tensor observables produced for each family point."""
        return ["lambda_S", "lambda_ST1", "lambda_ST2", "q"]
