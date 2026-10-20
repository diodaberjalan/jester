"""Small regressions for ST fallback and whole-family failure handling."""

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData, FamilyData
from jesterTOV.tov.scalar_tensor import (
    ScalarTensorTOVSolver,
    _correct_background_local,
    _recover_background,
)
from jesterTOV.tov import scalar_tensor


@pytest.mark.parametrize("slope", [1.5, 13.2, 156.9, -20.0])
@pytest.mark.parametrize("seed", [0.28, 0.31004, 0.34])
def test_local_corrector_handles_residual_slopes(slope, seed):
    def forward(params, accurate=False):
        phi = params[0]
        return jnp.array([slope * (phi - 0.31)]), (12000.0, 2000.0), jnp.isfinite(phi)

    state = jax.jit(lambda: _correct_background_local(forward, seed, 15, 1e-4))()
    assert state[-1]
    assert abs(state[4]) < 1e-4
    assert state[0] <= 15
    np.testing.assert_allclose(state[4], slope * (state[5][0] - 0.31))


def test_local_corrector_keeps_converged_lanes_frozen():
    def solve(seed):
        def forward(params, accurate=False):
            phi = params[0]
            return (
                jnp.array([100.0 * (phi - 0.31)]),
                (12000.0, 2000.0),
                jnp.isfinite(phi),
            )

        return _correct_background_local(forward, seed, 15, 1e-4)

    state = jax.jit(jax.vmap(solve))(jnp.array([0.31, 0.31004]))
    assert state[0][0] == 1
    assert state[0][1] > 1
    np.testing.assert_allclose(state[5][0], [0.31])
    assert np.all(state[-1])
    assert np.all(np.abs(state[4]) < 1e-4)


@pytest.mark.parametrize("failure", ["invalid", "worse"])
def test_local_corrector_finds_root_beyond_numerical_hole(failure):
    history = []

    def forward(params, accurate=False):
        phi = params[0]
        jax.debug.callback(lambda x: history.append(float(x)), phi)
        if failure == "invalid":
            bad = (phi > 0.3001) & (phi < 0.30035)
            residual = jnp.where(bad, jnp.nan, phi - 0.31)
        else:
            bad = (phi > 0.307) & (phi < 0.309)
            residual = jnp.where(bad, 0.02, phi - 0.31)
        valid = jnp.isfinite(phi) & ~(bad & (failure == "invalid"))
        return jnp.array([residual]), (12000.0, 2000.0), valid

    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    jax.block_until_ready(state)
    assert state[-1] and abs(state[4]) < 1e-4
    assert state[0] == len(history) <= 15
    np.testing.assert_allclose(state[5], [0.31], atol=1e-4)


def test_local_corrector_retreats_locally_from_invalid_nn_seed():
    history = []

    def forward(params, accurate=False):
        phi = params[0]
        jax.debug.callback(lambda x: history.append(float(x)), phi)
        return (
            jnp.array([phi - 0.31]),
            (12000.0, 2000.0),
            jnp.isfinite(phi) & (phi < 0.33),
        )

    state = jax.jit(lambda: _correct_background_local(forward, 0.335, 15, 1e-4))()
    jax.block_until_ready(state)
    assert state[-1] and abs(state[4]) < 1e-4
    np.testing.assert_allclose(history[:2], [0.335, 0.335 * 0.98])


def test_local_corrector_preserves_terminal_cutoff_trial():
    mass = 7.0 * scalar_tensor.utils.solar_mass_in_meter + 1.0

    def forward(params, accurate=False):
        return jnp.array([1e-5]), (12000.0, mass), jnp.array(False)

    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    assert state[0] == 1
    assert not state[-1]
    assert state[3] == mass
    assert state[5][0] == 0.3


def test_local_corrector_does_not_repeat_identical_failed_trial():
    history = []

    def forward(params, accurate=False):
        jax.debug.callback(lambda x: history.append(float(x)), params[0])
        return jnp.array([jnp.nan]), (12000.0, 2000.0), jnp.array(False)

    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    jax.block_until_ready(state)
    assert state[0] == len(history) == 8
    assert len(set(history)) == len(history)
    assert not state[-1]


def test_local_corrector_keeps_searching_after_intermediate_cutoff():
    cutoff = 7.0 * scalar_tensor.utils.solar_mass_in_meter

    def forward(params, accurate=False):
        phi = params[0]
        hit_cutoff = phi >= 0.33
        return (
            jnp.array([phi - 0.31]),
            (12000.0, jnp.where(hit_cutoff, cutoff + 1.0, 2000.0)),
            jnp.isfinite(phi) & ~hit_cutoff,
        )

    state = jax.jit(lambda: _correct_background_local(forward, 0.335, 15, 1e-4))()
    assert state[-1] and abs(state[4]) < 1e-4
    assert state[3] < cutoff


def test_local_corrector_does_not_early_reject_repeated_cutoff():
    cutoff = 7.0 * scalar_tensor.utils.solar_mass_in_meter

    def forward(params, accurate=False):
        return jnp.array([0.1]), (12000.0, cutoff + 1.0), jnp.array(False)

    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    assert state[0] == 15
    assert not state[-1]
    assert state[3] > cutoff


def _stagnating_forward(history, fail_accurate=False):
    def forward(params, accurate=False):
        phi = params[0]
        jax.debug.callback(
            lambda x, a: history.append((float(x), bool(a))), phi, accurate
        )
        # Coarse numerical sign jump makes a false bracket near 0.30998.
        # The tighter residual resolves the smooth root at 0.31 instead.
        coarse = 100.0 * (phi - 0.30998) + jnp.copysign(3e-4, phi - 0.30998)
        smooth = 100.0 * (phi - 0.31)
        coarse = jnp.where(jnp.abs(phi - 0.31) < 1e-6, smooth, coarse)
        residual = jnp.where(accurate, smooth, coarse)
        valid = jnp.isfinite(phi) & ~(jnp.asarray(fail_accurate) & accurate)
        return jnp.array([residual]), (12000.0 + phi, 2000.0 + phi), valid

    return forward


@pytest.mark.parametrize("budget", [9, 15])
def test_local_corrector_accuracy_burst_is_temporary_and_counted(budget):
    history = []
    forward = _stagnating_forward(history)
    state = jax.jit(lambda: _correct_background_local(forward, 0.3, budget, 1e-4))()
    jax.block_until_ready(state)
    modes = np.array([accurate for _, accurate in history])
    fine = np.flatnonzero(modes)
    assert len(fine) > 0
    assert len(fine) <= 2
    assert np.all(np.diff(fine) == 1)
    assert state[0] == len(history) <= budget
    if budget == 15:
        assert state[-1] and abs(state[4]) < 1e-4
        assert not modes[-1]  # Actual return to ordinary accuracy.
        assert len(fine) == 2
    # All returned fields describe the same last actually evaluated phi.
    np.testing.assert_allclose(state[5], [history[-1][0]])
    np.testing.assert_allclose(state[2], 12000.0 + history[-1][0])
    np.testing.assert_allclose(state[3], 2000.0 + history[-1][0])


def test_failed_accurate_probe_returns_to_ordinary_without_repeating_burst():
    history = []
    forward = _stagnating_forward(history, fail_accurate=True)
    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    jax.block_until_ready(state)
    modes = np.array([accurate for _, accurate in history])
    assert modes.sum() == 1
    assert not modes[-1]
    assert state[0] <= 15


def test_already_converged_nn_needs_no_probe_or_accuracy_burst():
    history = []

    def forward(params, accurate=False):
        jax.debug.callback(lambda a: history.append(bool(a)), accurate)
        return jnp.array([0.0]), (12000.0, 2000.0), jnp.array(True)

    state = jax.jit(lambda: _correct_background_local(forward, 0.31, 15, 1e-4))()
    jax.block_until_ready(state)
    assert state[0] == 1
    assert history == [False]


def test_one_sided_residuals_do_not_trigger_accuracy_burst():
    modes = []

    def forward(params, accurate=False):
        jax.debug.callback(lambda a: modes.append(bool(a)), accurate)
        return jnp.array([-0.01]), (12000.0, 2000.0), jnp.isfinite(params[0])

    state = jax.jit(lambda: _correct_background_local(forward, 0.3, 15, 1e-4))()
    jax.block_until_ready(state)
    assert state[0] == 15
    assert not state[-1] or abs(state[4]) >= 1e-4
    assert not any(modes)


def test_converged_lane_is_masked_while_another_lane_uses_accuracy_burst():
    history = []

    def solve(seed):
        def forward(params, accurate=False):
            phi = params[0]
            jax.debug.callback(
                lambda s, x, a: history.append((float(s), float(x), bool(a))),
                seed,
                phi,
                accurate,
            )
            coarse = 100.0 * (phi - 0.30998) + jnp.copysign(3e-4, phi - 0.30998)
            smooth = 100.0 * (phi - 0.31)
            coarse = jnp.where(jnp.abs(phi - 0.31) < 1e-6, smooth, coarse)
            return (
                jnp.array([jnp.where(accurate, smooth, coarse)]),
                (12000.0, 2000.0),
                jnp.isfinite(phi),
            )

        return _correct_background_local(forward, seed, 15, 1e-4)

    state = jax.jit(jax.vmap(solve))(jnp.array([0.31, 0.3]))
    jax.block_until_ready(state)
    assert state[0][0] == 1
    assert np.all(state[-1]) and np.all(np.abs(state[4]) < 1e-4)
    physical_first = [r for r in history if r[0] == 0.31 and np.isfinite(r[1])]
    assert physical_first == [(0.31, 0.31, False)]
    assert any(r[2] for r in history if r[0] == 0.3 and np.isfinite(r[1]))


def test_recovery_routing_uses_last_trial_and_masks_cutoff_lanes():
    # Exercise the actual routing block without compiling the expensive ODEs.
    source = ast.parse(
        inspect.getsource(inspect.unwrap(scalar_tensor._compiled_tov_solve))
    )
    body = source.body[0].body
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "state"
    )
    end = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "returnNAN"
    )
    wrapper = ast.parse("def route(trial):\n    pass").body[0]
    wrapper.body = (
        body[start : end + 1]
        + ast.parse("return phi0_final, M_inf_final, background_ok").body
    )
    wrapper.body[0] = ast.parse("state = make_state(trial)").body[0]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))

    def final_state(trial):
        # The fast loop already freezes the last real trial, excluding masked
        # NaN sentinels and any earlier cutoff hits.
        mass, residual, surface_ok, iterations = trial
        surface_ok = surface_ok.astype(bool)
        return (
            iterations,
            0.3,
            12000.0,
            mass,
            residual,
            jnp.array([0.3]),
            jnp.array([residual]),
            surface_ok,
        )

    def recover(active, previous, forward_args):
        return jax.tree.map(
            lambda new, old: jnp.where(active, new, old),
            (0.4, 13000.0, 2000.0, 0.0, active), previous,
        )

    # Build the wrapper's globals explicitly; only the routing block is run.
    env = dict(vars(scalar_tensor), make_state=final_state,
               max_iterations=15, tol=1e-4, forward_solver=None,
               h0=0., dh=0., r0=0., m0=0., eos_dict={}, phi_inf_target=.001,
               _deferred_recovery=recover)
    exec(compile(module, "<ST recovery routing>", "exec"), env)

    cutoff = 7.0 * scalar_tensor.utils.solar_mass_in_meter
    trials = jnp.array([
        [cutoff + 1., 1e-5, False, 6],   # cutoff despite small residual
        [cutoff + 1., 0.1, False, 15],   # cutoff at iteration limit
        [2000., 0.1, True, 15],         # unconverged, valid surface: recover
        [2000., 1e-5, False, 15],       # failed integration, not cutoff: recover
        [2000., 1e-5, True, 2],         # early success: no recovery
        [2000., 1e-5, True, 15],        # final success, even after earlier cutoff
    ])
    phi, mass, ok = jax.jit(jax.vmap(env["route"]))(trials)
    np.testing.assert_array_equal(ok, [False, False, True, True, True, True])
    np.testing.assert_allclose(phi[2:], [0.4, 0.4, 0.3, 0.3])
    assert np.isnan(phi[:2]).all()
    np.testing.assert_allclose(mass[:2], cutoff + 1.)


@pytest.mark.parametrize("indices", [[], [0], [1, 9], list(range(35))])
def test_deferred_recovery_compacts_and_preserves_order(monkeypatch, indices):
    history = []

    def forward(params, accurate, root):
        phi = params[0]
        jax.debug.callback(lambda p, r: history.append((float(p), float(r))), phi, root)
        return jnp.array([phi - root]), (12000., 2000.), jnp.isfinite(phi)

    monkeypatch.setattr(scalar_tensor, "_background_forward", forward)
    n = 37
    active = jnp.zeros(n, dtype=bool).at[jnp.array(indices, dtype=int)].set(True)
    roots = jnp.linspace(.1, .35, n)
    previous = (roots + 1., jnp.full(n, 11.), jnp.full(n, 22.),
                jnp.full(n, .5), ~active)
    args = (roots,)
    expected = jax.jit(jax.vmap(scalar_tensor._deferred_recovery.fun))(
        active, previous, args
    )
    jax.block_until_ready(expected)
    history.clear()
    actual = jax.jit(jax.vmap(scalar_tensor._deferred_recovery))(
        active, previous, args
    )
    jax.block_until_ready(actual)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, rtol=0, atol=0, equal_nan=True)
    evaluated_roots = {round(r, 6) for p, r in history if np.isfinite(p)}
    assert evaluated_roots == {round(float(roots[i]), 6) for i in indices}
    if not indices:
        assert not history  # The entire recovery queue is skipped.


def test_deferred_recovery_nested_vmap_and_shared_arguments(monkeypatch):
    def forward(params, accurate, root, scale):
        phi = params[0]
        return jnp.array([scale * (phi - root)]), (12000., 2000.), jnp.isfinite(phi)

    monkeypatch.setattr(scalar_tensor, "_background_forward", forward)
    roots = jnp.linspace(.1, .35, 26).reshape(2, 13)
    active = jnp.arange(26).reshape(2, 13) % 3 == 0
    previous = (roots + 1., jnp.full_like(roots, 11.), jnp.full_like(roots, 22.),
                jnp.full_like(roots, .5), ~active)

    def one_family(mask, old, root):
        return jax.vmap(scalar_tensor._deferred_recovery, in_axes=(0, 0, (0, None)))(
            mask, old, (root, 2.)
        )

    actual = jax.jit(jax.vmap(one_family))(active, previous, roots)
    expected = jax.jit(jax.vmap(jax.vmap(
        lambda mask, old, root: scalar_tensor._deferred_recovery.fun(mask, old, (root, 2.))
    )))(active, previous, roots)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("nested", [False, True])
def test_public_solver_traces_with_deferred_recovery(nested):
    # Exercise real Diffrax/custom-batching transformations, not only toy
    # residuals. No ODE execution or GPU is needed to catch tracing failures.
    values = jnp.linspace(1e-12, 1e-10, 8)
    eos = EOSData(values, values, values * 1e8, values * 3,
                  jnp.ones(8), jnp.full(8, .5))
    solver = ScalarTensorTOVSolver()

    def one_family(beta):
        return jax.vmap(lambda pc: solver.solve(
            eos, pc, {"beta_ST": beta, "phi_inf_tgt": .001}
        ))(values[:3])

    if nested:
        jax.make_jaxpr(jax.jit(jax.vmap(one_family)))(jnp.array([-9., -15.]))
    else:
        jax.make_jaxpr(jax.jit(one_family))(-9.)


@pytest.mark.parametrize("root,upper", [(0.31, 1.0), (0.329, 0.33)])
def test_recovery_finds_surface_root(root, upper):
    def forward(params):
        phi = params[0]
        return (
            jnp.array([phi - root]),
            (jnp.array(12000.0), jnp.array(2000.0)),
            jnp.isfinite(phi) & (phi < upper),
        )

    phi, radius, mass, residual, ok = jax.jit(
        lambda: _recover_background(forward, jnp.array(True))
    )()
    assert ok
    assert abs(phi - root) < 1e-4
    assert abs(residual) < 1e-4
    assert radius == 12000.0 and mass == 2000.0


def test_recovery_rejects_cutoff_endpoints():
    def forward(params):
        phi = params[0]
        # A zero residual beyond the valid surface domain is not convergence.
        return (
            jnp.array([phi - 0.31]),
            (jnp.array(12000.0), jnp.array(2000.0)),
            jnp.isfinite(phi) & (phi < 0.2),
        )

    assert not jax.jit(lambda: _recover_background(forward, True))()[-1]


def test_recovery_masks_successful_lanes():
    def forward(params):
        phi = params[0]
        return (
            jnp.array([phi - 0.31]),
            (jnp.array(12000.0), jnp.array(2000.0)),
            jnp.isfinite(phi),
        )

    out = jax.jit(jax.vmap(lambda active: _recover_background(forward, active)))(
        jnp.array([True, False])
    )
    np.testing.assert_array_equal(out[-1], [True, False])
    assert np.isnan(out[0][1])


@pytest.mark.parametrize("failed_field", [None, "mass", "k2", "extra", "processed"])
def test_whole_family_rejection(monkeypatch, failed_field):
    # Even a postprocessor that hides a raw failed point must not rescue it.
    def process(self, pcs, masses, radii, k2s, ndat, extra=None):
        result = jnp.ones(ndat)
        if failed_field == "processed":
            result = result.at[0].set(jnp.nan)
        return FamilyData(
            pcs, result, jnp.ones(ndat), jnp.ones(ndat), {"q": jnp.ones(ndat)}
        )

    monkeypatch.setattr(TOVSolverBase, "_create_family_data", process)
    values = jnp.ones(4)
    bad = values.at[0].set(jnp.nan)
    family = ScalarTensorTOVSolver()._create_family_data(
        values,
        bad if failed_field == "mass" else values,
        values,
        bad if failed_field == "k2" else values,
        4,
        {"q": bad if failed_field == "extra" else values},
    )
    for leaf in jax.tree.leaves(family):
        assert (
            np.all(np.isfinite(leaf))
            if failed_field is None
            else np.all(np.isnan(leaf))
        )


def test_no_tidal_family_allows_intentional_nan_love_numbers(monkeypatch):
    def process(self, pcs, masses, radii, k2s, ndat, extra=None):
        return FamilyData(pcs, masses, radii, k2s, extra)

    monkeypatch.setattr(TOVSolverBase, "_create_family_data", process)
    values = jnp.ones(4)
    family = ScalarTensorTOVSolver(calculate_tidal=False)._create_family_data(
        values,
        values,
        values,
        jnp.full(4, jnp.nan),
        4,
    )
    assert np.isfinite(family.masses).all()
    assert np.isnan(family.lambdas).all()
