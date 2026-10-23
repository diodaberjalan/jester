"""Regressions for the external, forward-only Modax compatibility wrapper."""

import ast
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.tov._modax.compat import _adapt_body, _ast_fingerprint


def test_wrapper_rejects_unknown_kernel_before_adaptation():
    with pytest.raises(RuntimeError, match="Unsupported Modax Tsit5 factory"):
        _adapt_body("def _make_body():\n    pass\n")


def test_adapted_kernel_matches_previous_forward_kernel():
    spec = importlib.util.find_spec("modax")
    if spec is None:
        pytest.skip("Optional Modax package is not installed")
    source = (Path(spec.origin).parent / "tsit5.py").read_text()
    factory = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_body"
    )
    start = min(factory.lineno, *(node.lineno for node in factory.decorator_list)) - 1
    adapted = ast.parse(
        _adapt_body("".join(source.splitlines(True)[start : factory.end_lineno]))
    )
    body = next(
        node
        for node in ast.walk(adapted)
        if isinstance(node, ast.FunctionDef) and node.name == "body"
    )
    # Fingerprint the entire previous device loop, not just the four extensions.
    # Changes to arithmetic, branching or step/save ordering must be reviewed.
    assert _ast_fingerprint(body) == (
        "2eedf4f7a1d68db3a8532578a88fb2a79883f64408d26f2bf92de58326cd156b"
    )


@pytest.fixture(scope="module")
def backend():
    if any(
        importlib.util.find_spec(name) is None for name in ("modax", "numba_cuda_mlir")
    ):
        pytest.skip("Optional Modax compiler dependencies are not installed")
    if jax.default_backend() != "gpu":
        pytest.skip("Modax execution requires CUDA")
    from jesterTOV.tov._modax import integrator

    return integrator


def _decay_and_mass(y, t, params):
    return -params[0] * y[0], params[1]


def test_cuda_accuracy_and_nested_batching(backend):
    solve = backend.make_solver(_decay_and_mass, 1e-8, 1e-9, 3000)
    parameters = jnp.array([10.0, 0.1, 1.0, 1.0])
    parameters = jnp.stack((parameters, parameters.at[3].set(10.0)))
    initial = jnp.array([1.0, 0.0])
    batched = jax.jit(jax.vmap(jax.vmap(solve, in_axes=(None, 0)), in_axes=(None, 0)))
    final, stats = batched(initial, jnp.stack((parameters, parameters)))
    np.testing.assert_allclose(final[..., 0], np.exp(-10.0), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(final[..., 1], 0.1, rtol=0, atol=1e-14)
    assert np.all(stats["accepted_steps"][:, 1] > stats["accepted_steps"][:, 0])


@pytest.mark.parametrize("initial", [[1.0, 2.0], [float("nan"), 0.0]])
def test_cuda_invalid_initial_state_skips_steps(backend, initial):
    solve = backend.make_solver(_decay_and_mass, 1e-8, 1e-9, 3000)
    final, stats = jax.jit(solve)(jnp.array(initial), jnp.array([10.0, 0.1, 1.0, 1.0]))
    np.testing.assert_equal(np.asarray(final), np.asarray(initial))
    assert all(int(value) == 0 for value in stats.values())


def test_cuda_mass_cutoff_preserves_event_endpoint(backend):
    solve = backend.make_solver(_decay_and_mass, 1e-8, 1e-9, 3000)
    final, stats = jax.jit(solve)(
        jnp.array([1.0, 0.0]), jnp.array([10.0, 1.0, 0.05, 1.0])
    )
    assert np.all(np.isfinite(final))
    assert 0.05 < final[1] < 1.0
    assert int(stats["accepted_steps"]) > 0


def test_cuda_unfinished_surface_stays_nan(backend):
    solve = backend.make_solver(_decay_and_mass, 1e-8, 1e-9, 1)
    final, stats = jax.jit(solve)(
        jnp.array([1.0, 0.0]), jnp.array([10.0, 0.1, 1.0, 1.0])
    )
    assert np.all(np.isnan(final))
    assert int(stats["loop_steps"]) == 1
