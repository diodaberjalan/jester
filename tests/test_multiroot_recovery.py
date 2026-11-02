import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.tov import scalar_tensor as st
from jesterTOV.tov.data_classes import EOSData, TOVSolution


@pytest.mark.parametrize("tidal", [False, True])
def test_multiroot_repair_vmap(monkeypatch, tidal):
    def forward(phi, accurate, h0, dh, r0, m0, eos, target):
        x = phi[0]
        return (
            jnp.array([(x - 0.2) * (x - 0.4) * (x - 0.6)]),
            (jnp.array(1.0), jnp.array(1.0)),
            jnp.array(True),
        )

    def solve(pc, beta, target, phi0, ns, ps, hs, es, derivatives, **kwargs):
        x = kwargs["forced_phi"]
        return (
            x * 10,
            jnp.ones_like(x),
            jnp.ones_like(x) if tidal else jnp.nan,
            jnp.ones_like(x),
            jnp.ones_like(x),
            jnp.ones_like(x),
            jnp.ones_like(x),
            x,
        )

    monkeypatch.setattr(st, "_background_forward", forward)
    monkeypatch.setattr(st, "_compiled_tov_solve_impl", solve)
    one = jnp.ones(4)
    eos = EOSData(one, one, one, one, one, one)

    def execute(active):
        sol = TOVSolution(
            jnp.array([1.0, 2.0, 3.0]),
            jnp.ones(3),
            jnp.ones(3),
            {
                key: (
                    jnp.array([0.2, 0.6, 0.2]) if key == "_central_phi" else jnp.ones(3)
                )
                for key in ["lambda_S", "lambda_ST1", "lambda_ST2", "q", "_central_phi"]
            },
        )
        repair = (
            st._repair_multiroot_tidal if tidal else st._repair_multiroot_background
        )
        return repair(active, eos, jnp.ones(3), {"beta_ST": -1.0}, sol)

    result = jax.jit(jax.vmap(execute))(jnp.array([False, True]))
    np.testing.assert_allclose(result.M[0], [1, 2, 3])
    np.testing.assert_allclose(result.M[1], [6, 2, 6], atol=1e-5)
    np.testing.assert_allclose(
        result.extra["_central_phi"][1], [0.6, 0.6, 0.6], atol=1e-5
    )
    nested = jax.jit(jax.vmap(jax.vmap(execute)))(jnp.array([[False, True]]))
    np.testing.assert_allclose(nested.M[0], result.M, atol=1e-5)


def test_missing_root_keeps_original_for_family_validation(monkeypatch):
    def forward(phi, *args):
        return jnp.array([-1.0]), (jnp.array(1.0), jnp.array(1.0)), jnp.array(True)

    monkeypatch.setattr(st, "_background_forward", forward)
    # No candidate is valid; the original jump remains for the family filter.
    monkeypatch.setattr(
        st, "_compiled_tov_solve_impl", lambda *args, **kwargs: (jnp.nan,) * 8
    )
    one = jnp.ones(4)
    eos = EOSData(one, one, one, one, one, one)
    extra = {
        key: jnp.array([0.01, 0.6]) if key == "_central_phi" else jnp.ones(2)
        for key in ["lambda_S", "lambda_ST1", "lambda_ST2", "q", "_central_phi"]
    }
    sol = TOVSolution(jnp.array([1.0, 2.0]), jnp.ones(2), jnp.ones(2), extra)
    actual = jax.jit(st._repair_multiroot_tidal)(
        True, eos, jnp.ones(2), {"beta_ST": -1.0}, sol
    )
    np.testing.assert_array_equal(actual.M, sol.M)
    np.testing.assert_array_equal(
        actual.extra["_central_phi"], sol.extra["_central_phi"]
    )
