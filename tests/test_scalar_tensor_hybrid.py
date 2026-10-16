"""Small regressions for ST fallback and whole-family failure handling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import FamilyData
from jesterTOV.tov.scalar_tensor import ScalarTensorTOVSolver, _recover_background


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
