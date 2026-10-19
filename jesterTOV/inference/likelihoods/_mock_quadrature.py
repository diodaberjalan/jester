"""Streaming quadrature for analytic mock densities.

Observation blocks bound temporary storage under an outer particle vmap. Grids,
weights, masks and penalties match the original mock likelihoods; arithmetic and
reduction ordering may differ at floating-point roundoff.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from jaxtyping import Array, Float

from jesterTOV.utils import get_MR_split_index


class _ObservationQuadrature:
    """Prepare fixed observation coefficients outside JAX tracing."""

    def __init__(self, reference: Any, observation_batch_size: int = 16) -> None:
        if not isinstance(observation_batch_size, int) or observation_batch_size < 1:
            raise ValueError("observation_batch_size must be a positive integer")
        self.K: int = reference.K
        if self.K < 1 or reference.N_masses_evaluation < 2:
            raise ValueError("Need at least one observation and two quadrature points")
        self.penalty_value: float = reference.penalty_value
        self.N_masses_evaluation: int = reference.N_masses_evaluation
        self.y_key: str = reference.y_key
        self.observation_batch_size = min(observation_batch_size, self.K)
        self.zero_skew = bool(np.all(np.asarray(reference.alpha_primes) == 0))
        inv = np.asarray(reference.inv_covs)
        self.diagonal = bool(np.all(inv[:, 0, 1] == 0) and np.all(inv[:, 1, 0] == 0))
        columns = dict(
            cx=np.asarray(reference.centers)[:, 0],
            cy=np.asarray(reference.centers)[:, 1],
            a=inv[:, 0, 0],
            b=inv[:, 0, 1],
            c=inv[:, 1, 0],
            d=inv[:, 1, 1],
            logdet=np.asarray(reference.log_det_covs),
            alpha_x=np.asarray(reference.alpha_primes)[:, 0],
            alpha_y=np.asarray(reference.alpha_primes)[:, 1],
            mass_std=np.asarray(reference.omegas)[:, 0],
        )
        for name, values in columns.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite observation coefficient: {name}")
        width = self.observation_batch_size
        padding = (-self.K) % width
        self.chunks = {
            name: jnp.asarray(
                np.pad(values, (0, padding), mode="edge").reshape(-1, width)
            )
            for name, values in columns.items()
        }
        self.chunks["active"] = jnp.asarray(
            (np.arange(self.K + padding) < self.K).reshape(-1, width)
        )

    def _branches(
        self, params: dict[str, Array], tidal: bool
    ) -> tuple[Array, list[tuple[Array, Array, Array, Array]]]:
        masses, y = params["masses_EOS"], params[self.y_key]
        valid = jnp.isfinite(masses) & jnp.isfinite(y)
        mtov = jnp.max(jnp.where(valid, masses, self.penalty_value))
        split = get_MR_split_index(masses, y)
        indices = jnp.arange(masses.shape[0])
        branches = []
        for mask in ((indices < split) & valid, (indices >= split) & valid):
            dummy = jnp.linspace(1e5, 1e5 + 10.0, masses.shape[0]) if tidal else 1e30
            safe_m = jnp.where(mask, masses, dummy)
            safe_y = jnp.where(mask, y, 0.0)
            order = jnp.argsort(safe_m)
            safe_m, safe_y = safe_m[order], safe_y[order]
            if tidal:
                lower = jnp.min(jnp.where(mask, masses, 1e30))
                upper = jnp.max(jnp.where(mask, masses, self.penalty_value))
            else:
                lower = safe_m[0]
                upper = jnp.max(jnp.where(safe_m == 1e30, self.penalty_value, safe_m))
            shifted = safe_m + jnp.arange(safe_m.shape[0]) * 1e-12
            branches.append((shifted, safe_y, lower, upper))
        return mtov, branches

    def _log_density(self, m: Array, y: Array, obs: dict[str, Array]) -> Array:
        dx, dy = m - obs["cx"][:, None], y - obs["cy"][:, None]
        quad = obs["a"][:, None] * dx**2 + obs["d"][:, None] * dy**2
        if not self.diagonal:
            quad = quad + (obs["b"] + obs["c"])[:, None] * dx * dy
        density = -0.5 * (obs["logdet"][:, None] + quad + 2 * jnp.log(2 * jnp.pi))
        if self.zero_skew:
            # The reference's zero*nonfinite coordinate gives NaN before masking.
            return jnp.where(jnp.isfinite(dx) & jnp.isfinite(dy), density, jnp.nan)
        skew = obs["alpha_x"][:, None] * dx + obs["alpha_y"][:, None] * dy
        return density + jnp.log(2.0) + norm.logcdf(skew)


class MockMRQuadrature(_ObservationQuadrature):
    """Reduce the original M-R quadrature over bounded observation blocks."""

    def evaluate(self, params: dict[str, Array]) -> Float:
        """Return the summed log likelihood for the supplied stellar curve."""
        mtov, branches = self._branches(params, tidal=False)
        grid = jnp.linspace(0.1, 3.5, self.N_masses_evaluation)
        dm = grid[1] - grid[0]
        prepared = [
            (jnp.interp(grid, m, y), lower, upper) for m, y, lower, upper in branches
        ]
        mass_penalty = jnp.where(grid > mtov, self.penalty_value, 0.0)

        def step(total: Array, obs: dict[str, Array]) -> tuple[Array, None]:
            probabilities = []
            for y, lower, upper in prepared:
                density = self._log_density(grid, y, obs)
                density = jnp.where(
                    ((grid >= lower) & (grid <= upper))[None, :],
                    density,
                    self.penalty_value,
                )
                probabilities.append(density + mass_penalty[None, :])
            combined = jnp.logaddexp(probabilities[0], probabilities[1])
            individual = logsumexp(combined, axis=1) + jnp.log(dm)
            return total + jnp.sum(jnp.where(obs["active"], individual, 0.0)), None

        dtype = jnp.result_type(params["masses_EOS"], self.chunks["a"])
        total, _ = jax.lax.scan(step, jnp.zeros((), dtype=dtype), self.chunks)
        return total


class MockLambdaQuadrature(_ObservationQuadrature):
    """Retain component-specific tidal grids while streaming observations."""

    def __init__(self, reference: Any, observation_batch_size: int = 16) -> None:
        super().__init__(reference, observation_batch_size)
        self.integration_sigma_cut: float = reference.integration_sigma_cut

    def evaluate(self, params: dict[str, Array]) -> Float:
        """Return the summed log likelihood for the supplied stellar curve."""
        mtov, branches = self._branches(params, tidal=True)
        grid = jnp.linspace(0.0, 1.0, self.N_masses_evaluation)
        weights = jnp.ones(self.N_masses_evaluation).at[0].set(0.5).at[-1].set(0.5)
        log_weights = jnp.log(weights)

        def step(total: Array, obs: dict[str, Array]) -> tuple[Array, None]:
            integrals = []
            for m, y, lower, upper in branches:
                window = self.integration_sigma_cut * obs["mass_std"]
                start = jnp.maximum(lower, obs["cx"] - window)
                stop = jnp.minimum(upper, obs["cx"] + window)
                support = stop > start
                m_grid = start[:, None] + (stop - start)[:, None] * grid[None, :]
                y_grid = jnp.interp(m_grid, m, y)
                density = self._log_density(m_grid, y_grid, obs)
                inside = (m_grid >= lower) & (m_grid <= upper) & support[:, None]
                density = jnp.where(inside, density, self.penalty_value)
                density = density + jnp.where(m_grid > mtov, self.penalty_value, 0.0)
                dm = (stop - start) / (self.N_masses_evaluation - 1)
                quadrature = jnp.log(dm[:, None]) + log_weights[None, :]
                quadrature = jnp.where(support[:, None], quadrature, self.penalty_value)
                integrals.append(logsumexp(density + quadrature, axis=1))
            individual = jnp.logaddexp(integrals[0], integrals[1])
            return total + jnp.sum(jnp.where(obs["active"], individual, 0.0)), None

        dtype = jnp.result_type(params["masses_EOS"], self.chunks["a"])
        total, _ = jax.lax.scan(step, jnp.zeros((), dtype=dtype), self.chunks)
        return total
