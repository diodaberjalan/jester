"""Forward-only Modax adapter for scalar-tensor surface integrations."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jax.extend import core
from jax.interpreters import batching, mlir
from modax._numba_common import PIDGains, SolveSettings
from .compat import _SurfaceIntegration, load_forward_tsit5, make_solve_impl
from .rhs import background_rhs, tidal_rhs


tsit5 = load_forward_tsit5()
_solve_impl = make_solve_impl(tsit5)


class Result(eqx.Module):
    _value: jax.Array


class Solution(NamedTuple):
    ys: tuple
    ts: jax.Array
    result: Result
    stats: dict


def make_solver(rhs, rtol, atol, max_steps):
    def implementation(y, params):
        leading = y.shape[:-1]
        flat_y = y.reshape((-1, y.shape[-1]))
        flat_p = params.reshape((-1, params.shape[-1]))
        history, stats = _solve_impl(
            rhs,
            flat_y,
            jnp.array([0.0, 1.0]),
            flat_p,
            settings=SolveSettings(rtol, atol, 1 / 999, max_steps, None),
            options=tsit5.KernelOptions(PIDGains(0.0, 1.0, 0.0), "local"),
            return_stats=True,
        )
        final = history[:, -1, :].reshape(leading + (y.shape[-1],))
        stats = jax.tree.map(lambda x: x.reshape(leading), stats)
        return (
            final,
            stats["accepted_steps"],
            stats["rejected_steps"],
            stats["loop_steps"],
        )

    primitive = core.Primitive("st_modax_" + rhs.__name__)
    primitive.multiple_results = True
    primitive.def_impl(implementation)
    primitive.def_abstract_eval(
        lambda y, p: (
            (y,)
            + tuple(
                jax.core.ShapedArray(y.shape[:-1], jnp.int32)  # type: ignore[reportAttributeAccessIssue]
                for _ in range(3)
            )
        )
    )
    mlir.register_lowering(
        primitive,
        mlir.lower_fun(implementation, multiple_results=True),
        platform="cuda",
    )

    def batched(args, dims):
        size = next(x.shape[d] for x, d in zip(args, dims) if d is not None)
        moved = [batching.bdim_at_front(x, d, size) for x, d in zip(args, dims)]
        return primitive.bind(*moved), (0, 0, 0, 0)

    batching.primitive_batchers[primitive] = batched

    def integrate(y, params):
        final, accepted, rejected, steps = primitive.bind(y, params)
        return final, dict(
            accepted_steps=accepted, rejected_steps=rejected, loop_steps=steps
        )

    return integrate


background = make_solver(background_rhs, 1e-10, 1e-11, 3000)
tidal = make_solver(tidal_rhs, 1e-10, 1e-11, 3000)


def integrate(y0, eos, h0, *, accurate=False, is_tidal=False, mass_limit=float("inf")):
    """Integrate one background or tidal state through normalized enthalpy."""
    y = jnp.stack(y0)
    parameters = jnp.concatenate(
        (
            jnp.array([eos["beta_ST"], h0, mass_limit, jnp.where(accurate, 10.0, 1.0)]),
            jnp.log(eos["h"]),
            jnp.log(eos["p"]),
            jnp.log(eos["e"]),
            eos["dloge_dlogp"],
        )
    )
    final, stats = (tidal if is_tidal else background)(y, parameters)
    valid = jnp.all(jnp.isfinite(final)) & (final[1] <= mass_limit)
    valid &= jnp.isfinite(h0) & (h0 > 0)
    ys = tuple(value[None] for value in final)
    return Solution(
        ys,
        jnp.array([jnp.where(valid, 0.0, jnp.nan)]),
        Result(jnp.where(valid, 0, 1)),
        stats,
    )


def surface(y0, h0, dh, eos_dict, accurate, mass_limit, kind) -> _SurfaceIntegration:
    """Forward-only Modax integration normalized to the shared result type."""
    solution = integrate(
        y0,
        eos_dict,
        h0,
        accurate=accurate,
        is_tidal=(kind == "tidal"),
        mass_limit=mass_limit if kind != "no_tidal" else float("inf"),
    )
    return _SurfaceIntegration(
        jnp.concatenate(solution.ys),
        (solution.result._value == 0) & (solution.ts[-1] == 0),
    )
