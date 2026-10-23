"""Jester's four Tsit5 extensions, applied to an installed Modax factory.

Modax currently has no event or per-lane tolerance hooks. Keep its integration
algorithm in the external package and specialize only these four sites in a
private namespace. Neither the installed files nor Modax's globals are changed.
The factory fingerprint deliberately requires the pinned upstream revision.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import linecache
from pathlib import Path
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable, NamedTuple

import jax


_BODY_FINGERPRINT = "953d25c29c9e2035307e5c6c02e94f4b22d1154d39e72db680b8dc7e05a98056"


class _SurfaceIntegration(NamedTuple):
    """Final state and surface-status flag shared by both ODE backends."""

    state: jax.Array
    reached_surface: jax.Array


def _is_differentiating(*values) -> bool:
    """Whether a JAX AD interpreter, rather than ordinary jit/vmap, is active."""
    return any(
        isinstance(value, jax.core.Tracer)  # type: ignore[reportAttributeAccessIssue]
        and ".interpreters.ad" in type(value).__module__
        for value in jax.tree.leaves(values)
    )


def _ast_fingerprint(node: ast.AST) -> str:
    """Hash syntax consistently across supported Python AST versions."""

    def normalize(value: Any) -> Any:
        if isinstance(value, ast.AST):
            return (
                type(value).__name__,
                [
                    (field, normalize(child))
                    for field, child in ast.iter_fields(value)
                    if field != "type_params" or child
                ],
            )
        if isinstance(value, list):
            return [normalize(child) for child in value]
        return value

    return hashlib.sha256(json.dumps(normalize(node)).encode()).hexdigest()


def _no_sensitivities(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Jester's Modax wrapper supports forward solves only")


def _load_forward_module(name: str, path: Path) -> ModuleType:
    """Load external forward code without importing the unused Enzyme backend."""
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom) and node.module == "modax._sensitivity"
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "modax._jax_common":
            node.module = "jesterTOV.tov._modax._forward_common"
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(
        SensitivitySpec=Any,
        augmented_error_weights=_no_sensitivities,
        augmented_y0=_no_sensitivities,
        make_augmented_local_writer=_no_sensitivities,
        make_sensitivity_solver=_no_sensitivities,
        clear_sensitivity_caches=lambda: None,
    )
    # dataclasses resolves annotations through the defining module's namespace.
    # Only private Jester modules are registered; upstream imports stay intact.
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), vars(module))
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_forward_tsit5() -> ModuleType:
    """Use installed Modax source with the same forward-only scope as before."""
    import modax

    root = Path(modax.__file__).parent
    _load_forward_module(
        "jesterTOV.tov._modax._forward_common", root / "_jax_common.py"
    )
    return _load_forward_module(
        "jesterTOV.tov._modax._forward_tsit5", root / "tsit5.py"
    )


def _adapt_body(source: str) -> str:
    """Specialize the pinned factory without duplicating its ODE algorithm."""
    factory = ast.parse(source).body[0]
    fingerprint = _ast_fingerprint(factory)
    if fingerprint != _BODY_FINGERPRINT:
        raise RuntimeError(
            "Unsupported Modax Tsit5 factory. Install Jester's pinned "
            "modax-cuda12 extra before using the CUDA scalar-tensor backend."
        )

    replacements = (
        (
            "        prow = params[i]\n",
            "        prow = params[i]\n"
            "        for save in range(times.shape[0]):\n"
            "            for j in range(n_system):\n"
            "                hist[i, save, j] = math.nan\n",
        ),
        (
            "        rejected_steps = 0\n",
            "        rejected_steps = 0\n"
            "        cutoff = prow[2]\n"
            "        accuracy_factor = prow[3]\n"
            "        initial_bad = False\n"
            "        for j in range(n_system):\n"
            "            if not math.isfinite(y[j]):\n"
            "                initial_bad = True\n"
            "        if initial_bad or y[1] > cutoff:\n"
            "            for j in range(n_system):\n"
            "                hist[i, n_save - 1, j] = y[j]\n"
            "            accepted_out[i] = 0\n"
            "            rejected_out[i] = 0\n"
            "            loop_out[i] = 0\n"
            "            return\n",
        ),
        (
            "                scale = atol + rtol * max(abs(y[j]), abs(u[j]))\n",
            "                scale = (atol + rtol * max(abs(y[j]), abs(u[j]))) / accuracy_factor\n",
        ),
        (
            "            n_steps += 1\n",
            "            n_steps += 1\n"
            "            if y[1] > cutoff:\n"
            "                for j in range(n_system):\n"
            "                    hist[i, n_save - 1, j] = y[j]\n"
            "                break\n",
        ),
    )
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise RuntimeError("Modax Tsit5 compatibility site no longer matches")
        source = source.replace(original, replacement, 1)
    return source


def make_solve_impl(tsit5: ModuleType) -> Callable:
    """Bind the external solver to Jester's private, cached kernel factories."""
    source = _adapt_body(inspect.getsource(tsit5._make_body))
    namespace = dict(vars(tsit5))
    filename = "<jester_modax_tsit5>"
    # CUDA compiler diagnostics and inspect can still locate the adapted body.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)

    # These functions resolve each other through globals. Clone only this call
    # chain so ordinary Modax callers keep their original kernels and caches.
    for name in ("_make_kernel", "_make_jax_launch", "_solve_impl"):
        original = getattr(tsit5, name)
        function = inspect.unwrap(original)
        cloned = FunctionType(
            function.__code__,
            namespace,
            name,
            function.__defaults__,
            function.__closure__,
        )
        cloned.__kwdefaults__ = function.__kwdefaults__
        cloned.__annotations__ = function.__annotations__
        namespace[name] = (
            functools.cache(cloned) if hasattr(original, "cache_info") else cloned
        )
    return namespace["_solve_impl"]
