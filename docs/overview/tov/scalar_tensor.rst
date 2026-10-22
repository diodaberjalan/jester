.. _tov-scalar-tensor:

Scalar-Tensor TOV Solver
========================

[Placeholder: Detailed explanation of modified gravity TOV solver]

This page will cover:

* Scalar-tensor theories of gravity
* Modified TOV equations
* Additional free parameters
* Physical implications for neutron star structure
* Usage examples
* API reference: :class:`jesterTOV.tov.scalar_tensor.ScalarTensorTOVSolver`

**Note**: This solver is currently work in progress.


Shooting initialization and recovery
------------------------------------

The neural-network prediction seeds a local shooting correction with up to
15 evaluations, using five damped updates followed by safeguarded secant.
If increasing the central field makes a negative residual more negative,
four additional probes at 0.001, 0.01, 0.1 and 0.5 are evaluated once.
Invalid integrations are excluded from endpoint selection. A positive probe
is paired with the largest-field valid negative endpoint; one geometric
midpoint is evaluated before secant resumes. With only negative probes, the
largest valid field starts a fresh correction with exactly one damped update.

For a negative-only search whose valid probe slopes decrease, the search
continues towards larger central fields until it finds a positive residual
or converges. If it encounters a failed integration, queued recovery can reuse
the largest valid negative field and its residual as a lower endpoint.
This restricts the upward crossing being tracked; other roots may exist below
that endpoint. If all four probes fail, recovery is skipped for that point.
Existing final-trial mass-cutoff eligibility is unchanged.

Recovery retains the original grid above an enabled lower endpoint, reuses
its cached residual, and evaluates points in parallel tiles of 16. It stops
after the first tile containing an adjacent valid sign bracket and selects
the first such bracket, including brackets crossing tile boundaries.
An earlier valid/invalid boundary cannot stop the grid scan because a later
sign bracket has priority. If no sign bracket exists, the first eligible
boundary receives up to 17 refinements; root bisection allows 40 evaluations.
The 17-refinement cap retained recovery coverage in the validation datasets;
it does not guarantee coverage for every unseen EOS. Completed batched lanes
are masked before evaluation to avoid repeated ODE integrations.

Recovery uses eight concurrent star slots for one family. Nested batches of
EOS families pool recovery requests across EOS and central-pressure axes,
using eight slots per EOS to reduce idle lanes. This changes execution layout
and can introduce small floating-point differences. Points after a bracket
within its tile may also be evaluated; they do not change bracket selection.

The Diffrax fallback retains relative tolerance 1e-5, absolute tolerance 1e-6
and a 500-step ceiling for ordinary background integrations. Temporary accurate
local evaluations, the residual threshold 1e-4, final tidal integration and
strict family-failure checks are unchanged. Existing YAML configurations require
no additional keys.

Optional CUDA forward integrator
-------------------------------

Installing both the ``cuda12`` and ``modax-cuda12`` optional dependencies
enables the bundled, forward-only Modax Tsit5 CUDA backend for scalar-tensor
background and tidal integrations. Its validated forward tolerances are
1e-10/1e-11, with a 3000-step ceiling. The shooting, recovery, EOS
interpolation, and exterior matching algorithms remain unchanged. Jester
automatically keeps Diffrax on
CPU, when the optional compiler dependencies are unavailable, and while JAX is
tracing a differentiated background integration. Full scalar-tensor shooting
does not currently support JAX automatic differentiation with either backend.
