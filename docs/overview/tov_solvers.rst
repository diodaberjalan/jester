.. _overview-tov:

TOV solvers
===========

JESTER provides TOV (Tolman-Oppenheimer-Volkoff) equation solvers to compute neutron star structure from a given equation of state. All solvers are JAX-accelerated and support automatic differentiation.

Stability-region selection
--------------------------

By default, family construction preserves the existing bounded stable-segment
detection. The optional YAML setting ``tov.stability_branch: first`` retains
only the first detected stable interval through its maximum-mass endpoint.
This gives current and future MR and tidal likelihoods, maximum-mass constraints,
and posterior curves the same selected family. The EOS and solver-failure
checks are unchanged. Omitting the setting or using ``default`` preserves the
existing behavior.

The Python API exposes the same static option on
:meth:`~jesterTOV.tov.base.TOVSolverBase.construct_family` and
:class:`~jesterTOV.inference.transforms.transform.JesterTransform`.

General Relativity
------------------

Standard TOV solver in general relativity.

:doc:`tov/gr`

Pressure Anisotropy
-------------------

Post-TOV solver including anisotropic effects.

:doc:`tov/anisotropy`

Modified Gravity
----------------

Scalar-tensor theories and alternative gravity frameworks.

:doc:`tov/scalar_tensor`

.. toctree::
   :hidden:

   tov/gr
   tov/scalar_tensor
   tov/anisotropy
