r"""Draft meta-model equation of state implementations.

This module contains updated metamodel implementations with:
- Exact proton fraction calculation as default (use proton_fraction='approx' for legacy behavior)
- Optional extra outputs (proton fractions, lepton fractions, DURCA density)
"""

from jesterTOV.eos.metamodel.base import MetaModel_EOS_model
from jesterTOV.eos.metamodel.metamodel_CSE import MetaModel_with_CSE_EOS_model
from jesterTOV.eos.metamodel.metamodel_peakCSE import (
    MetaModel_with_peakCSE_EOS_model,
)
from jesterTOV.eos.metamodel.metamodel_only import MetaModel_only

__all__ = [
    "MetaModel_EOS_model",
    "MetaModel_with_CSE_EOS_model",
    "MetaModel_with_peakCSE_EOS_model",
    "MetaModel_only",
]
