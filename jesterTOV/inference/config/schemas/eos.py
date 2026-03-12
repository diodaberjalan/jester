"""Pydantic models for EOS configuration."""

from typing import Literal, Union, Annotated
from pydantic import field_validator, ConfigDict, Discriminator

from ._base import JesterBaseModel


class BaseEOSConfig(JesterBaseModel):
    """Base configuration shared by all EOS types.

    Attributes
    ----------
    crust_name : Literal["DH", "BPS", "DH_fixed", "SLy"]
        Name of crust model to use (default: "DH")
    """

    model_config = ConfigDict(extra="forbid")

    crust_name: Literal["DH", "BPS", "DH_fixed", "SLy"] = "DH"


class BaseMetamodelEOSConfig(BaseEOSConfig):
    """Base configuration shared by all MetaModel-based EOS types.

    Holds the grid parameters that control the metamodel density grid.
    This base class is inherited by :class:`MetamodelEOSConfig` and
    :class:`MetamodelCSEEOSConfig` but not by the spectral EOS, which
    has a different parameterization.

    Attributes
    ----------
    ndat_metamodel : int
        Number of data points for MetaModel EOS grid (default: 100)
    nmax_nsat : float
        Maximum density in units of saturation density (default: 25.0)
    nmin_MM_nsat : float
        Starting density for metamodel grid as fraction of nsat (default: 0.75)
    """

    ndat_metamodel: int = 100
    nmax_nsat: float = 25.0
    nmin_MM_nsat: float = 0.75


class MetamodelEOSConfig(BaseMetamodelEOSConfig):
    """Configuration for MetaModel EOS (without CSE).

    Attributes
    ----------
    type : Literal["metamodel"]
        EOS type identifier
    nb_CSE : int
        Must be 0 for standard metamodel (no CSE extension)
    """

    type: Literal["metamodel"] = "metamodel"
    nb_CSE: int = 0

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for standard metamodel."""
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='metamodel'. "
                "Use type='metamodel_cse' for CSE extension."
            )
        return v


class MetamodelCSEEOSConfig(BaseMetamodelEOSConfig):
    """Configuration for MetaModel with CSE extension.

    Attributes
    ----------
    type : Literal["metamodel_cse"]
        EOS type identifier
    nb_CSE : int
        Number of CSE parameters (must be > 0, typically 4-8)
    ndat_CSE : int
        Number of density grid points for the CSE region (default: 100)
    max_nbreak_nsat : float | None
        Maximum allowed breaking density in units of nsat (default: None,
        meaning no upper bound beyond the prior). If specified, this must
        be consistent with the upper bound of the ``nbreak`` prior; an
        error is raised if they disagree.
    """

    type: Literal["metamodel_cse"] = "metamodel_cse"
    nb_CSE: int = 8
    ndat_CSE: int = 100
    max_nbreak_nsat: float | None = None

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is positive for metamodel_cse."""
        if v <= 0:
            raise ValueError(
                "nb_CSE must be > 0 for type='metamodel_cse'. "
                "Use type='metamodel' for standard metamodel without CSE."
            )
        return v


class SpectralEOSConfig(BaseEOSConfig):
    r"""Configuration for Spectral Decomposition EOS.

    Attributes
    ----------
    type : Literal["spectral"]
        EOS type identifier
    n_points_high : int
        Number of high-density points for spectral EOS (default: 500)
    nb_CSE : int
        Must be 0 for spectral (no CSE support)
    reparametrized : bool
        If False (default), sample directly in :math:`(\gamma_0, \gamma_1, \gamma_2, \gamma_3)`.
        If True, sample in a whitened space :math:`(\tilde{\gamma}_0, \tilde{\gamma}_1, \tilde{\gamma}_2, \tilde{\gamma}_3)`
        centred on a Gaussian fit to a radio-timing inference result.  The bijection
        :math:`\boldsymbol{\gamma} = \boldsymbol{\mu} + L_\text{wide}\,\tilde{\boldsymbol{\gamma}}` maps the
        unit-normal tilde parameters back to physical spectral coefficients, where
        :math:`L_\text{wide} = \sigma_\text{scale}\,L` and :math:`\boldsymbol{\mu}` is
        the posterior mean.  Use a ``MultivariateGaussianPrior`` with default (unit)
        parameters in the prior file when this option is enabled.
    sigma_scale : float
        Multiplicative factor applied to the base Cholesky factor :math:`L` to form
        :math:`L_\text{wide} = \sigma_\text{scale}\,L`.  Only used when
        ``reparametrized=True``.  Default 1.0 (exact radio posterior covariance).
        Increase to widen the prior around the radio posterior.
    """

    type: Literal["spectral"] = "spectral"
    n_points_high: int = 500
    nb_CSE: int = 0
    reparametrized: bool = False
    sigma_scale: float = 1.0

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for spectral."""
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='spectral'. "
                "CSE extension not supported for spectral EOS."
            )
        return v


class BaseSkyrmeEOSConfig(BaseEOSConfig):
    r"""Base configuration shared by all Skyrme-based EOS types.

    The Skyrme EOS uses infinite nuclear matter (INM) parameters that are
    converted to Skyrme force parameters through an inverse problem solver.

    Attributes
    ----------
    ndat_skyrme : int
        Number of density points for Skyrme EOS grid (default: 100)
    nmax_nsat : float
        Maximum density in units of saturation density (default: 12.0)
    nmin_Skyrme_nsat : float
        Starting density for Skyrme grid as fraction of nsat (default: 0.75)
    """

    ndat_skyrme: int = 100
    nmax_nsat: float = 12.0
    nmin_Skyrme_nsat: float = 0.75


class SkyrmeEOSConfig(BaseSkyrmeEOSConfig):
    r"""Configuration for Skyrme EOS (without CSE).

    The Skyrme EOS implements the energy density functional approach using
    INM parameters that are solved for Skyrme force parameters.

    Attributes
    ----------
    type : Literal["skyrme"]
        EOS type identifier
    nb_CSE : int
        Must be 0 for standard skyrme (no CSE extension)
    proton_fraction : str | float | None
        Proton fraction treatment: None (exact with muons), "approx"
        (without muons), or a fixed float value.
    """

    type: Literal["skyrme"] = "skyrme"
    nb_CSE: int = 0
    proton_fraction: str | float | None = None

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is 0 for standard skyrme."""
        if v != 0:
            raise ValueError(
                "nb_CSE must be 0 for type='skyrme'. "
                "Use type='skyrme_cse' for CSE extension."
            )
        return v


class SkyrmeCSEEOSConfig(BaseSkyrmeEOSConfig):
    r"""Configuration for Skyrme EOS with CSE extension.

    Combines the Skyrme approach for low-to-intermediate densities with
    piecewise constant speed-of-sound extensions at high densities.

    Attributes
    ----------
    type : Literal["skyrme_cse"]
        EOS type identifier
    nb_CSE : int
        Number of CSE parameters (must be > 0, typically 4-8)
    ndat_CSE : int
        Number of density grid points for the CSE region (default: 100)
    max_nbreak_nsat : float | None
        Maximum allowed breaking density in units of nsat (default: None).
        Should match the upper bound of the nbreak prior.
    proton_fraction : str | float | None
        Proton fraction treatment: None (exact with muons), "approx"
        (without muons), or a fixed float value.
    """

    type: Literal["skyrme_cse"] = "skyrme_cse"
    nb_CSE: int = 8
    ndat_CSE: int = 100
    max_nbreak_nsat: float | None = None
    proton_fraction: str | float | None = None

    @field_validator("nb_CSE")
    @classmethod
    def validate_nb_cse(cls, v: int) -> int:
        """Validate that nb_CSE is positive for skyrme_cse."""
        if v <= 0:
            raise ValueError(
                "nb_CSE must be > 0 for type='skyrme_cse'. "
                "Use type='skyrme' for standard Skyrme without CSE."
            )
        return v


# Discriminated union of all EOS types
EOSConfig = Annotated[
    Union[
        MetamodelEOSConfig,
        MetamodelCSEEOSConfig,
        SpectralEOSConfig,
        SkyrmeEOSConfig,
        SkyrmeCSEEOSConfig,
    ],
    Discriminator("type"),
]
