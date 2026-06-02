r"""Likelihood for enforcing direct Urca process activation at a chosen density.

This module implements a likelihood that checks whether the direct Urca neutrino
emission process is active in neutron star matter at a user-specified density.
Direct Urca operates when the proton fraction exceeds a threshold determined by
the lepton fractions. EOS configurations where direct Urca is NOT active at
the target density receive a log-likelihood penalty, while those where it is
active receive log L = 0.

The user can choose between two reference points:

- ``"n_reference"``: central density of a star with a user-configurable mass
  (default 1.4 :math:`M_{\odot}`)
- ``"n_TOV"``: central density of the maximum-mass (TOV) star

References
----------
Lattimer et al., "Direct URCA process in neutron stars," Phys. Rev. Lett. 66,
2701 (1991).
"""

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase

_NPE_XDU = 1.0 / 9.0  # X_DU in pure npe matter (x_e = 1)


class DirectUrcaLikelihood(LikelihoodBase):
    r"""Penalises EOS configurations where direct Urca is NOT active at a chosen density.

    Direct Urca operates when the proton fraction :math:`Y_p` exceeds the
    threshold :math:`X_{\rm DU}`, which depends on the lepton composition:

    .. math::
        X_{\rm DU} = \frac{1}{1 + \bigl(1 + x_e^{1/3}\bigr)^3},
        \qquad
        x_e = \frac{Y_e}{Y_e + Y_\mu}

    When lepton fractions are unavailable (approximate beta-equilibrium),
    pure npe matter (:math:`x_e = 1`) is assumed, giving
    :math:`X_{\rm DU} = 1/9`.

    Parameters
    ----------
    check_type : str
        Which reference density to check:
        - ``"n_reference"``: central density of a star with mass = ``reference_mass``
        - ``"n_TOV"``: central density of the maximum-mass star
    reference_mass : float, optional
        Stellar mass in :math:`M_{\odot}` at which to evaluate when
        ``check_type="n_reference"`` (default: 1.4).
    penalty_value : float, optional
        Log-likelihood penalty returned when :math:`Y_p < X_{\rm DU}` at the
        target density (default: -1e5).

    Examples
    --------
    .. code-block:: yaml

        - type: "direct_urca"
          enabled: true
          check_type: "n_reference"
          reference_mass: 1.4
          penalty_value: -1e5

    Notes
    -----
    Requires ``proton_fraction``, ``e_fraction``, and ``muon_fraction`` arrays
    from the EOS transform (these are populated when the EOS model computes
    beta-equilibrium with muons).  If lepton fractions are missing, the
    likelihood falls back to the npe-matter threshold :math:`X_{\rm DU} = 1/9`.
    """

    check_type: str
    reference_mass: float
    penalty_value: float

    def __init__(
        self,
        check_type: str = "n_reference",
        reference_mass: float = 1.4,
        penalty_value: float = -1e5,
    ) -> None:
        super().__init__()
        self.check_type = check_type
        self.reference_mass = reference_mass
        self.penalty_value = float(penalty_value)

    @staticmethod
    def _compute_xdu(
        proton_fraction: Float[Array, " n"],
        e_fraction: Float[Array, " n"] | None,
        muon_fraction: Float[Array, " n"] | None,
    ) -> Float[Array, " n"]:
        r"""Compute the direct Urca threshold :math:`X_{\rm DU}` on the density grid.

        When lepton fractions are available:

        .. math::
            x_e = \frac{Y_e}{Y_e + Y_\mu},
            \qquad
            X_{\rm DU} = \frac{1}{1 + (1 + x_e^{1/3})^3}

        When lepton fractions are unavailable (or ``None``), pure npe matter
        is assumed, giving :math:`X_{\rm DU} = 1/9`.

        Parameters
        ----------
        proton_fraction : array
            Proton fraction array (dimensionless).
        e_fraction : array or None
            Electron fraction array, or ``None`` if not computed.
        muon_fraction : array or None
            Muon fraction array, or ``None`` if not computed.

        Returns
        -------
        array
            :math:`X_{\rm DU}` threshold at each density point.
        """
        # Check if lepton fractions are available
        if e_fraction is not None and muon_fraction is not None:
            x_e = e_fraction / (e_fraction + muon_fraction + 1e-30)
            xdu = 1.0 / (1.0 + (1.0 + jnp.cbrt(x_e)) ** 3)
        else:
            # Pure npe matter: x_e = 1 → X_DU = 1/9
            xdu = jnp.full_like(proton_fraction, _NPE_XDU)
        return xdu

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the direct Urca log-likelihood.

        Returns 0.0 if :math:`Y_p \ge X_{\rm DU}` at the target density, or
        ``penalty_value`` otherwise.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing EOS and TOV quantities from the transform.
            Required keys:
            - ``"proton_fraction"``: proton fraction on density grid
            - ``"n_orig"``: density grid in fm\ :sup:`-3`
            - ``"masses_EOS"``: neutron star masses (:math:`M_{\odot}`)
            - ``"logpc_EOS"``: log\ :sub:`10` of central pressures (geometric)
            - ``"n"``: density grid (geometric units)
            - ``"p"``: pressure grid (geometric units)

            Optional keys:
            - ``"e_fraction"``: electron fraction (if available)
            - ``"muon_fraction"``: muon fraction (if available)
            - ``"n_TOV"``: central density at M\ :sub:`TOV` (geometric, only
              needed for ``check_type="n_TOV"``)

        Returns
        -------
        Float
            0.0 if direct Urca is active at the target density, else
            ``penalty_value``.
        """
        # Extract arrays from params
        proton_fraction: Float[Array, " n"] = params["proton_fraction"]
        n_orig: Float[Array, " n"] = params["n_orig"]

        # Lepton fractions may be None if not computed by the EOS
        e_fraction: Float[Array, " n"] | None = params.get("e_fraction")  # type: ignore[assignment]
        muon_fraction: Float[Array, " n"] | None = params.get("muon_fraction")  # type: ignore[assignment]

        # Compute X_DU on the full density grid
        xdu = self._compute_xdu(proton_fraction, e_fraction, muon_fraction)

        # Determine target density in fm^-3
        if self.check_type == "n_reference":
            # Central pressure log10 at the reference mass
            masses_eos: Float[Array, " n"] = params["masses_EOS"]
            logpc_eos: Float[Array, " n"] = params["logpc_EOS"]
            pc_ref_log10: Float = jnp.interp(self.reference_mass, masses_eos, logpc_eos)
            pc_ref_geom: Float = 10.0**pc_ref_log10  # geometric units

            # Interpolate density at that central pressure
            n_grid: Float[Array, " n"] = params["n"]
            p_grid: Float[Array, " n"] = params["p"]
            n_ref_geom: Float = jnp.interp(pc_ref_geom, p_grid, n_grid)  # geometric
            n_target_fm3: Float = n_ref_geom * utils.geometric_to_fm_inv3  # fm^-3

        elif self.check_type == "n_TOV":
            # n_TOV is already computed by the transform (in geometric units)
            n_tov_geom: Float = params["n_TOV"]  # type: ignore[assignment]
            n_target_fm3: Float = n_tov_geom * utils.geometric_to_fm_inv3  # fm^-3

        else:
            raise ValueError(
                f"Unknown check_type: '{self.check_type}'. "
                "Expected 'n_reference' or 'n_TOV'."
            )

        # Interpolate Y_p and X_DU at the target density
        # n_orig is in fm^-3, n_target_fm3 is in fm^-3 → same units
        yp_target: Float = jnp.interp(n_target_fm3, n_orig, proton_fraction)
        xdu_target: Float = jnp.interp(n_target_fm3, n_orig, xdu)

        # Apply penalty if Y_p < X_DU (direct Urca NOT active)
        log_likelihood = jnp.where(
            yp_target >= xdu_target,
            0.0,  # Direct Urca active → no penalty
            self.penalty_value,  # Direct Urca inactive → apply penalty
        )

        # Safety net for NaN from interpolation (e.g., target outside grid)
        log_likelihood = jnp.nan_to_num(
            log_likelihood,
            nan=self.penalty_value,
            posinf=self.penalty_value,
            neginf=self.penalty_value,
        )

        return log_likelihood