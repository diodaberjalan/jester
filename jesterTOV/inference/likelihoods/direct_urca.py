r"""Direct-Urca trigger-mass likelihoods.

This module implements trigger-mass likelihoods developed in
``Sandbox/durca_likelihood_playground/direct_urca.py``.  The observable is the
stellar mass at which fast cooling first turns on, :math:`m_{\rm trig}`.  The
upper-bound likelihood is the product of survival functions,

.. math::

    \mathcal{L}_{\rm upper}(m_{\rm trig})
    = \prod_i \left[1 - F_i(m_{\rm trig})\right].

The lower-bound likelihood uses a non-rapid-cooling object and evaluates

.. math::

    \mathcal{L}_{\rm lower}(m_{\rm trig}) = F_{\rm HESS}(m_{\rm trig}).

Three physical assumptions are supported:

``"durca_only"``
    :math:`m_{\rm trig}=m_{\rm dUrca}` only if direct Urca turns on below
    :math:`M_{\rm TOV}`.  If a CSE transition density ``nbreak`` exists,
    direct Urca must turn on before the CSE transition.

``"durca_or_cse"``
    When :math:`n_{\rm dUrca} \leq n_{\rm break}`, direct Urca triggers in the
    nucleonic branch and :math:`m_{\rm trig} = M(n_{\rm dUrca})`.  When
    :math:`n_{\rm dUrca} > n_{\rm break}`, the cooling density in the CSE
    branch is unknown and is marginalized over a uniform prior between
    :math:`\max(n_{\rm break}, n^*_{\rm min})` and :math:`n^*_{\rm max}`,
    together with a uniform prior on the stellar central density :math:`n^*`
    between :math:`n^*_{\rm min}` and :math:`n^*_{\rm max}` (with the
    constraint :math:`n_{\rm cool} \leq n^*`).

    .. math::

        \mathcal{L} = \int_{n^*_{\rm min}}^{n^*_{\rm max}}
        \int_{n_{\rm break}}^{n^*}
        \mathcal{L}_{\rm mtrig}\!\bigl(M(n_{\rm cool})\bigr) \;
        \frac{1}{n^* - n_{\rm break}} \;
        \frac{1}{n^*_{\rm max} - n^*_{\rm min}} \;
        dn_{\rm cool} \, dn^*.

``"durca_or_cse_simple"`` (legacy)
    :math:`m_{\rm trig}` is the lower-mass trigger from direct Urca or the CSE
    transition.  This assumes the CSE branch itself always allows direct Urca,
    so if ``n_durca > nbreak`` the trigger is ``nbreak``.  Retained for
    bookkeeping comparisons against the marginalized formulation.
"""

from typing import Literal

import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase

_NPE_XDU = 1.0 / 9.0
_N_SAT = 0.16  # nuclear saturation density [fm⁻³]
TriggerAssumption = Literal["durca_only", "durca_or_cse", "durca_or_cse_simple"]


class DirectUrcaLikelihood(LikelihoodBase):
    r"""Upper-bound likelihood for the direct-Urca or CSE trigger mass.

    Parameters
    ----------
    trigger_assumption : {"durca_only", "durca_or_cse"}
        Rule used to convert the direct-Urca onset density and optional CSE
        transition density into :math:`m_{\rm trig}`.
    name : str
        Identifier for this likelihood.
    penalty_value : float
        Log-likelihood returned when no valid trigger exists below
        :math:`M_{\rm TOV}` or required EOS quantities are absent.

    Notes
    -----
    The transform output must contain ``n_orig`` and ``proton_fraction``.  For
    the metamodel/skyrme EOS families this requires ``calculate_durca: true``.
    Electron and muon fractions are used when present; otherwise pure npe
    matter gives :math:`X_{\rm DU}=1/9`.
    """

    def __init__(
        self,
        trigger_assumption: TriggerAssumption = "durca_only",
        name: str = "Direct_Urca_Trigger_Mass",
        penalty_value: float = -1e5,
        nstar_min_nsat: float = 4.0,
        nstar_max_nsat: float = 10.0,
        nb_ncool: int = 400,
        nb_nstar: int = 200,
        **legacy_kwargs: object,
    ) -> None:
        super().__init__()
        if trigger_assumption not in (
            "durca_only",
            "durca_or_cse",
            "durca_or_cse_simple",
        ):
            raise ValueError(
                "trigger_assumption must be 'durca_only', 'durca_or_cse', "
                "or 'durca_or_cse_simple'"
            )
        self.trigger_assumption = trigger_assumption
        self.name = name
        self.penalty_value = float(penalty_value)
        self.nstar_min_nsat = float(nstar_min_nsat)
        self.nstar_max_nsat = float(nstar_max_nsat)
        self.nb_ncool = int(nb_ncool)
        self.nb_nstar = int(nb_nstar)
        self.legacy_kwargs = legacy_kwargs

        # SAX J1808.4-3658 Gaussian mixture
        self.sax_mu = jnp.array([1.46, 1.98, 1.93, 2.00])
        self.sax_sig = jnp.array([0.175, 0.130, 0.085, 0.060])
        self.sax_w = jnp.array([0.25, 0.25, 0.25, 0.25])

        # Cas A Gaussian
        self.cas_mu = 1.55
        self.cas_sig = 0.25

        # PSR B2334+61 uniform upper-limit distribution
        self.b2334_loc = 1.45
        self.b2334_scale = 1.60 - 1.45

        # Vela uniform distribution; upper support is M_TOV.
        self.vela_loc = 1.40

    @staticmethod
    def _compute_xdu(
        proton_fraction: Float[Array, " n"],
        e_fraction: Float[Array, " n"] | None,
        muon_fraction: Float[Array, " n"] | None,
    ) -> Float[Array, " n"]:
        r"""Compute the direct-Urca threshold :math:`X_{\rm DU}`."""
        if e_fraction is None or muon_fraction is None:
            return jnp.full_like(proton_fraction, _NPE_XDU)

        x_e = e_fraction / (e_fraction + muon_fraction + 1e-30)
        return 1.0 / (1.0 + (1.0 + jnp.cbrt(x_e)) ** 3)

    def _find_n_durca(self, params: dict[str, Float | Array]) -> Float:
        r"""Return first direct-Urca onset density in fm\ :sup:`-3`."""
        proton_fraction: Float[Array, " n"] = params["proton_fraction"]  # type: ignore[assignment]
        n_orig: Float[Array, " n"] = params["n_orig"]  # type: ignore[assignment]
        e_fraction: Float[Array, " n"] | None = params.get("e_fraction")  # type: ignore[assignment]
        muon_fraction: Float[Array, " n"] | None = params.get("muon_fraction")  # type: ignore[assignment]

        xdu = self._compute_xdu(proton_fraction, e_fraction, muon_fraction)
        active = proton_fraction >= xdu
        first_idx = jnp.argmax(active)
        return jnp.where(jnp.any(active), n_orig[first_idx], jnp.inf)

    @staticmethod
    def _mass_at_density(
        n_trigger_fm3: Float,
        params: dict[str, Float | Array],
    ) -> Float:
        r"""Interpolate stellar mass for a central density in fm\ :sup:`-3`."""
        n_grid_geom: Float[Array, " n"] = params["n"]  # type: ignore[assignment]
        p_grid_geom: Float[Array, " n"] = params["p"]  # type: ignore[assignment]
        logpc_eos: Float[Array, " n"] = params["logpc_EOS"]  # type: ignore[assignment]
        masses_eos: Float[Array, " n"] = params["masses_EOS"]  # type: ignore[assignment]

        n_trigger_geom = n_trigger_fm3 * utils.fm_inv3_to_geometric
        pc_trigger = jnp.interp(n_trigger_geom, n_grid_geom, p_grid_geom)
        logpc_trigger = jnp.log10(pc_trigger)
        return jnp.interp(logpc_trigger, logpc_eos, masses_eos)

    @staticmethod
    def _cse_transition_density(
        params: dict[str, Float | Array],
    ) -> tuple[Float, bool]:
        r"""Return the effective CSE transition density in fm\ :sup:`-3`.

        Manual CSE/peakCSE models preserve the sampled ``nbreak`` parameter in
        the transform output.  Adaptive CSE models expose their causality-driven
        transition density via ``extra_constraints["nbreak"]``, which the
        transform flattens to the same key.
        """
        if "nbreak" not in params:
            return jnp.asarray(jnp.inf), False
        return jnp.asarray(params["nbreak"]), True

    def _select_trigger_density(
        self,
        n_durca_fm3: Float,
        n_tov_fm3: Float,
        params: dict[str, Float | Array],
    ) -> tuple[Float, Float]:
        r"""Return ``(n_trig, valid)`` under the configured assumption.

        ``durca_or_cse`` is handled separately in :meth:`evaluate` because it
        involves a 2D marginalisation when ``n_durca > nbreak``.
        """
        n_cse_fm3, has_cse_transition = self._cse_transition_density(params)

        if self.trigger_assumption == "durca_only":
            before_tov = n_durca_fm3 < n_tov_fm3
            before_cse = n_durca_fm3 <= n_cse_fm3 if has_cse_transition else True
            before_break = before_cse
            valid = jnp.logical_and(before_tov, before_break)
            return n_durca_fm3, valid

        # durca_or_cse_simple (legacy): use min(n_durca, nbreak) as trigger
        n_trig_fm3 = jnp.minimum(n_durca_fm3, n_cse_fm3)
        valid = n_trig_fm3 < n_tov_fm3
        return n_trig_fm3, valid

    def _marginalize_trigger_likelihood(
        self,
        nbreak_fm3: Float,
        n_tov_fm3: Float,
        mtov: Float,
        params: dict[str, Float | Array],
    ) -> Float:
        r"""Marginalise over unknown cooling density in the CSE branch.

        Performs the 2D integral

        .. math::

            \mathcal{L}
            = \int_{n^*_{\rm min}}^{n^*_{\rm max}}
              \int_{n_{\rm break}}^{n^*}
              \mathcal{L}_{\rm mtrig}\!\bigl(M(n_{\rm cool})\bigr) \;
              \frac{1}{n^* - n_{\rm break}} \;
              \frac{1}{n^*_{\rm max} - n^*_{\rm min}} \;
              dn_{\rm cool} \, dn^*,

        where :math:`n^*` is the stellar central density and
        :math:`n_{\rm cool}` is the density at which direct Urca turns on
        in the CSE branch.  Both are uniformly distributed subject to
        :math:`n_{\rm cool} \leq n^*`, and a physical cutoff
        :math:`M(n_{\rm cool}) \leq M_{\rm TOV}` is applied.

        Returns the log of the marginalised likelihood.
        """
        nsat = _N_SAT  # fm⁻³
        nstar_min = jnp.maximum(nbreak_fm3, self.nstar_min_nsat * nsat)
        nstar_max = self.nstar_max_nsat * nsat

        # Guard against removable singularity at the lower boundary.
        # When nstar_min == nbreak_fm3 (CSE transition density exceeds
        # the default lower bound), the prior ∝ 1/(n⁎ − nbreak) diverges
        # at the first grid point while the inner-integration domain
        # collapses to zero width.  The trapezoidal rule cannot resolve
        # this 0/0 limit, so we shift nstar_min by one nstar grid
        # spacing to keep the integration well-conditioned.
        nstar_spacing = (nstar_max - nbreak_fm3) / (self.nb_nstar - 1)
        nstar_min = jnp.where(
            nstar_min <= nbreak_fm3,
            nbreak_fm3 + nstar_spacing,
            nstar_min,
        )

        ncool_grid = jnp.linspace(nbreak_fm3, nstar_max, self.nb_ncool)
        nstar_grid = jnp.linspace(nstar_min, nstar_max, self.nb_nstar)

        N_COOL, N_STAR = jnp.meshgrid(ncool_grid, nstar_grid, indexing="ij")

        # n_cool must not exceed n_star (cooling only when central density
        # reaches the threshold).
        valid_mask = jnp.where(N_COOL <= N_STAR, 1.0, 0.0)
        prior_nstar = 1.0 / (nstar_max - nstar_min)
        prior_ncool = jnp.where(
            N_COOL <= N_STAR,
            1.0 / jnp.maximum(N_STAR - nbreak_fm3, 1e-12),
            0.0,
        )

        # Convert n_cool grid to trigger mass via TOV interpolation.
        n_trigger_geom = N_COOL * utils.fm_inv3_to_geometric
        n_grid_geom: Float[Array, " n"] = params["n"]  # type: ignore[assignment]
        p_grid_geom: Float[Array, " n"] = params["p"]  # type: ignore[assignment]
        logpc_eos: Float[Array, " n"] = params["logpc_EOS"]  # type: ignore[assignment]
        masses_eos: Float[Array, " n"] = params["masses_EOS"]  # type: ignore[assignment]

        pc_trigger = jnp.interp(n_trigger_geom, n_grid_geom, p_grid_geom)
        logpc_trigger = jnp.log10(pc_trigger)
        m_cool = jnp.interp(logpc_trigger, logpc_eos, masses_eos)

        # Physical cutoff: trigger mass must be below M_TOV.
        cutoff_mask = jnp.where(m_cool <= mtov, 1.0, 0.0)

        # Evaluate the trigger-mass likelihood at each grid point.
        log_like_grid = self._log_mtrig_likelihood(m_cool, mtov)
        joint_likelihood = jnp.exp(log_like_grid)

        # Assemble integrand and integrate via 2D trapezoidal rule.
        integrand = (
            joint_likelihood * valid_mask * cutoff_mask * prior_ncool * prior_nstar
        )
        integral = jnp.trapezoid(
            jnp.trapezoid(integrand, x=ncool_grid, axis=0),
            x=nstar_grid,
            axis=0,
        )

        return jnp.log(jnp.maximum(integral, 1e-300))

    def _log_survival_likelihood(self, m_trig: Float, mtov: Float) -> Float:
        r"""Evaluate the playground upper-limit likelihood at ``m_trig``."""
        z_sax = (m_trig[..., None] - self.sax_mu) / self.sax_sig
        cdf_sax = jnp.sum(self.sax_w * norm.cdf(z_sax), axis=-1)
        log_sf_sax = jnp.log(jnp.maximum(1.0 - cdf_sax, 1e-300))

        z_cas = (m_trig - self.cas_mu) / self.cas_sig
        log_sf_cas = norm.logcdf(-z_cas)

        cdf_b2334 = jnp.minimum(
            jnp.maximum((m_trig - self.b2334_loc) / self.b2334_scale, 0.0),
            1.0,
        )
        log_sf_b2334 = jnp.log(jnp.maximum(1.0 - cdf_b2334, 1e-300))

        vela_scale = mtov - self.vela_loc
        cdf_vela = jnp.minimum(
            jnp.maximum((m_trig - self.vela_loc) / vela_scale, 0.0),
            1.0,
        )
        log_sf_vela = jnp.log(jnp.maximum(1.0 - cdf_vela, 1e-300))

        return log_sf_sax + log_sf_cas + log_sf_b2334 + log_sf_vela

    def _log_mtrig_likelihood(self, m_trig: Float, mtov: Float) -> Float:
        r"""Evaluate the trigger-mass likelihood at ``m_trig``."""
        return self._log_survival_likelihood(m_trig, mtov)

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the log-likelihood for the selected trigger assumption."""
        required = (
            "n_orig",
            "proton_fraction",
            "n",
            "p",
            "masses_EOS",
            "logpc_EOS",
            "n_TOV",
        )
        if any(key not in params for key in required):
            return jnp.asarray(self.penalty_value)

        n_tov_geom: Float = params["n_TOV"]  # type: ignore[assignment]
        n_tov_fm3 = n_tov_geom * utils.geometric_to_fm_inv3
        mtov = jnp.max(params["masses_EOS"])  # type: ignore[arg-type]

        n_durca_fm3 = self._find_n_durca(params)

        # --- durca_or_cse: marginalize when n_durca > nbreak ---
        if self.trigger_assumption == "durca_or_cse":
            n_cse_fm3, has_cse = self._cse_transition_density(params)

            # Marginalization only needed when direct Urca does NOT occur
            # in the nucleonic branch (n_durca > nbreak), a CSE
            # transition exists, and nbreak is below n_TOV (otherwise the
            # cooling threshold can never be reached in a stable star).
            needs_marginal = jnp.logical_and(
                has_cse,
                jnp.logical_and(
                    n_durca_fm3 > n_cse_fm3,
                    n_cse_fm3 < n_tov_fm3,
                ),
            )

            # Direct path: use n_durca when it is below nbreak (or no CSE).
            m_trig_direct = self._mass_at_density(n_durca_fm3, params)
            valid_direct = jnp.logical_and(
                n_durca_fm3 < n_tov_fm3,
                jnp.logical_and(m_trig_direct > 0.0, m_trig_direct < mtov),
            )
            log_like_direct = jnp.where(
                valid_direct,
                self._log_mtrig_likelihood(m_trig_direct, mtov),
                self.penalty_value,
            )

            # Marginalized path: integrate over n_cool, n_star.
            log_like_marginal = self._marginalize_trigger_likelihood(
                nbreak_fm3=n_cse_fm3,
                n_tov_fm3=n_tov_fm3,
                mtov=mtov,
                params=params,
            )

            log_likelihood = jnp.where(
                needs_marginal, log_like_marginal, log_like_direct
            )

            return jnp.nan_to_num(
                log_likelihood,
                nan=self.penalty_value,
                posinf=self.penalty_value,
                neginf=self.penalty_value,
            )

        # --- durca_only / durca_or_cse_simple ---
        n_trig_fm3, valid_trigger = self._select_trigger_density(
            n_durca_fm3, n_tov_fm3, params
        )
        m_trig = self._mass_at_density(n_trig_fm3, params)

        invalid_mass = jnp.logical_or(m_trig <= 0.0, m_trig >= mtov)
        valid = jnp.logical_and(valid_trigger, ~invalid_mass)
        log_likelihood = jnp.where(
            valid,
            self._log_mtrig_likelihood(m_trig, mtov),
            self.penalty_value,
        )

        return jnp.nan_to_num(
            log_likelihood,
            nan=self.penalty_value,
            posinf=self.penalty_value,
            neginf=self.penalty_value,
        )


class MtrigUpperLikelihood(DirectUrcaLikelihood):
    """Backward-compatible alias for the playground likelihood name."""


class MtrigLowerLikelihood(DirectUrcaLikelihood):
    r"""Lower-bound likelihood for the direct-Urca or CSE trigger mass.

    This likelihood evaluates the consistency of :math:`m_{\rm trig}` against
    lower limits from non-rapid-cooling objects. Since these objects have not
    triggered rapid cooling, their mass must be below :math:`m_{\rm trig}`.

    The likelihood is:

    .. math::

        \mathcal{L}(m_{\rm trig}) = F_{\rm HESS}(m_{\rm trig}).
    """

    def __init__(
        self,
        trigger_assumption: TriggerAssumption = "durca_only",
        name: str = "Mtrig_Lower_Bound",
        penalty_value: float = -1e5,
        nstar_min_nsat: float = 4.0,
        nstar_max_nsat: float = 10.0,
        nb_ncool: int = 400,
        nb_nstar: int = 200,
    ) -> None:
        super().__init__(
            trigger_assumption=trigger_assumption,
            name=name,
            penalty_value=penalty_value,
            nstar_min_nsat=nstar_min_nsat,
            nstar_max_nsat=nstar_max_nsat,
            nb_ncool=nb_ncool,
            nb_nstar=nb_nstar,
        )

        # HESS non-rapid-cooling object (Gaussian).
        self.hess_mu = 0.77
        self.hess_sig = 0.20

    def _log_mtrig_likelihood(self, m_trig: Float, mtov: Float) -> Float:
        r"""Evaluate the HESS lower-bound likelihood at ``m_trig``."""
        del mtov
        z_hess = (m_trig - self.hess_mu) / self.hess_sig
        return norm.logcdf(z_hess)

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        r"""Evaluate the lower-bound trigger-mass log-likelihood.

        If ``params["m_trig"]`` is present, it is used directly. Otherwise the
        trigger mass is computed from the EOS/TOV quantities via the inherited
        direct-Urca/CSE trigger logic.
        """
        if "m_trig" not in params:
            return super().evaluate(params)

        m_trig: Float = params["m_trig"]  # type: ignore[assignment]
        invalid_mass = m_trig <= 0.0
        log_likelihood = jnp.where(
            invalid_mass,
            self.penalty_value,
            self._log_mtrig_likelihood(m_trig, jnp.asarray(jnp.nan)),
        )

        return jnp.nan_to_num(
            log_likelihood,
            nan=self.penalty_value,
            posinf=self.penalty_value,
            neginf=self.penalty_value,
        )
