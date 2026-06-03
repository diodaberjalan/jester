r"""Meta-model EOS with adaptive piecewise constant speed-of-sound extensions.

The break density (``nbreak``) is not a free parameter — it is determined
automatically from the base meta-model EOS by locating the density where
the speed of sound squared crosses a user-specified threshold (default 0.95).

This is useful for:

- Starting the CSE extension *before* the EOS becomes acausal (``cs2_threshold=0.95``),
  replacing the near-superlumital tail with a controlled piecewise-constant
  parameterisation.
- Triggering the CSE extension near a phase-transition softening
  (``cs2_threshold=0.05``), capturing a drop in stiffness.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel.base import MetaModel_EOS_model
from jesterTOV.tov.data_classes import EOSData


class MetaModel_with_AdaptiveCSE_EOS_model(Interpolate_EOS_model):
    r"""
    Meta-model EOS with adaptive piecewise speed-of-sound extensions.

    Identical in spirit to :class:`MetaModel_with_CSE_EOS_model`, except that
    ``nbreak`` is **not** a free prior parameter.  Instead it is located
    automatically as the first density (going upward) where the meta-model's
    :math:`c_s^2` reaches or exceeds ``cs2_threshold``.

    The EOS is constructed in two regions:

    1. **Low-to-intermediate density** — Meta-model (crust + core) up to
       the auto-detected ``nbreak``.
    2. **High density** — Piecewise-constant speed-of-sound extension on
       :math:`[n_{\rm break}, n_{\rm max}]`.

    Parameters of the CSE grid (``n_CSE_i_u``, ``cs2_CSE_i``) are still
    sampled and work identically to the standard CSE model.
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        nb_CSE: Int = 8,
        cs2_threshold: Float = 0.95,
        **metamodel_kwargs,
    ):
        r"""
        Initialize the MetaModel with adaptive CSE.

        Args:
            nsat: Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Defaults to 0.16.
            nmin_MM_nsat: Starting density for meta-model region as fraction of
                :math:`n_0`.  Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat: Maximum density for EOS construction in units of
                :math:`n_0`.  Defines the upper reach of the CSE region.
                Defaults to 12.
            ndat_metamodel: Number of density points for meta-model region.
                Defaults to 100.
            ndat_CSE: Number of density points for the CSE region.
                Defaults to 100.
            nb_CSE: Number of CSE grid points (determines how many
                ``n_CSE_i_u`` / ``cs2_CSE_i`` pairs are expected).
                Defaults to 8.
            cs2_threshold: Speed-of-sound squared threshold for automatic
                ``nbreak`` detection.  The break density is the first point
                (going upward) where the base meta-model's :math:`c_s^2`
                reaches or exceeds this value.  Defaults to 0.95.
            **metamodel_kwargs: Passed to :class:`MetaModel_EOS_model`.
        """
        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE
        self.cs2_threshold = cs2_threshold

        # The meta-model is constructed up to the full nmax because we need
        # cs² everywhere to locate nbreak.  The max_nbreak_nsat optimisation
        # used in the non-adaptive CSE is not applicable here.
        self.metamodel = MetaModel_EOS_model(
            nsat=nsat,
            nmin_MM_nsat=nmin_MM_nsat,
            nmax_nsat=nmax_nsat,
            ndat=ndat_metamodel,
            **metamodel_kwargs,
        )

    @staticmethod
    def _find_nbreak_from_cs2(
        n_full: Float[Array, " n"],
        cs2_full: Float[Array, " n"],
        threshold: float,
        n_max: float,
        direction: str = "above",
    ) -> Float[Array, ""]:
        r"""Locate the first density where :math:`c_s^2` crosses a threshold.

        Args:
            n_full: Density grid in :math:`\\mathrm{fm}^{-3}` (monotonic).
            cs2_full: Speed-of-sound squared on the same grid.
            threshold: Target threshold value.
            n_max: Fallback density when the threshold is never crossed.
            direction: ``"above"`` (default) finds first point where
                ``cs2 >= threshold``; ``"below"`` finds first point
                where ``cs2 <= threshold``.

        Returns:
            Scalar array: the break density in :math:`\\mathrm{fm}^{-3}`.
        """
        mask = (
            cs2_full >= threshold if direction == "above"
            else cs2_full <= threshold
        )
        any_crossed = jnp.any(mask)
        indices = jnp.arange(len(cs2_full))
        masked_indices = jnp.where(mask, indices, len(cs2_full))
        first_idx = jnp.min(masked_indices)
        return jnp.where(any_crossed, n_full[first_idx], n_max)

    def construct_eos(
        self,
        params: dict[str, float],
        ngrids: Float[Array, " n_grid_point"] | None = None,
        cs2grids: Float[Array, " n_grid_point"] | None = None,
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ) -> EOSData | tuple:
        r"""
        Construct the EOS by combining meta-model and adaptive CSE regions.

        ``nbreak`` is auto-detected from :math:`c_s^2` of the base meta-model
        — it is **not** read from ``params``.

        Args:
            params: Dictionary of NEP parameters (``E_sat``, ``K_sat``, …)
                plus CSE grid parameters (``n_CSE_i_u``, ``cs2_CSE_i``).
                ``nbreak`` is **not** required and will be ignored if present.
            ngrids: Optional pre-built density grid for the CSE part.
            cs2grids: Optional pre-built speed-of-sound grid for the CSE part.
            return_extra: If True, return legacy tuple for backward compatibility.
            calculate_durca: If True, compute Direct Urca threshold density.

        Returns:
            :class:`EOSData` (or legacy tuple when ``return_extra=True``).
        """
        # ----------------------------------------------------------------
        # 1.  Construct the full meta-model (up to nmax)
        # ----------------------------------------------------------------
        (
            n_metamodel_full_arr,
            p_metamodel_full_arr,
            _hs,
            e_metamodel_full_arr,
            _dloge_dlogps,
            mu_metamodel_full_arr,
            cs2_metamodel_full_arr,
            extra_constraints,
        ) = self.metamodel.construct_eos(
            params, return_extra=True, calculate_durca=calculate_durca
        )

        # Convert from geometric units back to physical for interpolation
        n_metamodel_full = n_metamodel_full_arr / utils.fm_inv3_to_geometric
        p_metamodel_full = p_metamodel_full_arr / utils.MeV_fm_inv3_to_geometric
        e_metamodel_full = e_metamodel_full_arr / utils.MeV_fm_inv3_to_geometric
        mu_metamodel_full: Float[Array, "n_points"] = mu_metamodel_full_arr  # type: ignore[assignment]
        cs2_metamodel_full = cs2_metamodel_full_arr

        # ----------------------------------------------------------------
        # 2.  Automatic nbreak from cs² threshold
        # ----------------------------------------------------------------
        direction = "above" if self.cs2_threshold >= 0.5 else "below"
        nbreak = self._find_nbreak_from_cs2(
            n_metamodel_full, cs2_metamodel_full,
            self.cs2_threshold, self.nmax, direction,
        )

        # Expose nbreak in extra_constraints so downstream likelihoods
        # (e.g. DirectUrcaLikelihood with constraint_type="n_break") can
        # access it.
        extra_constraints["nbreak"] = nbreak

        # ----------------------------------------------------------------
        # 3.  Build CSE grid from params (identical to regular CSE)
        # ----------------------------------------------------------------
        if ngrids is None or cs2grids is None:
            ngrids_u = jnp.array([params[f"n_CSE_{i}_u"] for i in range(self.nb_CSE)])
            ngrids_u = jnp.sort(ngrids_u)
            cs2grids_local = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])

            width = self.nmax - nbreak
            ngrids = nbreak + ngrids_u * width

            ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
            cs2grids_local = jnp.append(
                cs2grids_local, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]])
            )
        else:
            cs2grids_local = cs2grids

        # ----------------------------------------------------------------
        # 4.  Interpolate meta-model up to nbreak
        # ----------------------------------------------------------------
        n_metamodel = jnp.logspace(
            jnp.log10(n_metamodel_full[0]),
            jnp.log10(nbreak),
            self.ndat_metamodel,
            endpoint=True,
        )
        p_metamodel = jnp.interp(n_metamodel, n_metamodel_full, p_metamodel_full)
        e_metamodel = jnp.interp(n_metamodel, n_metamodel_full, e_metamodel_full)
        mu_metamodel = jnp.interp(n_metamodel, n_metamodel_full, mu_metamodel_full)
        cs2_metamodel = jnp.interp(n_metamodel, n_metamodel_full, cs2_metamodel_full)

        # Values at break density
        p_break = jnp.interp(nbreak, n_metamodel, p_metamodel)
        e_break = jnp.interp(nbreak, n_metamodel, e_metamodel)
        mu_break = jnp.interp(nbreak, n_metamodel, mu_metamodel)
        cs2_break = jnp.interp(nbreak, n_metamodel, cs2_metamodel)

        # ----------------------------------------------------------------
        # 5.  Stitch CSE extension
        # ----------------------------------------------------------------
        ngrids_ext = jnp.concatenate((jnp.array([nbreak]), ngrids))
        cs2grids_ext = jnp.concatenate((jnp.array([cs2_break]), cs2grids_local))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids_ext, cs2grids_ext)

        n_CSE = jnp.logspace(
            jnp.log10(nbreak), jnp.log10(self.nmax), num=self.ndat_CSE,
        )
        cs2_CSE = cs2_extension_function(n_CSE)

        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # ----------------------------------------------------------------
        # 6.  Combine and build final EOSData
        # ----------------------------------------------------------------
        n = jnp.concatenate((n_metamodel, n_CSE))
        p = jnp.concatenate((p_metamodel, p_CSE))
        e = jnp.concatenate((e_metamodel, e_CSE))

        mu = jnp.concatenate((mu_metamodel, mu_CSE))
        cs2 = jnp.concatenate((cs2_metamodel, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        eos_data = EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=mu,
            extra_constraints=extra_constraints,
        )
        if return_extra:
            return (ns, ps, hs, es, dloge_dlogps, mu, cs2, extra_constraints)
        return eos_data

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of parameters required by MetaModel with adaptive CSE.

        ``nbreak`` is **not** included — it is auto-detected from
        :math:`c_s^2`.

        Returns:
            list[str]: NEP parameters + CSE grid parameters
        """
        params = [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
        for i in range(self.nb_CSE):
            params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        params.append(f"cs2_CSE_{self.nb_CSE}")
        return params