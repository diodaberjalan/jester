r"""Skyrme EOS with adaptive piecewise constant speed-of-sound extensions.

The break density (``nbreak``) is not a free parameter — it is determined
automatically from the base Skyrme EOS by locating the density where
the speed of sound squared crosses a user-specified threshold (default 0.95).

This is useful for:

- Starting the CSE extension *before* the EOS becomes acausal (``cs2_threshold=0.95``),
  replacing the near-superlumital tail with a controlled piecewise-constant
  parameterisation.
- Triggering the CSE extension near a phase-transition softening
  (``cs2_threshold=0.05``), capturing a drop in stiffness.
"""

from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.eos.skyrme.base import (
    Skyrme_EOS_model,
    create_skyrme_for_extension,
)


class Skyrme_with_AdaptiveCSE_EOS_model(Interpolate_EOS_model):
    r"""
    Skyrme EOS with adaptive piecewise speed-of-sound extensions.

    Identical in spirit to :class:`Skyrme_with_CSE_EOS_model`, except that
    ``nbreak`` is **not** a free prior parameter.  Instead it is located
    automatically as the first density (going upward) where the Skyrme EOS's
    :math:`c_s^2` reaches or exceeds ``cs2_threshold``.

    The EOS is constructed in two regions:

    1. **Low-to-intermediate density** — Skyrme (crust + core) up to
       the auto-detected ``nbreak``.
    2. **High density** — Piecewise-constant speed-of-sound extension on
       :math:`[n_{\rm break}, n_{\rm max}]`.

    Parameters of the CSE grid (``n_CSE_i_u``, ``cs2_CSE_i``) are still
    sampled and work identically to the standard CSE model.
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_Skyrme_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_skyrme: Int = 100,
        ndat_CSE: Int = 100,
        nb_CSE: int = 0,
        cs2_threshold: Float = 0.95,
        **skyrme_kwargs,
    ):
        r"""
        Initialize the Skyrme with adaptive CSE.

        Args:
            nsat: Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Defaults to 0.16.
            nmin_Skyrme_nsat: Starting density for Skyrme region as fraction of
                :math:`n_0`.  Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat: Maximum density for EOS construction in units of
                :math:`n_0`.  Defines the upper reach of the CSE region.
                Defaults to 12.
            ndat_skyrme: Number of density points for Skyrme region.
                Defaults to 100.
            ndat_CSE: Number of density points for the CSE region.
                Defaults to 100.
            nb_CSE: Number of CSE grid points.  If > 0, CSE grid parameters are
                generated from the params in ``construct_eos``.  Defaults to 0.
            cs2_threshold: Speed-of-sound squared threshold for automatic
                ``nbreak`` detection.  The break density is the first point
                (going upward) where the base Skyrme's :math:`c_s^2` reaches
                or exceeds this value.  Defaults to 0.95.
            **skyrme_kwargs: Passed to :class:`Skyrme_EOS_model`.
        """
        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skyrme_nsat = nmin_Skyrme_nsat
        self.ndat_skyrme = ndat_skyrme
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE
        self.cs2_threshold = cs2_threshold
        self.skyrme_kwargs = skyrme_kwargs

        # Store proton_fraction setting for later use
        self.proton_fraction_setting = skyrme_kwargs.get("proton_fraction", "exact")

        # The Skyrme model is constructed up to the full nmax because we need
        # cs² everywhere to locate nbreak.
        self.skyrme = Skyrme_EOS_model(
            nsat=nsat,
            nmin_Skyrme_nsat=nmin_Skyrme_nsat,
            nmax_nsat=nmax_nsat,
            ndat=ndat_skyrme,
            **skyrme_kwargs,
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
            n_full: Density grid in :math:`\mathrm{fm}^{-3}` (monotonic).
            cs2_full: Speed-of-sound squared on the same grid.
            threshold: Target threshold value.
            n_max: Fallback density when the threshold is never crossed.
            direction: ``"above"`` (default) finds first point where
                ``cs2 >= threshold``; ``"below"`` finds first point
                where ``cs2 <= threshold``.

        Returns:
            Scalar array: the break density in :math:`\mathrm{fm}^{-3}`.
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
        params: dict,
        ngrids: Float[Array, " n_grid_point"] | None = None,
        cs2grids: Float[Array, " n_grid_point"] | None = None,
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ) -> Union[EOSData, tuple]:
        r"""
        Construct the EOS by combining Skyrme and adaptive CSE regions.

        ``nbreak`` is auto-detected from :math:`c_s^2` of the base Skyrme EOS
        — it is **not** read from ``params``.

        Args:
            params: Dictionary of INM parameters (``t2``, ``t4``, ``x0``, …)
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
        # 1.  Build the full Skyrme model (up to nmax)
        # ----------------------------------------------------------------
        if return_extra or calculate_durca:
            skyrme_model = create_skyrme_for_extension(
                nsat=self.skyrme.nsat,
                nmin_Skyrme_nsat=self.skyrme.nmin_Skyrme_nsat,
                nbreak=self.nmax,
                ndat=self.skyrme.ndat,
                skyrme_kwargs=self.skyrme_kwargs,
                proton_fraction_setting=self.proton_fraction_setting,
            )
            skyrme_output = skyrme_model.construct_eos(
                params, return_extra=True, calculate_durca=calculate_durca
            )
        else:
            skyrme_output = self.skyrme.construct_eos(params, return_extra=True)

        (
            n_skyrme_full,
            p_skyrme_full,
            _,
            e_skyrme_full,
            _,
            mu_skyrme_full,
            cs2_skyrme_full,
            extra,
        ) = skyrme_output

        # Convert units back for interpolation
        n_skyrme_full = n_skyrme_full / utils.fm_inv3_to_geometric
        p_skyrme_full = p_skyrme_full / utils.MeV_fm_inv3_to_geometric
        e_skyrme_full = e_skyrme_full / utils.MeV_fm_inv3_to_geometric

        # ----------------------------------------------------------------
        # 2.  Automatic nbreak from cs² threshold
        # ----------------------------------------------------------------
        direction = "above" if self.cs2_threshold >= 0.5 else "below"
        nbreak = self._find_nbreak_from_cs2(
            n_skyrme_full, cs2_skyrme_full,
            self.cs2_threshold, self.nmax, direction,
        )

        # Expose nbreak in extra dict so downstream likelihoods
        # (e.g. DirectUrcaLikelihood with constraint_type="n_break") can
        # access it.
        extra["nbreak"] = nbreak

        # ----------------------------------------------------------------
        # 3.  Build CSE grid from params (identical to regular CSE)
        # ----------------------------------------------------------------
        if ngrids is None or cs2grids is None:
            ngrids_local = jnp.array([])
            cs2grids_local = jnp.array([])
            if self.nb_CSE > 0:
                for i in range(self.nb_CSE):
                    n_u = params.get(f"n_CSE_{i}_u", 0.5)
                    n_density = nbreak + n_u * (self.nmax - nbreak)
                    ngrids_local = jnp.append(ngrids_local, jnp.array([n_density]))
                    cs2grids_local = jnp.append(
                        cs2grids_local, jnp.array([params.get(f"cs2_CSE_{i}", 0.5)])
                    )
                ngrids_local = jnp.append(ngrids_local, jnp.array([self.nmax]))
                cs2grids_local = jnp.append(
                    cs2grids_local,
                    jnp.array([params.get(f"cs2_CSE_{self.nb_CSE}", 0.5)]),
                )
        else:
            cs2grids_local = cs2grids
            ngrids_local = ngrids

        # ----------------------------------------------------------------
        # 4.  Interpolate Skyrme up to nbreak
        # ----------------------------------------------------------------
        n_skyrme = jnp.linspace(
            n_skyrme_full[0], nbreak, self.ndat_skyrme, endpoint=True
        )
        p_skyrme = jnp.interp(n_skyrme, n_skyrme_full, p_skyrme_full)
        e_skyrme = jnp.interp(n_skyrme, n_skyrme_full, e_skyrme_full)
        mu_skyrme = jnp.interp(n_skyrme, n_skyrme_full, mu_skyrme_full)
        cs2_skyrme = jnp.interp(n_skyrme, n_skyrme_full, cs2_skyrme_full)

        # Values at break density
        p_break = jnp.interp(nbreak, n_skyrme, p_skyrme)
        e_break = jnp.interp(nbreak, n_skyrme, e_skyrme)
        mu_break = jnp.interp(nbreak, n_skyrme, mu_skyrme)
        cs2_break = jnp.interp(nbreak, n_skyrme, cs2_skyrme)

        # ----------------------------------------------------------------
        # 5.  Stitch CSE extension
        # ----------------------------------------------------------------
        ngrids_ext = jnp.concatenate((jnp.array([nbreak]), ngrids_local))
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
        n = jnp.concatenate((n_skyrme, n_CSE))
        p = jnp.concatenate((p_skyrme, p_CSE))
        e = jnp.concatenate((e_skyrme, e_CSE))

        mu = jnp.concatenate((mu_skyrme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skyrme, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        eos_data = EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=mu,
            extra_constraints=extra,
        )

        if return_extra:
            return (ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)
        return eos_data

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of parameters required by Skyrme with adaptive CSE.

        ``nbreak`` is **not** included — it is auto-detected from
        :math:`c_s^2`.

        Returns:
            list[str]: INM parameters + CSE grid parameters
        """
        params = [
            "t2",
            "t4",
            "x0",
            "x1",
            "x4",
            "alph",
            "beta",
            "gamma",
            "kfsat",
            "av",
            "J",
            "meffs",
            "meffv",
            "Kinf",
            "eNMhd",
        ]
        for i in range(self.nb_CSE):
            params.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        params.append(f"cs2_CSE_{self.nb_CSE}")
        return params