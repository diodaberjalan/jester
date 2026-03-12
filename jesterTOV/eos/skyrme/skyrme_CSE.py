r"""Skyrme EOS with piecewise constant speed-of-sound extensions (CSE)."""

from typing import Any, Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.eos.skyrme.base import (
    Skyrme_EOS_model,
    create_skyrme_for_extension,
)


class Skyrme_with_CSE_EOS_model(Interpolate_EOS_model):
    r"""
    Skyrme EOS combined with piecewise speed-of-sound extensions (CSE).

    This class extends the Skyrme approach by allowing for piecewise-constant
    speed-of-sound extensions at high densities. This is useful for modeling
    phase transitions or exotic matter components in neutron star cores that
    may not be captured by the Skyrme functional expansions.

    The EOS is constructed in two regions:

    1. **Low-to-intermediate density**: Skyrme approach (crust + core)
    2. **High density**: Speed-of-sound extension scheme
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_Skyrme_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        max_nbreak_nsat: Float | None = None,
        ndat_skyrme: Int = 100,
        ndat_CSE: Int = 100,
        nb_CSE: int = 0,
        **skyrme_kwargs,
    ):
        r"""
        Initialize the Skyrme with CSE EOS combining Skyrme and constant speed-of-sound extensions.

        This constructor sets up a hybrid EOS that uses the Skyrme approach for
        low-to-intermediate densities and allows for user-defined constant speed-of-sound
        extensions at high densities. The transition occurs at a break density specified
        in the params dictionary during EOS construction.

        Args:
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                Reference density for the Skyrme construction. Defaults to 0.16.
            nmin_Skyrme_nsat (Float, optional):
                Starting density for Skyrme region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Defines the high-density reach including CSE region. Defaults to 12.
            max_nbreak_nsat (Float | None, optional):
                Maximum value of nbreak prior in units of :math:`n_0`.
                Used to set the upper limit for the Skyrme region to avoid
                unnecessary computation. If None, defaults to nmax_nsat.
                Should be set to the maximum value from the nbreak prior distribution.
                Defaults to None.
            ndat_skyrme (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values give smoother Skyrme interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for constant speed-of-sound extension region.
                Controls resolution of high-density exotic matter modeling. Defaults to 100.
            nb_CSE (int, optional):
                Number of CSE grid points. If > 0, CSE grid parameters are generated
                from the params in construct_eos. Defaults to 0.
            **skyrme_kwargs:
                Additional keyword arguments passed to the underlying Skyrme_EOS_model.
                Includes parameters like crust_name, etc.
                See Skyrme_EOS_model.__init__ for complete parameter descriptions.

        See Also:
            Skyrme_EOS_model.__init__ : Base Skyrme parameters
            construct_eos : Method that defines CSE parameters and break density

        Note:
            The Skyrme model is created once in __init__ with max_nbreak_nsat as the maximum
            density to avoid re-instantiating the Skyrme class on each construct_eos call.
            During construct_eos, the Skyrme output is interpolated to the actual nbreak
            value (which varies with each sample) while maintaining fixed array sizes for JAX.
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skyrme_nsat = nmin_Skyrme_nsat
        self.ndat_skyrme = ndat_skyrme
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE
        self.skyrme_kwargs = skyrme_kwargs

        # Store proton_fraction setting for later use
        self.proton_fraction_setting = skyrme_kwargs.get("proton_fraction", "exact")

        # Use max_nbreak_nsat if provided, otherwise default to nmax_nsat
        # This allows optimization when the nbreak prior has a tighter upper bound
        metamodel_max_nsat = (
            max_nbreak_nsat if max_nbreak_nsat is not None else nmax_nsat
        )

        # Create the Skyrme instance once with max density from nbreak prior
        # This will be reused in construct_eos and interpolated to actual nbreak
        self.skyrme = Skyrme_EOS_model(
            nsat=nsat,
            nmin_Skyrme_nsat=nmin_Skyrme_nsat,
            nmax_nsat=metamodel_max_nsat,
            ndat=ndat_skyrme,
            **skyrme_kwargs,
        )

    def construct_eos(
        self,
        params: dict,
        ngrids: Float[Array, "n_grid_point"] | None = None,
        cs2grids: Float[Array, "n_grid_point"] | None = None,
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ) -> Union[EOSData, tuple]:
        r"""
        Construct the EOS by combining Skyrme and CSE regions.

        This method constructs the full EOS by:
        1. Building the Skyrme EOS up to the full nmax range
        2. Interpolating the Skyrme to a fixed-size grid up to nbreak
        3. Stitching the CSE extension on top from nbreak to nmax

        Args:
            params (dict): Dictionary with the parameters to be passed to the Skyrme EOS class.
                Must include 'nbreak' specifying the transition density between Skyrme and CSE.
                CSE grid parameters (n_CSE_i_u, cs2_CSE_i) are extracted from this dict if
                ngrids/cs2grids are not provided.
            ngrids (Float[Array, `n_grid_point`], optional): Density grid points for the CSE part.
                If None, extracted from params using nb_CSE.
            cs2grids (Float[Array, `n_grid_point`], optional): Speed-of-sound squared grid points.
                If None, extracted from params using nb_CSE.
            return_extra (bool, optional): Kept for backward compatibility. If True, returns
                a tuple ``(ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)`` for backward
                compatibility with older notebooks.
            calculate_durca (bool | None, optional): If True, calculate the Direct Urca threshold density.

        Returns:
            Union[EOSData, tuple]:
                - If ``return_extra=False`` (default): :class:`EOSData` object for inference compatibility.
                - If ``return_extra=True``: tuple ``(ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)``
                  where ``extra`` is a dict with Skyrme-specific quantities.
        """

        # Construct the Skyrme part
        # Get nbreak early for use in Skyrme creation
        nbreak = params.get("nbreak", self.skyrme.nmax)

        # Extract CSE grid parameters from params if not provided
        if ngrids is None or cs2grids is None:
            # Build CSE grid parameters from params
            # n_CSE_i_u are normalized positions (0 to 1), cs2_CSE_i are cs2 values
            ngrids = jnp.array([])
            cs2grids = jnp.array([])
            if self.nb_CSE > 0:
                # Extract n_CSE_i_u and cs2_CSE_i from params
                for i in range(self.nb_CSE):
                    n_u = params.get(f"n_CSE_{i}_u", 0.5)  # Default to midpoint
                    # Convert normalized position to actual density
                    n_density = nbreak + n_u * (self.nmax - nbreak)
                    ngrids = jnp.append(ngrids, jnp.array([n_density]))
                    cs2grids = jnp.append(cs2grids, jnp.array([params.get(f"cs2_CSE_{i}", 0.5)]))
                # Add final cs2 at nmax
                ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
                cs2grids = jnp.append(cs2grids, jnp.array([params.get(f"cs2_CSE_{self.nb_CSE}", 0.5)]))

        # We need to create a fresh instance when return_extra=True since the
        # pre-instantiated skyrme in __init__ doesn't support this parameter
        # We always request return_extra=True from the fresh instance and handle it here
        if return_extra or calculate_durca:
            # Use helper to create fresh skyrme instance limited to nbreak
            skyrme_model = create_skyrme_for_extension(
                nsat=self.skyrme.nsat,
                nmin_Skyrme_nsat=self.skyrme.nmin_Skyrme_nsat,
                nbreak=nbreak,
                ndat=self.skyrme.ndat,
                skyrme_kwargs=self.skyrme_kwargs,
                proton_fraction_setting=self.proton_fraction_setting,
            )
            # Always request extra to handle calculate_durca case
            skyrme_output = skyrme_model.construct_eos(
                params, return_extra=True, calculate_durca=calculate_durca
            )
        else:
            # Use pre-instantiated skyrme for efficiency
            skyrme_output = self.skyrme.construct_eos(params, return_extra=True)

        # Handle both return_extra=True and return_extra=False cases
        # Note: skyrme_output always has 8 values since we call with return_extra=True
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
        if not return_extra:
            extra = None

        # Convert units back for interpolation
        n_skyrme_full = n_skyrme_full / utils.fm_inv3_to_geometric
        p_skyrme_full = p_skyrme_full / utils.MeV_fm_inv3_to_geometric
        e_skyrme_full = e_skyrme_full / utils.MeV_fm_inv3_to_geometric

        # Re-interpolate to a fixed-size array up to nbreak
        # This maintains JAX compatibility while allowing variable nbreak
        n_skyrme = jnp.linspace(
            n_skyrme_full[0], nbreak, self.ndat_skyrme, endpoint=True
        )
        p_skyrme = jnp.interp(n_skyrme, n_skyrme_full, p_skyrme_full)
        e_skyrme = jnp.interp(n_skyrme, n_skyrme_full, e_skyrme_full)
        mu_skyrme = jnp.interp(n_skyrme, n_skyrme_full, mu_skyrme_full)
        cs2_skyrme = jnp.interp(n_skyrme, n_skyrme_full, cs2_skyrme_full)

        # Get values at break density
        p_break = jnp.interp(nbreak, n_skyrme, p_skyrme)
        e_break = jnp.interp(nbreak, n_skyrme, e_skyrme)
        mu_break = jnp.interp(nbreak, n_skyrme, mu_skyrme)
        cs2_break = jnp.interp(nbreak, n_skyrme, cs2_skyrme)

        # Define the speed-of-sound interpolation of the extension portion
        ngrids_ext = jnp.concatenate((jnp.array([nbreak]), ngrids))
        cs2grids_ext = jnp.concatenate((jnp.array([cs2_break]), cs2grids))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids_ext, cs2grids_ext)

        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(
            jnp.log10(nbreak), jnp.log10(self.nmax), num=self.ndat_CSE
        )
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine skyrme and CSE data
        n = jnp.concatenate((n_skyrme, n_CSE))
        p = jnp.concatenate((p_skyrme, p_CSE))
        e = jnp.concatenate((e_skyrme, e_CSE))

        # TODO: let's decide whether we want to save cs2 and mu or just use them for computation and then discard them.
        mu = jnp.concatenate((mu_skyrme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skyrme, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        # Build EOSData for inference compatibility
        # Include extra data from the skyrme if available
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

        # Return tuple for backward compatibility when return_extra=True
        # Expected order: (ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)
        if return_extra:
            return (ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)
        else:
            return eos_data

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of parameters required by Skyrme with CSE.

        Returns:
            list[str]: INM parameters + nbreak
        """
        return [
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
            "nbreak",
        ]