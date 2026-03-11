r"""Skyrme EOS with piecewise constant speed-of-sound extensions (CSE)."""

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
    phase transitions or exotic matter components in neutron star cores.

    The EOS is constructed in two regions:

    1. **Low-to-intermediate density**: Skyrme EOS (crust + core)
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
        **skyrme_kwargs,
    ):
        r"""
        Initialize the Skyrme with CSE EOS combining Skyrme and constant speed-of-sound extensions.

        This constructor sets up a hybrid EOS that uses the Skyrme approach for
        low-to-intermediate densities and allows for user-defined constant speed-of-sound
        extensions at high densities.

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
                Used to set the upper limit for the Skyrme region.
                If None, defaults to nmax_nsat. Defaults to None.
            ndat_skyrme (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values give smoother interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for constant speed-of-sound extension region.
                Controls resolution of high-density modeling. Defaults to 100.
            **skyrme_kwargs:
                Additional keyword arguments passed to the underlying Skyrme_EOS_model.
                Includes parameters like crust_name, etc. See Skyrme_EOS_model.__init__.

        See Also:
            Skyrme_EOS_model.__init__ : Base Skyrme parameters
            construct_eos : Method that defines CSE parameters and break density
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skyrme_nsat = nmin_Skyrme_nsat
        self.ndat_skyrme = ndat_skyrme
        self.nmax_nsat = nmax_nsat
        self.skyrme_kwargs = skyrme_kwargs

        # Store proton_fraction setting for later use
        self.proton_fraction_setting = skyrme_kwargs.get("proton_fraction", "exact")

        # Use max_nbreak_nsat if provided, otherwise default to nmax_nsat
        metamodel_max_nsat = (
            max_nbreak_nsat if max_nbreak_nsat is not None else nmax_nsat
        )

        # Create the Skyrme instance once with max density from nbreak prior
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
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ) -> tuple:
        r"""
        Construct the EOS by combining Skyrme and CSE regions.

        This method constructs the full EOS by:
        1. Building the Skyrme EOS up to the full nmax range
        2. Interpolating the Skyrme to a fixed-size grid up to nbreak
        3. Stitching the CSE extension on top from nbreak to nmax

        Args:
            params (dict): Dictionary containing Skyrme parameters and nbreak.
            ngrids (Float[Array, `n_grid_point`]): Density grid points for the CSE part of the EOS.
            cs2grids (Float[Array, `n_grid_point`]): Speed-of-sound squared grid points for the CSE part.
            return_extra (bool, optional): If True, return extra Skyrme-specific quantities. Defaults to False.
            calculate_durca (bool | None, optional): If True, calculate the Direct Urca threshold density. Defaults to None.

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential and speed of sound.
                If return_extra=True, also returns a dict with Skyrme-specific quantities.
        """

        # Extract break density
        nbreak = params.get("nbreak", self.skyrme.nmax)

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
        if return_extra:
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
        else:
            (
                n_skyrme_full,
                p_skyrme_full,
                _,
                e_skyrme_full,
                _,
                mu_skyrme_full,
                cs2_skyrme_full,
            ) = skyrme_output
            extra = None

        # Convert units back for interpolation
        n_skyrme_full = n_skyrme_full / utils.fm_inv3_to_geometric
        p_skyrme_full = p_skyrme_full / utils.MeV_fm_inv3_to_geometric
        e_skyrme_full = e_skyrme_full / utils.MeV_fm_inv3_to_geometric

        # Re-interpolate to a fixed-size array up to nbreak
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

        mu = jnp.concatenate((mu_skyrme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skyrme, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        if return_extra:
            return ns, ps, hs, es, dloge_dlogps, mu, cs2, extra
        return ns, ps, hs, es, dloge_dlogps, mu, cs2

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