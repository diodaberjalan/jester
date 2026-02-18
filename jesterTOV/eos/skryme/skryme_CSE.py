r"""Skyrme EOS with piecewise constant speed-of-sound extensions (CSE)."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.skryme.base import (
    Skryme_EOS_model,
    create_skryme_for_extension,
)


class Skryme_with_CSE_EOS_model(Interpolate_EOS_model):
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
        nmin_Skryme_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        max_nbreak_nsat: Float | None = None,
        ndat_skryme: Int = 100,
        ndat_CSE: Int = 100,
        **skryme_kwargs,
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
            nmin_Skryme_nsat (Float, optional):
                Starting density for Skyrme region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Defines the high-density reach including CSE region. Defaults to 12.
            max_nbreak_nsat (Float | None, optional):
                Maximum value of nbreak prior in units of :math:`n_0`.
                Used to set the upper limit for the Skyrme region.
                If None, defaults to nmax_nsat. Defaults to None.
            ndat_skryme (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values give smoother interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for constant speed-of-sound extension region.
                Controls resolution of high-density modeling. Defaults to 100.
            **skryme_kwargs:
                Additional keyword arguments passed to the underlying Skryme_EOS_model.
                Includes parameters like crust_name, etc. See Skryme_EOS_model.__init__.

        See Also:
            Skryme_EOS_model.__init__ : Base Skyrme parameters
            construct_eos : Method that defines CSE parameters and break density
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skryme_nsat = nmin_Skryme_nsat
        self.ndat_skryme = ndat_skryme
        self.nmax_nsat = nmax_nsat
        self.skryme_kwargs = skryme_kwargs

        # Store proton_fraction setting for later use
        self.proton_fraction_setting = skryme_kwargs.get("proton_fraction", "exact")

        # Use max_nbreak_nsat if provided, otherwise default to nmax_nsat
        metamodel_max_nsat = (
            max_nbreak_nsat if max_nbreak_nsat is not None else nmax_nsat
        )

        # Create the Skryme instance once with max density from nbreak prior
        self.skryme = Skryme_EOS_model(
            nsat=nsat,
            nmin_Skryme_nsat=nmin_Skryme_nsat,
            nmax_nsat=metamodel_max_nsat,
            ndat=ndat_skryme,
            **skryme_kwargs,
        )

    def construct_eos(
        self,
        INM_dict: dict,
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
            INM_dict (dict): Dictionary with the INM keys to be passed to the Skyrme EOS class.
                Must include 'nbreak' specifying the transition density between Skyrme and CSE.
            ngrids (Float[Array, `n_grid_point`]): Density grid points for the CSE part of the EOS.
            cs2grids (Float[Array, `n_grid_point`]): Speed-of-sound squared grid points for the CSE part.
            return_extra (bool, optional): If True, return extra Skyrme-specific quantities
                (proton fraction, lepton fractions, DURCA density) in the output.
                Defaults to False.
            calculate_durca (bool | None, optional): If True, calculate the Direct Urca threshold density.
                If None, uses the default from the skryme instance. Defaults to None.

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential and speed of sound.
                If return_extra=True, also returns a dict with Skyrme-specific quantities.
        """

        # Get nbreak for use in skryme creation
        nbreak = INM_dict.get("nbreak", self.skryme.nmax)

        # Create fresh Skryme instance limited to nbreak
        # This ensures proton_fraction setting is properly propagated
        if return_extra or calculate_durca:
            skryme = create_skryme_for_extension(
                nsat=self.skryme.nsat,
                nmin_Skryme_nsat=self.skryme.nmin_Skryme_nsat,
                nbreak=nbreak,
                ndat=self.skryme.ndat,
                skryme_kwargs={},
                proton_fraction_setting=self.proton_fraction_setting,
            )
            skryme_output = skryme.construct_eos(
                INM_dict, return_extra=True, calculate_durca=calculate_durca
            )
        else:
            # Use pre-instantiated skryme for efficiency
            skryme_output = self.skryme.construct_eos(INM_dict, return_extra=True)

        # Handle both return_extra=True and return_extra=False cases
        if return_extra or calculate_durca:
            (
                n_skryme_full,
                p_skryme_full,
                _,
                e_skryme_full,
                _,
                mu_skryme_full,
                cs2_skryme_full,
                extra,
            ) = skryme_output
        else:
            (
                n_skryme_full,
                p_skryme_full,
                _,
                e_skryme_full,
                _,
                mu_skryme_full,
                cs2_skryme_full,
            ) = skryme_output
            extra = None

        # Convert units back for interpolation
        n_skryme_full = n_skryme_full / utils.fm_inv3_to_geometric
        p_skryme_full = p_skryme_full / utils.MeV_fm_inv3_to_geometric
        e_skryme_full = e_skryme_full / utils.MeV_fm_inv3_to_geometric

        # Re-interpolate to a fixed-size array up to nbreak
        # This maintains JAX compatibility while allowing variable nbreak
        n_skryme = jnp.linspace(
            n_skryme_full[0], nbreak, self.ndat_skryme, endpoint=True
        )
        p_skryme = jnp.interp(n_skryme, n_skryme_full, p_skryme_full)
        e_skryme = jnp.interp(n_skryme, n_skryme_full, e_skryme_full)
        mu_skryme = jnp.interp(n_skryme, n_skryme_full, mu_skryme_full)
        cs2_skryme = jnp.interp(n_skryme, n_skryme_full, cs2_skryme_full)

        # Get values at break density
        p_break = jnp.interp(nbreak, n_skryme, p_skryme)
        e_break = jnp.interp(nbreak, n_skryme, e_skryme)
        mu_break = jnp.interp(nbreak, n_skryme, mu_skryme)
        cs2_break = jnp.interp(nbreak, n_skryme, cs2_skryme)

        # Define the speed-of-sound interpolation of the extension portion
        ngrids = jnp.concatenate((jnp.array([nbreak]), ngrids))
        cs2grids = jnp.concatenate((jnp.array([cs2_break]), cs2grids))
        cs2_extension_function = lambda n: jnp.interp(n, ngrids, cs2grids)

        # Compute n, p, e for CSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(jnp.log10(nbreak), jnp.log10(self.nmax), num=self.ndat_CSE)
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine skryme and CSE data
        n = jnp.concatenate((n_skryme, n_CSE))
        p = jnp.concatenate((p_skryme, p_CSE))
        e = jnp.concatenate((e_skryme, e_CSE))

        mu = jnp.concatenate((mu_skryme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skryme, cs2_CSE))

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
