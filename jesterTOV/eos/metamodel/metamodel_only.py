r"""Meta-model EOS with piecewise constant speed-of-sound extensions (CSE)."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.metamodel.base import MetaModel_EOS_model


class MetaModel_only(Interpolate_EOS_model):
    r"""
    Meta-model EOS only, used to extract microscopic properties in intermediate density (between ~0.5-2 n_sat)
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        **metamodel_kwargs,
    ):
        r"""
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs

    def construct_eos(
        self,
        NEP_dict: dict,
        ngrids: Float[Array, "n_grid_point"],
        cs2grids: Float[Array, "n_grid_point"],
    ) -> tuple:
        r"""
        Construct the EOS

        Args:
            NEP_dict (dict): Dictionary with the NEP keys to be passed to the metamodel EOS class.
            ngrids (Float[Array, `n_grid_point`]): Density grid points of densities for the CSE part of the EOS.
            cs2grids (Float[Array, `n_grid_point`]): Speed-of-sound squared grid points of densities for the CSE part of the EOS.

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential and speed of sound.
        """

        # Initializate the MetaModel part up to n_break
        metamodel = MetaModel_EOS_model(
            nsat=self.nsat,
            nmin_MM_nsat=self.nmin_MM_nsat,
            nmax_nsat=NEP_dict["nbreak"] / self.nsat,
            ndat=self.ndat_metamodel,
            **self.metamodel_kwargs,
        )

        # Construct the metamodel part:
        mm_output = metamodel.construct_eos(NEP_dict, return_proton_fraction=True)
        n_metamodel, p_metamodel, _, e_metamodel, _, mu_metamodel, cs2_metamodel, [n_metamodel_orig, proton_fraction, e_fraction, muon_fraction]  = (
            mm_output
        )

        # Convert units back for CSE initialization
        n_metamodel = n_metamodel / utils.fm_inv3_to_geometric
        p_metamodel = p_metamodel / utils.MeV_fm_inv3_to_geometric
        e_metamodel = e_metamodel / utils.MeV_fm_inv3_to_geometric

        # Combine metamodel and CSE data
        n = n_metamodel
        p = p_metamodel
        e = e_metamodel
        mu = mu_metamodel
        cs2 = cs2_metamodel

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return ns, ps, hs, es, dloge_dlogps, mu, cs2, [n_metamodel_orig, proton_fraction, e_fraction, muon_fraction]