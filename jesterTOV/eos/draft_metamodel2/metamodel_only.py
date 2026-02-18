r"""Meta-model EOS only - backward compatible wrapper.

This class is kept for backward compatibility. Use MetaModel_EOS_model with
return_extra=True to get the same functionality.
"""

from jaxtyping import Array, Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.draft_metamodel2.base import MetaModel_EOS_model


class MetaModel_only(Interpolate_EOS_model):
    r"""
    Meta-model EOS only, used to extract microscopic properties in intermediate density (between ~0.5-2 n_sat).

    Note:
        This class is deprecated. Use MetaModel_EOS_model with return_extra=True instead.
    """

    def __init__(
        self,
        nsat: Float = 0.16,
        nmin_MM_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_metamodel: Int = 100,
        ndat_CSE: Int = 100,
        calculate_durca: bool = False,
        **metamodel_kwargs,
    ):
        r""" """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_MM_nsat = nmin_MM_nsat
        self.ndat_metamodel = ndat_metamodel
        self.metamodel_kwargs = metamodel_kwargs
        self.calculate_durca = calculate_durca

    def construct_eos(
        self,
        NEP_dict: dict,
        ngrids: Float[Array, "n_grid_point"] | None = None,
        cs2grids: Float[Array, "n_grid_point"] | None = None,
    ) -> tuple:
        r"""
        Construct the EOS

        Args:
            NEP_dict (dict): Dictionary with the NEP keys to be passed to the metamodel EOS class.
            ngrids: Ignored (kept for backward compatibility).
            cs2grids: Ignored (kept for backward compatibility).

        Returns:
            tuple: EOS quantities (see Interpolate_EOS_model), as well as the chemical potential,
                speed of sound, and extra dict with metamodel-specific quantities.
        """

        # Initializate the MetaModel part up to n_break
        metamodel = MetaModel_EOS_model(
            nsat=self.nsat,
            nmin_MM_nsat=self.nmin_MM_nsat,
            nmax_nsat=NEP_dict["nbreak"] / self.nsat,
            ndat=self.ndat_metamodel,
            calculate_durca=self.calculate_durca,
            **self.metamodel_kwargs,
        )

        # Construct the metamodel part with return_extra=True
        mm_output = metamodel.construct_eos(
            NEP_dict, return_extra=True, calculate_durca=self.calculate_durca
        )
        (
            n_metamodel,
            p_metamodel,
            _,
            e_metamodel,
            _,
            mu_metamodel,
            cs2_metamodel,
            extra,
        ) = mm_output

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

        # Return extra dict for backward compatibility
        return ns, ps, hs, es, dloge_dlogps, mu, cs2, extra

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of parameters required by MetaModel_only.

        Returns:
            list[str]: NEP parameters + nbreak
                ["E_sat", "K_sat", "Q_sat", "Z_sat", "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym", "nbreak"]
        """
        return [
            "E_sat",
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
            "nbreak",
        ]
