r"""Skyrme EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE)."""

import jax.numpy as jnp
from jaxtyping import Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.eos.skyrme.base import (
    create_skyrme_for_extension,
)


class Skyrme_with_peakCSE_EOS_model(Interpolate_EOS_model):
    r"""
    Skyrme EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE).

    This class implements a sophisticated CSE parametrization based on the peakCSE model,
    which combines a Gaussian peak structure with logistic growth to model phase transitions
    while ensuring asymptotic consistency with perturbative QCD (pQCD) at the highest densities.

    **Mathematical Framework:**
    The speed of sound squared is parametrized as:

    .. math::
        c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}

    **Reference:** Greif:2018njt, arXiv:1812.08188

    Note:
        The peakCSE model provides greater physical realism than simple piecewise-constant
        CSE by incorporating smooth transitions and theoretically motivated high-density behavior.
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
        Initialize the Skyrme with peakCSE extensions for realistic phase transition modeling.

        This constructor sets up the peakCSE model that combines Skyrme physics at
        low-to-intermediate densities with sophisticated Gaussian peak + logistic growth
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
                Should extend to densities where pQCD limit is approached. Defaults to 12.
            max_nbreak_nsat (Float | None, optional):
                Maximum value of nbreak prior in units of :math:`n_0`.
                Used to set the upper limit for the Skyrme region.
                If None, defaults to nmax_nsat. Defaults to None.
            ndat_skyrme (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values give smoother Skyrme interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for peakCSE region discretization.
                Controls resolution of phase transition and pQCD approach modeling. Defaults to 100.
            **skyrme_kwargs:
                Additional keyword arguments passed to the underlying Skyrme_EOS_model.
                Includes parameters like crust_name, etc. See Skyrme_EOS_model.__init__.

        See Also:
            Skyrme_EOS_model.__init__ : Base Skyrme parameters
            construct_eos : Method that defines peakCSE parameters and break density
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skyrme_nsat = nmin_Skyrme_nsat
        self.ndat_skyrme = ndat_skyrme
        self.skyrme_kwargs = skyrme_kwargs

        # Store proton_fraction setting from skyrme_kwargs for later use
        self.proton_fraction_setting = skyrme_kwargs.get("proton_fraction", "exact")

        # Use max_nbreak_nsat if provided, otherwise default to nmax_nsat
        metamodel_max_nsat = (
            max_nbreak_nsat if max_nbreak_nsat is not None else nmax_nsat
        )

        # Create the Skyrme instance once with max density from nbreak prior
        from .base import Skyrme_EOS_model

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
    ) -> EOSData:
        r"""
        Construct the complete EOS using Skyrme + peakCSE extensions.

        This method builds the full EOS by combining the Skyrme approach with
        peakCSE extensions that model phase transitions through Gaussian peaks
        and approach the pQCD conformal limit at high densities.

        Args:
            params (dict): Combined parameter dictionary containing both INM and peakCSE parameters.
                INM parameters (see Skyrme_EOS_model.construct_eos):
                    - **t2**, **t4**: Input parameters
                    - **x0**, **x1**, **x4**: Exchange parameter inputs
                    - **alph**, **beta**, **gamma**: Density dependence exponents
                    - **kfsat**: Fermi momentum at saturation
                    - **av**: Saturation energy per nucleon
                    - **J**: Symmetry energy
                    - **meffs**, **meffv**: Effective masses
                    - **Kinf**: Incompressibility
                    - **eNMhd**: Energy density at high density
                    - **nbreak**: Break density between Skyrme and peakCSE regions

                peakCSE parameters:
                    - **gaussian_peak** (float): Amplitude :math:`A` of the Gaussian peak
                    - **gaussian_mu** (float): Peak location :math:`\mu` [:math:`\mathrm{fm}^{-3}`]
                    - **gaussian_sigma** (float): Peak width :math:`\sigma` [:math:`\mathrm{fm}^{-3}`]
                    - **logit_growth_rate** (float): Growth rate :math:`k` for pQCD approach
                    - **logit_midpoint** (float): Midpoint density :math:`n_{\mathrm{mid}}` for logistic transition

        Returns:
            EOSData: Complete EOS with all required arrays in geometric units.
        """

        # Extract nbreak for use in skyrme creation (density in fm^-3)
        nbreak = params["nbreak"]

        # Construct the Skyrme part using pre-instantiated skyrme instance
        skyrme_output = self.skyrme.construct_eos(params, return_extra=True)

        # Extract quantities from skyrme output (8 fields if return_extra=True)
        (n_skyrme, p_skyrme, _, e_skyrme, _, mu_skyrme, cs2_skyrme, extra) = (
            skyrme_output
        )

        # Convert units back for CSE initialization
        n_skyrme = n_skyrme / utils.fm_inv3_to_geometric
        p_skyrme = p_skyrme / utils.MeV_fm_inv3_to_geometric
        e_skyrme = e_skyrme / utils.MeV_fm_inv3_to_geometric

        # Re-interpolate to actual nbreak
        n_skyrme_interp = jnp.linspace(
            n_skyrme[0], nbreak, self.ndat_skyrme, endpoint=True
        )
        p_skyrme_interp = jnp.interp(n_skyrme_interp, n_skyrme, p_skyrme)
        e_skyrme_interp = jnp.interp(n_skyrme_interp, n_skyrme, e_skyrme)
        mu_skyrme_interp = jnp.interp(n_skyrme_interp, n_skyrme, mu_skyrme)
        cs2_skyrme_interp = jnp.interp(n_skyrme_interp, n_skyrme, cs2_skyrme)

        # Get values at break density
        p_break = jnp.interp(nbreak, n_skyrme_interp, p_skyrme_interp)
        e_break = jnp.interp(nbreak, n_skyrme_interp, e_skyrme_interp)
        mu_break = jnp.interp(nbreak, n_skyrme_interp, mu_skyrme_interp)
        cs2_break = jnp.interp(nbreak, n_skyrme_interp, cs2_skyrme_interp)

        # Define the speed-of-sound of the extension portion
        # the model is taken from arXiv:1812.08188
        offset = self.offset_calc(nbreak, cs2_break, params)
        cs2_extension_function = lambda x: (
            params["gaussian_peak"]
            * jnp.exp(
                -0.5
                * (
                    (x - params["gaussian_mu"]) ** 2
                    / params["gaussian_sigma"] ** 2
                )
            )
            + offset
            + (
                (1.0 / 3.0 - offset)
                / (
                    1.0
                    + jnp.exp(
                        -params["logit_growth_rate"]
                        * (x - params["logit_midpoint"])
                    )
                )
            )
        )

        # Compute n, p, e for peakCSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(
            jnp.log10(nbreak), jnp.log10(self.nmax), num=self.ndat_CSE
        )
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = (
            mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        )
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine Skyrme and CSE data
        n = jnp.concatenate((n_skyrme_interp, n_CSE))
        p = jnp.concatenate((p_skyrme_interp, p_CSE))
        e = jnp.concatenate((e_skyrme_interp, e_CSE))

        mu = jnp.concatenate((mu_skyrme_interp, mu_CSE))
        cs2 = jnp.concatenate((cs2_skyrme_interp, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            mu=mu,
            cs2=cs2,
            extra_constraints=extra,
        )
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = (
            mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        )
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine Skyrme and CSE data
        n = jnp.concatenate((n_skyrme, n_CSE))
        p = jnp.concatenate((p_skyrme, p_CSE))
        e = jnp.concatenate((e_skyrme, e_CSE))

        mu = jnp.concatenate((mu_skyrme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skyrme, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        if return_extra:
            return ns, ps, hs, es, dloge_dlogps, mu, cs2, extra # type: ignore[return-value]

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            mu=mu,
            cs2=cs2,
            extra_constraints=extra,
        )

    def offset_calc(self, nbreak, cs2_break, peakCSE_dict):
        r"""
        Calculate offset for peakCSE extension.

        This ensures continuity at the break density.

        Args:
            nbreak: Break density
            cs2_break: Speed of sound at break density
            peakCSE_dict: peakCSE parameters

        Returns:
            Offset value for the extension function
        """
        gaussian_part = peakCSE_dict["gaussian_peak"] * jnp.exp(
            -0.5
            * (nbreak - peakCSE_dict["gaussian_mu"]) ** 2
            / peakCSE_dict["gaussian_sigma"] ** 2
        )
        exp_part = jnp.exp(
            -peakCSE_dict["logit_growth_rate"]
            * (nbreak - peakCSE_dict["logit_midpoint"])
        )
        offset = ((1.0 + exp_part) * (cs2_break - gaussian_part) - 1.0 / 3.0) / exp_part
        return offset

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of parameters required by Skyrme with peakCSE.

        Returns:
            list[str]: INM parameters + nbreak + peakCSE parameters
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
            "gaussian_peak",
            "gaussian_mu",
            "gaussian_sigma",
            "logit_growth_rate",
            "logit_midpoint",
        ]
