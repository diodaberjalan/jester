r"""Skyrme EOS with Gaussian peak Constant Speed-of-sound Extensions (peakCSE)."""

import jax.numpy as jnp
from jaxtyping import Float, Int

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.skryme.base import (
    create_skryme_for_extension,
)


class Skryme_with_peakCSE_EOS_model(Interpolate_EOS_model):
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
        nmin_Skryme_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat_skryme: Int = 100,
        ndat_CSE: Int = 100,
        **skryme_kwargs,
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
            nmin_Skryme_nsat (Float, optional):
                Starting density for Skyrme region as fraction of :math:`n_0`.
                Must be above crust-core transition. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Should extend to densities where pQCD limit is approached. Defaults to 12.
            ndat_skryme (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values give smoother Skyrme interpolation. Defaults to 100.
            ndat_CSE (Int, optional):
                Number of density points for peakCSE region discretization.
                Controls resolution of phase transition and pQCD approach modeling. Defaults to 100.
            **skryme_kwargs:
                Additional keyword arguments passed to the underlying Skryme_EOS_model.
                Includes parameters like crust_name, etc. See Skryme_EOS_model.__init__.

        See Also:
            Skryme_EOS_model.__init__ : Base Skyrme parameters
            construct_eos : Method that defines peakCSE parameters and break density
        """

        self.nmax = nmax_nsat * nsat
        self.ndat_CSE = ndat_CSE
        self.nsat = nsat
        self.nmin_Skryme_nsat = nmin_Skryme_nsat
        self.ndat_skryme = ndat_skryme
        self.skryme_kwargs = skryme_kwargs

        # Store proton_fraction setting from skryme_kwargs for later use
        self.proton_fraction_setting = skryme_kwargs.get("proton_fraction", "exact")

    def construct_eos(
        self,
        params: dict,
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ):
        r"""
        Construct the complete EOS using Skyrme + peakCSE extensions.

        This method builds the full EOS by combining the Skyrme approach with
        peakCSE extensions that model phase transitions through Gaussian peaks
        and approach the pQCD conformal limit at high densities.

        Args:
            params (dict): Combined parameter dictionary containing both INM and peakCSE parameters.
                INM parameters (see Skryme_EOS_model.construct_eos):
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
            return_extra (bool, optional): If True, return extra Skyrme-specific quantities
                (proton fraction, lepton fractions, DURCA density) in the output.
                Defaults to False.
            calculate_durca (bool | None, optional): If True, calculate the Direct Urca threshold density.
                If None, uses the default from the skryme instance. Defaults to None.

        Returns:
            tuple: Complete EOS data containing:

                - **ns**: Number densities [geometric units]
                - **ps**: Pressures [geometric units]
                - **hs**: Specific enthalpies [geometric units]
                - **es**: Energy densities [geometric units]
                - **dloge_dlogps**: Logarithmic derivative :math:`\frac{d\ln\varepsilon}{d\ln p}`
                - **mu**: Chemical potential [geometric units]
                - **cs2**: Speed of sound squared including peakCSE structure
                - **extra** (if return_extra=True): Dict with Skyrme-specific quantities

        Note:
            The peakCSE speed of sound follows:
            :math:`c^2_s &= c^2_{s,{\rm break}} + \frac{\frac{1}{3} - c^2_{s,{\rm break}}}{1 + e^{-l_{\rm sig}(n - n_{\rm sig})}} + c^2_{s,{\rm peak}}e^{-\frac{1}{2}\left(\frac{n - n_{\rm peak}}{\sigma_{\rm peak}}\right)^2}`

            This ensures smooth transitions, realistic phase transition modeling,
            and asymptotic consistency with the pQCD conformal limit :math:`c_s^2 = 1/3`.
        """

        # Extract INM and peakCSE parameters from combined params dict
        # Get required INM parameters for Skryme
        INM_keys = [
            "t2", "t4", "x0", "x1", "x4",
            "alph", "beta", "gamma", "kfsat", "av",
            "J", "meffs", "meffv", "Kinf", "eNMhd", "nbreak"
        ]
        INM_dict = {k: params[k] for k in INM_keys if k in params}

        # Get peakCSE parameters
        peakCSE_keys = [
            "gaussian_peak", "gaussian_mu", "gaussian_sigma",
            "logit_growth_rate", "logit_midpoint"
        ]
        peakCSE_dict = {k: params[k] for k in peakCSE_keys if k in params}

        # Get nbreak for use in skryme creation
        nbreak = INM_dict["nbreak"]

        # Use helper to create fresh skryme instance limited to nbreak
        # This ensures proton_fraction setting is properly propagated
        skryme = create_skryme_for_extension(
            nsat=self.nsat,
            nmin_Skryme_nsat=self.nmin_Skryme_nsat,
            nbreak=nbreak,
            ndat=self.ndat_skryme,
            skryme_kwargs=self.skryme_kwargs,
            proton_fraction_setting=self.proton_fraction_setting,
        )

        # Construct the Skyrme part:
        skryme_output = skryme.construct_eos(
            INM_dict, return_extra=True, calculate_durca=calculate_durca
        )

        # Handle both return_extra=True and return_extra=False cases
        if return_extra:
            n_skryme, p_skryme, _, e_skryme, _, mu_skryme, cs2_skryme, extra = skryme_output
        else:
            (
                n_skryme,
                p_skryme,
                _,
                e_skryme,
                _,
                mu_skryme,
                cs2_skryme,
            ) = skryme_output
            extra = None

        # Convert units back for CSE initialization
        n_skryme = n_skryme / utils.fm_inv3_to_geometric
        p_skryme = p_skryme / utils.MeV_fm_inv3_to_geometric
        e_skryme = e_skryme / utils.MeV_fm_inv3_to_geometric

        # Get values at break density
        p_break = jnp.interp(INM_dict["nbreak"], n_skryme, p_skryme)
        e_break = jnp.interp(INM_dict["nbreak"], n_skryme, e_skryme)
        mu_break = jnp.interp(INM_dict["nbreak"], n_skryme, mu_skryme)
        cs2_break = jnp.interp(INM_dict["nbreak"], n_skryme, cs2_skryme)

        # Define the speed-of-sound of the extension portion
        # the model is taken from arXiv:1812.08188
        offset = self.offset_calc(INM_dict["nbreak"], cs2_break, peakCSE_dict)
        cs2_extension_function = lambda x: (
            peakCSE_dict["gaussian_peak"]
            * jnp.exp(
                -0.5
                * (
                    (x - peakCSE_dict["gaussian_mu"]) ** 2
                    / peakCSE_dict["gaussian_sigma"] ** 2
                )
            )
            + offset
            + (
                (1.0 / 3.0 - offset)
                / (
                    1.0
                    + jnp.exp(
                        -peakCSE_dict["logit_growth_rate"]
                        * (x - peakCSE_dict["logit_midpoint"])
                    )
                )
            )
        )

        # Compute n, p, e for peakCSE (number densities in unit of fm^-3)
        n_CSE = jnp.logspace(
            jnp.log10(INM_dict["nbreak"]), jnp.log10(self.nmax), num=self.ndat_CSE
        )
        cs2_CSE = cs2_extension_function(n_CSE)

        # We add a very small number to avoid problems with duplicates below
        mu_CSE = mu_break * jnp.exp(utils.cumtrapz(cs2_CSE / n_CSE, n_CSE)) + 1e-6
        p_CSE = p_break + utils.cumtrapz(cs2_CSE * mu_CSE, n_CSE) + 1e-6
        e_CSE = e_break + utils.cumtrapz(mu_CSE, n_CSE) + 1e-6

        # Combine Skyrme and CSE data
        n = jnp.concatenate((n_skryme, n_CSE))
        p = jnp.concatenate((p_skryme, p_CSE))
        e = jnp.concatenate((e_skryme, e_CSE))

        mu = jnp.concatenate((mu_skryme, mu_CSE))
        cs2 = jnp.concatenate((cs2_skryme, cs2_CSE))

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        if return_extra:
            return ns, ps, hs, es, dloge_dlogps, mu, cs2, extra
        return ns, ps, hs, es, dloge_dlogps, mu, cs2

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
