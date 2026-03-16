r"""Skyrme equation of state for nuclear matter."""

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, Int
from typing import Union

from jesterTOV import utils
from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.eos.crust import Crust
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class Skyrme_EOS_model(Interpolate_EOS_model):
    r"""
    Skyrme equation of state for nuclear matter.

    This class implements the Skyrme energy density functional approach for constructing
    neutron star equations of state. The Skyrme force parameters are derived from
    infinite nuclear matter (INM) properties through an inverse problem solver.

    The Skyrme EOS uses a density-dependent effective interaction with the form:

    .. math::
        V_{12} = t_0 (1 + x_0 P_\sigma) + \frac{1}{2} t_1 (1 + x_1 P_\sigma) (\mathbf{k}^2 + \mathbf{k}'^2)
        + t_2 (1 + x_2 P_\sigma) \mathbf{k}' \cdot \mathbf{k}
        + \frac{1}{6} t_3 (1 + x_3 P_\sigma) \rho^\alpha + \text{spin-gradient terms}

    **Reference:** Skyrme (1959), Vautherin & Brink (1972)
    """

    def __init__(
        self,
        # density parameters
        nsat: Float = 0.16,
        nmin_Skyrme_nsat: Float = 0.12 / 0.16,
        nmax_nsat: Float = 12,
        ndat: Int = 200,
        # crust parameters
        crust_name: str = "DH",
        max_n_crust_nsat: Float = 0.5,
        min_n_crust_nsat: Float = 2e-13,
        ndat_spline: Int = 10,
        # proton fraction
        proton_fraction: bool | float | str | None = None,
        calculate_durca: bool = False,
    ):
        r"""
        Initialize the Skyrme EOS with nuclear matter parameters.

        The Skyrme EOS is constructed by converting INM properties to Skyrme force
        parameters, then computing the energy density functional to obtain the EOS.

        Args:
            nsat (Float, optional):
                Nuclear saturation density :math:`n_0` [:math:`\mathrm{fm}^{-3}`].
                The density at which symmetric nuclear matter reaches minimum energy.
                Defaults to 0.16.
            nmin_Skyrme_nsat (Float, optional):
                Starting density for Skyrme region as fraction of :math:`n_0`.
                Must be above the crust-core transition density. Defaults to 0.75 (= 0.12/0.16).
            nmax_nsat (Float, optional):
                Maximum density for EOS construction in units of :math:`n_0`.
                Determines the high-density reach of the neutron star model. Defaults to 12.
            ndat (Int, optional):
                Number of density points for Skyrme region discretization.
                Higher values provide smoother interpolation at computational cost. Defaults to 200.
            crust_name (str, optional):
                Crust model name (e.g., 'DH', 'BPS') or path to custom .npz file.
                The crust provides low-density EOS data below nuclear saturation. Defaults to 'DH'.
            max_n_crust_nsat (Float, optional):
                Maximum crust density as fraction of :math:`n_0`. Defines the crust-core
                transition region where spline matching occurs. Defaults to 0.5.
            ndat_spline (Int, optional):
                Number of points for smooth spline interpolation across crust-core transition.
                Ensures thermodynamic consistency and causality preservation. Defaults to 10.
            proton_fraction (bool | float | None, optional):
                Proton fraction treatment strategy:

                - None: Calculate :math:`\beta`-equilibrium (charge neutrality + weak equilibrium)
                - float: Use fixed proton fraction value throughout the star
                - bool: Use simplified uniform composition model

                :math:`\beta`-equilibrium is the physical condition for neutron star matter.
                Defaults to None (exact with muons).

        Note:
            The Skyrme EOS calculates the energy density functional using automatic
            differentiation on the Skyrme functional. The INM parameters are converted
            to Skyrme force parameters (t0-t5, x0-x5) through a linear system solver.
        """

        # Save as attributes
        self.nsat = nsat
        self.nmin_Skyrme_nsat = nmin_Skyrme_nsat
        self.nmax_nsat = nmax_nsat
        self.ndat = ndat
        self.max_n_crust_nsat = max_n_crust_nsat
        self.min_n_crust_nsat = min_n_crust_nsat
        self.ndat_spline = ndat_spline

        # Proton fraction configuration
        # Default: exact beta-equilibrium with muons (proton_fraction=None)
        # "approx": approximate beta-equilibrium without muons
        # float: fixed proton fraction value
        # "exact": explicit exact calculation (same as default)
        if isinstance(proton_fraction, float):
            self.proton_fraction_val = proton_fraction
            self.proton_fraction = lambda x, y: self.proton_fraction_val
            self.with_muon = False
            logger.info(f"Proton fraction fixed to {self.proton_fraction_val}")
        elif proton_fraction == 'approx':
            # Approximate beta-equilibrium without muons (legacy behavior)
            self.proton_fraction = lambda x, y: self.compute_proton_fraction(x, y)
            self.with_muon = False
        elif proton_fraction == 'exact' or proton_fraction is None:
            # Exact beta-equilibrium with muons (default behavior)
            self.proton_fraction = lambda x, y: self.compute_proton_fraction_exact(y)
            self.with_muon = True
        else:
            # Default to exact calculation with muons
            self.proton_fraction = lambda x, y: self.compute_proton_fraction_exact(y)
            self.with_muon = True

        # DURCA calculation configuration
        self.calculate_durca = calculate_durca
        self.durca_density = {"ye": jnp.nan, "ym": jnp.nan, "nb_durca": jnp.nan}

        # Load and preprocess the crust
        crust = Crust(
            crust_name,
            min_density=min_n_crust_nsat * nsat,
            max_density=max_n_crust_nsat * nsat,
            filter_zero_pressure=True,
        )
        self.ns_crust: Float[Array, "n_crust"] = crust.n
        self.ps_crust: Float[Array, "n_crust"] = crust.p
        self.es_crust: Float[Array, "n_crust"] = crust.e
        self.mu_lowest: Float = crust.mu_lowest
        self.cs2_crust: Float[Array, "n_crust"] = crust.cs2

        # Make sure the Skyrme model starts above the crust
        self.max_n_crust = self.ns_crust[-1]

        # Create density arrays
        self.nmax = nmax_nsat * self.nsat
        self.ndat = ndat
        self.nmin_Skyrme = self.nmin_Skyrme_nsat * self.nsat
        self.n_Skyrme = jnp.linspace(
            self.nmin_Skyrme, self.nmax, self.ndat, endpoint=False
        )
        self.ns_spline = jnp.append(self.ns_crust, self.n_Skyrme)
        self.n_connection = jnp.linspace(
            self.max_n_crust + 1e-5, self.nmin_Skyrme, self.ndat_spline, endpoint=False
        )

    def solve_skyrme_system(self, p: dict) -> tuple:
        r"""
        Solve for Skyrme parameters from INM properties.

        This solves the inverse problem: given INM properties, find the Skyrme
        force parameters (t0-t5, x0-x5) that reproduce them.

        Args:
            p (dict): Dictionary of INM parameters:
                - t2, t4: Input parameters
                - x0, x1, x4: Exchange parameter inputs
                - alph: Maps to sigma (density dependence exponent)
                - beta, gamma: Density dependence exponents
                - kfsat: Fermi momentum at saturation
                - av: Saturation energy per nucleon
                - J: Symmetry energy
                - meffs, meffv: Effective masses (scalar, vector)
                - Kinf: Incompressibility
                - eNMhd: Energy density at high density

        Returns:
            tuple: (t_final, x_final) where each is an array of 6 Skyrme parameters
        """
        # Constants
        pi = jnp.pi
        hbm = 20.73553000
        hmt = 2 / (1/hbm + 1/hbm)

        # Derived densities
        kf = p['kfsat']
        rhosat = 2 * kf**3 / (3 * pi**2)
        rhoHD = 1.0
        kfnHD = (3 * pi**2)**(1/3)

        # Density dependent powers
        r3 = rhosat**(p['alph'] + 1)
        r4 = rhosat**(p['beta'] + 1)
        r5 = rhosat**(p['gamma'] + 1)

        rHD3 = rhoHD**(p['alph'] + 1)
        rHD4 = rhoHD**(p['beta'] + 1)
        rHD5 = rhoHD**(p['gamma'] + 1)

        # --- System 1: Isoscalar Channel (4x4) ---
        a1_00 = 3 * rhosat / 8
        a1_01 = 3 * rhosat * kf**2 / 80
        a1_02 = r3 / 16
        a1_03 = 3/80 * r5 * kf**2

        a1_10 = 9 * rhosat / 8
        a1_11 = 5 * a1_01
        a1_12 = 3 * (p['alph'] + 1) * r3 / 16
        a1_13 = 3/80 * kf**2 * r5 * (5 + 3*p['gamma'])

        a1_20 = 0.0
        a1_21 = rhosat / (hmt * 16)
        a1_22 = 0.0
        a1_23 = (r5 / 16) / hmt

        a1_30 = 9 * rhosat / 4
        a1_31 = 3/4 * rhosat * kf**2
        a1_32 = 3/16 * (p['alph'] + 1) * (3*p['alph'] + 2) * r3
        a1_33 = 3/80 * (3*p['gamma'] + 5) * (3*p['gamma'] + 4) * r5 * kf**2

        A1 = jnp.array([
            [a1_00, a1_01, a1_02, a1_03],
            [a1_10, a1_11, a1_12, a1_13],
            [a1_20, a1_21, a1_22, a1_23],
            [a1_30, a1_31, a1_32, a1_33]
        ])

        b1_0 = p['av'] - hmt * 0.6 * kf**2 - (9 * p['t4'] * kf**2 * r4 / 80)
        b1_1 = -hmt * 1.2 * kf**2 - 3 * (3 * p['t4'] * kf**2 * r4 / 80 * (5 + 3*p['beta']))
        b1_2 = 1/p['meffs'] - 1 - (3 * p['t4'] * r4 / 16) / hmt
        b1_3 = p['Kinf'] - 6/5 * hmt * kf**2 - 9/80 * p['t4'] * kf**2 * r4 * (3*p['beta'] + 5) * (3*p['beta'] + 4)
        b1 = jnp.array([b1_0, b1_1, b1_2, b1_3])

        res1 = jax.scipy.linalg.solve(A1, b1)
        t0_sol, term_t1_x2, t3_sol, term_t5_x5 = res1[0], res1[1], res1[2], res1[3]

        # --- System 2: Isovector Channel (3x3) ---
        a2_00 = -(p['x1'] + 5/4) * rhosat * kf**2 / 8
        a2_01 = -t3_sol * r3 / 24
        a2_02 = -9/4 * kf**2 * r5 / 24

        a2_10 = rhosat * (5/4 + p['x1']) / 8 / hmt
        a2_11 = 0.0
        a2_12 = 3/4 * r5 / 8 / hmt

        a2_20 = -3 * (5/4 + p['x1']) * rhoHD * kfnHD**2 / 40
        a2_21 = -t3_sol * rHD3 / 24
        a2_22 = -9 * rHD5 * kfnHD**2 / 40 / 4

        A2 = jnp.array([
            [a2_00, a2_01, a2_02],
            [a2_10, a2_11, a2_12],
            [a2_20, a2_21, a2_22]
        ])

        b2_0 = p['J'] - hmt * kf**2 / 3 \
               + t0_sol * rhosat * (2 * p['x0'] + 1) / 8 \
               + p['t4'] / 8 * kf**2 * r4 * p['x4'] \
               - 1.25 * term_t5_x5 * r5 * kf**2 / 24 \
               - 0.25 * (5 * term_t1_x2 - 9 * p['t2']) * rhosat * kf**2 / 24 \
               + t3_sol * r3 / 48

        b2_1 = 1/p['meffv'] - 1 - (term_t5_x5 * r5 / 4 \
               + p['t4'] * (2 + p['x4']) * r4 \
               + rhosat * (3 * p['t2'] + term_t1_x2) / 4) / hmt / 8

        b2_2 = p['eNMhd'] - 0.6 * hmt * kfnHD**2 \
               - 3 * p['t4'] * (1 - p['x4']) * rHD4 * kfnHD**2 / 40 \
               - 0.25 * t0_sol * (1 - p['x0']) * rhoHD - t3_sol * rHD3 / 24 \
               - 9/40 * kfnHD**2 / 4 * (term_t5_x5 * rHD5 + rhoHD * (term_t1_x2 - p['t2']))
        b2 = jnp.array([b2_0, b2_1, b2_2])

        res2 = jax.scipy.linalg.solve(A2, b2)
        t1_sol, x3_sol, t5_sol = res2[0], res2[1], res2[2]

        # --- Recover x2 and x5 ---
        x2_sol = (term_t1_x2 / p['t2'] - 3 * t1_sol / p['t2'] - 5) / 4
        x5_sol = (term_t5_x5 / t5_sol - 5) / 4

        t_final = jnp.array([t0_sol, t1_sol, p['t2'], t3_sol, p['t4'], t5_sol])
        x_final = jnp.array([p['x0'], p['x1'], x2_sol, x3_sol, p['x4'], x5_sol])

        return t_final, x_final

    def eDenSky(self, ron: Array, rop: Array) -> Array:
        r"""
        Skyrme energy density as function of neutron and proton densities.

        Args:
            ron: Neutron number density [:math:`\mathrm{fm}^{-3}`]
            rop: Proton number density [:math:`\mathrm{fm}^{-3}`]

        Returns:
            Array: Energy density [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
        """
        # Get Skyrme parameters
        t = self.t_skyrme
        x = self.x_skyrme

        t0, t1, t2, t3, t4, t5 = t
        x0, x1, x2, x3, x4, x5 = x

        # Density dependence powers
        alph = self.alph
        beta = self.beta
        gamm = self.gamma

        # Total density
        ro = ron + rop

        # Kinetic energy density
        tau_factor = 0.6 * (3 * jnp.pi**2)**(2/3)
        taun = tau_factor * ron**(5/3)
        taup = tau_factor * rop**(5/3)

        # Kinetic energy term
        Ekin = (0.5 * utils.hbarc**2) * (taun / utils.m_n + taup / utils.m_p)

        # Interaction terms
        ro2 = ro**2
        r_sq_sum = rop**2 + ron**2
        tausum = taun + taup
        tau_mix = rop * taup + ron * taun

        # t0 term
        E0 = 0.25 * t0 * ((2 + x0) * ro2 - (2 * x0 + 1) * r_sq_sum)

        # t3 term
        E3 = (t3 * ro**alph / 24) * ((2 + x3) * ro2 - (2 * x3 + 1) * r_sq_sum)

        # Effective mass term
        c_eff_1 = t1 * (2 + x1) + t2 * (2 * self.t2p + x2)
        c_eff_2 = t2 * (2 * x2 + self.t2p) - t1 * (2 * x1 + 1)
        Eeff = 0.125 * (c_eff_1 * ro * tausum + c_eff_2 * tau_mix)

        # t4 term (spin-gradient)
        c_4_1 = 1 + 0.5 * x4
        c_4_2 = 0.5 + x4
        E4 = 0.25 * t4 * ro**beta * (c_4_1 * ro * tausum - c_4_2 * tau_mix)

        # t5 term (spin-gradient)
        c_5_1 = 1 + 0.5 * x5
        c_5_2 = 0.5 + x5
        E5 = 0.25 * t5 * ro**gamm * (c_5_1 * ro * tausum + c_5_2 * tau_mix)

        return Ekin + E0 + E3 + Eeff + E4 + E5

    def compute_proton_fraction_exact(
        self, n: Array
    ) -> tuple:
        r"""
        Compute proton fraction from exact beta-equilibrium with muons.

        This follows the same method as metamodel2.

        Args:
            n: Total baryon density [:math:`\mathrm{fm}^{-3}`]

        Returns:
            tuple: (proton_fraction, electron_fraction, muon_fraction)
        """
        def muElec(ne):
            ne_safe = ne #jnp.clip(ne, 1e-25, None)
            kfe = jnp.power(3*jnp.pi*jnp.pi*ne_safe, 1.0/3.0)
            xe = utils.hbarc * kfe / utils.m_e
            mue = utils.m_e * jnp.sqrt(1 + xe*xe)
            return mue

        def muMuon(nm):
            nmu_safe = jnp.clip(nm, 1e-12, None)
            kf = (3*jnp.pi*jnp.pi*nmu_safe)**(1/3)
            xm = utils.hbarc * kf / utils.m_mu
            mu = utils.m_mu * jnp.sqrt(1 + xm*xm)
            return mu
            
        def guess_val_p(n):
            n_per_nsat = n/utils.fm_inv3_to_geometric / 0.16
            return 1/30 * n_per_nsat + 1/60
        def guess_val_mu(n):
            n_per_nsat = n/utils.fm_inv3_to_geometric / 0.16
            return 0.0075 * n_per_nsat
            
        # Energy density as function of densities
        total_energy_density = lambda n_n, n_p: self.eDenSky(n_n, n_p)
        nu_p = jax.grad(total_energy_density, argnums=1)
        nu_n = jax.grad(total_energy_density, argnums=0)

        def betaHMnpe_optimistix(guessYe, nb):
            def fn(z, args=nb):
                y = z
                n_n = nb*(1-y)
                n_p = nb*y
                mue = muElec(nb * y)
                mun = nu_n(n_n, n_p) + utils.m_n
                mup = nu_p(n_n, n_p) + utils.m_p
                f = (mun - mup - mue)/mun
                return f
            guessYp = guess_val_p(nb)
            z0 = jnp.array(guessYp)

            # Use Newton with Dogleg fallback for robustness and speed
            sol = optx.root_find(fn, optx.Dogleg(rtol=1e-5, atol=1e-6), z0, throw=False, max_steps=1000)
            return sol.value
            

        def betaHMnpemu_optimistix(guess, nb):
            def fn(z, args):
                y1, y2 = z
                y = y1 + y2
                n_n = nb*(1-y)
                n_p = nb*y
                mun = nu_n(n_n, n_p) + utils.m_n
                mup = nu_p(n_n, n_p) + utils.m_p
                mue = muElec(nb * y1)
                mumu = muMuon(nb * y2)
                f1 = (mun - mup - mue)/mun
                f2 = (mumu - mue)/mue
                return f1, f2
            guess = [guess_val_p(nb),1.0e-9]
            z0 = jnp.array(guess)
            # Use Newton with Dogleg fallback for robustness and speed
            sol = optx.root_find(fn, optx.Newton(rtol=1e-5, atol=1e-6), z0, throw=False, max_steps=1000)
            return sol.value

        @jax.jit
        def calc_ye_all_jit(guess_val, nb_array):
            return jax.vmap(lambda nb: betaHMnpe_optimistix(guess_val, nb))(nb_array)

        @jax.jit
        def calc_conditional_fractions(guess_vec, nb_full, cond_mask, ye_arr):
            def compute_for_single(nb, has_muon, ye):
                result = jax.lax.cond(
                    has_muon,
                    lambda: betaHMnpemu_optimistix(guess_vec, nb),
                    lambda: jnp.array([ye, 0.0])
                )
                return result
            return jax.vmap(compute_for_single)(nb_full, cond_mask, ye_arr)

        guess = [0.04, 1.e-9]
        ye_arr = calc_ye_all_jit(guess[0], n)
        cond = muElec(n * ye_arr) > utils.m_mu
        final_arr = calc_conditional_fractions(guess, n, cond, ye_arr)

        yp_arr = final_arr[:, 0] + final_arr[:, 1]

        ye_array = jnp.array(final_arr[:, 0])
        ymu_array = jnp.array(final_arr[:, 1])
        yp_array = jnp.array(yp_arr)

        proton_fraction = yp_array
        electron_fraction = ye_array
        muon_fraction = ymu_array

        # Direct Urca calculation (same as metamodel2)
        if self.calculate_durca:
            x_e = electron_fraction / (electron_fraction + muon_fraction + 1e-25)
            x_DU = 1 / (1 + (1 + jnp.cbrt(x_e)) ** 3)
            x_DU_curve = jnp.array([n, x_DU])
            y_p_curve = jnp.array([n, yp_array])
            nb_durca, yp_durca = utils.get_curve_intersection(x_DU_curve, y_p_curve)
            ye = jnp.interp(nb_durca, n, ye_array)
            ym = jnp.interp(nb_durca, n, ymu_array)
            self.durca_density = {"ye": ye, "ym": ym, "nb_durca": nb_durca}

        return proton_fraction, electron_fraction, muon_fraction

    def compute_proton_fraction(
        self, coefficient_sym: list, n: Array
    ) -> Float[Array, "n_points"]:
        r"""
        Compute proton fraction from approximate beta-equilibrium without muons.

        This is a simplified version without muon contributions.

        Args:
            coefficient_sym: Not used (kept for interface compatibility)
            n: Total baryon density [:math:`\mathrm{fm}^{-3}`]

        Returns:
            Float[Array, "n_points"]: Proton fraction
        """
        # Simplified beta-equilibrium without muons
        def muElec(ne):
            ne_safe = ne#jnp.clip(ne, 1e-25, None)
            kfe = jnp.power(3*jnp.pi*jnp.pi*ne_safe, 1.0/3.0)
            xe = utils.hbarc * kfe / utils.m_e
            mue = utils.m_e * jnp.sqrt(1 + xe*xe)
            return mue
        def guess_val_p(n):
            n_per_nsat = n/utils.fm_inv3_to_geometric / 0.16
            return 1/30 * n_per_nsat + 1/60
        total_energy_density = lambda n_n, n_p: self.eDenSky(n_n, n_p)
        nu_p = jax.grad(total_energy_density, argnums=1)
        nu_n = jax.grad(total_energy_density, argnums=0)

        def betaHMnpe(guessYe, nb):
            def fn(z, args=nb):
                y = z
                n_n = nb*(1-y)
                n_p = nb*y
                mue = muElec(nb * y)
                mun = nu_n(n_n, n_p) + utils.m_n
                mup = nu_p(n_n, n_p) + utils.m_p
                f = (mun - mup - mue)/mun
                return f
            guessYp = guess_val_p(nb)
            z0 = jnp.array(guessYp)

            # Use Newton with Dogleg fallback for robustness and speed
            sol = optx.root_find(fn, optx.Newton(rtol=1e-5, atol=1e-6), z0, throw=False, max_steps=1000)
            return sol.value

        @jax.jit
        def calc_ye_all_jit(guess_val, nb_array):
            return jax.vmap(lambda nb: betaHMnpe(guess_val, nb))(nb_array)

        guess = 0.04
        proton_fraction = calc_ye_all_jit(guess, n)

        return proton_fraction

    # type: ignore[override]
    def construct_eos(
        self,
        params: dict,
        return_extra: bool = False,
        calculate_durca: bool | None = None,
    ) -> Union[EOSData, tuple]:
        r"""
        Construct the complete equation of state from INM parameters.

        This method builds the full EOS by combining the crust model with the
        Skyrme core, ensuring thermodynamic consistency and causality.

        Args:
            params (dict): Infinite nuclear matter parameters including:
                - **t2**: Input parameter
                - **t4**: Input parameter
                - **x0**, **x1**, **x4**: Exchange parameter inputs
                - **alph**: Maps to sigma (density dependence exponent)
                - **beta**, **gamma**: Density dependence exponents
                - **kfsat**: Fermi momentum at saturation
                - **av**: Saturation energy per nucleon
                - **J**: Symmetry energy
                - **meffs**, **meffv**: Effective masses (scalar, vector)
                - **Kinf**: Incompressibility
                - **eNMhd**: Energy density at high density
            return_extra (bool, optional): If True, returns a tuple with extra Skyrme-specific quantities.
                If False (default), returns an EOSData object for inference compatibility.
            calculate_durca (bool | None, optional): If True, calculate the Direct Urca threshold density.
                If None, uses the instance default.

        Returns:
            Union[EOSData, tuple]:
                - If ``return_extra=False`` (default): :class:`EOSData` object for inference compatibility.
                - If ``return_extra=True``: tuple ``(ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)``
                  where ``extra`` is a dict with Skyrme-specific quantities.

                The EOSData contains:
                    - **ns**: Number densities [geometric units]
                    - **ps**: Pressures [geometric units]
                    - **hs**: Specific enthalpies [geometric units]
                    - **es**: Energy densities [geometric units]
                    - **dloge_dlogps**: Logarithmic derivative
                    - **mu**: Chemical potential [geometric units]
                    - **cs2**: Speed of sound squared
                    - **extra_constraints**: dict with proton fraction, lepton fractions, and DURCA threshold
        """

        # Handle calculate_durca: use instance default if not provided
        if calculate_durca is None:
            calculate_durca = getattr(self, 'calculate_durca', False)
        self.calculate_durca = calculate_durca
        self.durca_density = {"ye": jnp.nan, "ym": jnp.nan, "nb_durca": jnp.nan}

        # Solve for Skyrme parameters
        self.t_skyrme, self.x_skyrme = self.solve_skyrme_system(params)

        # Store density dependence parameters
        self.alph = params.get('alph', 0.2)
        self.beta = params.get('beta', 0.0833)
        self.gamma = params.get('gamma', 0.25)
        self.t2p = 1.0  # Standard value

        # Compute proton fraction
        if self.with_muon:
            proton_fraction, e_fraction, muon_fraction = self.proton_fraction(  # type: ignore[misc]
                None, self.n_Skyrme
            )
        else:
            proton_fraction = self.proton_fraction(None, self.n_Skyrme)
            e_fraction = None
            muon_fraction = None

        # Calculate energy density for each density point
        n_n = self.n_Skyrme * (1 - proton_fraction)  # type: ignore[operator]
        n_p = self.n_Skyrme * proton_fraction

        e_Skyrme = self.eDenSky(n_n, n_p)

        # Calculate pressure using thermodynamic identity
        # P = n * dE/dn - E (at fixed composition for beta-equilibrium)
        # But we need to account for the composition change with density
        # Using automatic differentiation for the pressure
        def pressure_from_eos(n, yp):
            n_n = n * (1 - yp)
            n_p = n * yp
            e = self.eDenSky(n_n, n_p)
            # dE/dn at fixed Y_p
            de_dn = jax.grad(lambda n_val: self.eDenSky(n_val * (1 - yp), n_val * yp))(n)
            p = n * de_dn - e
            return p

        p_Skyrme = jax.vmap(pressure_from_eos)(self.n_Skyrme, proton_fraction)

        # Add lepton contributions to pressure
        if self.with_muon and e_fraction is not None:
            # Ensure lepton fractions are non-negative to avoid NaNs in powers
            e_fraction_safe = e_fraction #jnp.clip(e_fraction, 1e-25, None)
            muon_fraction_arr = muon_fraction # jnp.clip(proton_fraction - e_fraction, 1e-25, None)

            # Electron pressure
            K_Fe = (3.0 * jnp.pi**2 * self.n_Skyrme * e_fraction_safe) ** (1.0/3.0) * utils.hbarc
            C_e = utils.m_e**4 / (8.0 * jnp.pi**2) / utils.hbarc**3
            x_e = K_Fe / utils.m_e
            f_e = x_e * (1 + 2 * x_e**2) * jnp.sqrt(1 + x_e**2) - jnp.arcsinh(x_e)
            e_electron = C_e * f_e
            p_electron = -e_electron + 8.0/3.0 * C_e * x_e**3 * jnp.sqrt(1 + x_e**2)

            # Muon pressure
            K_Fmu = (3.0 * jnp.pi**2 * self.n_Skyrme * muon_fraction_arr) ** (1.0/3.0) * utils.hbarc
            C_mu = utils.m_mu**4 / (8.0 * jnp.pi**2) / utils.hbarc**3
            x_mu = K_Fmu / utils.m_mu
            f_mu = x_mu * (1 + 2 * x_mu**2) * jnp.sqrt(1 + x_mu**2) - jnp.arcsinh(x_mu)
            e_muon = C_mu * f_mu
            p_muon = -e_muon + 8.0/3.0 * C_mu * x_mu**3 * jnp.sqrt(1 + x_mu**2)

            p_lepton = p_electron + p_muon
            e_lepton = e_electron + e_muon

            # Use safe fractions for rest mass density later too
            muon_final_arr = muon_fraction_arr
            e_final_arr = e_fraction_safe
        else:
            p_lepton = jnp.zeros_like(self.n_Skyrme)
            e_lepton = jnp.zeros_like(self.n_Skyrme)
            e_final_arr = None
            muon_final_arr = None

        p_total = p_Skyrme + p_lepton

        # Add rest mass energy density (same as draft_skyrme)
        # This is critical for proper EOS units
        rest_mass_energy_density = n_n * utils.m_n + n_p * utils.m_p
        if self.with_muon and e_final_arr is not None:
            n_mu = self.n_Skyrme * muon_final_arr
            n_e = self.n_Skyrme * e_final_arr
            rest_mass_energy_density += n_mu * utils.m_mu + n_e * utils.m_e

        e_total = e_Skyrme + e_lepton + rest_mass_energy_density

        # Ensure proton_fraction is array for compute_cs2
        if jnp.ndim(proton_fraction) == 0:  # type: ignore[arg-type]
            proton_fraction_arr = jnp.full_like(self.n_Skyrme, proton_fraction)  # type: ignore[arg-type]
        else:
            proton_fraction_arr = proton_fraction

        # Compute cs2 including lepton contributions
        cs2_Skyrme = self.compute_cs2(  # type: ignore[arg-type]
            self.n_Skyrme, p_total, e_total, proton_fraction_arr, e_final_arr
        )

        # Spline for speed of sound for the connection region
        cs2_spline = jnp.append(jnp.array(self.cs2_crust), cs2_Skyrme)

        cs2_connection = utils.cubic_spline(
            self.n_connection, self.ns_spline, cs2_spline
        )
        # cs2_connection = jnp.clip(cs2_connection, 1e-5, 1.0)

        # Concatenate the arrays
        n = jnp.concatenate([self.ns_crust, self.n_connection, self.n_Skyrme])
        cs2 = jnp.concatenate(
            [jnp.array(self.cs2_crust), cs2_connection, cs2_Skyrme]
        )

        # Compute pressure and energy from chemical potential
        log_mu = utils.cumtrapz(cs2, jnp.log(n)) + jnp.log(self.mu_lowest)
        mu = jnp.exp(log_mu)
        p = utils.cumtrapz(cs2 * mu, n) + self.ps_crust[0]
        e = mu * n - p

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)

        # Build extra dict for backward compatibility when return_extra=True
        extra = {
            "n_Skyrme_orig": self.n_Skyrme,
            "proton_fraction": proton_fraction,
        }
        if self.with_muon and e_fraction is not None:
            extra["e_fraction"] = e_fraction
            extra["muon_fraction"] = muon_fraction
        if self.calculate_durca:
            extra["durca_density"] = self.durca_density

        # Return tuple for backward compatibility when return_extra=True
        if return_extra:
            return (ns, ps, hs, es, dloge_dlogps, mu, cs2, extra)
        else:
            # Return EOSData for inference compatibility
            return EOSData(
                ns=ns,
                ps=ps,
                hs=hs,
                es=es,
                dloge_dlogps=dloge_dlogps,
                cs2=cs2,
                mu=mu,
                extra_constraints=extra,
            )

    def compute_cs2(
        self,
        n: Array,
        p: Array,
        e: Array,
        proton_fraction: Array,
        e_fraction: Array | None = None,
    ):
        r"""
        Compute speed of sound squared.

        Args:
            n: Number density [:math:`\mathrm{fm}^{-3}`]
            p: Pressure [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            e: Energy density [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`]
            proton_fraction: Proton fraction
            e_fraction: Electron fraction (if available)

        Returns:
            Array: Speed of sound squared
        """
        # Compute derivatives for cs2
        # cs2 = dp/dE = (dp/dn) / (de/dn)
        dn = n[1] - n[0]
        dp_dn = jnp.gradient(p, dn)
        de_dn = jnp.gradient(e, dn)

        cs2 = dp_dn / (de_dn + 1e-10)
        # cs2 = jnp.clip(cs2, 1e-5, 1.0)

        # Include lepton contributions to cs2 if present
        if e_fraction is not None:
            muon_fraction = jnp.maximum(1e-25, proton_fraction - e_fraction)

            # Electron contribution
            K_Fe = (3.0 * jnp.pi**2 * n * e_fraction) ** (1.0/3.0) * utils.hbarc
            C_e = utils.m_e**4 / (8.0 * jnp.pi**2) / utils.hbarc**3
            x_e = K_Fe / utils.m_e
            f_e = x_e * (1 + 2 * x_e**2) * jnp.sqrt(1 + x_e**2) - jnp.arcsinh(x_e)
            e_electron = C_e * f_e
            p_electron = -e_electron + 8.0/3.0 * C_e * x_e**3 * jnp.sqrt(1 + x_e**2)

            # Muon contribution
            K_Fmu = (3.0 * jnp.pi**2 * n * muon_fraction) ** (1.0/3.0) * utils.hbarc
            C_mu = utils.m_mu**4 / (8.0 * jnp.pi**2) / utils.hbarc**3
            x_mu = K_Fmu / utils.m_mu
            f_mu = x_mu * (1 + 2 * x_mu**2) * jnp.sqrt(1 + x_mu**2) - jnp.arcsinh(x_mu)
            e_muon = C_mu * f_mu
            p_muon = -e_muon + 8.0/3.0 * C_mu * x_mu**3 * jnp.sqrt(1 + x_mu**2)

            e_lepton = e_electron + e_muon
            p_lepton = p_electron + p_muon

            # Total cs2 with leptons
            e_total = e + e_lepton
            p_total = p + p_lepton

            dp_dn_total = jnp.gradient(p_total, n[1] - n[0])
            de_dn_total = jnp.gradient(e_total, n[1] - n[0])

            cs2 = dp_dn_total / (de_dn_total + 1e-10)
            # cs2 = jnp.clip(cs2, 1e-5, 1.0)

        return cs2

    def get_required_parameters(self) -> list[str]:
        r"""
        Return list of INM parameters required by Skyrme EOS.

        Returns:
            list[str]: INM parameter names
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
        ]


# =============================================================================
# Helper functions for CSE/peakCSE extensions
# =============================================================================


def create_skyrme_for_extension(
    nsat: Float,
    nmin_Skyrme_nsat: Float,
    nbreak: Float,
    ndat: Int,
    skyrme_kwargs: dict,
    proton_fraction_setting: str = "exact",
) -> Skyrme_EOS_model:
    r"""
    Create a Skyrme_EOS_model instance for use with CSE/peakCSE extensions.

    Args:
        nsat: Nuclear saturation density [:math:`\mathrm{fm}^{-3}`]
        nmin_Skyrme_nsat: Starting density for Skyrme region as fraction of nsat
        nbreak: Break density for transition to extension region [:math:`\mathrm{fm}^{-3}`]
        ndat: Number of density points for discretization
        skyrme_kwargs: Additional keyword arguments for Skyrme_EOS_model
        proton_fraction_setting: Either "exact" (with muons) or "approx" (without muons)

    Returns:
        Skyrme_EOS_model instance configured for the extension
    """
    # Fix: Safely copy the dictionary and pop out `proton_fraction` to avoid kwarg collisions.
    safe_kwargs = skyrme_kwargs.copy()
    safe_kwargs.pop("proton_fraction", None)

    return Skyrme_EOS_model(
        nsat=nsat,
        nmin_Skyrme_nsat=nmin_Skyrme_nsat,
        nmax_nsat=nbreak / nsat,
        ndat=ndat,
        proton_fraction=proton_fraction_setting,
        **safe_kwargs,
    )