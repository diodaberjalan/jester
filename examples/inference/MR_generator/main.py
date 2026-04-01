import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jax import random, config

config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jesterTOV import utils
from jesterTOV.eos.metamodel import MetaModel_with_CSE_EOS_model
from jesterTOV.tov.gr import GRTOVSolver
from jesterTOV.tov.scalar_tensor import ScalarTensorTOVSolver
from jesterTOV.tov.data_classes import EOSData

from cases import get_case_data, parse_input_data
from submit_generator import generate_submit_script
from config_generator import generate_yaml_config
from prior_generator import generate_prior_file
from likelihood import SkewedCorrelatedFlow, MockMRLikelihood
import plotting

def main():
    cases_to_run = ["case1", "case2", "case3", "case4"]
    N_obj_list = [5, 10, 15, 20, 25]

    # *** EXPERIMENT TOGGLES ***
    run_multi_cases = True

    if run_multi_cases:
        experiment_modes = [
            {"sys": False, "corr": False, "skew": False},  # 1. Gaussian
            # {"sys": False, "corr": True,  "skew": True},   # 2. No sys
            # {"sys": True,  "corr": False, "skew": False},  # 3. Gaussian with systematic
            # {"sys": True,  "corr": True,  "skew": True}    # 4. All on
        ]
    else:
        # Default single run config
        experiment_modes = [
            {"sys": True, "corr": True, "skew": True} 
        ]

    for mode in experiment_modes:
        sys = mode["sys"]
        corr = mode["corr"]
        skew = mode["skew"]

        toggle_str = ""
        if sys: toggle_str += "_sys"
        if corr: toggle_str += "_corr"
        if skew: toggle_str += "_skew"
        if not toggle_str: toggle_str = "_gaussian"

        print(f"\n========================================")
        print(f"Starting execution for mode: {toggle_str}")
        print(f"========================================")

        nsat = 0.16
        nmax_nsat = 25.0
        nb_CSE = 8
        phi_c = 1.0
        phi_inf_tgt = 1e-3
        r_max = 50 

        gr_solver = GRTOVSolver()
        st_solver = ScalarTensorTOVSolver()

        # Dictionary to store log-likelihoods for the final plot
        ll_results = {case: [] for case in cases_to_run}

        for case_name in cases_to_run:
            print(f"\nProcessing {case_name}...")
            input_data_str = get_case_data(case_name)
            input_dict = parse_input_data(input_data_str)
            
            beta_ST = input_dict.pop("beta_ST")
            input_dict["n_CSE_8_u"] = nmax_nsat

            eos = MetaModel_with_CSE_EOS_model(
                nmax_nsat=nmax_nsat, nb_CSE=nb_CSE, nmin_MM_nsat=0.75, ndat_metamodel=80, ndat_CSE=70
            )
            eos_output = eos.construct_eos(input_dict)

            eos_data = EOSData(
                ns=eos_output.ns, ps=eos_output.ps, hs=eos_output.hs, 
                es=eos_output.es, dloge_dlogps=eos_output.dloge_dlogps, cs2=eos_output.cs2,
            )

            tov_params = {"beta_ST": beta_ST, "phi_c": phi_c, "phi_inf_tgt": phi_inf_tgt}
            
            # Calculate ST
            family = st_solver.construct_family(eos_data, ndat=200, min_nsat=0.1, tov_params=tov_params)
            radii_km = family.radii
            masses_solar = family.masses
            mask = (radii_km < r_max) & (radii_km > 5) & (masses_solar > 1.2) & jnp.isfinite(family.extra["lambda_S"])
            
            data = {
                "masses": masses_solar[mask],
                "radii": radii_km[mask]
            }
            results = {beta_ST: data}

            # Calculate GR for reference plotting
            gr_family = gr_solver.construct_family(eos_data, ndat=300, min_nsat=0.5, tov_params={})
            gr_mask = (gr_family.radii < r_max) & (gr_family.radii > 5) & (gr_family.masses > 0.5)
            gr_masses = gr_family.masses[gr_mask]
            gr_radii = gr_family.radii[gr_mask]
            
            # *** PRE-GENERATE MAX DATASET TO ENSURE INCLUSIVE SUBSETS ***
            N_max = max(N_obj_list)
            key = jax.random.PRNGKey(43)
            split_idx = int(utils.get_MR_split_index(data["radii"], data["masses"]))
            
            rad_seg1, rad_seg2 = data["radii"][:split_idx], data["radii"][split_idx:]
            mass_seg1, mass_seg2 = data["masses"][:split_idx], data["masses"][split_idx:]
            
            key, subkey_u, subkey_coin = jax.random.split(key, 3)
            
            if len(mass_seg2) == 0:
                m1_min, m1_max = mass_seg1[0], mass_seg1[-1]
                key, subkey_u = jax.random.split(key, 2)
                sample_masses_max = jax.random.uniform(subkey_u, shape=(N_max,), minval=m1_min, maxval=m1_max)
                sample_radii_max = jnp.interp(sample_masses_max, mass_seg1, rad_seg1)
            else:
                m1_min, m1_max = mass_seg1[0], mass_seg1[-1]
                m2_min, m2_max = mass_seg2[0], mass_seg2[-1]
                key, subkey_u, subkey_coin, subkey_force = jax.random.split(key, 4)
                
                cond = m1_min <= m2_min
                min_a = jnp.where(cond, m1_min, m2_min)
                max_a = jnp.where(cond, m1_max, m2_max)
                min_b = jnp.where(cond, m2_min, m1_min)
                max_b = jnp.where(cond, m2_max, m1_max)
                
                is_gap = min_b > max_a
                width_a = max_a - min_a
                width_b = max_b - min_b
                
                total_domain_width = jnp.where(is_gap, width_a + width_b, jnp.maximum(max_a, max_b) - min_a)
                u = jax.random.uniform(subkey_u, shape=(N_max,), minval=0.0, maxval=total_domain_width)
                
                mass_if_gap = jnp.where(u <= width_a, min_a + u, min_b + (u - width_a))
                mass_if_overlap = min_a + u
                sample_masses_max = jnp.where(is_gap, mass_if_gap, mass_if_overlap)
                
                r1 = jnp.interp(sample_masses_max, mass_seg1, rad_seg1)
                
                # === REVERTED TO ORIGINAL ===
                # Reason: User confirmed mass_seg2 is NOT decreasing. jnp.interp will function correctly.
                r2 = jnp.interp(sample_masses_max, mass_seg2, rad_seg2)
                # ============================
                
                valid1 = (sample_masses_max >= m1_min) & (sample_masses_max <= m1_max)
                valid2 = (sample_masses_max >= m2_min) & (sample_masses_max <= m2_max)
                
                coin_flip = jax.random.bernoulli(subkey_coin, p=0.5, shape=(N_max,))
                use_seg1 = jnp.where(valid1 & valid2, coin_flip, valid1)
                sample_radii_max = jnp.where(use_seg1, r1, r2)
                
                missed_seg2 = jnp.all(use_seg1)
                forced_m2 = jax.random.uniform(subkey_force, shape=(), minval=m2_min, maxval=m2_max)
                forced_r2 = jnp.interp(forced_m2, mass_seg2, rad_seg2)
                
                # === FIX: INCLUSIVE FORCED INDEX ===
                # Reason: Forcing the point at N_max - 1 (index 24) meant subsets N=5, 10, 15, 20 completely 
                # excluded it. Setting it to index 0 ensures all subsets from N=5 upwards contain the forced point.
                # OLD CODE:
                # overwrite_mask = (jnp.arange(N_max) == (N_max - 1)) & missed_seg2
                # NEW CODE:
                overwrite_mask = (jnp.arange(N_max) == 0) & missed_seg2
                # ===================================
                
                sample_masses_max = jnp.where(overwrite_mask, forced_m2, sample_masses_max)
                sample_radii_max = jnp.where(overwrite_mask, forced_r2, sample_radii_max)

            # --- SYSTEMATIC SHIFT GENERATION ---
            key_r, key_m = random.split(key, 2)
            std_m_max = jax.random.uniform(key_m, shape=sample_masses_max.shape, minval=0.02, maxval=0.05)
            std_r_max = jax.random.uniform(key_r, shape=sample_radii_max.shape, minval=0.02, maxval=0.05)
            
            delta_r_max = std_r_max * sample_radii_max
            delta_m_max = std_m_max * sample_masses_max
            
            if not sys:
                noise_r_max = jnp.zeros_like(sample_radii_max)
                noise_m_max = jnp.zeros_like(sample_masses_max)
            else:
                noise_r_max = random.normal(key_r, shape=sample_radii_max.shape) * delta_r_max
                noise_m_max = random.normal(key_m, shape=sample_masses_max.shape) * delta_m_max

            sample_radii_noise_max = sample_radii_max + noise_r_max
            sample_masses_noise_max = sample_masses_max + noise_m_max
            
            # === FIX: PRE-GENERATE PDF ERRORS (STATISTICAL NOISE) ===
            # Reason: We must generate the PDF error percentages OUTSIDE the N_obj loop. 
            # If generated inside, the random numbers change per subset, violating the requirement 
            # that "it has to be consistent for each 5 ndat in 10 ndat".
            # NEW CODE:
            key, key_pdf_m, key_pdf_r, key_corr, key_skew = random.split(key, 5)
            
            pdf_frac_m_max = jax.random.uniform(key_pdf_m, shape=(N_max,), minval=0.02, maxval=0.05)
            pdf_frac_r_max = jax.random.uniform(key_pdf_r, shape=(N_max,), minval=0.02, maxval=0.05)
            
            if corr:
                corr_vals_max = jax.random.uniform(key_corr, shape=(N_max,), minval=0.0, maxval=1.0)
            else:
                corr_vals_max = jnp.zeros(N_max)
                
            if skew:
                skew_vals_max = jax.random.uniform(key_skew, shape=(N_max, 2), minval=-3.0, maxval=3.0)
            else:
                skew_vals_max = jnp.zeros((N_max, 2))
            # ========================================================

            for N_obj in N_obj_list:
                print(f"  Generating setup for N_obj = {N_obj}")

                folder_name = f"output_{case_name}_N{N_obj}{toggle_str}"
                os.makedirs(folder_name, exist_ok=True)
                
                # *** INJECT FILES INTO FOLDER ***
                generate_submit_script(folder_name)
                generate_prior_file(folder_name, sys, corr, skew)
                
                # *** SLICE PRE-GENERATED DATA ***
                sample_masses = sample_masses_max[:N_obj]
                sample_radii = sample_radii_max[:N_obj]
                sample_masses_noise = sample_masses_noise_max[:N_obj]
                sample_radii_noise = sample_radii_noise_max[:N_obj]
                
                # === FIX: SLICE PRE-GENERATED PDF PROPERTIES ===
                # Reason: Ensure the exact same fractional errors, skews, and correlations 
                # are used for the first N_obj items across all subset iterations.
                # NEW CODE:
                pdf_frac_m = pdf_frac_m_max[:N_obj]
                pdf_frac_r = pdf_frac_r_max[:N_obj]
                corr_vals = corr_vals_max[:N_obj]
                skew_vals = skew_vals_max[:N_obj]
                # ===============================================
                
                # *** EXECUTE CASE-LEVEL PLOTS ***
                plotting.plot_eos(eos_output.ns, eos_output.ps, eos_output.es, eos_output.cs2, input_dict, nsat, folder_name)
                plotting.plot_mr_comparison(results, gr_radii, gr_masses, folder_name)

                dat_filename = os.path.join(folder_name, "reference_mr.dat")
                np.savetxt(dat_filename, np.column_stack((data["masses"], data["radii"])), header="Mass Radius")

                # *** EXECUTE SPLIT PLOT ***
                plotting.plot_splits(rad_seg1, mass_seg1, rad_seg2, mass_seg2, folder_name)

                # *** EXECUTE NOISE PLOT ***
                plotting.plot_noise_addition(data["radii"], data["masses"], sample_radii, sample_masses, sample_radii_noise, sample_masses_noise, folder_name)

                # === FIX: REMOVE NP.RANDOM.SEED(42) FROM INNER LOOP ===
                # Reason: It resets the global numpy state every subset pass, which caused 
                # the previous inconsistencies when using np.random.uniform inside the loop.
                # OLD CODE:
                # np.random.seed(42)
                # ======================================================
                
                table_data = []
                flows = []
                
                for i in range(N_obj):
                    m = float(sample_masses_noise[i])
                    r = float(sample_radii_noise[i])
                    
                    # === FIX: APPLY SLICED PDF PROPERTIES ===
                    # Reason: By referencing the pre-generated JAX arrays, point [i] is statistically 
                    # identical whether it is evaluated in the N=5 subset or the N=25 subset.
                    # OLD CODE:
                    # std_m_val = np.random.uniform(0.02, 0.05) * m 
                    # std_r_val = np.random.uniform(0.02, 0.05) * r
                    # if not corr: corr_val = 0.0
                    # else: corr_val = np.random.uniform(0, 1) 
                    # NEW CODE:
                    std_m_val = float(pdf_frac_m[i]) * m
                    std_r_val = float(pdf_frac_r[i]) * r
                    corr_val = float(corr_vals[i])
                    # ========================================
                    
                    cov_val = corr_val * std_m_val * std_r_val
                    cov_m = jnp.array([[std_m_val**2, cov_val], [cov_val, std_r_val**2]])
                    
                    # === FIX: APPLY SLICED SKEW PROPERTIES ===
                    # OLD CODE:
                    # if not skew: skew_v = jnp.array([0.0, 0.0])
                    # else: skew_v = jnp.array([np.random.uniform(-3.0, 3.0), np.random.uniform(-3.0, 3.0)])
                    # NEW CODE:
                    skew_v = jnp.array([skew_vals[i, 0], skew_vals[i, 1]])
                    # =========================================
                    
                    flows.append(SkewedCorrelatedFlow(m, r, cov_m, skew_v))
                    
                    table_data.append({
                        "Sample": i + 1,
                        "Mass_Center": f"{sample_masses[i]:.3f}",
                        "Radius_Center": f"{sample_radii[i]:.3f}",
                        "Mass_Center_Noise": f"{m:.3f}",
                        "Radius_Center_Noise": f"{r:.3f}",
                        "Std_Mass": f"{std_m_val:.3f}",
                        "Std_Radius": f"{std_r_val:.3f}",
                        "Covariance": f"{cov_val:.3f}",
                        "Skew_Mass": f"{skew_v[0]:.3f}",
                        "Skew_Radius": f"{skew_v[1]:.3f}"
                    })

                df_params = pd.DataFrame(table_data)
                csv_filename = f"{case_name}_{N_obj}ndat{toggle_str}.csv"
                csv_path = os.path.join(folder_name, csv_filename)
                df_params.to_csv(csv_path, index=False)
                
                generate_yaml_config(folder_name, csv_filename)

                # *** EXECUTE CONTOUR PLOT ***
                plotting.plot_mock_flows_contours(flows, sample_masses_noise, sample_radii_noise, data["radii"], data["masses"], case_name, N_obj, folder_name)

                # *** EVALUATE AND PLOT FINAL LIKELIHOOD ***
                mock_likelihood = MockMRLikelihood(csv_path, N_masses_evaluation=300)
                m_eos = jnp.array(data["masses"])
                r_eos = jnp.array(data["radii"])
                params_eval = {"masses_EOS": m_eos, "radii_EOS": r_eos}
                ll_val = mock_likelihood.evaluate(params_eval)
                print(f"    -> Total Normalized Log-Likelihood: {ll_val:.4f}")
                
                # Store the log-likelihood for the final aggregation plot
                ll_results[case_name].append(ll_val)
                
                plotting.plot_final_mock_likelihood(mock_likelihood, m_eos, r_eos, folder_name)

                print(f"    -> Saved all plots and configs successfully to {folder_name}/ (⁠✿⁠^⁠‿⁠^⁠)")

        # *** EXECUTE GLOBAL LIKELIHOOD VS N_OBJ PLOT ***
        print(f"\nGenerating final Log-Likelihood vs N_obj plot for {toggle_str}...")
        plt.figure(figsize=(10, 6))
        
        for case_name, ll_vals in ll_results.items():
            plt.plot(N_obj_list, ll_vals, marker='o', linestyle='-', linewidth=2, markersize=8, label=case_name)
            
        plt.title(f"Log-Likelihood Across Subset Sizes ({toggle_str.strip('_')})")
        plt.xlabel("N_obj")
        plt.ylabel("Normalized Log-Likelihood")
        plt.xticks(N_obj_list)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(title="MR Cases")
        
        final_plot_filename = f"likelihood_vs_N_obj{toggle_str}.png"
        plt.tight_layout()
        plt.savefig(final_plot_filename, dpi=96)
        plt.close()
        print(f"Saved {final_plot_filename} to current working directory. (⁠✿⁠^⁠‿⁠^⁠)")

if __name__ == "__main__":
    main()