import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.stats import norm
from matplotlib.lines import Line2D
from jesterTOV import utils

# Keep your exact LaTeX and font configurations
params = {"text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"],
          "xtick.labelsize": 16,
          "ytick.labelsize": 16,
          "axes.labelsize": 16,
          "legend.fontsize": 16,
          "legend.title_fontsize": 16}
plt.rcParams.update(params)

def plot_eos(ns, ps, es, cs2, input_dict, nsat, save_path):
    ns_plots = ns / utils.fm_inv3_to_geometric / 0.16
    es_plots = es / utils.MeV_fm_inv3_to_geometric
    ps_plots = ps / utils.MeV_fm_inv3_to_geometric

    plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # p(n)
    plt.subplot(221)
    plt.plot(ns_plots, ps_plots)
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV/fm$^3$]")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(4e-1, 25)
    plt.ylim(1e-2, 10000)
    
    # e(n)
    plt.subplot(222)
    plt.plot(ns_plots, es_plots)
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$e$ [MeV/fm$^3$]")

    # cs2(n)
    plt.subplot(223)
    plt.plot(ns_plots, cs2)
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$c_s^2$")
    plt.axvline(0.5, color="red", label="Crust-core transition")
    plt.axvline(input_dict["nbreak"] / nsat, color="black", label=r"$n_{\rm{break}}$")
    plt.legend()

    # p(e)
    plt.subplot(224)
    plt.plot(es_plots, ps_plots)
    plt.xlabel(r"$e$ [MeV/fm$^3$]")
    plt.ylabel(r"$p$ [MeV/fm$^3$]")
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "eos_plot.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_mr_comparison(results, gr_radii, gr_masses, save_path):
    plt.figure(figsize=(6, 4))
    for beta_ST, data in results.items():
        plt.scatter(data["radii"], data["masses"], label=rf'$\beta$ = {beta_ST}', s=1)
    plt.plot(gr_radii, gr_masses, label='GR', color='black', linewidth=2)
    plt.ylabel(r"Mass [$M_\odot$]")
    plt.xlabel(r"$R$ [km]")
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "mr_comparison.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_splits(rad_seg1, mass_seg1, rad_seg2, mass_seg2, save_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(rad_seg1, mass_seg1, label="Segment 1")
    plt.scatter(rad_seg2, mass_seg2, label="Segment 2")
    plt.xlabel("Radii")
    plt.ylabel("Masses")
    plt.legend()
    plt.savefig(os.path.join(save_path, "split_segments.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_noise_addition(data_radii, data_masses, sample_radii, sample_masses, sample_radii_noise, sample_masses_noise, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_radii, data_masses, alpha=0.3, label="Original Data", color='gray')
    plt.scatter(sample_radii, sample_masses, label="Selected Samples", color='blue')
    plt.scatter(sample_radii_noise, sample_masses_noise, label="Samples + Noise", color='red', marker='x')
    plt.legend()
    plt.xlabel("Radii")
    plt.ylabel("Masses")
    plt.savefig(os.path.join(save_path, "noise_injection.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_mock_flows_contours(flows, samples_m, samples_r, data_radii, data_masses, case_name, n_obj, save_path):
    m_grid = jnp.linspace(jnp.min(samples_m) - 1, jnp.max(samples_m) + 1, 200)
    r_grid = jnp.linspace(jnp.min(samples_r) - 2, jnp.max(samples_r) + 2, 200)
    M, R = jnp.meshgrid(m_grid, r_grid)
    grid_points = jnp.stack([M, R], axis=-1)

    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap('viridis')
    n_flows = len(flows)

    for i, flow in enumerate(flows):
        color = cmap(i / n_flows)
        lp = flow.log_prob(grid_points)
        prob = jnp.exp(lp)
        
        prob_flat = prob.flatten()
        sorted_prob = jnp.sort(prob_flat)[::-1]
        cum_prob = jnp.cumsum(sorted_prob)
        cum_prob = cum_prob / cum_prob[-1] 
        
        idx_1sigma = jnp.searchsorted(cum_prob, 0.6827)
        idx_2sigma = jnp.searchsorted(cum_prob, 0.9545)
        
        level_1sigma = sorted_prob[idx_1sigma]
        level_2sigma = sorted_prob[idx_2sigma]
        
        plt.contour(R, M, prob, levels=[level_2sigma, level_1sigma], 
                    colors=[color], alpha=0.8, linestyles=['dashed', 'solid'])
        
        plt.scatter(flow.center[1], flow.center[0], color='red', 
                    marker='*', s=50, edgecolors='black', 
                    label='Data Centers' if i == 0 else "")
                    
    plt.scatter(data_radii, data_masses, alpha=0.3, label="Reference masses radii", color='gray')
    plt.scatter(samples_r, samples_m, label="Selected Samples", color='blue')
    plt.ylabel(r"Mass ($M_{\odot}$)")
    plt.xlabel("Radius (km)")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='solid'),
                    Line2D([0], [0], color='black', lw=1, linestyle='dashed'),
                    Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=8)]
    plt.legend(custom_lines, [r'$1\sigma$', r'$2\sigma$', 'Data Centeroids'])
    
    plot_filename = os.path.join(save_path, f"{case_name}_{n_obj}ndat_plot.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    plt.close()

def plot_final_mock_likelihood(mock_ll, m_eos, r_eos, save_path):
    m_grid = jnp.linspace(0.0, jnp.max(m_eos) + 0.5, 200)
    r_grid = jnp.linspace(jnp.min(r_eos) - 2.0, jnp.max(r_eos) + 2.0, 200)
    M, R = jnp.meshgrid(m_grid, r_grid)
    grid_points = jnp.stack([M, R], axis=-1)

    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap('viridis')

    for k in range(mock_ll.K):
        color = cmap(k / mock_ll.K)
        diff = grid_points - mock_ll.centers[k]
        diff_transformed = jnp.einsum('ij,nmj->nmi', mock_ll.inv_covs[k], diff)
        quad = jnp.sum(diff * diff_transformed, axis=-1)
        
        log_norm = -0.5 * (mock_ll.log_det_covs[k] + quad + 2 * jnp.log(2 * jnp.pi))
        skew_arg = jnp.sum(mock_ll.alpha_primes[k] * diff, axis=-1)
        log_skew = jnp.log(2.0) + norm.logcdf(skew_arg)
        
        prob = jnp.exp(log_norm + log_skew)

        prob_flat = prob.flatten()
        sorted_prob = jnp.sort(prob_flat)[::-1]
        cum_prob = jnp.cumsum(sorted_prob)
        cum_prob = cum_prob / cum_prob[-1] 

        idx_1sigma = min(jnp.searchsorted(cum_prob, 0.6827), len(sorted_prob)-1)
        idx_2sigma = min(jnp.searchsorted(cum_prob, 0.9545), len(sorted_prob)-1)

        level_1sigma = sorted_prob[idx_1sigma]
        level_2sigma = sorted_prob[idx_2sigma]

        plt.contour(R, M, prob, levels=[float(level_2sigma), float(level_1sigma)], 
                    colors=[color], alpha=0.8, linestyles=['dashed', 'solid'])
        
        plt.scatter(mock_ll.centers[k, 1], mock_ll.centers[k, 0], color='red', 
                    marker='*', s=50, edgecolors='black', label='Data Centroids' if k == 0 else "")

    plt.plot(r_eos, m_eos, color='black', linewidth=2.5, zorder=10)
    plt.ylabel(r"Mass ($M_{\odot}$)")
    plt.xlabel("Radius (km)")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    custom_lines = [Line2D([0], [0], color='black', lw=1, linestyle='solid'),
                    Line2D([0], [0], color='black', lw=1, linestyle='dashed'),
                    Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=8),
                    Line2D([0], [0], color='black', lw=2.5, linestyle='solid')]
    plt.legend(custom_lines, [r'$1\sigma$', r'$2\sigma$', 'Data Centroids', 'Reference EOS'])
    plt.title("Mock MR Likelihood Contours with Reference EOS")
    
    plt.savefig(os.path.join(save_path, "final_likelihood_contours.pdf"), format='pdf', bbox_inches='tight')
    plt.close()