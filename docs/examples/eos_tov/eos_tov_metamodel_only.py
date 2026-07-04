#!/usr/bin/env python
# coding: utf-8

# # Constructing EOS and solving TOV equations — MetaModel only

# This example notebook shows how to construct the equation of state using the meta-model (nuclear empirical parameter) parametrisation, as well as solve the TOV equations.
# 
# Unlike the companion notebook `eos_tov_beta-test.ipynb`, this version uses only the baseline meta-model without the speed-of-sound extension (CSE). The density range is therefore limited to the meta-model region (up to `nmax_nsat`).

# In[1]:


import matplotlib.pyplot as plt

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
}
plt.rcParams.update(params)

import jax.numpy as jnp

import jesterTOV
from jesterTOV.eos.metamodel.base import MetaModel_EOS_model
from jesterTOV.tov.gr import GRTOVSolver
from jesterTOV.tov.data_classes import EOSData
import jesterTOV.utils as utils


# # Equation of state

# In[2]:


nsat = 0.16  # nuclear saturation density in fm^-3

# Define the EOS object — the baseline meta-model without CSE
# Two proton-fraction treatments are compared:
#   exact (default / None): beta-equilibrium with muons
#   approx: fast approximate proton fraction (no muons)
eos = MetaModel_EOS_model(nmax_nsat=8.0)
eos_approx = MetaModel_EOS_model(nmax_nsat=8.0, proton_fraction="approx")

# Define the nuclear empirical parameters (NEPs) — all in MeV
# NEP_dict = {
#     "E_sat": -16.0,  # saturation parameters
#     "K_sat": 200.0,
#     "Q_sat": 0.0,
#     "Z_sat": 0.0,
#     "E_sym": 32.0,  # symmetry parameters
#     "L_sym": 70.0,
#     "K_sym": -100.0,
#     "Q_sym": 0.0,
#     "Z_sym": 0.0,
# }
NEP_dict = {'E_sat': -15.451024510181709, 'K_sat': 168.98175714704948, 'Q_sat': -645.4341634743986, 'Z_sat': -4854.55334332118, 'E_sym': 33.11527652897152, 'L_sym': 147.47942171365827, 'K_sym': -207.75595490026768, 'Q_sym': 31.162227083051675, 'Z_sym': 4041.586944937484}
# Now create the EOS. The standalone metamodel uses the same legacy
# tuple return as the CSE/peakCSE examples when return_extra=True.
# We then rebuild EOSData explicitly for the TOV workflow below.
ns, ps, hs, es, dloge_dlogps, mu, cs2, extra = eos.construct_eos(NEP_dict, return_extra=True, calculate_durca=True)
ns_approx, ps_approx, hs_approx, es_approx, dloge_dlogps_approx, mu_approx, cs2_approx, extra_approx = eos_approx.construct_eos(NEP_dict, return_extra=True)

eos_data = EOSData(ns=ns, ps=ps, hs=hs, es=es, dloge_dlogps=dloge_dlogps, cs2=cs2, mu=mu, extra_constraints=extra)
eos_data_approx = EOSData(ns=ns_approx, ps=ps_approx, hs=hs_approx, es=es_approx, dloge_dlogps=dloge_dlogps_approx, cs2=cs2_approx, mu=mu_approx, extra_constraints=extra_approx)

# Extract extra info for diagnostics
n_orig = extra["n_orig"]
proton_fraction = extra["proton_fraction"]
e_fraction = extra["e_fraction"]
muon_fraction = extra["muon_fraction"]
durca_density = extra["durca_density"]

proton_fraction_approx = extra_approx["proton_fraction"]
n_orig_approx = extra_approx["n_orig"]

# Find the density where the EOS becomes acausal (cs2 >= 1 or cs2 < 0).
# This acts as the effective nbreak — the maximum density at which the EOS
# is causal and can be used for TOV integration.
cs2_full = cs2
ns_full_geom = ns  # geometric units — needs conversion to fm^-3
acausal_mask = (cs2_full >= 1.0) | (cs2_full < 0.0)
nbreak_geom = ns_full_geom[jnp.argmax(acausal_mask)]
nbreak = nbreak_geom / utils.fm_inv3_to_geometric   # now in fm^-3

print("-" * 50)
ye_durca = durca_density["ye"]
ym_durca = durca_density["ym"]
n_durca = durca_density["nb_durca"]

print("--- DUrca Threshold Results ---")
print(f"Ye            : {ye_durca:.6f}")
print(f"Ymu           : {ym_durca:.6f}")
print(f"Yp (Total)    : {ye_durca + ym_durca:.6f}")
print(f"n_b/n_sat     : {n_durca / nsat:.4f}")
print("--- Causality Limit ---")
print(f"n_break/n_sat : {nbreak / nsat:.4f}")
print("-" * 50)


# In[3]:


# Unpack the EOSData NamedTuples
ns, ps, hs, es, dloge_dlogps = (
    eos_data.ns,
    eos_data.ps,
    eos_data.hs,
    eos_data.es,
    eos_data.dloge_dlogps,
)
cs2, mu = eos_data.cs2, eos_data.mu

ns_a, ps_a, hs_a, es_a, dloge_dlogps_a = (
    eos_data_approx.ns,
    eos_data_approx.ps,
    eos_data_approx.hs,
    eos_data_approx.es,
    eos_data_approx.dloge_dlogps,
)
cs2_a, mu_a = eos_data_approx.cs2, eos_data_approx.mu

# Convert to common units for plotting
ns_plots = ns / utils.fm_inv3_to_geometric / nsat
es_plots = es / utils.MeV_fm_inv3_to_geometric
ps_plots = ps / utils.MeV_fm_inv3_to_geometric

ns_plots_approx = ns_a / utils.fm_inv3_to_geometric / nsat
es_plots_approx = es_a / utils.MeV_fm_inv3_to_geometric
ps_plots_approx = ps_a / utils.MeV_fm_inv3_to_geometric

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# p(n)
axes[0, 0].plot(ns_plots, ps_plots, label="exact")
axes[0, 0].plot(ns_plots_approx, ps_plots_approx, "--", label="approx")
axes[0, 0].set_xlabel(r"$n$ [$n_{\rm{sat}}$]")
axes[0, 0].set_ylabel(r"$p$ [MeV/fm$^3$]")
axes[0, 0].legend()

# e(n)
axes[0, 1].plot(ns_plots, es_plots)
axes[0, 1].plot(ns_plots_approx, es_plots_approx, "--")
axes[0, 1].set_xlabel(r"$n$ [$n_{\rm{sat}}$]")
axes[0, 1].set_ylabel(r"$e$ [MeV/fm$^3$]")

# cs2(n)
axes[1, 0].plot(ns_plots, cs2)
axes[1, 0].plot(ns_plots_approx, cs2_a)
axes[1, 0].axvline(0.5, color="red", alpha=0.5, label=r"Crust–core")
axes[1, 0].axvline(nbreak / nsat, color="tab:red", ls="--", lw=1.5, label=r"$n_{\rm{break}}$ (causality)")
axes[1, 0].axhline(0.0, color="gray", ls=":", lw=0.8)
axes[1, 0].axhline(1.0, color="gray", ls=":", lw=0.8)
axes[1, 0].set_xlabel(r"$n$ [$n_{\rm{sat}}$]")
axes[1, 0].set_ylabel(r"$c_s^2$")
axes[1, 0].legend()

# p(e)
axes[1, 1].plot(es_plots, ps_plots)
axes[1, 1].plot(es_plots_approx, ps_plots_approx, "--")
axes[1, 1].set_xlabel(r"$e$ [MeV/fm$^3$]")
axes[1, 1].set_ylabel(r"$p$ [MeV/fm$^3$]")
axes[1, 1].set_yscale("log")
axes[1, 1].set_xscale("log")

fig.tight_layout()
plt.show()
plt.close(fig)


# In[4]:


# Compare exact vs approximate proton fraction
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    n_orig / nsat,
    proton_fraction_approx,
    ls="--",
    lw=2,
    color="tab:blue",
    label=r"Approx $Y_p$",
)
ax.plot(
    n_orig / nsat,
    proton_fraction,
    ls="-",
    lw=2,
    color="tab:blue",
    alpha=0.6,
    label=r"Exact $Y_p$",
)
if e_fraction is not None:
    ax.plot(
        n_orig / nsat,
        e_fraction,
        ls="-",
        lw=2,
        color="tab:orange",
        label=r"Exact $Y_e$",
    )
if muon_fraction is not None:
    ax.plot(
        n_orig / nsat,
        muon_fraction,
        ls="-",
        lw=2,
        color="tab:green",
        label=r"Exact $Y_{\mu}$",
    )

ax.set_xlabel(r"$n/n_{\rm{sat}}$", fontsize=12)
ax.set_ylabel(r"Particle Fraction $Y_i$", fontsize=12)
ax.axvline(
    x=n_durca / nsat,
    color="black",
    ls="--",
    linewidth=1.0,
    alpha=0.7,
    label=r"$n_{\rm{dUrca}}$",
)
ax.scatter(n_durca / nsat, ye_durca, color="tab:orange", marker="x", label=r"$Y_{e}$ DUrca")
ax.scatter(
    n_durca / nsat, ym_durca, color="tab:green", marker="x", label=r"$Y_{\mu}$ DUrca"
)
ax.scatter(
    n_durca / nsat,
    ye_durca + ym_durca,
    color="tab:blue",
    marker="x",
    label=r"$Y_{e} + Y_{\mu}$ DUrca",
)

# DU fraction diagnostic
if e_fraction is not None and muon_fraction is not None:
    x_e = e_fraction / (e_fraction + muon_fraction)
    x_DU = 1 / (1 + (1 + jnp.cbrt(x_e)) ** 3)
    ax.plot(
        n_orig / nsat,
        x_DU,
        ls="-",
        lw=2,
        color="k",
        alpha=0.25,
        label=r"$x_{DU}$",
    )

ax.grid(True, which="major", linestyle=":", alpha=0.6)
ax.legend(frameon=True, loc="best", fontsize=10, ncol=2)
fig.tight_layout()
plt.show()
plt.close(fig)


# # Neutron stars

# In[5]:


from scipy.interpolate import interp1d
from jesterTOV.tov.data_classes import EOSData


def truncate_to_causal(eos_data_in):
    """Slice EOSData to keep only the causal region (cs2 in [0, 1])."""
    cs2_arr = eos_data_in.cs2
    acausal = (cs2_arr >= 1.0) | (cs2_arr < 0.0)
    if bool(jnp.any(acausal)):
        last = int(jnp.argmax(acausal))
    else:
        last = len(cs2_arr)
    return EOSData(
        ns=eos_data_in.ns[:last],
        ps=eos_data_in.ps[:last],
        hs=eos_data_in.hs[:last],
        es=eos_data_in.es[:last],
        dloge_dlogps=eos_data_in.dloge_dlogps[:last],
        cs2=eos_data_in.cs2[:last],
        mu=eos_data_in.mu[:last] if eos_data_in.mu is not None else None,
        extra_constraints=eos_data_in.extra_constraints,
    )


# Truncate to causal region (construct_family rejects EOS with any cs2 outside [0, 1])
eos_data_causal = truncate_to_causal(eos_data)
eos_data_approx_causal = truncate_to_causal(eos_data_approx)

# Solve TOV equations for both EOS variants
solver = GRTOVSolver()

family = solver.construct_family(eos_data_causal, ndat=200, min_nsat=1.0, tov_params={})
log10pcs, masses, radii, Lambdas = (
    family.log10pcs,
    family.masses,
    family.radii,
    family.lambdas,
)

family_approx = solver.construct_family(
    eos_data_approx_causal, ndat=200, min_nsat=1.0, tov_params={}
)
log10pcs_a, masses_a, radii_a, Lambdas_a = (
    family_approx.log10pcs,
    family_approx.masses,
    family_approx.radii,
    family_approx.lambdas,
)

# Interpolate the exact solution at the approximate mass points for comparison
f_exact = interp1d(masses, radii, kind="cubic", bounds_error=False)
f_exact_lam = interp1d(masses, Lambdas, kind="cubic", bounds_error=False)
radii_exact_interp = f_exact(masses_a)
Lambdas_exact_interp = f_exact_lam(masses_a)
radius_diff = radii_a - radii_exact_interp
Lambdas_diff = Lambdas_a - Lambdas_exact_interp

# --- M(R) and M(Lambda) ---
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].plot(radii_a, masses_a, label="approx")
axes[0].plot(radii, masses, "--", label="exact")
axes[0].set_xlabel(r"$R$ [km]")
axes[0].set_ylabel(r"$M$ [$M_\odot$]")
axes[0].legend()

axes[1].plot(masses_a, Lambdas_a, label="approx")
axes[1].plot(masses, Lambdas, "--", label="exact")
axes[1].set_xlabel(r"$M$ [$M_\odot$]")
axes[1].set_ylabel(r"$\Lambda$")
axes[1].set_yscale("log")

fig.tight_layout()
plt.show()
plt.close(fig)

# --- Central pressure vs mass ---
plt.plot(masses_a, jnp.power(10, log10pcs_a) / utils.MeV_fm_inv3_to_geometric, label="approx")
plt.plot(masses, jnp.power(10, log10pcs) / utils.MeV_fm_inv3_to_geometric, "--", label="exact")
plt.xlabel(r"$M$ [$M_\odot$]")
plt.ylabel(r"$P_c$ [MeV/fm$^3$]")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# --- Fractional differences ---
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].plot(masses_a, radius_diff / radii_exact_interp)
axes[0].set_ylabel(r"$(R-R_{\rm{exact}})/R_{\rm{exact}}$")
axes[0].set_xlabel(r"$M$ [$M_\odot$]")

axes[1].plot(masses_a, jnp.abs(Lambdas_diff / Lambdas_exact_interp))
axes[1].set_xlabel(r"$M$ [$M_\odot$]")
axes[1].set_ylabel(r"$|\Lambda-\Lambda_{\rm{exact}}|/\Lambda_{\rm{exact}}$")
axes[1].set_yscale("log")

fig.tight_layout()
plt.show()
plt.close(fig)

# --- Absolute differences ---
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].plot(masses_a, radius_diff)
axes[0].set_ylabel(r"$(R-R_{\rm{exact}})$ [km]")
axes[0].set_xlabel(r"$M$ [$M_\odot$]")

axes[1].plot(masses_a, jnp.abs(Lambdas_diff))
axes[1].set_xlabel(r"$M$ [$M_\odot$]")
axes[1].set_ylabel(r"$|\Lambda-\Lambda_{\rm{exact}}|$")
axes[1].set_yscale("log")

fig.tight_layout()
plt.show()
plt.close(fig)


# In[6]:


import numpy as np
import jax
import jax.numpy as jnp
# Assuming 'eos' is imported from your module
# import eos 
# ---------------------------------------------------------
# MONKEY PATCH: Fix Dynamic Shape Error without editing files
# ---------------------------------------------------------
from jesterTOV.eos.crust import Crust

# Define a new, JIT-safe preprocessing function
def safe_preprocess(self, n, p, e, min_density, max_density, filter_zero_pressure):
    """
    JIT-Safe Replacement: 
    Instead of slicing the array (which changes shape and crashes JIT),
    we simply return the full data. 

    The downstream logic will handle the transition density (nbreak)
    using values, not array sizes.
    """
    # We bypass the 'mask' and 'n[mask]' logic entirely.
    # We return the full arrays so the shape is always static (e.g. 500 points).
    return n, p, e

# Overwrite the method in the loaded class
print("🩹 Applying JIT-safe monkey patch to Crust._preprocess...")
Crust._preprocess = safe_preprocess
print("✅ Patch applied. You can now run JAX JIT without dynamic shape errors.")
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. Setup Parameters
# ---------------------------------------------------------
# nep_ranges = {
#     "E_sat": (-16.1, -15.9),
#     "K_sat": (150.0, 300.0),
#     "Q_sat": (-500.0, 1100.0),
#     "Z_sat": (-2500.0, 1500.0),
#     "E_sym": (28.0, 45.0),
#     "L_sym": (10.0, 200.0),
#     "K_sym": (-400.0, 200.0),
#     "Q_sym": (-1000.0, 1500.0),
#     "Z_sym": (-2000.0, 1500.0),
#     "nbreak": (0.16, 0.32),
# }
nep_ranges = {
    "E_sat": (-17.0, -14.8),
    "K_sat": (120.0, 350.0),
    "Q_sat": (-1000.0, 1600.0),
    "Z_sat": (-5000.0, 5000.0),

    "E_sym": (24.0, 50.0),
    "L_sym": (0.0, 250.0),
    "K_sym": (-600.0, 400.0),
    "Q_sym": (-1500.0, 2000.0),
    "Z_sym": (-5000.0, 5000.0),

    # if this is the density where you switch from nucleonic meta-model
    # to high-density agnostic / c_s^2 model:
    "nbreak": (0.16, 0.48),   # 1 to 3 n_sat, assuming n_sat = 0.16 fm^-3
}
param_names = list(nep_ranges.keys())
# Keep this comfortably above idx=171 while staying practical to run in-notebook.
n_tests = 1024
np.random.seed(42)

# Generate random parameters (N_tests x N_params)
n_params = len(param_names)
bounds = np.array([nep_ranges[p] for p in param_names])
random_params = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_tests, n_params))

# ---------------------------------------------------------
# 2. KEY FIX: Structure of Arrays (SoA)
# ---------------------------------------------------------
# Instead of a list of 100 dicts, we make ONE dict where 
# each key holds an array of 100 values.
# JAX vmap loves this format.
nep_batch = {
    name: jnp.array(random_params[:, i])
    for i, name in enumerate(param_names)
}

# Constants
nsat = 0.1578
ngrids = jnp.array([4.0, 5.0, 6.0, 7.0]) * nsat
cs2grids = jnp.array([0.5, 0.4, 0.3, 0.2])

# ---------------------------------------------------------
# 3. Vectorized JIT Compilation
# ---------------------------------------------------------
@jax.jit
def run_batch_eos(nep_batch, ngrids, cs2grids):
    """
    Runs construct_eos for the entire batch at once.
    """

    # Keep the same batch-driver signature as the CSE notebook,
    # but the plain metamodel does not consume ngrids/cs2grids.
    # `calculate_durca=True` is not vmap-safe for the plain metamodel,
    # so batch diagnostics skip that extra calculation here.
    def single_step(nep):
        return eos.construct_eos(
            nep,
            return_extra=True,
            calculate_durca=False,
        )

    # jax.vmap transforms the function to accept batches.
    # in_axes=(0): The first arg (nep) has a batch dimension at axis 0.
    # The other args (ngrids, cs2grids) are treated as constants for the batch.
    batch_fn = jax.vmap(single_step)

    return batch_fn(nep_batch)

# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
print(f"Running {n_tests} EOS constructions in parallel...")

# This runs ONE compiled kernel on the device
# Results are returned already stacked as (n_tests, ...)
results = run_batch_eos(nep_batch, ngrids, cs2grids)

# Unpack directly (results is a tuple of arrays)
# ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all = results

# ---------------------------------------------------------
# 5. NaN Detection (Fixed Unpacking)
# ---------------------------------------------------------

# The *extras will catch the Durca results (8th item) without crashing
ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all, *extras = results

# If you want to check the Durca output for NaNs too, you can grab it:
if extras:
    durca_all = extras[0]
    # Add it to the list for checking
    output_names = ['ns', 'ps', 'hs', 'es', 'dloge_dlogps', 'mu', 'cs2', 'durca']
    all_outputs = [ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all, durca_all]
else:
    output_names = ['ns', 'ps', 'hs', 'es', 'dloge_dlogps', 'mu', 'cs2']
    all_outputs = [ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all]

print(f"\n{'='*60}")
print(f"NaN Detection Results (Unpacked {len(all_outputs)} variables):")
print(f"{'='*60}")

total_nan_tests = 0
global_nan_mask = jnp.zeros(n_tests, dtype=bool)

for name, arr in zip(output_names, all_outputs):

    # CASE 1: The output is a Dictionary (e.g., Durca results)
    if isinstance(arr, dict):
        # We must check every array INSIDE the dictionary
        for sub_key, sub_arr in arr.items():
            full_name = f"{name}['{sub_key}']"

            # Skip non-array items inside dict if any
            if not hasattr(sub_arr, 'shape'): 
                continue

            # Check for NaNs
            if sub_arr.ndim > 1:
                row_is_nan = jnp.any(jnp.isnan(sub_arr), axis=tuple(range(1, sub_arr.ndim)))
            else:
                row_is_nan = jnp.isnan(sub_arr)

            # Update global mask
            global_nan_mask = global_nan_mask | row_is_nan

            nan_count = jnp.sum(row_is_nan)
            if nan_count > 0:
                indices = jnp.where(row_is_nan)[0]
                print(f"\nVariable '{full_name}': {nan_count} tests with NaN")
                print(f"  Indices: {indices[:10].tolist()}{'...' if len(indices)>10 else ''}")
        continue

    # CASE 2: The output is a standard JAX Array
    if not hasattr(arr, 'ndim'):
        continue # Skip unknown types

    if arr.ndim > 1:
        # Check all dimensions after the batch dim (axis 0)
        row_is_nan = jnp.any(jnp.isnan(arr), axis=tuple(range(1, arr.ndim)))
    else:
        row_is_nan = jnp.isnan(arr)

    global_nan_mask = global_nan_mask | row_is_nan

    nan_count = jnp.sum(row_is_nan)
    if nan_count > 0:
        indices = jnp.where(row_is_nan)[0]
        print(f"\nVariable '{name}': {nan_count} tests with NaN")
        print(f"  Indices: {indices[:10].tolist()}{'...' if len(indices)>10 else ''}")

# ---------------------------------------------------------
# Summary Stats
# ---------------------------------------------------------
total_fails = jnp.sum(global_nan_mask)
successful = n_tests - total_fails

print(f"\n{'='*60}")
print(f"Summary: {successful}/{n_tests} successful, {total_fails}/{n_tests} failed.")

if total_fails > 0:
    first_fail_idx = int(jnp.where(global_nan_mask)[0][0])
    print(f"First failing parameters (Index {first_fail_idx}):")
    bad_params = {k: float(v[first_fail_idx]) for k, v in nep_batch.items()}
    print(bad_params)
print(f"{'='*60}")

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 6. Visualization: "The Red Zone" Corner Plot
# ---------------------------------------------------------

# Convert JAX dictionary to standard NumPy matrix (N_samples, N_params)
# We cast to np.array to ensure matplotlib compatibility
param_matrix = np.stack([np.array(nep_batch[p]) for p in param_names], axis=1)
is_bad = np.array(global_nan_mask)
is_good = ~is_bad

print(f"Plotting {len(param_names)}x{len(param_names)} corner plot...")
print(f"🔴 Red points = NaN (Failed)")
print(f"🔵 Blue points = Success")

# Create a grid of subplots
n_dims = len(param_names)
fig, axes = plt.subplots(n_dims, n_dims, figsize=(20, 20))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for i in range(n_dims):
    for j in range(n_dims):
        ax = axes[i, j]

        # X-axis data (Column j)
        x_data = param_matrix[:, j]
        # Y-axis data (Row i)
        y_data = param_matrix[:, i]

        # --- DIAGONAL PLOTS (Histograms) ---
        if i == j:
            # Plot distribution of ALL parameters in grey
            ax.hist(x_data, bins=15, color='gray', alpha=0.3, density=True, label='All')

            # Overlay distribution of BAD parameters in Red (if any exist)
            if np.sum(is_bad) > 0:
                ax.hist(x_data[is_bad], bins=15, color='red', alpha=0.6, density=True, label='NaNs')

            ax.set_yticklabels([]) # Hide Y ticks on diagonal

            # Add legend only on the first diagonal
            if i == 0:
                ax.legend(loc='upper right', fontsize='x-small')

        # --- OFF-DIAGONAL PLOTS (Scatter) ---
        else:
            if i < j:
                # Hide upper triangle to keep it clean (standard corner plot style)
                ax.axis('off')
                continue

            # Plot GOOD points first (background)
            ax.scatter(x_data[is_good], y_data[is_good], 
                       c='dodgerblue', s=10, alpha=0.5, edgecolor='none')

            # Plot BAD points on top (foreground)
            if np.sum(is_bad) > 0:
                ax.scatter(x_data[is_bad], y_data[is_bad], 
                           c='red', s=20, alpha=0.9, marker='x')

        # --- LABELS & TICKS ---
        # Only show x-labels on the bottom row
        if i == n_dims - 1:
            ax.set_xlabel(param_names[j], fontsize=10, rotation=45)
        else:
            ax.set_xticklabels([])

        # Only show y-labels on the left column (and not on diagonal)
        if j == 0 and i != j:
            ax.set_ylabel(param_names[i], fontsize=10, rotation=45)
        else:
            ax.set_yticklabels([])

plt.suptitle(f"NaN Failure Analysis (Red = Failed)", y=0.92, fontsize=20)
plt.show()


# In[7]:


import numpy as np
import jax
import jax.numpy as jnp
# Assuming 'eos' is imported from your module
# import eos 

# =========================================================
# MONKEY PATCH: Fix Dynamic Shape Error without editing files
# =========================================================
from jesterTOV.eos.crust import Crust

# Define a new, JIT-safe preprocessing function
def safe_preprocess(self, n, p, e, min_density, max_density, filter_zero_pressure):
    """
    JIT-Safe Replacement: 
    Instead of slicing the array (which changes shape and crashes JIT),
    we simply return the full data. 

    The downstream logic will handle the transition density (nbreak)
    using values, not array sizes.
    """
    return n, p, e

# Overwrite the method in the loaded class
print("🩹 Applying JIT-safe monkey patch to Crust._preprocess...")
Crust._preprocess = safe_preprocess
print("✅ Patch applied. You can now run JAX JIT without dynamic shape errors.")

# =========================================================
# 1. Setup Parameters
# =========================================================
nep_ranges = {
    "E_sat": (-17.0, -14.8),
    "K_sat": (120.0, 350.0),
    "Q_sat": (-1000.0, 1600.0),
    "Z_sat": (-5000.0, 5000.0),

    "E_sym": (24.0, 50.0),
    "L_sym": (0.0, 250.0),
    "K_sym": (-600.0, 400.0),
    "Q_sym": (-1500.0, 2000.0),
    "Z_sym": (-5000.0, 5000.0),

    "nbreak": (0.16, 0.48),   
}
param_names = list(nep_ranges.keys())
n_tests = 1024
np.random.seed(42)

n_params = len(param_names)
bounds = np.array([nep_ranges[p] for p in param_names])
random_params = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_tests, n_params))

# =========================================================
# 2. KEY FIX: Structure of Arrays (SoA)
# =========================================================
nep_batch = {
    name: jnp.array(random_params[:, i])
    for i, name in enumerate(param_names)
}

# Constants
nsat = 0.1578
ngrids = jnp.array([4.0, 5.0, 6.0, 7.0]) * nsat
cs2grids = jnp.array([0.5, 0.4, 0.3, 0.2])

# =========================================================
# 3. Vectorized JIT Compilation
# =========================================================
@jax.jit
def run_batch_eos(nep_batch, ngrids, cs2grids):
    """
    Runs construct_eos for the entire batch at once.
    """
    def single_step(nep):
        return eos.construct_eos(
            nep,
            return_extra=True,
            calculate_durca=False,
        )

    batch_fn = jax.vmap(single_step)
    return batch_fn(nep_batch)

# =========================================================
# 4. Execution
# =========================================================
print(f"Running {n_tests} EOS constructions in parallel...")
results = run_batch_eos(nep_batch, ngrids, cs2grids)

# =========================================================
# 5. Causality Cut and Unpacking
# =========================================================
ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all, *extras = results

# Identify regions violating causality (or already containing NaNs in cs2)
invalid_cs2 = (cs2_all < 0.0) | (cs2_all > 1.0) | jnp.isnan(cs2_all)
cut_mask = jnp.cumsum(invalid_cs2, axis=1) > 0

# --- THE FIX: Physical Phase Space Mapping ---
has_invalid = jnp.any(cut_mask, axis=1)
first_invalid_idx = jnp.argmax(cut_mask, axis=1) 

# If the EOS is completely causal, set the threshold to infinity
n_cut = jnp.where(
    has_invalid, 
    ns_all[jnp.arange(n_tests), first_invalid_idx], 
    jnp.inf
)

def get_active_mask(arr_shape, base_mask, arr_ns=None, n_cut_thresholds=None):
    """Generates the correct boolean mask based on array shapes or physical density."""
    if arr_shape == base_mask.shape:
        return base_mask
    elif arr_ns is not None and n_cut_thresholds is not None:
        if arr_ns.ndim == 1:
            arr_ns = jnp.broadcast_to(arr_ns, (n_cut_thresholds.shape[0], arr_ns.shape[0]))
        return arr_ns >= jnp.expand_dims(n_cut_thresholds, axis=1)
    else:
        # Fallback if no physical reference exists
        return jnp.zeros(arr_shape[:2], dtype=bool)

def apply_cut(arr, active_mask):
    """Applies the pre-calculated active mask to fill acausal regions with NaNs."""
    if arr.ndim == 2:
        return jnp.where(active_mask, jnp.nan, arr)
    elif arr.ndim > 2:
        expanded_mask = jnp.expand_dims(active_mask, axis=tuple(range(2, arr.ndim)))
        return jnp.where(expanded_mask, jnp.nan, arr)
    return arr

# Apply the cuts to primary outputs
ns_all = apply_cut(ns_all, cut_mask)
ps_all = apply_cut(ps_all, cut_mask)
hs_all = apply_cut(hs_all, cut_mask)
es_all = apply_cut(es_all, cut_mask)
dloge_dlogps_all = apply_cut(dloge_dlogps_all, cut_mask)
mu_all = apply_cut(mu_all, cut_mask)
cs2_all = apply_cut(cs2_all, cut_mask)

# Apply density-mapped cuts to secondary arrays
if extras:
    durca_all = extras[0]

    # Locate the density array within the Durca dictionary to serve as the physical reference
    durca_ns_key = next((k for k in ['ns', 'n', 'n_b'] if k in durca_all), None)
    durca_ns_arr = durca_all[durca_ns_key] if durca_ns_key else None

    for key in list(durca_all.keys()):
        if hasattr(durca_all[key], 'shape'):
            active_mask = get_active_mask(durca_all[key].shape, cut_mask, durca_ns_arr, n_cut)
            durca_all[key] = apply_cut(durca_all[key], active_mask)

    output_names = ['ns', 'ps', 'hs', 'es', 'dloge_dlogps', 'mu', 'cs2', 'durca']
    all_outputs = [ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all, durca_all]
else:
    output_names = ['ns', 'ps', 'hs', 'es', 'dloge_dlogps', 'mu', 'cs2']
    all_outputs = [ns_all, ps_all, hs_all, es_all, dloge_dlogps_all, mu_all, cs2_all]

# =========================================================
# 6. NaN Detection (Ignoring Causal Cuts)
# =========================================================
print(f"\n{'='*60}")
print(f"NaN Detection Results (Unpacked {len(all_outputs)} variables):")
print("Note: NaNs in the acausal tail are expected and ignored.")
print(f"{'='*60}")

total_nan_tests = 0
global_nan_mask = jnp.zeros(n_tests, dtype=bool)

def check_true_nans(arr, active_mask):
    """Finds NaNs that are NOT part of the intentional causal cut."""
    if arr.ndim == 2:
        return jnp.isnan(arr) & (~active_mask)
    elif arr.ndim > 2:
        expanded_mask = jnp.expand_dims(active_mask, axis=tuple(range(2, arr.ndim)))
        return jnp.isnan(arr) & (~expanded_mask)
    return jnp.isnan(arr)

for name, arr in zip(output_names, all_outputs):
    if isinstance(arr, dict):
        sub_ns_key = next((k for k in ['ns', 'n', 'n_b'] if k in arr), None)
        sub_ns_arr = arr[sub_ns_key] if sub_ns_key else None

        for sub_key, sub_arr in arr.items():
            if not hasattr(sub_arr, 'shape'): 
                continue

            if sub_arr.ndim > 1:
                active_mask = get_active_mask(sub_arr.shape, cut_mask, sub_ns_arr, n_cut)
                true_nan_mask = check_true_nans(sub_arr, active_mask)
                row_is_nan = jnp.any(true_nan_mask, axis=tuple(range(1, sub_arr.ndim)))
            else:
                row_is_nan = jnp.isnan(sub_arr)

            global_nan_mask = global_nan_mask | row_is_nan

            nan_count = jnp.sum(row_is_nan)
            if nan_count > 0:
                indices = jnp.where(row_is_nan)[0]
                print(f"\nVariable '{name}['{sub_key}']': {nan_count} tests with true NaN")
                print(f"  Indices: {indices[:10].tolist()}{'...' if len(indices)>10 else ''}")
        continue

    if not hasattr(arr, 'ndim'):
        continue

    if arr.ndim > 1:
        # Primary outputs naturally match the cut_mask shape
        true_nan_mask = check_true_nans(arr, cut_mask)
        row_is_nan = jnp.any(true_nan_mask, axis=tuple(range(1, arr.ndim)))
    else:
        row_is_nan = jnp.isnan(arr)

    global_nan_mask = global_nan_mask | row_is_nan

    nan_count = jnp.sum(row_is_nan)
    if nan_count > 0:
        indices = jnp.where(row_is_nan)[0]
        print(f"\nVariable '{name}': {nan_count} tests with true NaN")
        print(f"  Indices: {indices[:10].tolist()}{'...' if len(indices)>10 else ''}")

# =========================================================
# Summary Stats
# =========================================================
total_fails = jnp.sum(global_nan_mask)
successful = n_tests - total_fails

print(f"\n{'='*60}")
print(f"Summary: {successful}/{n_tests} successful, {total_fails}/{n_tests} failed.")

if total_fails > 0:
    first_fail_idx = int(jnp.where(global_nan_mask)[0][0])
    print(f"First failing parameters (Index {first_fail_idx}):")
    bad_params = {k: float(v[first_fail_idx]) for k, v in nep_batch.items()}
    print(bad_params)
print(f"{'='*60}")

import matplotlib.pyplot as plt

# =========================================================
# 7. Visualization: "The Red Zone" Corner Plot
# =========================================================
param_matrix = np.stack([np.array(nep_batch[p]) for p in param_names], axis=1)
is_bad = np.array(global_nan_mask)
is_good = ~is_bad

print(f"Plotting {len(param_names)}x{len(param_names)} corner plot...")
print(f"🔴 Red points = NaN (Failed)")
print(f"🔵 Blue points = Success")

n_dims = len(param_names)
fig, axes = plt.subplots(n_dims, n_dims, figsize=(20, 20))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for i in range(n_dims):
    for j in range(n_dims):
        ax = axes[i, j]
        x_data = param_matrix[:, j]
        y_data = param_matrix[:, i]

        if i == j:
            ax.hist(x_data, bins=15, color='gray', alpha=0.3, density=True, label='All')
            if np.sum(is_bad) > 0:
                ax.hist(x_data[is_bad], bins=15, color='red', alpha=0.6, density=True, label='NaNs')
            ax.set_yticklabels([])
            if i == 0:
                ax.legend(loc='upper right', fontsize='x-small')
        else:
            if i < j:
                ax.axis('off')
                continue
            ax.scatter(x_data[is_good], y_data[is_good], c='dodgerblue', s=10, alpha=0.5, edgecolor='none')
            if np.sum(is_bad) > 0:
                ax.scatter(x_data[is_bad], y_data[is_bad], c='red', s=20, alpha=0.9, marker='x')

        if i == n_dims - 1:
            ax.set_xlabel(param_names[j], fontsize=10, rotation=45)
        else:
            ax.set_xticklabels([])

        if j == 0 and i != j:
            ax.set_ylabel(param_names[i], fontsize=10, rotation=45)
        else:
            ax.set_yticklabels([])

plt.suptitle(f"NaN Failure Analysis (Red = Failed)", y=0.92, fontsize=20)
plt.show()


# In[8]:


idx = 56
bad_params = {k: float(v[idx]) for k, v in nep_batch.items()}
print(bad_params)


# In[9]:


# 1. Base population: Filter out samples with NaNs during construction
base_indices = jnp.where(~global_nan_mask)[0]

# 2. Segregate by physics
valid_idxs = []
invalid_idxs = []

for idx in base_indices:
    # Check if the entire array satisfies stability and causality
    if np.all((cs2_all[idx] >= 0.0) & (cs2_all[idx] <= 1.0)):
        valid_idxs.append(idx)
    else:
        invalid_idxs.append(idx)

valid_idxs = np.array(valid_idxs)
invalid_idxs = np.array(invalid_idxs)

print(f"\nTotal NaN-free EOS: {len(base_indices)}")
print(f"Physically valid EOS: {len(valid_idxs)}")
print(f"Physically invalid EOS: {len(invalid_idxs)}")

# 3. Subsample to avoid memory overflow (adjust limits as needed)
n_plot_valid = min(500, len(valid_idxs))
n_plot_invalid = min(500, len(invalid_idxs))

plot_valid = np.random.choice(valid_idxs, size=n_plot_valid, replace=False) if n_plot_valid > 0 else []
plot_invalid = np.random.choice(invalid_idxs, size=n_plot_invalid, replace=False) if n_plot_invalid > 0 else []

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Pressure vs Density ---
ax_p = axes[0]

# Render invalid EOS first (Background)
for idx in plot_invalid:
    n_i = ns_all[idx]
    p_i = ps_all[idx] / utils.MeV_fm_inv3_to_geometric
    n_norm = n_i / utils.fm_inv3_to_geometric / nsat
    ax_p.plot(n_norm, p_i, color='gray', alpha=0.1, linewidth=1, zorder=1)

# Render valid EOS second (Foreground)
for idx in plot_valid:
    n_i = ns_all[idx]
    p_i = ps_all[idx] / utils.MeV_fm_inv3_to_geometric
    n_norm = n_i / utils.fm_inv3_to_geometric / nsat
    ax_p.plot(n_norm, p_i, color='tab:blue', alpha=0.2, linewidth=1.5, zorder=2)

ax_p.set_xlabel(r"$n/n_{\rm{sat}}$", fontsize=14)
ax_p.set_ylabel(r"$P$ [MeV/fm$^3$]", fontsize=14)
# ax_p.set_title(r"Pressure vs. Density (Valid & Invalid)", fontsize=14)
ax_p.grid(True, which='both', linestyle=':', alpha=0.6)
ax_p.set_yscale('log')
ax_p.set_xscale('log')
ax_p.set_ylim(1e-1, 1e3) 
ax_p.set_xlim(1e-1, 6) 
# --- Plot 2: Speed of Sound vs Density ---
ax_cs2 = axes[1]

# Render invalid EOS first (Background)
for idx in plot_invalid:
    n_i = ns_all[idx]
    cs2_i = cs2_all[idx]
    n_norm = n_i / utils.fm_inv3_to_geometric / nsat
    ax_cs2.plot(n_norm, cs2_i, color='gray', alpha=0.1, linewidth=1, zorder=1)

# Render valid EOS second (Foreground)
for idx in plot_valid:
    n_i = ns_all[idx]
    cs2_i = cs2_all[idx]
    n_norm = n_i / utils.fm_inv3_to_geometric / nsat
    ax_cs2.plot(n_norm, cs2_i, color='tab:red', alpha=0.2, linewidth=1.5, zorder=2)

ax_cs2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label=r'Causality ($c_s^2=1$)')
ax_cs2.axhline(0.0, color='black', linestyle='-', linewidth=1.0) 

ax_cs2.set_xlabel(r"$n/n_{\rm{sat}}$", fontsize=14)
ax_cs2.set_ylabel(r"$c_s^2$ [dimensionless]", fontsize=14)
# ax_cs2.set_title(r"Speed of Sound vs. Density (Valid & Invalid)", fontsize=14)
ax_cs2.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
plt.close()


# In[ ]:




