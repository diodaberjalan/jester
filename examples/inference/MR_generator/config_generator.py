import os

def generate_yaml_config(folder_path: str, csv_filename: str, gravity_theory: str = "ST"):
    
    tov_type = "scalar_tensor" if gravity_theory == "ST" else "gr"
    tidal_str = "\n  calculate_tidal: false" if gravity_theory == "ST" else ""
    
    config_content = f"""seed: 44
dry_run: false
validate_only: false
eos:
  type: metamodel_cse
  ndat_metamodel: 100
  nmax_nsat: 25.0
  nb_CSE: 8
  crust_name: DH
tov:
  min_nsat_TOV: 0.75
  ndat_TOV: 100
  nb_masses: 100
  type: {tov_type}{tidal_str}
prior:
  specification_file: prior.prior
likelihoods:
- type: constraints_eos
  enabled: true
- type: "mock_mr"
  enabled: true
  csv_file: "{csv_filename}"
  penalty_value: -10000000000
  N_masses_evaluation: 200
sampler:
  type: smc-rw
  n_particles: 1000
  n_mcmc_steps: 10
  target_ess: 0.9
  random_walk_sigma: 0.1
  n_eos_samples: 1000
  output_dir: ./outdir/
postprocessing:
  enabled: true
  make_cornerplot: false
  make_massradius: true
  make_pressuredensity: true
"""
    config_path = os.path.join(folder_path, "config.yml")
    with open(config_path, "w") as f:
        f.write(config_content)