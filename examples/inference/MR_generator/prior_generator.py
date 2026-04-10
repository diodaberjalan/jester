import os


def generate_prior_file(
    folder_name, sys=True, corr=True, skew=True, gravity_theory="ST"
):
    """
    Generates the prior.prior file in the specified folder.
    Dynamically includes scalar-tensor priors based on gravity_theory.
    """
    prior_lines = [
        'E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])',
        'K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])',
        'Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])',
        'Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])',
        'E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])',
        'L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])',
        'K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])',
        'Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])',
        'Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])',
        'nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])',
    ]

    if gravity_theory == "ST":
        prior_lines.extend(
            [
                'beta_ST = UniformPrior(-6, 0, parameter_names=["beta_ST"])',
                'phi_c = Fixed(1, parameter_names=["phi_c"])',
                'phi_inf_tgt = Fixed(1e-3, parameter_names=["phi_inf_tgt"])',
            ]
        )

    prior_content = "\n".join(prior_lines) + "\n"

    filepath = os.path.join(folder_name, "prior.prior")
    with open(filepath, "w") as f:
        f.write(prior_content)
