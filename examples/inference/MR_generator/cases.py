def get_case_data(case_name: str) -> str:
    cases = {
        "case1": """
E_sat           | -1.59577e+01
E_sym           | 4.02592e+01
K_sat           | 2.81422e+02
K_sym           | -2.73364e+02
L_sym           | 2.97490e+01
Q_sat           | 4.71850e+02
Q_sym           | -4.54879e+02
Z_sat           | -1.15918e+03
Z_sym           | 3.16026e+02
_random_key     | 1.36150e+09
beta_ST         | -2.00280e+00
cs2_CSE_0       | 9.64526e-01
cs2_CSE_1       | 8.10706e-01
cs2_CSE_2       | 7.43328e-01
cs2_CSE_3       | 8.65659e-01
cs2_CSE_4       | 3.07420e-02
cs2_CSE_5       | 7.21023e-01
cs2_CSE_6       | 9.65283e-01
cs2_CSE_7       | 4.96315e-01
cs2_CSE_8       | 7.14064e-01
n_CSE_0_u       | 1.38456e-01
n_CSE_1_u       | 1.77157e-01
n_CSE_2_u       | 6.00186e-01
n_CSE_3_u       | 4.93667e-01
n_CSE_4_u       | 8.50130e-01
n_CSE_5_u       | 3.96461e-01
n_CSE_6_u       | 9.58902e-01
n_CSE_7_u       | 8.73815e-01
nbreak          | 2.16532e-01
phi_c           | 9.99953e-01
phi_inf_tgt     | 9.99978e-04
""",
        "case2": """
E_sat           | -1.59577e+01
E_sym           | 4.02592e+01
K_sat           | 2.81422e+02
K_sym           | -2.73364e+02
L_sym           | 2.97490e+01
Q_sat           | 4.71850e+02
Q_sym           | -4.54879e+02
Z_sat           | -1.15918e+03
Z_sym           | 3.16026e+02
_random_key     | 1.36150e+09
beta_ST         | -4.80280e+00
cs2_CSE_0       | 9.64526e-01
cs2_CSE_1       | 8.10706e-01
cs2_CSE_2       | 7.43328e-01
cs2_CSE_3       | 8.65659e-01
cs2_CSE_4       | 3.07420e-02
cs2_CSE_5       | 7.21023e-01
cs2_CSE_6       | 9.65283e-01
cs2_CSE_7       | 4.96315e-01
cs2_CSE_8       | 7.14064e-01
n_CSE_0_u       | 1.38456e-01
n_CSE_1_u       | 1.77157e-01
n_CSE_2_u       | 6.00186e-01
n_CSE_3_u       | 4.93667e-01
n_CSE_4_u       | 8.50130e-01
n_CSE_5_u       | 3.96461e-01
n_CSE_6_u       | 9.58902e-01
n_CSE_7_u       | 8.73815e-01
nbreak          | 2.16532e-01
phi_c           | 9.99953e-01
phi_inf_tgt     | 9.99978e-04
""",
        "case3": """
E_sat           | -1.60193e+01
E_sym           | 3.94645e+01
K_sat           | 1.89552e+02
K_sym           | -2.81984e+01
L_sym           | 1.82195e+01
Q_sat           | 1.54928e+02
Q_sym           | 1.22737e+03
Z_sat           | 1.45468e+03
Z_sym           | -7.32391e+01
_random_key     | 2.07058e+09
beta_ST         | -5.77032e+00
cs2_CSE_0       | 8.39565e-01
cs2_CSE_1       | 8.15221e-01
cs2_CSE_2       | 4.12762e-01
cs2_CSE_3       | 4.93620e-01
cs2_CSE_4       | 1.62055e-01
cs2_CSE_5       | 7.72167e-01
cs2_CSE_6       | 3.76947e-01
cs2_CSE_7       | 2.76801e-01
cs2_CSE_8       | 5.74852e-01
n_CSE_0_u       | 2.73534e-01
n_CSE_1_u       | 6.74102e-01
n_CSE_2_u       | 3.12529e-01
n_CSE_3_u       | 8.44386e-01
n_CSE_4_u       | 2.96292e-01
n_CSE_5_u       | 1.50952e-01
n_CSE_6_u       | 4.84994e-01
n_CSE_7_u       | 3.37661e-01
nbreak          | 2.24007e-01
phi_c           | 9.99935e-01
phi_inf_tgt     | 9.99955e-04
""",
        "case4": """
E_sat           | -1.60667e+01
E_sym           | 3.92920e+01
K_sat           | 1.96292e+02
K_sym           | 9.83514e+01
L_sym           | 1.30970e+01
Q_sat           | 4.22457e+02
Q_sym           | -9.55569e+02
Z_sat           | -1.56207e+03
Z_sym           | 1.25714e+03
_random_key     | 3.97777e+09
beta_ST         | -5.61005e+00
cs2_CSE_0       | 5.58593e-01
cs2_CSE_1       | 3.07976e-01
cs2_CSE_2       | 7.96442e-01
cs2_CSE_3       | 5.40990e-01
cs2_CSE_4       | 3.73679e-01
cs2_CSE_5       | 4.53546e-01
cs2_CSE_6       | 4.88443e-01
cs2_CSE_7       | 9.81306e-02
cs2_CSE_8       | 1.71824e-01
n_CSE_0_u       | 5.41364e-01
n_CSE_1_u       | 4.44460e-01
n_CSE_2_u       | 7.82756e-01
n_CSE_3_u       | 4.30642e-01
n_CSE_4_u       | 6.71844e-01
n_CSE_5_u       | 5.33934e-02
n_CSE_6_u       | 4.40534e-01
n_CSE_7_u       | 3.11815e-01
nbreak          | 2.91341e-01
phi_c           | 9.99995e-01
phi_inf_tgt     | 9.99938e-04
"""
    }
    
    if case_name not in cases:
        raise ValueError(f"Unknown case_name: {case_name}. Cek lagi dong kak! (⁠✿⁠ ⁠♡⁠‿⁠♡⁠)")
    
    return cases[case_name]

def parse_input_data(input_data_str: str) -> dict:
    input_dict = {}
    for line in input_data_str.strip().split('\n'):
        if '|' in line:
            key, val = line.split('|')
            input_dict[key.strip()] = float(val.strip())
    return input_dict