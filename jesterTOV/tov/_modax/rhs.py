"""Existing Jester ST equations translated to scalar arithmetic for CUDA."""

import math

from numba_cuda_mlir import cuda  # type: ignore[reportMissingImports]


@cuda.jit(device=True)
def thermodynamics(h, params):
    n = (params.shape[0] - 4) // 4
    x = math.log(max(h, math.exp(params[4])))
    lo = 0
    hi = n - 1
    if x >= params[4 + hi]:
        lo = hi
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x < params[4 + mid]:
                hi = mid
            else:
                lo = mid
    if lo == n - 1:
        pressure = math.exp(params[4 + n + lo])
        energy = math.exp(params[4 + 2 * n + lo])
        derivative = params[4 + 3 * n + lo]
    else:
        theta = (x - params[4 + lo]) / (params[4 + lo + 1] - params[4 + lo])
        pressure = math.exp(
            params[4 + n + lo] + theta * (params[4 + n + lo + 1] - params[4 + n + lo])
        )
        energy = math.exp(
            params[4 + 2 * n + lo]
            + theta * (params[4 + 2 * n + lo + 1] - params[4 + 2 * n + lo])
        )
        linear = (max(h, math.exp(params[4])) - math.exp(params[4 + lo])) / (
            math.exp(params[4 + lo + 1]) - math.exp(params[4 + lo])
        )
        derivative = params[4 + 3 * n + lo] + linear * (
            params[4 + 3 * n + lo + 1] - params[4 + 3 * n + lo]
        )
    return pressure, energy, energy / pressure * derivative


def background_rhs(y, u, params):
    beta_ST = params[0]
    h = params[1] * (1 - u)
    p, e, dedp = thermodynamics(h, params)
    r = y[0]
    m = y[1]
    nu = y[2]
    psi = y[3]
    phi = y[4]
    A_phi = math.exp(0.5 * beta_ST * phi**2)
    alpha_phi = beta_ST * phi
    EPS = 1e-25
    dpdr = -(e + p) * (
        (m + 4.0 * math.pi * A_phi**4 * r**3 * p) / (r * (r - 2.0 * m + EPS))
        + 0.5 * r * psi**2
        + alpha_phi * psi
    )
    safe_dpdr = math.copysign(EPS, dpdr) if abs(dpdr) < EPS else dpdr
    drdh = (e + p) / safe_dpdr
    dmdh = (
        4.0 * math.pi * A_phi**4 * r**2 * e + 0.5 * r * (r - 2.0 * m) * psi**2
    ) * drdh
    dnudh = (
        2 * (m + 4.0 * math.pi * A_phi**4 * r**3 * p) / (r * (r - 2.0 * m + EPS))
        + r * psi**2
    ) * drdh
    dpsidh = (
        4.0
        * math.pi
        * A_phi**4
        * r
        / (r - 2.0 * m + EPS)
        * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        - 2.0 * (r - m) / (r * (r - 2.0 * m + EPS)) * psi
    ) * drdh
    dphidh = psi * drdh
    return (
        -params[1] * drdh,
        -params[1] * dmdh,
        -params[1] * dnudh,
        -params[1] * dpsidh,
        -params[1] * dphidh,
    )


def tidal_rhs(y, u, params):
    beta_ST = params[0]
    h = params[1] * (1 - u)
    p, e, dedp = thermodynamics(h, params)
    r = y[0]
    m = y[1]
    nu = y[2]
    psi = y[3]
    phi = y[4]
    H0 = y[5]
    H0_prime = y[6]
    delta_phi = y[7]
    delta_phi_prime = y[8]
    EPS = 1e-25
    A_phi = math.exp(0.5 * beta_ST * phi**2)
    alpha_phi = beta_ST * phi
    A_phi4 = A_phi**4
    four_pi_Aphi4 = 4.0 * math.pi * A_phi4
    r2 = r * r
    r3 = r2 * r
    denom_non_tidal = r - 2.0 * m + EPS
    dpdr = -(e + p) * (
        (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal)
        + 0.5 * r * psi**2
        + alpha_phi * psi
    )
    safe_dpdr = math.copysign(EPS, dpdr) if abs(dpdr) < EPS else dpdr
    drdh = (e + p) / safe_dpdr
    dmdh = (four_pi_Aphi4 * r2 * e + 0.5 * r * (r - 2.0 * m) * psi**2) * drdh
    dnudh = (
        2 * (m + four_pi_Aphi4 * r3 * p) / (r * denom_non_tidal) + r * psi**2
    ) * drdh
    dpsidh = (
        four_pi_Aphi4
        * r
        * (alpha_phi * (e - 3.0 * p) + r * (e - p) * psi)
        / denom_non_tidal
        - 2.0 * (r - m) / (r * denom_non_tidal) * psi
    ) * drdh
    dphidh = psi * drdh
    denom_pert = r - 2.0 * m + EPS
    F1 = (4.0 * math.pi * r**3 * A_phi4 * (p - e) + 2.0 * (r - m)) / (r * denom_pert)
    F0_num = (
        4.0 * math.pi * r**3 * p * A_phi4 * (r * (dedp + 9.0) - 2.0 * m * (dedp + 13.0))
        + 4.0 * math.pi * r**3 * e * A_phi4 * (dedp + 5.0) * (r - 2.0 * m)
        - 4.0 * r**2 * (r - 2.0 * m) * psi**2 * (4.0 * math.pi * r**3 * p * A_phi4 + m)
        - 64.0 * math.pi**2 * r**6 * p**2 * A_phi4**2
        - 6.0 * r * (r - 2.0 * m)
        - r**4 * (r - 2.0 * m) ** 2 * psi**4
        - 4.0 * m**2
    )
    F0 = F0_num / (r**2 * (r - 2.0 * m) ** 2)
    Fs_num = (
        4.0
        * r**2
        * (
            2.0
            * math.pi
            * A_phi4
            * (-alpha_phi * ((dedp - 9.0) * p + (dedp - 1.0) * e) + 4.0 * r * p * psi)
            + (r - 2.0 * m) * psi**3
        )
        + 8.0 * m * psi
    )
    Fs = Fs_num / (r * (r - 2.0 * m))
    G1 = F1
    G0 = (
        4.0
        * math.pi
        * r
        * A_phi4
        / (r - 2.0 * m)
        * (
            alpha_phi**2 * ((dedp + 9.0) * p + (dedp - 7.0) * e)
            + (e - 3.0 * p) * -beta_ST
        )
        - 6.0 / (r * (r - 2.0 * m))
        - 4.0 * psi**2
    )
    Gs = Fs / 4.0
    dH0dh = H0_prime * drdh
    dH0_primedh = (-F1 * H0_prime - F0 * H0 + Fs * delta_phi) * drdh
    ddelta_phidh = delta_phi_prime * drdh
    ddelta_phi_primedh = (-G1 * delta_phi_prime - G0 * delta_phi + Gs * H0) * drdh
    return (
        -params[1] * drdh,
        -params[1] * dmdh,
        -params[1] * dnudh,
        -params[1] * dpsidh,
        -params[1] * dphidh,
        -params[1] * dH0dh,
        -params[1] * dH0_primedh,
        -params[1] * ddelta_phidh,
        -params[1] * ddelta_phi_primedh,
    )
