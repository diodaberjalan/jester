"""Tests for derived Skyrme nuclear empirical parameters."""

import numpy as np
import pytest

from jesterTOV.eos.skyrme import Skyrme_EOS_model
from jesterTOV.eos.skyrme.neps import (
    SKYRME_RESULT_NEP_KEYS,
    compute_skyrme_neps,
    compute_skyrme_neps_from_force,
    compute_skyrme_neps_from_params,
    has_skyrme_parameters,
)


def _sample_params() -> dict[str, float]:
    return {
        "t2": 0.0653197680269105,
        "t4": -393.95938126131114,
        "x0": -0.5220046395768974,
        "x1": 63.97673418075088,
        "x4": -0.06647266088744885,
        "alph": 0.08514875984757903,
        "beta": 0.019629226545955426,
        "gamma": 0.2650239864740867,
        "kfsat": 1.3296560983138088,
        "av": -16.14607403242108,
        "J": 32.062082553105675,
        "meffs": 0.7021275438323112,
        "meffv": 0.9040752064991122,
        "Kinf": 267.780792702404,
        "eNMhd": 243.98249010800095,
    }


def test_compute_skyrme_neps_from_params_matches_core_inm_inputs() -> None:
    params = _sample_params()
    neps = compute_skyrme_neps_from_params(params)

    assert neps["E_sat"] == pytest.approx(params["av"], abs=1e-6)
    assert neps["P_sat"] == pytest.approx(0.0, abs=1e-4)
    assert neps["K_sat"] == pytest.approx(params["Kinf"], abs=1e-6)
    assert neps["E_sym"] == pytest.approx(params["J"], abs=1e-6)

    assert neps["esm0"] == pytest.approx(neps["E_sat"])
    assert neps["psm0"] == pytest.approx(neps["P_sat"])
    assert neps["Ksat0"] == pytest.approx(neps["K_sat"])
    assert neps["Qsat0"] == pytest.approx(neps["Q_sat"])
    assert neps["J0"] == pytest.approx(neps["E_sym"])
    assert neps["L0"] == pytest.approx(neps["L_sym"])
    assert neps["lsymt0"] == pytest.approx(neps["L_sym"])
    assert neps["ksymt20"] == pytest.approx(neps["K_sym"])
    assert neps["Ksym20"] == pytest.approx(neps["K_sym"])
    assert neps["Qsym20"] == pytest.approx(neps["Q_sym"])


def test_compute_skyrme_neps_vectorized_supports_fixed_params() -> None:
    params = _sample_params()
    posterior = {
        "t2": np.array([params["t2"], params["t2"] * 1.01]),
        "t4": np.array([params["t4"], params["t4"] * 0.99]),
        "x0": np.array([params["x0"], params["x0"]]),
        "x1": np.array([params["x1"], params["x1"]]),
        "x4": np.array([params["x4"], params["x4"]]),
        "alph": np.array([params["alph"], params["alph"]]),
        "beta": np.array([params["beta"], params["beta"]]),
        "gamma": np.array([params["gamma"], params["gamma"]]),
        "kfsat": np.array([params["kfsat"], params["kfsat"]]),
        "av": np.array([params["av"], params["av"]]),
        "J": np.array([params["J"], params["J"]]),
        "meffs": np.array([params["meffs"], params["meffs"]]),
        "meffv": np.array([params["meffv"], params["meffv"]]),
        "eNMhd": np.array([params["eNMhd"], params["eNMhd"]]),
        "log_prob": np.array([0.0, 0.0]),
    }
    fixed_params = {"Kinf": params["Kinf"]}

    assert has_skyrme_parameters(posterior, fixed_params)
    computed = compute_skyrme_neps(posterior, fixed_params)

    for key in SKYRME_RESULT_NEP_KEYS:
        assert key in computed
        assert computed[key].shape == (2,)

    np.testing.assert_allclose(computed["J0"], computed["E_sym"])
    np.testing.assert_allclose(computed["L0"], computed["L_sym"])
    np.testing.assert_allclose(computed["psm0"], computed["P_sat"])
    np.testing.assert_allclose(computed["Ksat0"], computed["K_sat"])
    np.testing.assert_allclose(computed["Qsat0"], computed["Q_sat"])
    np.testing.assert_allclose(computed["lsymt0"], computed["L_sym"])
    np.testing.assert_allclose(computed["ksymt20"], computed["K_sym"])
    np.testing.assert_allclose(computed["Ksym20"], computed["K_sym"])
    np.testing.assert_allclose(computed["Qsym20"], computed["Q_sym"])


def test_skyrme_construct_eos_extra_constraints_include_neps() -> None:
    model = Skyrme_EOS_model(ndat=8, proton_fraction="approx")
    result = model.construct_eos(_sample_params(), return_extra=True)
    extra = result[-1]

    for key in ("E_sat", "P_sat", "K_sat", "Q_sat", "E_sym", "L_sym", "K_sym", "Q_sym"):
        assert key in extra

    assert extra["esm0"] == pytest.approx(extra["E_sat"])
    assert extra["J0"] == pytest.approx(extra["E_sym"])
    assert extra["psm0"] == pytest.approx(extra["P_sat"])


def test_compute_skyrme_neps_from_force_matches_bsk24_reference_values() -> None:
    # Reference values were evaluated from Sandbox/skyrme_sample_converter/reference_codes
    # using BSk24.py + nucMatter_dio.py at n_sat = 0.1578 fm^-3.
    t = np.array([-3970.29, 395.766, 0.00010, 22648.6, -100.0, -150.0])
    x = np.array([0.894371, 0.0563535, -13896100.0, 1.05119, 2.0, -11.0])
    neps = compute_skyrme_neps_from_force(
        t,
        x,
        nsat=0.1578,
        alph=1.0 / 12.0,
        beta=0.50,
        gamma=1.0 / 12.0,
    )

    assert neps["E_sat"] == pytest.approx(-16.048278169860254, abs=1e-5)
    assert neps["K_sat"] == pytest.approx(245.418050976217, abs=1e-5)
    assert neps["E_sym"] == pytest.approx(29.99606058353131, abs=1e-5)
    assert neps["L_sym"] == pytest.approx(46.38826519902523, abs=1e-5)
    assert neps["K_sym"] == pytest.approx(-37.650431681357986, abs=1e-5)
    assert neps["psm0"] == pytest.approx(neps["P_sat"])
    assert neps["Ksat0"] == pytest.approx(neps["K_sat"])
    assert neps["Qsat0"] == pytest.approx(neps["Q_sat"])
    assert neps["lsymt0"] == pytest.approx(neps["L_sym"])
    assert neps["ksymt20"] == pytest.approx(neps["K_sym"])
    assert neps["Qsym20"] == pytest.approx(neps["Q_sym"])
