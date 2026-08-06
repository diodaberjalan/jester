"""Tests for derived Skyrme nuclear empirical parameters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.eos.skyrme import Skyrme_EOS_model
from jesterTOV.eos.skyrme.neps import (
    SKYRME_RESULT_NEP_KEYS,
    _energy_per_particle,
    _nsat_from_kfsat,
    _solve_skyrme_system,
    compute_skyrme_neps,
    compute_skyrme_neps_from_force,
    compute_skyrme_neps_from_params,
    has_skyrme_parameters,
    normalize_skyrme_saturation_parameter,
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
    assert np.isfinite(neps["Z_sat"])
    assert np.isfinite(neps["Z_sym"])

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


def test_nsat_input_matches_the_legacy_kfsat_parameterization() -> None:
    legacy = _sample_params()
    nsat = float(_nsat_from_kfsat(jnp.asarray(legacy["kfsat"])))
    from_nsat = {key: value for key, value in legacy.items() if key != "kfsat"}
    from_nsat["nsat"] = nsat

    normalized = normalize_skyrme_saturation_parameter(from_nsat)
    assert "nsat" not in normalized
    assert normalized["kfsat"] == pytest.approx(legacy["kfsat"])

    legacy_neps = compute_skyrme_neps_from_params(legacy)
    nsat_neps = compute_skyrme_neps_from_params(from_nsat)
    for name, value in legacy_neps.items():
        assert nsat_neps[name] == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize("saturation_keys", [(), ("nsat", "kfsat")])
def test_skyrme_requires_exactly_one_saturation_parameter(
    saturation_keys: tuple[str, ...],
) -> None:
    params = _sample_params()
    params.pop("kfsat")
    nsat = float(_nsat_from_kfsat(jnp.asarray(_sample_params()["kfsat"])))
    for name in saturation_keys:
        params[name] = nsat if name == "nsat" else _sample_params()["kfsat"]

    with pytest.raises(ValueError, match="exactly one of 'nsat' or 'kfsat'"):
        compute_skyrme_neps_from_params(params)


def test_compute_skyrme_neps_from_params_is_jittable() -> None:
    """Derived NEPs must remain arrays when called by a traced EOS transform."""
    params = {key: jnp.asarray(value) for key, value in _sample_params().items()}

    neps = jax.jit(compute_skyrme_neps_from_params)(params)

    for value in neps.values():
        assert jnp.isfinite(value)

    params_nsat = _sample_params()
    params_nsat["nsat"] = float(
        _nsat_from_kfsat(jnp.asarray(params_nsat.pop("kfsat")))
    )
    nsat_neps = jax.jit(compute_skyrme_neps_from_params)(
        {key: jnp.asarray(value) for key, value in params_nsat.items()}
    )
    for value in nsat_neps.values():
        assert jnp.isfinite(value)


def test_skyrme_neps_are_the_defined_x_derivatives_at_saturation() -> None:
    """Check every reported NEP against direct autodiff of its definition."""
    params = {key: jnp.asarray(value) for key, value in _sample_params().items()}
    t, x_force = _solve_skyrme_system(params)
    nsat = _nsat_from_kfsat(params["kfsat"])

    def e_sat(xvar: jax.Array) -> jax.Array:
        density = nsat * (1.0 + 3.0 * xvar)
        return _energy_per_particle(
            density, 0.0, t, x_force, params["alph"], params["beta"], params["gamma"]
        )

    def e_sym(xvar: jax.Array) -> jax.Array:
        density = nsat * (1.0 + 3.0 * xvar)

        def energy(delta: jax.Array) -> jax.Array:
            return _energy_per_particle(
                density,
                delta,
                t,
                x_force,
                params["alph"],
                params["beta"],
                params["gamma"],
            )

        return 0.5 * jax.grad(jax.grad(energy))(0.0)

    def derivative(function, order: int) -> jax.Array:
        for _ in range(order):
            function = jax.jacfwd(function)
        return function(jnp.asarray(0.0))

    neps = compute_skyrme_neps_from_params(params)
    expected = {
        "E_sat": derivative(e_sat, 0),
        "P_sat": derivative(e_sat, 1),
        "K_sat": derivative(e_sat, 2),
        "Q_sat": derivative(e_sat, 3),
        "Z_sat": derivative(e_sat, 4),
        "E_sym": derivative(e_sym, 0),
        "L_sym": derivative(e_sym, 1),
        "K_sym": derivative(e_sym, 2),
        "Q_sym": derivative(e_sym, 3),
        "Z_sym": derivative(e_sym, 4),
    }
    for name, value in expected.items():
        assert neps[name] == pytest.approx(value, abs=1e-5)

    # The first isoscalar x derivative is zero exactly when the physical SNM
    # pressure n^2 de_sat/dn vanishes at the same density.
    pressure_snm = nsat**2 * jax.grad(
        lambda density: _energy_per_particle(
            density, 0.0, t, x_force, params["alph"], params["beta"], params["gamma"]
        )
    )(nsat)
    assert pressure_snm == pytest.approx(0.0, abs=1e-6)


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

    nsat_posterior = dict(posterior)
    nsat_posterior["nsat"] = np.asarray(
        _nsat_from_kfsat(jnp.asarray(nsat_posterior.pop("kfsat")))
    )
    computed_from_nsat = compute_skyrme_neps(nsat_posterior, fixed_params)
    for name in SKYRME_RESULT_NEP_KEYS:
        np.testing.assert_allclose(computed_from_nsat[name], computed[name])


def test_skyrme_construct_eos_extra_constraints_include_neps() -> None:
    params = _sample_params()
    model = Skyrme_EOS_model(ndat=8, proton_fraction="approx")
    result = model.construct_eos(params, return_extra=True)
    extra = result[-1]

    for key in (
        "E_sat", "P_sat", "K_sat", "Q_sat", "Z_sat",
        "E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym",
    ):
        assert key in extra

    assert extra["esm0"] == pytest.approx(extra["E_sat"])
    assert extra["J0"] == pytest.approx(extra["E_sym"])
    assert extra["psm0"] == pytest.approx(extra["P_sat"])
    # These quantities must be evaluated at the pressure-zero density implied
    # by kfsat, not at the fixed density used to construct the EOS grid.
    assert extra["E_sat"] == pytest.approx(params["av"], abs=1e-6)
    assert extra["P_sat"] == pytest.approx(0.0, abs=1e-4)
    assert extra["K_sat"] == pytest.approx(params["Kinf"], abs=1e-6)


def test_skyrme_construct_eos_accepts_nsat() -> None:
    params = _sample_params()
    params["nsat"] = float(_nsat_from_kfsat(jnp.asarray(params.pop("kfsat"))))
    model = Skyrme_EOS_model(ndat=8, proton_fraction="approx")
    extra = model.construct_eos(params, return_extra=True)[-1]

    assert extra["E_sat"] == pytest.approx(params["av"], abs=1e-6)
    assert extra["P_sat"] == pytest.approx(0.0, abs=1e-4)
    assert extra["K_sat"] == pytest.approx(params["Kinf"], abs=1e-6)


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
