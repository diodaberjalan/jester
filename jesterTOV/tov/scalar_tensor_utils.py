import jax
import jax.numpy as jnp
import jax.nn as jnn

# --------------------------------------------
# This part, define matrix multiplication to solve for matching conditions (as well as it's derivative)
# [ H0OnlyQT(M,q,R)   H0OnlyET(M,q,R)   H0OnlyQS(M,q,R)   H0OnlyES(M,q,R) ]   [ cQT ]   [ H0_int(M,q,R) ]
# [ H0OnlyQT'(M,q,R)  H0OnlyET'(M,q,R)  H0OnlyQS'(M,q,R)  H0OnlyES'(M,q,R) ]  [ cET ] = [ H0'_int(M,q,R) ]
# [ φpOnlyQT(M,q,R)   φpOnlyET(M,q,R)   φpOnlyQS(M,q,R)   φpOnlyES(M,q,R) ]  [ cQS ]   [ φp_int(M,q,R) ]
# [ φpOnlyQT'(M,q,R)  φpOnlyET'(M,q,R)  φpOnlyQS'(M,q,R)  φpOnlyES'(M,q,R) ]  [ cES ]   [ φp'_int(M,q,R) ]
# Left matrix from infinity expansion, right matrix from tov solver, and c matrix is what we solve for to determine lambdas.


def build_exterior_basis(M, q, R):
    r"""
    Build exterior basis functions for matching at stellar surface.

    Constructs the 2x4 matrix of exterior basis functions (H0 and φ' components)
    evaluated at the stellar surface radius R.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    R : jax.numpy.ndarray
        Stellar radius (matching point).

    Returns
    -------
    tuple
        Two lists of length 4 each:
        - First list: H0 basis functions [QT, ET, QS, ES]
        - Second list: φ' basis functions [QT, ET, QS, ES]
    """
    return [
        H0OnlyQT_jax(M, q, R),
        H0OnlyET_jax(M, q, R),
        H0OnlyQS_jax(M, q, R),
        H0OnlyES_jax(M, q, R),
    ], [
        PhiPOnlyQT_jax(M, q, R),
        PhiPOnlyET_jax(M, q, R),
        PhiPOnlyQS_jax(M, q, R),
        PhiPOnlyES_jax(M, q, R),
    ]


def build_exterior_basis_autodiff(M, q, R):
    r"""
    Build derivatives of exterior basis functions with respect to radius.

    Uses automatic differentiation (JAX grad) to compute radial derivatives
    of the basis functions at the stellar surface.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    R : jax.numpy.ndarray
        Stellar radius (matching point).

    Returns
    -------
    tuple
        Two lists of length 4 each:
        - First list: dH0/dr basis functions [QT', ET', QS', ES']
        - Second list: dφ'/dr basis functions [QT', ET', QS', ES']
    """
    H0OnlyQT_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyQT_jax, argnums=2)(M, q, R)
    H0OnlyET_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyET_jax, argnums=2)(M, q, R)
    H0OnlyQS_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyQS_jax, argnums=2)(M, q, R)
    H0OnlyES_autodiff_jax = lambda M, q, R: jax.grad(H0OnlyES_jax, argnums=2)(M, q, R)
    PhiPOnlyQT_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyQT_jax, argnums=2)(
        M, q, R
    )
    PhiPOnlyET_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyET_jax, argnums=2)(
        M, q, R
    )
    PhiPOnlyQS_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyQS_jax, argnums=2)(
        M, q, R
    )
    PhiPOnlyES_autodiff_jax = lambda M, q, R: jax.grad(PhiPOnlyES_jax, argnums=2)(
        M, q, R
    )
    return [
        H0OnlyQT_autodiff_jax(M, q, R),
        H0OnlyET_autodiff_jax(M, q, R),
        H0OnlyQS_autodiff_jax(M, q, R),
        H0OnlyES_autodiff_jax(M, q, R),
    ], [
        PhiPOnlyQT_autodiff_jax(M, q, R),
        PhiPOnlyET_autodiff_jax(M, q, R),
        PhiPOnlyQS_autodiff_jax(M, q, R),
        PhiPOnlyES_autodiff_jax(M, q, R),
    ]


def coeff_solver(interior_sol, exterior_basis, exterior_basis_prime):
    r"""
    Solve for exterior basis coefficients matching interior solution.

    Solves the linear system:
    .. math::
        A \\cdot \\mathbf{c} = \\mathbf{b}
    where :math:`A` is the 4x4 matrix of exterior basis functions (and their
    derivatives), :math:`\\mathbf{c}` are the coefficients (cQT, cET, cQS, cES),
    and :math:`\\mathbf{b}` is the interior solution vector (H0, H0', δφ, δφ').

    Parameters
    ----------
    interior_sol : tuple
        Tuple :math:`(H_0, H_0', \\delta\\phi, \\delta\\phi')` evaluated at surface.
    exterior_basis : tuple
        Two lists of length 4: H0 basis values and φ' basis values.
    exterior_basis_prime : tuple
        Two lists of length 4: dH0/dR basis values and dφ'/dR basis values.

    Returns
    -------
    jax.numpy.ndarray
        Coefficient vector :math:`[c_{QT}, c_{ET}, c_{QS}, c_{ES}]`.
    """
    H0_int, H0_prime_int, delta_phi_int, delta_phi_prime_int = interior_sol
    H0_basis, delta_phi_basis = exterior_basis
    H0_basis_prime, delta_phi_basis_prime = exterior_basis_prime

    # Build the matching matrix (4x4 system)
    # We solve particular solutions coeffs: interior = c1*basis1 + c2*basis2 + c3*basis3 + c4*basis4
    # We use this function also for (4x4) = (2x4) system by setting one column to 0 (particular solution) and move one of rhs term to lhs
    # in above case, we leave this function as it is: only input changed.
    A = jnp.array(
        [
            [H0_basis[0], H0_basis[1], H0_basis[2], H0_basis[3]],  # H0 values
            [
                H0_basis_prime[0],
                H0_basis_prime[1],
                H0_basis_prime[2],
                H0_basis_prime[3],
            ],  # H0' vals
            [
                delta_phi_basis[0],
                delta_phi_basis[1],
                delta_phi_basis[2],
                delta_phi_basis[3],
            ],  # dphi vals
            [
                delta_phi_basis_prime[0],
                delta_phi_basis_prime[1],
                delta_phi_basis_prime[2],
                delta_phi_basis_prime[3],
            ],  # dphi' vals
        ]
    )

    b = jnp.array([H0_int, H0_prime_int, delta_phi_int, delta_phi_prime_int])

    # Solve for coefficients
    coefficients = jnp.linalg.solve(A, b)

    return coefficients


def compute_tidal_deformabilities(coefficients):
    r"""
    Compute tidal deformabilities from matched coefficients.

    Converts the coefficient vector into four tidal deformabilities:
    tensor (:math:`\\Lambda_T`), scalar (:math:`\\Lambda_S`), and two mixed
    scalar-tensor (:math:`\\Lambda_{\\mathrm{ST}1}` and :math:`\\Lambda_{\\mathrm{ST}2}`)
    using the relations from Creci et al. (2023).

    Parameters
    ----------
    coefficients : tuple
        Tuple of six coefficients :math:`(c_{QT1}, c_{QT2}, c_{ET}, c_{QS1}, c_{QS2}, c_{ES})`.

    Returns
    -------
    tuple
        Four tidal deformabilities:
        - lambda_T: tensor deformability
        - lambda_S: scalar deformability
        - lambda_ST1: mixed deformability (tensor perturbation zero)
        - lambda_ST2: mixed deformability (scalar perturbation zero)
        These should satisfy :math:`\\Lambda_{\\mathrm{ST}1} = \\Lambda_{\\mathrm{ST}2}`.
    """
    cQT1, cQT2, cET, cQS1, cQS2, cES = coefficients
    # case 1: scalar = 1, tensor = 0
    # case 2: scalar = 0, tensor = 1
    double_factorial = 3.0  # (2l-1)!! = 3 for l=2

    # Tensor deformability (response to tensor tidal field)
    lambda_T = (1.0 / double_factorial) * (cQT2 / cET)  # scalar pert=0

    # Scalar deformability (response to scalar tidal field)
    lambda_S = (1.0 / double_factorial) * (cQS1 / cES)  # tensor pert=0

    # Mixed scalar-tensor deformability
    lambda_ST1 = (1.0 / double_factorial) * (cQT1 / (2.0 * cES))  # tensor pert = 0
    lambda_ST2 = (1.0 / double_factorial) * (2.0 * cQS2 / cET)  # scalar pert = 0

    return lambda_T, lambda_S, lambda_ST1, lambda_ST2


# Infinity expansion for boundary conditions:
# From Creci et al Jupyter notebook https://community.wolfram.com/groups/-/m/t/3459453
# Used as exterior basis to match conditions


def H0OnlyQT_jax(M, q, r):
    r"""
    Compute the H0 (only QT) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the H0 (only QT) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        29491200.0 * jnp.power(M, 19.0) / (23.0 * jnp.power(r, 22.0))
        - 159004594585600.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 2.0)
        / (22309287.0 * jnp.power(r, 22.0))
        + 6.230429751360435e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 4.0)
        / (4216455243.0 * jnp.power(r, 22.0))
        - 3.438074420771897e17
        * jnp.power(M, 19.0)
        * jnp.power(q, 6.0)
        / (23301463185.0 * jnp.power(r, 22.0))
        + 2.5485251228132252e20
        * jnp.power(M, 19.0)
        * jnp.power(q, 8.0)
        / (33204585038625.0 * jnp.power(r, 22.0))
        - 2.0993833349937666e20
        * jnp.power(M, 19.0)
        * jnp.power(q, 10.0)
        / (99613755115875.0 * jnp.power(r, 22.0))
        + 7.558555508042696e17
        * jnp.power(M, 19.0)
        * jnp.power(q, 12.0)
        / (2554198849125.0 * jnp.power(r, 22.0))
        - 2933799166774592.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 14.0)
        / (150246991125.0 * jnp.power(r, 22.0))
        + 14934094118912.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 16.0)
        / (29515186701.0 * jnp.power(r, 22.0))
        - 24576000.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 18.0)
        / (7436429.0 * jnp.power(r, 22.0))
        + 161873920.0 * jnp.power(M, 18.0) / (253.0 * jnp.power(r, 21.0))
        - 268178337056768.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 2.0)
        / (81800719.0 * jnp.power(r, 21.0))
        + 2.274123155544045e16
        * jnp.power(M, 18.0)
        * jnp.power(q, 4.0)
        / (3681032355.0 * jnp.power(r, 21.0))
        - 4822171191567592.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 6.0)
        / (876436275.0 * jnp.power(r, 21.0))
        + 1.4542063357777052e16
        * jnp.power(M, 18.0)
        * jnp.power(q, 8.0)
        / (5842908500.0 * jnp.power(r, 21.0))
        - 645419377518649.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 10.0)
        / (1124838000.0 * jnp.power(r, 21.0))
        + 3.60184371231355e16
        * jnp.power(M, 18.0)
        * jnp.power(q, 12.0)
        / (560919216000.0 * jnp.power(r, 21.0))
        - 2316422291259147.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 14.0)
        / (747892288000.0 * jnp.power(r, 21.0))
        + 45672613590959.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 16.0)
        / (927846678528.0 * jnp.power(r, 21.0))
        - 1154725.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 18.0)
        / (10551296.0 * jnp.power(r, 21.0))
        + 24576000.0 * jnp.power(M, 17.0) / (77.0 * jnp.power(r, 20.0))
        - 112034631397376.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 2.0)
        / (74687613.0 * jnp.power(r, 20.0))
        + 3.6019803593214464e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 4.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        - 2.833519571131187e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 6.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 9.891112547831296e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 8.0)
        / (127043629713.0 * jnp.power(r, 20.0))
        - 567872463104000.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 10.0)
        / (3849806961.0 * jnp.power(r, 20.0))
        + 31632352013824.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 12.0)
        / (2491051563.0 * jnp.power(r, 20.0))
        - 1967882196992.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 14.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        + 12091392.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 16.0)
        / (3556553.0 * jnp.power(r, 20.0))
        + 1114112.0 * jnp.power(M, 16.0) / (7.0 * jnp.power(r, 19.0))
        - 6316583848448.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 2.0)
        / (9258795.0 * jnp.power(r, 19.0))
        + 3.3344712951069376e16
        * jnp.power(M, 16.0)
        * jnp.power(q, 4.0)
        / (32081724675.0 * jnp.power(r, 19.0))
        - 1.0412053914895918e16
        * jnp.power(M, 16.0)
        * jnp.power(q, 6.0)
        / (14582602125.0 * jnp.power(r, 19.0))
        + 2.6960628142255503e18
        * jnp.power(M, 16.0)
        * jnp.power(q, 8.0)
        / (11549420883000.0 * jnp.power(r, 19.0))
        - 1.6342736992086124e18
        * jnp.power(M, 16.0)
        * jnp.power(q, 10.0)
        / (46197683532000.0 * jnp.power(r, 19.0))
        + 1.0581779822765676e16
        * jnp.power(M, 16.0)
        * jnp.power(q, 12.0)
        / (4738223952000.0 * jnp.power(r, 19.0))
        - 9908370878981.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 14.0)
        / (219011240448.0 * jnp.power(r, 19.0))
        + 546975.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 16.0)
        / (4358144.0 * jnp.power(r, 19.0))
        + 1507328.0 * jnp.power(M, 15.0) / (19.0 * jnp.power(r, 18.0))
        - 1494491556352.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 2.0)
        / (4849845.0 * jnp.power(r, 18.0))
        + 23572004454464.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 4.0)
        / (56581525.0 * jnp.power(r, 18.0))
        - 69746327683944.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 6.0)
        / (282907625.0 * jnp.power(r, 18.0))
        + 56461198314712.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 8.0)
        / (848722875.0 * jnp.power(r, 18.0))
        - 2202458527328.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 10.0)
        / (282907625.0 * jnp.power(r, 18.0))
        + 95267171072.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 12.0)
        / (282907625.0 * jnp.power(r, 18.0))
        - 2424832.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 14.0)
        / (692835.0 * jnp.power(r, 18.0))
        + 2252800.0 * jnp.power(M, 14.0) / (57.0 * jnp.power(r, 17.0))
        - 1205476363264.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 2.0)
        / (8729721.0 * jnp.power(r, 17.0))
        + 50070815769328.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 4.0)
        / (305540235.0 * jnp.power(r, 17.0))
        - 86903812826108.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 6.0)
        / (1057639275.0 * jnp.power(r, 17.0))
        + 1959948682174993.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 8.0)
        / (109994484600.0 * jnp.power(r, 17.0))
        - 225207514065389.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 10.0)
        / (146659312800.0 * jnp.power(r, 17.0))
        + 427651340075.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 12.0)
        / (10429106688.0 * jnp.power(r, 17.0))
        - 96525.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 14.0)
        / (661504.0 * jnp.power(r, 17.0))
        + 1003520.0 * jnp.power(M, 13.0) / (51.0 * jnp.power(r, 16.0))
        - 28169575424.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 2.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 1013852314624.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 4.0)
        / (16081065.0 * jnp.power(r, 16.0))
        - 2712695658496.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 6.0)
        / (103378275.0 * jnp.power(r, 16.0))
        + 3196414281728.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 8.0)
        / (723647925.0 * jnp.power(r, 16.0))
        - 63199271936.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 10.0)
        / (241215975.0 * jnp.power(r, 16.0))
        + 7168.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 12.0)
        / (1989.0 * jnp.power(r, 16.0))
        + 166400.0 * jnp.power(M, 12.0) / (17.0 * jnp.power(r, 15.0))
        - 1374576608.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 2.0)
        / (51051.0 * jnp.power(r, 15.0))
        + 8439472838.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 4.0)
        / (357357.0 * jnp.power(r, 15.0))
        - 24597221.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 6.0)
        / (3094.0 * jnp.power(r, 15.0))
        + 11333729111.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 8.0)
        / (11435424.0 * jnp.power(r, 15.0))
        - 417084431.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 10.0)
        / (11435424.0 * jnp.power(r, 15.0))
        + 3003.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 12.0)
        / (17408.0 * jnp.power(r, 15.0))
        + 4864.0 * jnp.power(M, 11.0) / jnp.power(r, 14.0)
        - 58413792.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 2.0)
        / (5005.0 * jnp.power(r, 14.0))
        + 5785354852.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 4.0)
        / (675675.0 * jnp.power(r, 14.0))
        - 4567772324.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 6.0)
        / (2027025.0 * jnp.power(r, 14.0))
        + 14607952.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 8.0)
        / (75075.0 * jnp.power(r, 14.0))
        - 3712.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 10.0)
        / (1001.0 * jnp.power(r, 14.0))
        + 16896.0 * jnp.power(M, 10.0) / (7.0 * jnp.power(r, 13.0))
        - 1569899984.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 2.0)
        / (315315.0 * jnp.power(r, 13.0))
        + 2013679711.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 4.0)
        / (675675.0 * jnp.power(r, 13.0))
        - 11078698957.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 6.0)
        / (18918900.0 * jnp.power(r, 13.0))
        + 3187331519.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 8.0)
        / (100900800.0 * jnp.power(r, 13.0))
        - 693.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 10.0)
        / (3328.0 * jnp.power(r, 13.0))
        + 108800.0 * jnp.power(M, 9.0) / (91.0 * jnp.power(r, 12.0))
        - 43769248.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 2.0)
        / (21021.0 * jnp.power(r, 12.0))
        + 703808.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 4.0)
        / (715.0 * jnp.power(r, 12.0))
        - 4736224.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 6.0)
        / (35035.0 * jnp.power(r, 12.0))
        + 80000.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 8.0)
        / (21021.0 * jnp.power(r, 12.0))
        + 7680.0 * jnp.power(M, 8.0) / (13.0 * jnp.power(r, 11.0))
        - 7648268.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 2.0)
        / (9009.0 * jnp.power(r, 11.0))
        + 1517584.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 4.0)
        / (5005.0 * jnp.power(r, 11.0))
        - 1578641.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 6.0)
        / (60060.0 * jnp.power(r, 11.0))
        + 4725.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 8.0)
        / (18304.0 * jnp.power(r, 11.0))
        + 3200.0 * jnp.power(M, 7.0) / (11.0 * jnp.power(r, 10.0))
        - 231988.0
        * jnp.power(M, 7.0)
        * jnp.power(q, 2.0)
        / (693.0 * jnp.power(r, 10.0))
        + 58724.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (693.0 * jnp.power(r, 10.0))
        - 128.0 * jnp.power(M, 7.0) * jnp.power(q, 6.0) / (33.0 * jnp.power(r, 10.0))
        + 1568.0 * jnp.power(M, 6.0) / (11.0 * jnp.power(r, 9.0))
        - 145616.0
        * jnp.power(M, 6.0)
        * jnp.power(q, 2.0)
        / (1155.0 * jnp.power(r, 9.0))
        + 94613.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (4620.0 * jnp.power(r, 9.0))
        - 175.0 * jnp.power(M, 6.0) * jnp.power(q, 6.0) / (528.0 * jnp.power(r, 9.0))
        + 208.0 * jnp.power(M, 5.0) / (3.0 * jnp.power(r, 8.0))
        - 4664.0 * jnp.power(M, 5.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 8.0))
        + 136.0 * jnp.power(M, 5.0) * jnp.power(q, 4.0) / (35.0 * jnp.power(r, 8.0))
        + 100.0 * jnp.power(M, 4.0) / (3.0 * jnp.power(r, 7.0))
        - 296.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 7.0))
        + 25.0 * jnp.power(M, 4.0) * jnp.power(q, 4.0) / (56.0 * jnp.power(r, 7.0))
        + 110.0 * jnp.power(M, 3.0) / (7.0 * jnp.power(r, 6.0))
        - 26.0 * jnp.power(M, 3.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0))
        + 50.0 * jnp.power(M, 2.0) / (7.0 * jnp.power(r, 5.0))
        - 9.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / (14.0 * jnp.power(r, 5.0))
        + 3.0 * M / jnp.power(r, 4.0)
        + jnp.power(r, -3.0)
    )


def PhiPOnlyQT_jax(M, q, r):
    r"""
    Compute the PhiP (only QT) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the PhiP (only QT) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        93388800.0 * jnp.power(M, 19.0) * q / (253.0 * jnp.power(r, 22.0))
        - 1982408089600.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 3.0)
        / (1312311.0 * jnp.power(r, 22.0))
        + 9202652339847040.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 5.0)
        / (4216455243.0 * jnp.power(r, 22.0))
        - 6709158052328800.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 7.0)
        / (4660292637.0 * jnp.power(r, 22.0))
        + 3.095625106257937e18
        * jnp.power(M, 19.0)
        * jnp.power(q, 9.0)
        / (6640917007725.0 * jnp.power(r, 22.0))
        - 1.4702022486024916e18
        * jnp.power(M, 19.0)
        * jnp.power(q, 11.0)
        / (19922751023175.0 * jnp.power(r, 22.0))
        + 1.0618124700348515e17
        * jnp.power(M, 19.0)
        * jnp.power(q, 13.0)
        / (19922751023175.0 * jnp.power(r, 22.0))
        - 327839775523072.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 15.0)
        / (2213639002575.0 * jnp.power(r, 22.0))
        + 7602176.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 17.0)
        / (7436429.0 * jnp.power(r, 22.0))
        + 317521920.0 * jnp.power(M, 18.0) * q / (1771.0 * jnp.power(r, 21.0))
        - 54248329565184.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 3.0)
        / (81800719.0 * jnp.power(r, 21.0))
        + 11457631522208.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 5.0)
        / (13483635.0 * jnp.power(r, 21.0))
        - 424943704339756.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 7.0)
        / (876436275.0 * jnp.power(r, 21.0))
        + 919918368991283.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 9.0)
        / (7011490200.0 * jnp.power(r, 21.0))
        - 231217475648777.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 11.0)
        / (14022980400.0 * jnp.power(r, 21.0))
        + 194652894683933.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 13.0)
        / (224367686400.0 * jnp.power(r, 21.0))
        - 926258551895.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 15.0)
        / (62822952192.0 * jnp.power(r, 21.0))
        + 182325.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 17.0)
        / (5275648.0 * jnp.power(r, 21.0))
        + 6684672.0 * jnp.power(M, 17.0) * q / (77.0 * jnp.power(r, 20.0))
        - 21515986997248.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 3.0)
        / (74687613.0 * jnp.power(r, 20.0))
        + 4574082043518976.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 5.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        - 2225270466590720.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 7.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 4431902475622400.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 9.0)
        / (127043629713.0 * jnp.power(r, 20.0))
        - 47407471925248.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 11.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 1697292904448.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 13.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        - 33357824.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 15.0)
        / (32008977.0 * jnp.power(r, 20.0))
        + 5570560.0 * jnp.power(M, 16.0) * q / (133.0 * jnp.power(r, 19.0))
        - 2518049131264.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 3.0)
        / (20369349.0 * jnp.power(r, 19.0))
        + 773884913000656.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 5.0)
        / (6416344935.0 * jnp.power(r, 19.0))
        - 63088965880649.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 7.0)
        / (1283268987.0 * jnp.power(r, 19.0))
        + 363573767832881.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 9.0)
        / (41997894120.0 * jnp.power(r, 19.0))
        - 2248987038532789.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 11.0)
        / (3695814682560.0 * jnp.power(r, 19.0))
        + 8772734045155.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 13.0)
        / (657033721344.0 * jnp.power(r, 19.0))
        - 10725.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 15.0)
        / (272384.0 * jnp.power(r, 19.0))
        + 1146880.0 * jnp.power(M, 15.0) * q / (57.0 * jnp.power(r, 18.0))
        - 16908159232.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 3.0)
        / (323323.0 * jnp.power(r, 18.0))
        + 493734180064.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 5.0)
        / (11316305.0 * jnp.power(r, 18.0))
        - 495090090644.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 7.0)
        / (33948915.0 * jnp.power(r, 18.0))
        + 455469736.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 9.0)
        / (230945.0 * jnp.power(r, 18.0))
        - 1070016608.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 11.0)
        / (11316305.0 * jnp.power(r, 18.0))
        + 146944.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 13.0)
        / (138567.0 * jnp.power(r, 18.0))
        + 9318400.0 * jnp.power(M, 14.0) * q / (969.0 * jnp.power(r, 17.0))
        - 189892217600.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 3.0)
        / (8729721.0 * jnp.power(r, 17.0))
        + 932113326568.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 5.0)
        / (61108047.0 * jnp.power(r, 17.0))
        - 1598428684781.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 7.0)
        / (392837445.0 * jnp.power(r, 17.0))
        + 1363525754411.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 9.0)
        / (3384445680.0 * jnp.power(r, 17.0))
        - 232077111443.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 11.0)
        / (19554575040.0 * jnp.power(r, 17.0))
        + 15015.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 13.0)
        / (330752.0 * jnp.power(r, 17.0))
        + 232960.0 * jnp.power(M, 13.0) * q / (51.0 * jnp.power(r, 16.0))
        - 4074131968.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 3.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 82274075648.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 5.0)
        / (16081065.0 * jnp.power(r, 16.0))
        - 152179698688.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 7.0)
        / (144729585.0 * jnp.power(r, 16.0))
        + 10307691008.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 9.0)
        / (144729585.0 * jnp.power(r, 16.0))
        - 494080.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 11.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 36608.0 * jnp.power(M, 12.0) * q / (17.0 * jnp.power(r, 15.0))
        - 25723376.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 3.0)
        / (7293.0 * jnp.power(r, 15.0))
        + 194177083.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 5.0)
        / (119119.0 * jnp.power(r, 15.0))
        - 58637561.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 7.0)
        / (238238.0 * jnp.power(r, 15.0))
        + 235028551.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 9.0)
        / (22870848.0 * jnp.power(r, 15.0))
        - 231.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 11.0)
        / (4352.0 * jnp.power(r, 15.0))
        + 7040.0 * jnp.power(M, 11.0) * q / (7.0 * jnp.power(r, 14.0))
        - 1362448.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 3.0)
        / (1001.0 * jnp.power(r, 14.0))
        + 13150838.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 5.0)
        / (27027.0 * jnp.power(r, 14.0))
        - 4090336.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 7.0)
        / (81081.0 * jnp.power(r, 14.0))
        + 9760.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 9.0)
        / (9009.0 * jnp.power(r, 14.0))
        + 42240.0 * jnp.power(M, 10.0) * q / (91.0 * jnp.power(r, 13.0))
        - 31877096.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 3.0)
        / (63063.0 * jnp.power(r, 13.0))
        + 50453185.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 5.0)
        / (378378.0 * jnp.power(r, 13.0))
        - 1618798.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 7.0)
        / (189189.0 * jnp.power(r, 13.0))
        + 105.0 * jnp.power(M, 10.0) * jnp.power(q, 9.0) / (1664.0 * jnp.power(r, 13.0))
        + 19200.0 * jnp.power(M, 9.0) * q / (91.0 * jnp.power(r, 12.0))
        - 1251520.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 3.0)
        / (7007.0 * jnp.power(r, 12.0))
        + 227776.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 5.0)
        / (7007.0 * jnp.power(r, 12.0))
        - 7552.0 * jnp.power(M, 9.0) * jnp.power(q, 7.0) / (7007.0 * jnp.power(r, 12.0))
        + 13440.0 * jnp.power(M, 8.0) * q / (143.0 * jnp.power(r, 11.0))
        - 531766.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 3.0)
        / (9009.0 * jnp.power(r, 11.0))
        + 80291.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 5.0)
        / (12012.0 * jnp.power(r, 11.0))
        - 175.0 * jnp.power(M, 8.0) * jnp.power(q, 7.0) / (2288.0 * jnp.power(r, 11.0))
        + 448.0 * jnp.power(M, 7.0) * q / (11.0 * jnp.power(r, 10.0))
        - 12302.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (693.0 * jnp.power(r, 10.0))
        + 724.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (693.0 * jnp.power(r, 10.0))
        + 560.0 * jnp.power(M, 6.0) * q / (33.0 * jnp.power(r, 9.0))
        - 1070.0 * jnp.power(M, 6.0) * jnp.power(q, 3.0) / (231.0 * jnp.power(r, 9.0))
        + 25.0 * jnp.power(M, 6.0) * jnp.power(q, 5.0) / (264.0 * jnp.power(r, 9.0))
        + 20.0 * jnp.power(M, 5.0) * q / (3.0 * jnp.power(r, 8.0))
        - 20.0 * jnp.power(M, 5.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 8.0))
        + 50.0 * jnp.power(M, 4.0) * q / (21.0 * jnp.power(r, 7.0))
        - 5.0 * jnp.power(M, 4.0) * jnp.power(q, 3.0) / (42.0 * jnp.power(r, 7.0))
        + 5.0 * jnp.power(M, 3.0) * q / (7.0 * jnp.power(r, 6.0))
        + jnp.power(M, 2.0) * q / (7.0 * jnp.power(r, 5.0))
    )


def H0OnlyET_jax(M, q, r):
    r"""
    Compute the H0 (only ET) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the H0 (only ET) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        2.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / 3.0
        - 1795948544.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 2.0)
        / (1311.0 * jnp.power(r, 22.0))
        + 3026355961364480.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 4.0)
        / (423876453.0 * jnp.power(r, 22.0))
        - 9.934062256455117e16
        * jnp.power(M, 24.0)
        * jnp.power(q, 6.0)
        / (7323317001.0 * jnp.power(r, 22.0))
        + 1.66948524541456e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 8.0)
        / (1387951654614525.0 * jnp.power(r, 22.0))
        - 5.494650013306427e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 10.0)
        / (1.0409637409608938e17 * jnp.power(r, 22.0))
        + 3.449339897623955e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 12.0)
        / (3.122891222882681e17 * jnp.power(r, 22.0))
        - 1.712772250439323e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 14.0)
        / (1.9119742180914376e16 * jnp.power(r, 22.0))
        - 1.0276506807305192e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 16.0)
        / (2.810602100594413e18 * jnp.power(r, 22.0))
        + 1.6998731009782561e18
        * jnp.power(M, 24.0)
        * jnp.power(q, 18.0)
        / (6373247393638125.0 * jnp.power(r, 22.0))
        - 70522962808832.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 20.0)
        / (16328842995465.0 * jnp.power(r, 22.0))
        - 301613056.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 22.0)
        / (13987922949.0 * jnp.power(r, 22.0))
        - 172228608.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 2.0)
        / (253.0 * jnp.power(r, 21.0))
        + 1331923272265728.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 4.0)
        / (409003595.0 * jnp.power(r, 21.0))
        - 1.1419773419217792e16
        * jnp.power(M, 23.0)
        * jnp.power(q, 6.0)
        / (2045017975.0 * jnp.power(r, 21.0))
        + 3431446897968256.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 8.0)
        / (786545375.0 * jnp.power(r, 21.0))
        - 2.373750847096053e16
        * jnp.power(M, 23.0)
        * jnp.power(q, 10.0)
        / (14607271250.0 * jnp.power(r, 21.0))
        + 259748182786263.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 12.0)
        / (965770000.0 * jnp.power(r, 21.0))
        - 6318613505327741.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 14.0)
        / (467432680000.0 * jnp.power(r, 21.0))
        - 2127504567244611.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 16.0)
        / (3739461440000.0 * jnp.power(r, 21.0))
        + 1.7427823814337472e16
        * jnp.power(M, 23.0)
        * jnp.power(q, 18.0)
        / (418819681280000.0 * jnp.power(r, 21.0))
        - 721589919151.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 20.0)
        / (5360891920384.0 * jnp.power(r, 21.0))
        - 37791.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 22.0)
        / (21102592.0 * jnp.power(r, 21.0))
        - 11141120.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 2.0)
        / (33.0 * jnp.power(r, 20.0))
        + 1651285448531968.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 4.0)
        / (1120314195.0 * jnp.power(r, 20.0))
        - 4.790558621575209e17
        * jnp.power(M, 22.0)
        * jnp.power(q, 6.0)
        / (211739382855.0 * jnp.power(r, 20.0))
        + 6.511366545685005e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 8.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        - 6.968485180692378e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 10.0)
        / (146588803515.0 * jnp.power(r, 20.0))
        + 2.247140858365107e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 12.0)
        / (381130889139.0 * jnp.power(r, 20.0))
        - 2818816980099584.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 14.0)
        / (2450127144465.0 * jnp.power(r, 20.0))
        - 328915430605184.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 16.0)
        / (1905654445695.0 * jnp.power(r, 20.0))
        + 187412715008.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 18.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        + 8192.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 20.0)
        / (440895.0 * jnp.power(r, 20.0))
        - 17563648.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 2.0)
        / (105.0 * jnp.power(r, 19.0))
        + 48176937714688.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 4.0)
        / (72747675.0 * jnp.power(r, 19.0))
        - 8.67524307235769e16
        * jnp.power(M, 21.0)
        * jnp.power(q, 6.0)
        / (96245174025.0 * jnp.power(r, 19.0))
        + 1.2627038878827817e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 8.0)
        / (2406129350625.0 * jnp.power(r, 19.0))
        - 2.8324664939136896e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 10.0)
        / (21655164155625.0 * jnp.power(r, 19.0))
        + 1.9253399483808276e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 12.0)
        / (173241313245000.0 * jnp.power(r, 19.0))
        + 2253864884321707.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 14.0)
        / (14920304490000.0 * jnp.power(r, 19.0))
        - 7.457196187887162e17
        * jnp.power(M, 21.0)
        * jnp.power(q, 16.0)
        / (2.217488809536e16 * jnp.power(r, 19.0))
        + 1225448279513.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 18.0)
        / (5631717611520.0 * jnp.power(r, 19.0))
        + 26741.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 20.0)
        / (13074432.0 * jnp.power(r, 19.0))
        - 1572864.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 2.0)
        / (19.0 * jnp.power(r, 18.0))
        + 30971278336.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 4.0)
        / (104975.0 * jnp.power(r, 18.0))
        - 99607926089216.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 6.0)
        / (282907625.0 * jnp.power(r, 18.0))
        + 243697859241888.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 8.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        - 46998257904314.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 10.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        + 2255447520022.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 12.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        + 136801791864.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 14.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        - 6221321152.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 16.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        - 16384.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 18.0)
        / (1154725.0 * jnp.power(r, 18.0))
        - 6995968.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 2.0)
        / (171.0 * jnp.power(r, 17.0))
        + 17041946358784.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 4.0)
        / (130945815.0 * jnp.power(r, 17.0))
        - 205224782811392.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 6.0)
        / (1527701175.0 * jnp.power(r, 17.0))
        + 1.1114967845506816e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 8.0)
        / (206239658625.0 * jnp.power(r, 17.0))
        - 1.2460283668445272e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 10.0)
        / (1649917269000.0 * jnp.power(r, 17.0))
        + 5405548485623711.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 12.0)
        / (59397021684000.0 * jnp.power(r, 17.0))
        + 4999186602827.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 14.0)
        / (198486288000.0 * jnp.power(r, 17.0))
        - 113605554383.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 16.0)
        / (375447840768.0 * jnp.power(r, 17.0))
        - 25025.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 18.0)
        / (10584064.0 * jnp.power(r, 17.0))
        - 3088384.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 2.0)
        / (153.0 * jnp.power(r, 16.0))
        + 391057583104.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 4.0)
        / (6891885.0 * jnp.power(r, 16.0))
        - 7139858176.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 6.0)
        / (143325.0 * jnp.power(r, 16.0))
        + 172020744946432.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 8.0)
        / (10854718875.0 * jnp.power(r, 16.0))
        - 15670155663872.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 10.0)
        / (10854718875.0 * jnp.power(r, 16.0))
        - 3725198288384.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 12.0)
        / (97692469875.0 * jnp.power(r, 16.0))
        + 45271423744.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 14.0)
        / (10854718875.0 * jnp.power(r, 16.0))
        + 7936.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 16.0)
        / (984555.0 * jnp.power(r, 16.0))
        - 168960.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 2.0)
        / (17.0 * jnp.power(r, 15.0))
        + 22808576.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 4.0)
        / (935.0 * jnp.power(r, 15.0))
        - 2124241628.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 6.0)
        / (119119.0 * jnp.power(r, 15.0))
        + 1459391631.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 8.0)
        / (340340.0 * jnp.power(r, 15.0))
        - 1859605213.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 10.0)
        / (9529520.0 * jnp.power(r, 15.0))
        - 57914453.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 12.0)
        / (3465280.0 * jnp.power(r, 15.0))
        + 528387.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 14.0)
        / (1361360.0 * jnp.power(r, 15.0))
        + 3861.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 16.0)
        / (1392640.0 * jnp.power(r, 15.0))
        - 73216.0 * jnp.power(M, 16.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 14.0))
        + 6965811968.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 4.0)
        / (675675.0 * jnp.power(r, 14.0))
        - 12370991536.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 6.0)
        / (2027025.0 * jnp.power(r, 14.0))
        + 6254576641.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 8.0)
        / (6081075.0 * jnp.power(r, 14.0))
        - 42789671.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 10.0)
        / (18243225.0 * jnp.power(r, 14.0))
        - 7546076.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 12.0)
        / (2027025.0 * jnp.power(r, 14.0))
        + 32.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 14.0)
        / (45045.0 * jnp.power(r, 14.0))
        - 7168.0 * jnp.power(M, 15.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 13.0))
        + 20171485856.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 4.0)
        / (4729725.0 * jnp.power(r, 13.0))
        - 139608820654.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 6.0)
        / (70945875.0 * jnp.power(r, 13.0))
        + 171954172043.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 8.0)
        / (851350500.0 * jnp.power(r, 13.0))
        + 1955084741.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 10.0)
        / (224532000.0 * jnp.power(r, 13.0))
        - 8558330503.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 12.0)
        / (18162144000.0 * jnp.power(r, 13.0))
        - 11.0 * jnp.power(M, 15.0) * jnp.power(q, 14.0) / (3328.0 * jnp.power(r, 13.0))
        - 105984.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 2.0)
        / (91.0 * jnp.power(r, 12.0))
        + 781056.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 4.0)
        / (455.0 * jnp.power(r, 12.0))
        - 102244488.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 6.0)
        / (175175.0 * jnp.power(r, 12.0))
        + 626256.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 8.0)
        / (25025.0 * jnp.power(r, 12.0))
        + 531288.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 10.0)
        / (175175.0 * jnp.power(r, 12.0))
        - 96.0 * jnp.power(M, 14.0) * jnp.power(q, 12.0) / (7007.0 * jnp.power(r, 12.0))
        - 22016.0 * jnp.power(M, 13.0) * jnp.power(q, 2.0) / (39.0 * jnp.power(r, 11.0))
        + 989368.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 4.0)
        / (1485.0 * jnp.power(r, 11.0))
        - 20588222.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 6.0)
        / (135135.0 * jnp.power(r, 11.0))
        - 17175089.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 8.0)
        / (9729720.0 * jnp.power(r, 11.0))
        + 1179523.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 10.0)
        / (2162160.0 * jnp.power(r, 11.0))
        + 147.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 12.0)
        / (36608.0 * jnp.power(r, 11.0))
        - 8960.0 * jnp.power(M, 12.0) * jnp.power(q, 2.0) / (33.0 * jnp.power(r, 10.0))
        + 2554984.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 4.0)
        / (10395.0 * jnp.power(r, 10.0))
        - 65621.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 6.0)
        / (2079.0 * jnp.power(r, 10.0))
        - 194983.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 8.0)
        / (93555.0 * jnp.power(r, 10.0))
        + 32.0 * jnp.power(M, 12.0) * jnp.power(q, 10.0) / (945.0 * jnp.power(r, 10.0))
        - 7104.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (55.0 * jnp.power(r, 9.0))
        + 161824.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 4.0)
        / (1925.0 * jnp.power(r, 9.0))
        - 933.0 * jnp.power(M, 11.0) * jnp.power(q, 6.0) / (275.0 * jnp.power(r, 9.0))
        - 18399.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 8.0)
        / (30800.0 * jnp.power(r, 9.0))
        - 7.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (1408.0 * jnp.power(r, 9.0))
        - 544.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 8.0))
        + 39916.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 4.0)
        / (1575.0 * jnp.power(r, 8.0))
        + 12514.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 6.0)
        / (14175.0 * jnp.power(r, 8.0))
        - 106.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (1575.0 * jnp.power(r, 8.0))
        - 248.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 7.0))
        + 1856.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (315.0 * jnp.power(r, 7.0))
        + 241.0 * jnp.power(M, 9.0) * jnp.power(q, 6.0) / (405.0 * jnp.power(r, 7.0))
        + 25.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (4032.0 * jnp.power(r, 7.0))
        - 12.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / jnp.power(r, 6.0)
        + 33.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (70.0 * jnp.power(r, 6.0))
        + 9.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (70.0 * jnp.power(r, 6.0))
        - 100.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 5.0))
        - 289.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (630.0 * jnp.power(r, 5.0))
        - jnp.power(M, 7.0) * jnp.power(q, 6.0) / (140.0 * jnp.power(r, 5.0))
        - 22.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 4.0))
        - 47.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (180.0 * jnp.power(r, 4.0))
        + 2.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 2.0))
        + jnp.power(M, 3.0) * jnp.power(q, 2.0) / (3.0 * r)
        - 2.0 * M * r
        + jnp.power(r, 2.0)
    )


def PhiPOnlyET_jax(M, q, r):
    r"""
    Compute the PhiP (only ET) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the PhiP (only ET) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        -0.3333333333333333 * (jnp.power(M, 2.0) * q)
        - 1048576.0 * jnp.power(M, 24.0) * q / (57.0 * jnp.power(r, 22.0))
        - 108047362342912.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 3.0)
        / (423876453.0 * jnp.power(r, 22.0))
        + 1.2223025594248648e18
        * jnp.power(M, 24.0)
        * jnp.power(q, 5.0)
        / (1016814398985.0 * jnp.power(r, 22.0))
        - 2.415993268667906e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 7.0)
        / (1387951654614525.0 * jnp.power(r, 22.0))
        + 1.1372707315866402e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 9.0)
        / (1.0409637409608938e17 * jnp.power(r, 22.0))
        - 1.56978117441629e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 11.0)
        / (4.99662595661229e17 * jnp.power(r, 22.0))
        + 1.2677820661047214e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 13.0)
        / (3.52703008702044e17 * jnp.power(r, 22.0))
        + 8.765493299428505e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 15.0)
        / (5.756113102017358e19 * jnp.power(r, 22.0))
        - 1.3175059756885878e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 17.0)
        / (4.263787482975821e19 * jnp.power(r, 22.0))
        + 4.357109942093411e19
        * jnp.power(M, 24.0)
        * jnp.power(q, 19.0)
        / (2.35405796161536e18 * jnp.power(r, 22.0))
        - 3666737926669729.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 21.0)
        / (1.178630380781568e16 * jnp.power(r, 22.0))
        + 1547.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 23.0)
        / (2097152.0 * jnp.power(r, 22.0))
        - 381026304.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 3.0)
        / (1771.0 * jnp.power(r, 21.0))
        + 336489492363264.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 5.0)
        / (409003595.0 * jnp.power(r, 21.0))
        - 2288720172911296.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 7.0)
        / (2045017975.0 * jnp.power(r, 21.0))
        + 54530206363968.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 9.0)
        / (76880375.0 * jnp.power(r, 21.0))
        - 1344862073331039.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 11.0)
        / (5842908500.0 * jnp.power(r, 21.0))
        + 1844788271586391.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 13.0)
        / (46743268000.0 * jnp.power(r, 21.0))
        - 657087845981487.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 15.0)
        / (186973072000.0 * jnp.power(r, 21.0))
        + 1547821973166531.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 17.0)
        / (10470492032000.0 * jnp.power(r, 21.0))
        - 131287333403.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 19.0)
        / (58270564352.0 * jnp.power(r, 21.0))
        + 109395.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 21.0)
        / (21102592.0 * jnp.power(r, 21.0))
        - 40108032.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 3.0)
        / (385.0 * jnp.power(r, 20.0))
        + 44652955148288.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 5.0)
        / (124479355.0 * jnp.power(r, 20.0))
        - 2032958894531584.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 7.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        + 17319075678208.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 9.0)
        / (72837765.0 * jnp.power(r, 20.0))
        - 2774132700214784.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 11.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        + 115371183150592.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 13.0)
        / (12455257815.0 * jnp.power(r, 20.0))
        - 15246453790208.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 15.0)
        / (23526598095.0 * jnp.power(r, 20.0))
        + 90748965376.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 17.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        - 8339456.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 19.0)
        / (53348295.0 * jnp.power(r, 20.0))
        - 6684672.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 3.0)
        / (133.0 * jnp.power(r, 19.0))
        + 5249385256448.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 5.0)
        / (33948915.0 * jnp.power(r, 19.0))
        - 1746066195088352.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 7.0)
        / (10693908225.0 * jnp.power(r, 19.0))
        + 30531884705802.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 9.0)
        / (396070675.0 * jnp.power(r, 19.0))
        - 131506055976748.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 11.0)
        / (7403474925.0 * jnp.power(r, 19.0))
        + 4110722687299.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 13.0)
        / (2026214190.0 * jnp.power(r, 19.0))
        - 660940017641191.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 15.0)
        / (6159691137600.0 * jnp.power(r, 19.0))
        + 1795939470341.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 17.0)
        / (876044961792.0 * jnp.power(r, 19.0))
        - 6435.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 19.0)
        / (1089536.0 * jnp.power(r, 19.0))
        - 458752.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 3.0)
        / (19.0 * jnp.power(r, 18.0))
        + 106328069632.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 5.0)
        / (1616615.0 * jnp.power(r, 18.0))
        - 13790462592.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 7.0)
        / (229075.0 * jnp.power(r, 18.0))
        + 1360480816336.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 9.0)
        / (56581525.0 * jnp.power(r, 18.0))
        - 51536125009.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 11.0)
        / (11316305.0 * jnp.power(r, 18.0))
        + 23158612446.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 13.0)
        / (56581525.0 * jnp.power(r, 18.0))
        - 874515016.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 15.0)
        / (56581525.0 * jnp.power(r, 18.0))
        + 36736.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 17.0)
        / (230945.0 * jnp.power(r, 18.0))
        - 3727360.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 3.0)
        / (323.0 * jnp.power(r, 17.0))
        + 80154360320.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 5.0)
        / (2909907.0 * jnp.power(r, 17.0))
        - 115607264944.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 7.0)
        / (5360355.0 * jnp.power(r, 17.0))
        + 2987661500984.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 9.0)
        / (416645775.0 * jnp.power(r, 17.0))
        - 13367945464759.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 11.0)
        / (12221609400.0 * jnp.power(r, 17.0))
        + 21903222813317.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 13.0)
        / (293318625600.0 * jnp.power(r, 17.0))
        - 478357587061.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 15.0)
        / (260727667200.0 * jnp.power(r, 17.0))
        + 9009.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 17.0)
        / (1323008.0 * jnp.power(r, 17.0))
        - 93184.0 * jnp.power(M, 18.0) * jnp.power(q, 3.0) / (17.0 * jnp.power(r, 16.0))
        + 8672948096.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 5.0)
        / (765765.0 * jnp.power(r, 16.0))
        - 66732268672.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 7.0)
        / (8933925.0 * jnp.power(r, 16.0))
        + 69925152512.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 9.0)
        / (34459425.0 * jnp.power(r, 16.0))
        - 58660306688.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 11.0)
        / (241215975.0 * jnp.power(r, 16.0))
        + 222168704.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 13.0)
        / (18555075.0 * jnp.power(r, 16.0))
        - 24704.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 15.0)
        / (153153.0 * jnp.power(r, 16.0))
        - 219648.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 3.0)
        / (85.0 * jnp.power(r, 15.0))
        + 11074592.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 5.0)
        / (2431.0 * jnp.power(r, 15.0))
        - 1480173854.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 7.0)
        / (595595.0 * jnp.power(r, 15.0))
        + 183740283.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 9.0)
        / (340340.0 * jnp.power(r, 15.0))
        - 938679283.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 11.0)
        / (19059040.0 * jnp.power(r, 15.0))
        + 244740253.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 13.0)
        / (152472320.0 * jnp.power(r, 15.0))
        - 693.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 15.0)
        / (87040.0 * jnp.power(r, 15.0))
        - 8448.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 14.0))
        + 8929728.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 5.0)
        / (5005.0 * jnp.power(r, 14.0))
        - 7099640.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 7.0)
        / (9009.0 * jnp.power(r, 14.0))
        + 3280691.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 9.0)
        / (24570.0 * jnp.power(r, 14.0))
        - 1198264.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 11.0)
        / (135135.0 * jnp.power(r, 14.0))
        + 488.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 13.0)
        / (3003.0 * jnp.power(r, 14.0))
        - 50688.0 * jnp.power(M, 15.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 13.0))
        + 71072272.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 5.0)
        / (105105.0 * jnp.power(r, 13.0))
        - 10623001.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 7.0)
        / (45045.0 * jnp.power(r, 13.0))
        + 5873381.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 9.0)
        / (194040.0 * jnp.power(r, 13.0))
        - 27428833.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 11.0)
        / (20180160.0 * jnp.power(r, 13.0))
        + 63.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (6656.0 * jnp.power(r, 13.0))
        - 23040.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 12.0))
        + 1723584.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 5.0)
        / (7007.0 * jnp.power(r, 12.0))
        - 329328.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 7.0)
        / (5005.0 * jnp.power(r, 12.0))
        + 216144.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 9.0)
        / (35035.0 * jnp.power(r, 12.0))
        - 5664.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 11.0)
        / (35035.0 * jnp.power(r, 12.0))
        - 16128.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 3.0)
        / (143.0 * jnp.power(r, 11.0))
        + 1275212.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 5.0)
        / (15015.0 * jnp.power(r, 11.0))
        - 253378.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 7.0)
        / (15015.0 * jnp.power(r, 11.0))
        + 87641.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 9.0)
        / (80080.0 * jnp.power(r, 11.0))
        - 105.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 11.0)
        / (9152.0 * jnp.power(r, 11.0))
        - 2688.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (55.0 * jnp.power(r, 10.0))
        + 6332.0 * jnp.power(M, 12.0) * jnp.power(q, 5.0) / (231.0 * jnp.power(r, 10.0))
        - 9047.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 7.0)
        / (2310.0 * jnp.power(r, 10.0))
        + 181.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (1155.0 * jnp.power(r, 10.0))
        - 224.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (11.0 * jnp.power(r, 9.0))
        + 624.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (77.0 * jnp.power(r, 9.0))
        - 249.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (308.0 * jnp.power(r, 9.0))
        + 5.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (352.0 * jnp.power(r, 9.0))
        - 8.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / jnp.power(r, 8.0)
        + 15.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 8.0))
        - jnp.power(M, 10.0) * jnp.power(q, 7.0) / (7.0 * jnp.power(r, 8.0))
        - 20.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 7.0))
        + jnp.power(M, 9.0) * jnp.power(q, 5.0) / (2.0 * jnp.power(r, 7.0))
        - jnp.power(M, 9.0) * jnp.power(q, 7.0) / (56.0 * jnp.power(r, 7.0))
        - 6.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 6.0))
        + 3.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (28.0 * jnp.power(r, 6.0))
        - 6.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (35.0 * jnp.power(r, 5.0))
        + 3.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (140.0 * jnp.power(r, 5.0))
    )


def H0OnlyQS_jax(M, q, r):
    r"""
    Compute the H0 (only QS) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the H0 (only QS) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        373555200.0 * jnp.power(M, 19.0) * q / (253.0 * jnp.power(r, 22.0))
        - 7929632358400.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 3.0)
        / (1312311.0 * jnp.power(r, 22.0))
        + 3.681060935938816e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 5.0)
        / (4216455243.0 * jnp.power(r, 22.0))
        - 2.68366322093152e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 7.0)
        / (4660292637.0 * jnp.power(r, 22.0))
        + 1.2382500425031748e19
        * jnp.power(M, 19.0)
        * jnp.power(q, 9.0)
        / (6640917007725.0 * jnp.power(r, 22.0))
        - 5.880808994409967e18
        * jnp.power(M, 19.0)
        * jnp.power(q, 11.0)
        / (19922751023175.0 * jnp.power(r, 22.0))
        + 4.247249880139406e17
        * jnp.power(M, 19.0)
        * jnp.power(q, 13.0)
        / (19922751023175.0 * jnp.power(r, 22.0))
        - 1311359102092288.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 15.0)
        / (2213639002575.0 * jnp.power(r, 22.0))
        + 30408704.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 17.0)
        / (7436429.0 * jnp.power(r, 22.0))
        + 1270087680.0 * jnp.power(M, 18.0) * q / (1771.0 * jnp.power(r, 21.0))
        - 216993318260736.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 3.0)
        / (81800719.0 * jnp.power(r, 21.0))
        + 45830526088832.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 5.0)
        / (13483635.0 * jnp.power(r, 21.0))
        - 1699774817359024.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 7.0)
        / (876436275.0 * jnp.power(r, 21.0))
        + 919918368991283.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 9.0)
        / (1752872550.0 * jnp.power(r, 21.0))
        - 231217475648777.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 11.0)
        / (3505745100.0 * jnp.power(r, 21.0))
        + 194652894683933.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 13.0)
        / (56091921600.0 * jnp.power(r, 21.0))
        - 926258551895.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 15.0)
        / (15705738048.0 * jnp.power(r, 21.0))
        + 182325.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 17.0)
        / (1318912.0 * jnp.power(r, 21.0))
        + 26738688.0 * jnp.power(M, 17.0) * q / (77.0 * jnp.power(r, 20.0))
        - 86063947988992.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 3.0)
        / (74687613.0 * jnp.power(r, 20.0))
        + 1.8296328174075904e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 5.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        - 8901081866362880.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 7.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 1.77276099024896e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 9.0)
        / (127043629713.0 * jnp.power(r, 20.0))
        - 189629887700992.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 11.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 6789171617792.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 13.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        - 133431296.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 15.0)
        / (32008977.0 * jnp.power(r, 20.0))
        + 22282240.0 * jnp.power(M, 16.0) * q / (133.0 * jnp.power(r, 19.0))
        - 10072196525056.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 3.0)
        / (20369349.0 * jnp.power(r, 19.0))
        + 3095539652002624.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 5.0)
        / (6416344935.0 * jnp.power(r, 19.0))
        - 252355863522596.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 7.0)
        / (1283268987.0 * jnp.power(r, 19.0))
        + 363573767832881.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 9.0)
        / (10499473530.0 * jnp.power(r, 19.0))
        - 2248987038532789.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 11.0)
        / (923953670640.0 * jnp.power(r, 19.0))
        + 8772734045155.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 13.0)
        / (164258430336.0 * jnp.power(r, 19.0))
        - 10725.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 15.0)
        / (68096.0 * jnp.power(r, 19.0))
        + 4587520.0 * jnp.power(M, 15.0) * q / (57.0 * jnp.power(r, 18.0))
        - 67632636928.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 3.0)
        / (323323.0 * jnp.power(r, 18.0))
        + 1974936720256.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 5.0)
        / (11316305.0 * jnp.power(r, 18.0))
        - 1980360362576.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 7.0)
        / (33948915.0 * jnp.power(r, 18.0))
        + 1821878944.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 9.0)
        / (230945.0 * jnp.power(r, 18.0))
        - 4280066432.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 11.0)
        / (11316305.0 * jnp.power(r, 18.0))
        + 587776.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 13.0)
        / (138567.0 * jnp.power(r, 18.0))
        + 37273600.0 * jnp.power(M, 14.0) * q / (969.0 * jnp.power(r, 17.0))
        - 759568870400.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 3.0)
        / (8729721.0 * jnp.power(r, 17.0))
        + 3728453306272.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 5.0)
        / (61108047.0 * jnp.power(r, 17.0))
        - 6393714739124.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 7.0)
        / (392837445.0 * jnp.power(r, 17.0))
        + 1363525754411.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 9.0)
        / (846111420.0 * jnp.power(r, 17.0))
        - 232077111443.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 11.0)
        / (4888643760.0 * jnp.power(r, 17.0))
        + 15015.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 13.0)
        / (82688.0 * jnp.power(r, 17.0))
        + 931840.0 * jnp.power(M, 13.0) * q / (51.0 * jnp.power(r, 16.0))
        - 16296527872.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 3.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 329096302592.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 5.0)
        / (16081065.0 * jnp.power(r, 16.0))
        - 608718794752.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 7.0)
        / (144729585.0 * jnp.power(r, 16.0))
        + 41230764032.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 9.0)
        / (144729585.0 * jnp.power(r, 16.0))
        - 1976320.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 11.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 146432.0 * jnp.power(M, 12.0) * q / (17.0 * jnp.power(r, 15.0))
        - 102893504.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 3.0)
        / (7293.0 * jnp.power(r, 15.0))
        + 776708332.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 5.0)
        / (119119.0 * jnp.power(r, 15.0))
        - 117275122.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 7.0)
        / (119119.0 * jnp.power(r, 15.0))
        + 235028551.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 9.0)
        / (5717712.0 * jnp.power(r, 15.0))
        - 231.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 11.0)
        / (1088.0 * jnp.power(r, 15.0))
        + 28160.0 * jnp.power(M, 11.0) * q / (7.0 * jnp.power(r, 14.0))
        - 5449792.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 3.0)
        / (1001.0 * jnp.power(r, 14.0))
        + 52603352.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 5.0)
        / (27027.0 * jnp.power(r, 14.0))
        - 16361344.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 7.0)
        / (81081.0 * jnp.power(r, 14.0))
        + 39040.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 9.0)
        / (9009.0 * jnp.power(r, 14.0))
        + 168960.0 * jnp.power(M, 10.0) * q / (91.0 * jnp.power(r, 13.0))
        - 127508384.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 3.0)
        / (63063.0 * jnp.power(r, 13.0))
        + 100906370.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 5.0)
        / (189189.0 * jnp.power(r, 13.0))
        - 6475192.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 7.0)
        / (189189.0 * jnp.power(r, 13.0))
        + 105.0 * jnp.power(M, 10.0) * jnp.power(q, 9.0) / (416.0 * jnp.power(r, 13.0))
        + 76800.0 * jnp.power(M, 9.0) * q / (91.0 * jnp.power(r, 12.0))
        - 5006080.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 3.0)
        / (7007.0 * jnp.power(r, 12.0))
        + 911104.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 5.0)
        / (7007.0 * jnp.power(r, 12.0))
        - 30208.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 7.0)
        / (7007.0 * jnp.power(r, 12.0))
        + 53760.0 * jnp.power(M, 8.0) * q / (143.0 * jnp.power(r, 11.0))
        - 2127064.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 3.0)
        / (9009.0 * jnp.power(r, 11.0))
        + 80291.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 5.0)
        / (3003.0 * jnp.power(r, 11.0))
        - 175.0 * jnp.power(M, 8.0) * jnp.power(q, 7.0) / (572.0 * jnp.power(r, 11.0))
        + 1792.0 * jnp.power(M, 7.0) * q / (11.0 * jnp.power(r, 10.0))
        - 49208.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (693.0 * jnp.power(r, 10.0))
        + 2896.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (693.0 * jnp.power(r, 10.0))
        + 2240.0 * jnp.power(M, 6.0) * q / (33.0 * jnp.power(r, 9.0))
        - 4280.0 * jnp.power(M, 6.0) * jnp.power(q, 3.0) / (231.0 * jnp.power(r, 9.0))
        + 25.0 * jnp.power(M, 6.0) * jnp.power(q, 5.0) / (66.0 * jnp.power(r, 9.0))
        + 80.0 * jnp.power(M, 5.0) * q / (3.0 * jnp.power(r, 8.0))
        - 80.0 * jnp.power(M, 5.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 8.0))
        + 200.0 * jnp.power(M, 4.0) * q / (21.0 * jnp.power(r, 7.0))
        - 10.0 * jnp.power(M, 4.0) * jnp.power(q, 3.0) / (21.0 * jnp.power(r, 7.0))
        + 20.0 * jnp.power(M, 3.0) * q / (7.0 * jnp.power(r, 6.0))
        + 4.0 * jnp.power(M, 2.0) * q / (7.0 * jnp.power(r, 5.0))
    )


def PhiPOnlyQS_jax(M, q, r):
    r"""
    Compute the PhiP (only QS) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the PhiP (only QS) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        137625600.0 * jnp.power(M, 19.0) / (253.0 * jnp.power(r, 22.0))
        - 25044285849600.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 2.0)
        / (7436429.0 * jnp.power(r, 22.0))
        + 1.0386679483380224e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 4.0)
        / (1405485081.0 * jnp.power(r, 22.0))
        - 1.1083481260520846e18
        * jnp.power(M, 19.0)
        * jnp.power(q, 6.0)
        / (147575933505.0 * jnp.power(r, 22.0))
        + 4.276358632435258e19
        * jnp.power(M, 19.0)
        * jnp.power(q, 8.0)
        / (11068195012875.0 * jnp.power(r, 22.0))
        - 3.4122519275204542e19
        * jnp.power(M, 19.0)
        * jnp.power(q, 10.0)
        / (33204585038625.0 * jnp.power(r, 22.0))
        + 4.5715105084355814e18
        * jnp.power(M, 19.0)
        * jnp.power(q, 12.0)
        / (33204585038625.0 * jnp.power(r, 22.0))
        - 8.537907977396265e17
        * jnp.power(M, 19.0)
        * jnp.power(q, 14.0)
        / (99613755115875.0 * jnp.power(r, 22.0))
        + 459851552390656.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 16.0)
        / (2213639002575.0 * jnp.power(r, 22.0))
        - 65536.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 18.0)
        / (52003.0 * jnp.power(r, 22.0))
        + 498073600.0 * jnp.power(M, 18.0) / (1771.0 * jnp.power(r, 21.0))
        - 130349638000640.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 2.0)
        / (81800719.0 * jnp.power(r, 21.0))
        + 331514716669952.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 4.0)
        / (105172353.0 * jnp.power(r, 21.0))
        - 165519445666736.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 6.0)
        / (58429085.0 * jnp.power(r, 21.0))
        + 5507212513696126.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 8.0)
        / (4382181375.0 * jnp.power(r, 21.0))
        - 5.856179718704707e16
        * jnp.power(M, 18.0)
        * jnp.power(q, 10.0)
        / (210344706000.0 * jnp.power(r, 21.0))
        + 1378981216484473.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 12.0)
        / (46743268000.0 * jnp.power(r, 21.0))
        - 2.093033358474397e16
        * jnp.power(M, 18.0)
        * jnp.power(q, 14.0)
        / (15705738048000.0 * jnp.power(r, 21.0))
        + 14826685826357.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 16.0)
        / (753875426304.0 * jnp.power(r, 21.0))
        - 60775.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 18.0)
        / (1507328.0 * jnp.power(r, 21.0))
        + 11206656.0 * jnp.power(M, 17.0) / (77.0 * jnp.power(r, 20.0))
        - 18678269390848.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 2.0)
        / (24895871.0 * jnp.power(r, 20.0))
        + 6246198807072256.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 4.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        - 1.473649069109248e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 6.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 1.66641507094784e16
        * jnp.power(M, 17.0)
        * jnp.power(q, 8.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        - 9022651836532736.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 10.0)
        / (127043629713.0 * jnp.power(r, 20.0))
        + 243121395256832.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 12.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        - 2479639181312.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 14.0)
        / (14115958857.0 * jnp.power(r, 20.0))
        + 42106880.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 16.0)
        / (32008977.0 * jnp.power(r, 20.0))
        + 10027008.0 * jnp.power(M, 16.0) / (133.0 * jnp.power(r, 19.0))
        - 3974494584832.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 2.0)
        / (11316305.0 * jnp.power(r, 19.0))
        + 5891336352527072.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 4.0)
        / (10693908225.0 * jnp.power(r, 19.0))
        - 6.006610594366005e16
        * jnp.power(M, 16.0)
        * jnp.power(q, 6.0)
        / (160408623375.0 * jnp.power(r, 19.0))
        + 4.534986186885946e17
        * jnp.power(M, 16.0)
        * jnp.power(q, 8.0)
        / (3849806961000.0 * jnp.power(r, 19.0))
        - 3.8909336700647725e17
        * jnp.power(M, 16.0)
        * jnp.power(q, 10.0)
        / (23098841766000.0 * jnp.power(r, 19.0))
        + 3.6571209266836563e17
        * jnp.power(M, 16.0)
        * jnp.power(q, 12.0)
        / (369581468256000.0 * jnp.power(r, 19.0))
        - 24255807439991.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 14.0)
        / (1314067442688.0 * jnp.power(r, 19.0))
        + 10725.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 16.0)
        / (229376.0 * jnp.power(r, 19.0))
        + 2228224.0 * jnp.power(M, 15.0) / (57.0 * jnp.power(r, 18.0))
        - 264027403264.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 2.0)
        / (1616615.0 * jnp.power(r, 18.0))
        + 1816686703232.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 4.0)
        / (8083075.0 * jnp.power(r, 18.0))
        - 15774907358576.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 6.0)
        / (121246125.0 * jnp.power(r, 18.0))
        + 1350428153472.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 8.0)
        / (40415375.0 * jnp.power(r, 18.0))
        - 11352273008.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 10.0)
        / (3108875.0 * jnp.power(r, 18.0))
        + 17642708288.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 12.0)
        / (121246125.0 * jnp.power(r, 18.0))
        - 318464.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 14.0)
        / (230945.0 * jnp.power(r, 18.0))
        + 6553600.0 * jnp.power(M, 14.0) / (323.0 * jnp.power(r, 17.0))
        - 219264332288.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 2.0)
        / (2909907.0 * jnp.power(r, 17.0))
        + 27457227271648.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 4.0)
        / (305540235.0 * jnp.power(r, 17.0))
        - 199469520616378.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 6.0)
        / (4583103525.0 * jnp.power(r, 17.0))
        + 488099722330459.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 8.0)
        / (54997242300.0 * jnp.power(r, 17.0))
        - 77665594008593.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 10.0)
        / (109994484600.0 * jnp.power(r, 17.0))
        + 1343666476931.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 12.0)
        / (78218300160.0 * jnp.power(r, 17.0))
        - 2145.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 14.0)
        / (38912.0 * jnp.power(r, 17.0))
        + 179200.0 * jnp.power(M, 13.0) / (17.0 * jnp.power(r, 16.0))
        - 5274612736.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 2.0)
        / (153153.0 * jnp.power(r, 16.0))
        + 564114925568.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 4.0)
        / (16081065.0 * jnp.power(r, 16.0))
        - 479162181632.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 6.0)
        / (34459425.0 * jnp.power(r, 16.0))
        + 1571540384768.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 8.0)
        / (723647925.0 * jnp.power(r, 16.0))
        - 84964553728.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 10.0)
        / (723647925.0 * jnp.power(r, 16.0))
        + 667648.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 12.0)
        / (459459.0 * jnp.power(r, 16.0))
        + 93184.0 * jnp.power(M, 12.0) / (17.0 * jnp.power(r, 15.0))
        - 794581696.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 2.0)
        / (51051.0 * jnp.power(r, 15.0))
        + 4753519492.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 4.0)
        / (357357.0 * jnp.power(r, 15.0))
        - 142857509.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 6.0)
        / (34034.0 * jnp.power(r, 15.0))
        + 341843419.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 8.0)
        / (714714.0 * jnp.power(r, 15.0))
        - 723367669.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 10.0)
        / (45741696.0 * jnp.power(r, 15.0))
        + 1155.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 12.0)
        / (17408.0 * jnp.power(r, 15.0))
        + 19968.0 * jnp.power(M, 11.0) / (7.0 * jnp.power(r, 14.0))
        - 34722112.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 2.0)
        / (5005.0 * jnp.power(r, 14.0))
        + 1096169384.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 4.0)
        / (225225.0 * jnp.power(r, 14.0))
        - 24147776.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 6.0)
        / (20475.0 * jnp.power(r, 14.0))
        + 185505904.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 8.0)
        / (2027025.0 * jnp.power(r, 14.0))
        - 1984.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 10.0)
        / (1287.0 * jnp.power(r, 14.0))
        + 135168.0 * jnp.power(M, 10.0) / (91.0 * jnp.power(r, 13.0))
        - 106489536.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 2.0)
        / (35035.0 * jnp.power(r, 13.0))
        + 383469712.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 4.0)
        / (225225.0 * jnp.power(r, 13.0))
        - 5709620857.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 6.0)
        / (18918900.0 * jnp.power(r, 13.0))
        + 1085909833.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 8.0)
        / (75675600.0 * jnp.power(r, 13.0))
        - 21.0 * jnp.power(M, 10.0) * jnp.power(q, 10.0) / (256.0 * jnp.power(r, 13.0))
        + 70400.0 * jnp.power(M, 9.0) / (91.0 * jnp.power(r, 12.0))
        - 27389728.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 2.0)
        / (21021.0 * jnp.power(r, 12.0))
        + 2813376.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 4.0)
        / (5005.0 * jnp.power(r, 12.0))
        - 2382944.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 6.0)
        / (35035.0 * jnp.power(r, 12.0))
        + 34688.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 8.0)
        / (21021.0 * jnp.power(r, 12.0))
        + 57600.0 * jnp.power(M, 8.0) / (143.0 * jnp.power(r, 11.0))
        - 1630432.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 2.0)
        / (3003.0 * jnp.power(r, 11.0))
        + 15476827.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 4.0)
        / (90090.0 * jnp.power(r, 11.0))
        - 511029.0
        * jnp.power(M, 8.0)
        * jnp.power(q, 6.0)
        / (40040.0 * jnp.power(r, 11.0))
        + 175.0 * jnp.power(M, 8.0) * jnp.power(q, 8.0) / (1664.0 * jnp.power(r, 11.0))
        + 2304.0 * jnp.power(M, 7.0) / (11.0 * jnp.power(r, 10.0))
        - 50312.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (231.0 * jnp.power(r, 10.0))
        + 32672.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (693.0 * jnp.power(r, 10.0))
        - 1240.0 * jnp.power(M, 7.0) * jnp.power(q, 6.0) / (693.0 * jnp.power(r, 10.0))
        + 3584.0 * jnp.power(M, 6.0) / (33.0 * jnp.power(r, 9.0))
        - 95716.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (1155.0 * jnp.power(r, 9.0))
        + 25469.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (2310.0 * jnp.power(r, 9.0))
        - 25.0 * jnp.power(M, 6.0) * jnp.power(q, 6.0) / (176.0 * jnp.power(r, 9.0))
        + 56.0 * jnp.power(M, 5.0) / jnp.power(r, 8.0)
        - 3064.0 * jnp.power(M, 5.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 8.0))
        + 208.0 * jnp.power(M, 5.0) * jnp.power(q, 4.0) / (105.0 * jnp.power(r, 8.0))
        + 200.0 * jnp.power(M, 4.0) / (7.0 * jnp.power(r, 7.0))
        - 191.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (21.0 * jnp.power(r, 7.0))
        + 5.0 * jnp.power(M, 4.0) * jnp.power(q, 4.0) / (24.0 * jnp.power(r, 7.0))
        + 100.0 * jnp.power(M, 3.0) / (7.0 * jnp.power(r, 6.0))
        - 16.0 * jnp.power(M, 3.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0))
        + 48.0 * jnp.power(M, 2.0) / (7.0 * jnp.power(r, 5.0))
        - 5.0 * jnp.power(M, 2.0) * jnp.power(q, 2.0) / (14.0 * jnp.power(r, 5.0))
        + 3.0 * M / jnp.power(r, 4.0)
        + jnp.power(r, -3.0)
    )


def H0OnlyES_jax(M, q, r):
    r"""
    Compute the H0 (only ES) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the H0 (only ES) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        -4.0 * jnp.power(M, 2.0) * q / 3.0
        - 4194304.0 * jnp.power(M, 24.0) * q / (57.0 * jnp.power(r, 22.0))
        - 432189449371648.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 3.0)
        / (423876453.0 * jnp.power(r, 22.0))
        + 4.889210237699459e18
        * jnp.power(M, 24.0)
        * jnp.power(q, 5.0)
        / (1016814398985.0 * jnp.power(r, 22.0))
        - 9.663973074671625e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 7.0)
        / (1387951654614525.0 * jnp.power(r, 22.0))
        + 4.549082926346561e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 9.0)
        / (1.0409637409608938e17 * jnp.power(r, 22.0))
        - 1.56978117441629e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 11.0)
        / (1.2491564891530725e17 * jnp.power(r, 22.0))
        + 1.2677820661047214e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 13.0)
        / (8.8175752175511e16 * jnp.power(r, 22.0))
        + 8.765493299428505e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 15.0)
        / (1.4390282755043396e19 * jnp.power(r, 22.0))
        - 1.3175059756885878e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 17.0)
        / (1.0659468707439553e19 * jnp.power(r, 22.0))
        + 4.357109942093411e19
        * jnp.power(M, 24.0)
        * jnp.power(q, 19.0)
        / (5.8851449040384e17 * jnp.power(r, 22.0))
        - 3666737926669729.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 21.0)
        / (2946575951953920.0 * jnp.power(r, 22.0))
        + 1547.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 23.0)
        / (524288.0 * jnp.power(r, 22.0))
        - 1524105216.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 3.0)
        / (1771.0 * jnp.power(r, 21.0))
        + 1345957969453056.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 5.0)
        / (409003595.0 * jnp.power(r, 21.0))
        - 9154880691645184.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 7.0)
        / (2045017975.0 * jnp.power(r, 21.0))
        + 218120825455872.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 9.0)
        / (76880375.0 * jnp.power(r, 21.0))
        - 1344862073331039.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 11.0)
        / (1460727125.0 * jnp.power(r, 21.0))
        + 1844788271586391.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 13.0)
        / (11685817000.0 * jnp.power(r, 21.0))
        - 657087845981487.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 15.0)
        / (46743268000.0 * jnp.power(r, 21.0))
        + 1547821973166531.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 17.0)
        / (2617623008000.0 * jnp.power(r, 21.0))
        - 131287333403.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 19.0)
        / (14567641088.0 * jnp.power(r, 21.0))
        + 109395.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 21.0)
        / (5275648.0 * jnp.power(r, 21.0))
        - 160432128.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 3.0)
        / (385.0 * jnp.power(r, 20.0))
        + 178611820593152.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 5.0)
        / (124479355.0 * jnp.power(r, 20.0))
        - 8131835578126336.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 7.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        + 69276302712832.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 9.0)
        / (72837765.0 * jnp.power(r, 20.0))
        - 1.1096530800859136e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 11.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        + 461484732602368.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 13.0)
        / (12455257815.0 * jnp.power(r, 20.0))
        - 60985815160832.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 15.0)
        / (23526598095.0 * jnp.power(r, 20.0))
        + 362995861504.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 17.0)
        / (4705319619.0 * jnp.power(r, 20.0))
        - 33357824.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 19.0)
        / (53348295.0 * jnp.power(r, 20.0))
        - 26738688.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 3.0)
        / (133.0 * jnp.power(r, 19.0))
        + 20997541025792.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 5.0)
        / (33948915.0 * jnp.power(r, 19.0))
        - 6984264780353408.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 7.0)
        / (10693908225.0 * jnp.power(r, 19.0))
        + 122127538823208.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 9.0)
        / (396070675.0 * jnp.power(r, 19.0))
        - 526024223906992.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 11.0)
        / (7403474925.0 * jnp.power(r, 19.0))
        + 8221445374598.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 13.0)
        / (1013107095.0 * jnp.power(r, 19.0))
        - 660940017641191.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 15.0)
        / (1539922784400.0 * jnp.power(r, 19.0))
        + 1795939470341.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 17.0)
        / (219011240448.0 * jnp.power(r, 19.0))
        - 6435.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 19.0)
        / (272384.0 * jnp.power(r, 19.0))
        - 1835008.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 3.0)
        / (19.0 * jnp.power(r, 18.0))
        + 425312278528.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 5.0)
        / (1616615.0 * jnp.power(r, 18.0))
        - 55161850368.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 7.0)
        / (229075.0 * jnp.power(r, 18.0))
        + 5441923265344.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 9.0)
        / (56581525.0 * jnp.power(r, 18.0))
        - 206144500036.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 11.0)
        / (11316305.0 * jnp.power(r, 18.0))
        + 92634449784.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 13.0)
        / (56581525.0 * jnp.power(r, 18.0))
        - 3498060064.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 15.0)
        / (56581525.0 * jnp.power(r, 18.0))
        + 146944.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 17.0)
        / (230945.0 * jnp.power(r, 18.0))
        - 14909440.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 3.0)
        / (323.0 * jnp.power(r, 17.0))
        + 320617441280.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 5.0)
        / (2909907.0 * jnp.power(r, 17.0))
        - 462429059776.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 7.0)
        / (5360355.0 * jnp.power(r, 17.0))
        + 11950646003936.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 9.0)
        / (416645775.0 * jnp.power(r, 17.0))
        - 13367945464759.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 11.0)
        / (3055402350.0 * jnp.power(r, 17.0))
        + 21903222813317.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 13.0)
        / (73329656400.0 * jnp.power(r, 17.0))
        - 478357587061.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 15.0)
        / (65181916800.0 * jnp.power(r, 17.0))
        + 9009.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 17.0)
        / (330752.0 * jnp.power(r, 17.0))
        - 372736.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 3.0)
        / (17.0 * jnp.power(r, 16.0))
        + 34691792384.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 5.0)
        / (765765.0 * jnp.power(r, 16.0))
        - 266929074688.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 7.0)
        / (8933925.0 * jnp.power(r, 16.0))
        + 279700610048.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 9.0)
        / (34459425.0 * jnp.power(r, 16.0))
        - 234641226752.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 11.0)
        / (241215975.0 * jnp.power(r, 16.0))
        + 888674816.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 13.0)
        / (18555075.0 * jnp.power(r, 16.0))
        - 98816.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 15.0)
        / (153153.0 * jnp.power(r, 16.0))
        - 878592.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 3.0)
        / (85.0 * jnp.power(r, 15.0))
        + 44298368.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 5.0)
        / (2431.0 * jnp.power(r, 15.0))
        - 5920695416.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 7.0)
        / (595595.0 * jnp.power(r, 15.0))
        + 183740283.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 9.0)
        / (85085.0 * jnp.power(r, 15.0))
        - 938679283.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 11.0)
        / (4764760.0 * jnp.power(r, 15.0))
        + 244740253.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 13.0)
        / (38118080.0 * jnp.power(r, 15.0))
        - 693.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 15.0)
        / (21760.0 * jnp.power(r, 15.0))
        - 33792.0 * jnp.power(M, 16.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 14.0))
        + 35718912.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 5.0)
        / (5005.0 * jnp.power(r, 14.0))
        - 28398560.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 7.0)
        / (9009.0 * jnp.power(r, 14.0))
        + 6561382.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 9.0)
        / (12285.0 * jnp.power(r, 14.0))
        - 4793056.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 11.0)
        / (135135.0 * jnp.power(r, 14.0))
        + 1952.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 13.0)
        / (3003.0 * jnp.power(r, 14.0))
        - 202752.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 3.0)
        / (91.0 * jnp.power(r, 13.0))
        + 284289088.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 5.0)
        / (105105.0 * jnp.power(r, 13.0))
        - 42492004.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 7.0)
        / (45045.0 * jnp.power(r, 13.0))
        + 5873381.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 9.0)
        / (48510.0 * jnp.power(r, 13.0))
        - 27428833.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 11.0)
        / (5045040.0 * jnp.power(r, 13.0))
        + 63.0 * jnp.power(M, 15.0) * jnp.power(q, 13.0) / (1664.0 * jnp.power(r, 13.0))
        - 92160.0 * jnp.power(M, 14.0) * jnp.power(q, 3.0) / (91.0 * jnp.power(r, 12.0))
        + 6894336.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 5.0)
        / (7007.0 * jnp.power(r, 12.0))
        - 1317312.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 7.0)
        / (5005.0 * jnp.power(r, 12.0))
        + 864576.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 9.0)
        / (35035.0 * jnp.power(r, 12.0))
        - 22656.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 11.0)
        / (35035.0 * jnp.power(r, 12.0))
        - 64512.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 3.0)
        / (143.0 * jnp.power(r, 11.0))
        + 5100848.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 5.0)
        / (15015.0 * jnp.power(r, 11.0))
        - 1013512.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 7.0)
        / (15015.0 * jnp.power(r, 11.0))
        + 87641.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 9.0)
        / (20020.0 * jnp.power(r, 11.0))
        - 105.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 11.0)
        / (2288.0 * jnp.power(r, 11.0))
        - 10752.0 * jnp.power(M, 12.0) * jnp.power(q, 3.0) / (55.0 * jnp.power(r, 10.0))
        + 25328.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 5.0)
        / (231.0 * jnp.power(r, 10.0))
        - 18094.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 7.0)
        / (1155.0 * jnp.power(r, 10.0))
        + 724.0 * jnp.power(M, 12.0) * jnp.power(q, 9.0) / (1155.0 * jnp.power(r, 10.0))
        - 896.0 * jnp.power(M, 11.0) * jnp.power(q, 3.0) / (11.0 * jnp.power(r, 9.0))
        + 2496.0 * jnp.power(M, 11.0) * jnp.power(q, 5.0) / (77.0 * jnp.power(r, 9.0))
        - 249.0 * jnp.power(M, 11.0) * jnp.power(q, 7.0) / (77.0 * jnp.power(r, 9.0))
        + 5.0 * jnp.power(M, 11.0) * jnp.power(q, 9.0) / (88.0 * jnp.power(r, 9.0))
        - 32.0 * jnp.power(M, 10.0) * jnp.power(q, 3.0) / jnp.power(r, 8.0)
        + 60.0 * jnp.power(M, 10.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 8.0))
        - 4.0 * jnp.power(M, 10.0) * jnp.power(q, 7.0) / (7.0 * jnp.power(r, 8.0))
        - 80.0 * jnp.power(M, 9.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 7.0))
        + 2.0 * jnp.power(M, 9.0) * jnp.power(q, 5.0) / jnp.power(r, 7.0)
        - jnp.power(M, 9.0) * jnp.power(q, 7.0) / (14.0 * jnp.power(r, 7.0))
        - 24.0 * jnp.power(M, 8.0) * jnp.power(q, 3.0) / (7.0 * jnp.power(r, 6.0))
        + 3.0 * jnp.power(M, 8.0) * jnp.power(q, 5.0) / (7.0 * jnp.power(r, 6.0))
        - 24.0 * jnp.power(M, 7.0) * jnp.power(q, 3.0) / (35.0 * jnp.power(r, 5.0))
        + 3.0 * jnp.power(M, 7.0) * jnp.power(q, 5.0) / (35.0 * jnp.power(r, 5.0))
    )


def PhiPOnlyES_jax(M, q, r):
    r"""
    Compute the PhiP (only ES) basis function for exterior solution.

    Parameters
    ----------
    M : jax.numpy.ndarray
        Gravitational mass (geometric units).
    q : jax.numpy.ndarray
        Scalar charge.
    r : jax.numpy.ndarray
        Radial coordinate (distance from center).

    Returns
    -------
    jax.numpy.ndarray
        Value of the PhiP (only ES) basis function.

    Notes
    -----
    Series expansion from Creci et al. (2023) for scalar-tensor tidal perturbations.
    Used to construct exterior basis for matching boundary conditions.
    """
    return (
        2.0 * jnp.power(M, 2.0) / 3.0
        - 8041005056.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 2.0)
        / (14421.0 * jnp.power(r, 22.0))
        + 1385977507643392.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 4.0)
        / (423876453.0 * jnp.power(r, 22.0))
        - 9.989620396474737e18
        * jnp.power(M, 24.0)
        * jnp.power(q, 6.0)
        / (1468731909645.0 * jnp.power(r, 22.0))
        + 9.225803659557211e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 8.0)
        / (1387951654614525.0 * jnp.power(r, 22.0))
        - 1.6043003271337948e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 10.0)
        / (462650551538175.0 * jnp.power(r, 22.0))
        + 6.755117363294697e23
        * jnp.power(M, 24.0)
        * jnp.power(q, 12.0)
        / (6.245782445765362e17 * jnp.power(r, 22.0))
        - 2.710363445407059e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 14.0)
        / (1.18967284681245e17 * jnp.power(r, 22.0))
        + 1.1950718113172306e25
        * jnp.power(M, 24.0)
        * jnp.power(q, 16.0)
        / (3.597570688760849e20 * jnp.power(r, 22.0))
        - 5.054567548874643e21
        * jnp.power(M, 24.0)
        * jnp.power(q, 18.0)
        / (1.75705528144608e18 * jnp.power(r, 22.0))
        + 3.3056744555097832e22
        * jnp.power(M, 24.0)
        * jnp.power(q, 20.0)
        / (2.842524988650547e20 * jnp.power(r, 22.0))
        - 3.243763865234007e16
        * jnp.power(M, 24.0)
        * jnp.power(q, 22.0)
        / (2.062603166367744e16 * jnp.power(r, 22.0))
        + 1547.0
        * jnp.power(M, 24.0)
        * jnp.power(q, 24.0)
        / (524288.0 * jnp.power(r, 22.0))
        - 443547648.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 2.0)
        / (1771.0 * jnp.power(r, 21.0))
        + 7430031507456.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 4.0)
        / (6292363.0 * jnp.power(r, 21.0))
        - 695487629952512.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 6.0)
        / (409003595.0 * jnp.power(r, 21.0))
        + 15860600113536.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 8.0)
        / (22472725.0 * jnp.power(r, 21.0))
        + 1854140157001254.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 10.0)
        / (7303635625.0 * jnp.power(r, 21.0))
        - 607478734116071.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 12.0)
        / (2247272500.0 * jnp.power(r, 21.0))
        + 1.6931295578153756e16
        * jnp.power(M, 23.0)
        * jnp.power(q, 14.0)
        / (233716340000.0 * jnp.power(r, 21.0))
        - 426893055188763.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 16.0)
        / (54083120000.0 * jnp.power(r, 21.0))
        + 3.578520927133202e16
        * jnp.power(M, 23.0)
        * jnp.power(q, 18.0)
        / (104704920320000.0 * jnp.power(r, 21.0))
        - 6233510141877.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 20.0)
        / (1340222980096.0 * jnp.power(r, 21.0))
        + 25857.0
        * jnp.power(M, 23.0)
        * jnp.power(q, 22.0)
        / (3014656.0 * jnp.power(r, 21.0))
        - 149291008.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 2.0)
        / (1155.0 * jnp.power(r, 20.0))
        + 614110969716736.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 4.0)
        / (1120314195.0 * jnp.power(r, 20.0))
        - 2.883604164704051e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 6.0)
        / (42347876571.0 * jnp.power(r, 20.0))
        + 346354717217408.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 8.0)
        / (1749912255.0 * jnp.power(r, 20.0))
        + 7143052099063808.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 10.0)
        / (54447269877.0 * jnp.power(r, 20.0))
        - 1.7261848214515635e17
        * jnp.power(M, 22.0)
        * jnp.power(q, 12.0)
        / (1905654445695.0 * jnp.power(r, 20.0))
        + 2.463306532016589e16
        * jnp.power(M, 22.0)
        * jnp.power(q, 14.0)
        / (1319299231635.0 * jnp.power(r, 20.0))
        - 574469521314688.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 16.0)
        / (381130889139.0 * jnp.power(r, 20.0))
        + 833697187328.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 18.0)
        / (19249034805.0 * jnp.power(r, 20.0))
        - 3137536.0
        * jnp.power(M, 22.0)
        * jnp.power(q, 20.0)
        / (10669659.0 * jnp.power(r, 20.0))
        - 133169152.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 2.0)
        / (1995.0 * jnp.power(r, 19.0))
        + 128568127768576.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 4.0)
        / (509233725.0 * jnp.power(r, 19.0))
        - 1022368992317056.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 6.0)
        / (3849806961.0 * jnp.power(r, 19.0))
        + 1.0601170091752902e17
        * jnp.power(M, 21.0)
        * jnp.power(q, 8.0)
        / (2406129350625.0 * jnp.power(r, 19.0))
        + 1.2755055261297347e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 10.0)
        / (21655164155625.0 * jnp.power(r, 19.0))
        - 2.5958300267679514e17
        * jnp.power(M, 21.0)
        * jnp.power(q, 12.0)
        / (9117963855000.0 * jnp.power(r, 19.0))
        + 1.6227134070813253e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 14.0)
        / (366863957460000.0 * jnp.power(r, 19.0))
        - 2.797703590745652e18
        * jnp.power(M, 21.0)
        * jnp.power(q, 16.0)
        / (1.108744404768e16 * jnp.power(r, 19.0))
        + 341356715454037.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 18.0)
        / (78844046561280.0 * jnp.power(r, 19.0))
        - 6721.0
        * jnp.power(M, 21.0)
        * jnp.power(q, 20.0)
        / (688128.0 * jnp.power(r, 19.0))
        - 655360.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 2.0)
        / (19.0 * jnp.power(r, 18.0))
        + 49009399808.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 4.0)
        / (425425.0 * jnp.power(r, 18.0))
        - 28330659115776.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 6.0)
        / (282907625.0 * jnp.power(r, 18.0))
        + 5361605413888.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 8.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        + 33909814164736.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 10.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        - 11786514354528.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 12.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        + 1338458164964.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 14.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        - 50397087952.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 16.0)
        / (1414538125.0 * jnp.power(r, 18.0))
        + 350976.0
        * jnp.power(M, 20.0)
        * jnp.power(q, 18.0)
        / (1154725.0 * jnp.power(r, 18.0))
        - 51838976.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 2.0)
        / (2907.0 * jnp.power(r, 17.0))
        + 6805873168384.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 4.0)
        / (130945815.0 * jnp.power(r, 17.0))
        - 55166563457312.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 6.0)
        / (1527701175.0 * jnp.power(r, 17.0))
        - 738796077908144.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 8.0)
        / (206239658625.0 * jnp.power(r, 17.0))
        + 1.4811340694832936e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 10.0)
        / (1649917269000.0 * jnp.power(r, 17.0))
        - 6.6700843335613576e16
        * jnp.power(M, 19.0)
        * jnp.power(q, 12.0)
        / (29698510842000.0 * jnp.power(r, 17.0))
        + 1344096953129401.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 14.0)
        / (7542478944000.0 * jnp.power(r, 17.0))
        - 18704857702771.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 16.0)
        / (4693098009600.0 * jnp.power(r, 17.0))
        + 7007.0
        * jnp.power(M, 19.0)
        * jnp.power(q, 18.0)
        / (622592.0 * jnp.power(r, 17.0))
        - 1411072.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 2.0)
        / (153.0 * jnp.power(r, 16.0))
        + 159389998336.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 4.0)
        / (6891885.0 * jnp.power(r, 16.0))
        - 9361528576.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 6.0)
        / (765765.0 * jnp.power(r, 16.0))
        - 34191514009088.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 8.0)
        / (10854718875.0 * jnp.power(r, 16.0))
        + 437170363904.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 10.0)
        / (140970375.0 * jnp.power(r, 16.0))
        - 4870862105344.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 12.0)
        / (8881133625.0 * jnp.power(r, 16.0))
        + 308710599424.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 14.0)
        / (10854718875.0 * jnp.power(r, 16.0))
        - 2167808.0
        * jnp.power(M, 18.0)
        * jnp.power(q, 16.0)
        / (6891885.0 * jnp.power(r, 16.0))
        - 405504.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 2.0)
        / (85.0 * jnp.power(r, 15.0))
        + 24589248.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 4.0)
        / (2431.0 * jnp.power(r, 15.0))
        - 2234310352.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 6.0)
        / (595595.0 * jnp.power(r, 15.0))
        - 4198013377.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 8.0)
        / (2382380.0 * jnp.power(r, 15.0))
        + 425842269.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 10.0)
        / (433160.0 * jnp.power(r, 15.0))
        - 184250867.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 12.0)
        / (1555840.0 * jnp.power(r, 15.0))
        + 1102175551.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 14.0)
        / (304944640.0 * jnp.power(r, 15.0))
        - 3663.0
        * jnp.power(M, 17.0)
        * jnp.power(q, 16.0)
        / (278528.0 * jnp.power(r, 15.0))
        - 259072.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 2.0)
        / (105.0 * jnp.power(r, 14.0))
        + 2923899008.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 4.0)
        / (675675.0 * jnp.power(r, 14.0))
        - 1943073856.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 6.0)
        / (2027025.0 * jnp.power(r, 14.0))
        - 4953879404.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 8.0)
        / (6081075.0 * jnp.power(r, 14.0))
        + 5152567744.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 10.0)
        / (18243225.0 * jnp.power(r, 14.0))
        - 44152796.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 12.0)
        / (2027025.0 * jnp.power(r, 14.0))
        + 2096.0
        * jnp.power(M, 16.0)
        * jnp.power(q, 14.0)
        / (6435.0 * jnp.power(r, 14.0))
        - 348160.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 2.0)
        / (273.0 * jnp.power(r, 13.0))
        + 8505963776.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 4.0)
        / (4729725.0 * jnp.power(r, 13.0))
        - 132451952.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 6.0)
        / (921375.0 * jnp.power(r, 13.0))
        - 5406426616.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 8.0)
        / (16372125.0 * jnp.power(r, 13.0))
        + 183798767107.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 10.0)
        / (2554051500.0 * jnp.power(r, 13.0))
        - 520303969.0
        * jnp.power(M, 15.0)
        * jnp.power(q, 12.0)
        / (162162000.0 * jnp.power(r, 13.0))
        + jnp.power(M, 15.0) * jnp.power(q, 14.0) / (64.0 * jnp.power(r, 13.0))
        - 4608.0 * jnp.power(M, 14.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 12.0))
        + 1935744.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 4.0)
        / (2695.0 * jnp.power(r, 12.0))
        + 6987672.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 6.0)
        / (175175.0 * jnp.power(r, 12.0))
        - 20830608.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 8.0)
        / (175175.0 * jnp.power(r, 12.0))
        + 2749368.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 10.0)
        / (175175.0 * jnp.power(r, 12.0))
        - 11808.0
        * jnp.power(M, 14.0)
        * jnp.power(q, 12.0)
        / (35035.0 * jnp.power(r, 12.0))
        - 145408.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 2.0)
        / (429.0 * jnp.power(r, 11.0))
        + 36596752.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 4.0)
        / (135135.0 * jnp.power(r, 11.0))
        + 6926398.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 6.0)
        / (135135.0 * jnp.power(r, 11.0))
        - 18342487.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 8.0)
        / (486486.0 * jnp.power(r, 11.0))
        + 1703357.0
        * jnp.power(M, 13.0)
        * jnp.power(q, 10.0)
        / (617760.0 * jnp.power(r, 11.0))
        - 63.0 * jnp.power(M, 13.0) * jnp.power(q, 12.0) / (3328.0 * jnp.power(r, 11.0))
        - 28672.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 2.0)
        / (165.0 * jnp.power(r, 10.0))
        + 193808.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 4.0)
        / (2079.0 * jnp.power(r, 10.0))
        + 323198.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 6.0)
        / (10395.0 * jnp.power(r, 10.0))
        - 957112.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 8.0)
        / (93555.0 * jnp.power(r, 10.0))
        + 722.0
        * jnp.power(M, 12.0)
        * jnp.power(q, 10.0)
        / (2079.0 * jnp.power(r, 10.0))
        - 4864.0 * jnp.power(M, 11.0) * jnp.power(q, 2.0) / (55.0 * jnp.power(r, 9.0))
        + 52224.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 4.0)
        / (1925.0 * jnp.power(r, 9.0))
        + 55563.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 6.0)
        / (3850.0 * jnp.power(r, 9.0))
        - 34537.0
        * jnp.power(M, 11.0)
        * jnp.power(q, 8.0)
        / (15400.0 * jnp.power(r, 9.0))
        + 3.0 * jnp.power(M, 11.0) * jnp.power(q, 10.0) / (128.0 * jnp.power(r, 9.0))
        - 400.0 * jnp.power(M, 10.0) * jnp.power(q, 2.0) / (9.0 * jnp.power(r, 8.0))
        + 1138.0 * jnp.power(M, 10.0) * jnp.power(q, 4.0) / (225.0 * jnp.power(r, 8.0))
        + 77314.0
        * jnp.power(M, 10.0)
        * jnp.power(q, 6.0)
        / (14175.0 * jnp.power(r, 8.0))
        - 556.0 * jnp.power(M, 10.0) * jnp.power(q, 8.0) / (1575.0 * jnp.power(r, 8.0))
        - 1376.0 * jnp.power(M, 9.0) * jnp.power(q, 2.0) / (63.0 * jnp.power(r, 7.0))
        - 37.0 * jnp.power(M, 9.0) * jnp.power(q, 4.0) / (45.0 * jnp.power(r, 7.0))
        + 18493.0
        * jnp.power(M, 9.0)
        * jnp.power(q, 6.0)
        / (11340.0 * jnp.power(r, 7.0))
        - 17.0 * jnp.power(M, 9.0) * jnp.power(q, 8.0) / (576.0 * jnp.power(r, 7.0))
        - 72.0 * jnp.power(M, 8.0) * jnp.power(q, 2.0) / (7.0 * jnp.power(r, 6.0))
        - 51.0 * jnp.power(M, 8.0) * jnp.power(q, 4.0) / (35.0 * jnp.power(r, 6.0))
        + 12.0 * jnp.power(M, 8.0) * jnp.power(q, 6.0) / (35.0 * jnp.power(r, 6.0))
        - 464.0 * jnp.power(M, 7.0) * jnp.power(q, 2.0) / (105.0 * jnp.power(r, 5.0))
        - 38.0 * jnp.power(M, 7.0) * jnp.power(q, 4.0) / (45.0 * jnp.power(r, 5.0))
        + jnp.power(M, 7.0) * jnp.power(q, 6.0) / (28.0 * jnp.power(r, 5.0))
        - 22.0 * jnp.power(M, 6.0) * jnp.power(q, 2.0) / (15.0 * jnp.power(r, 4.0))
        - 47.0 * jnp.power(M, 6.0) * jnp.power(q, 4.0) / (180.0 * jnp.power(r, 4.0))
        + 2.0 * jnp.power(M, 4.0) * jnp.power(q, 2.0) / (3.0 * jnp.power(r, 2.0))
        + jnp.power(M, 3.0) * jnp.power(q, 2.0) / (3.0 * r)
        - 2.0 * M * r
        + jnp.power(r, 2.0)
    )


def predict_phi0_last(beta_st, log_pc, log_ps_nsat):
    """
    Ultra-fast JAX deeper surrogate NN to predict Phi0_Last.
    Trained with asymmetric weighted loss to preserve strong-field precision.
    """
    mean_X = jnp.array([ -4.42618296, -10.63824663, -14.63619966, -14.43587915, -14.22044974,
 -13.9987255 , -13.77216608, -13.54816404, -13.32503666, -13.10395111,
 -12.87803343, -12.63535503, -12.35816543, -11.90796452, -11.39730826,
 -10.91834557, -10.44242476, -10.01819168,  -9.65440263,  -9.44942628,
  -9.40458713,  -9.36166979])
    scale_X = jnp.array([2.63167192, 0.79867976, 0.00395935, 0.00259443, 0.00260551, 0.00319084,
 0.00256253, 0.00274802, 0.00313575, 0.00199759, 0.0047761 , 0.00626588,
 0.01704002, 0.12846375, 0.25013029, 0.32437632, 0.4232025 , 0.52674144,
 0.57944155, 0.58293415, 0.53192898, 0.49180826])

    W1 = jnp.array([[-1.1386395 ,  0.4569696 ,  0.3350527 ,  0.9766735 , -0.1293833 ,
   0.65234524,  0.18947078, -1.197769  , -1.1301382 ,  0.55866295,
   2.5874267 ,  2.4117153 ,  1.6768534 ,  3.721848  , -0.2298734 ,
   0.01762259,  9.832739  ,  1.2524139 ,  0.45459193, -0.18989123,
  -0.0893924 ,  1.1917748 ,  1.3245072 ,  3.1921606 ,  0.8893822 ,
  -0.60999715,  0.6879903 , -0.5511441 ,  0.3359553 , -0.00341565,
   1.4512548 ,  0.994621  ,  0.482206  ,  2.3645675 ,  2.9231834 ,
   1.2477752 , -0.13841374, -0.1981384 , -0.16026764,  2.8454108 ,
   1.4944698 , -0.34909654,  0.5062528 , -0.06381705,  0.24525245,
  -1.0277663 ,  0.03104181,  1.6123308 , -1.2140692 ,  0.7150779 ,
   1.5223364 ,  1.0669206 ,  1.2873937 ,  0.6703071 ,  0.17361632,
   2.1119242 ,  0.35706216,  0.6842505 , -0.2694804 ,  0.12932323,
   0.18823674, -1.0784993 ,  0.7649975 ,  0.9786644 , -0.52941227,
   1.4248862 ,  1.055422  ,  1.7580054 ,  0.21963803,  1.4737152 ,
   1.6343782 ,  0.00648889,  1.0163519 ,  1.2314306 , -0.991847  ,
   0.0483778 , -0.62003577,  0.25507468,  0.5036708 , -0.52042776,
  -0.2680015 ,  0.20698284,  0.7981427 ,  0.2784445 ,  0.12011344,
   2.058131  ,  1.8289093 ,  5.285212  ,  1.752274  ,  0.2620105 ,
   0.58602667,  1.0303255 ,  1.9776112 , -0.3427927 ,  0.70568985,
  -1.0296139 ,  1.4590194 ,  0.82603824, -1.2521338 ,  1.3368864 ,
   0.22514129,  1.5517735 ,  0.9127522 ,  1.297568  ,  0.08865566,
   3.9044697 , -0.19699167,  1.8003839 ,  3.5541725 ,  0.5201241 ,
  -0.21960695,  0.14338852,  3.232612  , -0.12968129,  0.17061771,
   1.6301558 ,  0.19743825, -0.34863716,  0.9065168 ,  1.8940185 ,
   0.8386743 ,  0.9006205 ,  0.481351  ,  0.81777817,  1.8482888 ,
  -0.00482969, -0.5362099 ,  1.2572191 ],
 [ 0.6934552 , -0.7829678 , -0.5360227 , -0.9696803 ,  1.6746113 ,
  -2.1120842 ,  0.07091888,  1.5784919 ,  1.9225813 ,  0.13266172,
  -0.353529  , -0.9212145 ,  0.04132276,  2.0125363 , -1.4110262 ,
  -1.0773823 ,  1.5079232 , -0.31105027, -1.7663952 , -1.4386299 ,
   0.760382  ,  0.12289087,  0.24676651, -0.7685786 , -1.3754227 ,
   0.3766113 , -0.5293132 ,  1.1426309 , -0.25077984, -0.8298053 ,
  -1.4694967 , -0.03045086, -0.48060507, -0.9837242 ,  0.23459707,
  -1.4152824 , -0.6960037 ,  1.357435  ,  0.10977649,  0.4311444 ,
  -0.09462247,  1.0075254 , -1.0636873 , -1.3970982 ,  0.03087936,
  -1.7371981 , -0.50534827, -0.5107413 , -1.2340618 , -0.04881754,
  -1.1887456 , -1.4314127 , -1.3551319 , -0.2973968 , -0.7183229 ,
  -1.7034549 , -0.74251145, -1.0367374 , -0.55732185, -0.47297078,
  -0.7145135 ,  1.2386317 ,  0.10384753, -0.31705678, -0.87896454,
  -1.5708201 ,  0.5734229 , -0.35398373,  0.05426713, -1.1520256 ,
  -0.16346297, -0.8369786 , -0.02871419, -1.291973  ,  1.6638261 ,
   1.6042297 , -0.49237493, -0.7758489 , -0.5441482 ,  1.1787426 ,
   0.3389586 , -0.3231176 , -0.33073026,  0.19343866, -1.7600816 ,
  -0.8477071 , -0.571412  ,  1.1653477 ,  0.4129487 , -0.08402974,
  -0.616701  , -1.702946  , -1.0809158 , -2.5932279 , -0.31150237,
   0.40043688, -0.6399262 , -1.9012922 ,  2.0574913 ,  0.8240793 ,
  -0.33070216, -0.7400175 , -0.15151888, -1.2457433 ,  0.9196509 ,
  -1.07003   , -0.28073496, -2.109752  , -0.536843  , -2.005476  ,
  -0.92013556, -1.2481471 ,  1.1159134 , -2.055637  , -1.6590466 ,
  -0.07111073,  0.4771241 ,  1.0178775 , -0.6117003 , -1.23331   ,
  -1.4202092 , -0.55258304,  0.9011838 , -0.75462353, -0.8740935 ,
  -0.33761472,  1.5092967 , -0.5589899 ],
 [-0.00269996, -1.3339895 , -0.7158871 , -0.5329447 , -0.12129378,
   0.34206194, -0.5815846 ,  0.12308454,  0.117212  ,  0.01171502,
  -0.89927983, -0.08116415,  0.2805485 , -0.21627031, -0.16755088,
  -0.34262243, -0.08687045, -0.6129106 , -0.14571996, -0.00498511,
  -0.12373693, -0.7695563 , -0.18169257, -0.40017015, -0.8004107 ,
   0.20933868, -0.39539877,  0.01674055, -1.5691168 ,  0.7238879 ,
   0.03729643,  0.2885559 , -0.44043747, -0.34745535,  0.81399375,
   1.340641  ,  0.21606235,  0.10848286, -1.0800941 , -0.16587122,
   0.23359759, -0.6707936 , -0.10693025,  0.49945733,  0.90244186,
   0.21715717, -0.16462034, -0.5033661 , -0.00592235,  0.16434921,
   0.04379985,  0.00957467, -0.09647318, -0.86246115,  1.0239352 ,
  -0.9695324 , -0.06809712,  0.8868766 , -0.06014645, -0.1505063 ,
  -0.17622371,  0.6358449 , -0.73580885, -0.23108594, -0.29020506,
  -1.2345188 , -0.55281764, -0.01594846,  0.08355381, -0.09697629,
  -0.05558243,  0.31198803,  1.3294741 , -0.48280412,  0.14033672,
  -0.1557464 ,  0.2462957 ,  0.03807751,  0.4189767 , -0.03929263,
   0.58202994, -0.23782422, -0.02989089, -1.4308202 ,  0.88756496,
   0.4240181 , -0.1900193 ,  0.10491569, -0.45006657, -0.28171298,
  -0.17964068, -0.10952899, -0.29420862, -0.06799375, -0.10447203,
   0.0566837 , -0.2548005 , -0.22520745,  0.11710317,  0.25683212,
  -0.55287826, -0.6089965 , -0.18644366, -1.2839135 ,  0.18796279,
  -0.05116891,  0.7020643 ,  0.38959822, -0.26050472, -2.4385273 ,
  -0.31114656, -0.93591666, -0.30099055,  0.06222507, -0.29853997,
  -0.8830373 , -0.8271657 ,  0.31813547,  0.52103925, -0.5841128 ,
  -0.41772762, -0.42142332,  0.0065315 ,  0.49510628, -0.32641718,
   0.32241777, -0.11946298, -0.8056197 ],
 [-0.08400376, -0.66927063, -0.6783971 , -0.4252228 ,  0.00459236,
   0.22020991, -0.48982206, -0.12749536,  0.00149188,  0.36074498,
  -0.28545117,  0.12123685, -0.13446765,  0.13997777, -0.18530233,
  -0.39012378,  0.23577164, -0.31750014, -0.03359651, -0.27787519,
   0.02647898, -0.7712668 ,  0.13469109, -0.39631948, -0.36668593,
   0.05587393, -0.4061252 , -0.3113718 , -1.4344212 ,  0.5962595 ,
   0.05471693,  0.09316307, -0.6009622 , -0.324515  ,  0.13884518,
   0.751783  , -0.05422296, -0.2586233 , -0.7682591 , -0.15801406,
   0.17613055, -0.76307803,  0.0682494 , -0.05022748,  0.6734789 ,
  -0.12204211, -0.33969197, -0.33488098,  0.01847313,  0.19526488,
   0.18464154,  0.21189539,  0.30027798, -0.8346686 ,  0.4339392 ,
  -0.64979553,  0.01435485,  0.30144954, -0.15562811, -0.20599627,
   0.03043354, -0.6180331 , -0.5819917 , -0.20433001, -0.3773118 ,
  -0.39278004, -0.7936569 ,  0.15920307, -0.02438264,  0.00496798,
   0.08427316,  0.2802548 ,  0.36509347, -0.09093705, -0.2266746 ,
  -0.13619581, -0.07855091, -0.14824559,  0.06321634, -0.37618938,
   0.59073687, -0.3545895 , -0.10431996, -0.99109685,  0.66716576,
   0.310854  , -0.05987335, -0.22560044, -0.23338108, -0.31797636,
   0.30227715,  0.01392112,  0.24851549,  0.0965201 , -0.18942961,
   0.02195039, -0.06799722, -0.09882651,  0.01370097,  0.1146797 ,
  -0.68266994, -0.31686616,  0.09407277, -0.29240078, -0.02730257,
   0.21402952,  0.18415813,  0.2248179 ,  0.02299972, -2.193164  ,
  -0.40841866, -0.589317  ,  0.1829    ,  0.02655775, -0.22973357,
  -0.38943532, -0.6046811 ,  0.1806107 ,  0.22093946, -0.5907048 ,
  -0.09802694, -0.29453018,  0.02732726,  0.19015682, -0.20833586,
   0.0742234 , -0.18889894, -0.6211211 ],
 [ 0.04403172, -0.5669145 , -0.0236721 , -0.06979612,  0.06169877,
  -0.02311367,  0.18986122, -0.07135583,  0.06350613,  0.44233555,
   0.1682461 , -0.05079539,  0.03711972, -0.01730146, -0.02154552,
   0.4202809 , -0.10605452, -0.18558216, -0.09517179, -0.06982784,
   0.15995102,  0.08438506, -0.2077277 , -0.19260648, -0.01440576,
   0.0857326 , -0.13803889,  0.04962527, -0.32292774,  0.00993406,
  -0.17574531,  0.41950122, -0.54000354,  0.1949052 , -0.6187606 ,
  -0.269581  , -0.2513741 ,  0.11477959, -0.37646   ,  0.05588785,
   0.16961932, -0.268182  ,  0.07496922, -0.03162389,  0.00631318,
  -0.05878981, -0.25565895,  0.12853453, -0.05124135,  0.47341537,
  -0.281708  ,  0.09935492, -0.16359161, -0.28534842, -0.2435434 ,
  -0.07402688,  0.06385408, -0.27214903,  0.2696309 ,  0.84080803,
  -0.08145429, -0.16032363, -0.13416089,  0.0595274 ,  0.2684716 ,
   0.0909636 , -0.3679352 ,  0.44856283, -0.05104308, -0.29559872,
   0.21526444,  0.30057123,  0.20769282, -0.09241968, -0.02449859,
   0.41617706,  0.3494731 , -0.6203029 ,  0.33762822,  0.47513548,
   0.28518838,  0.15437321, -0.03268029,  0.7500401 , -0.07950789,
  -0.12567197, -0.10374402, -0.09778472,  0.00368281, -0.20825504,
   0.6965181 , -0.02487426, -0.06245562, -0.03404459,  0.22182785,
  -0.02541719,  0.20616725,  0.16237056, -0.00551517, -0.19118795,
  -0.12742065,  0.23497216, -0.05795413, -0.3401854 ,  0.19195953,
   0.02414175,  0.33400023,  0.21951409,  0.5325568 ,  0.64517707,
   0.27189165,  0.28057495, -0.18216887, -0.08062307, -0.02769752,
   0.44499594, -0.2867744 ,  0.0963627 ,  0.40870044, -0.09824928,
  -0.1480903 ,  0.19398125,  0.18200852, -0.14691424,  0.226063  ,
  -0.13671812,  0.18002345,  0.02210461],
 [-0.05409513, -0.05172573, -0.34049794,  0.2085832 , -0.1109184 ,
  -0.10637511, -0.24101146,  0.07270015, -0.09699985,  0.07214174,
  -0.04901825,  0.3781062 , -0.017925  , -0.18498625,  0.19040045,
  -0.03333414,  0.11279763, -0.40869483,  0.15837657,  0.10148265,
  -0.09822459, -0.81100094,  0.04916604, -0.13571787, -0.32095513,
  -0.25209966, -0.1452523 , -0.09326791,  0.9022498 , -0.26114693,
  -0.19862325, -0.10346551, -0.18239927, -0.1768155 ,  0.2888248 ,
  -0.36771286, -0.21588513,  0.10386528, -0.22471485, -0.40929434,
   0.23739406, -0.20556374, -0.1885027 ,  0.09136353, -0.03208803,
   0.05460311, -0.5804474 , -0.28274474,  0.13019413,  0.30332005,
   0.20959519, -0.6109112 ,  0.05888527, -0.73294425, -0.48899606,
  -0.29189026,  0.04415788, -0.6446263 , -0.43039948,  0.08048798,
   0.01168624, -0.46829304, -0.49447733,  0.01398211, -0.1275332 ,
   0.22467534, -0.4861581 , -0.00179537,  0.07040796,  0.4497858 ,
  -0.28478444, -0.26015094, -0.242646  ,  0.03639962, -0.04096484,
  -0.3116764 ,  0.15034433,  0.40302584,  0.23167458, -0.46024755,
  -0.53120923,  0.02606631, -0.07236407, -0.20114142, -0.41645506,
  -0.06540689, -0.0394449 , -0.15998234, -0.2612537 , -0.73177814,
   0.34444723,  0.11714911,  0.33274457,  0.00736615, -0.14456023,
   0.03909623, -0.02283888,  0.16444883,  0.05152372, -0.08920158,
   0.05181058,  0.0161289 , -0.29707953, -0.13732894, -0.04340796,
   0.15264072, -0.44747064, -0.12252352,  0.15366645, -1.5002484 ,
  -0.20526639,  0.7597128 ,  0.19119406,  0.02449821, -0.24011956,
  -0.41339895, -0.07098977, -0.12390527,  0.04330694, -0.27800527,
   0.3502601 , -0.15059093, -0.45387608, -0.03802824,  0.16403924,
  -0.02043791, -0.1455296 , -0.47370026],
 [-0.10052466,  0.09386207,  0.3025075 ,  0.28142384,  0.11416074,
   0.086498  ,  0.179883  ,  0.0872146 ,  0.08022778,  0.10077471,
   0.1455465 ,  0.23892206,  0.11367105,  0.39293858,  0.11083968,
  -0.1125698 ,  0.09426376, -0.19054255, -0.0274474 ,  0.11972782,
   0.38865677,  0.24337001,  0.05268812, -0.07987749,  0.23358665,
   0.20478831, -0.24539624,  0.08900707,  0.02261567,  0.34525773,
  -0.03557396, -0.03599024,  0.24238737, -0.08970231,  0.3441676 ,
  -0.22676097,  0.34606948, -0.19806582,  0.6944522 ,  0.3344206 ,
  -0.18407199,  0.21628462, -0.27682492,  0.11677232, -0.10779139,
   0.03215259, -0.0022203 ,  0.17767948, -0.19212833, -0.1875883 ,
  -0.1507916 , -0.09221815,  0.05523708,  0.11036399, -0.13139416,
  -0.1410833 ,  0.45062557, -0.1304107 ,  0.07453405,  0.98778427,
   0.18604562,  0.2332616 , -0.2172335 ,  0.18849397,  0.27686453,
   0.31945148,  0.19697025,  0.44671655,  0.3706768 , -0.3333529 ,
   0.04526031,  0.04377693,  0.16278867, -0.18799528,  0.1650198 ,
   0.13879839,  0.31240985, -0.2495971 ,  0.04336175,  0.12711367,
   0.00302724,  0.09781454, -0.24680269,  0.16238211, -0.11237606,
  -0.3927304 , -0.10762962,  0.46877575, -0.06745335,  0.256759  ,
   0.49994916,  0.03336374, -0.33596095,  0.01926282,  0.34648415,
  -0.29310378, -0.00875066,  0.48272237, -0.17932127,  0.03169682,
   0.5424637 ,  0.5387733 ,  0.02681268, -0.23946238,  0.4259194 ,
  -0.23486196,  0.03413322, -0.16516928,  0.17124726,  1.5662967 ,
   0.11683441,  1.2496706 ,  0.17044203,  0.05664225,  0.36370906,
   0.23263387,  0.18304822, -0.07341861, -0.22315755,  0.09407764,
  -0.08808174, -0.06820747,  0.51862395,  0.04683687,  0.01541752,
   0.18487242, -0.07763037,  0.19779909],
 [ 0.09462878,  0.19565535, -0.02183949,  0.41492525, -0.0687265 ,
  -0.19855098,  0.04916567,  0.06911591, -0.18273681, -0.16655184,
   0.24583443, -0.05287092, -0.15295738, -0.21768755, -0.13218217,
   0.06503101, -0.1974284 ,  0.19574831,  0.20691264, -0.4113458 ,
  -0.10676596,  0.28077114, -0.30326137, -0.02637984,  0.4284639 ,
  -0.12945831, -0.00617234,  0.18245038,  0.4414801 ,  0.14960933,
  -0.33148843, -0.17712355,  0.25092915, -0.06900455, -0.52830696,
  -0.2487795 ,  0.13212746, -0.02600188,  0.2672165 , -0.06733982,
  -0.30637756,  0.14690311, -0.3022576 , -0.47598588, -0.5041239 ,
   0.01204811, -0.11661402,  0.16600457,  0.11224546, -0.27842265,
   0.19988437, -0.15936546, -0.19667497,  0.02665736, -0.6283906 ,
   0.00297659, -0.00737167, -0.35755056, -0.05879559,  0.02395404,
   0.33714423, -0.08413411,  0.16204299, -0.00180708, -0.08853745,
   0.0249724 ,  0.07316104, -0.39819345,  0.1344652 , -0.09696795,
  -0.14342657, -0.16909009, -0.73162407,  0.02188731,  0.13778928,
  -0.35568136,  0.24654308, -0.26037973, -0.0472372 , -0.27708232,
   0.16350791,  0.00346449,  0.06112454,  0.7642946 , -0.24446836,
   0.13631243,  0.15449879,  0.2009982 , -0.00829011, -0.17866923,
   0.14911456, -0.27844438,  0.14295529,  0.09807955, -0.09070012,
   0.07363505,  0.13588211, -0.13351649,  0.0501797 ,  0.02522823,
   0.22884032,  0.44060665, -0.03724149, -0.088421  , -0.4523349 ,
   0.00124445, -0.4222525 , -0.26125264,  0.07083387,  1.419626  ,
   0.04179994,  0.6292048 , -0.04876782,  0.05623575,  0.05528305,
  -0.11818016, -0.02247959, -0.04160752, -0.4392028 ,  0.23739153,
   0.22193635, -0.06696398, -0.03346978, -0.04865471, -0.20385712,
  -0.16592488,  0.06372286,  0.28170362],
 [ 0.16305125,  0.2398449 ,  0.0596975 ,  0.3012889 , -0.08763951,
  -0.26852912,  0.20869586, -0.1371243 , -0.12761845,  0.18476515,
   0.39560583, -0.2621119 ,  0.5171871 ,  0.05742701,  0.1329757 ,
   0.05168077, -0.12863217,  0.36811274, -0.05331375,  0.68314135,
  -0.45634148,  0.48047394, -0.15338743,  0.05486332,  0.74741846,
  -0.13719736,  0.0111712 ,  0.03848456,  0.5904979 , -0.06308446,
  -0.2078242 , -0.45150968,  0.58701265,  0.23725775,  0.49624905,
   0.33252138,  0.2676967 ,  0.2442074 ,  0.83775234,  0.4787379 ,
   0.1169185 ,  0.09092914, -0.12997736, -0.06973706, -0.14853097,
  -0.04834091,  0.23536824,  0.34779882, -0.0695939 , -0.24612202,
  -0.13032484,  0.32414794, -0.38614795, -0.0855731 , -0.47344783,
   0.25186378,  0.20179388, -0.3760494 ,  0.02938103,  0.20476353,
   0.08781961,  0.49865875,  0.08231787,  0.22392069, -0.21183279,
   0.57470655,  0.2557856 ,  0.7161432 ,  0.22520289,  0.14188002,
   0.40207148,  0.24940442,  0.6648606 , -0.01067238, -0.00938382,
   0.01341159,  0.20058416,  1.2684528 ,  0.11444858,  0.44046542,
  -0.15908979, -0.15352659,  0.07198444, -0.0916972 , -0.430522  ,
  -0.09884261,  0.12474137, -0.18303923, -0.12531778,  0.0222077 ,
  -0.97936445,  0.16162784, -0.20115818, -0.2157481 ,  0.14509389,
   0.23041408,  0.04824534,  0.03911319,  0.12745945, -0.0129868 ,
   0.511983  ,  0.19140628, -0.09679075, -0.06190457,  0.14358588,
  -0.25392023,  0.4011812 , -0.32047173,  0.11593628,  1.7393866 ,
   0.34916827,  0.36532018, -0.2300746 , -0.25885642,  0.50612634,
   0.6881876 ,  0.08060651, -0.04961502,  0.16380607, -0.05547189,
   0.13895766,  0.10256706,  0.52345604,  0.13798626, -0.06694768,
   0.18651682, -0.01212934,  0.6292207 ],
 [-0.21181163,  0.2852409 , -0.07537257,  0.02718994,  0.0491927 ,
  -0.07504547,  0.03416924,  0.05920541,  0.02062368, -0.38968286,
   0.28063136,  0.02354574, -0.24523097,  0.03070873, -0.22827744,
  -0.09208499,  0.09791007,  0.16964713,  0.01525855,  0.03430582,
  -0.21355642, -0.18575367, -0.25491047,  0.28940415,  0.00698337,
   0.03099588, -0.20390357, -0.07661429, -0.0883349 , -0.5152578 ,
   0.04020872, -0.4922297 , -0.52944916, -0.06645437, -0.24306057,
   0.37813142,  0.07712341, -0.01778062, -0.45808414,  0.11774062,
  -0.03074556, -0.10617026, -0.15124962,  0.16057968, -0.18845414,
  -0.03138892,  0.23595756, -0.01138881,  0.06856734, -0.01545891,
   0.11876137,  0.28064492, -0.12547818, -0.3133598 ,  0.6529304 ,
  -0.25907636, -0.76576483, -0.09158349,  0.06434676, -0.6649152 ,
  -0.30435812,  0.08658694,  0.31241605,  0.28825116, -0.05569223,
   0.17598481,  0.1880772 , -0.5312081 , -0.1583906 ,  0.26936847,
  -0.01437549,  0.11269497, -0.0308612 ,  0.28666604, -0.1446594 ,
  -0.08262372, -0.20068742, -0.38191816, -0.26699167,  0.13840018,
  -0.11299468, -0.04202212,  0.23195606, -0.28084555,  0.15950745,
   0.12775423,  0.05868479,  0.0166018 ,  0.19474533, -0.55006117,
  -0.7548868 , -0.04684767, -0.01860567, -0.03122457, -0.02967854,
   0.07715826, -0.01568633, -0.15412526, -0.07555244,  0.04593883,
  -0.3571802 , -0.02484667, -0.11907002,  0.40599644,  0.23126623,
   0.00323154,  0.13832594,  0.10288282,  0.19863144, -0.80374265,
  -0.5174272 , -0.63053256,  0.05899704,  0.12746125, -0.05272263,
   0.05787343, -0.48318836, -0.00178035, -0.22254848, -0.13851364,
   0.05906326,  0.01466174, -0.11337525,  0.15490176,  0.0633906 ,
  -0.01527907,  0.02922191,  0.26398236],
 [ 0.03673176, -0.51584053, -0.22787654, -0.21278062,  0.0940723 ,
   0.43451715, -0.2503785 ,  0.09100834,  0.0254006 , -0.75923324,
  -0.71431   ,  0.04053756, -0.96221083,  0.01327718,  0.23980342,
  -0.30312687, -0.00500726, -0.54298836,  0.03457499, -1.6169266 ,
   0.13777795, -0.71473366, -0.3357974 ,  0.01329513, -0.32397687,
   0.2792044 , -0.5503856 , -0.05048243, -2.286576  , -0.34960017,
   0.03772706, -0.56181556, -0.993407  , -0.14451551, -0.9289698 ,
  -0.47432947, -0.14702457, -0.30067992, -1.2519441 , -0.06597257,
  -0.5101604 , -0.4616632 , -0.27742782, -0.177107  , -0.28767344,
   0.17865904, -0.44372758, -0.3779611 , -0.049445  ,  0.11612919,
  -0.04808266,  0.13222364,  0.11431418, -0.5702133 ,  0.2197273 ,
  -0.47460097, -1.2899657 ,  1.0246158 , -0.15886652, -2.2300987 ,
  -0.24916784, -0.30266288, -0.14215593,  0.13823862,  0.12767136,
  -0.38794318, -0.10418017, -1.2748345 , -1.3027173 ,  0.03202934,
  -0.24918833, -0.7455401 ,  0.14669691,  0.13797471,  0.04558498,
  -0.02059294, -0.63326454, -0.84166265, -0.8220969 , -0.5443331 ,
  -0.18377335, -0.43739086, -0.16411813,  0.0090176 ,  0.6372363 ,
  -0.01760587, -0.5212402 , -0.22554006, -0.16885248, -0.11808872,
   0.28890666, -0.07385586,  0.16814382,  0.0495977 , -0.7831251 ,
  -0.30682117, -0.1411845 , -0.02478585,  0.15446681,  0.19614922,
  -0.8347789 , -0.584571  , -0.6306735 ,  0.09345984, -0.09472507,
  -0.01120712, -0.43138272,  0.2471727 ,  0.16671938, -2.4459362 ,
  -0.8241622 , -1.5458324 ,  0.25725025,  0.02015211, -0.4532401 ,
  -0.5919092 , -1.2868738 ,  0.10225113, -0.75799036, -0.7915928 ,
  -0.15961605, -0.32470876, -0.28654715, -0.37388313, -0.9022181 ,
  -0.99860066, -0.51959   , -0.46604982],
 [-0.0330913 ,  0.11149633, -0.44122976, -0.5118439 ,  0.19437593,
   0.25024483, -0.49651805, -0.06414219,  0.16344464, -0.68540037,
  -0.40972474, -0.3108178 , -0.23032288, -0.0987373 , -0.02786405,
   0.13060385,  0.22821006, -0.7017368 ,  0.00147558, -0.0513148 ,
   0.672829  , -0.73785555, -0.25200891,  0.4025376 , -0.30689234,
   0.03652949, -0.57868433, -0.5542822 , -0.19064401, -0.22172964,
  -0.18261993, -1.4719509 , -1.4183385 ,  0.08048125,  0.18223497,
   0.5379288 ,  0.4362438 , -0.05924493, -1.2665938 , -0.4170245 ,
  -0.00481794, -0.04101919,  0.00568361,  0.5010063 , -0.31152597,
  -0.03700757, -0.14002089, -0.44509965, -0.06872606,  0.51718426,
   0.01075422, -0.71266234, -0.1374012 , -1.4090316 ,  0.44688064,
  -0.5075299 , -1.4854732 , -0.7611914 ,  0.01234564, -1.8344662 ,
  -0.3512118 ,  0.26864263,  0.15683247,  0.15763775, -0.0823354 ,
  -0.1238305 , -0.15660632, -0.8024317 , -0.95602506,  0.04457249,
  -0.1019507 ,  0.18088157,  0.23283313,  0.09462027, -0.03268605,
  -0.02720168, -0.02316622,  0.06682114,  0.10165147, -0.12338032,
  -1.7878908 , -0.79523575,  0.03316259, -0.7998229 , -0.1987013 ,
   0.20661582, -0.22252405, -0.30370083,  0.10305023, -1.6701021 ,
  -1.4835888 ,  0.26589286, -0.03525852,  0.12924342, -0.22978547,
   0.17903869,  0.01467787, -0.58314943, -0.0172216 , -0.14064258,
  -0.68912274, -0.5357913 , -0.9165629 ,  0.0976644 , -0.58502614,
   0.22330654,  0.19057795,  0.11773021, -0.2520277 , -2.3909526 ,
  -0.5255343 , -1.3942335 , -0.03821517,  0.15035255, -0.33933085,
  -0.4405458 , -2.0675066 ,  0.05515531, -0.5782743 , -0.58734775,
   0.08139084, -0.33126625, -0.04331015, -0.05574753, -0.7760452 ,
  -0.293821  ,  0.27280304, -0.6759131 ],
 [-0.14863947,  0.44102532,  0.4773472 , -0.34118885,  0.05441829,
  -0.14531144, -0.4047139 , -0.06413043,  0.02481888,  0.10833771,
   0.2264514 , -0.00717319, -0.17350382, -0.0406612 ,  0.00309291,
  -0.2208394 ,  0.00986984,  0.00673708,  0.00026668, -0.20804736,
   0.19707632, -0.2388029 , -0.5577579 ,  0.08475318,  0.09505431,
  -0.22988427,  0.05150697,  0.07737149, -0.9146526 , -0.44277185,
   0.0284127 , -0.8705191 ,  0.12191741,  0.16579337, -0.38325018,
  -0.03725686,  0.03061594, -0.08276094, -0.39176828,  0.4114326 ,
   0.45204633,  0.41700754,  0.5871487 , -0.09712182, -0.8069074 ,
   0.01274139, -0.43014568, -0.40498388,  0.03287833,  0.9288453 ,
   0.00993166, -0.26097268,  0.08953089,  0.23106195, -0.0819133 ,
   0.03933754, -0.5168418 , -1.1276189 , -0.991283  , -0.5656698 ,
  -0.7249081 , -0.04146441, -0.03214502, -0.20403145,  0.12103391,
   0.21841523,  0.5937687 ,  0.5407483 , -0.12844044,  0.19932395,
   0.17623362, -0.05682246, -0.9608995 ,  0.23878714,  0.12486385,
   0.04464321,  0.2971266 ,  0.03127188, -0.4690296 ,  0.00849574,
  -0.77427596,  0.24876978, -0.54402506, -1.7334167 , -0.14890108,
  -0.01815675,  0.07313005, -0.15425234, -0.42175233, -0.6913087 ,
  -0.47543877, -0.1538274 ,  0.04490143,  0.09719821,  0.08779418,
  -0.21811123,  0.61792684, -0.13984793,  0.02646967, -0.00733354,
   0.21503252,  0.4496804 , -1.4837576 , -0.18484972, -0.17779402,
  -0.02400789, -0.6363166 , -0.02037977, -0.02796423, -0.02227694,
  -0.05077229, -0.58612764,  0.0398065 ,  0.02506973, -0.1813724 ,
   0.35969025, -0.28993505, -0.12932573,  0.70382845,  0.05787145,
  -0.04350096,  0.066687  ,  0.14561416, -0.26595762, -0.5285228 ,
  -0.32144102,  0.11186935, -0.2749765 ],
 [-0.0797497 , -0.4892036 ,  1.1362646 , -0.4953508 ,  0.15821406,
   0.54440963,  0.7747148 , -0.1897655 , -0.15888707, -0.14574328,
   0.18644688,  0.2743556 , -0.38296923,  0.20072298,  0.00517588,
   0.08723031,  0.1734842 ,  0.04724894, -0.079578  ,  0.07015917,
  -0.00282966, -0.02298988,  0.35959816, -0.8298798 ,  0.16574821,
  -0.10957818, -0.73608965,  0.04220238, -0.86949223,  0.66367775,
  -0.86619586, -0.16430075, -0.500659  , -0.12572826, -0.29940996,
  -0.6251274 ,  0.29163396,  0.19908795, -0.4865862 , -0.38540444,
  -0.45185933, -0.30214694, -0.24980135, -0.08346403,  0.04551218,
   0.04581121,  0.00362063,  0.09193426, -0.06737377, -0.565546  ,
   0.08233459, -0.05465638, -0.19134699,  0.5380848 ,  0.09310256,
   0.09094808, -0.10638754, -0.1981967 , -0.5458136 ,  0.51679474,
  -0.5639467 , -0.34630772, -0.30962837, -0.12789579,  0.20049289,
  -0.3838421 , -0.43534952, -0.17157131,  0.16966446,  0.00336932,
   0.42637533, -0.24327391, -0.26425707, -0.29784667,  0.04555257,
  -0.03693919, -0.23873603, -1.0704606 , -0.27936482, -0.22601208,
   0.02315667,  0.45368677,  0.6288843 , -0.43734694,  0.14360887,
   0.03152456, -0.44288608,  0.02568387,  0.11832395,  0.48391145,
   0.64957565, -0.19371785, -0.09745255,  0.09148054,  0.39734703,
  -0.20618801,  0.11486144,  0.2382008 ,  0.07085511, -0.19828913,
  -0.44764107,  0.13897286,  0.20217487, -0.38588297,  0.15415974,
  -0.0931313 ,  0.26319623,  0.17258215,  0.21540864, -0.31593615,
  -0.32433835, -0.07124863,  0.08226673, -0.06446874, -0.11430501,
   0.00771395,  0.03835924,  0.41879806, -0.5161086 ,  0.01587408,
  -0.01029493, -0.11077711, -0.5983714 ,  0.09984694,  0.2663262 ,
  -0.46405232, -0.18896331, -0.05856283],
 [ 0.17629656,  0.3734754 ,  0.59823143,  0.04762467, -0.05014875,
  -0.11251473,  0.3082175 , -0.03865674,  0.14647903, -0.37463564,
   0.01563787, -0.229431  , -0.17739472, -0.31180447,  0.02482969,
  -0.26148665, -0.28598514, -0.02629589, -0.01909613, -0.4060832 ,
  -0.2651012 ,  0.22580367, -0.0346586 ,  0.24364081,  0.08756977,
   0.374474  , -0.05943975, -0.11000377,  1.4548169 ,  0.8979543 ,
  -0.46477428,  0.3267917 , -0.13738637,  0.12133685, -0.5971593 ,
   0.40410778,  0.01564215, -0.50978154, -0.6259004 , -0.7315892 ,
  -0.09449386,  0.40881684, -0.16091122, -0.25690287,  0.59421015,
  -0.05304336,  0.36900768,  0.03982732,  0.1697933 ,  0.03170359,
  -0.0395702 , -0.09587517,  0.06357071,  0.03645398, -0.48354205,
   0.303485  ,  0.6510228 , -0.04126094,  0.95562786,  0.14108539,
  -0.4965001 ,  0.5575439 , -0.30409414, -0.07509534,  0.89675415,
   0.07052512,  0.31988358,  0.03235422,  0.14025877,  0.40263608,
   0.03451025,  0.5146772 , -0.28499848, -0.17320701,  0.11074844,
   0.50449127,  0.06586698,  0.3374637 ,  0.6181676 ,  0.33571118,
  -0.18659878,  0.5270398 ,  0.10836294,  0.3812862 ,  0.12563491,
   0.05519565,  0.09036381, -0.0328913 ,  0.16235012,  0.7132177 ,
  -0.24979122,  0.13009249,  0.06268335, -0.016295  ,  0.22837833,
   0.2119906 , -0.38336346,  0.05273908,  0.10365368,  0.10248888,
   0.27677736, -0.12179601,  0.24161518,  0.16832452,  0.3980555 ,
   0.20303376,  0.60737556, -0.29700735, -0.03636103, -0.12356043,
  -0.38112053, -0.8903229 , -0.10703984,  0.08679156,  0.0387044 ,
   0.24708946,  0.47764692, -0.26250502,  0.05215212,  0.15642138,
  -0.04046082, -0.3793002 ,  0.11821841, -0.31336883, -0.01593587,
  -0.39116508, -0.10150312,  0.20661655],
 [ 0.41071597,  0.5498701 , -0.4256569 , -0.8235198 , -0.3634483 ,
  -0.8075442 ,  0.06477252,  0.0698786 ,  0.15513113, -0.13886966,
   0.21125992, -0.18069997, -0.44098008,  0.07066294, -0.505496  ,
   0.48904416,  0.02996838, -0.23935364, -0.24897227,  0.19158101,
   0.9983779 ,  0.18968448,  0.01379858,  1.0197808 , -0.3278441 ,
   0.6329779 , -0.5008147 ,  0.27235666, -0.45712116,  0.95222425,
   0.21034063,  0.48592907,  0.31827757, -0.28282565, -0.9394966 ,
  -0.15088323, -0.57087713, -0.7208148 ,  0.04075322, -0.33499068,
  -0.15596408,  0.5482779 ,  0.25192496,  1.3044921 ,  0.17981417,
   0.03736199, -0.40368065,  0.49546102, -0.23099597, -0.59896296,
  -0.06129674, -0.06554016,  0.12994404, -0.28532064,  0.65935934,
  -0.42821243,  0.4505947 ,  0.17589098,  0.45243245, -0.25266886,
   0.51655346, -0.7125567 , -0.12691522,  0.10846775,  1.0806998 ,
  -0.19848308, -0.33663064, -0.07375804, -0.4572168 , -0.3860619 ,
  -0.28700274,  0.46864986, -0.3178514 , -0.41617754, -0.10925002,
  -0.57658404, -0.7479513 , -0.35353327,  0.25539896, -0.5306845 ,
   0.92835224,  0.06456909, -0.46761346,  0.8239029 , -0.29612148,
  -0.1469178 ,  0.0328519 ,  0.64073515,  0.09805612,  0.24925433,
  -0.18093316,  0.07911971, -0.07431847, -0.64674467,  0.44018734,
   0.4736744 , -0.64591897, -0.5623697 ,  0.17800325,  0.29923084,
   0.608138  , -0.9281362 ,  0.86655635, -0.17274344,  0.6391152 ,
  -0.14432216, -0.26047045, -0.02235892,  0.37880507, -0.09278198,
   0.03129042, -0.30776113,  0.02490703, -0.02037182, -0.30374998,
  -0.08815778,  0.39103296, -1.202484  ,  0.440375  ,  0.03773891,
  -0.42211455, -1.1209855 ,  0.0932532 , -0.20659788,  0.01972938,
  -0.7019405 ,  0.3098532 , -0.2721896 ],
 [ 1.0732158 ,  0.7924826 , -1.6664876 , -1.6505646 , -1.2510328 ,
   0.26019433, -0.02286951,  0.6074588 ,  0.29481152,  0.05054524,
  -0.0796411 ,  0.30257648,  1.0668488 , -0.24104299, -0.16804208,
   1.113337  , -0.12213015, -0.07299982,  0.4843158 ,  2.2050962 ,
  -2.4772282 ,  0.41786498,  0.63506246,  0.00778087, -0.2723086 ,
  -0.70189273, -0.073472  ,  0.3331691 ,  1.2155128 , -0.9404042 ,
   0.46361634, -0.34435698,  1.2686495 ,  0.451454  ,  1.7811328 ,
  -0.7562656 , -2.0324829 ,  0.77878433, -0.05353216,  0.5219802 ,
   0.70331043, -0.11475152,  0.51423085,  1.5120229 , -1.8501295 ,
  -0.15383764,  0.589333  , -0.04826724,  0.16399884,  0.23959394,
   0.2203511 , -0.20932283, -0.33617687, -0.86203516,  1.3031636 ,
   0.02857172, -1.4601679 ,  0.02015563, -0.2724758 , -0.40257534,
   1.9831884 ,  0.8030538 ,  0.43716106, -0.17614365, -0.960637  ,
  -0.30775696,  1.2130563 , -0.30559134, -1.2954714 , -0.8672958 ,
  -0.64583415, -0.04440733,  1.7663453 ,  0.17077   ,  0.32992473,
  -1.9038092 ,  0.2915934 ,  1.7497936 , -0.96663874,  1.0595908 ,
  -0.0901178 ,  1.2095487 , -0.72672176,  0.23412465, -0.3969873 ,
  -0.27201796,  0.73297733,  0.19924584, -0.28251708, -0.43257213,
  -0.0794702 , -0.477108  , -0.27200726,  0.68958867,  0.95241326,
  -1.354389  ,  0.06123723, -0.46504992,  0.65274334, -0.4777798 ,
   0.35226235,  0.03890122,  0.0995    ,  0.3299511 ,  0.15207149,
   0.01366608,  1.0734802 , -0.6548201 , -0.25311732, -0.47624552,
  -0.19465367,  0.44041973, -0.00326312, -0.36387992, -0.304557  ,
   0.04012554,  0.10529712,  0.06754611, -0.01906462, -0.6442145 ,
   0.1836454 , -0.17108998,  1.2067584 , -0.06857193,  0.31214216,
  -0.5501243 ,  0.01062741, -0.03665247],
 [ 0.00010154,  0.4224544 ,  0.11553687, -0.1296124 ,  0.63365185,
   0.5881756 ,  0.07890782,  0.24915732,  0.56189936, -0.17426182,
  -1.0929832 , -0.1841684 ,  0.2080831 ,  0.88030946,  0.03579246,
  -2.4222565 ,  0.47228274, -0.2079592 , -0.21064147, -1.9180588 ,
  -0.08812384, -0.62676346,  0.15225339, -0.7833235 , -0.06186979,
  -0.2811491 ,  1.0817283 ,  0.3670889 ,  0.00837935, -0.42471427,
  -0.9804006 , -0.42497778,  0.04531105,  0.16853811,  0.40802637,
  -0.3708889 ,  0.7897185 ,  0.10618788,  0.14840865,  0.30762175,
   0.66771835, -0.42515683, -0.90186524, -1.2573842 ,  1.8876755 ,
  -0.08285835,  0.5258551 ,  0.4348906 , -0.9652338 ,  0.46026865,
  -0.24358055, -0.85146976,  0.8191927 ,  0.5676105 ,  0.14844908,
  -0.18568836,  0.35068628,  0.28881216, -0.43683037, -0.20777269,
  -0.8930032 , -0.29659536,  0.7199054 , -0.3619013 ,  0.14468822,
   0.10614551,  1.2612007 , -0.6037007 , -0.673949  ,  0.4834827 ,
  -0.6466153 ,  1.0367967 , -0.5982258 ,  0.2502059 ,  0.2792038 ,
   0.6969105 , -0.5813502 , -1.6498159 ,  0.31223744, -1.3596163 ,
  -0.37185365, -1.9404901 ,  0.73493594, -0.59825665,  0.25107232,
  -0.6225293 , -0.91560835,  0.01991955,  0.1955026 , -0.03306039,
   0.91724026, -0.09198908,  0.30905455,  1.3724219 , -0.76189333,
   0.8988859 , -0.8425027 ,  0.6091805 , -0.16266328,  0.90505695,
   0.24564432,  0.17194115,  0.24382418,  0.8669686 , -0.29835692,
  -0.2910431 , -1.5075613 , -0.12181131,  0.0111649 , -0.820756  ,
  -0.52067876,  0.23665519,  0.4696069 ,  0.139084  , -0.13394094,
   0.5241968 ,  0.6062118 ,  0.45429948, -1.862441  , -0.00566544,
   0.04868411,  0.03767707,  1.2657442 ,  0.28464606, -0.18363854,
   1.1274505 , -3.0789847 ,  0.3543193 ],
 [-0.34236372, -0.16940065, -0.48722458,  0.08612178,  0.46856177,
   0.30748802,  0.49734926,  0.10793624,  0.24503212,  0.39227188,
  -1.18103   , -0.05933239, -1.2314293 ,  0.17983074,  0.11701505,
   0.65914524,  0.16505691, -0.3440581 ,  0.25751287, -0.4226217 ,
   0.8402488 , -1.1496971 , -0.28302497, -0.49892855, -0.23610605,
   0.41227683,  0.16214931,  0.30916345, -0.11879122,  0.02254229,
  -0.70611364, -0.8668756 , -0.6048234 , -0.03459479, -1.3323195 ,
  -0.6249719 , -0.97223306,  0.4102831 , -0.10220275, -0.11326766,
  -1.4287692 ,  0.24312437, -0.3837661 , -0.09851878, -0.7934761 ,
   0.04181533, -0.14247493,  0.07098858, -0.5563388 , -1.2000213 ,
  -0.06702196, -0.7983802 ,  0.11524878, -0.22934876, -0.63095677,
  -0.06054879,  0.5040409 ,  0.10153483,  0.03784007, -0.00224411,
  -0.3886582 , -3.053288  , -0.46299836, -0.23412292,  0.12100142,
   0.07160254, -0.7827671 , -0.4406959 , -0.75088   , -0.7041142 ,
   0.81539804,  0.2329367 ,  0.36684892,  0.6745436 ,  0.21028921,
  -0.0181231 , -0.16854902,  0.01341822, -0.71909356, -2.7862804 ,
  -0.0339649 , -0.2939761 , -0.79502654, -1.0670751 ,  0.25751954,
  -0.30493125, -0.35129136,  0.6619253 , -0.69843256, -0.67645967,
  -0.57543117, -0.36389416,  0.10171715, -0.15069915, -1.117064  ,
   0.5952655 ,  0.21472889, -0.08368675,  0.01317031,  0.63725555,
  -0.0405051 ,  0.26886353, -0.55946916,  0.11595277,  0.1573481 ,
   0.2630849 , -1.2956395 , -0.02804584, -0.38199815, -0.6744961 ,
  -0.34784123,  0.32927167,  0.03223212,  1.3188553 , -0.70910645,
  -0.8973275 , -0.5914795 , -0.24734326, -0.6188376 , -0.69293886,
  -0.02674488, -0.45676842, -0.33450365, -0.16015561, -0.29828978,
  -0.9515365 ,  0.53874314,  0.04908313],
 [-0.36452165, -0.04625337, -0.47287667, -0.16346644, -0.32407296,
  -0.10251388, -0.12292403, -0.288887  , -0.13669078, -0.16161883,
  -0.19442745, -0.26637116, -0.2229004 , -0.17401788,  0.25893816,
  -0.09104232, -0.18478097, -0.14509259,  0.12910673, -0.03699882,
  -0.42853606,  0.3285817 , -0.50800323, -0.29012784, -0.05446962,
   0.0538918 ,  0.09125529, -0.3864014 ,  0.03721917, -0.06432604,
  -0.11167008, -0.05979676, -1.0200063 ,  0.01956459, -0.38966024,
  -0.0779557 , -0.73249096, -0.2538164 , -0.23481931, -0.6632057 ,
  -0.5160285 , -0.7863904 ,  0.12895572, -0.11211766, -0.20700891,
   0.08389486, -0.5478023 ,  0.39732298,  0.24135344, -0.23722816,
   0.2113644 , -0.38407308, -0.07937759, -0.511263  , -0.15554604,
   0.0618243 ,  0.43431035, -0.17655271, -0.15256353,  0.15723747,
  -0.3221906 , -0.68363273, -0.5171267 , -0.08309747, -0.10015389,
  -0.3829167 , -0.5677015 , -0.4410463 ,  0.02151266,  0.21757652,
   0.3568813 , -0.9185219 , -0.7382738 , -0.34863663, -0.27028874,
  -0.39790916, -0.7311963 ,  0.35781142, -0.22591376,  0.24938263,
  -0.30988538, -0.10101385, -0.3133891 , -0.00303447,  0.00825007,
   0.3175715 ,  0.23753008, -0.25076756, -0.6993757 , -0.24618429,
   0.25765783,  0.14623924,  0.14552025,  0.2551973 , -0.57218695,
  -0.32566792,  0.14528072,  0.08146633, -0.06968044, -0.28202328,
   0.35869285,  0.2614694 , -0.5782842 ,  0.00362738, -0.39766103,
   0.12572704, -0.03629564,  0.13734643, -0.7143889 , -0.6311585 ,
   0.2656175 , -0.05402704, -0.04923843,  0.3830721 ,  0.00852704,
  -0.06720628, -0.70626014, -0.4065875 ,  0.06167223, -0.17911606,
   0.12375134,  0.07437076, -0.94558287, -0.00027691, -0.1385991 ,
  -0.31499857, -0.27996618,  0.46414456],
 [-0.34581774, -0.09942278,  0.07254487, -0.24037431, -0.06289848,
  -0.00157977, -0.26134792,  0.03924416, -0.06711293,  0.24349698,
  -0.34345523, -0.22620939, -0.21759003,  0.02986687,  0.03550204,
   0.1141645 , -0.04175945,  0.01281366,  0.02914377,  0.15522318,
  -0.14831524,  0.4668963 , -0.35642272, -0.35951266,  0.01877047,
   0.37118807,  0.06145256, -0.02560953, -0.18593872, -0.17266506,
  -0.02030182,  0.2145839 , -1.0036619 , -0.05428441, -0.09258725,
  -0.00776593, -0.42145425,  0.13223824,  0.07271   , -0.6220168 ,
  -0.55130285, -0.67932075,  0.14510156,  0.16367458, -0.3754991 ,
  -0.11837099, -0.2961734 ,  0.79396856,  0.09740269, -0.19088018,
  -0.05491598, -0.18778281,  0.00199382, -0.3379526 , -0.19534706,
  -0.04878384,  0.4999122 , -0.02719432, -0.03811228,  0.26047397,
  -0.19153404, -0.5683199 , -0.46638355, -0.12778927, -0.21275316,
  -0.42766595, -0.7965698 , -0.29886913, -0.03516153,  0.1755859 ,
   0.27606085, -0.9534874 , -0.5008723 , -0.19303085,  0.08654844,
  -0.17332165, -0.5753799 ,  0.0883415 ,  0.05043287,  0.52407336,
  -0.12580313, -0.11705738, -0.4064346 , -0.06073058,  0.07454886,
   0.10483561,  0.15170963, -0.20334437, -0.4921471 ,  0.04298859,
   0.03844372, -0.05125675,  0.00363609, -0.01172897, -0.34649187,
  -0.20971456,  0.3115124 , -0.13989612, -0.0079869 , -0.05187246,
   0.33700514,  0.40654814, -0.607078  ,  0.13956109, -0.11713597,
   0.01267839,  0.06097849,  0.00886042, -0.712078  , -0.4437371 ,
   0.749051  ,  0.09485934,  0.1998455 ,  0.00641697,  0.02745659,
   0.01086394, -0.3795197 ,  0.05366248,  0.12123843, -0.07799632,
  -0.03607374,  0.16291015, -0.6292952 ,  0.16853057,  0.19433507,
   0.18299322,  0.00412294,  0.7170747 ],
 [ 0.02418183, -0.07367806,  0.4345327 , -0.41859543,  0.4317066 ,
   0.07258017, -0.04823544,  0.18526135,  0.21803738,  0.38359535,
   0.04021054, -0.05879007, -0.31981042,  0.14867157, -0.40890664,
   0.03467265,  0.07775706, -0.21046267, -0.14636578,  0.16109835,
   0.20735736,  0.972984  , -0.2466139 , -0.27443996,  0.22268811,
   0.68642175,  0.11751731,  0.37486905, -0.22181106,  0.07797182,
  -0.04361426,  0.21690324, -1.0131884 , -0.09193348,  0.035366  ,
   0.05302989, -0.23092932,  0.03934537,  0.02513693, -0.69904906,
  -0.9839923 , -0.6648217 ,  0.12493722,  0.24200822, -0.3366049 ,
  -0.21227649, -0.5669423 ,  0.86494225, -0.27907357, -0.46659398,
  -0.22350031,  0.17312066, -0.1822945 , -0.11989237, -0.04476634,
  -0.30778915,  0.44490218,  0.04053956,  0.17785825,  0.42963272,
  -0.2163951 , -0.5416372 , -0.45265108,  0.23139217,  0.04213534,
  -0.6423793 , -0.72454774, -0.18418644,  0.3789162 , -0.03777563,
   0.776429  , -0.8305266 , -0.592774  , -0.34390232,  0.11670028,
   0.08843676, -0.64816123, -0.04439151,  0.1154567 ,  0.4072918 ,
   0.21578208, -0.00063101, -0.4629907 ,  0.01383431, -0.20693697,
  -0.18746473, -0.14108714,  0.32410103, -0.31918296,  0.34040543,
  -0.03903825, -0.18802413, -0.40226516, -0.28413376, -0.09514481,
   0.13001446,  0.2972064 , -0.33851296,  0.13738152,  0.22050408,
   0.31618145,  0.55025536, -0.39072773, -0.0862854 ,  0.12625293,
  -0.1912424 ,  0.33298087, -0.05138207, -0.6303236 , -0.21378534,
   1.379979  ,  0.22547574, -0.09676325, -0.15215242,  0.12074694,
   0.31214836, -0.24572343,  0.16740292,  0.406298  , -0.20922346,
  -0.18730362,  0.4636057 , -0.7031409 ,  0.48382214,  0.38177142,
   0.24347946,  0.28288585,  0.93598926]])
    b1 = jnp.array([-0.48746228, -0.51183707, -1.7235086 , -0.15221316, -1.9221895 ,
 -1.4076462 , -1.7974483 , -0.47574526, -0.6949402 , -2.2589538 ,
 -0.31537023, -1.644476  , -1.3534561 , -1.818027  ,  0.5764546 ,
  0.38713664, -0.8059508 , -1.5055209 ,  0.8143762 , -0.46387756,
 -1.3713657 , -1.3370082 , -2.2331793 , -2.0072486 , -0.96443194,
 -1.4445049 , -2.4694517 , -1.9891298 ,  1.1305411 , -1.716047  ,
 -0.8118631 , -2.0908868 , -0.3162158 , -1.6152251 , -1.6289986 ,
 -0.5300504 , -2.07029   , -0.9735304 , -1.503496  , -1.9446973 ,
 -1.6730446 , -1.2708678 , -2.1471848 , -0.94691056, -1.4197403 ,
  0.71051335, -1.6207314 , -2.176967  ,  2.0587418 , -2.7797048 ,
  1.5329161 , -1.6066731 ,  0.08141399, -1.5604917 , -1.4670682 ,
 -0.79719037, -1.5380297 , -0.7853396 , -2.0526276 , -0.9338266 ,
 -1.5406599 , -0.7672982 , -1.9916843 , -2.3289044 , -0.75039005,
 -0.33917207, -1.4635341 , -0.88990647, -1.3402079 , -1.5769559 ,
 -1.9529161 , -2.2928548 , -1.3939942 , -1.6500151 , -0.668117  ,
 -1.5696063 , -1.6999389 , -0.3215994 , -1.4760838 , -1.6540208 ,
 -1.0183293 , -1.1121949 , -1.4434973 , -0.11746735,  0.05293589,
  1.274098  , -0.86085117, -1.8791595 , -2.207372  , -2.4628742 ,
  0.31237522, -1.7781438 ,  1.5178804 ,  0.7046932 , -1.5670662 ,
 -1.5096929 , -1.8723633 , -1.0728012 , -0.29264286, -1.0640731 ,
 -1.3152803 , -1.8702542 , -2.0786648 , -1.999288  , -1.962469  ,
 -0.4509166 , -0.9291199 , -0.75682837, -0.30476445,  0.0763353 ,
 -1.9066232 , -0.7802661 ,  0.10308808,  1.560515  , -1.4480671 ,
 -2.024955  , -1.2604029 , -1.3215615 , -1.912145  ,  0.21522792,
 -0.74998975, -2.1054308 , -2.0529838 , -1.8991631 , -1.2197727 ,
 -2.772863  , -1.6730053 , -1.5846205 ])
    W2 = jnp.array([[-0.38350958, -0.11290994,  0.25700116, -0.2614587 ,  0.9433003 ,
  -0.67563087,  0.27047774,  0.29551   , -0.33728814, -0.10276582,
   0.120444  , -0.2810172 , -0.2292217 ,  0.11650736,  0.04686295,
  -0.13262153, -0.13488844,  0.18454492,  0.13976264, -0.37596014,
  -0.14652063,  0.05868563, -0.28347704, -0.58002687, -0.7827372 ,
   0.02302615, -0.5401852 , -0.10111296, -0.49386117,  0.17995493,
  -0.24722356,  0.00063574, -1.1107364 ,  0.5440528 , -0.85190624,
   0.28027838, -0.80959564, -0.76594025, -2.2447875 , -0.30991703,
  -0.00613575,  0.7253939 , -0.5372374 , -1.0154715 , -0.47364792,
  -0.65333146, -0.23873226, -0.32478535, -0.03480557, -1.1328375 ,
  -0.8076816 , -0.56372845,  0.14488542, -1.9883399 ,  0.07265127,
   0.09696146, -0.01556521, -0.10854793,  0.00878607, -0.46592867,
   0.14526589, -0.89071494, -0.32391837, -0.548333  , -0.08014149,
   0.07734638, -0.16138574,  0.62696624,  0.03296224, -1.1022197 ,
  -0.06445509, -0.62563324,  0.23636343, -0.42912865, -1.3691221 ,
  -0.50068825,  0.8914054 , -0.14994898, -0.25152278, -0.13393055,
  -1.088713  , -0.4816396 , -0.92251873, -0.09012656, -0.56307656,
  -0.34731424, -0.53539705, -0.33878067, -0.5934908 , -0.7958855 ,
  -0.22615558,  0.11648059, -0.32709253, -0.16281024, -0.64858896,
   0.32988286,  0.08036523, -0.57762337, -0.12641484, -0.7201568 ,
   0.560778  , -0.18869717,  0.08929458, -0.6402969 , -0.56904995,
  -0.29673734,  0.04809175,  0.19264174, -0.05040414, -0.90400845,
  -0.08478172, -0.15656896, -0.21642946, -0.8214796 , -0.20008993,
  -0.00024602,  0.13546301, -0.6315944 , -0.90752375, -0.8608916 ,
  -0.30027425, -0.64947766, -0.92208904,  0.08514041, -0.0921324 ,
  -0.04145217, -0.19133356, -0.45203876],
 [ 0.18103558, -0.4119442 , -0.77379644,  0.00239691, -2.087526  ,
   0.13844928, -0.9409168 , -0.502496  ,  0.35798264, -1.234334  ,
  -0.7827189 ,  0.02849922,  0.73884875,  0.56580216,  0.91673833,
  -0.43075377,  0.620345  ,  0.93759495, -0.25716028,  0.05032958,
   0.75408775, -0.2614933 ,  0.5785643 , -0.33987492,  0.26129296,
  -0.19355176, -0.20822784, -0.09115873, -1.0467142 , -0.0758538 ,
  -0.65516996,  0.301008  , -0.15555006,  0.6247363 , -0.10919876,
   0.2675491 ,  0.32701826, -0.26323175, -0.9490111 ,  0.19412032,
   0.95745593,  0.4509994 ,  0.18222414, -0.01119185, -0.19747649,
  -0.23794654, -0.03566766, -0.3191801 ,  0.75210285,  0.27309698,
  -0.6057073 , -0.0171997 ,  0.7605724 ,  2.390525  ,  0.0806469 ,
  -0.22953816,  0.23300931,  1.3941618 , -0.58461344,  0.5468044 ,
  -0.5220814 , -0.12319426,  0.41880178,  0.08237141,  0.26067284,
   0.04338383, -0.32896376, -0.00771264,  0.48876667,  1.6283021 ,
   0.12391277, -0.06048728,  0.4212829 , -0.1681174 ,  0.14824812,
   0.50886405,  1.0673975 ,  0.06605054, -0.39115584,  0.12553208,
   0.3052824 , -0.0091208 ,  0.0473299 , -0.56564426, -0.5930364 ,
   0.11116087,  0.5833929 , -0.05908791, -1.5430332 ,  0.14478445,
  -0.20766905,  0.2824979 ,  0.14670406,  0.5340581 , -0.30234787,
  -0.00833459,  0.4283967 ,  0.45396787,  0.08937397, -1.3587549 ,
   0.63125753, -0.725115  , -0.5736442 ,  0.3585511 , -0.07255332,
  -0.29467013,  0.6205754 , -0.12956718,  0.01290911, -1.4863604 ,
  -0.86698145, -1.0931393 ,  0.02817475,  0.59545076,  0.57865363,
   0.02846346, -0.76359737,  0.20863065, -0.08353569,  0.39077502,
   0.263607  , -0.2724619 ,  0.9510565 , -0.15139784,  0.30531076,
  -0.19555405,  0.44278717, -0.29523322],
 [ 0.94793844, -0.2732949 , -0.44326878,  0.26177233, -0.9053038 ,
   0.1297951 ,  0.49395165,  0.8322896 , -0.56075317, -0.27796414,
   0.4465555 ,  0.6122716 ,  0.49369657, -1.9516842 , -0.11741097,
   0.30508238,  1.041127  ,  2.377608  , -0.12363865,  0.37522155,
   0.86377937, -0.36062428,  0.00915465,  0.09598894,  0.22259177,
  -0.28649703,  0.02590775,  0.20764376,  0.8845473 ,  0.73012334,
  -0.34411952,  0.4954956 ,  0.2621415 , -0.32352394,  0.5028181 ,
   1.6323303 , -0.44435295,  0.22533944,  1.8575082 ,  0.15176155,
  -2.0699477 , -1.9583638 ,  0.3540941 , -0.24983104, -0.02158237,
   1.0014378 ,  0.31139207,  0.27669528, -0.13934465,  0.06623585,
  -0.324906  ,  0.22590858,  0.6309074 ,  0.86465794,  0.06306595,
   0.09695121,  0.29176134, -0.1623496 , -0.71685183, -0.2257616 ,
   1.187723  ,  0.13274728, -0.46453506, -0.5542985 , -0.43626294,
   0.02141289,  0.30426434,  0.3103415 ,  0.9991997 ,  0.29116717,
  -0.89270043,  1.5706574 ,  0.51671255,  0.04422846,  0.47126126,
   0.17489262, -1.639064  , -0.6958056 ,  0.17101233,  0.43981168,
   0.36546826,  0.41708624, -0.8067764 ,  0.6042661 ,  0.23981607,
  -0.45904478, -0.25030366, -0.11348579,  0.28989688,  0.133953  ,
  -0.73902225, -0.07971536,  0.092223  ,  0.54684126,  0.32064167,
  -0.3566261 ,  0.14524247, -0.20214072, -0.56562746, -0.37833726,
  -1.8566517 ,  0.30458435, -0.33287802,  0.45115286, -0.20547226,
   1.1877562 , -1.2424637 , -0.0509645 ,  0.23816137,  0.66768306,
   0.09191269,  1.0139606 ,  0.02446977,  0.3469894 , -0.95370543,
   0.3689515 , -0.06576546, -0.07727   , -0.22594352,  0.16587083,
   0.1975315 ,  0.4601064 , -0.3867855 , -0.25959393, -0.21385248,
  -0.34041974, -0.932872  , -0.27300084],
 [ 0.3095483 ,  0.5964683 , -1.3358116 , -0.05252602, -1.1526011 ,
  -0.52679545,  0.68591666,  0.9919132 ,  0.48397675, -0.3096036 ,
  -0.6758425 , -0.1972698 , -0.96448356,  0.41813648, -0.11769934,
   0.11378097, -0.24955207, -0.7589036 , -0.3351928 , -0.15640913,
  -0.6662084 , -0.1663928 , -0.00757051, -0.32735822,  0.281189  ,
  -0.56050634, -0.77653044, -1.3593767 , -0.75126123, -0.07301933,
  -0.04033668,  0.23451117, -0.98872316,  0.14644632, -0.20998473,
   0.21066967, -1.0239067 , -0.46404696, -0.09493106, -0.18250519,
   0.704472  , -1.9005847 , -1.7580322 , -0.37141326,  0.32094392,
  -1.1089183 , -1.7247845 ,  0.10697924,  0.30154487,  0.07796139,
  -1.2281163 , -0.28938603, -1.527271  , -0.04712082,  0.20661157,
  -0.45556974, -0.03815747, -0.49691468,  0.1531399 , -0.96132946,
  -0.14568971,  0.23982704,  0.26831818,  0.29090774, -1.4425848 ,
  -0.19819704,  0.50585425, -0.20812312, -0.4969323 , -0.16733152,
  -0.5563304 , -0.23331536, -0.05320814,  0.14322448,  0.30420515,
  -0.28327715,  0.03390962,  0.41451672,  0.01927789,  0.07884929,
  -0.03561024,  0.13798414, -0.3922499 , -1.0207262 ,  0.21579058,
  -0.3281392 , -0.724076  ,  0.21032219, -1.173266  , -0.25801548,
   0.17402896,  0.4021875 ,  0.05599833, -0.69564277, -0.11021306,
   0.17870206,  0.10801294, -0.09357354, -0.05001819,  0.17822649,
   1.3073916 , -1.3560971 , -0.0815331 , -0.23145673,  0.17359573,
   0.06258556,  0.27993366, -0.08258452,  0.26491573, -0.29839456,
   0.7136087 , -0.1436566 ,  0.416745  , -0.38403687,  0.23436557,
   0.17826816, -0.3379308 , -0.3495508 , -0.04583991, -0.5367377 ,
  -0.31579   , -0.04771712, -0.08247791,  0.18434736, -0.2897481 ,
  -0.45988935,  0.50061244, -1.1230642 ],
 [-0.00138782,  0.11399828, -1.5220433 , -0.04719659, -0.81645584,
   0.2908962 , -1.6616672 , -1.8502179 , -0.7007699 ,  0.71640635,
  -0.42644864, -0.3537544 ,  0.05920842,  0.31004834,  0.24244219,
  -0.6181945 ,  0.7622279 , -3.081636  , -0.19397676, -0.7317352 ,
   1.096629  , -0.02583089,  0.18082134, -1.4435432 , -1.6693684 ,
  -0.7361673 , -0.24732895, -0.6776554 , -0.11495411,  0.08314425,
  -1.6512175 , -0.58365273, -0.539351  ,  0.4077386 , -1.1948434 ,
  -0.5301088 , -0.68748116, -0.47015336, -0.71230257,  0.6020521 ,
  -1.6972171 , -1.2591417 ,  0.43660238, -1.0131292 , -2.1370835 ,
  -0.6364324 , -0.39011303, -0.83067274, -0.80258924, -0.9061281 ,
   0.00364766,  0.08194912,  0.2477529 , -1.3043734 ,  0.5426747 ,
  -0.68746257,  0.45263338,  0.98615944, -0.49591932,  0.1098897 ,
  -1.7555943 , -1.4734275 , -0.4926288 , -0.65066445, -0.8092293 ,
   0.07776835, -0.318319  , -0.67627853,  0.32620564, -1.3282478 ,
   0.3027899 , -0.8971663 , -0.5142425 , -1.8160399 , -0.23933895,
   0.43039143,  1.1933216 , -0.12663507, -1.3867764 , -0.46794185,
   1.5489651 , -0.09043343, -1.2492459 ,  0.43961367, -1.5702746 ,
   0.825108  ,  0.38898015, -0.30428824, -1.2261754 , -0.7234241 ,
  -0.2635586 ,  0.01489936, -0.61986625,  0.36995795, -1.2283008 ,
  -0.5456812 , -0.14205098, -1.9889941 , -0.38193753, -1.6477772 ,
   0.6319853 , -1.2313867 , -0.37052715,  0.59850454, -1.3823154 ,
  -0.29921493, -1.0129535 ,  0.27509263, -1.1625752 , -0.7668791 ,
  -0.78257805, -0.6201769 , -0.90472096, -1.549642  , -1.0298133 ,
   0.06435965, -0.19786873, -0.39783528, -0.40649092,  0.25457644,
  -2.5081358 , -0.22167018, -0.7942945 , -1.8623198 , -0.48292556,
   0.11279207,  0.01286425, -0.9306972 ],
 [-0.2527575 , -0.4116552 ,  1.5282912 , -0.01897023,  0.05319406,
   0.43216443, -0.30815724, -0.20365328,  0.71767193,  1.012948  ,
   1.5155994 , -0.28492436, -0.441792  ,  0.7861621 , -1.581542  ,
  -0.4643135 ,  0.8369383 ,  0.37720472,  0.25677863,  1.0650274 ,
   2.7370217 , -0.2147481 , -0.31567416, -0.2101617 ,  0.6280474 ,
   0.51181644,  0.4208337 , -0.98991036, -2.0603242 , -0.8084802 ,
  -0.2778566 ,  0.42831   , -0.00575504, -0.5438628 ,  0.5015375 ,
   0.11475407,  0.5141331 ,  0.42143467, -0.49931064, -0.69478834,
   2.037876  , -2.2854023 , -0.45224115,  0.57564867,  0.39869097,
   0.33340907,  1.5056602 , -0.40303242,  1.212727  ,  0.44347   ,
   0.15333752, -0.27730316,  0.18923885,  0.7427281 ,  0.02158109,
  -0.64812696, -0.19142775,  0.90674084, -0.12737858,  0.0192369 ,
   0.46995357,  0.89812803, -0.28388375,  0.08225683,  0.5231403 ,
  -0.3919305 ,  1.294541  , -0.7427128 ,  0.24489023,  0.30838946,
  -0.21724491,  0.6615358 ,  0.0939865 , -0.52688175,  0.08948707,
   0.41756865,  0.86704165, -1.0673386 ,  1.1907188 , -0.25513926,
   0.26624927, -1.0950475 ,  0.08225144,  0.25598654,  1.1355399 ,
   1.33833   ,  0.27695158, -0.40836525,  1.4276925 ,  0.8864458 ,
  -0.51893514,  0.9217048 , -0.321216  , -0.22764264,  0.93984544,
  -0.15767954, -0.12819919,  0.83166814, -0.23581159, -2.5369775 ,
   0.7526091 , -0.26180226, -0.04100409,  0.9624946 ,  0.57588917,
  -0.967474  ,  0.10404846,  0.2548634 , -0.7434959 , -0.99664915,
  -1.7092233 , -0.2967468 ,  0.2476833 ,  0.44513553, -0.52640045,
   0.2292099 ,  0.4256496 , -0.15615807,  0.17000994, -0.65944153,
   0.15071067,  0.04886647,  0.95524496, -0.40739357,  0.08106238,
   0.29172832, -0.08503286,  0.11216031],
 [-0.3642799 ,  0.6819667 ,  0.20347832, -0.26091403,  0.4971459 ,
  -0.32929122,  0.08511541,  0.4597884 , -0.2720535 , -0.4593408 ,
   0.32539794,  0.16253798,  0.21641204, -0.03249664,  0.35848942,
   0.23378329, -0.0114938 ,  0.7134867 ,  1.2152575 ,  0.09130909,
   0.2097716 , -0.08978028,  0.7024234 , -0.19999339, -0.11861265,
   0.6670973 , -0.14683062, -0.06304549,  0.202331  ,  0.17284092,
  -0.17722517, -0.28845945,  0.2980991 ,  0.76922584, -0.06228276,
   0.17412516,  0.22713332, -0.10440919,  2.3175137 ,  0.16047864,
   1.8203714 ,  0.0544273 ,  0.13268702,  0.0616033 , -0.30356318,
  -0.5615914 , -2.0542777 ,  0.82456225,  0.5135315 ,  0.35537967,
   0.08615783,  0.5214193 , -0.09887899,  0.25767887,  0.03914578,
   0.2321931 ,  0.6507312 ,  0.15999678, -0.05875045,  0.00092443,
   0.2940074 ,  0.3471122 , -0.24475442,  0.08585281,  0.2327335 ,
  -0.18933661,  0.82413924,  0.4892413 ,  0.3862797 ,  0.09201164,
   0.04162107,  0.47194424,  0.18477932,  0.3620754 ,  0.18854001,
  -0.20242839, -0.3246645 ,  0.838919  ,  0.09702004,  0.15362848,
   0.4974052 , -0.37330535,  0.1918499 ,  0.24005036, -0.5712942 ,
   0.23543616,  0.5226349 , -0.25007057, -0.00145347,  0.22241907,
  -0.13046864, -0.2802339 , -0.15594742,  0.00860731,  0.35300642,
   0.3809191 ,  0.37423748, -0.37899476,  0.14963499,  0.61742973,
   1.0434717 , -0.3846272 ,  0.12897709, -0.00884893,  0.3522576 ,
  -0.1406925 ,  0.86519474,  0.16269052, -0.54438055, -0.15608907,
   0.20843329,  0.43116027,  0.00510045, -0.9556716 ,  0.16516098,
  -0.46868324, -0.02869359,  0.20783195,  0.12674345, -0.29147765,
   0.01302626,  0.37129354,  0.514734  ,  0.22710428,  0.38539416,
  -0.02593309, -0.13910761,  0.14835638],
 [ 0.18149364, -0.21868023, -0.32372192, -1.034912  , -0.38929123,
  -0.6054764 , -0.5321477 , -0.40757096, -0.26831803,  0.0728373 ,
  -0.35992646, -0.48801333, -0.5585896 , -0.0516668 , -0.18758236,
  -0.59928733, -0.1917159 , -0.33737645, -0.4609613 , -0.5838943 ,
  -0.47208527, -0.60564274, -0.33514103, -0.6341437 , -0.68772316,
  -0.5497138 , -0.71307886, -0.48731223,  0.00828608,  0.28164852,
  -0.53995407, -0.3294121 , -0.056405  , -0.13182195, -0.5171098 ,
   0.15036537, -0.5772907 , -0.5312566 , -0.46853194, -0.67509174,
  -0.2857173 , -0.11563188, -0.1971845 , -0.5379993 , -0.5802194 ,
  -0.75491136,  0.19277407, -0.81826913, -0.25884947, -0.7368911 ,
  -0.46244454, -0.31995186, -0.11105449, -0.45183852,  0.28805366,
  -0.54407495,  0.11684934, -0.19478686, -0.9152111 , -0.10540594,
  -0.17063256, -0.10134804, -0.42414272, -0.23115656, -0.7393358 ,
  -0.47859988, -0.27628577, -0.26328927, -0.28515688, -0.28063333,
  -1.0495584 , -0.7110122 , -0.23114596, -0.3436807 , -0.5658417 ,
  -0.3398487 , -0.9379005 , -0.80939776, -0.9088214 , -0.19133304,
  -1.2306429 , -0.4674999 , -0.9489639 , -0.18113585, -0.50980943,
  -0.27062678, -0.31165123, -0.49158475, -0.78896856, -0.798572  ,
  -0.40662643, -0.60178936, -0.48237568, -1.1220745 , -0.7421294 ,
  -0.3051484 , -0.01157296, -0.7869784 , -0.6248128 , -0.48874107,
  -0.28233555, -0.48614118, -0.32060075, -0.6923139 , -0.5668877 ,
  -0.6082109 , -0.26882362,  0.42672968, -0.13685533, -0.71992195,
  -0.29097605, -0.2848428 , -0.56365436, -1.0684252 , -0.4953908 ,
  -0.14374559, -0.34515405, -0.55494076, -1.2632291 , -0.3267305 ,
  -0.92212635, -0.5814612 , -1.0226592 , -0.50590634, -0.64226073,
  -0.3536133 , -0.17770597, -0.63291097],
 [ 0.13912228, -0.15093085, -0.6956117 ,  0.54488206, -1.1076403 ,
  -1.5885761 , -0.6423196 , -0.7315397 , -0.44231334,  0.02833383,
  -0.3356924 , -0.4933596 , -0.5182737 , -0.12491824, -0.26982364,
  -0.6811593 ,  0.07486908, -0.44241625, -1.1679032 , -0.3186699 ,
  -0.5978648 , -0.8613531 , -0.18161887, -0.5027099 , -0.7892909 ,
  -0.58435816, -0.608346  , -0.45382255, -0.0321555 ,  0.19307539,
  -0.56520927, -0.57658833,  0.08870377, -0.21591996, -0.6059307 ,
  -0.00600012, -0.7088937 , -0.6808101 , -0.26431668, -0.5141634 ,
  -0.10503177,  0.06644444, -0.46241084, -0.30460176, -0.4697122 ,
  -0.5472803 ,  0.16499306, -0.56787115, -0.26137754, -0.74176943,
  -0.36718145, -0.3818096 ,  0.08921913, -0.17673483,  0.4760535 ,
  -0.52071387,  0.41143453, -0.3951081 , -0.48575163, -0.08925801,
  -0.1934584 , -0.06288777, -0.6102712 , -0.2641677 , -0.71762574,
  -0.5784279 , -0.15400296, -0.25381956, -0.3203083 , -0.37570757,
  -0.43811995, -0.7542237 , -0.5695375 , -0.3315915 , -0.7867915 ,
  -0.2331404 , -0.13106367, -0.3845507 , -0.58115774, -0.30465585,
  -2.2251036 , -0.49900728, -0.95084286, -0.5402818 , -0.43890163,
  -0.1532086 , -0.4026331 , -0.96156204, -1.1509782 , -0.7512531 ,
  -0.15947227, -0.8409634 , -0.8217772 , -0.6743015 , -0.84446055,
  -0.5556001 , -0.1374468 , -0.71696705, -0.8774972 , -0.5894283 ,
  -0.18444525, -0.74191535, -0.11362027, -0.48042277, -0.6560881 ,
  -0.6285966 , -0.2708807 ,  0.568619  , -0.46886587, -0.47807157,
  -0.12731065, -0.34714085, -0.6142536 , -0.9856068 , -0.41402245,
  -0.71528167, -0.40865943, -0.526091  , -0.2735065 , -0.12490849,
  -0.9320108 , -0.657296  , -0.57715285, -0.3187983 , -0.705726  ,
  -0.34466228, -0.01447627, -0.7058136 ],
 [-0.13930328, -0.0560105 , -0.4378058 , -0.10202679,  1.6249784 ,
   0.4203796 ,  0.57975495,  0.60824037, -0.01059252, -0.26965454,
   0.19820529, -0.3481383 ,  0.5466758 ,  1.4595402 ,  0.1830013 ,
   0.29398698, -0.1228118 , -1.1057346 ,  0.3870718 ,  0.70455706,
  -0.35904452,  0.53320426,  0.39973527, -0.07702962, -0.58522713,
   0.05722788, -0.09252585,  0.37260374,  1.1949009 , -0.19914319,
   0.14851114, -0.30124715,  0.11547419, -0.24274585,  0.2775342 ,
  -0.43345192,  0.7887643 , -0.22211595,  0.35875583,  0.28380057,
   0.77086747,  1.2924563 ,  0.22671884,  0.07675485, -0.03773525,
   0.6187524 ,  1.3511764 ,  0.2905664 , -0.66030645,  0.6345976 ,
   0.4578614 ,  0.54347867,  0.47464514,  0.57562673, -0.05004761,
   0.7114527 ,  0.29380918,  1.381991  ,  0.13168876, -0.08568486,
  -0.01425881,  0.02665936, -0.48479915, -0.35263965, -0.18450779,
   1.1990348 ,  0.4663862 , -0.15422766,  1.0787402 ,  0.3832413 ,
   1.0404186 , -0.00704187, -0.34046772, -0.3730514 ,  0.48882326,
   0.1941147 , -0.22671121, -0.22227031,  0.55509484, -0.35594466,
   1.7696054 ,  0.18368821,  0.24816312,  1.0983008 , -0.18497233,
   0.20224261,  0.11540957, -0.24115162,  1.0223761 ,  0.5719799 ,
   0.78971434, -0.04465967, -0.11036796,  0.78094786, -0.05688686,
   0.6257019 ,  0.23129645,  0.61922526, -0.05636242,  1.0712193 ,
  -0.08669609,  0.09471621,  0.39021093,  0.5262734 , -0.23046952,
   0.8098205 , -0.84813404, -0.2506727 , -1.5521765 ,  0.3942767 ,
   0.21481882, -0.38032055, -0.16952054,  0.45081422,  0.03307479,
  -0.35872516, -0.22478202,  0.6297594 ,  0.17954397,  0.66070926,
  -0.18680677,  0.33229697,  0.10008053, -0.25574142,  0.34467384,
   0.39892328, -0.630903  , -0.00173408],
 [-0.4316932 ,  0.02620307,  0.06254824,  0.05160601, -0.7063228 ,
   0.16234812, -0.8060733 , -0.11967263, -1.28143   , -0.26168844,
  -1.0449655 , -0.43284252,  0.08149067,  0.19442537, -0.6142698 ,
  -0.16804235, -0.35656932, -1.4136491 ,  0.8872427 , -0.8451118 ,
   0.06606124, -0.50083923,  0.28760067,  0.3661529 , -0.32739738,
  -0.51575303, -0.5807708 , -0.5967142 , -0.12016614, -0.72896177,
   0.24235843, -0.21819998,  0.15598679, -0.68986905, -0.2984665 ,
   0.28125906, -0.23995571, -0.5000781 , -0.60114795, -0.2571484 ,
  -0.6596038 , -1.1292793 , -1.3061779 , -0.5120666 ,  0.32839382,
  -0.6019911 , -1.25382   ,  0.00803893,  1.1352934 ,  0.44075432,
  -0.12004918, -0.2183244 , -0.8276135 ,  0.2740026 , -0.43029374,
   0.16391256,  0.24638158, -0.7746169 , -0.1470321 ,  0.5853484 ,
  -0.11302768, -0.01865521, -0.16289802,  0.2634523 ,  0.14966016,
   0.05444375, -0.5419551 , -0.37287706,  0.04564082, -0.04150505,
  -0.155311  ,  0.73520464,  0.6185949 ,  0.06328214, -0.42667213,
   0.07620346, -0.11081661, -0.0900515 ,  0.4507138 ,  0.11921827,
   0.00098381, -0.24918187, -0.8423313 , -0.4812758 ,  0.7117328 ,
  -1.0226367 , -0.07710376,  0.27533045,  0.18568392, -0.02879023,
   0.5391243 , -0.04113756, -0.21546577,  0.9194222 ,  0.12706739,
   0.21245055,  0.32299176, -0.39012232,  0.2599639 ,  0.33441547,
  -0.651419  , -1.360692  ,  0.23926075, -0.0155871 , -0.47021616,
  -0.16479221, -0.7184389 ,  0.16565908, -0.72935706, -0.05430979,
  -0.09414054,  0.30586782,  0.33368802, -0.7346752 , -0.3651066 ,
   0.80614525, -0.24311621, -0.1703208 ,  0.09135777, -0.44592154,
  -0.63594806, -0.2534207 , -0.14022553, -0.62598294, -0.10400164,
  -0.4280124 ,  0.04573108, -0.41265115],
 [-0.46405664,  0.43411994,  0.2979079 , -0.13218917, -0.1860899 ,
  -0.5519774 , -0.39898542, -0.3970776 , -0.853312  ,  0.578717  ,
   0.05736905,  0.19091585, -0.25960174,  0.66650975,  0.20024721,
  -0.39064094,  1.472709  ,  0.64947885,  1.3012738 , -1.1340358 ,
   0.19163929,  0.10334002,  0.49311268, -0.24812846, -0.78178376,
  -0.00180003, -0.18591896, -0.6839121 , -1.6110399 ,  0.181375  ,
   0.19118288,  0.5678988 , -0.27520692,  0.1315873 , -0.23212448,
   0.19936496,  0.48600823,  0.6681977 ,  0.18405479,  0.02023795,
   0.40339768, -0.03741794,  0.24162029, -0.30768567, -0.5591771 ,
   1.254975  ,  1.5119026 ,  0.4297569 ,  1.233308  , -1.0829434 ,
   0.27172536,  0.695813  ,  1.8041285 ,  1.6332467 ,  0.10337656,
   0.18943459, -1.1817183 , -0.04586469,  0.1956944 , -0.3951161 ,
  -1.7364608 ,  0.17587577, -0.6768586 ,  0.09276304, -0.34543303,
   0.35774103,  0.23951499,  0.5037554 , -0.23662314,  0.86842656,
  -0.18252379,  0.73077875, -0.7771992 , -0.55827683, -1.0022635 ,
  -0.81164813, -0.87396663,  0.8584223 , -0.91622984, -1.0333177 ,
  -0.48940364, -0.80440104,  0.60717124,  0.3753921 ,  0.39893058,
  -2.3285327 , -0.32123348,  0.24061356, -0.2231426 ,  0.6040457 ,
  -0.9898528 ,  0.05349471, -0.28526533, -0.56445295,  0.25555685,
  -0.31829268,  0.927262  ,  0.3034353 ,  0.19660404, -0.6931617 ,
   1.1829424 , -0.99106777,  0.10118139, -0.37176988, -0.32423538,
  -0.06068446,  1.9305887 ,  0.13540767, -0.19142774,  0.24192569,
   0.5058728 , -0.5796418 , -0.11328963, -0.42556763, -0.7027039 ,
   1.9636122 ,  0.95362693, -0.16595076, -0.31444797,  0.2335011 ,
  -0.15588099,  1.062031  , -0.2865783 ,  0.38114488,  0.17987198,
   0.07309821,  0.26879102,  0.459173  ],
 [-0.30129626, -0.19352078,  0.40897697,  0.12542422, -1.3868978 ,
  -0.3464131 ,  0.05396156,  0.58804363, -0.4747034 , -0.5072503 ,
   0.05213298, -0.06838647, -0.33819574,  0.28150672, -0.729268  ,
   0.05138081,  0.1592509 ,  1.5835081 ,  0.73155403,  0.3449113 ,
  -0.8329241 ,  0.27158558,  0.02417018, -0.6480792 , -0.3737958 ,
   0.24150184, -0.051223  , -0.33944976,  0.67427367,  0.2218149 ,
  -0.00851572, -0.11117323, -0.3215012 ,  0.5070931 ,  0.4719111 ,
   1.1445768 ,  0.2037703 ,  0.02776188, -0.04758178, -0.40781724,
  -0.27523196,  2.1847382 ,  0.03672764,  1.085071  , -0.57174486,
  -2.0622678 ,  1.0745429 ,  0.24168068,  1.1187993 ,  0.2626551 ,
  -0.2694186 ,  0.7061866 , -0.07561474,  0.09222823,  0.16380756,
  -0.38018215, -0.0253892 , -0.44433555, -0.63055086, -0.4266201 ,
  -0.63503945, -0.2405611 ,  0.6765647 ,  0.33120993,  0.3735617 ,
  -1.449587  , -0.21740797,  0.5626717 , -0.6485667 ,  0.4099209 ,
   0.42839745,  0.10661419, -0.05592031,  0.20221698, -0.5645562 ,
   0.23732558,  0.12980755,  0.21003713,  1.0250534 ,  0.01064727,
  -0.28684106,  0.21171188,  1.080869  , -0.14579561, -0.00132765,
   0.49927345, -0.5237031 ,  0.42068377, -0.30496195,  0.452063  ,
   0.7811906 ,  0.37767982, -0.23011369,  0.27763402, -0.12585047,
  -0.07321427, -0.3441264 , -0.45636445, -0.09683391,  0.30559   ,
  -0.37701836,  0.45586082, -0.11550841,  0.21651874,  0.05436971,
   0.00923619,  0.10304867,  0.1443799 , -0.03769852,  0.19199029,
   0.5863392 ,  0.12738174, -0.05046092,  0.06574498, -0.41071078,
  -0.7015168 ,  0.46331775, -0.700563  , -0.14247602,  0.18049395,
  -0.24290171,  0.16247305,  0.1297167 , -0.5475059 , -0.3395353 ,
  -0.21614651, -0.8838211 , -0.92246944],
 [-1.028326  , -2.0233564 , -1.3251176 , -0.24257323, -0.0312397 ,
  -0.29670754,  0.48436826,  0.7341976 ,  0.04950063, -2.9115689 ,
   0.771517  , -1.1556377 ,  0.48425147, -0.37903652, -0.4091361 ,
   0.11814771,  0.96681625,  2.6815572 ,  0.53002584, -0.5221291 ,
  -0.5950179 , -0.06200447,  2.142973  , -0.6865267 , -1.3180906 ,
  -0.77029955,  0.43130782, -0.38970435,  0.6545468 , -1.2272207 ,
  -1.0102348 , -0.37277713, -0.25039408, -0.18183525, -0.00353612,
  -1.0111983 , -0.04790065, -1.8513087 , -0.2134361 ,  0.7432396 ,
   0.61474514,  1.33501   ,  0.1474963 , -0.9957127 , -1.1630723 ,
   0.18958235,  2.3031602 , -0.99566865,  0.8152342 , -0.0610485 ,
  -1.0052278 , -0.4675139 ,  1.0180317 , -0.3251501 ,  0.28449506,
  -1.1963999 ,  0.9258075 , -0.2940801 ,  0.5500711 , -0.5504924 ,
  -1.0371392 ,  0.5684797 , -1.1088455 , -2.2872725 , -0.8335969 ,
   0.18723209,  0.5106638 ,  0.29010445, -0.33267665, -1.6024383 ,
  -0.34810153,  0.12776184,  1.3186342 , -0.06305458, -0.18077126,
  -0.4120238 , -0.8075072 , -0.26401   , -0.6133804 , -0.84999037,
   0.467316  , -1.5904323 , -1.1475178 , -0.04882256, -0.55228543,
  -0.10534125, -0.06755885,  0.9909671 , -0.89750165,  0.17020267,
  -0.674234  , -0.46942177, -0.9276884 , -0.92799395, -0.77370006,
  -1.22644   ,  0.31499112, -0.9152278 , -1.84998   , -0.33838278,
   0.12521076, -0.33290753,  0.02203159,  0.38674653, -1.0482703 ,
   0.20008622, -0.37408513, -1.3044971 , -1.1845046 , -0.56907225,
   0.00904181, -0.10855228, -1.138087  , -1.44653   , -1.328529  ,
  -1.8575535 , -1.0237865 ,  0.1511609 ,  0.24174468,  0.14181124,
  -0.1314119 , -0.8176308 , -0.3489488 , -2.4348927 ,  0.12461267,
   0.12063719, -0.5618388 , -0.28168058],
 [ 0.35271022, -0.48324788, -0.74559635,  1.5530729 ,  0.28819928,
  -0.18102367, -0.09494471,  0.7622098 , -0.02081674, -1.0445017 ,
   0.09738332, -0.8061103 , -0.23292015, -0.713083  , -0.7548041 ,
   0.04164511,  1.6884589 , -0.26455677,  0.19120477,  0.10722218,
  -1.5888085 , -0.23851816,  1.066271  ,  0.16663633, -0.6887529 ,
  -1.161435  , -0.16716631, -0.47405794,  1.2059164 , -0.0664719 ,
  -0.922565  , -0.72289497,  0.2624249 ,  1.3922839 , -1.7064893 ,
  -0.19628929, -1.0362434 , -1.6502922 ,  1.2992002 ,  0.53696936,
  -0.11288363, -0.39382723, -1.0756233 , -0.06294328,  0.60293514,
   0.35929492,  0.69524354, -0.10728323, -0.9409343 ,  0.1878304 ,
  -0.8711774 , -0.8678009 ,  1.4725924 , -0.06126492,  2.9652803 ,
  -1.6965189 ,  3.4869874 , -0.11071716, -0.40973195, -0.6939302 ,
  -0.42453867,  1.7839615 ,  0.6163353 ,  0.01202016, -1.5010868 ,
   0.16696467,  0.08912558, -0.633333  ,  0.8563838 ,  0.30170557,
   0.11045393, -1.0292016 , -0.10754296,  0.8230543 , -0.38810423,
  -1.3048316 ,  0.04083439, -0.06103671,  0.1393672 ,  0.34974906,
   1.5268384 , -0.38120848, -1.1411387 , -0.18409039, -0.34505785,
  -0.47086954, -0.88953465,  0.36327216, -0.3176304 , -0.31322345,
   0.170137  , -0.11684582,  0.58315116,  0.24548641, -0.6195554 ,
  -1.0348744 , -0.15225853, -0.57901627,  0.3602188 , -0.0521109 ,
   0.00084175, -0.3689949 ,  0.73548687,  1.373955  , -0.5114181 ,
  -1.7216843 ,  0.88011575, -0.5604787 ,  0.9554992 , -0.02752231,
   0.28720555, -0.47497025,  0.5662713 ,  0.15197438, -0.08277674,
   0.33029705, -0.7582631 , -0.17789952,  0.19168979, -0.22124696,
   0.6333579 , -1.2151757 ,  0.33090928, -0.7367423 ,  0.23638876,
   0.26393467, -0.72462183, -0.46830818],
 [-0.3128436 , -0.73578304,  0.20319352,  0.30009463, -0.57651335,
   0.10909474, -0.54920346, -0.84003586,  0.3426502 , -0.10224453,
   0.7718201 , -0.07748874, -0.9201723 , -2.1214602 ,  0.30116978,
  -0.24729694, -0.38394907,  1.6018976 ,  0.3193489 , -0.20822506,
   1.4941021 ,  0.5498147 ,  0.22639816,  0.00745715,  0.66183805,
   0.7058602 ,  0.24421829, -0.5732181 ,  0.1962155 ,  0.7898896 ,
   0.35463288,  0.1408685 , -1.3315951 , -0.60718846,  0.19508126,
   0.15431656,  0.9471122 , -0.05771825,  0.13642636,  0.14661057,
  -0.21295476, -0.88053495, -0.31006593,  0.03226738, -0.4899105 ,
  -0.3870172 ,  0.88246244,  0.64185447,  0.13654292, -0.5664774 ,
  -0.15575752, -0.74116606,  0.6352267 ,  0.54256797,  0.50140196,
  -0.4271548 ,  0.2468215 , -1.0703818 , -0.04612391, -0.16991444,
   1.1762289 ,  0.47009403,  0.06435307, -0.670993  ,  0.05876612,
  -0.22088346, -1.2209222 , -0.06381585, -0.19568439, -0.07350985,
  -1.1232401 , -0.74613553,  0.2721062 ,  0.11166995, -1.0454093 ,
  -0.01074952, -0.02642117,  0.05563675,  1.1012458 ,  0.04970192,
  -0.15545553, -0.20052607,  0.10264762, -0.70808065,  0.15303293,
   0.08976807, -0.6067388 ,  0.01906375, -0.4031461 ,  0.32430327,
   1.0546457 , -0.14305027, -0.00379905, -1.3791271 ,  0.00807135,
  -0.24338669, -0.22532117,  0.801487  , -0.11208176, -0.16793351,
   0.61266524,  0.28516483, -0.26773405, -0.6186981 ,  0.9789991 ,
  -0.4969487 , -0.05523393,  0.02109953,  0.31317958, -0.71419954,
  -0.4053994 , -0.03517755, -0.00751904, -0.2852985 ,  0.34099445,
  -0.13516885,  0.90169466, -0.6594582 , -0.81049746, -0.60770935,
   0.38363907,  1.1764799 ,  0.7676718 , -0.67782676,  0.1097745 ,
  -0.48877874, -0.39672598,  0.77231276],
 [-0.17235248,  0.4170252 ,  0.03030423, -0.03540597, -0.51259923,
   0.8106073 , -0.24328193, -0.1951516 ,  0.5846801 , -0.25516626,
  -0.16052759, -0.6854323 , -0.4606856 , -0.83679295, -0.9008988 ,
   1.6281066 ,  0.52151185, -2.8243842 , -0.21446565, -0.47109592,
  -0.7693922 , -0.36468267,  1.4996386 ,  0.36523035,  0.58426744,
  -0.07847872, -0.49908435,  0.47172835,  0.3641564 , -0.92591363,
  -0.00334791,  0.40191838, -0.0359001 , -0.62383705,  0.45601025,
  -0.42757994,  0.25456676,  0.13219845, -0.7228955 ,  0.43516746,
  -1.3630291 , -5.898059  , -0.86419857, -0.27942312, -0.39641562,
   0.09437617, -0.29240566, -0.40675917, -0.9860216 ,  0.58444095,
  -0.50590765, -0.93410754, -0.43412548, -0.60166687,  0.6745    ,
  -0.10737661,  0.02040238, -0.23906825,  0.3893096 , -1.6806306 ,
   0.49835795,  0.41483808, -0.7295344 , -0.97910625,  0.15324843,
   0.58987206,  0.9438869 ,  0.0902141 , -0.99824554, -0.01188192,
  -0.13623264,  0.27998963,  2.136459  , -0.8169811 ,  0.01031273,
  -0.5592233 , -0.18265764, -0.37142056,  0.38494465, -0.42848346,
   0.44580343, -0.05194454, -0.1239817 , -0.18650833, -0.2182696 ,
   0.01113711, -0.05932362,  1.3849734 , -0.73278743, -0.23136696,
  -0.3101953 , -0.3803459 ,  0.6434884 ,  0.70474035, -1.3076077 ,
  -0.5472229 , -0.453249  , -0.31339237, -0.07037196,  0.19884102,
  -0.7119265 ,  1.5156021 ,  0.8331653 ,  1.1784725 ,  0.16147695,
  -1.3954914 , -0.18514529, -1.5756515 , -0.5684332 , -0.5260845 ,
  -0.48023918,  0.4090686 ,  0.3500608 ,  0.0198082 ,  0.3548357 ,
  -0.35944015,  0.03917356, -0.50053537, -0.04991673,  0.35049954,
  -0.09598291, -0.35301462, -1.3294014 , -1.9584154 , -0.19402835,
   0.27028355, -1.717138  ,  0.26781547],
 [-0.03417511, -0.02425843, -0.59129447, -0.20969248,  0.05056144,
   0.3410469 ,  0.19224949, -0.4424714 , -1.6086823 , -0.671695  ,
  -0.1785721 ,  0.23858368, -0.01078201, -0.19686696,  0.4736579 ,
   0.42297053, -0.49129823, -0.28599572,  0.18489555, -0.21031837,
   0.26410514, -0.0650176 ,  0.15990104,  0.00162112,  0.7461118 ,
   0.3361305 , -0.37981892,  0.29560336, -0.17321232, -0.4665633 ,
   0.07489187,  0.19683295, -0.31864548,  0.28413886,  0.593883  ,
  -0.26231623,  0.02796614,  0.42199743, -1.1638014 , -0.01457993,
  -0.08282067, -0.12419977, -0.6136844 , -0.34861508,  0.48953912,
   0.64188474,  0.47632894,  0.02377819, -0.06241828,  0.41034076,
  -0.8865774 , -0.11706839, -0.6127468 , -0.4510087 , -0.2569072 ,
   0.0846285 ,  0.41271505, -0.4125871 , -0.04891803, -0.4965302 ,
   0.23758069, -0.6868041 , -0.52882946,  0.55471826,  0.47076204,
  -0.58247644,  1.1588318 ,  0.6452559 , -0.20609137, -0.2695494 ,
  -0.03759497,  0.47464308, -0.8622818 ,  0.1298059 ,  0.08701629,
  -0.40951514,  0.09895269,  0.26422283, -0.211854  , -0.6786177 ,
  -0.25414512, -0.458191  ,  0.9053802 , -0.480763  ,  0.18181734,
   0.8726201 , -0.3253211 , -0.1956623 , -0.5755431 ,  0.8436053 ,
  -0.328879  ,  0.29701766, -0.48577222,  1.1964653 , -0.14057277,
   0.07584988,  1.2086179 ,  0.5489495 ,  0.442118  , -0.7489825 ,
  -1.2949096 ,  0.18975854,  0.01105131,  0.56399643,  0.17718446,
  -0.43868172,  0.27085984,  0.06525045, -1.6972177 , -0.02361218,
  -0.19465919, -0.8248179 , -0.20987825, -0.3269834 ,  0.50680894,
  -2.110747  ,  0.25333276, -0.03901194, -0.12690958, -0.6990445 ,
  -0.021747  , -0.7434814 ,  0.16579056,  0.6297161 , -0.1429727 ,
   0.28531834,  0.03943934,  0.42832583],
 [-0.6567232 , -0.471822  ,  0.1353129 ,  0.30172026, -0.43367848,
   0.38702613, -0.49978903, -0.28166974,  0.11121085, -1.0510112 ,
  -0.7384252 , -0.25161043, -1.1391257 , -0.15729673, -0.6030388 ,
   0.13896032,  0.3246738 , -0.14803198,  0.07265195,  0.17191531,
  -1.2011286 ,  0.02423408,  1.3263161 , -0.04535965, -0.3576477 ,
   0.16072233, -0.344832  ,  0.00739449,  1.1565758 , -0.57272166,
  -1.0528848 , -0.7631327 , -1.1704024 , -0.44638276, -1.1847913 ,
  -1.1306101 , -1.2481707 , -0.23164801, -0.75590664,  1.0949826 ,
  -0.33961073,  0.14697155, -2.2462468 , -0.4452848 , -0.65087944,
  -0.5182201 ,  1.5093443 ,  0.10403786, -1.9595392 , -0.6985111 ,
  -1.317675  , -1.5442753 ,  1.0470464 , -0.7609658 ,  2.9096055 ,
   0.04034664,  2.4515455 , -0.53146166, -1.3080007 ,  0.77751625,
  -1.289869  ,  1.0389243 ,  0.95514673, -1.056174  ,  0.3679223 ,
   0.6150273 , -0.07453518,  0.28973395,  0.16468272, -1.1075383 ,
  -1.0414835 , -0.1089116 ,  0.5961392 ,  0.4586195 , -1.3328552 ,
  -2.0751934 , -0.18018101, -0.07622769, -0.82276464,  0.4030804 ,
   1.4993516 ,  0.21938021, -1.0129964 , -1.1542153 ,  0.27990538,
   0.10374668, -1.1992642 ,  0.64040893, -0.14580992, -0.4927138 ,
   0.42300385, -0.4248297 ,  0.9091596 ,  0.0494478 , -0.5367034 ,
   0.4109494 , -0.05610648, -0.87772214,  0.26280653, -0.60058755,
   1.3138325 , -0.27648592,  0.6232505 ,  0.89311695, -0.5432072 ,
  -0.4437127 ,  0.7844132 , -0.8401381 ,  0.6626257 ,  0.15182629,
  -0.10043721, -1.3385605 ,  0.5335907 , -0.57245845,  0.28279522,
   0.10302502,  0.37092814, -0.09671356, -0.44983146,  0.85572845,
   0.00356777, -1.3699635 , -0.6282233 ,  0.12370063, -0.6888708 ,
  -0.2302373 , -0.7534279 , -0.7229644 ],
 [ 0.57526124,  0.07697484,  0.37319374, -0.49865058, -0.8523601 ,
   0.1537425 , -0.38198093, -0.4984301 , -0.34618175, -0.12696743,
   0.8800696 , -0.49905133, -0.9203702 , -0.8913009 , -0.31722283,
   0.25627074,  0.03351344, -0.9260391 , -0.552201  , -0.00673019,
  -0.28354198,  0.41378805,  0.572692  , -0.8820721 , -1.603515  ,
  -0.71000695,  0.5181865 ,  1.4294285 ,  0.35489887, -0.5801274 ,
   0.09263348, -0.52198696, -0.34717697, -0.25375652, -0.24225643,
  -1.0112877 , -0.18136811, -0.6550738 ,  0.44754001,  0.43524095,
   0.176541  ,  1.01925   , -0.04806106,  0.5378454 , -0.06401747,
  -0.24991754,  2.0965395 ,  0.13356651,  0.03797997,  0.58348554,
  -0.3439227 , -0.55487406,  0.68240756,  0.3142241 ,  0.69153976,
   0.50436944,  0.1033582 ,  0.7934704 ,  0.28399143, -0.75350386,
   0.0297513 ,  0.7267714 ,  0.00286913, -0.2082998 , -0.10251014,
  -0.31358993,  0.8338137 ,  0.23550078, -0.83755344,  0.2407333 ,
   0.1661862 , -0.7000946 ,  0.12937544,  0.13268694, -0.4944801 ,
   0.23429549, -0.4709527 , -0.3320496 ,  0.0726111 , -0.14339519,
  -0.02862418, -0.11132922, -0.28887352, -0.7821307 ,  0.3213597 ,
   0.49541134, -0.29202652,  0.13239793, -0.09607591, -0.0949232 ,
  -0.7730596 , -1.3283973 ,  0.27380073,  0.95529085,  0.46829826,
   0.557471  , -1.4164573 ,  0.14538084,  0.35621306,  0.621195  ,
   2.0970247 , -1.2490823 , -0.2466032 ,  0.74384487, -0.7902933 ,
  -0.1732125 , -0.20940909, -0.17343271,  0.65896803,  0.7146956 ,
  -1.0168774 , -0.36665338, -0.01437723,  0.17492904,  0.35171637,
   0.49356169, -1.0436382 , -0.02845939, -1.4443251 ,  2.0212543 ,
  -1.1582682 , -0.51169163,  0.34464908, -0.7700644 ,  0.03434595,
  -0.17357565,  0.00525231, -0.33024222],
 [ 1.1529905 , -0.34009305, -0.3074197 , -0.45029402, -0.02429199,
  -0.07498064,  0.7238834 ,  0.578322  ,  0.5244573 ,  0.372587  ,
  -0.17711206, -0.609089  , -0.19450366, -0.12538642, -0.04272449,
  -0.00483388, -0.41064876, -1.7118437 , -0.26378453, -0.3152044 ,
  -1.762076  , -0.7366691 ,  1.1154199 , -0.7026184 , -0.16407847,
  -0.40505424, -0.91648793, -0.5807263 , -1.0195289 ,  0.26501372,
  -0.36259785, -0.35395885, -1.6197786 , -0.02654048,  0.61126494,
  -0.15091275, -0.58225405, -0.27706617, -0.42013037,  0.21048424,
  -0.34990728,  0.39110592, -0.9080242 , -0.43236506,  0.09748831,
   0.16125235,  0.36127555, -0.11240972, -0.9288498 , -0.2990946 ,
  -0.4488947 , -1.4174323 ,  0.48668015,  0.09463257, -0.06079924,
   0.77767164,  0.25271013, -0.71630687, -0.4385931 , -0.64994335,
   0.6869604 ,  0.17694883,  0.5584197 , -0.18424833,  0.924571  ,
  -0.37608665, -0.377982  , -0.7625116 , -0.27556267, -0.7911608 ,
  -0.296688  , -0.5768189 ,  0.6313434 ,  0.11400781,  0.04405376,
  -0.3409978 , -1.5758399 , -0.25175214, -0.91229886, -0.31630883,
   0.09456325, -1.0028471 , -0.6151669 , -0.41741955, -0.55615103,
   0.35053697, -0.62810093, -0.16075613, -0.5853483 , -0.09084084,
  -0.9671815 , -0.6070982 ,  0.4093168 , -0.16338666, -0.9274948 ,
  -0.5948517 , -0.27495733, -1.1155049 ,  0.7648102 ,  0.6917432 ,
   0.49950755, -0.65777475, -0.1343997 ,  1.293915  , -1.4578142 ,
  -0.7365711 , -0.21478422, -0.39202598,  0.8048973 ,  0.03930671,
  -0.45807126, -0.17333192, -0.09031799, -0.01949999, -0.39482895,
  -0.14490952,  0.40926445, -0.5423329 ,  0.27239808,  0.45534554,
  -0.52222073, -1.1907511 ,  0.2240539 , -0.06595176, -0.4929212 ,
  -0.22782806, -0.7798711 , -1.0433424 ],
 [-0.07270997,  0.57289064, -0.50134265, -0.08827125,  0.7228733 ,
   0.17011514, -0.2587928 , -0.26627192, -0.5157629 , -1.1437564 ,
  -0.9482404 ,  0.09539503, -0.09282705, -0.32162145, -0.318582  ,
   0.22291505,  0.9372363 , -0.5993073 ,  0.32195973,  0.4008935 ,
   0.8369043 , -0.49497432,  0.18306518,  0.13981584, -0.09890793,
  -0.32092944, -0.02049227, -0.3274667 , -0.43994132, -0.7182475 ,
  -0.8155546 ,  0.3919734 , -0.41033977,  0.59552515,  0.05324192,
  -1.8314488 , -0.717801  , -0.13842303, -0.5150028 , -0.09676167,
  -0.12956972, -0.7032028 ,  0.17072424, -0.06037863,  0.67326903,
  -0.38661796,  1.0342729 , -0.15898493,  0.93785733,  0.2553681 ,
   0.01897055, -0.6995236 , -0.31995317, -0.05685932,  0.1455058 ,
   0.59831524,  0.69341505, -1.4282808 , -0.9815413 ,  0.17807922,
  -0.6646165 ,  0.05546226, -0.7852662 , -0.14044048,  0.04285292,
   0.21878651, -0.6349914 , -0.37010565,  1.2486144 , -0.8332568 ,
   0.08944681, -0.3937904 , -0.72887117, -0.08872312,  0.36313522,
   0.7318746 , -0.79735076,  0.44016773, -0.12636407, -0.14041248,
   0.40304312,  0.6307391 ,  0.05681739, -0.28532928,  0.76250374,
   0.5990578 ,  0.7299676 , -0.37555856, -0.16236193, -0.26536006,
  -0.09603583,  0.13907248, -0.5354317 ,  0.17572   , -0.32201868,
  -0.32240796, -0.17435302,  0.24143162, -0.21687683, -0.13838986,
  -0.9865567 ,  0.00955721,  0.14904591,  0.33314466, -0.30941868,
   0.10304558, -0.19221379,  0.375844  ,  0.27021357, -0.15406929,
  -0.7403676 , -0.3786357 , -0.01268378, -0.34300116,  0.7617837 ,
  -1.9279342 ,  1.1639961 , -0.04625384, -0.36743358,  1.0567398 ,
  -0.32707694, -0.18528402, -0.53537273, -0.46450448,  0.11704263,
  -0.06920215, -1.1442063 ,  0.51931703],
 [ 0.1805627 ,  0.5282809 , -0.2989716 ,  0.12983873,  1.945441  ,
  -0.28184113,  1.3456328 ,  0.31639007,  0.04992395,  0.5790522 ,
   0.02072972,  0.67581195,  0.70513785, -0.03560153, -0.33718124,
  -0.13519295, -0.9116633 ,  0.124981  ,  2.0474892 ,  0.48998514,
  -0.1949718 ,  0.42019904,  0.48198152, -0.23254752, -0.25641635,
  -0.03480544,  0.17166299,  0.26491332,  2.0031037 , -0.38556188,
  -0.31467107, -1.1274483 , -0.3836928 ,  0.9081244 ,  0.36395115,
  -0.30151367, -0.34380114, -0.41547453,  0.5129    ,  0.16980338,
   0.66231054,  1.5599228 ,  0.4298499 ,  0.10546452, -0.2342875 ,
   0.4122389 , -0.07955337,  0.39015564,  0.6936161 ,  0.7237757 ,
   0.30419466,  0.5262773 ,  0.34734228,  0.07319389,  0.05148204,
   0.7225151 ,  0.39759025, -0.5110417 ,  0.4066243 , -0.04548488,
   0.3685296 , -0.1143304 , -0.3233565 ,  0.94994015,  0.29504848,
  -0.54278576,  0.74339664,  0.18848473,  1.5648397 , -0.03541812,
  -0.7054291 , -0.13198793, -0.45063862,  0.55199355,  0.28549355,
   0.06795254,  0.08570922,  0.9929434 ,  0.14048885, -0.10995157,
  -0.28634223, -0.01329767, -0.03179583,  0.81644076,  0.1522082 ,
  -0.87688756,  0.41374132, -0.15680793, -0.7498774 ,  0.45886546,
  -0.52032447,  0.40107328, -0.43965158,  1.0208561 , -0.06076518,
  -0.31479448,  0.26629323,  0.330046  , -0.14098868,  0.11698426,
  -0.65923405,  0.26010272,  0.77926075,  0.24346706, -0.08683396,
   0.1775304 ,  1.5580575 ,  0.1129063 , -0.18146394,  1.082479  ,
   0.15766744, -0.550888  ,  0.08935614,  0.13554388, -0.4921488 ,
  -0.72338474,  0.83070433,  0.10470895, -0.18226236, -0.6136896 ,
   0.5550398 ,  0.5994705 , -0.20179181,  0.5394921 , -0.06165339,
   0.09078996,  0.10200561, -0.02870496],
 [-0.04148341,  0.40783066,  0.19391912,  0.33269194,  1.6871665 ,
   0.34356195,  0.90476215, -0.37237224,  0.6847615 , -0.5547381 ,
   0.33519953, -0.48696634, -0.3356676 ,  0.02591157,  0.21395878,
   0.23602821,  0.2005572 ,  0.14484258,  0.79991895, -1.1259829 ,
   0.05232178,  0.6686526 ,  0.38222402, -0.1731336 ,  0.16360873,
  -0.12083848,  0.02183854, -0.6494941 , -0.56020963,  0.42228308,
   0.70897615, -1.460245  ,  0.9735431 ,  0.524186  , -0.6413459 ,
  -0.27376482, -0.14950429, -0.9368488 , -1.5819999 , -0.18836612,
  -0.39989248,  2.734913  , -0.30923912,  0.7223908 , -1.0297736 ,
  -1.1535097 , -1.4100232 ,  0.33405522,  0.8866091 ,  0.22285777,
  -0.52679175, -0.515801  ,  0.5182995 ,  0.01995882,  0.22375299,
  -0.49388564, -0.40618545,  0.16850969, -0.6017424 , -0.30565444,
  -0.8300582 , -0.11572205, -0.48187405, -0.4091328 , -1.2867813 ,
  -0.26918706, -0.56329775,  0.6421381 ,  0.06372844,  0.74034786,
   0.09809877, -0.41955698,  0.08013511,  0.08869689, -0.06225086,
   1.0178889 ,  0.47188208, -0.22232501,  0.87683773, -0.68820447,
   0.08135261, -0.24964492,  0.24931732, -0.21498504, -0.5952741 ,
  -0.7909113 , -0.02138469, -0.55772805, -0.33817843, -0.8181856 ,
  -0.4172647 ,  1.2437962 , -0.54981357, -1.5937688 , -0.02057928,
  -0.50367993,  0.61859494,  0.26338273, -1.3217622 ,  0.3659422 ,
  -0.62521946, -0.6123829 ,  0.28105244,  0.33259702, -0.30961254,
  -0.20036626,  0.83199143,  0.27255055, -0.09723103,  0.43477973,
   0.9198418 , -0.03386756, -0.18155168,  0.54389966, -0.5741496 ,
   2.302092  , -0.11263726, -0.28437313, -0.2708601 , -0.6532327 ,
   0.5299902 , -1.0214038 ,  0.7865688 ,  1.425664  ,  0.00043068,
  -0.47082686,  0.02198932, -0.2726375 ],
 [ 0.37437358, -0.327808  ,  0.62310785, -0.15170163,  1.0307752 ,
  -0.3415267 , -0.33092588,  0.3681308 , -0.31589285, -0.6149323 ,
  -0.37370938,  0.74624074,  0.314786  , -0.38363993,  0.2241255 ,
   0.19665815,  1.4845946 , -0.82407975,  0.62532294, -0.3414282 ,
   0.8661131 ,  0.05504795,  0.11130492,  0.14539438, -0.25695813,
   0.4944107 , -0.5421395 , -0.8844839 , -0.8365763 , -0.03955805,
  -0.2878203 ,  0.08214962, -0.45652723,  0.5073818 , -0.21841137,
   0.49281326,  0.3299738 , -0.16946582,  0.8079499 ,  0.04858946,
  -0.63105863,  0.2763623 , -0.6016741 ,  0.41487506, -0.146005  ,
  -0.83302355,  1.1806914 , -0.8300626 ,  0.0152856 , -0.04569972,
  -0.5770812 , -0.05962411, -1.767674  , -0.00661062,  0.44050348,
   0.46308675, -0.20101938,  0.35313597, -0.4656705 , -0.12010337,
  -0.39600065, -0.13755426, -0.19267698,  0.20188823,  0.996757  ,
  -0.46940213,  0.21012884,  0.56467324, -0.05164649, -0.28853464,
  -0.2784477 ,  0.9470715 , -0.21804582,  0.17191733, -0.37966266,
   0.7450146 ,  1.3549542 ,  0.1464705 , -0.18615696,  0.10154833,
  -0.1346749 , -0.42405885, -0.21690516, -0.7075268 , -0.49457887,
   0.27469608,  0.6975294 , -0.05735961,  0.6498746 ,  0.902294  ,
   0.7424114 , -0.62432265, -0.43890798, -0.73961985, -0.03518726,
   0.8538767 , -0.19828144,  0.04926359, -0.3765646 , -0.21400134,
  -0.2952556 ,  0.53483295,  0.15580484,  0.16935585,  0.04572913,
  -0.42791623, -0.45359963,  0.39104146,  0.22573987, -0.84352815,
   0.36784944,  0.11234719,  0.02504635,  0.32834598, -0.2596715 ,
  -0.4009831 ,  0.6314741 , -0.15416381,  0.34120405,  0.411676  ,
  -0.46000603,  0.04047317,  0.37232372,  0.61287946, -0.17906679,
   0.37429857,  0.29877636,  0.8517551 ],
 [ 0.07431848, -0.25636786, -0.26153472,  0.44825026, -0.02130674,
   0.32528076,  0.51801366,  0.44770777, -0.13744603, -0.33179274,
   0.3389564 ,  0.34350684,  0.99789715,  0.3674385 ,  0.11390986,
   0.67274386,  0.7664414 ,  0.2465797 , -0.35090816,  0.5639485 ,
  -0.4714423 ,  0.46329424,  0.21546336,  0.41226482,  0.19535974,
   0.00362504,  0.41559365, -0.16979127,  0.92186415,  0.0344219 ,
  -0.17304514,  0.41328225, -0.04695506, -0.36208138,  1.4427031 ,
  -0.5864152 , -0.45868063,  0.07335562,  0.89599764,  0.04469515,
   1.2953565 , -2.084677  , -0.42249474,  0.25060105, -0.17393363,
  -0.6273502 , -0.05868122,  0.5619775 ,  0.14740098, -0.56939834,
  -0.5623775 ,  0.29059982,  0.05753168, -0.65158737,  0.12673216,
  -0.42244846,  0.20615081,  0.88768166, -0.33978352,  0.39303312,
   0.72325945,  0.5381089 , -0.11783089, -0.933852  ,  0.21595673,
  -0.34517634,  0.3557635 ,  0.80998236,  1.812743  ,  0.86687005,
  -0.4576925 , -0.26589715,  0.3536358 , -0.48463956,  0.67970675,
   0.22560166,  1.0593712 , -0.41882563,  0.59924483, -0.2502676 ,
   0.91477615, -0.35436308,  0.33239982, -0.11312275,  0.5119745 ,
  -0.16820039,  0.17023508, -0.15110536,  0.41542977,  1.0594454 ,
  -0.26690826, -0.41518763, -0.34653816,  0.34296757,  0.37978747,
  -0.14561424,  0.7940379 , -0.24836366,  0.16626176,  0.48977238,
   0.02529373, -0.476163  ,  0.88447464, -0.10253297,  0.15435162,
   0.52449274, -0.16249429, -0.03254924, -0.15154293,  0.4727848 ,
  -0.7343148 ,  0.5742781 , -0.11157168, -0.00941883, -0.06205522,
   0.36585546, -1.0485077 ,  0.40496776,  0.04968873, -0.62764186,
  -0.8746983 , -0.13535905,  0.5992392 ,  0.21523604,  0.6597706 ,
   0.49392766,  0.96652377, -0.02784265],
 [ 0.144211  ,  0.2665654 ,  0.23803063, -0.2380226 ,  0.24175628,
   0.6760399 , -0.4649684 , -0.04761435, -0.15144967, -0.02586485,
  -0.3987495 ,  0.01451188, -0.42922184,  0.6487432 ,  1.0774521 ,
   0.02879376,  0.04086675, -0.2047436 ,  0.8062745 , -0.16599573,
   0.3388    , -0.03721014,  0.37608165,  0.35235435,  0.19616188,
   0.18778066, -0.41238466, -0.09870823,  0.21868846,  0.65329343,
  -0.20463094,  0.09376287, -0.0680246 ,  0.7455874 , -0.07167022,
   2.3013327 ,  0.06330408,  0.4595927 , -0.6571208 ,  0.06542668,
   1.6583552 , -0.26239124,  0.1929172 , -0.1772162 , -0.36890915,
  -0.765107  , -0.33539677,  0.31720054,  0.10314838,  0.4590249 ,
   0.4800716 , -0.811558  ,  0.41090968,  1.1014618 , -0.0185024 ,
   0.8963804 ,  0.00634773, -0.01071594, -1.4595876 , -0.16771193,
  -0.07840215,  0.09245479,  0.23891692,  0.178539  ,  0.14442474,
  -0.04005321,  0.42508692, -0.11327355, -0.84125584,  0.40050086,
  -0.94375753, -0.0491012 ,  0.15663877,  0.14809914, -0.09027121,
   0.48758498,  0.28352958,  0.51035935, -0.4692664 , -0.05138727,
   0.5993277 ,  0.0394676 , -0.20386004,  0.8386774 , -0.6514515 ,
   0.19712394,  0.17244051, -0.09850982, -1.3958051 ,  0.2436996 ,
   0.14406049, -0.59872705,  0.01515276,  0.7933883 ,  0.09101998,
   0.06899056,  0.23833793,  0.02731364, -0.08502754, -0.6256807 ,
   0.19885421, -0.13836823,  0.00581502,  0.99627537, -0.27075094,
   0.33298966, -0.5223962 ,  0.01627275,  0.4730005 ,  0.4553532 ,
   0.6774558 , -1.0978653 ,  0.17954339,  0.25227347, -0.0544224 ,
   0.50741005, -0.14409478, -0.09867423, -0.14577597, -0.19577993,
   0.08781481, -0.44243795,  0.27870196,  0.14648846, -0.16365878,
   0.00901328,  0.7138439 , -0.02766189],
 [-0.05395817, -0.09885138, -0.5320351 ,  1.486325  ,  1.2345846 ,
  -0.70105594,  0.3830913 ,  0.36191148, -0.0598992 , -0.55962795,
   1.1602911 ,  0.6714454 ,  0.7100175 , -0.4357562 ,  0.9606903 ,
   0.45869127,  0.63160557,  1.3921541 ,  0.7640401 ,  0.25971395,
   0.18504436,  0.25650358,  1.1540815 , -0.19750586, -0.23292682,
  -0.40267023,  0.0823184 , -0.05971502,  0.584548  ,  0.24694464,
  -0.14450783, -0.15073958,  0.43257165,  1.2821134 , -0.34228325,
   0.09539243, -0.31917298,  0.26979578,  1.3030605 ,  0.33760098,
   0.7363447 ,  1.6874511 ,  0.26841825,  0.2644377 , -0.62680954,
   0.18774335,  0.18614654, -0.41555375, -0.15820734,  0.5881529 ,
   0.14968842,  0.64307344,  0.72106665,  0.10816973,  0.28204778,
   0.27359924,  0.50558245,  0.848238  ,  0.1942808 ,  0.08909714,
  -0.26181877,  0.3821261 ,  0.02940864,  0.2598771 , -0.61200154,
   0.48668668,  0.5679366 ,  0.30969658,  0.24303065,  0.7833389 ,
   0.36730456,  0.17355399,  0.31689253,  0.6295289 ,  0.08542124,
   1.1778898 , -0.5330983 ,  0.31402496, -0.19259836,  0.24225537,
   0.46266696,  0.46105286,  0.04726452,  0.21749462, -0.36356342,
   0.17110507,  0.12590386,  0.6070303 , -0.8088662 ,  0.20336947,
   0.04625817, -0.16329661, -0.06755922,  0.15243655, -0.3869225 ,
  -0.3934826 ,  0.13287665, -0.601503  , -0.09903794,  0.33225212,
   0.43998876,  0.07969596,  1.7295651 , -0.08627764, -0.21688356,
   0.24521701,  1.3217682 , -0.11209963,  0.27778566, -0.32262105,
   0.17153686,  0.74552083, -0.11322103,  0.04805468, -0.06416016,
   1.0350035 , -0.26399046,  0.37512076,  1.1564326 , -0.15726884,
  -0.14016564,  0.25009173,  0.31282926, -0.42170396,  0.48160866,
   0.38857326,  0.20681189,  0.02777502],
 [-0.20736027,  1.2378913 , -0.38512367,  0.61205775,  3.3945673 ,
   0.59950364,  1.412637  ,  0.8257101 , -0.78422815, -0.20055118,
  -0.40238345, -1.051605  , -0.14576696, -0.31747225,  0.65912557,
   0.9329183 ,  0.8105907 ,  0.7347574 ,  1.076382  , -0.56424665,
   0.7125617 , -0.13219662,  0.5673468 , -0.15316561, -0.45886824,
   0.5986606 ,  0.14821734,  0.7088954 , -0.13353369, -0.38607058,
  -0.21761341,  0.56566495,  1.0662469 , -0.52203524,  1.0187254 ,
   0.13528986, -0.17623726, -0.9895036 ,  0.00986664, -0.14445217,
   0.5511133 , -2.1455834 ,  0.14124697, -0.08547878,  2.6640167 ,
   0.54752207, -0.6628122 ,  0.5493075 , -0.10124796, -0.614293  ,
  -0.30623874,  0.49251908, -0.1185757 , -2.1250045 ,  0.12647392,
   0.02138401,  0.26209742, -1.2483923 ,  1.0819812 , -0.32038495,
   1.3218834 , -1.1049728 , -0.81668496, -0.41108787,  1.9757669 ,
   0.09993252,  0.62441564, -0.21370177,  1.3670799 ,  2.3550622 ,
   0.8128106 ,  0.05080605, -0.98048335,  0.4436221 , -0.06636092,
   0.4693914 ,  0.57714736, -0.2500828 , -0.3918602 , -0.11615799,
  -0.6439815 , -0.10301845, -0.75798804,  0.00604271, -0.00961933,
  -0.31102097,  0.8258615 , -0.7698323 , -0.86173004,  0.7596279 ,
  -0.6676604 ,  1.5262477 , -1.0323961 ,  1.6442709 ,  0.07309575,
  -1.231887  , -0.6859038 , -0.08939195,  0.9465816 ,  2.3820946 ,
  -0.0760441 , -0.61950827,  0.3632269 ,  0.34789076, -0.5345664 ,
  -0.37901056,  0.32292253,  0.58272976, -1.1949171 , -0.28325596,
   0.42581135,  1.0113252 ,  0.30593413,  0.01894878, -1.035348  ,
   0.19445652, -0.47776842,  0.44092867,  0.22812088, -0.27502877,
  -0.66697747,  0.08584657,  0.45447916, -0.08989209,  0.2685515 ,
  -0.01874863, -0.7450364 ,  0.6275251 ],
 [ 0.02025263, -0.965455  ,  1.5377129 ,  0.41667953,  0.6484574 ,
  -0.65916646,  0.39454517,  1.2815844 ,  0.11558533,  0.14964297,
   0.21589687,  0.85073656, -2.1829233 ,  0.27004895,  0.42989555,
   0.0702427 , -0.26218483,  1.6172245 ,  0.00599426,  0.03848291,
   1.3369998 , -0.06418507,  0.4873482 , -0.83539855,  0.7326973 ,
   0.2706498 ,  0.66051906,  0.67487013,  1.6794647 , -0.13301909,
   0.6317611 , -0.7311469 , -0.55627227,  1.912431  ,  1.7152548 ,
   0.93170404,  1.1754491 , -0.6609468 , -0.69123745, -0.09852654,
   1.0171332 ,  0.8564759 ,  0.9331321 ,  0.43679357, -0.05435452,
   0.88012   ,  0.18317802,  0.10220775, -0.06957111,  0.3595249 ,
   0.34242946, -1.1941215 , -0.6093915 , -0.23107927,  0.736234  ,
   1.0583138 ,  0.73243934,  0.20715658, -0.48043323,  0.13463399,
   0.72014844,  0.4223559 , -0.04448406, -0.41684029,  0.9116886 ,
  -0.54430246, -0.27735847, -1.2875085 , -1.102397  , -0.04889352,
  -0.33710986, -0.4293101 , -0.29684567,  0.16003335, -0.29377058,
  -0.71798867,  0.8328878 , -2.0218377 ,  0.7905806 , -0.67101043,
  -1.1266618 , -0.13382362,  1.1191795 , -0.79910713, -0.8606837 ,
  -0.24873975, -0.4791044 , -0.42225385, -0.7984417 ,  0.44936794,
   0.6614692 , -1.5662938 , -0.2795412 ,  0.03615912,  1.2512114 ,
   0.3280384 ,  0.9494907 ,  1.1924433 , -0.1350958 , -0.16569656,
  -0.2100491 , -0.22673918,  0.09149579,  0.02531711,  0.4762889 ,
  -0.31629223,  0.3500838 ,  0.1815274 ,  0.26239783,  0.34840357,
  -0.23105888,  0.51922023, -0.40389276,  0.00832348, -0.04665646,
  -0.28174773,  1.1463363 , -0.38729987,  1.3719934 ,  0.24403289,
   1.4094707 ,  0.82871693,  0.78528374, -0.18886055, -0.06849843,
   0.53048146, -0.93721485,  0.8796492 ],
 [ 0.31154934,  0.14932257, -1.0805832 , -0.13629921,  0.60686   ,
  -0.5338179 ,  0.64303756, -0.72955406, -0.01963827,  0.1674061 ,
   0.1669188 , -0.19547535, -0.15073276,  1.2938409 ,  0.27481943,
   0.51707727, -0.5462661 ,  1.2718947 , -0.19111721, -0.3131036 ,
   1.220034  ,  0.5067136 , -0.19522162,  0.2734406 , -0.8245551 ,
   0.23692697,  0.17489292, -0.18068832,  0.4269822 , -0.28837138,
   0.07880228,  0.04413876,  0.15023461,  0.22063164, -0.6471611 ,
   0.68631965, -0.4594208 , -0.72087806, -0.8770145 , -0.00755726,
   0.5777769 , -1.0383723 ,  0.6976289 , -0.49787822, -0.22829942,
  -0.29073733,  0.6949255 ,  0.02322764,  0.2777516 ,  0.2896312 ,
  -0.08519164, -0.5968032 ,  0.44657534,  1.1729063 ,  0.20763165,
   0.31168988,  0.332765  , -0.01790537, -0.1291524 , -0.2673924 ,
  -0.2441193 ,  0.208349  , -0.13407995, -0.43678325, -1.0243374 ,
   0.04056552,  0.69286805, -0.2747814 , -0.48064685, -0.01675584,
   0.35138273, -0.46933514, -0.3991256 ,  0.1515266 , -0.78435504,
   0.17392275,  1.8255507 , -0.53876007,  0.1434655 ,  0.17122932,
   0.03684102, -0.32034054, -1.2001165 , -0.35619536,  0.0598345 ,
  -0.24005964, -0.03137832,  0.5802551 , -1.291278  , -0.18426326,
  -0.01095238, -0.750797  , -0.06005435, -0.12274764,  0.8041059 ,
   0.00590906,  0.33975202,  0.79514927,  0.06351985, -1.0485502 ,
   0.9834072 , -0.1194057 ,  0.12601487, -0.69206893,  0.12115641,
   1.1139868 ,  0.26061192, -0.33666548,  0.00467564,  0.13812572,
   0.15569173, -0.9124264 ,  0.12115402, -0.23416382, -0.08147486,
   0.29892904,  0.6446524 , -0.07503738,  0.20558508,  0.6181204 ,
  -0.32100993, -0.05873185,  1.554382  , -0.07455703,  0.08028617,
  -0.786974  ,  0.88960415, -0.11559573],
 [ 0.5999637 , -0.5698173 ,  0.33477867,  0.36592615,  0.5316137 ,
   0.21106231,  0.3814565 ,  0.9180544 , -0.6335052 ,  1.149356  ,
  -0.10567481,  0.3126243 , -0.32386553,  2.0731132 , -0.5927453 ,
   0.06638881, -0.5980873 ,  1.643245  ,  1.33785   ,  0.69981027,
  -0.46566787,  1.2532005 ,  0.81021607, -0.3454415 , -0.10119485,
   0.27958795, -0.24787936,  0.03393151,  0.7794567 ,  0.22860484,
   0.6072963 , -0.34516582, -0.073107  ,  1.4316576 , -0.14812206,
   1.3435391 ,  0.11435606,  0.22675197,  0.4040449 ,  0.40192434,
   1.8664448 ,  1.605273  ,  0.55581695,  0.00673454, -0.42982906,
  -0.34671494,  2.0120232 ,  0.940925  ,  1.0176018 ,  0.4333475 ,
   0.3743055 ,  0.8635856 , -1.0899109 ,  0.06560645, -0.12852691,
   0.8679698 , -0.15988202,  0.5835667 , -0.265264  ,  0.05085932,
  -0.08084621,  1.2779695 ,  0.37934858,  0.21978828,  0.0539755 ,
   1.261856  ,  0.9300327 ,  1.3451471 ,  0.08263851,  0.77170706,
   0.58947104,  0.00103504, -0.67968935,  0.15440717,  0.03636521,
   0.26951975,  0.6069908 ,  0.00972032,  0.03290347, -0.24782826,
   0.582913  ,  0.87881285,  0.06428176,  0.6127429 ,  0.2373056 ,
  -0.11012069,  0.19943343,  0.23651479, -0.46363184, -0.2859856 ,
   1.0708662 , -0.01864592, -0.30553898,  1.3139501 ,  0.17776239,
   0.35630473,  1.0061884 ,  0.24000672, -0.8060972 ,  0.69540083,
  -0.8966584 ,  0.02787979,  0.69446635,  0.49438933,  0.18484977,
   0.21002044, -0.5730398 , -0.00961372,  0.69841427,  0.31229764,
   0.13623174,  1.0216771 , -0.4251077 ,  0.34993973, -0.18515092,
   0.05393942,  2.1089668 ,  0.39382055, -0.1196021 , -0.76322454,
  -0.5377902 ,  0.01880028,  0.05842668, -0.3662244 ,  0.14443398,
  -0.22200836,  0.10473636,  0.24822405],
 [-0.35682112, -0.16766322, -0.8480027 , -0.5168741 ,  0.7825058 ,
   0.34030065, -0.95917135,  0.12247948, -0.7442641 , -0.29506636,
   1.3233792 , -0.81674033,  0.69874114,  0.09998302, -0.18002057,
   0.19762523, -0.30118525, -1.3235086 , -0.526726  , -0.3286727 ,
  -0.27793893,  0.3058706 ,  0.1337365 , -0.02936619, -0.21782693,
  -0.64079237, -0.7351841 ,  0.26586285,  0.5484116 , -0.38503602,
  -0.03888418,  0.0397318 , -0.36085594, -0.3316121 , -0.0658205 ,
  -0.44950974,  0.1484595 , -0.17701235, -0.18252614,  0.02303293,
   0.78932905,  0.8674484 ,  0.21093436, -0.4044264 , -1.0463132 ,
   0.00699896, -3.4035692 , -0.29021752, -2.9953935 , -0.5152286 ,
  -0.9530921 ,  0.952435  ,  0.03905959, -1.6597703 , -0.43709934,
  -1.0316951 ,  1.3213266 , -1.1333519 , -0.0563985 , -0.61227953,
  -0.8900596 , -0.17924592, -0.09049441, -0.05116288, -0.3928904 ,
  -0.87689847,  0.8685818 , -1.1477306 , -1.2188289 ,  0.08086619,
   0.27601352,  0.4868064 , -0.22419463,  0.31063536,  0.15061598,
  -0.02928601,  0.5029044 ,  0.4004588 ,  0.7139675 ,  0.12874277,
  -0.33457989, -0.15284844,  0.8639213 , -1.2148364 , -0.34641272,
  -0.07982513,  0.25357473,  0.19886054, -1.4620074 ,  0.34745768,
   0.5539959 , -0.252498  , -0.02639478, -0.87144876,  0.02268358,
   0.3138097 ,  0.0117174 , -0.13298853,  0.02455869, -0.20561716,
   1.2426919 , -0.08544096,  0.13573883,  0.04690935, -0.26005453,
  -0.24287488,  0.3296259 , -0.12011109, -0.03146284, -0.4399171 ,
   0.07151107, -1.0406588 , -0.31581992,  0.04491394,  0.26302406,
  -0.63262844,  0.17932475, -0.20501176,  0.08235628, -2.0621455 ,
  -0.38219607, -0.87817204,  0.21041311, -0.32769945, -0.03726243,
  -0.78290904, -0.3263249 , -0.1475695 ],
 [ 0.9345391 ,  0.14188041,  0.5201318 ,  0.62373143,  0.57604593,
   0.3279709 , -0.19601099, -1.6016141 ,  0.58968234,  0.29460043,
   0.56661505, -0.5340376 ,  0.34031123,  0.6438216 ,  0.89602   ,
   0.29768512,  1.7556366 ,  1.073303  ,  0.4136335 , -0.995347  ,
   0.29892743, -0.10967343,  0.63889545,  0.27936792,  0.29787752,
   0.23150556, -0.18795165, -1.5938456 , -1.8994062 ,  0.55750346,
   0.16405834, -0.00861578, -0.04813026,  0.7954955 , -0.03579997,
   0.2659017 , -0.37480333,  0.6004559 , -0.7951285 , -0.00091701,
   0.10675704, -0.24739811, -0.0883126 ,  0.7502097 ,  0.29119745,
   0.48176205,  0.4081151 ,  0.5802373 ,  0.50163233, -0.9535771 ,
  -0.07059854,  0.25497887,  0.6780159 ,  0.9403064 ,  0.760555  ,
   0.08776137, -0.61110353,  0.2877769 , -0.38764268, -0.49184787,
   0.21893826, -0.1940982 , -0.7742993 ,  0.47028863,  0.1082603 ,
  -0.8435572 , -1.0282012 , -0.03496564,  0.10954617,  0.08068439,
   0.3452819 ,  0.07016116, -0.19854735,  0.09153567,  0.04963266,
   0.30356795, -0.71841466,  0.15029402,  0.00820213, -0.75398815,
  -0.3636854 , -0.5936664 ,  0.3656092 ,  0.54749966,  0.04598808,
  -1.6118841 , -0.1977376 , -0.29874465, -0.48419818,  1.3116757 ,
  -0.867724  , -0.7382387 , -0.27086058, -0.633454  ,  0.3889447 ,
  -0.04710134,  0.38714907,  0.4619347 , -0.06945518, -0.9683792 ,
   0.75079936, -0.7359955 ,  0.39372188,  0.13907164,  0.1642061 ,
  -0.68002284, -0.85350287,  0.05252308, -0.86193275, -0.15395495,
  -0.00187563, -0.28082892, -0.28364226, -0.39341748,  0.48048222,
   0.16646679, -0.66095793,  0.27592734,  0.06316712, -1.2068397 ,
   0.68914604, -0.5370846 , -0.28487977,  0.0283711 ,  0.3949168 ,
   0.132767  , -0.07408956, -0.2579709 ],
 [ 0.51419145,  0.5842033 ,  1.0840715 , -0.10631435, -0.77695686,
   0.4391327 ,  0.76673716,  0.664293  , -0.23899372, -0.82839876,
   0.26420838,  0.4769066 , -0.5415749 ,  0.0123993 , -0.1375474 ,
  -0.5570221 , -0.36647362,  0.33275884, -0.22331375,  0.19764887,
  -1.0990303 , -0.83244944,  0.13883854, -0.3017467 ,  0.05206979,
  -0.41455364, -0.30072302,  0.01340591, -0.5265705 ,  0.41217345,
  -0.78986156,  0.32435197, -0.08217241,  0.25258273, -0.4535072 ,
   0.43740696, -0.0430643 ,  1.0616564 ,  0.64084405,  0.18786906,
   0.25677842,  0.09613658, -0.8854219 , -0.358941  , -0.1267752 ,
   0.14570938, -1.1834326 , -1.0960891 ,  0.31874034, -0.31894618,
  -0.2914971 , -0.04370906,  0.7808707 ,  0.07195462, -0.40711537,
   0.9590609 , -0.03742747,  0.44941443,  0.09755689, -0.41191995,
   0.44557852, -0.3666411 , -0.00848754,  0.09030116,  1.1518482 ,
   0.32214254,  0.01150637, -0.10667561, -0.27806658, -0.1302406 ,
  -0.3276854 , -0.516962  ,  0.37682658, -0.11149108,  0.02492317,
  -0.92317814,  0.25395542,  0.08143973, -0.12156352, -0.15637219,
  -0.3036416 ,  0.03336746,  0.57109016,  0.00168974,  0.15641552,
   0.04940401, -0.7284853 , -0.37309688, -0.6315928 , -0.09628967,
   0.29863334,  0.8470947 , -0.02588497,  0.18677089,  0.00096179,
   0.52483606, -0.40936646, -0.48841423,  0.07130237,  0.50972694,
  -0.5660765 , -0.30858827,  0.18329826, -1.1831508 , -0.23653257,
  -0.4907495 , -0.01722662,  0.04895283,  0.10320106, -0.94809663,
  -0.83440226,  0.394875  , -0.06673894, -0.19490172,  0.04441952,
   0.98376924,  0.20205973, -0.24770123,  0.01132868, -0.3598677 ,
   0.04817916,  0.06600513, -0.06270801, -0.9982534 ,  0.30118358,
   0.278809  , -1.076258  , -0.01280216],
 [ 0.66163564,  0.01773833, -0.43887907, -0.16454555,  0.7626181 ,
   0.26953223, -0.37048233,  0.1083945 ,  0.20360212,  0.34181353,
  -0.70951974, -0.8054761 ,  0.14182347,  1.725393  ,  0.51728624,
  -0.12158321, -0.02548535, -0.742433  ,  0.25690746,  0.6589837 ,
  -0.27453205,  0.07044839,  0.22170645, -0.2346061 , -0.67210406,
  -0.6503945 , -0.41730306,  0.78596413, -0.4573213 , -0.12445389,
   0.32120994, -0.52870494,  0.10978057,  0.7694471 , -0.4839214 ,
   0.90163666, -0.90058684, -0.14296162,  0.07171038,  0.37388036,
  -0.13996604,  0.46700987, -0.21013755, -0.5926319 ,  0.48302004,
  -0.6716408 ,  0.16228494, -0.585607  , -1.0901569 , -0.17833349,
  -0.07355148,  0.12488735, -0.03260745, -0.30675155,  0.21565895,
  -0.24176994, -0.00784343, -1.2556517 , -0.11535496,  0.30604938,
  -0.80239445, -0.03618598,  0.05132058, -0.23701996, -1.1753733 ,
  -0.12698492, -0.02502147, -0.04673585, -0.8119719 , -1.1078932 ,
   0.02439659,  0.22074199,  0.47457168, -0.01114979, -0.09719218,
   0.64286876, -0.50984263,  0.35059947,  0.19680399,  0.2292754 ,
   0.38852587, -0.00111545, -0.37659413, -0.1686769 , -0.4400719 ,
   0.04311465,  0.10208935,  0.5013419 ,  1.0174214 ,  0.14889354,
   0.03763886,  0.5698075 ,  0.39210835,  0.31783852,  0.8087406 ,
  -0.80595124, -0.31093958,  0.35633615,  0.0933819 , -0.5292183 ,
   1.4174244 , -0.71997607, -0.3359979 , -0.4118105 , -0.05152682,
  -0.74393094, -0.41783243, -0.18895517,  0.09800366, -0.48918873,
  -0.5275407 ,  0.09133609,  0.3357686 ,  0.07435685, -0.01120504,
  -0.18448652,  0.03017557, -0.53660285, -0.38294226,  0.81741893,
   0.11861117,  0.40177932,  1.3250413 , -0.18917163, -0.65924144,
  -0.04846504, -1.0783845 , -0.3420747 ],
 [-1.5615782 ,  0.36664808,  0.26036605,  0.888029  ,  0.1164643 ,
  -0.03492513, -0.23559855, -0.33798137,  0.8367649 , -0.20320807,
  -0.32034898,  0.62488455,  0.6281878 ,  0.5796746 ,  0.47866523,
  -0.04757926, -0.01271926, -0.9637328 , -0.18507622, -0.32792643,
   0.6343187 ,  0.11866081, -0.36819547,  0.10350408,  0.39279634,
   0.36201614,  0.38067272, -0.22798656, -0.44415578,  0.2666504 ,
  -0.6258362 ,  0.26597375, -0.07732509, -0.00802816,  0.67452174,
  -0.9026004 ,  1.0565938 ,  0.09063888,  0.07376517,  0.23457323,
   0.38303664, -0.562932  , -0.998103  , -0.1637184 , -0.5057355 ,
  -0.6436021 , -2.661157  ,  1.0272294 ,  0.18861541,  0.02762804,
  -0.07737068, -0.7192444 , -0.99603724, -0.80713624,  0.80409086,
   0.817002  ,  0.6116686 ,  1.4458153 , -0.6958529 ,  0.8915022 ,
  -0.5630007 , -0.16623229, -0.71474195, -1.1639963 ,  0.8075805 ,
   1.6519656 ,  0.3995516 ,  0.5292593 ,  0.646026  ,  0.58930725,
  -0.11263296,  0.6746864 ,  0.37952352, -0.22475877, -0.07016346,
  -0.48050353,  0.27957436,  0.7140299 ,  0.722429  , -0.3054384 ,
  -0.3982648 , -0.6669321 ,  0.12243777,  0.29359657, -0.01450026,
   0.2405537 ,  0.6369586 , -0.14547361, -0.06102871,  0.5325214 ,
  -0.21422616, -0.42444506, -0.40615788,  1.1042564 ,  0.7374306 ,
  -0.1267882 ,  0.20644277,  0.25905007, -0.49334025, -0.5012365 ,
   0.84214526,  0.15830156, -0.35726082, -0.6221847 ,  0.23498207,
   1.222869  , -0.6900829 ,  0.42931953, -0.5851859 , -0.3508915 ,
  -0.08574894, -0.28450245,  0.4525613 ,  0.12449021, -0.27093714,
  -0.9061628 ,  0.19468425,  0.35828605,  1.0199087 ,  1.7088704 ,
  -0.18187241,  0.7604585 ,  0.4994131 ,  0.3172651 ,  0.32908875,
   0.18098874,  0.89698696, -0.43434078],
 [-0.25155386, -1.1882491 , -0.0885042 , -0.9923465 , -0.5150148 ,
  -0.7513984 , -3.4855144 , -0.1290659 ,  0.07151296, -0.41551584,
  -0.0112894 , -1.5999625 , -0.1821094 ,  0.20357656,  0.01677332,
  -0.4356899 ,  0.3233908 , -0.3998499 ,  0.4921246 , -0.11823323,
  -0.01533878, -1.1315591 , -0.29706553, -0.73835987,  0.0886107 ,
  -1.3022124 , -0.42965704, -1.0138087 , -0.6119329 ,  0.29917446,
  -0.18320648,  0.16412257, -0.3940132 , -0.7302681 , -1.0213057 ,
  -0.05465383, -0.11623393,  0.21914044, -0.6432187 , -0.31431407,
  -1.6766059 ,  1.1359804 , -0.01625581, -0.2754898 , -1.6739353 ,
  -0.60198957,  0.13193867, -0.18490428, -0.68187666, -0.42134982,
  -0.42994916, -0.29570618,  0.1252926 , -0.6540085 ,  0.30752036,
  -1.5766946 ,  0.35179335, -0.599076  , -1.1291083 ,  0.25841653,
   0.02058554,  0.07692948, -0.75809306, -0.34163606, -3.418552  ,
  -0.98103815, -0.45766062, -0.50705296, -1.7206607 , -0.49810562,
  -0.58912104, -1.1101834 ,  0.25202036, -0.5575902 , -0.03962549,
  -0.35664785, -0.73665375, -0.4869017 , -0.39120832, -0.14297055,
  -0.01133164, -0.14157178, -0.6018266 , -0.2350326 , -0.02669543,
  -0.6904169 , -0.21000223, -0.9431863 , -0.26044342, -0.76827055,
  -0.7954475 , -1.9571292 , -0.77711564, -0.26799303, -0.8569241 ,
  -1.9004964 , -0.2201985 , -0.63393104, -1.641708  , -0.35851932,
   0.55224925,  0.33225352, -0.08949175, -0.1952914 , -0.16033009,
  -0.88144636, -0.77783585,  0.41505235, -0.7016015 , -0.9816446 ,
  -0.57181484, -0.47091645, -0.78253555, -1.6925482 ,  0.34589234,
  -1.1105918 ,  0.14094879, -0.40214816, -0.02115746, -0.2789048 ,
  -0.2794383 , -0.38999608, -0.7378071 , -0.3004222 , -0.45587936,
  -0.3458769 , -1.6628146 , -0.77118057],
 [-1.4362788 ,  0.479974  , -0.6056359 , -0.49323887,  1.3629094 ,
   0.35863695, -0.51020527,  0.4446377 , -0.05912081, -1.4524285 ,
   0.33174536,  0.03945762,  0.14544609,  0.21423043, -0.16931537,
   0.39067116, -0.05248484,  0.9991735 ,  1.0474875 , -0.61258346,
  -0.73408574,  0.05701662,  0.847224  ,  0.10453653, -0.281527  ,
   0.1458015 , -0.00364114, -0.15803835,  0.30553707, -0.33467564,
  -0.31066975, -0.30715698,  0.16281061,  0.02306181, -0.25518194,
  -1.2654055 , -0.4285554 , -0.37579265,  0.42307413,  0.36994293,
   1.206652  ,  1.4102126 ,  0.13099837, -0.24638388, -1.006324  ,
   0.51147574, -1.2791442 ,  0.3168148 , -0.80825084, -0.0384533 ,
   0.4013194 , -0.5067264 ,  0.5088938 ,  1.4040734 ,  0.29024208,
   0.25857362,  0.33016694,  0.6536851 , -0.35077894,  0.14125554,
  -0.56536686, -0.17213413,  0.26616105,  0.22661307,  0.39935052,
  -0.10074581, -0.2683916 ,  0.680487  , -0.44054258, -1.0111214 ,
   0.64541775, -0.07902485,  0.03136221,  0.92794764,  0.3616333 ,
   0.13474883, -1.0129362 ,  1.328479  , -0.9168516 ,  0.04244124,
   0.39585155,  0.30483994, -0.14836119,  0.19940123, -0.22867598,
   0.1877857 ,  0.11669253,  0.02555781, -0.4295689 , -0.24971275,
  -0.01034148, -1.045457  ,  0.32245812,  0.39049682, -0.18808116,
  -0.67814374,  0.03674542,  0.06585888, -0.7752051 , -0.6064988 ,
   0.40760505, -0.22196877,  0.04104144,  0.3708088 , -0.6347419 ,
   0.02259358,  0.552656  , -0.09796788,  0.8668314 , -0.06566614,
  -0.01737115,  0.14708047, -0.0416771 , -0.34119174,  0.02054471,
  -1.2699203 ,  0.21783935, -0.16568375,  0.9991273 ,  0.284836  ,
  -0.5978308 , -0.30062512,  0.4051889 , -0.4054572 , -0.37485084,
  -0.04393509,  0.9115046 , -0.31353632],
 [-0.9422828 , -0.6962568 ,  0.753481  ,  0.14090472, -0.28999147,
  -0.5407905 , -0.11207163,  0.63783145, -0.68117684, -0.32479906,
  -0.01436805, -0.15843792, -0.37336126,  0.23556228, -0.4490422 ,
   0.03693759, -0.11092373, -0.4687402 , -0.27218148,  0.84216803,
   0.44824505,  1.0029614 ,  0.04462518, -0.40172744,  0.03615877,
   0.18764628, -0.20231986, -0.49426907, -0.65326387, -0.37211058,
   0.26479977, -0.41678762,  0.36532214, -0.31541055, -0.16108446,
   0.9964354 ,  0.03693628, -0.10005604, -0.15939069, -0.38328794,
   0.4113008 ,  2.0588248 ,  0.25868297,  0.59221286, -1.1967654 ,
  -1.0334976 ,  1.4164224 , -0.4973816 ,  0.22428674,  1.1268826 ,
  -0.36703718,  0.9941782 , -0.38537294,  0.74884474, -0.16679767,
   0.62908345, -0.4360167 , -0.75378335,  0.13482334, -0.47691205,
  -0.83907324,  0.23430608,  0.01294585, -0.8243621 ,  0.5937829 ,
   0.3912575 , -1.0034434 , -0.34719804, -0.9145786 ,  0.52781564,
  -0.07220872, -0.7524916 ,  0.14236149,  0.33344066, -0.6807762 ,
  -0.18284541,  0.04221184, -0.47156855, -0.26190826,  0.11318742,
   0.07197096, -0.58188105, -0.48911116,  0.6214426 , -0.5827705 ,
  -1.2092404 ,  0.3410771 , -1.1217759 ,  0.18804084, -1.1546544 ,
  -0.10130332,  0.6101679 , -0.23780277, -1.5224082 , -0.44059616,
  -0.71452385, -0.2790537 , -0.28610095, -0.5524191 ,  0.6869806 ,
   0.0108106 , -0.7217683 ,  0.22617759,  0.42823467,  0.3417962 ,
  -0.56285423,  0.31383544, -0.45881116, -0.3380014 ,  0.35979035,
  -0.5587602 ,  0.38021314, -0.5270836 , -0.10532793,  0.74791855,
  -1.6753147 , -1.2124625 , -0.06230953,  0.67256135, -0.37609777,
  -0.2892746 ,  0.47903192, -0.6021964 , -0.69970155,  0.12438444,
  -0.36853102,  0.32045972, -0.47680116],
 [ 1.5315238 ,  1.132874  ,  0.16446157,  0.10277585,  0.12966254,
  -0.10316164, -0.40876716,  0.54595834,  0.19195028, -0.41119397,
   0.92171985,  0.5751446 , -0.51081204,  0.1764138 , -0.49057797,
  -0.17699987, -0.57402784,  0.7189649 , -0.10017725, -0.31968227,
  -0.72050637,  0.9422743 , -0.21154419, -0.218279  , -0.06204043,
  -0.6370666 , -0.19792965,  0.5804355 , -1.2666    ,  0.27298468,
   0.29114968, -0.44194174, -0.08502014,  1.6557591 ,  0.84892064,
   0.90928733,  0.70645183, -0.05531108,  0.1943477 , -0.19829255,
   1.2183876 , -0.06794077, -0.25393683, -0.22253595, -0.50389683,
  -0.42322096,  0.5188286 ,  0.70783097,  1.5636148 ,  0.45839015,
   0.6792885 ,  0.78556263, -0.15588951, -0.09317783, -0.23307683,
  -0.29623836,  0.90394866,  0.5313317 ,  0.23772778,  1.0653946 ,
  -0.76593906, -0.3872692 ,  0.33499077, -0.2314557 , -0.1984749 ,
   0.6266242 , -0.02433023, -0.14808135, -0.7619073 ,  0.32439   ,
   0.04132314, -0.27767253, -0.06873176,  0.08322575,  0.19322546,
   0.6255473 , -1.1237109 , -0.6511178 ,  0.72295976, -0.01080401,
   0.09673462,  0.07988767, -0.19708133,  0.31345975,  1.2899666 ,
   0.8401333 ,  0.46551025,  0.22678162, -0.18563105, -0.03547024,
  -0.19699085, -1.1542467 ,  0.0539662 ,  0.21582353, -0.07069899,
  -0.7674204 ,  0.30511376,  0.10122082, -0.21012354,  0.41713065,
  -1.486152  , -0.7637724 ,  0.19732112,  0.40002084, -0.5940563 ,
  -0.57357293,  0.701009  ,  0.4454186 ,  0.10294809,  0.41193393,
   0.44977596,  0.99044126, -0.15389314,  0.48702115, -0.16339439,
   0.77280396, -0.09554625,  0.06628395, -0.6133561 ,  0.1131457 ,
   0.24238876,  0.5468002 , -0.26442528, -0.7317455 ,  0.11821963,
   0.25535142, -0.33120283, -0.22046748],
 [ 0.2328938 ,  0.91131127, -0.09824359,  0.0734041 , -0.629393  ,
   0.5361064 , -0.41488975, -0.36240292,  0.02299364,  0.14651485,
   0.2229512 , -0.20229347, -0.01573426,  1.1518513 ,  0.5557966 ,
   0.4879209 ,  0.21026199,  0.25655487, -0.43334344, -0.02747946,
  -0.48617426, -0.05800787,  1.242641  , -0.14713678, -0.17636637,
  -0.07685454, -0.43872687,  0.27324688,  0.35633737, -0.40709946,
  -0.19408838,  0.21514551,  0.24077156,  1.0430169 , -0.33478394,
   0.11172336, -0.3116529 , -0.7993484 ,  0.5474917 , -0.95242506,
   0.7168393 ,  0.40167293,  0.381662  ,  0.41588938,  0.06692681,
   0.5151241 ,  0.12982734, -0.57962143,  0.00658996, -0.02016122,
  -0.49503487,  1.15367   ,  0.37060156,  1.1011217 , -0.10192105,
  -0.24145903,  0.58034706,  0.6709149 , -0.4543297 ,  1.1432365 ,
  -0.36906916,  1.4094412 ,  0.04333521,  0.17100587, -0.300702  ,
  -0.42466715,  0.4533073 , -0.5197856 ,  0.11492617,  1.046455  ,
  -0.5071946 ,  0.04083361,  0.48237872,  0.26405206,  0.35637423,
  -0.30683845, -0.49301612,  0.2450374 ,  0.05582555,  0.00903657,
   0.4168912 , -0.05253533, -0.3397173 ,  0.30378607,  0.07971506,
  -0.64823467, -0.08205315,  0.1234986 , -0.495584  , -0.30033872,
   0.4323857 ,  0.08183496,  0.05399634,  0.9679979 ,  0.10870963,
   0.21578461,  0.2943475 , -0.4686236 , -0.2133685 , -0.30031568,
  -0.18305795,  0.00167882,  0.387828  , -0.17975481, -0.06547905,
  -0.66909903,  0.38593617, -0.01706313,  0.08590705, -0.02356951,
   0.30661726, -0.67101824, -0.2263927 ,  0.20381978, -0.17499042,
   0.8417269 ,  0.12236128,  0.41369137, -0.6211429 ,  0.62719285,
  -0.07622543, -0.25753525, -0.00044521, -0.4703916 ,  0.3965492 ,
  -0.14554064, -0.1373742 , -0.68947905],
 [ 0.49055547, -0.09480583,  0.33745924,  0.5652824 ,  0.50362   ,
   0.88808537,  0.84580594,  0.42893344,  0.5901525 ,  0.04969563,
  -0.01219958, -0.38391966,  0.17393309,  0.1930047 ,  0.19077058,
  -0.00455079,  0.53281015,  0.21225329, -0.06816682,  0.47766945,
   0.51241535,  0.77582383, -0.6128939 , -0.11534332, -0.25932848,
  -0.00186875,  0.12158407,  1.5780734 ,  0.15693907,  0.04875198,
   0.24477462, -0.8164343 ,  0.5437779 ,  1.1115081 ,  0.02350741,
  -0.00061826, -0.31370315, -0.02921885, -0.42499703, -0.11362343,
   0.73330116,  0.74274755, -0.16046758, -0.09056185,  0.9549983 ,
  -0.15455112, -0.33769935, -1.189368  , -0.15081157,  0.8076109 ,
   0.2729107 ,  0.16396953,  0.4155528 , -0.03838992,  0.42093784,
  -0.08410039,  0.09857205,  0.43461493,  0.09245925,  0.40839586,
  -0.03079587, -0.13927399,  0.03215437,  0.48753497,  0.14757508,
   0.737186  ,  0.97665244, -0.27933198, -0.47057185,  0.8817357 ,
   0.76873726, -0.07294675, -0.18758044, -0.4774614 ,  0.24638598,
   1.0185828 ,  0.27372888, -0.10215628,  0.79021853, -0.10256287,
   0.05559114,  0.4269883 ,  0.7171656 ,  0.14311716,  0.09176155,
   1.120012  ,  0.17485113, -0.2681595 , -0.521871  ,  0.18559055,
  -0.24206026, -0.00755647,  0.198711  ,  1.2759191 , -0.04210797,
  -0.3546749 ,  0.30384126, -0.17907561,  0.4086359 ,  0.56298685,
  -0.37235487, -0.13292263, -0.4282381 ,  1.3743745 , -0.30290917,
   0.7427625 ,  0.00380283, -0.31228074, -0.34829125, -0.07368874,
  -0.10982159,  0.18535067,  0.01890408,  0.20917055, -0.11760852,
   0.689053  ,  0.7834295 , -0.01071178, -0.10713921, -0.05591312,
   0.24923523, -0.38205078, -0.38077223, -0.21488059,  0.29480386,
   0.09336617,  0.6912765 ,  0.00862858],
 [ 0.77404827, -0.5040765 , -1.4142898 ,  0.04942893, -0.55648696,
   1.1173732 ,  0.27093685, -1.5903383 , -0.37573668,  0.07195516,
   0.5477165 , -0.336826  , -0.28495798,  0.06718339,  0.2522872 ,
  -0.3230753 ,  0.7357853 , -1.4320976 ,  0.5520526 , -0.27055874,
  -2.2217941 ,  0.45870644,  0.1664033 , -0.83042353, -1.5152944 ,
  -1.6902243 , -0.46197212, -0.13371904,  0.27502838,  0.5372972 ,
  -1.4623222 , -1.172559  ,  0.40546882,  0.46544617, -0.7794437 ,
   0.23703678, -1.3359258 ,  0.49651223, -0.16771162,  0.01488542,
  -0.55514246,  0.5237586 ,  0.14658436, -0.4113396 , -1.0876054 ,
  -0.06350618, -0.08244709, -0.22085102,  2.3816755 , -1.0699892 ,
   0.10748536,  1.0799974 ,  2.0360727 , -0.8687681 ,  1.1381364 ,
  -0.44483772,  0.3388217 , -1.5792552 , -1.3601962 , -0.8553228 ,
  -1.8071867 ,  1.609539  ,  0.378146  ,  0.3672086 , -1.0941324 ,
  -0.19195804,  0.52837914, -0.69421434,  0.03214139, -0.5400008 ,
   0.36876974,  0.6380599 ,  0.22286178,  0.09372054, -1.5021524 ,
   0.79748154, -0.7032281 ,  0.27862683, -0.88605934,  0.14040847,
   2.1645033 , -0.3172831 , -1.7936835 ,  1.2191509 ,  0.82296515,
   0.6968643 ,  0.60289603,  0.08714293, -3.043659  , -1.0665454 ,
   0.49622744, -0.03363389,  0.1744308 , -1.4337015 , -0.8728794 ,
   1.0741377 ,  0.46623403, -1.6069576 , -0.04296382, -0.27688116,
   2.5697253 ,  0.02353567,  0.6533802 , -0.22720589, -0.5936822 ,
  -0.7692317 ,  0.3325961 , -0.0534231 ,  0.3706611 , -1.8410575 ,
  -0.15766786, -1.8149769 , -0.92367387, -1.0587463 ,  0.48504627,
   0.6637604 , -0.51258606,  0.43159935,  0.23912804,  0.2817794 ,
  -1.3140062 , -0.18034121, -1.1938953 ,  0.06041323, -0.20357968,
  -0.05245242, -0.7608925 , -1.9922949 ],
 [ 1.0486037 , -1.184713  ,  0.17079027, -0.28451863, -0.1259091 ,
   1.0255342 ,  0.23722818,  0.20268269,  0.07074739, -0.21303026,
   0.8627522 ,  0.07717646, -0.28645056,  1.4741348 , -0.24434893,
  -0.5844826 ,  0.84555084, -1.4007903 , -0.44207406,  0.28727037,
   0.05909888,  0.3775976 ,  0.7041093 ,  0.12083983, -0.92065233,
  -0.32725823, -0.76808226,  0.7417016 , -0.4864486 ,  0.19456187,
   0.14000559, -0.34191725,  0.4819911 , -0.10221709, -0.26186943,
   0.84760284, -1.063374  ,  0.0947203 , -0.4475829 ,  0.06859005,
  -0.40216094,  2.5917096 , -0.15733664,  0.87485486,  0.6787479 ,
   1.2785659 , -0.9259248 ,  1.3253605 ,  0.30327225,  1.0718161 ,
   0.60580724,  0.2079719 , -0.4918106 , -0.19950841, -0.08227397,
  -0.2656189 ,  0.43167502, -0.04847732,  0.2549257 ,  0.6551107 ,
   0.0556394 ,  0.0764712 ,  0.17518371,  0.01256165, -0.39371732,
   0.74281555, -0.6105667 , -0.04729559, -0.17651086, -0.4327345 ,
  -0.2507473 , -0.14399926, -0.34335825, -0.10294634,  0.42831975,
   0.93791807,  0.7904281 , -0.27553287, -0.742652  ,  0.11172166,
  -0.14898916, -0.01639796, -0.2611011 ,  0.6611221 , -0.4350942 ,
  -0.4760919 , -0.32933778,  0.693373  ,  0.22052787, -0.23416777,
  -0.1447235 ,  0.51755524,  0.20366928, -0.93028575, -0.65627015,
   0.79900837,  0.12365835, -0.31917804,  0.5841165 ,  0.5392167 ,
  -0.5359878 , -0.03839752,  0.5650832 , -0.2477295 , -0.03877855,
  -0.58910084,  0.16410057, -0.0702661 , -0.3578085 , -0.63034594,
   0.995973  , -1.5246147 ,  0.15283607,  0.15002348,  0.06326705,
  -0.01515282,  0.8244059 ,  0.40581366,  0.26217976,  1.7885473 ,
   0.796579  ,  0.2517453 , -0.348589  , -1.0061139 ,  0.30744478,
  -0.29895005, -0.5134379 , -0.21446525],
 [-1.7240956 , -0.15347931, -0.9574478 ,  0.15075305,  0.47730404,
  -0.01382413, -0.544321  , -0.16209026, -0.7152993 , -0.7727655 ,
   0.3131312 , -1.6651388 , -1.6116439 ,  1.6129977 , -0.32647824,
  -0.2455145 ,  1.8123674 , -1.2281423 ,  0.12703063, -0.9639011 ,
  -0.6855556 , -0.16774069,  0.47058904, -1.568603  , -1.3832405 ,
  -1.0700883 , -0.7866667 , -0.59167105,  1.2501317 , -1.1037523 ,
  -0.742733  , -0.51972795, -0.22542232, -0.07880735, -1.7534393 ,
  -0.6042437 , -1.3476307 , -0.90487695,  0.12536581,  0.13196556,
   0.20786424, -1.8351444 , -1.7059913 , -0.610872  , -0.3488896 ,
  -0.18473764, -0.6857249 , -0.08852775, -0.7314667 ,  0.13350329,
  -1.2347088 , -0.81098306,  1.1279941 , -0.02984217,  2.1863196 ,
  -1.289146  ,  3.3109105 ,  0.39730445,  0.0733669 , -1.0175484 ,
  -0.33345443,  1.1372002 , -0.09047779, -0.3271694 , -1.050614  ,
  -0.14864053,  0.3360733 ,  0.4055    , -0.15262206,  0.2878469 ,
   0.29577464, -0.941679  ,  0.26600298, -0.47911245, -0.48660403,
  -3.599531  , -0.01664436, -0.21452631, -1.4558952 , -0.30899417,
   1.282939  , -1.110881  , -2.1366615 ,  0.17830348, -1.2459155 ,
  -1.1940093 , -1.7414255 , -0.0322979 , -0.62243676, -1.1051463 ,
  -0.07135867,  0.08661167,  0.24861461,  0.39523077, -1.6232916 ,
  -1.0639724 ,  0.17792761, -1.6217569 , -0.0373079 , -0.26305184,
   0.6505675 , -0.80353004,  0.00920481,  1.1319618 , -1.6354574 ,
  -1.3850114 , -0.3049651 , -0.5389295 ,  0.09811281, -0.26193014,
   0.568302  , -0.6924567 , -0.83371085, -0.43550354, -0.205277  ,
   0.2738912 , -0.17546353, -0.39005414, -1.0856863 ,  0.0971702 ,
  -0.08465461, -1.8596945 , -1.0822964 , -1.6185637 , -0.48682404,
  -0.23746406, -0.12690654, -1.3995787 ],
 [ 0.43848187,  0.01595135, -0.13082837, -0.44382918, -0.24127589,
  -0.00392409, -0.4093333 ,  0.32953665, -0.15280789, -0.374004  ,
   0.04570145,  0.39067546,  0.28461185,  0.29946104,  0.5378135 ,
  -0.15431766, -0.19742817,  0.7506431 , -0.38950327, -0.38863888,
   0.55159366,  0.02436523,  0.4688102 ,  0.15688044, -0.01979993,
   0.49265033, -0.17659889, -0.35435194,  0.4860431 ,  0.28584045,
  -0.63258994,  0.00726708, -0.33488885,  1.3043112 , -0.30406106,
   0.855494  , -0.1746156 , -0.04410001,  0.51484555, -0.27553296,
   0.9394671 ,  0.69317454,  0.10583206,  0.31980428, -0.29806605,
   0.11506739, -0.59365875,  0.22364883, -0.6647805 ,  0.7847062 ,
  -0.1783697 ,  0.05255371,  0.42190233,  0.74238294,  0.11810069,
   0.11712851,  0.22823165,  1.1649776 , -0.5162132 , -0.2356497 ,
  -0.7144475 ,  0.56103176, -0.34428164,  0.11788481,  0.24405354,
  -0.3534361 ,  0.07785485,  0.15532617,  0.83510476,  0.9083744 ,
  -0.18621229,  0.37736776,  0.32224393,  0.195522  ,  0.40092862,
   0.5343101 , -0.33660594,  0.71701884,  0.6086608 ,  0.04630249,
  -0.08137591, -0.41363624,  0.55884755,  0.07163717, -0.11930754,
  -0.5612533 , -0.1773708 ,  0.11472461, -0.62418354,  0.00339023,
   0.02278822, -0.13879313,  0.05550346,  0.5779377 ,  0.22747144,
   0.28645307,  0.32919738, -0.02682025,  0.26030758, -0.37280527,
   1.1049187 , -0.26228714, -0.09887231,  0.1282374 ,  0.05609882,
  -0.00164896,  0.6034684 , -0.2989607 , -0.11199002, -0.02706846,
   0.3006765 , -0.8543728 , -0.00554101,  0.34017044, -0.22022285,
  -0.03552856,  0.10996111, -0.06205057, -0.44167063,  0.24824221,
   0.61689186,  0.03217339,  0.38706964,  0.08288957,  0.07032739,
  -0.40458646, -0.45455596,  0.12710789],
 [-1.3856909 , -0.7021078 , -0.31240705,  0.36093256, -1.44604   ,
  -0.01200718, -0.20687729,  0.22026964, -0.6228977 , -1.4179789 ,
   0.2852147 , -0.19923481, -0.17559339, -0.84836614, -0.11273947,
   0.44227287, -0.5987263 , -0.02502237, -1.5751226 , -1.0599099 ,
  -0.28351685,  0.01744723,  0.21377586, -0.0147895 , -0.7393712 ,
  -0.0994527 , -0.5161725 ,  0.2329026 , -0.05961423,  0.10595318,
   0.15771545,  0.44836044,  0.42899516, -0.11672425, -0.3330223 ,
  -1.1941564 , -0.49853894,  0.09843022, -0.50027853, -0.2725174 ,
  -0.38765994, -0.20661512, -0.5695143 , -0.29306355, -0.94046384,
  -0.9058068 , -1.0308311 ,  0.32679453, -0.8421238 , -1.7797905 ,
  -0.08526471,  0.08080164,  0.40162447, -2.3006783 , -0.34223542,
  -0.0318115 , -0.08078875, -0.59740543,  0.27689204, -0.68955374,
   0.2919296 , -0.35361412, -0.22501725, -0.44874462,  1.0869387 ,
   0.39575925, -0.32012755,  0.10933641, -0.6194403 , -0.6329548 ,
  -1.1477693 , -0.0378334 , -0.00215989,  0.10959607,  0.4887865 ,
  -0.06467616,  0.53795576, -1.0854754 , -2.238122  , -0.02258037,
  -0.3620822 , -0.41138172, -0.73696345,  0.38608617,  0.11544336,
   0.41445088, -0.73944265, -0.39947596, -1.4431915 , -0.19985986,
   0.40926716, -0.46871507, -0.1543708 ,  1.2177995 , -0.21192405,
   0.52133054,  0.79586387, -0.2225782 ,  0.15246066, -0.847738  ,
  -0.38965717, -0.99222356, -0.038811  ,  0.28908843,  0.19803667,
  -0.40121278, -0.8113779 , -0.18354258,  0.18633132,  1.0511737 ,
   0.28853008,  0.30588847, -0.13211994, -0.12837002, -0.65706515,
  -0.02490008,  1.2288873 , -0.05461086,  0.23256803,  0.288727  ,
  -0.4657555 , -0.3545767 , -0.7586265 , -0.8700702 ,  0.09137169,
   0.03514363,  0.289312  , -1.0404536 ],
 [ 0.04746957, -0.03374586, -0.22638415, -0.24864636,  0.02692965,
  -0.3171279 , -0.28661475, -0.08501914, -0.6835045 , -0.35697109,
  -1.037242  , -0.4802286 , -2.4472435 , -1.0751256 , -0.27845615,
   0.34597558, -1.232001  ,  0.02434656, -0.20106454, -0.9018164 ,
  -0.1672001 ,  0.11651801, -0.7109763 , -0.4877445 , -0.13848108,
  -0.30109364, -0.7312046 , -0.34392658, -1.128916  , -0.29625124,
  -0.4385384 , -0.08433131, -1.044999  , -1.0017225 , -0.51036924,
  -0.7913121 , -0.1908436 , -0.4403603 , -0.46155736, -0.17464972,
  -0.00893749, -1.0564756 , -3.3511248 , -0.65461445, -0.01827502,
  -0.716487  , -0.5233208 , -0.23278326, -0.33732754, -0.17730358,
  -0.8606807 , -1.076211  , -2.0547316 , -0.21484345, -2.5263505 ,
  -0.25477988, -3.3623893 , -0.70440155, -0.21876018, -1.4281216 ,
  -0.28886873, -1.5227098 ,  0.01537509, -0.19228882, -0.31358147,
  -0.87606615, -0.164419  , -1.8989093 , -0.94813675, -0.12501235,
  -0.6918805 , -0.46267292,  0.13831827,  0.15278481, -0.541015  ,
  -1.6779072 , -0.62854743, -0.20529214, -0.29793248,  0.05753939,
  -1.3298137 , -0.42736527, -0.12528592, -0.6093913 , -0.711359  ,
  -1.8164456 , -1.7176931 ,  0.03633687, -0.47856802, -0.29095602,
  -0.31219664, -0.15578109,  0.31374943, -0.36744255, -0.48529682,
  -0.5428111 , -0.6495158 ,  0.00759147, -0.01112496, -0.14360261,
  -1.8173056 , -0.41147223, -0.1827137 , -1.3830148 , -0.25287354,
  -0.70433265, -0.43579495, -0.24721858,  0.03020614, -0.5395004 ,
  -1.1108247 ,  0.01919674, -0.3134324 , -0.40654528, -0.3319709 ,
  -0.34647834, -0.50383365, -1.0209718 , -0.45558408, -1.5193479 ,
  -0.2659052 , -0.649766  , -0.47487754, -0.70352024, -0.7712729 ,
  -1.145148  , -0.74271023, -0.19136557],
 [ 0.8424993 ,  1.2457443 ,  0.40152135,  0.12836707, -0.48950037,
   1.5409042 ,  0.00833383,  0.25098148,  0.23022662, -0.00825311,
  -0.28733757,  0.19012423,  0.45466244,  0.2083559 , -0.16876951,
  -0.09001368,  0.23677583, -0.14732282,  0.45730746,  0.6246552 ,
   0.03671856, -0.11461122, -0.3111269 ,  0.1249539 ,  0.43585107,
   0.08745759,  0.61467105, -0.4225687 ,  0.12177054,  0.06038494,
   0.15436342,  0.11957501, -0.287559  ,  1.078276  ,  1.1415372 ,
   0.45582154,  0.02373558,  0.37796333,  0.2982538 ,  0.11693061,
   0.7334818 ,  1.5195196 ,  0.25588384,  0.12960333,  0.5881246 ,
   0.74623483, -0.51115614,  0.43244147,  2.6745245 ,  0.08073167,
   0.5433669 , -0.36801612, -0.9135785 , -0.39519295,  0.06093649,
   0.20514159,  0.5251003 , -0.27416593, -0.5990632 , -0.26190898,
   1.0025272 , -0.63417584, -0.17821415,  0.1064873 ,  0.06477137,
   0.399115  ,  0.63282686, -0.3514644 ,  0.15770936,  0.00649099,
  -1.2625598 ,  0.7108904 ,  0.15119663, -0.98300797,  0.14730144,
  -0.11117971, -1.1747049 , -0.9912848 ,  0.01798138, -0.0678543 ,
   0.8863592 , -0.13341433,  0.18239103,  1.0111974 ,  0.13572156,
  -1.6818376 ,  0.70629025,  0.09370738,  0.59620166, -0.41970092,
  -0.9071405 ,  0.08218503, -0.2423867 , -0.33765042,  0.14603533,
  -0.08111446, -0.19341095,  0.22881499,  0.55692536,  0.5563275 ,
  -1.3488357 ,  0.44124195, -0.51062953,  3.5794592 ,  0.02069102,
  -0.16943838,  0.16075659,  0.2729322 ,  0.4824498 ,  0.2883426 ,
   0.6648976 ,  0.0973297 ,  0.14520985, -0.06129002, -0.6926229 ,
   1.0305026 ,  0.44411105,  0.3175397 ,  0.43837973,  0.67459446,
   0.65957683,  0.4976016 , -0.2310713 , -0.04467225, -0.33358562,
   0.290046  ,  0.864227  , -0.61101764],
 [-0.0119508 , -0.40921214, -0.7899315 ,  0.3073574 , -0.7922173 ,
  -0.20147479, -0.29305008, -1.1992927 , -0.642406  , -1.7202309 ,
  -0.6154857 ,  0.09734194, -0.77077675, -1.3646325 ,  0.41084138,
   0.32358405, -0.7357767 , -1.3533729 , -0.5378831 , -0.27023515,
  -0.07969575, -0.6200052 , -0.34967518,  0.3348714 ,  0.166361  ,
  -0.11564478, -0.55573004, -0.5261387 , -0.65016365, -0.28889522,
  -1.6944822 , -0.15034604, -0.64252853, -0.77437323, -0.15864863,
  -0.5833163 ,  0.16343078,  0.11836819, -0.00004323,  0.28479326,
  -0.2823422 , -0.27188885, -0.5414918 , -0.84005   , -0.59996074,
  -0.44785473, -1.058373  , -0.32916203, -2.4004228 , -0.42376885,
  -0.22018792, -0.5665389 , -1.2100981 , -0.36449748, -0.32573703,
   0.12682351, -0.43818426, -1.2800921 , -0.0302351 , -0.13971576,
  -0.70598894, -0.55829173,  0.4172749 , -0.7075285 ,  0.10311084,
  -0.38092738, -1.3756065 , -0.6679333 , -1.0245327 , -0.7189301 ,
  -0.34648868, -0.1357338 ,  0.38915858,  0.6900327 , -0.2524701 ,
  -0.2910431 , -1.6221192 , -0.4142281 , -0.8752984 ,  0.30512396,
  -0.25863123,  0.06461244, -0.17829975, -0.2275392 , -0.35922822,
  -0.8329337 , -0.95116824,  0.5098796 ,  0.07680439,  0.10873042,
  -0.5823681 , -0.11744519,  0.6766729 , -0.89380836, -0.22057275,
  -0.3788207 , -0.46447557,  0.19756974, -0.14492594, -0.6842733 ,
  -0.25733954, -0.34263006, -0.21216072, -0.8776635 ,  0.00803232,
  -0.3423893 , -0.26208594, -0.4394623 ,  0.22388306, -0.01758935,
  -1.2593309 , -1.2127589 ,  0.29859212, -0.45439285, -0.16998029,
  -0.47278613, -0.15909356, -0.59247684, -0.21890776, -0.38492048,
   0.11647392,  0.19321223, -0.05543578, -0.2417565 , -0.7777713 ,
  -0.8254841 , -0.8647557 ,  0.12731798],
 [ 0.46141395,  0.15875056,  0.06708817,  0.14361385, -0.28025872,
   0.07305765, -0.78848594,  0.3299841 , -1.738545  , -0.25300118,
   0.30281103, -0.5035739 ,  0.03059838, -1.7469432 , -0.20331496,
   0.12873279, -1.4824443 ,  0.12196898, -0.58492607, -0.08313482,
  -0.3803165 , -0.50291777, -0.09883334, -0.43423432,  0.00242385,
   0.180539  ,  0.42838776, -0.46708778, -0.56882155,  0.3144643 ,
   0.39964867,  0.8133587 ,  1.2059385 , -0.8079954 ,  0.11656277,
   0.21318468,  0.49921173, -0.39386958, -1.3430192 ,  0.16806272,
  -0.77313024, -1.7292001 , -0.18745129, -0.2370005 ,  0.36981893,
   0.47334495, -0.24288441, -0.29307202,  0.12773335,  0.40944156,
  -0.01628322,  0.2610038 ,  0.36186   ,  0.27860826,  0.64360267,
  -0.0377467 ,  1.0941758 ,  0.7242158 , -1.438061  ,  0.52608705,
   0.57148546, -0.06423713,  0.7764158 , -0.04483593, -0.23807149,
  -0.28205854,  0.00187006,  0.5395588 ,  0.4008223 , -0.37821263,
   0.5915128 ,  0.7123132 , -0.36080578, -0.12436549,  0.31671688,
  -0.13520946, -0.30123323, -0.03976141, -0.26424542, -0.106541  ,
  -0.2031407 , -0.598153  ,  0.5360712 ,  0.0314831 , -0.12849021,
   0.30147573,  0.45405462,  0.17611594, -0.08284628,  0.6237227 ,
   1.0997083 , -0.7898747 , -0.11379528, -2.173972  ,  0.07598363,
  -0.13480867, -1.0485647 , -0.01439495, -0.31884965, -0.8302737 ,
   0.00599887, -0.06336489, -0.37417185,  0.12548862,  0.6426897 ,
   0.436262  , -1.0597218 ,  0.03184439, -0.5977033 , -1.0387391 ,
  -1.1085621 ,  0.50438   ,  0.34259614,  0.40895122, -0.4542614 ,
  -1.4163218 , -2.090296  , -0.3564919 ,  0.38493237, -0.10411263,
  -0.41852418,  0.7540462 , -0.4398422 ,  0.28635293, -0.18582411,
  -0.20711237,  0.09424908,  0.04136706],
 [-0.25666133, -0.4340536 , -0.5844016 , -0.0922352 ,  0.12915857,
  -0.53358144, -0.4954045 , -1.3269075 ,  0.7507615 , -0.3090023 ,
  -0.7366616 , -1.3079017 , -0.8223492 ,  0.12233935, -0.46899566,
  -0.46114534, -0.64302975,  2.3183007 ,  0.49556977, -1.3898814 ,
   0.25690678,  0.3318923 , -0.43213508, -0.13678952,  0.1623365 ,
  -0.6320207 ,  0.00428682, -0.3832375 , -0.40444475,  0.30269754,
  -0.02894225, -0.29328436, -0.78553283, -0.08986776, -0.5317939 ,
   0.21454549, -0.3616921 , -0.33905128,  0.3468423 ,  0.13162445,
   0.55273366, -0.00130632,  0.5550632 , -0.4480626 ,  0.03590821,
   0.2266419 ,  1.4305912 , -0.3105112 ,  0.14399835,  0.8136106 ,
   0.42316917, -0.2450244 ,  0.13290595, -0.38905847, -0.06215165,
  -0.36944762,  0.06608312,  1.0761452 ,  0.24951205,  0.28597805,
  -1.1006424 , -0.06026479,  0.09862627, -1.4641691 , -1.1474314 ,
  -0.8845582 ,  0.01650003,  0.18091622, -0.3607572 ,  1.5270624 ,
  -0.3466285 , -1.1121867 , -0.27100363,  0.1406186 ,  0.10717659,
  -0.35981488,  0.06395967,  0.3715    , -0.7048229 ,  0.17315367,
  -0.00406441,  0.07504727,  0.20151998, -1.1555008 ,  0.43064675,
  -0.29778135, -0.1970978 ,  0.24402589,  0.2810671 , -0.19319762,
  -0.7331175 , -0.16033617,  0.25940332, -1.1509603 , -0.3199387 ,
  -0.5647094 ,  0.09454931, -0.20135531, -0.34942338,  0.18221919,
   0.88660395, -0.66372967, -0.16175973, -0.72938114, -0.5468916 ,
   0.16905819, -0.5296292 , -0.2657232 , -0.28659567,  0.18727222,
   0.3075071 ,  0.21700205, -0.18071216, -0.42350233, -0.13521098,
   0.01466598, -0.21512382, -0.4227658 , -0.21849696, -0.17617705,
  -0.90322393,  1.143304  , -1.0126556 , -0.13628672, -0.00266024,
  -0.7922987 ,  0.2976855 , -0.55661005],
 [ 0.27704528, -0.56243354, -0.2128382 ,  0.03916095,  0.7210756 ,
  -0.24014142,  0.43443537,  0.65670955,  0.69388396, -0.34458625,
  -1.2670896 ,  0.03096518,  1.0854902 ,  0.33172065, -0.09294814,
   0.20854367,  0.4102938 ,  2.0567088 ,  0.6511039 ,  0.72949475,
   1.3309771 ,  0.17934586,  0.18332314,  0.06076498,  0.26696762,
  -0.24984367, -0.6036281 ,  0.16723205,  1.051737  ,  0.57878965,
  -0.47780207,  0.6715238 ,  0.03289749,  0.51771283,  0.4534277 ,
   0.77328926, -0.31822452, -0.684199  ,  0.628618  ,  0.08974157,
  -0.28716522, -0.73380256,  0.66155744, -0.28289694, -0.5410595 ,
  -0.26136062,  1.4603457 ,  0.87124586, -0.52141976,  0.53919053,
   0.7133533 ,  0.61330324,  1.4429035 , -0.05410213,  0.22648436,
   0.49048656, -0.0776355 ,  0.32649124, -0.6230966 ,  0.10049518,
   1.1564902 ,  0.17842917, -0.2221705 , -0.40725076,  0.00467955,
  -0.06748734,  0.03224554,  1.1305484 ,  0.7284808 ,  0.20364043,
  -0.35680595,  0.17271215,  0.6671484 ,  0.1632617 ,  0.42170396,
   0.15792383,  0.40064788, -0.55796635,  0.26284388,  0.12827373,
   0.53277045, -0.5470779 , -0.03872037,  2.2767987 , -0.1257228 ,
  -0.30367765,  1.0224015 ,  0.22436994, -1.3823012 ,  0.11628888,
   0.09782439,  0.05058862, -0.12175691,  1.1172284 ,  0.10389872,
   0.2399415 ,  0.27768305, -0.11999046, -0.05214316,  0.9844056 ,
  -0.33214313, -0.55364764, -0.06103625,  0.36355832,  0.20956713,
  -0.19148237, -0.05056014,  0.02404786, -0.0282017 ,  0.09000015,
   0.746439  , -0.482294  , -0.1889196 ,  0.04766144,  0.36452204,
   0.6922702 ,  0.94648874,  0.40977874, -0.4230209 , -0.6370475 ,
  -0.67562044,  0.41384563,  0.29556492, -0.3467297 ,  0.6197423 ,
   0.2176405 , -0.12237266,  0.16824844],
 [ 0.23912035, -0.26231566, -0.30115741, -0.3791776 , -1.6893294 ,
   0.4150323 , -0.50867915, -0.8672006 ,  0.12046674,  0.04398585,
   0.24362399, -0.47266158, -0.45364517, -0.65471995, -0.32619533,
  -0.21726264,  0.30080745, -1.289014  , -0.49592265, -1.1857237 ,
   0.00758603,  0.01814063,  0.26340047, -0.09229647,  0.39404458,
  -0.7835581 , -1.0316974 , -0.69506377,  0.8196232 , -0.23161352,
  -0.77587175,  0.6394671 , -0.10887453,  0.8156636 , -1.0760245 ,
  -0.31734228, -1.106001  ,  0.04324701,  1.2585428 ,  0.21083435,
   0.6647376 , -1.7149746 ,  1.0512035 , -0.47467014, -1.8482814 ,
   0.16702855,  0.21815737,  0.15063773, -0.758372  , -0.5667967 ,
   1.4275774 ,  0.49204877,  0.54213923,  1.1752746 ,  0.55731237,
   0.49721608,  0.40826848, -0.10123818, -0.37633342,  0.1516615 ,
  -0.89643353,  0.45244986,  0.23467046,  0.2744433 , -0.15044138,
   0.4452458 , -0.7398207 , -1.437318  ,  0.14921132, -0.29728216,
   0.39821035, -0.87412673, -0.37051114,  0.48042452,  0.9758054 ,
   0.26271674,  0.32415754,  0.0399811 , -1.6705455 ,  0.30107257,
   0.89677733,  0.43998295, -0.6745152 ,  0.7741083 , -0.6030465 ,
  -0.18544698,  0.46376637,  0.18183178, -2.43313   , -0.26370546,
   0.13328987, -0.39037797,  0.44648024, -0.78667927, -1.3591834 ,
   0.53492236, -0.1658977 , -1.0136788 , -1.1041417 , -0.79453564,
  -0.00713657, -0.41517928, -0.07072738, -0.18263307, -1.364044  ,
   0.09008153,  0.48912472, -0.23560631,  0.29393244, -0.6301028 ,
  -0.60058355, -1.7081774 ,  0.14662178,  0.4927636 ,  0.33248672,
  -0.09359733, -0.16853413, -0.40043414,  0.5329575 ,  0.19734879,
  -0.994823  , -0.00091528,  0.48458576, -0.4183214 , -1.3415003 ,
  -0.26351175, -0.56157714, -0.21830955],
 [-0.28935724, -0.28945145, -0.37236613,  0.10895173,  0.01055348,
  -0.48511788,  0.5233157 , -0.4506178 ,  0.6590023 , -0.19954187,
   0.82695395,  0.12866266, -0.17028703, -0.5295642 , -0.0545388 ,
   0.44412813, -0.6177045 , -0.02862185, -0.280312  , -0.35249716,
   0.9337162 ,  0.06166167,  0.5445001 ,  0.31706542,  0.39297798,
   0.4738214 , -0.643287  , -1.0353291 , -0.8897133 , -0.03746746,
  -0.7029963 , -1.2976004 ,  0.12441029,  0.7837964 ,  0.5066963 ,
  -0.7439064 , -0.05648609, -0.27039695, -0.5704617 ,  0.03594895,
   0.5703111 ,  0.67114466, -0.37055624,  0.42185855,  0.45001557,
   0.3035655 , -0.7466035 , -0.02444501, -0.21628948, -0.4297476 ,
   0.12578145,  0.09918166,  0.15325041, -0.09586316,  0.35312608,
   0.33029738,  0.62238437, -0.5139198 , -0.07351191, -0.63888955,
   0.5567727 ,  0.11498589,  0.15097795, -0.12400624,  0.49308363,
  -0.22798237,  0.22818735,  0.25394157, -0.7206784 , -0.20648104,
   0.38404992,  0.47583133, -0.51991296,  0.3215029 ,  1.1070536 ,
   0.8457115 , -0.88210934,  0.5095444 ,  0.18708894, -0.35920835,
  -0.38206658, -0.3382671 ,  0.23327962, -0.4831633 , -0.18566331,
   0.10970341,  0.5284774 , -1.2730395 ,  0.4003315 ,  0.36066666,
  -0.06722675,  0.30458784, -0.4754906 , -0.8967726 ,  0.20763874,
  -0.2651733 , -0.6330656 , -0.03498576, -0.26804104, -0.38942584,
   0.59475625, -0.7559429 ,  0.5535133 , -0.14105594, -0.02816962,
  -0.00618166,  0.5430337 ,  0.0114409 , -0.3802951 ,  0.09912179,
   0.2670667 , -1.0139449 , -0.2695238 ,  0.3021861 ,  0.19290502,
   0.32771716,  0.52868867, -0.19627997, -0.08878994,  0.73602784,
   0.7158375 , -0.143754  , -0.8431134 , -0.04048422, -0.00517494,
   0.20104928, -0.11420131,  0.7139993 ],
 [-0.7748763 , -0.19946367,  0.56478   , -0.081307  ,  0.2739259 ,
   0.03021144,  0.52052075,  1.3368444 , -1.0279644 ,  0.1064867 ,
   0.41124097, -0.02426403, -0.17380424,  1.6998788 ,  0.05996763,
   0.20584044, -0.01922557, -0.53050816,  0.78022665, -0.01721048,
  -0.9718242 ,  0.80700356,  0.31711018, -0.38466606, -0.9489277 ,
  -0.16046564, -0.56672686, -0.00438715, -0.62070215,  0.29807466,
  -0.61655474,  0.6110269 , -0.25323656, -0.7789655 , -0.38192412,
   0.8196443 ,  0.15702786,  0.12560377,  0.03896954, -0.11868453,
   0.11540342,  1.1156693 , -0.58519703, -1.2558719 , -0.44430593,
  -0.12567522, -2.1612089 ,  2.0896108 ,  0.4397262 , -1.3559618 ,
  -0.21502793,  0.7345191 , -0.51442957, -0.1540564 , -0.41044822,
   0.44636658,  0.655663  , -0.41583315,  0.3151542 , -0.78112227,
  -0.39750794, -0.01157421,  0.1876654 , -0.28336102, -0.18155468,
  -0.8822528 ,  0.19300795, -0.41307884, -1.0124221 ,  0.28728306,
  -0.9440987 ,  0.53760666, -0.08069846,  0.37469068, -0.10983688,
   0.11388794, -3.3150642 ,  0.50743765,  0.73708254, -0.00666639,
   0.5029167 ,  0.22112165,  0.34602797,  0.53634804, -1.1123458 ,
   0.01717992, -0.40052626, -0.22953841, -0.69033045, -0.07612061,
  -0.05296361, -1.0645665 ,  0.01992161, -0.10525024,  0.33803076,
   0.4505419 , -0.16774394,  0.20533709, -0.36619902,  0.21431383,
  -1.0294315 ,  0.521007  ,  0.39095378,  0.18217884,  0.05204518,
   0.19889611,  0.4102081 , -0.21314514,  0.07638882, -0.2883772 ,
   0.72573674,  0.9871502 ,  0.15648864, -0.16366677,  0.2916234 ,
  -0.94702107, -0.7377721 , -0.15088747, -0.15793955,  0.5359346 ,
  -1.1272709 , -0.03109085,  0.6108612 , -0.12873259,  0.41992617,
   0.5563189 , -0.3683948 ,  0.5185255 ],
 [ 0.27958128, -0.1004128 , -0.6902803 , -0.4017809 , -0.67772055,
  -0.38628227, -0.73837113, -0.41520703, -0.31780764, -0.20622616,
   0.40292305,  0.48350078, -0.70750266, -0.65959644, -0.3746775 ,
  -0.22849247, -0.23173688,  0.56490797, -0.56761575, -1.6924015 ,
  -0.1879171 , -1.69651   ,  0.22743298,  0.05764389,  0.22805949,
  -1.7096441 , -0.5359233 , -0.0032681 ,  0.41149917,  0.13423227,
  -0.9543473 , -0.70116454, -0.4142543 ,  0.663879  ,  0.46825844,
  -0.50338507, -0.24554007, -0.00357551,  0.02143274,  0.07736114,
  -0.4683579 ,  0.4129592 ,  0.5248868 , -1.303185  , -0.6590383 ,
   0.91363275, -0.34173796, -2.7823596 , -1.162662  , -1.0222999 ,
  -0.49778378,  0.63726735,  0.49868515,  0.43810314,  0.0008163 ,
  -1.357376  , -0.05104634, -0.9703873 , -0.5645158 ,  0.19453295,
  -1.1636996 ,  0.3647535 ,  0.34588057, -0.1036423 , -0.75341415,
  -0.00103397, -0.5703929 ,  0.13296418,  0.03503045, -0.5252609 ,
  -0.33273265,  0.4831451 ,  0.42691937, -0.16995558,  0.51874804,
   0.22447924, -0.24331354, -0.29699782, -0.02063349,  0.1336934 ,
  -0.26181352,  0.13211603,  0.02260525,  0.03705413, -0.9965953 ,
   0.76587653,  0.41907427,  0.18071386, -0.1495106 , -0.05557254,
  -1.1475325 , -0.32998168,  0.37618047,  0.24193329, -0.9883928 ,
  -2.2420886 ,  0.12458836, -0.57498914, -0.3520221 , -0.87137073,
   0.18487467, -0.7549537 , -0.27909607, -0.5881948 , -0.17259341,
  -0.40642875,  0.15343893, -0.28921208,  0.4289355 ,  0.08619264,
  -1.2677761 , -0.93540573, -0.16295204,  0.01088015,  0.43745646,
  -0.2767763 , -0.6196627 , -0.8592735 , -0.3697911 , -0.21265264,
  -1.1525832 , -0.9528192 , -0.6704748 , -0.7022518 , -0.6145412 ,
   0.34299752, -0.34984043,  0.16807939],
 [ 0.10955847,  0.355437  ,  0.08287871, -1.4427482 , -0.63983685,
   0.22866686, -0.9104485 , -0.42050084,  0.3254012 ,  0.3097755 ,
   0.44186607, -0.18793117, -0.38937703,  0.23940639,  0.57505554,
   0.07274118,  1.3060226 ,  0.10860241, -0.7131336 ,  0.24531679,
   0.07284616, -0.41587403,  0.9493557 , -0.09307987,  0.01631907,
  -0.19526616,  0.07839587, -0.21356015,  0.34974813, -0.18540108,
   0.4290089 , -0.45796475,  0.3593683 , -0.03260645, -0.7174511 ,
  -1.2111578 ,  0.36567152,  0.64675176,  0.32715756, -0.24253136,
   0.19017845, -0.05112768, -0.11150451,  0.13126254, -0.22876872,
  -0.37412116,  0.8882304 ,  0.4857458 , -0.9487789 ,  0.31851473,
   0.10027276, -0.08496744,  0.2030837 ,  0.65762013,  0.63654983,
  -0.08660668,  0.34289783, -0.56938785, -1.1827925 ,  0.2679965 ,
  -0.21657585,  0.60919124,  0.02509914,  0.4028257 , -0.6859289 ,
   0.66766363, -1.6544793 , -0.579425  ,  0.24438126,  0.7661263 ,
   0.7331472 ,  0.03421582,  0.4648551 , -0.12865447, -0.29991242,
   0.34979236, -1.585638  ,  0.38834044, -0.39921597,  0.2460232 ,
   1.169278  ,  0.44092628,  0.47368163,  0.82750547, -0.03489961,
   0.5721017 ,  0.16578753,  1.3006343 ,  0.19917633, -0.68773156,
   1.0142033 , -0.6519332 ,  0.02766303,  0.75723416,  0.80286545,
  -0.16785677,  0.669514  ,  0.4585778 , -0.53445256, -0.4440767 ,
   0.3874398 , -0.39262655, -0.12702936,  0.69587564,  0.8135255 ,
  -0.00033456,  0.472508  , -0.06454814,  0.88449574, -0.696843  ,
   0.15679796, -0.5682361 , -0.2580137 ,  0.01689624,  0.7867159 ,
   0.15584102,  0.38353738,  0.44573244,  1.0906134 ,  1.2133157 ,
  -0.09508   ,  0.24572615,  0.7593461 , -0.13156411, -0.2743161 ,
  -0.1636859 , -0.5466735 ,  0.14012259],
 [-1.3762155 ,  0.29186887, -0.18778464,  1.0941334 ,  0.02447823,
   0.11530862,  0.55444413, -0.5850773 , -0.8065001 ,  0.6669689 ,
  -0.16586328, -0.9275468 ,  0.59011054, -0.4314928 ,  0.29511273,
   0.7356172 , -1.8459187 , -0.64339274,  0.08994259, -0.99606764,
   0.40250742, -0.7707613 ,  0.55718255,  0.27522096, -0.54861164,
  -1.1013583 , -0.7256221 ,  0.6905135 , -0.3027091 , -0.15620257,
  -0.48331302,  0.2072159 , -0.30183542, -1.1603131 , -0.57691634,
  -2.512389  ,  0.03928212, -0.121994  ,  0.3034913 ,  0.12318144,
  -0.54872847,  1.0464772 , -0.1564457 ,  0.65648764,  0.04703852,
  -0.61538684,  0.47343385, -0.3290635 , -1.6846899 ,  0.7295689 ,
   0.15150803, -0.2676119 ,  0.2661799 , -1.6151936 ,  0.16181217,
  -0.11686447,  0.50809145,  1.003002  , -1.2392226 , -0.9290433 ,
  -0.19615817, -0.07829403, -0.5681319 , -1.0651528 , -1.9456655 ,
  -0.08151157, -1.3812858 , -1.0147628 ,  0.69626445,  1.1056994 ,
   1.2625415 , -1.1346062 , -0.74107534,  0.43988755, -0.00586603,
  -0.10809598, -0.44396642,  0.2981636 , -0.87527907,  0.13244385,
  -0.3675949 , -0.489506  , -0.01131574,  0.24647082, -0.9474925 ,
   0.53659385, -0.6810317 , -0.34409088, -0.9973768 ,  0.12477053,
  -1.2037305 , -0.87512594, -0.6109238 , -1.1401474 , -0.8556814 ,
  -0.2952419 , -1.2886469 , -0.07452028,  0.36729157,  0.6552921 ,
  -0.869562  ,  0.5856996 ,  0.17796774,  0.314565  , -0.9186577 ,
  -0.36095607,  0.08178028,  0.8600672 , -0.76653886,  0.07861485,
   0.2914324 ,  0.47420684,  0.6894715 ,  0.21780501,  0.31826872,
  -2.8466372 , -1.7191901 , -0.4032862 ,  0.3414172 , -0.6409416 ,
  -0.19359575,  0.20615798, -0.63568956, -0.3248201 ,  0.27221474,
   0.33470115, -0.5132135 , -0.50512767],
 [ 0.7392928 ,  0.59089524, -0.9742534 , -0.18460281,  0.4402548 ,
  -0.34104982,  0.38763335,  0.82426625, -0.6292602 , -1.2308966 ,
  -0.45605785,  0.94375104,  1.838668  ,  0.8717494 , -0.4326177 ,
   0.79637015,  1.5750009 ,  1.7801528 , -0.46173102, -0.09052457,
  -1.439773  , -0.04092921, -0.08538571, -0.74562126, -1.0731039 ,
  -0.33812714, -0.7384881 ,  0.90789664, -0.6489124 , -0.52792805,
   0.52441216, -0.17833449,  0.15709853, -0.4384698 ,  0.21416973,
  -0.9491626 , -0.62836367,  0.05302864,  1.0468309 , -0.26831374,
   0.1769337 ,  0.7838497 ,  0.9777924 , -0.51387215,  0.30822614,
   1.6144292 , -2.754711  ,  0.26032183, -1.1526612 , -1.066362  ,
   0.9346447 ,  0.9059724 ,  0.35390997, -1.3852215 ,  0.87492925,
  -0.4140177 ,  0.7305551 , -1.8990804 ,  0.18376924, -0.7769652 ,
  -0.70474964,  0.82832605,  0.0888546 ,  0.02264424,  0.35375223,
  -0.77264965, -0.32097176,  0.8835848 ,  0.07340959,  1.0324056 ,
   0.42044637,  0.02234426,  0.08649148,  0.0539178 , -0.04032435,
   0.66985047, -0.95053893, -1.999862  ,  0.18569414,  0.0633915 ,
  -0.21021138, -0.04745023, -0.39857653,  0.0042745 , -0.27267534,
   1.0099968 ,  0.06909312, -0.11151015, -0.15790868,  0.20738497,
   0.9938456 , -0.6468791 ,  0.02992204,  1.2950847 , -0.26469702,
  -0.21375854, -0.5237904 , -0.7766238 ,  0.00919597,  0.8690744 ,
   0.20079502,  0.18207055,  0.5441095 ,  0.05477056, -1.2696875 ,
  -0.491044  , -0.17951047, -0.10580522, -0.2902657 , -0.90967363,
   0.94013804,  0.4638117 , -0.3602701 , -0.31164765, -0.40946224,
  -0.9094461 , -0.5421451 ,  0.2364023 , -0.36290398, -0.18311763,
  -0.87626576, -1.3067428 ,  0.31927523, -0.10994455, -0.02461256,
   0.23265474,  0.07773545, -0.29417685],
 [ 0.15814975, -0.5939848 , -0.14023381,  0.1058989 , -0.6017014 ,
  -0.00354658, -0.64730006, -0.15731986, -0.8170197 , -0.05727968,
  -0.12463682, -0.18882862, -1.025367  , -0.5655673 , -0.43852568,
   0.20767277, -1.2301667 ,  0.4279    ,  0.06143922,  0.4486353 ,
  -0.0628676 , -0.6244539 , -0.21846692,  0.10414275, -0.18191205,
  -0.361574  , -0.37285212,  0.09632791,  0.33500534,  0.1256949 ,
   0.7393477 , -0.6765875 , -0.5289108 ,  0.44874567, -0.1560984 ,
   0.05744743, -0.06306519,  0.02318548,  0.29175422, -0.62359095,
  -0.96839523, -1.2815925 , -2.351305  , -0.23018269, -0.06198292,
  -1.1914418 , -0.36462238,  0.12676619, -1.595433  ,  0.26947635,
   0.10296086, -0.4758821 , -2.2341533 ,  0.39097926, -0.6641569 ,
  -1.080374  , -0.26338768, -1.7748265 , -0.24468467, -0.44682312,
   0.19883694, -0.78181964,  0.02648352,  0.08121417, -0.13070402,
  -0.46072546, -1.3763345 ,  1.1557425 , -0.7332332 , -0.29479995,
   0.08229768,  0.127139  , -0.10843955,  0.16385661, -0.57623714,
  -1.5203083 , -1.9927138 ,  0.08479277, -0.03963264,  0.16848177,
  -0.09201875, -0.50015175, -0.20417264, -1.1115774 ,  0.41865826,
  -0.9655319 , -1.0685712 ,  0.08363456, -0.10824118, -0.12739927,
  -1.4890184 , -0.28825057,  0.06826841, -0.2026864 , -0.1577396 ,
   0.6627679 , -1.7949079 ,  0.09419888, -0.11308735, -0.45015612,
   0.692739  , -0.4957293 ,  0.41780725, -0.01933493, -0.12778515,
  -0.18322416,  0.14269137, -0.01265768, -0.13970643, -0.37020957,
  -0.70794606,  0.12806147, -0.06884849, -0.11156664, -0.20763491,
  -0.11913225,  0.07575619, -0.23635893, -0.06432162, -0.6152603 ,
  -0.06900099,  0.09926391, -0.67592674,  0.07957041, -0.3000939 ,
  -0.22419344, -0.8224365 ,  0.26286244],
 [ 0.7397526 ,  0.50576967, -0.21709631,  0.13538924,  1.5377471 ,
  -0.15981016, -0.31584772, -0.51436245, -0.38598874, -0.5325024 ,
  -0.18338178, -0.44419265, -0.17759372,  0.5131539 ,  0.46018904,
  -0.0816604 ,  0.95169646,  1.9219189 ,  0.8882695 , -0.3185078 ,
   0.971768  ,  0.11298049,  0.81461024, -0.01699661,  0.23740514,
  -0.00548049, -0.2042529 , -0.45021534,  1.3188798 , -0.21742532,
   0.538498  ,  0.42048144,  0.20084518,  0.97878945, -0.4314293 ,
   0.04514127, -0.21107724,  0.19371627, -0.08171131, -0.47083467,
   0.13169952,  0.59862053, -0.24347514, -0.38716862, -0.11737554,
  -0.03026134, -1.3139741 ,  0.736853  ,  1.5499568 ,  0.91346514,
  -0.02636086, -0.2782266 ,  0.840445  ,  0.68137145,  0.09252595,
   0.06577227, -0.09517253,  1.4037805 , -0.262353  ,  0.7385286 ,
  -0.27831283,  0.24606583, -0.2609263 , -0.28422362, -0.34790462,
   0.03501755, -0.5078806 ,  0.56368357,  0.5817491 ,  1.043613  ,
   0.87141645, -0.08614226,  0.16772425,  0.45270172,  0.53748506,
   0.230693  ,  1.2607608 ,  0.718722  , -0.07009615,  0.02656957,
   0.47227982, -0.22886184,  0.37962562,  0.5421625 ,  0.00723226,
   0.7651392 , -0.0931735 , -0.13109775, -0.4229628 , -0.02103203,
   0.25721526,  1.1752951 , -0.06991784,  1.6603514 ,  0.18588541,
  -0.5712518 ,  0.5711617 , -0.08659714, -0.6582834 , -0.14906259,
   1.8124312 , -0.7501005 ,  0.3802429 ,  0.09806783, -0.04162047,
  -0.14410041,  0.7011626 ,  0.06040612,  0.00935805,  0.58436614,
  -0.427841  ,  0.28270686, -0.32316285,  0.61361295,  0.2944465 ,
   0.58237404,  0.10892762,  0.35657337, -0.1029245 , -0.27164212,
   0.21606466, -0.549691  ,  0.06827599,  0.21604794,  0.3307001 ,
  -0.12467853,  0.08579485,  0.48169994],
 [ 0.35213527,  0.10597434,  0.5768238 , -0.04271871, -0.33063868,
  -0.07632209, -0.43032086, -0.07029051, -0.2657491 , -0.16412458,
   0.6454329 , -0.30056083,  0.20744798,  1.3272544 ,  0.7173409 ,
   0.38730997,  0.8425338 ,  0.14235586,  0.3410986 ,  0.3817489 ,
   0.04080658,  0.64471257, -0.00299557, -0.17943323,  0.06538737,
   0.23089208, -0.10587461, -0.31065667,  1.2626656 ,  0.24986078,
   0.57835066,  0.68247485,  0.5638398 ,  0.60668784, -0.27533612,
   0.36786294,  0.5630821 , -0.11718885, -0.7801324 , -0.01772528,
   0.2711839 , -0.36895663, -0.10155325,  0.15491106, -0.41322574,
   0.25930938,  1.5318102 , -0.06986842,  2.160594  ,  0.6124282 ,
   0.18192562,  0.3680053 , -0.1310495 ,  0.01589705, -0.19197524,
   0.21321936,  0.70699877,  0.40503088,  0.01529892,  0.4173102 ,
   0.3879297 , -0.23841436,  0.2772943 , -0.08729535, -0.43655327,
   0.83453643, -1.0627776 ,  1.010307  ,  0.13401   ,  0.5771907 ,
  -0.27877912, -0.13006896, -1.008762  ,  0.1476866 ,  0.54098046,
  -0.04501874,  0.557646  ,  0.07418342,  0.3669872 , -0.370435  ,
  -0.23154442,  0.01577557,  0.12659216,  0.2807379 , -0.02024173,
   1.0159507 ,  0.7722144 , -0.03882872,  0.22625749,  0.13547644,
   0.04780904, -0.8260253 , -0.06846715, -0.4301368 , -0.04933303,
  -0.09368036, -0.02025675, -0.22266291, -0.54958403,  0.2460079 ,
   0.534474  , -0.20229271,  0.41783583, -0.28786024, -0.04416102,
   0.702379  , -0.01716872, -0.37717286,  0.12399507,  0.32979977,
   0.15590766, -1.0045921 , -0.4022555 ,  0.41973734, -0.3321899 ,
  -0.7351376 ,  0.14225513,  0.6365578 , -0.37750867,  1.2766054 ,
  -0.05965083,  0.5344917 , -0.07124344,  0.4749591 ,  0.12646885,
   0.38684875,  0.34640867, -0.29027045],
 [ 0.00321468,  0.1415907 ,  0.37877506,  0.39142188, -1.7759936 ,
  -0.37031925, -0.2268609 , -0.9705976 , -0.9571499 , -0.12015998,
  -0.40356424, -0.07552534,  0.39241293,  2.211818  ,  0.6349368 ,
  -0.03143347,  0.33420542,  0.45429325,  0.21216264, -0.35730815,
  -2.3669388 , -0.2605106 ,  0.40572718,  0.03809501,  0.59824944,
   0.50872606,  0.11406135,  0.0081482 , -0.81476694,  0.01159534,
  -0.9436898 ,  0.7888049 ,  0.3495456 , -1.5867846 ,  0.20626993,
   0.08319073,  1.2042481 ,  0.55710846, -0.04967569, -0.34058452,
  -0.1896664 , -3.389169  ,  1.0509099 , -0.608606  , -0.5711713 ,
  -1.2216569 ,  0.1107914 , -0.3392539 , -0.00783045,  0.13778667,
   0.90377057,  1.620742  , -0.9041233 , -1.7366084 , -0.11217017,
   0.25247526, -0.22623388,  0.59933144,  0.18838316, -0.5873776 ,
  -0.53534853, -0.2709064 , -0.3027166 , -0.5325976 , -0.3835559 ,
   0.23973097, -1.3261139 ,  1.2569441 , -0.22961988, -1.5452648 ,
   0.6921898 , -0.25358683, -0.05454331, -0.34887788, -1.2552491 ,
   0.34463716, -1.9388276 , -0.36246333,  0.5694627 , -0.452778  ,
  -2.6345787 ,  0.00890301,  0.8134179 ,  0.14640531, -0.30691928,
  -0.28374735,  1.2428918 , -0.5261031 ,  0.2420407 ,  0.94426817,
  -0.61752445,  0.40968493, -0.5436828 ,  0.4842986 , -0.08155759,
  -0.84569234,  0.21757211,  0.701334  , -0.01190817, -0.62836134,
   0.3353194 ,  0.26935902,  0.76310474, -1.1037916 ,  0.2808227 ,
   0.96257013,  0.01599138,  0.25410822, -0.6631586 , -0.97920114,
  -1.5848155 , -0.04387258,  0.12287021, -1.3055886 , -0.43994865,
   0.05606484,  0.00164234, -0.54363763,  0.10906349, -0.54997087,
   0.13108885,  0.87943554,  1.7442452 ,  0.03701806,  0.41284028,
   0.40226826,  0.57756406,  0.76454884],
 [-0.7805378 , -0.6548216 , -0.35888448,  0.72192085,  0.38927   ,
  -0.16922872, -0.36706454, -0.17672938, -0.20075779,  0.45382053,
  -1.1608828 , -0.7273459 ,  0.35409772, -1.626985  ,  0.8657046 ,
   0.06451317, -0.49262717, -0.95479983,  0.0932877 ,  0.22314365,
   0.04208599, -0.6485668 ,  0.29831383, -0.2241827 , -0.04363922,
  -0.309375  , -0.22945674,  0.82921803, -0.5967452 ,  0.7555191 ,
   0.00203247, -0.66674703, -0.53881633,  0.18999329,  0.04192648,
   0.04664283,  0.05732002, -0.94783115,  0.08611572,  0.30633143,
  -0.91321784,  0.00883532, -0.18752626, -0.4636474 , -0.33919072,
  -0.46877027,  0.822313  ,  0.13092601,  1.1997898 ,  0.0287481 ,
  -1.4908339 ,  1.6349742 , -0.17138714,  1.0039855 ,  0.4056738 ,
  -0.5614917 ,  0.3411634 , -0.02081349,  0.6432416 , -0.8266118 ,
   1.1344244 ,  0.3559709 , -0.00281105, -0.7631982 , -0.5677251 ,
  -1.0864419 ,  0.779434  , -0.9619665 ,  0.23102053, -0.41024974,
  -0.5768293 , -1.1845689 , -0.43924218,  0.20078798, -0.17137703,
  -0.0217371 ,  1.2414742 ,  0.53280646,  0.01164883,  0.3972993 ,
  -0.20076829,  0.21260591,  0.14452519,  0.7628449 , -0.04892355,
  -1.2625906 ,  0.7393167 , -0.93381685, -0.6267788 , -0.39241534,
  -0.31225666, -0.89850056, -0.38226816,  0.0879838 ,  0.14089093,
  -0.32669866, -1.0503805 , -0.2601595 , -0.41900116,  0.85257757,
  -0.17344321, -0.68224853, -0.12676752, -0.69376475, -0.15582629,
  -0.08530042, -0.07527816,  0.6245374 , -0.40330306, -0.26932582,
  -0.1666704 , -0.9337407 ,  0.00569197, -0.17537951, -0.3042273 ,
  -0.9592914 , -0.48425934,  0.13143957,  0.8584902 ,  0.14779663,
   0.08621909, -0.48040068, -1.363883  ,  0.23728564, -0.36235806,
  -0.45686173, -0.27693918, -0.5611278 ],
 [ 0.286533  , -0.23207223, -1.0075784 , -0.31014717,  1.7464106 ,
   0.30809495, -0.06113858, -0.48545277,  0.21964744, -1.2464654 ,
   0.36381888,  0.24401394,  0.5048851 ,  1.8405883 ,  0.03147434,
   0.19890966,  0.44086787,  1.3095154 ,  0.50007635, -0.36042324,
   0.5877927 ,  0.48921794,  0.08122252, -0.03923399,  0.4966644 ,
  -0.759779  ,  0.23287717, -0.7382121 , -0.16504033, -0.39729565,
   0.49871683, -0.99931943,  0.31996927, -0.41743737,  0.12150127,
   0.29062107,  0.12531446,  0.25474444, -0.33911526, -0.08007351,
   0.90087706,  1.1693138 ,  0.25667474, -0.5909831 , -0.68625754,
   0.978154  ,  0.5794307 ,  0.708199  ,  1.367145  , -0.1699138 ,
   0.99527764,  1.1046245 ,  0.19172949,  1.130253  ,  0.06956307,
  -0.43351397, -0.10362837,  0.50727147, -0.06658179,  0.90618706,
   0.8321798 ,  0.11565544,  0.13460359, -1.2280176 , -0.70145804,
  -0.5475128 , -0.08062052,  0.5283387 ,  0.4609491 ,  0.9097553 ,
  -0.7784253 ,  0.13749751,  0.3441447 ,  0.40219682,  0.38148883,
   0.2554132 , -0.07027521, -0.38248026,  0.12664303, -0.00767094,
   0.11034106, -0.40830553,  0.44332054,  0.73987013,  0.2221997 ,
   0.95883334,  0.29075927, -1.2714752 ,  0.32169124, -0.05534204,
  -1.5787432 , -0.58864933, -0.12195209,  0.6351424 ,  0.2753682 ,
  -0.2880435 ,  0.30935332,  0.10352612, -0.62318444, -0.9082524 ,
   0.43087208, -1.0427251 ,  0.16766584,  0.10231575, -0.24474852,
  -0.8532432 ,  0.83767074, -0.35808003, -0.43902394,  0.88621527,
  -0.41722754, -1.4580159 ,  0.47107995,  0.29913694,  0.89389044,
  -1.1092361 , -0.82127154,  0.416526  ,  0.00600941,  0.75247365,
   0.3888621 , -0.6316093 ,  0.9498216 ,  0.33166802,  0.3323711 ,
  -0.03429044,  0.19636944, -0.3823699 ],
 [ 0.13680531,  0.12586549,  0.12591557,  1.5469108 , -0.00908534,
   0.14622824, -0.8737348 ,  1.0733688 ,  0.5346896 , -0.56284195,
  -0.68203974, -0.27132434, -0.4099662 , -0.1534611 , -1.0552304 ,
  -0.08125141, -0.39835814, -0.27230632,  0.4751498 , -0.3167407 ,
  -0.24580221,  0.08999138,  0.485659  , -0.4166664 , -0.6265416 ,
  -0.06332054, -0.31527266,  0.17258653, -0.10587136,  1.0189468 ,
   0.01568238, -0.1906459 , -0.8920775 , -0.6356104 ,  0.00925814,
  -0.24876614,  0.5601511 ,  0.12254149, -1.5067354 , -0.22465979,
   0.40743268,  0.2389698 ,  0.4174951 , -0.7542604 , -0.47261405,
   0.787867  ,  0.6908047 , -0.02353887,  1.2444366 , -0.79138505,
  -0.2872919 , -0.09324118,  0.11813975, -1.0673184 , -0.48622954,
  -0.97745585,  0.3981442 ,  0.56570536,  0.27457598, -0.11448866,
  -0.30994597, -0.56991994, -0.4544109 ,  0.46705317, -0.87314355,
  -0.39931357, -0.30590245, -0.5996117 , -1.7780449 , -0.12303229,
   0.0405177 , -0.4372379 , -0.62780446,  0.4398945 ,  0.03537238,
  -0.83682877, -0.08969948, -0.4592432 , -0.17903732, -0.03279267,
  -0.2797767 ,  0.04005071, -0.49338207, -0.35240275,  0.37700528,
   0.50499004, -0.7297603 ,  0.5518224 , -0.6603336 , -0.67641675,
  -0.58858603, -1.3741466 , -0.09965213,  1.173943  , -0.46615037,
  -0.6411714 ,  0.60113764,  0.04000418, -0.17890221,  0.57987595,
   0.2585979 , -0.37700808,  0.10536774,  0.7059541 , -0.9628696 ,
  -0.8403144 , -0.04835828,  0.0151047 , -0.25379017,  0.23217437,
  -0.19663547,  1.1473992 , -0.46657205, -0.50775534,  0.25170356,
   0.44960603, -1.9377928 , -0.2766451 ,  0.8237207 , -0.03947026,
  -0.1292044 , -0.21504597, -1.1002356 , -1.2213697 , -0.23222674,
  -0.14669652,  0.19250256,  0.03439014],
 [ 0.19279972,  0.6491812 , -0.32943955, -0.9523278 , -0.4779114 ,
  -0.00491245, -0.16290554, -0.00579319, -1.0175357 , -0.6621709 ,
  -0.01450337, -0.8807886 ,  0.0750604 ,  0.20426555,  0.03822518,
   0.5386305 , -0.5164491 ,  0.06349865,  0.14547859,  0.50620437,
  -0.8961213 ,  0.1964974 ,  0.17468956, -0.00451819, -0.04522423,
  -0.06913781, -0.5082573 ,  0.54605466, -0.2127846 , -1.1607105 ,
   0.33674717,  0.2946123 , -0.5225344 , -0.15027851,  0.14125189,
  -0.75829124, -0.51418895, -0.59107643, -0.07981473,  0.10343553,
  -0.48608443, -0.24203946, -0.13708371, -0.16080219,  0.07369982,
  -1.0601659 , -0.86542165,  0.61288476, -1.6619711 ,  0.10347667,
  -0.11201962,  0.16280732,  0.39989537, -0.63769126,  0.40674022,
   0.39711118,  0.00186939,  0.4275336 , -0.16130963, -0.36866188,
  -0.11660007, -0.29384378, -0.41177467, -0.75296223,  0.2747524 ,
  -0.55064213,  0.5921262 , -0.78072596, -0.3865475 , -0.3756859 ,
  -0.40999624,  0.1739695 ,  0.19724253, -0.18744077, -0.05660067,
  -0.06646299,  0.5635416 , -0.44639573, -0.20813723,  0.04039379,
  -0.07071564,  0.12341823,  0.05596787,  0.41096488, -0.11301934,
  -0.04449632, -0.7913983 , -0.6293691 , -0.35572314,  0.17269216,
   0.46642148, -0.38959184, -0.06386464,  0.4771907 ,  0.4165876 ,
  -0.1390041 ,  0.3718308 ,  0.19923288,  0.38529024,  0.12844372,
  -0.6124267 , -0.41052675,  0.18775193,  0.2349867 ,  0.13739441,
  -0.59721744,  0.13720971,  0.24970868,  0.13179713, -0.04633157,
  -0.37526995,  0.7935524 ,  0.20624955, -0.04527342,  0.29048294,
  -1.2836492 , -0.29603317, -0.06142573, -1.2522743 , -1.1127948 ,
  -0.44958353,  0.03857602, -0.4734625 , -0.3278867 ,  0.08942375,
  -0.51458097,  0.34814465,  0.2723982 ],
 [-0.09404244,  0.5436239 ,  0.615614  , -0.07744375, -0.6084806 ,
  -0.29835588, -0.632641  ,  0.28199932,  0.4842959 , -0.35524562,
   0.33911958, -0.10380839,  0.00043233,  1.4490851 ,  0.02451077,
   0.33619076, -0.78301734,  0.6694404 , -0.6482347 , -0.31392732,
   0.6234111 ,  0.0752504 ,  0.29004925, -0.07257099, -0.2543945 ,
   0.68496996, -0.34326738, -1.2614442 , -0.8544921 ,  0.07227421,
  -0.05602361,  0.59376746,  0.7579445 ,  0.6238565 ,  0.19779809,
   0.22441399,  0.6241532 ,  0.09685148, -0.5187325 , -0.0490518 ,
   0.488013  ,  1.1526538 , -0.09617202,  0.26009563, -0.57344216,
  -0.3782778 ,  0.4987074 ,  0.47402227, -0.01730324,  0.6816182 ,
   0.05515372,  0.07451054,  0.7246306 ,  0.6958904 ,  0.24669206,
   0.14888683,  0.07114542,  0.39674845, -0.3089643 , -0.7972832 ,
   0.6355368 , -0.11443582, -0.28158966, -0.27977252,  0.43305975,
  -0.82068425, -0.7413306 , -0.6407584 ,  1.062012  ,  0.3191287 ,
  -0.7736123 ,  0.5215064 , -0.4855303 , -0.09071281,  0.20352665,
   0.11927687, -0.00620406, -0.24443793,  0.5999106 , -0.3589532 ,
  -0.3766967 , -0.28601018,  0.27099007, -0.58099043,  0.36784643,
   0.5754467 ,  0.18904987,  0.3657228 , -0.5361889 ,  1.1517278 ,
   0.18524903, -0.3798641 , -0.32201222,  0.21761915,  0.35190153,
   0.22679523, -1.6111169 ,  0.26566282,  0.4234187 , -0.55237556,
   0.8218994 ,  0.49486876, -0.17699791,  0.24305892, -0.05522706,
  -0.22638889,  1.0172765 , -0.09011602, -0.41663378,  0.00998131,
   0.09795038, -1.057315  , -0.10180816, -0.02451939, -0.2901155 ,
  -0.6335885 , -0.63081276, -0.01847037,  0.06851868, -0.37454692,
  -0.12460896,  0.08097893, -0.32201618,  0.39148614, -0.04190482,
   0.38478637,  0.07339934,  0.30100495],
 [ 1.2087523 , -0.64368576, -0.26257357,  0.25845435,  0.6681478 ,
  -0.13240992, -0.312806  ,  0.6629364 , -0.3535346 ,  1.1678655 ,
  -0.8378    , -0.8464607 ,  0.53010845,  1.054099  , -0.13421835,
   0.6078241 ,  0.00413108, -0.3479308 ,  0.8852366 , -0.16760091,
  -0.10774559,  0.6809787 ,  1.0963372 , -0.5334845 , -0.70025235,
  -0.00923058,  0.1510895 ,  0.7232854 , -0.40642986, -0.12933275,
   0.34786874, -0.02949438,  0.06318291,  0.50740063,  0.3008482 ,
  -0.65056485,  0.11773645, -0.250269  ,  0.35407698, -0.08603913,
   0.9530944 ,  0.40397048,  0.7591908 , -0.08962737, -0.71322703,
   1.575562  ,  0.8643852 , -0.27339846, -0.71148   , -1.0513567 ,
  -0.29228792, -0.63753235,  0.15283771,  1.6345781 , -0.06137392,
   0.19104658,  0.07984035,  0.11296584, -0.21839923, -0.5956926 ,
   0.17711128, -0.5137969 , -0.04314246, -0.67292875, -0.0327189 ,
   0.53941315,  1.2750883 , -0.59592295,  0.41503516, -0.26916507,
   0.4963706 ,  0.21875851,  0.19699536,  0.02776161,  0.88580024,
  -0.28389862, -0.45277026, -0.6384721 ,  0.7671722 , -0.45541483,
  -0.48429114, -0.18260546, -0.59697306,  0.5904038 ,  0.36806223,
  -0.7894382 ,  0.562852  , -0.42818448, -0.32651943, -0.2724295 ,
  -0.23530093, -0.9617086 ,  0.00806975,  2.05602   , -0.1380331 ,
   0.35120916,  0.3265932 , -0.06869317, -0.01682174, -0.3839556 ,
   1.1193284 , -0.6578443 ,  0.68670315,  0.3754475 , -0.20436625,
   0.7089059 ,  0.03972318, -0.44029602, -0.40242884,  0.1929367 ,
  -0.20423545, -1.1738867 ,  0.00744935,  0.01268665, -0.24911596,
   0.9626096 , -0.26641765,  0.29624784,  0.3419747 ,  0.5552775 ,
  -0.2006802 , -0.18623812, -0.91924405,  0.17772445,  0.1328785 ,
   0.36265787,  0.87227315, -0.14376988],
 [-1.1097713 ,  1.7587292 , -0.22891743, -0.5613659 , -0.36635807,
   0.19440566,  0.25747374,  0.68613476,  0.826825  , -0.9341422 ,
   0.24998873, -0.36084828, -0.17720293,  0.00513639, -0.30300435,
  -0.32497194,  0.21537668,  1.2452725 ,  0.57911515,  0.7253992 ,
  -0.59077   , -0.5400197 , -0.79138166, -0.4748299 ,  0.5349198 ,
   0.06242004, -0.08672855,  1.3399442 ,  0.02143901, -0.40047583,
   1.2090856 ,  0.5840823 ,  0.01029453,  0.66278356,  0.649878  ,
  -1.5031649 , -0.04249629, -0.20719415,  1.5091224 ,  0.7164479 ,
   0.4512924 , -1.3713171 , -0.31116453,  0.25653815,  0.3143482 ,
   0.1448465 , -2.2236874 ,  0.11874819,  0.0572227 ,  0.13445769,
   0.10833436, -0.66617507,  0.17412679,  0.01463389, -0.11000764,
  -0.30336553,  1.6892277 ,  0.804947  , -0.07703685,  0.6516453 ,
   2.533379  , -0.5503983 ,  0.01598478, -0.02949509,  0.16978243,
   0.14831974,  0.5041073 ,  0.5621829 ,  0.15456514,  0.02973833,
  -0.80817765,  0.18650903, -0.44458148, -0.07721066, -0.2462702 ,
   0.24163869,  0.5039936 ,  0.6876429 ,  0.29487   ,  0.19209968,
   1.0901963 , -0.55699086, -0.18917233,  0.05182995, -0.64489955,
  -1.0873877 ,  0.59510833,  0.1768304 ,  0.13694713, -0.35992754,
   0.10989729,  0.06553978, -0.09508011, -0.06362958, -0.05108132,
  -0.10673742, -0.02997816, -0.2384    , -0.07967284,  2.5862575 ,
   0.21937202, -0.5805434 ,  0.16088873,  0.6805016 , -0.29238486,
  -0.5131525 , -1.2352151 ,  0.09701482,  0.0948698 ,  0.45914087,
   0.93803   ,  0.25085908, -0.0802479 , -0.21990617, -0.2547089 ,
  -1.582443  ,  0.62440133, -0.01164594,  1.0218043 ,  0.4292614 ,
  -0.25493637,  0.00884727, -0.2802267 , -0.16873349,  0.28886625,
  -0.3801392 , -0.66657466, -0.07106073],
 [ 0.11695565,  0.5803449 , -0.77821904,  0.5453187 ,  1.3238318 ,
   0.6994861 ,  0.46535456, -0.37502804, -0.38973632, -0.16303387,
   0.34206152, -0.11391403, -0.785585  , -0.901751  ,  0.04643055,
   0.04420163,  0.2546771 , -0.7510381 ,  0.22555588, -1.0260222 ,
  -0.23836088, -0.00680816, -0.27334726,  0.28456602,  0.53721786,
  -0.6469246 ,  0.53897697, -0.43806198,  0.00624103,  0.02809512,
  -0.49000686, -0.1636119 , -0.37588084, -1.2540274 ,  0.8789136 ,
  -0.21049157,  0.12606142,  0.15693071, -1.3038611 , -0.264439  ,
   0.8589859 , -0.32323998, -0.72295433,  0.72573674,  0.53404015,
   0.05756226, -0.11555716, -0.17416853,  0.18489417,  1.0589797 ,
  -0.12668858, -0.5142035 ,  0.42636207, -0.12166438, -0.1184579 ,
  -0.3890484 , -0.16121535,  1.2939605 ,  0.07919125, -0.5156869 ,
  -0.16218452, -0.25859493, -0.26021037, -0.06706722, -0.16294487,
  -0.6151041 , -0.7696429 ,  0.04209266,  0.26516482,  0.76465565,
  -1.2290833 , -0.44572118, -0.23628539, -0.03793678,  0.82441086,
   0.14888503, -1.1839857 , -0.07959467,  0.1792001 , -0.07196748,
  -0.4072166 , -0.18181047,  0.56713086, -1.3064042 , -0.20173863,
   0.5068408 , -0.3332607 , -0.16309811,  0.04892252,  0.3760477 ,
   0.5345175 ,  0.17411736, -0.22886603, -0.70018303,  0.93400544,
  -0.0338545 , -1.0455703 ,  0.4486406 ,  0.09397   , -0.2513784 ,
  -0.63276577,  0.30176228, -0.19018176,  0.44046548,  0.08954471,
   0.15565538,  0.09771606,  0.14702417, -0.23715779,  1.0060002 ,
  -0.54916924, -0.23495048, -0.00481378,  0.27273008, -0.46176472,
  -0.5983293 , -0.37821308,  0.2978648 , -0.09667437, -0.37934673,
  -0.28211114,  0.5677487 ,  0.5892135 ,  0.06312015,  0.3749557 ,
  -0.37831426, -0.0456399 ,  0.09971428],
 [ 0.03864872,  0.25108933,  0.04127422,  0.15035538,  0.06205912,
   0.26695287,  0.936972  ,  0.6253937 , -0.3105926 ,  0.3961372 ,
   0.31900483, -0.37026548,  0.00906041, -0.5725707 ,  0.6348313 ,
  -0.4460798 ,  0.21109363,  0.48935324,  0.13600981, -0.58072984,
  -0.19398907, -0.07982682,  0.01243341, -0.48203442,  0.00431393,
   0.65094465,  0.07960817,  2.121376  ,  0.708518  ,  0.09432942,
   0.71004385, -0.19954078,  1.3017356 ,  1.3453858 ,  0.7973572 ,
   0.20330207,  1.0321956 ,  0.47285342,  2.0756292 , -0.5032767 ,
   0.33921215,  1.1476191 , -0.1128309 ,  1.0314716 ,  0.12184826,
   0.27280214,  0.10527705,  1.5462842 ,  2.9161077 ,  0.86092144,
  -0.5731184 ,  0.5039658 ,  1.7944704 ,  0.7547579 ,  0.42855516,
   0.16682199, -1.3813056 ,  0.608465  ,  0.8680845 ,  0.6211109 ,
   0.8096839 ,  0.5944029 , -0.09886965, -0.16889903, -0.12490883,
   1.745479  ,  2.4236844 ,  0.46752644,  0.2820634 ,  0.8968799 ,
  -0.43795407, -0.7548418 , -0.47257578, -0.4888445 ,  0.34987205,
  -0.2682912 ,  0.08996747,  0.07572251, -0.2429261 , -0.37245998,
   0.5086801 ,  0.05059918,  0.07186009, -0.35278434, -0.22599475,
   0.44622585,  1.058826  , -0.06239896, -0.3245658 , -0.45576465,
  -0.34026188, -0.42527318, -0.21688588,  0.4036698 ,  0.11696901,
  -0.04762055,  0.12839821,  0.05688703, -0.5158216 , -0.6150048 ,
   1.2066575 , -0.11752535, -0.88464737,  0.06145422,  0.34912416,
  -0.41029307, -0.15640697,  0.23054451,  0.22803141, -0.25574586,
  -0.8114434 , -0.22034164, -0.4644098 ,  0.01680657,  0.03927534,
   0.7129366 ,  0.4829025 ,  0.09788954, -0.02837255,  0.5165533 ,
  -0.17307633,  0.12616633,  0.50296175,  0.19541444,  0.2678226 ,
   0.40149972,  0.8016836 ,  0.21255033],
 [ 0.25948068,  0.12675793, -0.50421584,  0.22031508, -0.81470007,
  -0.30295768, -0.39543232, -0.2088617 , -0.342361  ,  0.09597644,
  -0.21948263, -0.44971225, -0.3042907 , -0.06141338, -0.46057788,
   0.01982156,  0.09294594, -0.03773651, -0.22748582, -0.34298426,
  -0.4963003 , -0.88604164, -0.05113008, -0.6517334 , -0.6731722 ,
  -0.47039405, -0.57350993, -0.84276277, -0.48181027,  0.255214  ,
  -0.4002673 , -0.26627266, -0.7342983 ,  0.01668103, -0.69817585,
   0.0991316 , -0.8241221 , -0.3801472 ,  0.0427796 , -0.3508665 ,
  -0.11083152, -0.43842217, -0.3037271 , -0.3366725 , -0.29757133,
  -1.2014453 ,  0.2530202 , -0.7263647 , -0.0911696 , -0.42426968,
  -0.29861408, -0.22724067,  0.23885572, -0.7912598 ,  0.37413087,
  -0.04277279,  0.28305206, -0.077957  , -0.65530944, -0.41455388,
  -0.38795817, -0.7107726 , -0.5280172 , -0.2706885 , -0.7573018 ,
  -0.18995385, -0.06125266, -0.3901736 , -0.22398818, -0.19562007,
  -0.3084463 , -0.49419254, -0.26899293, -0.83932257, -0.70599496,
  -0.11783814, -0.29305148, -0.3658947 , -0.6232628 , -0.2695427 ,
   0.04936957, -0.31633064, -0.76228386, -0.26837814, -0.5769122 ,
  -0.17919557, -0.42921868, -0.7077776 , -0.9021973 , -0.79063755,
   0.06836867, -0.7331047 , -0.72133714, -0.21036217, -0.8160376 ,
  -0.69579613, -0.2544144 , -0.7042588 , -0.5215307 , -0.43940184,
   0.05068085, -0.5348241 , -0.10752466, -0.37298983, -0.68449897,
  -0.67367387, -0.48654506,  0.36769727, -0.9458343 , -0.5371551 ,
  -0.20488013,  0.17467797, -0.6136071 , -1.0780791 , -0.58127856,
  -1.169907  , -0.5874988 , -0.4788989 ,  0.0488428 , -0.22597596,
  -1.0747113 , -0.67830235, -0.47821367, -0.6351977 , -0.58376265,
  -0.24475491, -0.4484368 , -0.6188814 ],
 [-0.701034  , -0.30132064,  0.24794649,  0.3306547 , -0.47013855,
   0.02465691, -0.6691806 , -0.4984281 ,  0.3454751 ,  0.00872881,
   0.29023626, -0.9016736 , -0.40174824,  0.14359222, -1.4779936 ,
  -0.10167149, -0.47498092,  0.9965269 ,  0.09846634, -0.15223819,
   0.01541834, -0.02325022,  0.69623727,  0.3059779 , -1.1170905 ,
  -1.4698094 , -0.35494542, -0.8872551 ,  0.6536292 , -0.16428234,
   0.5286357 , -0.17713955,  0.16140449, -0.4137241 , -0.92868394,
   1.2379206 , -0.28435466,  0.15636186,  0.40538773, -0.74612355,
  -3.6197836 ,  0.41918495, -1.4863847 , -0.28396103, -1.929828  ,
  -1.6757082 , -0.42619944, -0.19493562, -0.23133612, -1.6520376 ,
   0.36721528, -1.9142226 , -0.5665231 ,  0.7578444 ,  0.2329036 ,
  -0.88138866, -0.03622863, -0.21129839, -0.0079449 , -0.6486879 ,
  -1.7743253 , -0.18767878, -2.2049813 , -0.8735654 , -1.6257801 ,
   0.42106652, -0.59259695, -0.17354767, -1.4763407 , -0.07889577,
   0.32424587,  0.16652772, -0.67131424,  0.21169913, -0.32585284,
  -0.95648676,  0.77533305, -0.35830292, -1.0793431 , -0.194185  ,
  -0.28249773,  0.30442733, -1.5129228 , -0.40052465, -0.71635133,
  -0.4636951 , -1.2158837 , -0.42717707, -0.01685706, -1.5015068 ,
  -0.6684315 ,  0.6769661 , -0.47756875, -2.0296295 , -0.9613645 ,
  -0.88099587,  0.33998606, -0.9668395 , -1.3731217 , -1.1737847 ,
  -0.43074945,  0.813311  ,  0.4704222 ,  0.9443264 , -0.08941408,
  -0.6117989 , -0.30308944,  0.62654155,  0.02567938,  0.5170938 ,
  -1.1514112 , -0.546934  , -1.5075208 ,  0.03463364, -1.228719  ,
  -0.08278969,  0.19827454, -0.60122   , -0.84302646,  0.14236733,
   0.2618128 , -0.7297261 , -0.9490543 ,  0.8573831 , -0.8299198 ,
  -0.41431653, -1.0722969 ,  0.3956649 ],
 [ 0.0496266 , -0.11551219,  0.29689112, -0.529363  ,  0.26350278,
   0.09948669,  0.49573627,  0.39086178, -0.23242918,  0.40579098,
  -0.18798761,  0.42826203,  0.10452259,  0.026885  , -0.53464043,
   0.09016991,  1.4054731 ,  1.2265729 , -0.531213  ,  0.22889003,
  -0.3971565 , -0.21969406,  0.30849367,  0.06449872, -0.08644392,
   0.13014305, -0.22444716,  0.5935456 ,  0.5173914 , -0.12585805,
   0.37737626, -0.32337216, -0.567306  , -0.75607246,  0.55513257,
   0.6853572 , -0.3389538 , -0.89176977,  0.25638643, -0.01811511,
   0.09926352, -0.14488009,  0.31946597,  0.22436348, -0.44118556,
   0.73702437, -0.04269349,  0.39536017,  0.7232305 ,  0.1550836 ,
   0.6838054 , -0.06110531,  0.10670805, -0.7927565 ,  1.0082803 ,
  -0.07436499, -0.35674644,  0.74632543,  0.04391167,  0.2700872 ,
   0.2721833 ,  0.40052867, -0.4337159 ,  0.19404323,  0.5474624 ,
   0.01585093, -0.15969   ,  1.0494263 , -0.05673822, -0.08645148,
  -0.626564  ,  0.04475814, -0.04380247,  0.25034997,  0.24791045,
   0.05521406, -0.26044264,  0.33005604,  0.84697586, -0.14082138,
   0.85645896,  0.19427525,  0.5965278 ,  0.78468   , -0.04875831,
  -0.3351912 , -0.16180786, -0.10010066,  0.11092115,  0.69880515,
  -0.03617951,  0.06801035,  0.02415914,  0.02383091,  0.2706052 ,
   0.27842268,  0.32255116, -0.03873879, -0.07622587,  0.96729827,
   0.40086544,  0.5567512 ,  0.26912636,  0.4575833 ,  0.0880203 ,
   0.49911076,  0.46937513, -0.07176971,  0.00208033,  0.28774822,
   0.6449375 , -0.00196191,  0.3895055 ,  0.14006008, -0.20889337,
  -0.25367787, -0.17515998,  0.23635566,  0.6093909 ,  0.22588146,
  -0.32033312,  0.23349787, -0.10736893, -0.08417847,  0.5728778 ,
   0.09971777, -0.88616997,  0.18108176],
 [-0.7136312 , -0.1828167 , -0.73747665,  0.7629199 ,  0.9525566 ,
  -0.43184724,  0.9118375 ,  1.3522067 ,  0.4161741 , -0.64639115,
  -0.70699334,  0.4492398 , -0.07896612,  0.71762717,  0.37485856,
  -0.46026352,  0.04413215, -0.4049498 ,  0.6517518 , -0.00956025,
  -0.18303788, -0.22367191,  0.635384  , -0.28135788,  0.20528893,
  -0.41808507, -0.12501238,  0.51712674,  0.00225313,  0.1808305 ,
   0.8386352 , -1.0569667 , -0.3049395 ,  0.60253555,  0.4006054 ,
   0.12195045, -0.11955062, -0.17762133,  0.4428999 , -0.14189054,
   1.3998222 ,  0.37322742,  0.07186196,  0.4104478 ,  0.7963312 ,
  -0.25581637,  0.5287334 ,  0.06656118,  0.27866423, -0.6179466 ,
   0.30365267,  0.44734257, -0.39354002, -0.3148625 ,  1.129892  ,
   0.03508093,  1.4374213 ,  0.996854  ,  0.18513903, -0.37297747,
   1.559097  ,  0.53734946, -0.19173267, -0.39465687, -0.1228256 ,
  -1.0732187 , -0.04372937,  1.0633855 ,  0.23915705,  0.17508379,
   0.70530826, -0.52882516, -1.0930696 , -0.13990887,  0.48732227,
  -0.18143503, -1.1145619 ,  0.5034641 ,  0.0243138 , -0.09342029,
  -0.26489037,  0.14011216, -0.27590963,  0.4727184 , -0.00317239,
   0.9863603 , -0.4808908 , -0.69075996,  0.00544124,  1.1924393 ,
  -0.04761968,  0.10247114, -0.389148  ,  1.5629531 ,  0.22252801,
   0.16666824,  0.68532103, -0.05183357,  0.03630761,  0.17584449,
   0.5026054 , -0.4444179 ,  0.47638622,  0.3455473 , -0.6168704 ,
   0.94962806, -0.261573  ,  0.41326284, -0.63919705,  1.2652197 ,
   0.4176517 ,  0.2976899 ,  0.14408402, -0.34591037, -0.4927904 ,
  -0.49798208,  1.1930982 ,  0.29160932, -0.41199604, -0.53154534,
  -1.0477113 ,  0.01142811,  0.49465615,  0.20125376,  0.2093608 ,
   0.65246254,  0.20922431,  0.37226677],
 [ 0.13814618,  0.62088615, -0.1604329 , -0.20408641,  0.56885874,
  -0.1870061 ,  0.73368526,  0.9690506 , -0.9847482 ,  1.0609577 ,
  -0.21282753,  0.36544153, -0.0594786 ,  0.6553002 ,  0.55924547,
   0.36644992,  0.59894955,  1.3929183 , -0.7825483 ,  0.16306569,
   0.6570279 , -0.04880488, -0.18378232, -0.31232798,  0.08461611,
  -0.31477058, -0.66797984,  1.006824  , -0.19794725,  0.19486254,
   0.12615427, -0.4188663 ,  0.09176338,  0.37228134,  0.0333744 ,
   0.41771886,  0.12013117, -0.17367712,  0.7032243 ,  0.14114699,
   0.62274206, -2.074597  , -0.2724421 , -0.21447705,  0.05139469,
   0.12513883, -1.105997  ,  2.3107543 , -0.90244275, -0.44059548,
  -0.09760665, -0.32924846, -0.13138294, -0.08469929,  0.02694557,
  -0.10138851,  0.14276157,  0.8320993 ,  0.08673847, -0.60129666,
   0.27125865, -0.27899143,  0.27407303, -0.19609426, -0.15311325,
   0.8560928 ,  0.1662346 , -0.15991534,  0.5511338 ,  0.1349412 ,
  -0.36565226, -0.21434498,  0.23309332, -0.36655137,  0.27095318,
  -0.14675537,  0.69835454,  0.5805929 ,  0.4938642 ,  0.11316443,
  -0.39827406, -0.14071815,  0.15661164,  0.43669856,  0.13419208,
  -0.76271784, -0.6424434 ,  0.98203284,  0.17786583, -0.5183639 ,
   0.2676855 , -0.18157928, -0.21429883,  0.83477104,  0.12824614,
   0.24827904, -1.0310475 ,  0.05266123,  0.8004502 ,  0.9069307 ,
   0.95465416,  0.00854538, -0.02471344, -0.16843456,  0.05813294,
   0.5841153 , -0.49312   ,  0.12498185,  0.11498076, -0.42941615,
   0.43881807, -0.2309443 , -0.04583719, -0.18461634, -0.07934554,
  -0.68443227, -0.31481358,  0.19544193,  1.0382516 ,  0.1603346 ,
   0.02484024,  0.13092141, -0.29894602,  0.05879216,  0.05065073,
  -0.3045401 , -1.7277699 , -0.5092224 ],
 [-0.21716169,  1.0362204 , -0.19360745,  0.44477046, -0.02806857,
   0.73945695,  0.70012146,  0.2988017 ,  0.13792224, -0.5179781 ,
   0.71143365,  0.72453266, -0.6014877 ,  1.0257552 , -0.24294579,
   0.53636354,  2.4532595 ,  0.33472478, -1.2394656 , -0.35485363,
   0.3313826 ,  0.5220919 ,  0.2691783 ,  0.11642535,  0.22287624,
   0.00354247,  0.04599729, -0.06576891,  0.95189726, -0.553789  ,
   0.76344115,  0.35013866,  0.3776265 ,  0.65062225,  0.63073415,
  -3.0806293 , -0.22492638,  0.05874686, -0.3514359 , -0.06665193,
  -0.42180532,  1.2723556 ,  0.27067012,  0.03986933,  1.1474086 ,
   0.5821302 ,  1.1670971 , -0.21452594,  0.16856034,  0.30327308,
   0.30498984,  0.7715259 ,  1.4698073 ,  0.96280426,  1.0554227 ,
   0.24682334,  1.9709188 ,  0.24111106, -0.7913107 ,  0.3075843 ,
   0.8590176 , -0.14246239, -0.37490815,  1.1413853 , -0.06473608,
  -1.2114339 , -0.8378969 , -3.0426579 , -0.35655075,  0.6622493 ,
   0.22006279,  0.19669048, -0.35169306,  0.16081306,  0.9864635 ,
   3.4314158 ,  0.91407955,  0.51094604,  0.09492776,  0.17480104,
   0.6703194 ,  1.418453  ,  0.16117553,  0.3260887 ,  0.97400075,
   0.7034568 , -0.27163047,  0.27398792,  0.26919818,  0.3671875 ,
   0.419321  , -0.36412942, -0.22490495,  2.836447  , -0.2276226 ,
   0.39171755, -0.18230848, -0.17826076, -0.15522403,  1.1107364 ,
  -1.1356196 ,  0.16607411, -0.16204116,  0.361686  , -0.22831954,
  -0.01453691, -0.04329224,  0.4785607 , -0.01598914,  0.79216737,
  -0.7776113 , -0.20777176, -0.00876906,  0.5464303 ,  0.7707209 ,
  -0.35642248, -0.19793154,  0.40654257,  0.19826609, -1.2082872 ,
   0.34442827, -0.05851463,  0.2569046 ,  0.33647224,  0.47859356,
   0.58002424, -0.13190763, -0.00572904],
 [ 0.3068811 , -0.44005138, -0.22504562, -1.1528769 , -1.6452438 ,
  -0.06848692,  0.5981401 ,  0.22439416, -1.6997538 ,  0.17500508,
   0.2459492 ,  0.45148313,  1.8072324 ,  0.5761796 ,  0.11851498,
   0.07990395, -0.64575684,  0.2725379 , -0.5872192 ,  1.4225721 ,
  -0.13284396,  0.26385123,  0.02783136,  0.11677156, -0.39719892,
  -0.2597898 ,  0.7004699 ,  0.82353604,  0.42139044, -0.12910958,
  -0.3130944 ,  0.47133902, -0.7677537 ,  0.1622184 ,  0.5177939 ,
  -0.38229257,  0.19245228, -1.9141145 , -0.1275287 ,  0.23234628,
   0.71329653,  0.26569963,  1.0835738 ,  0.6620721 , -0.22048186,
   0.37976873, -0.2519255 ,  0.37915614,  0.4726613 ,  1.3417239 ,
   0.70145583,  0.11720836, -0.13689409, -0.14266236,  0.08914296,
  -0.24636115,  0.23657025, -0.62161076,  0.2530821 ,  0.4614124 ,
  -0.34562886, -0.06298957,  0.63771784,  0.81026524, -0.68898225,
   0.2511357 ,  0.8711319 , -0.256933  ,  0.37591347, -1.4774739 ,
  -1.3068067 ,  0.55007404,  0.22776237,  0.25833863, -0.1950195 ,
   0.5952453 , -1.2922927 , -0.5491901 , -0.13138266, -0.07502022,
   0.09839223,  0.46942797,  0.14568996,  1.1055568 ,  0.3912903 ,
  -0.5413014 ,  0.84222543,  0.40735596,  1.279545  , -0.06756294,
  -1.2426581 , -0.7916758 ,  0.11652257, -0.5782469 ,  0.6353308 ,
   1.1304363 , -0.61612076,  0.6543397 ,  0.18338014,  0.8800528 ,
   0.29835248, -0.7362971 ,  0.39133757, -0.15361363,  0.5265319 ,
   0.2644446 , -0.11809829, -0.21100219, -0.19807175,  0.2624113 ,
  -0.13855298, -0.3403923 ,  0.389835  , -0.5343824 , -0.6119395 ,
  -0.0347927 ,  0.3099057 ,  0.5126386 , -1.1100498 , -0.558263  ,
   0.10828783,  1.2035455 , -0.0549424 ,  0.07892215,  0.47838357,
   0.33960322, -0.12086235, -0.01567498],
 [-1.2108086 , -0.5104028 ,  0.8926658 ,  0.03448609, -0.41143748,
   0.2974801 , -0.99050677,  0.66462874,  0.49536505,  0.42942026,
   0.27086818,  0.11129256, -0.79822415, -0.12610161,  0.382862  ,
  -0.66945815, -1.6291809 , -0.8651367 , -0.18854864,  0.6993437 ,
  -0.5210741 ,  0.7468342 ,  0.47443786,  0.24971753,  0.04291284,
   0.21341513,  0.14215448, -0.4432192 , -0.06265911,  0.20401022,
  -0.22375041, -0.8014744 ,  0.27349347,  0.10441881, -0.43745893,
   1.9710464 ,  0.35459876, -0.06188839, -0.63098603,  0.10395864,
  -0.21875948,  0.87058574,  0.2674947 , -0.0388024 , -0.67714477,
  -0.9045134 ,  0.18024947,  0.04821204,  0.9546387 , -0.10316616,
   0.6143304 , -1.2456852 , -0.01738553,  1.9104642 ,  0.02572287,
  -0.11284336,  0.05219923,  1.0497506 ,  0.38701895,  0.5327801 ,
  -0.74155116,  0.17667449,  0.34224293,  0.8354922 , -0.1810461 ,
   0.22006711, -1.0326114 , -0.00454569, -0.91745746,  0.5234826 ,
  -0.30907318,  0.11486821, -0.27094683,  0.04758867,  0.55706626,
  -0.90463006,  1.1019908 ,  0.06127173,  0.37741575,  0.03732695,
   0.30264902,  0.02750488,  0.5047649 ,  0.2099903 ,  0.13767442,
   0.38402468, -0.570906  , -0.11889718,  0.36639103, -0.30080524,
   0.19486411, -0.02201743,  0.03401905, -0.6412925 ,  0.58267915,
   0.12872352, -0.8427373 , -0.09121123, -0.01572651,  0.03399774,
  -2.5095892 , -0.55690265,  0.25193247,  0.2793557 ,  0.6028904 ,
  -2.3362834 ,  0.15020844, -0.0012887 ,  0.1558933 ,  0.84097767,
   0.15305711,  1.0247834 , -0.23382516,  0.1482748 , -0.04196944,
  -0.6383166 , -0.8603045 , -0.08858018, -1.0780101 , -0.232474  ,
   0.9102112 , -0.38505045, -0.60372245,  0.06299008,  0.20767668,
   1.1835235 ,  0.16489887,  0.5078674 ],
 [ 0.3052753 ,  0.3868272 ,  0.5562657 ,  0.18010905, -0.3286111 ,
  -0.05031623,  0.05438164, -0.15778483, -1.0118003 , -0.12645257,
   1.306055  ,  0.24927889, -0.46807814,  0.38349834,  0.2459434 ,
  -0.11123356, -0.27552798,  0.9804837 ,  0.18798307, -0.95961297,
   0.45027995,  0.13268325, -0.0659873 ,  0.4477562 ,  0.32165056,
   0.5177689 ,  0.10983363,  0.3258067 , -1.0830357 ,  0.3077343 ,
  -0.22733666,  0.20439616,  0.05589309,  0.3074738 ,  0.66699004,
  -1.1620708 ,  0.36413953,  0.4435039 , -0.98120147, -0.15011148,
   1.0060351 ,  0.3717025 ,  0.2513951 ,  0.2060657 , -0.17531312,
   0.34917   , -2.9740229 , -0.22367449, -1.309272  ,  0.23101674,
  -0.07763296,  0.23937762, -0.45462635,  0.48064983,  0.20741579,
   0.5974683 , -0.3590307 ,  0.5562678 , -0.3205208 ,  0.24587739,
   0.18071659, -0.10726155,  0.30045095,  0.39152914,  0.5706983 ,
  -0.07543839,  1.297092  , -0.2845739 ,  1.3072555 ,  0.49481824,
  -0.7781817 , -0.1019867 ,  0.02129078, -0.06171485,  0.6970937 ,
   0.14060643, -0.25451717,  0.21012188,  0.23858207,  0.09479711,
  -0.00196347, -0.80623853,  1.0769658 ,  0.3236531 ,  0.14873458,
  -1.2522978 ,  0.00775681, -0.04333732, -0.11862861, -0.16624731,
  -0.8679773 , -0.23609214, -0.54972124, -0.8673834 ,  0.4257447 ,
   0.13869233,  0.31044722,  0.42648077,  0.13763079, -0.69298923,
   3.225309  ,  0.25207978,  0.21171504,  0.08840303,  0.44324133,
   0.25286478, -0.4166096 ,  0.40274101, -0.6312808 , -0.65048563,
   0.0743597 , -0.2219082 , -0.01748538, -0.16044484,  0.3214744 ,
  -0.83133954, -0.20797892,  0.3086791 , -0.14802434, -0.06924684,
   0.20736599, -0.17025413,  0.4049661 ,  1.0886686 ,  0.19389918,
   0.54727596,  0.07846259,  0.38669416],
 [ 0.4243707 ,  0.70358384,  0.87995905, -0.32457817, -0.17028205,
   0.73216164, -0.40521365, -1.1873119 ,  0.584273  , -0.28159347,
   0.1226372 ,  0.29258484,  0.5793659 ,  0.57803935, -0.42118943,
   0.71645766, -0.3007511 , -1.2396969 , -0.6514517 , -0.06111381,
   0.88936055, -0.23078781,  0.518572  ,  0.21968496,  0.09820081,
   0.15850006, -0.472025  , -0.6009863 ,  0.33374414,  0.15774623,
   0.8274679 , -0.13930778,  0.04727286, -0.69028616,  0.48044443,
   0.3019004 ,  0.683025  ,  0.60528004,  0.6526188 ,  0.24354951,
   0.61296886,  1.266724  , -0.5637129 , -0.59440476,  0.2545061 ,
   1.1640543 , -0.5999235 , -0.04969801,  0.9580109 ,  0.61255705,
   0.9711428 , -0.89707893,  0.90930694,  1.2373968 , -0.15618332,
   0.67174286, -0.15177667, -0.3092388 , -0.11216772,  0.592175  ,
   0.00880459, -0.6108904 ,  0.23315203, -0.27497178,  0.48016286,
   0.73507965, -1.1809851 , -0.01436179, -0.14401889,  1.4239649 ,
  -0.21038277,  0.5447901 , -0.49828747,  0.49739784, -0.44492492,
   0.10955872, -0.16583085,  0.4545244 ,  0.77934164, -0.39587384,
  -0.16787341, -0.07450593,  0.11866458, -1.3403229 , -0.31037158,
  -0.21654007, -0.1277908 , -0.40183222,  0.3225856 , -0.0166645 ,
   0.5592836 , -0.6034545 , -0.16921116, -0.23840185, -0.5105341 ,
  -0.17610826,  0.26107687,  0.15981609,  0.288804  , -0.68488127,
   0.56922364,  0.06358856,  0.23407912,  0.22164832,  0.26225778,
  -0.6004542 , -0.10767239,  0.17530294,  0.00840971,  0.9589865 ,
  -0.71988165,  0.30383936,  0.52000713,  0.703072  , -0.37982956,
   0.69573665, -0.67332506,  0.13582702, -0.71623915,  0.7348792 ,
   0.05565453,  0.5320093 ,  0.37161872,  0.16346183,  0.7876799 ,
  -0.09497461, -0.6277722 ,  0.26613432],
 [-0.42554593,  0.33952042, -2.8364284 , -0.12922598, -0.9627851 ,
   0.2476649 ,  0.2326513 ,  0.0307911 , -3.3490593 , -0.13796374,
  -0.88562435,  0.30138278, -1.0697805 , -0.01505816, -1.0960443 ,
  -0.16499889,  0.13924037,  0.76572245, -0.39874068, -0.33959016,
  -2.6929026 , -1.3810226 , -0.36048096,  0.79865575, -0.11998051,
  -0.6464973 , -0.44047642,  0.04241445,  0.55489194, -0.65125847,
   0.97671366, -0.09576889,  1.1391764 ,  0.46735924,  1.110558  ,
  -0.88679105, -1.084268  ,  0.10886118,  0.23713292,  0.17701976,
  -0.1037162 ,  0.2785784 , -0.582096  , -0.12520754, -0.01861885,
   0.5132457 , -0.30626428, -0.43260384,  0.33054668, -1.4668361 ,
  -0.5680929 , -1.0373156 ,  1.3605578 ,  0.09545739,  1.1648382 ,
  -0.696704  ,  0.9607185 ,  0.29708946, -0.4943789 ,  0.9089371 ,
   0.5911999 ,  0.15902145, -0.13798544, -0.40785986, -0.20382522,
  -0.01886088, -0.21933536, -1.064958  ,  0.32550213,  0.05747411,
  -0.4023376 ,  0.7304133 ,  0.36639154,  0.23591118,  0.74496657,
  -2.153128  , -0.2989812 , -0.09876931, -2.017321  ,  0.20580067,
   1.5462029 ,  0.41074806, -1.0936482 ,  1.3924588 , -0.3220284 ,
   0.40550894, -1.102894  ,  0.6053596 , -1.3263569 ,  0.09930128,
  -0.29144394,  0.16381332,  0.36686975, -0.07690979, -0.89002144,
  -0.31571892, -0.19883753, -0.4483025 , -0.00025431, -0.05362856,
  -0.4355498 ,  0.09365781, -0.5701674 , -0.27257282, -1.5013943 ,
   0.8347148 ,  0.44553328, -0.25866812,  0.2056855 ,  0.31665626,
  -0.30324134, -0.2782593 ,  0.00356988,  0.95260394,  0.52414733,
  -0.06673591, -1.3478858 , -0.29587892,  0.8518083 ,  0.45457324,
  -1.4277911 , -1.1078192 , -1.2848337 , -1.2412875 , -0.33483484,
   0.2710598 ,  0.12753886,  0.03358247],
 [-0.26182944, -0.56987554,  0.1070191 ,  0.28897646,  0.05307092,
   0.25065157, -1.028569  ,  0.9969577 , -0.95180494, -0.65226233,
  -0.8420993 , -0.3540013 ,  0.01222059, -3.1790004 ,  0.5673899 ,
   0.4359464 , -0.5562808 , -0.43030936, -0.54019034,  0.02471444,
  -0.65447813, -0.35531092, -0.86896956,  0.24566756,  0.0464    ,
  -0.13856713, -0.43039715,  0.5391508 , -1.058533  , -0.48296827,
  -0.24127315,  0.2414844 , -0.87549514, -0.37506112, -0.26292503,
  -0.22198789, -0.43816057, -1.0794433 , -0.04850724,  0.6509274 ,
  -0.15476452,  1.2676638 , -1.0737941 , -0.65026253, -0.24291408,
  -0.41131532, -0.76194274,  0.0144752 , -1.4648921 , -0.54686934,
  -0.3409184 , -0.24778572, -0.83479846, -0.7404292 , -0.22278856,
  -0.46415576, -0.28952554, -0.9033507 , -0.4256139 , -0.07126097,
   0.6436154 , -0.28737438,  1.1762037 ,  0.24239688, -0.23598623,
  -1.1666112 , -0.6287082 , -0.5505488 , -0.24324493, -1.6414347 ,
   0.2111222 , -0.5771411 ,  0.16360842,  0.81238973, -0.7509878 ,
   0.13398774,  0.3939438 ,  0.18303682, -0.10049456,  0.7999328 ,
  -0.4279338 , -0.09951717, -0.3098224 , -0.25945762, -0.8257028 ,
  -0.70737636,  0.22042377,  0.96451855, -0.40902323,  0.02356136,
  -0.6341552 , -1.334508  ,  1.1253548 ,  0.14796343, -0.15773995,
  -0.15755746, -0.72497666,  0.2574237 ,  0.1841144 , -1.5146918 ,
  -1.0767391 , -0.36736006, -0.21119116, -0.43864033, -0.08687472,
  -0.00634168,  0.37013388, -0.6407779 ,  0.7792033 , -0.673459  ,
  -0.6264293 , -0.6828631 ,  0.71307415, -0.92781264,  0.45081958,
   0.4719214 , -0.10414232, -0.3364249 , -0.3589497 , -0.10991468,
  -0.3420767 , -0.2194248 , -0.04363852,  0.36065862, -0.5769016 ,
  -0.6403067 , -0.30749813, -0.09767435],
 [ 0.2281254 ,  0.49837762,  0.02405543, -0.04777826,  0.15633477,
  -0.03330226, -0.0184116 ,  0.6266169 , -1.0642407 , -1.3825041 ,
   0.5408213 ,  0.25929758, -0.54245627,  1.5281249 ,  0.2802121 ,
  -0.4355494 ,  0.8815649 ,  0.1365161 ,  0.86360985,  0.28344256,
  -0.21051033,  0.62928283,  0.7158593 ,  0.11036502, -0.16421553,
  -0.09958895,  0.01792884,  0.03241548,  0.18684806, -0.18532895,
   0.04763773, -0.3385962 , -0.08598544,  0.11076275,  0.48503932,
   0.6036273 ,  0.0441293 , -0.40006492, -0.35586306, -0.2487772 ,
   0.64921486,  1.647266  , -0.08598398,  0.09128629,  0.0071964 ,
   0.34130743,  1.4298286 ,  0.01197519,  1.0821822 ,  0.18193473,
   0.48018378,  0.947624  , -0.07401666, -0.06294575,  0.21352397,
  -0.2173645 ,  0.612159  ,  0.15139456, -0.45202565, -0.02509472,
  -0.9093624 , -0.20098227,  0.01257585,  0.08938726,  0.17700793,
   0.08046568,  0.02148291, -1.0644331 , -0.27970865,  0.90732837,
   0.3405743 , -0.12837863,  0.13714336,  0.04966161,  0.22654122,
   0.29023346,  0.38160545, -0.07535055, -0.28469416,  0.25763297,
   0.33858487, -0.6736618 ,  0.3014429 ,  0.1081296 ,  0.16920267,
   0.18262796, -0.04351959,  0.36031237, -0.30066115,  0.2062363 ,
  -0.16668724,  0.00527792, -0.17172204,  0.715112  , -0.00151061,
  -0.15777303,  0.2763702 ,  0.0940284 , -0.37804842,  0.130525  ,
  -0.07023967, -0.4966452 ,  0.2475342 ,  0.71955013,  0.40908995,
  -0.04192203,  0.36283958,  0.15816332, -0.12166318, -0.13447623,
  -0.02376946, -0.61885595,  0.20043951,  0.31349185,  1.2012732 ,
  -2.2329156 ,  0.43576375,  0.03813776, -0.52734935,  0.29532054,
  -0.5921028 , -0.04335603,  0.6690831 , -0.20898081, -0.05792767,
  -0.17672686, -0.07783482, -0.9047187 ],
 [ 1.0079441 , -0.3007644 , -0.5119829 ,  0.12595108, -0.31467313,
  -0.36325568,  0.08838218,  0.3917154 , -0.40242597,  1.4864217 ,
  -0.52975404, -0.20591938, -0.80096763, -0.195982  , -0.43727455,
   0.38351673,  0.42996317,  1.4854926 ,  1.0692209 , -0.52899885,
   0.07527329,  0.3192622 ,  0.7239285 , -0.3928375 , -0.66458595,
  -0.2351554 ,  0.20000105, -2.379002  , -0.28199127,  0.70526963,
  -1.0665109 ,  0.23860015,  0.10590346,  0.8118007 , -0.17936425,
   0.46675342,  0.5302353 , -0.18532054, -1.5353453 ,  0.00177114,
   0.7600308 , -1.8970851 , -0.27043262, -0.35035986, -1.2465701 ,
  -0.32150635,  0.66317517, -0.6220499 ,  0.7393726 , -0.8956273 ,
   0.17992564, -1.174462  , -0.36860728, -0.7275635 , -0.12171371,
  -0.4041765 ,  0.4005164 , -0.30396944, -0.17898674, -0.4186257 ,
   0.2658004 ,  0.483509  , -1.1117976 , -0.9097555 , -0.6294352 ,
  -0.10179388, -0.28863195, -0.06651645, -0.5134005 , -1.0543877 ,
  -0.26728797, -0.5620524 ,  0.9585244 , -0.5503237 ,  0.6488535 ,
  -0.67892754, -1.0381122 ,  0.1016578 , -0.33202696, -0.02666536,
   0.19444676, -0.28439313, -0.39056623, -0.44150206,  0.33561128,
   0.2853311 , -0.02766107,  0.8103702 , -1.1001358 , -0.17787454,
   1.0183159 , -0.8209838 , -0.6915453 ,  0.7300173 , -0.24469867,
  -0.6542844 ,  0.9153989 , -0.19886875, -1.287242  , -0.94966483,
   0.5985458 , -0.82220936,  0.19951566, -0.32677564, -0.7571939 ,
   0.26419732, -0.28400385, -0.23350336, -0.48863196, -0.54699725,
   0.08418896,  0.47358373, -1.1041055 ,  0.20303126, -0.96226245,
  -2.3737392 , -0.20488586,  0.23168164,  0.1768088 , -0.62012213,
  -0.4246946 , -0.19121051, -0.32679412, -1.4445263 ,  0.25120533,
  -0.12498637, -0.4594002 ,  0.3447093 ],
 [ 0.40689433, -0.29608908, -0.7711608 ,  0.14457993,  1.139883  ,
   0.73884153,  0.3977125 , -0.8473783 , -0.82810104, -0.61979187,
   1.7887208 , -0.17510876, -0.59902483,  0.06895974, -0.6776043 ,
   0.1629183 ,  0.034061  ,  0.40814176,  0.34684053, -0.69106305,
   0.1651582 ,  0.04457106, -0.01444595, -0.3025596 ,  0.53791094,
  -0.09720679, -0.63195425,  0.22503136,  0.8283681 , -0.82382745,
  -0.72473216, -0.01953638,  0.16591465,  0.5632169 , -0.13831754,
  -0.9606426 ,  0.00088927, -0.00143012,  0.02879345,  0.09540477,
   0.17624962, -0.8467629 ,  0.6203823 , -0.24154483, -0.82859665,
  -0.49253586, -0.5080483 , -0.10321829,  1.0110897 , -0.10628704,
   0.1063806 ,  0.26749766, -0.42487884,  0.42638844,  0.03137881,
   0.17420642,  0.31880307, -1.0261858 ,  0.11544682, -0.06551214,
  -0.3090466 , -0.52856624, -0.41384217, -0.08420267, -0.39946845,
  -0.07988162, -1.2939167 , -0.983999  ,  0.7194595 , -0.4012517 ,
   0.91801304, -0.65337867, -0.21548654,  0.09225316, -0.2295429 ,
  -0.4238788 ,  0.42948902,  0.42094007, -0.53239775,  0.12096463,
   0.34780824,  0.21860574,  0.39254496, -0.06561763,  0.7266663 ,
  -1.7174251 , -0.14347778,  0.69149715, -0.9935939 , -0.30030584,
  -0.5872123 ,  0.32505137, -0.5725105 , -0.318901  , -0.53503686,
  -0.18724342,  0.92091906, -0.22849613, -0.26889223,  0.79265577,
   1.9476362 , -0.21374851,  0.06702062,  0.3353544 ,  0.2283393 ,
  -0.30830884, -0.28181994,  0.48794773, -2.4535656 ,  0.33976236,
   0.3615404 ,  0.16017112, -0.13040096, -0.3195407 ,  0.1419223 ,
  -1.6398438 ,  0.2256417 ,  0.09539137,  0.33102986,  0.80892575,
   0.24931552, -0.29053298, -0.5852    ,  0.5820631 , -0.15420152,
  -0.02989451,  0.02818563,  0.00567879],
 [ 0.02082451, -0.48574927, -0.32107428, -0.1723826 , -0.00487275,
  -0.42323625,  0.04719713, -0.9046817 , -0.37765625,  0.23926294,
  -0.6299107 ,  0.09592314,  0.43425164,  0.14063537,  0.37652418,
  -0.64307904,  2.2123675 ,  1.2475463 , -0.25844657,  0.5897707 ,
   1.3949018 ,  1.0572155 ,  0.74642545, -0.30858615,  0.09971757,
   0.13304068,  0.21525003,  1.1119398 ,  0.5540825 , -0.2624892 ,
  -1.0297662 ,  0.00409361, -0.46078765,  0.50983703,  0.11305007,
   0.70007   ,  0.13388519, -0.6317134 ,  0.8711405 ,  0.682818  ,
   1.1210394 ,  3.1934967 ,  1.2942424 , -0.40347335, -0.5956075 ,
  -0.47250316,  1.0073547 , -0.06953412,  0.85287213,  0.525791  ,
   0.23077185, -0.13208313,  0.48359543, -0.33686456,  0.5181624 ,
   1.0892029 ,  0.4567892 ,  0.30902562, -0.02571575,  0.5445553 ,
   0.24321951,  0.82179266, -0.3524528 , -0.3268882 , -0.09511752,
   0.32066277, -1.6834996 , -0.22284031,  2.4933093 ,  0.00359721,
   0.59500396, -0.17260124,  0.1274789 , -0.38277575,  0.31109062,
   0.4192975 ,  1.9157343 , -2.2430174 , -0.57481456, -0.21246056,
   0.8322821 ,  1.0442598 , -0.1769066 ,  0.9984735 ,  0.55339146,
  -1.1070992 ,  1.0220605 , -0.00841539, -0.634666  , -0.1729716 ,
   0.7174306 , -1.1647929 ,  0.0409939 ,  0.19605348,  0.05747535,
   0.41082415,  0.8869007 , -0.06900178, -0.7594241 ,  0.6551237 ,
  -0.38757414, -0.53995705,  0.43861893,  0.66096777,  0.15951838,
   0.58041483, -0.23343949, -0.08743889,  0.5876847 ,  0.07438742,
   0.1997739 ,  0.5388069 , -0.41058263,  0.40903535,  0.11481711,
   0.99247205,  1.5472684 ,  0.18069229,  0.762962  , -0.51802725,
  -0.794107  ,  0.5988579 , -0.0739238 ,  0.11533092,  0.28823638,
   0.21992904, -0.35756916,  0.8035672 ],
 [-0.28059167, -0.03415943, -0.08407569, -0.0965708 , -1.3189265 ,
  -0.6618686 , -1.0912983 ,  1.1680803 ,  0.75212884, -0.49078   ,
  -0.17536683,  0.09413406,  0.72675097, -1.8372189 , -0.5860308 ,
   0.0452712 ,  0.03589997, -1.1963336 , -0.2892112 ,  0.06401203,
  -1.3000251 ,  0.51871   ,  0.19171911,  0.04109671, -0.43865275,
   0.9640231 , -0.46261868, -1.8665249 ,  0.5026862 , -0.13064216,
   1.0506716 ,  0.09075001,  0.81975925,  0.28360754, -2.5283587 ,
  -0.7836764 , -0.61868757, -2.4493802 , -1.1165748 ,  0.5350837 ,
  -1.0771921 , -0.99167293, -0.2694559 , -1.037486  , -1.2614458 ,
   0.3508404 ,  0.48208356,  0.14662938,  0.09653839, -1.454466  ,
  -0.0895981 , -1.224159  ,  0.50025386,  0.9623842 ,  0.23461536,
  -0.3304498 ,  0.2613692 , -0.59750587, -0.57703507,  1.4791808 ,
   0.895963  ,  0.02800765,  0.07186759, -0.46462774, -0.73582244,
   0.4805957 ,  0.19294214,  0.06210326,  0.39401194,  0.44010064,
  -0.12550233,  0.32986882, -0.00835409,  0.33762068, -0.16609691,
  -0.47448182, -1.1021674 ,  0.44092324,  0.01602287,  0.40266907,
   0.46183679,  0.13756919, -0.29354253,  0.72995585,  0.73287314,
   1.2456836 , -0.89641815,  0.47199088, -0.06689402, -0.46798643,
  -0.8417033 , -0.59243995,  0.5988043 ,  0.19830766,  0.4025077 ,
  -0.6201847 , -1.7089049 ,  0.0452164 ,  0.10054532, -0.34837484,
  -0.9907983 , -0.38504687, -0.15021558,  0.4597585 , -0.35702345,
  -0.12146071, -0.12460636, -0.41174942,  0.40091607,  0.01493693,
  -1.8591422 , -1.1588931 , -0.40186432,  0.38828552,  0.46036744,
  -1.834465  , -0.55589014, -0.52066827,  0.19521876,  0.05751189,
   0.4743773 , -0.8245119 ,  0.16300848, -0.5301181 , -0.3941448 ,
  -0.68339413, -0.41844222, -0.50566626],
 [ 0.35874957,  0.9315688 ,  0.41711417, -1.6466731 , -0.32667932,
   0.12931415, -0.909208  ,  1.0796009 , -0.10912103,  0.4553069 ,
   0.04772192, -0.4328111 ,  0.04021696,  0.01225706, -0.89147437,
  -0.41880405,  0.24954796,  0.4195664 , -0.17179242, -0.9902815 ,
  -0.48808762, -1.2697388 , -0.04829048, -1.2895916 , -0.5681641 ,
   0.27733257, -0.05105127,  0.1523421 , -1.998126  ,  0.28775617,
   0.96583027,  1.1926886 ,  0.35060644, -1.4233948 ,  0.31179196,
   0.4076964 ,  0.44947943,  0.870771  , -0.75734687, -0.3419939 ,
  -0.02716494,  0.71303076,  0.11531566,  0.19645047,  0.4326922 ,
  -0.30917424, -1.4903811 , -0.04498778,  0.5458137 ,  1.0351417 ,
   0.39871576,  0.19926962,  1.0699087 ,  1.0855192 ,  0.78934985,
   0.4949125 ,  0.85849655,  1.3051022 , -0.6033395 , -1.2550145 ,
   0.4323953 ,  0.56768954, -0.86285394, -0.41518265, -0.25239006,
   0.75116   ,  0.17792612,  0.09973797,  0.36184156,  0.9941497 ,
  -0.49146947,  0.29150134, -1.1736264 , -0.593319  ,  1.1040834 ,
   0.3376417 , -0.7542904 ,  0.2379776 , -0.26877066, -0.5307952 ,
   0.33196974,  0.32657602,  1.5129133 ,  0.01518631,  0.44347554,
  -0.5618676 ,  0.14296238, -0.06411939,  0.08707333, -0.30236214,
   0.29986605, -1.8768239 , -0.844276  ,  1.5453515 ,  0.6140125 ,
   0.25170645, -0.23801309,  0.16813777, -0.5629039 , -1.2441002 ,
   0.26174837,  1.2019781 , -0.49949637,  0.18880157,  1.1343062 ,
   0.78252715, -1.3390931 , -0.15965532, -0.6569255 , -1.7094221 ,
  -0.04416306, -0.6606064 ,  0.38106588, -0.37191245,  0.07653418,
  -0.2960943 , -1.0422462 , -0.09375221, -0.05787025,  1.5301214 ,
  -0.33444113, -0.21885014, -0.94511634,  0.49687183,  0.02736855,
  -0.04460036, -0.16345465, -0.5987275 ],
 [ 0.551586  , -0.5976558 , -1.3357728 ,  0.42184588, -1.302516  ,
  -0.06865177, -0.1662348 ,  0.02418195, -0.10154247, -1.7146598 ,
  -1.158067  , -0.33140123, -0.7539944 , -2.542883  ,  0.21381308,
   0.57755226, -1.5560974 , -1.0141443 , -0.04188733,  0.07650267,
  -0.975785  , -2.035205  , -1.2549452 ,  0.22165465, -0.25497842,
  -0.54551184, -0.41180685, -0.47003356, -0.45351666, -0.24693975,
  -1.9012349 , -0.11744007, -0.20948459, -0.53781766, -0.42099994,
  -0.31011817, -0.67202425, -0.56594735, -0.44908348,  0.31520584,
  -0.57212347,  0.34638435, -0.02203411, -1.4333835 , -0.16539592,
  -0.29390752,  0.3600309 , -0.7862451 , -0.8657852 , -1.0282379 ,
  -0.4365551 , -1.0161543 , -1.3326317 , -0.47745764, -0.39721704,
  -0.45227233, -0.3147554 , -0.6971943 , -0.17913456, -0.33126748,
  -0.94376683, -0.3820985 ,  0.6013002 , -0.30379316,  0.11203494,
  -0.23624368, -1.2721766 , -1.109198  , -0.37714443, -0.2649625 ,
   0.31530637, -0.35177508,  0.41913053,  0.8829531 , -0.23897924,
  -0.11150957, -0.5032859 , -0.7014139 , -0.8249534 ,  0.7302965 ,
  -0.31836212, -0.1499144 , -0.80703115, -1.0245241 , -0.5668802 ,
  -0.26578298, -0.08253884,  0.67393094, -0.03695693,  0.0205628 ,
  -0.2615818 , -0.15233344,  1.3092265 ,  0.13019074, -0.4584007 ,
  -0.66211617, -0.77700275, -0.68871784,  0.18357703, -0.563588  ,
  -0.9834263 , -0.75825846, -0.42680776, -0.7040178 , -0.5273863 ,
  -0.3445902 ,  0.09533357, -0.89045817, -0.16751325,  0.15843175,
  -0.67732775, -0.21233363,  0.7384014 , -0.01034139, -0.02205832,
  -1.5536704 , -0.29531613, -1.0345134 , -0.1252155 , -0.16220668,
  -0.6963616 ,  0.2536991 , -0.2322529 , -0.48469633, -1.2360065 ,
  -1.0789684 , -0.44395262, -0.28955904],
 [-1.0595282 , -0.52314997,  1.2791685 , -0.63141066, -0.00081575,
  -0.10441513,  0.23163746,  0.7312363 ,  0.08846097, -0.01017696,
   0.5011397 ,  0.19656129, -0.7688944 , -0.3765146 ,  0.05283929,
   0.5002356 , -1.8302897 ,  1.34935   , -0.01637908,  0.56371325,
   2.108846  , -0.14378059,  0.784817  , -0.34305173,  0.25022164,
   0.755502  , -0.00634003,  1.0570118 ,  0.5574045 , -0.45561093,
   0.87161696,  0.4133538 , -0.40339708, -0.31530255,  0.77170014,
  -0.04807128,  0.9222307 , -0.13644867, -0.00454549,  0.59245723,
  -2.1483583 , -0.9677959 , -1.1740173 ,  0.1881474 ,  0.00198153,
  -0.4556838 , -1.3235039 ,  1.1233659 , -1.3930238 , -0.22580494,
  -1.450736  , -0.85637295,  0.5932148 , -0.5369386 , -1.385789  ,
   0.45456687, -0.2936015 ,  0.32321644, -0.09112402, -0.15525152,
   0.7791017 , -0.5074731 ,  0.71466446,  0.08997434,  0.6055619 ,
   0.43890497, -0.1739513 , -0.01451394, -0.12090254, -1.1122947 ,
  -0.71538854, -0.43306133,  0.60081214,  0.6114303 , -0.2774444 ,
  -3.1780057 ,  0.79052025,  0.02463375,  0.81655604,  0.24302885,
   0.06852838,  0.34635043,  1.2073063 ,  0.6359828 ,  0.57648736,
   0.40675277, -0.44228855,  0.92344546,  0.9775896 ,  0.46715945,
  -1.9527253 , -0.14439413,  0.83795136,  0.07528583,  0.8703872 ,
   0.9617988 ,  0.3682897 ,  1.2651491 ,  0.7509758 ,  0.16918248,
   0.7580794 ,  0.88904643,  0.1276885 , -0.60528547,  0.72989595,
   0.5295498 ,  0.3779517 , -0.6090888 ,  0.8420096 , -0.43746504,
   0.33635113,  0.3630793 ,  0.29485   , -1.2621248 ,  0.49893036,
   0.9337014 ,  0.2617992 , -0.38803762, -0.4836808 ,  1.200293  ,
   0.07641614, -0.93558383,  0.4714567 ,  0.6782143 , -0.43613565,
   0.36937344,  0.34569654,  0.5868401 ],
 [-0.03926441, -0.18947114,  0.37262607,  0.48460042, -0.06662215,
   0.03142857, -0.9844533 , -0.38215593,  0.5315867 , -0.7661964 ,
  -0.07695638,  0.25327355,  0.1821419 , -0.04129764,  0.08564781,
  -0.4249633 ,  0.6687701 , -0.09302969,  0.37713376,  0.5643818 ,
   0.1463538 ,  0.41967964,  0.05979311, -0.2337879 , -0.24982898,
  -0.02378927,  0.26217192,  2.223526  ,  0.13326743, -0.6838709 ,
   0.05746098,  0.02500405,  0.47422826,  0.99928075, -0.25627187,
  -0.36865586, -0.03622333,  0.16448887, -0.18955961, -0.05169562,
  -0.11470281, -1.0811651 ,  0.10386349,  0.34381598,  0.0909754 ,
  -1.4553992 , -1.1917406 , -0.5322907 ,  1.031987  ,  0.56082916,
   0.09747458, -0.83208436,  0.7823865 ,  0.5119914 , -0.16299719,
  -0.04188035,  0.63884896,  0.39489552, -0.98511726,  1.5204902 ,
   0.12677851,  0.24631628, -0.46204397,  0.49068165, -0.5874911 ,
  -0.9758251 ,  0.19394162, -0.19937213, -1.2228297 ,  0.26279625,
  -0.68848544,  0.48237354, -0.23880707, -0.01977417, -0.63272744,
  -0.00675891,  0.20487349,  0.9953069 ,  0.53743595,  0.06974997,
   0.06923018,  0.69570434,  0.3863889 ,  0.22133332,  0.45425123,
  -0.42102632, -0.04849256, -0.1590888 ,  0.01800413,  0.303512  ,
  -1.0831136 , -0.3820636 , -0.07807168, -0.2842625 ,  0.42110214,
   0.0698417 ,  0.4895909 , -0.22631821,  0.09341086, -0.04300866,
   0.6595514 , -0.509976  ,  0.17474978, -0.023995  ,  0.19879764,
   0.11449438, -1.1337013 , -0.07109948, -0.88384396,  0.14968607,
  -0.16706698,  0.8728224 ,  0.3741371 ,  0.10688952, -0.6128096 ,
  -0.24654905, -0.02258059, -0.0164238 ,  0.27617782, -0.83691573,
   0.38294548,  0.18956359, -0.18745695, -0.3668994 , -0.22446193,
  -0.07309065,  0.5944562 , -0.09480848],
 [ 0.71770895,  0.23650639, -0.24084145, -0.518805  ,  0.23745103,
  -0.7474197 ,  0.06729957,  0.44433105, -0.28541452, -0.18248329,
   0.66513854, -0.2929502 ,  0.01930809,  1.4797573 , -0.02769893,
   0.30698347,  0.84496653,  1.1672595 ,  0.6982052 ,  0.11817777,
  -0.40511152, -0.08789206,  0.3702723 ,  0.00072253,  0.0280775 ,
  -0.04506928, -0.48956332, -0.861004  ,  0.01511913, -0.17524634,
   0.8212845 , -0.69898254,  0.3078765 ,  0.45572132, -0.3382341 ,
   0.00277973, -0.02321035, -0.20348622,  0.25397667, -0.09670241,
   1.2979171 ,  1.8229986 , -0.10879853,  0.03430975,  0.22056906,
   0.43020594,  0.45931432, -0.17243496, -0.23931758, -0.31519866,
  -0.51165134,  0.04524563,  0.48063052,  0.37536043,  0.34110376,
   0.29959586, -0.05808284,  0.4728973 , -0.37306967,  0.15331832,
   1.3309269 ,  1.0821533 , -0.4673247 ,  0.61535037, -0.34377882,
   0.34913033,  0.28187686,  0.00781845,  0.76716864,  0.22221132,
  -0.02462229,  0.06684266, -0.00241884,  0.40840834,  0.90242726,
   0.2859678 ,  0.0057301 ,  0.05768115, -0.44749188,  0.10173295,
   0.6583739 ,  0.2197453 , -0.68090665,  0.19989915, -0.5823465 ,
  -0.12824571, -0.44363934, -0.5744541 , -0.70750153, -0.20360082,
   0.32860813, -1.4891897 , -0.02884022,  1.0615871 , -0.05807446,
  -0.21463323,  0.7119334 , -0.07148435, -0.04868217, -0.45133325,
   1.2376214 ,  0.12141767,  1.3015325 , -0.6624833 ,  0.26261094,
   0.17327558,  0.6041058 , -0.03574926, -0.06986926, -0.6948499 ,
  -0.42321253,  0.7678531 , -0.24476734,  0.03032752,  0.5171622 ,
  -0.4595535 , -0.5467536 ,  0.5991289 ,  0.5209622 , -0.41453502,
  -0.23201874,  0.02010567,  0.3148476 , -0.1922618 ,  0.39693594,
  -0.48872924,  0.15271229, -0.26899037],
 [ 0.78170305, -0.6247996 ,  0.9214761 , -0.16201288,  0.16637549,
  -0.68618137, -0.09870612,  0.3784967 ,  0.19767992, -0.21846268,
  -0.07528715, -0.6340938 ,  1.4731895 ,  0.19753954,  0.9591608 ,
   0.40874416,  1.6894512 , -1.3315024 ,  2.3756013 , -0.2848819 ,
   0.7425709 ,  0.20230757,  0.5234235 ,  0.02934118,  0.08780782,
   0.7335916 , -0.13582718, -1.4615453 ,  1.0067656 ,  0.2699377 ,
   0.3044467 , -0.83505857,  0.29639697,  0.22891556,  1.2793765 ,
  -0.48106223,  0.35336587,  0.09606593, -0.42678228, -0.7591244 ,
  -0.2663428 ,  0.3987164 ,  0.5672108 ,  0.79388726, -0.13707705,
   0.12595841,  0.23630485, -0.69969493, -0.39914903, -1.5435103 ,
  -0.19726428, -0.2399817 ,  0.22707394,  0.0640405 ,  0.01371153,
   0.46847108,  0.08275322, -0.2095419 , -0.6055201 , -0.49926713,
   0.9907376 ,  0.25947222,  0.05224151, -0.5970425 ,  0.48867813,
   0.24618891,  0.89223444,  0.00367543,  0.04781436, -0.56008947,
  -0.02027869,  0.8838041 , -0.9559026 , -0.01334672,  0.3097137 ,
   0.8350123 , -0.08163618, -1.4578909 ,  0.7542956 , -0.25010797,
  -0.19148134,  0.2706625 , -0.4914752 , -0.5069518 , -0.15380521,
   0.47847867,  0.8858982 , -1.0508789 ,  1.0576265 ,  0.59292704,
   0.1974128 ,  0.16187358, -0.74313813, -0.09271142,  0.0957288 ,
   0.5006511 ,  0.8894648 ,  0.27594778, -0.1678022 , -1.0744116 ,
   0.35585588,  0.22108844,  0.22205627, -0.10509349,  0.5458195 ,
  -0.30022037, -0.00344811, -0.39466918, -0.04996753, -0.04528954,
   0.24834938, -0.68029284,  0.02786926,  0.59377205,  0.59948796,
   1.5844997 ,  0.13865024, -0.3477534 , -0.9105048 , -0.45265597,
   0.6280746 , -0.16332975, -0.79636276,  0.34986064,  0.01444832,
  -0.2213214 ,  0.51018125,  0.3696844 ],
 [ 0.24458368, -0.12789346,  0.8551581 , -1.4381711 ,  0.58033603,
  -0.49076146, -0.07814354,  0.8316757 ,  0.15610896,  1.2731874 ,
   0.41103324,  0.5816334 ,  0.48103946, -0.03431487,  0.8880433 ,
  -0.06153294,  0.20351957,  0.7350108 ,  0.88165295,  0.55681807,
   0.51909494, -0.6301925 ,  0.22104244, -0.82344973,  0.4979359 ,
   0.60175705,  0.48100924,  1.5251974 , -0.2987252 ,  0.35117936,
   0.59716946, -0.88022333, -0.04601939, -0.69562733,  0.2964821 ,
  -0.6296752 ,  0.03950046, -0.19674772,  0.9647706 ,  0.09318344,
  -0.03203051, -0.10083841,  0.5341276 ,  0.4333033 , -0.20587721,
  -1.2364405 , -1.3939884 ,  0.51114905, -0.11757616,  0.49129528,
  -0.00584434, -0.17357706,  1.0995446 ,  0.8297785 , -0.60467356,
  -0.64976865,  0.28430226,  0.79433835, -0.17088279, -1.1830685 ,
   1.7454668 ,  0.6519708 , -0.16234723,  0.39811647,  0.23680909,
   0.18380183,  0.9206759 , -0.95444256,  0.14526734,  0.55067575,
  -0.9812811 ,  1.4744191 , -0.12529482,  0.04954338, -0.09849931,
   0.4827207 , -2.0719903 , -0.03275094, -1.1556458 ,  0.596208  ,
   0.69260174,  1.0157518 ,  0.12597136, -0.5772049 ,  0.43185097,
  -0.40928283,  0.2310841 , -0.29114908,  0.14625372,  0.22869287,
   1.4456528 ,  0.02084413, -0.610076  , -1.3552258 ,  0.17125495,
   0.9027927 , -1.1670536 ,  0.5115473 ,  0.0895374 ,  2.8482373 ,
   0.6531181 ,  0.11394445, -0.7990356 , -0.15468176,  0.6900405 ,
  -0.5484656 , -1.6072721 ,  0.40505192,  0.05261251, -0.04750292,
   0.05523564, -1.4966117 ,  0.2591247 ,  0.16323496, -0.8374939 ,
   1.2054454 ,  1.11774   , -0.00135922,  0.430273  , -0.30620947,
  -1.8696414 ,  0.42289513,  0.8197236 , -0.26432073,  0.05495254,
   0.35973895, -0.6321879 ,  0.39258683],
 [-0.00715979, -0.50744873, -0.57254696, -0.872451  , -0.35222057,
  -0.6685855 , -0.3802415 , -1.0224918 , -0.4915173 ,  0.36646807,
  -0.3548374 , -0.4707293 , -0.2513447 , -0.35758218, -0.52735925,
  -0.32437494, -0.09616336, -0.05325096, -0.63752836, -0.58966726,
  -0.12455817, -0.4763624 , -0.24410203, -0.60178596, -0.5201706 ,
  -0.6267873 , -0.4903166 , -0.55140346, -0.50741124,  0.5675696 ,
  -0.48854813, -0.22757155, -0.83215773, -0.5970209 , -0.4665798 ,
   0.0932305 , -0.43934187, -0.59794086, -0.29002386, -0.7748217 ,
  -0.6354464 , -0.5738387 , -0.3242504 , -0.37596732, -0.3849668 ,
  -1.3788253 , -0.00438937, -0.48629737, -0.35672772, -0.40487912,
  -0.32674786, -0.46525788,  0.02939213, -0.65055865,  0.18793082,
  -0.41267028,  0.14002727, -0.32676032, -0.6767518 , -0.5032565 ,
  -0.51085365,  0.3925299 , -0.6811857 , -0.66408014, -0.5533428 ,
  -0.40161046, -0.19179808, -0.46554533, -0.4090678 , -0.6408684 ,
  -0.6537867 , -0.5747598 , -0.59924567, -1.2302994 , -0.5160534 ,
  -0.45170447, -0.3222778 , -0.55889034, -0.21399325, -0.4957502 ,
   0.19487956, -0.42522022, -0.6909029 , -0.5050677 , -0.51818126,
  -0.48797524, -0.4519115 , -0.6836893 , -0.6591049 , -0.48305777,
  -0.37024218, -0.5917029 , -0.81366265, -0.43487054, -0.6773415 ,
  -0.58803713, -0.32750583, -0.4720258 , -0.5588178 , -0.48754779,
  -0.18935727, -0.5909982 , -0.10931756, -0.44289488, -0.61625624,
  -0.62231594, -0.82466793,  0.64084566, -0.9086176 , -0.76644105,
  -0.25806358, -0.41737482, -0.7621801 , -0.6261957 , -0.25930488,
  -1.3465253 , -0.7114722 , -0.61291254, -0.5988295 , -0.18093942,
  -0.7654706 , -0.6585677 , -0.6139451 , -0.4607887 , -0.6798525 ,
  -0.3148061 , -0.52950877, -0.48669606],
 [ 0.67605215,  0.00557876, -1.0432781 , -0.78771347,  1.2980157 ,
   0.36828533, -1.1254199 ,  1.7109216 , -1.229311  , -1.1192323 ,
   0.50638765, -0.8673802 ,  0.6264283 ,  0.0744721 ,  0.73742944,
  -0.3872328 ,  0.70293826,  0.32580984,  0.4996951 ,  0.5969006 ,
   0.22666648,  0.20559415,  0.7077581 ,  0.02637503,  0.26705825,
  -1.015225  , -0.15661518, -0.13709262, -0.35297227, -0.52548194,
  -0.4774753 ,  0.08308648,  0.4116949 , -0.00762772, -0.7395602 ,
  -0.4035344 , -0.4574859 , -1.4927648 ,  0.0089633 ,  0.6654656 ,
   1.5049655 , -2.2691298 , -0.05687547,  0.04450046,  0.21515428,
  -0.07607094, -4.168864  , -0.33670083, -0.5219071 ,  0.00063612,
  -0.9026474 , -0.41580263,  0.5126836 ,  0.76474756,  0.1045891 ,
  -1.8790138 ,  0.438569  , -0.5651308 , -0.12877536,  0.73758274,
  -0.04348341,  0.12726034,  0.7111899 ,  0.19334608, -1.1588689 ,
   0.05471573, -0.3951367 ,  1.0705956 ,  0.97979015, -0.04117642,
  -0.22340538, -0.02355571,  0.34591666, -0.17660233, -0.1768021 ,
   0.8667362 ,  0.08381699,  0.48469168, -0.5504228 , -0.06709985,
   0.5462721 ,  0.2374967 , -0.19032334, -0.59505516, -0.19431682,
   0.49421266,  0.6265073 , -0.5002081 ,  0.1875404 ,  0.34622115,
  -0.9124595 ,  0.42405802,  0.7124246 , -0.2849994 , -0.7143894 ,
  -1.2617005 ,  0.30537772, -0.10093778,  0.85598934,  0.25721514,
   0.08969251, -0.9597236 , -0.4914155 ,  0.17512332, -0.8713736 ,
  -0.4167058 ,  0.61707824, -0.2563677 ,  0.9530912 , -0.13337351,
   0.13140777, -1.3381635 ,  0.34857142, -0.82287186, -0.06016373,
   1.0182118 ,  0.10147943,  0.4156507 ,  0.03237934,  0.2501896 ,
  -0.05356055, -0.25991198,  0.72358394, -0.5644251 ,  0.44395804,
  -0.07087874, -0.66412336, -0.480564  ],
 [-0.7466328 ,  0.44242898, -0.46962   , -0.93158567, -0.17950301,
   0.03290767,  0.13680698,  0.62209725, -0.36336413,  0.65363014,
   0.8189357 ,  0.11486653, -0.21492909,  1.5166903 ,  0.15579289,
   0.31393802,  1.6802307 , -0.22128683,  0.56434256,  0.9195349 ,
  -0.43859246,  0.7346266 ,  0.17565776, -0.53288287, -0.63566935,
  -0.25885057,  0.10509996, -0.07086429, -0.39917722,  0.25330782,
  -0.25236037,  0.440715  , -0.17566119,  0.3195931 ,  0.24633616,
   0.7044287 , -0.9507766 , -0.39372873,  0.69145393,  0.10266475,
   0.8270059 , -0.23928243, -0.17887627,  0.2771143 , -0.805735  ,
   0.5092046 , -0.8110699 , -0.54314715, -0.4832281 , -0.8321137 ,
   0.15627754, -0.14403813,  0.7996    ,  0.21350439, -0.07194021,
   0.6695763 ,  0.04891521, -0.02361483, -1.2334327 , -0.7566214 ,
  -0.2614664 ,  0.60112   , -0.20401426,  0.51734114, -0.05380768,
  -0.30729437,  1.2029455 , -0.4531711 ,  0.7786153 ,  0.58112633,
   0.40571493, -0.02496643,  0.2805437 ,  0.24546218,  0.08661155,
   0.33109936, -1.113246  ,  0.23649384,  0.69285405,  0.17615758,
   0.3614221 ,  0.07929311,  0.6487371 ,  0.1510945 , -0.14088397,
   0.2635781 ,  0.43798906,  0.3455009 , -0.9732832 , -0.23445937,
  -0.46496233, -0.47083646,  0.18285266, -0.36247125,  0.2900693 ,
   0.01201547,  0.3532178 , -0.162691  ,  0.02200974,  0.94437903,
   0.55810875, -0.38232803,  0.10153434,  0.4350273 , -0.12056802,
   1.4680555 , -0.96198076, -0.18241778, -0.03383863, -0.16761743,
  -0.67726356, -0.0627351 , -0.72547954,  0.4268965 , -0.01208237,
  -0.6518427 , -0.1827095 ,  0.23711403, -0.26211634,  0.42297414,
  -0.09377109,  0.03516467, -0.11559905, -0.79849124,  0.43770003,
  -0.42070982, -0.11130562, -0.41538566],
 [ 0.01539962, -0.3056195 , -0.07619033,  0.08119912,  1.8829491 ,
   0.29700428, -0.45712116, -0.2636323 , -0.64300656,  0.11065733,
  -0.65677035, -0.07982652,  0.76415914,  0.22712113,  0.127592  ,
   0.10649365, -0.33626434,  0.89533395, -0.8363046 , -0.28555635,
   0.83739793,  0.2127584 ,  0.7866359 ,  0.17676412, -0.31499973,
   0.13558984, -0.27331638, -0.5815172 ,  0.08885099, -0.14770521,
  -0.231008  , -0.5540574 , -0.11699183, -0.37154508,  1.3094057 ,
  -0.7774151 ,  0.1762785 , -0.24419205, -0.24342512,  0.07904622,
  -0.3888609 ,  2.457027  ,  0.32052243,  0.04568155,  0.01172389,
   0.04793981, -1.4967519 , -1.0573802 , -1.9402344 , -1.9584435 ,
  -0.6428914 , -0.65408576, -1.185531  , -1.002412  ,  0.01005849,
   0.11677331, -0.44947687, -0.26778778, -0.8561402 , -0.394871  ,
   1.2497965 , -0.33024958,  0.4121062 , -0.3879006 ,  0.5297609 ,
  -0.94595736, -0.06353158,  0.3646334 ,  0.52871305, -0.12834543,
   1.0158451 , -0.21499087, -0.25010955,  0.06793784, -0.08299688,
  -0.19449842,  0.64249456, -0.15695354, -1.4237777 , -0.10184666,
  -0.36075428,  0.04448272, -0.09713084, -0.446622  , -0.39824167,
  -1.0771197 , -0.17120004,  0.03667027, -2.2723894 , -0.46213844,
   0.26089036, -1.0581225 , -0.49265683,  1.3172317 , -0.22120923,
   0.1019289 , -0.7308462 ,  0.2222246 ,  0.02677858, -0.39705056,
   0.55159533, -0.8901257 ,  0.21848391,  0.79483545, -0.37341642,
   0.34937695,  0.19048946, -0.18348993, -0.21955582, -0.15466785,
   1.1466658 , -0.9836947 ,  0.03617704, -0.47263667, -0.3332456 ,
   0.23874554, -0.05813483, -0.0052285 ,  0.35239872, -0.8182926 ,
  -0.34652713, -0.01156749, -1.2683305 ,  0.5547694 , -0.38801774,
  -0.16696708, -0.0769758 ,  0.14503802],
 [ 0.11282607,  0.30343258, -0.9652723 ,  0.30916435,  1.6614499 ,
   0.42789388,  0.34498268,  1.0341588 , -0.35098818,  1.8369023 ,
   0.41828638,  0.4031437 , -0.43772984, -0.53800255, -0.1764404 ,
   0.04683221, -0.4600231 , -0.45648843, -0.11495266, -0.43123537,
  -0.3363611 ,  0.72248477, -0.0550061 ,  0.28047007,  0.42910442,
   0.395104  ,  0.20650329,  0.74307805,  1.1888177 ,  0.06646344,
  -0.02300867,  0.1869721 , -0.35432768,  0.08509143,  0.58310765,
   0.19273663,  0.10914663, -0.22253409,  0.02578344,  0.01856966,
   0.7622283 ,  1.5704262 ,  0.719411  , -0.5886355 , -0.283217  ,
   0.5939428 , -0.05625567,  0.70184606,  0.12430937, -0.98263764,
  -1.1185206 , -0.10147229, -0.9279423 , -1.2206837 ,  0.5455964 ,
  -0.14147522,  0.21275555, -1.4137146 , -0.16713747, -0.563666  ,
  -1.114249  , -0.5625275 ,  0.03723065,  0.25490308,  0.35379106,
  -0.02574809, -0.2874596 , -0.45181823,  1.4803994 ,  0.09831192,
   0.4871084 , -0.45077813, -0.19834277,  0.11618915,  0.01429609,
   0.2627425 ,  0.6284764 , -0.36568505, -0.0364007 , -0.24958557,
   0.23014024,  0.76406205,  0.39574137, -0.2567636 , -0.53850883,
  -1.4606693 ,  0.26076254, -0.19458178, -0.36271593,  0.43436956,
   0.36759216, -0.32492825, -0.51549965, -0.6547211 ,  0.58254254,
  -0.33509117,  1.6574112 ,  0.43309715, -0.03209189, -0.14269702,
  -1.6517419 , -0.20755546, -0.14857367,  0.29393268,  0.37365726,
  -0.4891617 ,  0.07874338,  0.49903703, -0.8461119 ,  0.08372083,
  -0.21409093,  0.88206744,  0.09650531,  0.25074762, -0.34238935,
  -1.8704236 ,  1.1390059 , -0.18855444, -0.7354592 , -1.0054129 ,
  -0.8926662 , -0.13832842,  0.41733956,  0.8017793 , -0.25446254,
  -0.36194682, -0.17360446,  0.09934349],
 [-0.1349053 , -0.259853  ,  0.22163706, -0.29997817, -1.6630871 ,
  -0.57954305, -0.53569007, -0.6136529 , -0.45480448,  0.20857763,
   0.04824673,  0.1870553 , -0.3118427 , -0.9455172 , -0.3022461 ,
   0.47741026, -0.1628715 , -1.3733926 , -0.42653027, -2.7815158 ,
   0.38655   ,  0.6474554 , -0.1971605 ,  0.8205485 ,  0.9201424 ,
  -0.24223404, -0.07798324,  0.14140831,  0.5681573 ,  0.4674042 ,
  -0.39885765,  0.5328291 ,  0.2307063 ,  0.2671714 , -0.73799825,
   1.4079307 ,  0.5514753 ,  0.43734336,  0.34288377, -0.1852017 ,
   0.23824577, -1.4613222 , -0.782806  ,  0.4872011 ,  0.25743267,
   0.6449928 ,  1.5804285 ,  0.9732377 ,  0.94500256,  2.3579915 ,
   0.9862674 , -0.94727194,  0.90281165, -0.00945309,  0.17445825,
  -0.60849804, -0.9216549 ,  1.7060729 , -0.3743963 ,  0.7678911 ,
  -1.4515655 ,  0.09387016, -0.29696277,  0.50799495,  0.17504358,
  -1.4773697 ,  0.5423297 , -0.7068704 , -0.84668165,  0.10083344,
   1.0172281 ,  0.80101025, -0.00539734,  0.1877949 ,  0.06682235,
   0.7223529 , -1.1750329 ,  1.4519829 , -0.36766964,  0.15315397,
  -0.5786539 , -0.05896928, -0.09166183,  0.41306585, -0.8798658 ,
   0.03549435, -0.33168587, -0.27887827, -0.47359332,  0.16828893,
  -0.53980744, -0.12054304, -0.3016874 , -0.25153333, -0.16354007,
  -0.21651644,  0.28797996, -0.1198411 ,  0.957427  , -1.5297954 ,
   1.0099822 , -0.8352287 ,  0.029539  , -0.17578627, -0.5778762 ,
   0.32620195, -0.57531   ,  0.06025214, -0.25452423, -0.5596757 ,
  -0.5283976 , -0.49136958,  0.40808818, -0.49009582, -0.2889036 ,
  -0.53685534, -0.05083742, -0.2531995 ,  0.47969013, -0.17031226,
  -0.84174824, -0.55622894, -0.5561758 ,  0.13030028, -0.27802926,
  -0.04323121,  0.04501129,  0.8251093 ],
 [ 0.7496662 ,  0.08406688, -0.11400069,  0.9662644 ,  0.62501264,
  -0.05929399, -0.20980567,  0.629487  , -0.02607341, -0.8548968 ,
   1.6230015 , -0.13810448,  0.00634737, -0.21765463,  0.11548033,
  -0.2334447 ,  0.1533887 , -0.22772065,  0.6519188 ,  0.7035262 ,
   0.687577  ,  0.15402523,  0.40904653, -0.38897762, -0.0331945 ,
  -0.3520362 ,  0.01881642,  0.42367947,  0.19728012, -0.39589646,
   0.05026697,  0.00984857, -0.08337896,  1.2365396 , -0.272335  ,
  -0.30768147,  0.12789942,  0.05157669,  0.20105538, -0.01382479,
   0.79768085, -0.80464756, -0.32084408, -0.3293384 ,  0.24715878,
   0.3208387 , -0.14407645, -0.33558932,  0.5682216 , -0.24217345,
   0.18289831,  0.7529963 ,  0.59237736, -0.9770772 , -0.15681292,
  -0.31998166,  0.2851802 ,  0.42219546, -0.09694521,  0.9242798 ,
   0.82792956,  0.2472265 , -0.45379293, -0.48333305,  0.07019908,
  -0.5238849 , -0.3839996 ,  0.47367668,  1.4185836 , -0.03373544,
   0.08178346, -0.33875093,  0.33696815, -0.34829676, -0.45908687,
   0.81368065, -1.0673137 ,  0.1219103 , -0.09301962, -0.02415667,
  -1.1211774 , -0.37737265, -0.47419393,  0.1994589 , -0.62909263,
   0.47461805, -0.16749154,  0.1721639 , -0.5070025 , -0.1905135 ,
  -0.36474183, -0.12437314, -0.00000093, -1.2854371 , -0.18668906,
   0.14374064, -0.246052  , -0.15366131, -0.22172612,  0.09708247,
   0.04087719,  0.00317539,  0.4585402 ,  0.33384624, -0.30076802,
  -0.24413374,  0.5239836 , -0.19880942,  0.10396148, -0.54465383,
  -1.0796927 , -0.00402447, -0.1477116 ,  0.01108333,  0.30087438,
   0.31682712, -0.2986796 ,  0.4878482 ,  0.610163  ,  0.6151302 ,
  -0.27797475,  0.18808098,  0.00554763, -0.19325309,  0.07099332,
  -0.39420864, -0.25307482,  0.13367687],
 [-0.37274113, -0.89546496,  0.34602737,  0.73566556,  0.19945496,
  -1.7641755 , -0.7282137 , -2.0741363 , -0.34179965,  1.2854753 ,
  -0.16019449, -0.9920063 , -0.34012288, -0.09895051, -0.21120885,
  -0.51209724, -0.8432065 ,  0.3498289 , -2.1476498 , -0.6643708 ,
   0.5930593 ,  0.40079385,  0.73717356, -0.37428272, -0.41884506,
  -0.38742217,  0.03495779, -1.1459605 ,  1.3385447 ,  1.4361656 ,
   0.1819564 , -2.179312  , -0.01284336, -0.3138772 ,  0.01484035,
  -0.04665916,  0.32957965, -0.79028356, -0.33681425,  0.16527419,
   0.28129432, -1.4552141 , -0.70311683,  0.36157185, -0.34340507,
  -0.51611805, -0.7891443 , -0.24673411, -0.31716442,  0.11842377,
  -0.79499567, -0.5227969 ,  0.2893665 , -1.2020298 ,  0.03613206,
  -0.50457454, -0.83634067, -0.71208286, -1.6626792 ,  0.6432502 ,
   0.05731859,  0.15915553, -0.32119113, -0.9801218 , -0.49233872,
  -1.8044072 , -0.34776425,  0.08175226, -0.40547273, -0.63141936,
   0.08680037, -0.95409566,  0.00024536, -0.4098186 , -0.6195972 ,
  -0.43152058, -0.5564771 ,  1.1763183 , -0.16398452, -0.3674079 ,
   0.32914722,  0.20039342, -0.61713123,  1.1990649 , -0.04110509,
  -1.0653255 , -0.37343344, -1.2577403 , -0.92590785, -0.09247281,
   1.4860793 , -1.2288216 , -1.6012046 , -2.6271687 ,  0.48491856,
  -0.2464249 ,  0.31272382,  0.03458248, -1.0255454 , -1.0945776 ,
  -0.11117503, -1.4802605 ,  0.6282516 , -0.461427  , -0.65022856,
  -0.8825395 , -0.47251323,  0.18370485, -0.34770802, -0.44576895,
   0.11733554, -0.71615636, -0.67954177,  0.3209499 , -0.7945948 ,
   1.5726267 , -0.4186587 ,  0.25664774,  0.5030308 , -0.44276246,
  -0.51669496, -0.370991  , -0.32075724, -0.7720947 ,  0.02728463,
  -0.30466148, -0.6977501 , -0.66253406],
 [ 0.35168713, -0.02646646,  0.6922513 ,  0.2054229 , -0.78682894,
   0.6563163 , -0.42643866,  0.24818258, -1.0594038 , -0.64998484,
   0.5040388 ,  0.14989637, -0.10348172,  0.5438614 ,  0.5618482 ,
  -0.23056318,  0.13087767,  0.44194785, -0.16198814,  1.3731306 ,
  -0.00870352, -0.05357735,  0.28387812,  0.03494156,  0.1394091 ,
   0.50632536,  0.10618098, -0.07085636,  0.91046125,  0.52551335,
   1.9241439 ,  0.7884391 , -0.5017714 , -0.06213814,  0.3434426 ,
  -0.84991884,  0.0631225 , -1.2932132 , -0.95822024,  0.19552757,
  -0.03518874,  3.3129604 , -0.2932146 ,  0.40124333,  0.6956426 ,
  -0.2126602 , -0.40698045, -0.12064028, -0.6719239 ,  1.5833445 ,
  -0.7563262 , -0.16355549,  0.46221787,  0.22752155,  0.53569746,
   0.1643555 ,  0.31737325, -1.0639375 , -0.5406019 ,  0.20969726,
   0.26178724, -0.03189812,  0.18223031, -0.26406878, -0.2410431 ,
   0.15310614, -0.6683411 ,  0.99099207, -0.06704626, -0.02008975,
  -0.743406  , -0.21327488,  0.00138808, -0.25268108,  0.05045883,
   0.863996  , -0.57414544,  0.9981281 ,  0.8213817 , -0.12478665,
   0.47054222, -0.1863837 , -0.07314287, -0.72157925, -0.176199  ,
   0.50555825,  0.2975951 ,  0.02529984,  0.81222844,  1.0401138 ,
  -0.66733915,  0.03616107,  0.01639606,  1.2576126 , -0.01970808,
   0.43323734, -0.301602  ,  1.3224653 ,  0.0160906 ,  0.7810377 ,
  -0.16332725, -0.65724224,  0.50490206,  0.44812652,  0.3107562 ,
  -0.13497853, -0.44615126, -0.01426296, -0.12833408, -0.09964118,
  -0.28034598, -0.66260636,  0.2550367 , -0.5186273 , -0.40838948,
   0.35101345,  1.0132985 ,  0.03837571,  0.22564115, -0.23402748,
  -0.27717653, -0.49053818,  0.29566455,  0.42152062,  0.1622619 ,
   0.17852435, -0.30793032,  0.43008968],
 [-0.23650184, -1.1743426 ,  0.77751106,  1.4260598 ,  0.6661946 ,
   0.11588349,  0.31058097, -0.27722535,  1.8688575 ,  0.21282126,
   0.7157676 , -0.6138704 , -0.28922307,  0.5434257 ,  0.3397922 ,
  -0.38054982,  1.7439141 ,  0.24786323, -0.25325873,  0.6424488 ,
   0.46694186, -0.13924582,  0.54406345,  0.5880478 , -0.33564022,
   0.19193585,  0.5134265 ,  0.5396243 , -0.19751078,  0.8995827 ,
   0.7559494 ,  0.43151125,  0.46758986, -0.36299112, -0.12725218,
  -0.1219438 ,  0.5604302 , -0.12463187, -0.37878805,  0.30225205,
   0.4853261 , -0.66066784, -0.22131039,  0.7106334 ,  0.29363033,
   0.6003567 , -0.35789275,  0.30374315,  0.3425179 , -0.35142535,
   0.20222427,  0.2811985 ,  0.82150596,  0.28557166, -0.02370916,
  -0.20912924, -0.9109461 ,  1.8029248 ,  0.17057392,  0.5455012 ,
   0.17493907,  0.40690538,  0.27267691, -0.6310763 , -0.28195143,
  -0.03012821, -1.0203615 ,  1.6078993 , -0.02799097,  0.33766383,
   1.7377099 , -0.47029215,  0.56931823, -0.33068788, -0.34555808,
  -0.6205535 ,  0.53630877, -0.24943121,  1.2945681 , -0.13495068,
   0.8187605 , -0.13308698, -0.16594861,  1.6518204 ,  0.01875961,
   1.1479632 ,  0.1562133 , -0.79931575,  1.4018298 , -0.5215541 ,
  -0.85294026,  0.3088766 , -0.7103799 ,  1.0063717 ,  0.44947585,
   0.17022   ,  0.5881634 ,  0.06784588, -0.05624215, -0.14292051,
   0.43234792, -0.4863796 , -0.00600607, -0.24298143, -0.2451008 ,
  -0.02204243, -0.26063767,  0.2838877 , -0.61115146,  0.2021031 ,
   0.7942396 , -0.09010997,  0.22955826,  0.32545605, -0.30544978,
  -0.56387436,  0.46502674,  0.32653823,  0.31095093, -0.892566  ,
   1.2231245 ,  0.2296503 , -0.63299614,  0.31961337,  0.18157095,
   0.7141745 , -0.129198  ,  0.4254633 ],
 [-0.5525342 ,  0.48568556, -0.2922528 , -0.11429685, -1.0904845 ,
  -0.5966002 ,  0.24551958, -0.25673974,  0.20297834, -0.07724237,
  -0.32937202, -0.24462645, -0.05315445, -0.55801195, -0.40520313,
   0.03852472, -0.21873572,  0.20315607,  0.04641948,  0.67999274,
  -0.03759198,  0.6435754 , -0.8893104 , -0.11088944, -0.13639243,
  -0.544172  ,  0.09960619,  0.35294378,  0.73926294, -0.66106164,
  -1.1022229 , -0.09448248, -0.99025905, -1.327255  , -0.06655616,
  -0.74510235, -0.26648903, -0.53998256, -0.74899596,  0.2569056 ,
   0.36847895, -1.8601667 ,  0.1738411 , -0.43303907,  0.0134313 ,
  -1.1110501 , -0.80677265, -0.56566834,  0.41746482, -1.3031993 ,
  -0.66903305, -1.6682488 ,  0.35033372, -1.0228755 ,  0.21411832,
  -0.6404976 , -0.2627401 , -0.73518604, -0.12230345, -0.29241058,
  -0.48170674,  0.18524407,  0.30147398,  0.00711154, -0.32747018,
  -0.21242818, -0.6285665 ,  0.07197864,  0.04365371, -0.6057312 ,
  -0.4549181 , -0.68784153,  0.17763785, -0.04928377, -0.51147145,
  -0.37339982,  1.1744914 , -0.5132463 ,  0.10672484,  0.08064109,
  -0.1372738 ,  0.23959664, -0.05862667, -0.46602738,  0.20703782,
  -0.25837502, -0.30816066,  0.2676007 ,  0.33373833, -0.24325693,
  -0.2331591 , -0.2577709 ,  0.20681265, -1.7081676 ,  0.4813549 ,
  -0.02649137, -0.31279096, -0.15336882, -0.11158963,  0.5853287 ,
  -0.7214857 , -0.26962477, -0.10687156, -0.41768706, -0.73840564,
   0.17710547, -0.22859791, -0.04615102,  0.3536654 ,  0.26998323,
  -0.24619213, -0.30312774, -0.10362171, -0.0316106 ,  0.22188288,
  -0.26844332,  0.28162804,  0.0069243 , -0.22008139,  0.7748182 ,
   0.11703297, -0.17816418,  0.26569042, -0.27067634, -0.5099463 ,
  -0.21433666, -0.978387  , -0.1775803 ],
 [-0.5561813 , -1.0907638 , -0.4642735 , -1.7834023 , -2.0298245 ,
   0.093715  , -1.0682522 , -2.0007596 , -0.5195038 , -0.3819812 ,
  -0.34698963, -0.1250275 , -0.70130247, -1.0816029 ,  0.0543376 ,
   0.70510525, -0.80852413, -1.4523166 , -0.6842129 , -0.07982587,
  -1.0759711 , -0.66468185,  0.3136306 , -1.0605637 , -1.0717454 ,
  -0.63231206, -0.70364   , -1.0567124 , -0.65435463, -0.24483043,
  -0.82697654,  0.3064205 , -1.0935742 , -1.9518602 , -1.1656024 ,
  -0.17873771, -0.53418326,  0.02708986,  0.06676082,  0.05037137,
  -1.544899  , -0.92753357, -0.456035  , -0.40113497, -0.8647233 ,
  -0.14895941, -1.2726797 , -1.5014676 , -0.62614363, -0.90870667,
  -0.19135799, -0.8288644 , -1.465061  , -0.7507548 , -3.7101865 ,
  -0.5112107 , -3.851958  , -0.4629731 , -2.353548  , -0.6671688 ,
  -1.8393257 , -0.35366306, -2.6053464 ,  0.52736276, -1.0619    ,
  -0.35989544, -0.6412597 , -0.02464578, -1.2741109 , -1.1315459 ,
  -0.76374567, -0.596807  ,  0.09410118,  0.4483272 , -0.81373125,
  -1.112599  , -0.7503296 , -0.7233798 , -0.6135596 , -1.7709022 ,
  -0.24064115, -0.149332  , -0.47303525, -0.9718541 , -0.41975084,
  -0.9556518 , -0.2988792 , -1.3658266 , -0.29342622, -0.58957267,
  -0.42414778, -0.95256066, -0.475101  , -2.9859078 , -0.78285944,
  -0.14006971, -0.5328725 , -0.839115  , -0.17721464, -1.3882606 ,
  -1.5542129 , -0.12284841, -0.00091687, -0.21020909, -0.70244914,
  -0.5072334 ,  0.16874774, -0.75897   , -0.74694824, -0.4305215 ,
  -0.42614657, -0.97313976,  0.05739004, -0.19287086, -2.1293437 ,
  -1.8914531 , -1.2765146 , -0.8149929 , -1.2815711 , -1.035953  ,
  -0.22303869, -0.4794605 , -1.2253858 , -0.31596476, -0.6665293 ,
  -0.603652  , -0.4204995 , -0.5701626 ],
 [ 0.71203744, -0.32165244, -1.2563736 , -0.60926104, -0.7479695 ,
  -0.70167565, -0.20279634,  0.23978156, -0.5686462 , -0.714628  ,
  -0.11049502, -0.5683246 ,  0.36772102,  0.06215483,  0.15298294,
   0.9029686 ,  0.00667557, -1.0149078 ,  0.64008886, -0.01145101,
   0.03395171,  0.08993178,  0.7560444 , -0.09576976,  0.5686394 ,
   0.13861075, -0.38068503, -2.055933  , -0.13689521, -0.31505612,
  -0.2094216 , -0.9004632 ,  0.78660953, -0.2671629 ,  0.02318592,
  -4.3947015 , -0.20560983,  0.11703609, -1.0730983 ,  0.26151377,
   1.1528779 ,  0.95652246,  0.02951425, -0.6775413 ,  0.82239115,
  -0.90744996,  0.89958084,  0.15551075,  0.03317736,  0.64223224,
   0.06564555, -0.8001162 , -0.22243446, -0.24109381,  1.3720329 ,
   0.6845015 ,  0.6206625 , -0.5553228 ,  0.20949076,  0.06557378,
  -0.2244823 ,  0.63342726,  0.07825009,  0.31249887,  0.80505097,
   0.0820751 , -1.130343  ,  0.38688996, -0.00323382, -0.1885417 ,
  -0.94463974,  0.87965447, -0.13361311,  0.7002774 , -0.146647  ,
   0.57393885, -0.04440487, -0.07974212, -0.88985735,  0.38781458,
  -0.0705782 ,  0.05346748,  1.024125  , -0.2058489 , -0.40279925,
   0.8781327 ,  0.3622329 ,  0.23176226, -2.0948083 , -0.09176347,
   0.26920575, -1.8828292 ,  0.19173875, -1.1336393 , -0.20242897,
  -1.175223  , -0.3090319 , -0.02906247, -0.7710268 ,  0.1441266 ,
   1.879389  ,  0.0161506 ,  0.11272311, -0.50305   , -0.24435356,
   0.27187952, -1.3214422 ,  0.12253308, -0.04816846,  1.1551751 ,
  -0.19483891, -1.8717616 , -0.60014445, -0.5664787 ,  0.72219414,
   0.08173785, -1.5856684 ,  0.33543763,  0.49048084,  0.19600222,
   0.7176364 , -0.2286702 ,  0.4982695 ,  0.10262958,  0.06973478,
   0.0547027 ,  1.2203106 , -0.01402153],
 [-0.19783945,  0.05868777, -0.47940627, -0.4198263 ,  0.05083268,
  -0.46670195,  0.31153888,  0.00835508, -0.42139378,  0.30830467,
   0.54906094,  0.14683326,  1.0215001 , -0.44743997, -0.29511362,
   0.39009133, -0.52588856, -0.45282218, -0.07527521,  0.9384866 ,
  -0.6127    , -0.55580884,  0.1220876 , -0.01252807, -0.40145656,
  -0.8683587 , -0.3884187 , -0.0184785 ,  0.3401695 , -0.16929246,
  -0.5799963 ,  0.03313359, -1.2890396 ,  0.58404267,  0.39405343,
   0.41918755,  0.3548257 , -0.44514427, -2.9470835 ,  0.1455409 ,
  -1.8638467 , -1.7621822 , -1.1238624 ,  1.194173  ,  0.09479368,
   0.9218114 , -1.6166523 ,  0.09659123,  0.48585045, -0.7885215 ,
   0.36549613,  0.02322501, -0.01862753, -0.6188322 ,  0.16496374,
   0.2928787 ,  0.3638114 , -1.08283   , -0.27636796, -0.4263813 ,
   0.32083192,  0.3827081 , -0.0287264 , -0.4004016 , -1.160529  ,
  -0.2506702 , -0.72621745,  0.36733627,  0.05705857, -1.950906  ,
  -1.5142618 , -0.26499155,  0.06833177,  0.21375464, -0.00078669,
   0.19530393,  0.4049721 , -0.59295976, -0.08986043,  0.0446002 ,
  -0.3450933 , -0.016109  ,  0.3261003 ,  0.36567724,  0.00486251,
  -0.40358534, -0.6002128 ,  0.12293375, -0.88214123, -0.10147551,
  -0.22268456, -1.1339959 ,  0.15703604,  0.24233444,  0.43705985,
  -0.72881126, -0.43514106,  0.31083518,  0.0632311 ,  0.19627239,
  -0.9486847 ,  0.37888768,  0.23503573,  0.383512  , -0.32380065,
  -0.25631255,  0.26952237, -0.10895611,  0.07554502, -0.34754717,
   0.91143435, -0.953732  , -0.08165751, -0.58334947,  0.01862052,
  -0.692811  , -0.18303151, -1.0192901 , -0.16928409, -1.2170913 ,
   0.2828865 , -0.7068915 , -1.0973825 , -0.0949418 , -0.7325058 ,
  -0.14649782, -1.7058996 ,  0.40160823],
 [-0.64674884, -0.12762061, -0.12713407,  0.0402323 ,  0.3170992 ,
   0.6125615 , -1.6092764 , -1.1751151 ,  0.11915526, -0.76811874,
   0.17624904,  0.17712636,  0.42626527,  0.17953634,  0.6685969 ,
   0.33813164, -0.10469358, -1.6111528 , -0.4982448 ,  0.45665675,
  -0.01220361, -0.5867387 ,  0.61235106,  0.3957451 ,  0.10431749,
  -0.30863824, -0.41638368, -0.9125746 , -0.9190617 , -1.1023474 ,
   0.4997965 ,  0.33036882,  0.3800868 , -0.66250974, -0.78273666,
  -0.05863725, -0.6147724 ,  0.39639792,  0.14198266,  0.44245723,
  -0.29716045,  0.09746373,  0.38704497, -0.04417633,  0.11396869,
  -0.20970912, -0.9795991 , -0.0307818 , -0.45056915,  0.20339453,
  -0.02146737,  0.5560261 , -0.5083441 ,  0.16575807,  0.2431294 ,
  -0.16918771, -0.3069524 ,  0.6564417 , -0.14073117, -0.08899415,
  -0.47734636,  0.0719086 ,  0.73791194,  0.03476983, -0.15309542,
   0.40052798, -0.16792536,  0.07625131,  0.4526507 ,  0.5288864 ,
   0.00806496,  0.02606198,  0.13229486,  0.5980454 , -0.3731558 ,
   0.1416718 ,  0.30765516, -0.64513916, -0.2730621 ,  0.59437066,
  -0.22806297,  0.32020113, -0.0125228 ,  0.64610976, -0.59485835,
   0.3630415 ,  0.446121  ,  1.1094573 ,  0.07915025,  0.02745593,
  -0.4648809 , -0.291796  ,  0.94164395,  0.00816951, -0.3100841 ,
   0.14078559, -0.04869367, -0.43275493,  0.71411633,  0.29531816,
  -0.35891962, -0.14807065, -0.10477432,  0.25519118, -0.10578188,
  -0.43275297, -0.08828862, -0.51475173,  0.19812654, -0.40325657,
   0.20171282, -1.2687169 ,  0.4475264 ,  0.26855537,  0.07987072,
  -0.47407526,  0.13203937, -0.01583616, -0.25011086,  0.4436493 ,
   0.06257718,  0.25831994,  0.25967374,  0.4995751 , -0.32918093,
   0.27981415, -0.2030115 , -0.22636925],
 [-0.6265433 ,  0.1021293 , -0.35909873, -0.06054582, -0.22517322,
  -0.5888695 , -0.09066924,  0.19393846,  0.20167004,  0.10210753,
  -2.386655  ,  0.48834893, -1.5795413 , -2.3173783 , -0.0737783 ,
   0.23661563, -0.9285856 , -0.9242224 , -0.22725047, -0.46203732,
  -2.2787771 , -0.04252426, -0.6517134 ,  0.0720302 , -0.4584289 ,
  -0.3997954 , -0.9434682 ,  0.0489275 , -0.2963737 , -0.27773088,
  -0.03748254, -0.39838716,  0.47645512, -0.1594068 , -0.1580465 ,
  -0.07997718, -0.8699338 ,  0.25443864, -0.26105472, -0.11711007,
  -0.35130516, -0.78756183, -1.6540384 , -0.5620093 , -0.2811435 ,
  -0.1715418 , -0.8225879 , -0.20243749, -0.73530704, -0.24330741,
   0.16961342, -0.6473143 , -1.7987539 ,  0.2572264 , -2.0358572 ,
  -0.05693236, -2.260725  , -0.04227838, -0.10027255, -0.04080999,
  -0.1375764 , -1.1173357 ,  0.23804043,  0.39026845, -0.4244024 ,
   0.04708861, -1.167875  ,  0.04694028, -1.7487252 ,  0.3193498 ,
  -0.02183027,  0.11609301,  0.05778061,  0.53635937,  0.24613094,
  -1.1513588 , -0.2270466 , -0.11923169, -0.9267839 ,  0.35818398,
  -1.1659514 ,  0.20774195, -0.02586818, -1.6248481 , -0.09629644,
  -0.72455525, -1.6502467 ,  0.27443856, -0.5033371 , -0.17856641,
  -0.19767247, -0.35648766,  0.5739959 ,  0.5935507 , -1.3077273 ,
  -0.22017762, -0.43559226, -0.8025606 , -0.13754074,  0.27479357,
  -2.2066078 , -0.34902802, -0.60453784, -1.658166  , -0.2630411 ,
   0.03889747, -0.04555213, -0.3071618 ,  0.23346354, -0.1774756 ,
  -1.0154643 , -0.41387507,  0.07767702, -0.33068958, -0.1662742 ,
   0.17517908,  0.2289432 , -0.6174804 , -0.62210643, -0.46539873,
  -0.47889245, -0.10452403, -0.19317429, -0.38873446, -0.67015135,
  -0.734291  ,  0.05473707, -0.09442459],
 [-1.1971534 , -0.534417  , -0.00305861, -1.976612  , -0.7864388 ,
  -0.39018804, -0.7202015 , -0.36260846, -0.8757138 , -1.2859849 ,
  -0.42317513, -0.11131499, -0.5357444 , -1.1948665 ,  0.30405307,
   0.35271308, -0.44339216, -1.9006497 , -0.9093572 , -0.67552376,
  -0.57238954, -0.01537327,  0.02157705, -0.4672369 ,  0.03601286,
   0.03268376, -0.59221   , -0.18481587, -0.3712861 , -0.12812167,
  -0.00205625,  0.34829715, -0.19385238, -1.634636  , -0.7826635 ,
  -0.4019894 , -0.15231657,  0.23181668, -0.22200081, -0.07101136,
  -0.5320713 ,  0.1317829 , -0.39301974,  0.02607465,  1.224413  ,
  -1.340016  , -1.2648051 , -0.517598  , -1.3395727 ,  1.1546158 ,
  -0.49266696, -0.08637722, -0.08039513, -1.6716958 ,  1.7715605 ,
  -0.18769744,  1.4302778 ,  0.7224245 , -3.7219214 , -0.06446432,
   0.05778603, -0.07754661, -1.9921771 ,  1.337808  ,  0.33233005,
   0.539942  ,  0.0830216 , -0.35807708, -0.5402296 ,  0.17715147,
   0.05833563,  0.75159705,  0.23680367,  0.138337  ,  0.71486163,
  -0.06557864, -2.0409687 , -0.83062863, -0.6985296 , -0.79573464,
  -0.09829137, -1.3953559 ,  0.5708824 ,  1.7013445 , -0.00791996,
  -0.346626  , -0.16683236, -0.5174406 , -0.3868808 , -0.57796484,
  -1.4200077 , -1.5609757 , -0.03629737,  0.4709357 ,  0.02795731,
   0.12989438,  1.4704388 , -0.9721029 , -0.77209413, -0.58239347,
  -0.24103032,  0.78939587,  0.05851331, -0.16523656,  0.3155196 ,
  -0.17418815, -0.2613219 ,  0.537542  , -1.1707962 , -1.3746778 ,
  -0.3474845 , -0.34827894, -1.0101161 ,  0.06383403,  0.61360604,
  -4.281963  , -0.18410826, -0.2557721 , -0.8017877 ,  2.1539319 ,
  -0.10948639, -0.16458283, -0.31405398,  0.09543201, -0.03992595,
   0.17395893, -0.20932035, -1.0628916 ],
 [-0.6462073 ,  0.10589898,  0.29315156,  0.6586421 , -0.3800417 ,
   0.39398783, -0.132351  , -0.40222692, -0.05597901, -0.9016571 ,
  -0.9294049 , -0.01459293,  0.65176445, -0.35074595, -0.74635375,
   0.05813719,  0.03212151, -0.14010946, -0.6542282 , -0.71255267,
   0.3882472 , -0.37672725,  0.72411925,  0.10486475,  0.7094235 ,
   0.39841115, -0.41684404,  0.00985278, -0.08518957,  0.07931314,
   0.5022027 ,  1.3356278 , -0.10785777,  0.02825681, -0.6884047 ,
  -0.10195666, -0.4643006 , -0.0006691 ,  0.4421848 , -0.6398104 ,
  -0.8019853 , -0.03942408, -0.61522174, -0.7656234 ,  0.09897462,
  -0.1093794 ,  0.6446804 ,  0.22301483,  0.7113685 ,  0.05435801,
  -0.0964504 ,  0.43131253, -0.734183  ,  0.7059055 , -0.17490982,
   0.8891092 ,  0.41774505, -0.6654067 , -0.9406511 , -0.3638548 ,
   0.7010758 , -0.39702043,  0.1834135 , -0.00338126,  0.23184288,
  -0.42466465,  0.4268451 ,  0.51290786,  0.3070945 ,  0.01383103,
  -0.50763273,  0.34268928, -1.017093  ,  0.18466116, -0.29610398,
  -0.49798974,  0.08848744, -0.05042888, -0.01881783, -0.37030777,
   0.23180261,  0.18137093, -0.167385  ,  0.2356524 , -0.14153896,
  -0.0396322 , -0.3285429 ,  0.5454966 , -0.64265496, -0.01433506,
  -0.5653708 ,  0.03348171, -0.43279064, -0.27161348, -0.07916722,
   0.20525748, -0.8870666 ,  0.14404567,  0.36939684, -0.5285515 ,
  -0.5553811 , -0.05931492, -0.21614897,  0.5094917 ,  0.19174549,
  -0.8456464 ,  0.25733736, -0.08360716, -2.8892145 ,  0.14461431,
   0.59041953,  0.22594611, -0.09666064,  0.03168022,  0.5955827 ,
  -0.48173374,  0.42397243, -0.20646554, -0.33240566,  0.10961968,
  -0.5115428 ,  0.4620883 ,  0.26505432,  0.85038203, -0.1843797 ,
  -0.08656623,  0.5192288 ,  0.13731621],
 [ 0.40250632, -0.25038427, -0.9214625 , -0.7489165 ,  0.80431247,
   0.2063371 , -1.0671898 ,  1.0230943 ,  0.9721971 ,  0.1924397 ,
  -0.2624324 ,  0.6474737 ,  0.05649641,  0.32205653,  0.25715527,
  -0.25004688, -0.6171392 ,  0.51263374, -1.2439026 ,  0.02214975,
  -0.00237889,  0.8987955 ,  0.5211514 , -0.13281609,  0.00630671,
  -0.6185542 ,  0.17895159,  0.00343693,  0.05146657, -0.13782991,
  -0.6210072 , -0.1014899 , -0.18466578,  0.8062552 ,  0.03073347,
  -0.02171979, -0.31203613, -0.7648153 ,  0.07004552, -0.3325142 ,
  -0.04758478,  1.493067  ,  0.50343233, -0.28973773, -0.66257364,
   1.1731504 , -0.3240361 ,  0.28824598, -1.1579872 ,  0.51602733,
  -0.6089849 , -0.09224793, -0.02345608, -0.18740247,  0.6414599 ,
  -0.09297815,  0.61683476,  0.10412262, -0.250237  ,  0.45975757,
   0.50938463,  0.389932  , -0.17393945, -0.16365746,  0.02134453,
  -0.5962064 ,  0.6687331 ,  0.9310261 ,  1.1606565 , -0.42267564,
   0.34648386, -0.47035992,  0.31937584,  0.33750835, -0.13874085,
   0.7101215 ,  0.98581487, -0.07750738, -0.4315692 , -0.4067841 ,
  -0.08782922,  0.23955978, -0.3410329 ,  0.04651609,  0.04779083,
  -0.49299493,  0.27679417,  0.17750223, -1.1037493 ,  0.03494813,
  -0.33211708, -0.7387193 ,  0.2132803 ,  0.41036794,  0.06894443,
  -0.33556774,  0.07789244, -0.00636955, -0.32442415, -0.42175648,
  -0.32331696, -0.9783501 ,  0.23074797, -0.05421704, -0.2945342 ,
   0.6408242 ,  0.24224515, -0.08780793, -0.57694507,  0.2520738 ,
  -0.42629793,  0.43559262, -0.5051742 , -0.03781825, -0.00379941,
  -0.25640723,  1.1507583 , -0.16311559, -0.63767755,  0.9656969 ,
  -0.85245794, -0.06354298, -0.3658961 , -0.18672195, -0.06544992,
  -0.34251714,  0.30064282, -0.10198429],
 [-0.2061877 ,  0.6203962 ,  0.29163766, -0.23591712, -1.2217962 ,
   0.35287273, -2.1851258 , -0.62663585,  0.25392425,  0.20958093,
   0.0142579 , -1.037811  , -0.3488655 ,  0.23612481, -0.6382263 ,
   0.3615853 ,  0.2303162 ,  0.0889956 ,  0.6836987 ,  0.07922587,
  -0.04647023, -0.21319368,  0.3555683 ,  0.41356236, -0.95861214,
  -1.1336398 , -0.34233636,  0.09710529, -1.1022849 , -0.11340178,
   0.47462007,  0.4244868 , -0.25869235, -0.3152038 , -1.9290154 ,
  -0.6488095 , -0.60560715,  0.33795634,  0.72524595, -0.5184395 ,
  -1.809359  , -0.12937294,  0.06530648,  0.70324785, -1.8623055 ,
  -0.3894688 ,  0.07469209, -1.4348419 , -0.22635219, -1.130341  ,
   0.26970568, -0.42580554, -0.23643447,  0.18195237,  0.18085498,
  -1.1211641 , -0.00441406, -0.63039374,  0.30326083,  0.44744316,
  -0.0373445 ,  0.93055385, -0.83301175, -0.7859967 , -1.8273984 ,
  -0.03032167,  0.41912696,  0.6648036 , -1.5735505 ,  0.9156065 ,
   0.9135471 , -0.31792942, -0.42978093, -0.24583055, -0.09395073,
  -0.67587924,  0.54517454, -0.0062202 , -0.39511073, -0.1226858 ,
   0.22745274,  0.01458428, -0.30401722,  0.06055848, -0.04741361,
  -0.5345255 , -0.3564071 , -0.5073005 , -1.0469414 , -0.8958041 ,
  -0.64721   , -0.32005495, -0.40450612, -0.41275573, -1.1979911 ,
  -0.29619682, -0.5729027 , -1.0048624 , -1.2799605 , -0.45484516,
   0.01647454, -0.6987262 , -0.23301563,  0.419966  ,  0.41149586,
  -0.44612986, -0.30611682,  0.33822203, -0.42771143, -0.7536958 ,
  -1.6090304 ,  0.54982966, -0.5830854 ,  0.28792557, -0.44527987,
   1.1275536 ,  0.11480931, -0.47662956, -0.5208083 , -0.20589146,
   0.606394  , -0.3167395 , -0.65570647, -0.3896532 , -0.14343373,
  -0.51958925, -0.5350923 , -0.04239128],
 [ 1.448553  ,  0.58703786,  1.1144474 ,  0.6516859 , -0.14626001,
  -0.8461718 , -0.10459343,  0.918326  ,  0.44275913, -0.72484905,
  -0.30458385,  0.27951822, -0.06952778,  1.3481301 ,  0.2622871 ,
  -1.4446822 , -1.5680968 ,  0.678121  ,  0.69446903,  0.41706645,
  -0.43483487,  0.3196453 , -0.56233454,  0.3541748 , -0.04183061,
  -0.65525866, -0.23228632,  0.38420445,  0.3738332 ,  0.8757254 ,
   1.5646596 , -0.04992326,  0.54725194,  0.7106    ,  0.24671705,
  -1.3437592 , -0.2059778 , -0.02179701,  1.338675  , -0.11592863,
   1.2845691 ,  2.5848854 ,  0.63617975, -0.09495309,  0.7435982 ,
   0.47754532,  3.1109807 , -1.4288977 ,  3.105319  , -0.7346444 ,
   0.1865731 , -0.01116573,  1.7018403 , -1.035634  , -0.27120847,
  -1.7189673 , -0.01423035,  2.1453743 , -0.8425336 ,  2.242589  ,
   0.44867384,  0.3525601 , -0.23938511,  1.4401941 , -0.11981482,
  -0.6233646 , -0.92806613, -0.8453838 , -1.7929306 ,  1.9772457 ,
  -0.44164762,  0.35998306, -0.43742982, -0.10086059,  0.35641888,
  -0.3551105 ,  0.633029  ,  1.1906108 , -0.00595921,  0.3915136 ,
   0.04839947,  0.03126173, -0.23052756, -0.01609137, -0.404272  ,
   1.5894382 ,  0.26068664,  0.13171254, -0.4318887 , -0.21778728,
   0.85525924, -1.1472733 , -0.25535417,  1.0711629 ,  0.36171332,
  -0.78017145,  1.1273562 ,  0.04479029, -0.24052659,  1.9650713 ,
   1.1514602 , -0.0665598 ,  0.04827131, -0.3273146 , -0.92814696,
  -0.70786613, -0.6908796 ,  0.25075576,  0.12201245,  0.783054  ,
   0.6989291 ,  1.1220912 , -0.35847792,  0.03169489, -0.25127202,
   0.60834926, -1.1040928 ,  0.08913255, -0.18277997, -0.3964651 ,
  -0.9713795 , -0.38656205,  0.01188855, -1.134388  ,  0.16908552,
   0.14706315,  0.7429204 , -0.30076453],
 [ 0.13892911, -0.36049286, -0.52878886,  0.19257921,  0.8182327 ,
  -0.2951388 ,  0.10037886, -0.55041534, -0.60388756, -0.4204408 ,
   0.21104993, -0.5906253 , -0.29468653, -1.4981272 ,  0.18143308,
   0.13052388,  0.31293574,  0.4140611 , -1.217272  ,  0.67493993,
   0.18089959, -0.00394315, -0.42408383,  0.03701596, -0.2893567 ,
   0.04363732, -0.47567454, -1.2737544 , -0.56053376, -0.16534321,
  -0.17487903, -0.42651254,  0.05049524,  0.04959768,  0.31995443,
  -0.5031693 , -0.2915052 , -0.12821129,  0.2136749 ,  0.01138901,
   0.23105548, -0.06231044,  0.4844241 , -0.09310385,  0.23355639,
  -1.4365208 , -0.4895094 ,  0.09937587, -1.1135204 , -0.7007921 ,
  -1.0565858 ,  0.6154918 , -0.43784702, -0.8909907 ,  0.01731405,
  -0.34447706, -0.7493229 , -0.3881428 ,  0.63919395, -1.0781112 ,
   0.15670037,  0.02978299, -0.08434212, -0.4444636 , -0.33752418,
   0.43543196,  1.0219307 ,  0.14986683, -0.8174351 ,  0.02009923,
   0.3518072 ,  0.27459583,  0.4044137 ,  0.05418983,  0.14691202,
   0.09092078,  0.23986824,  0.18397292, -0.02815037,  0.39686534,
  -0.07263742,  0.5975399 , -1.015936  , -0.0053749 , -0.04863029,
  -0.60781336,  0.20297019, -0.39177638,  0.4481571 ,  0.1075417 ,
   0.60355616,  0.02800617,  0.12191784,  0.316913  , -0.3256859 ,
  -0.1423518 , -0.2619704 ,  0.4111671 , -0.40534678, -0.610887  ,
  -0.958678  , -0.18547937, -0.697528  , -0.7865821 ,  0.3506548 ,
  -0.25346762, -0.83091897, -0.01814221,  0.40211645, -0.3176274 ,
  -0.3041338 , -0.5808014 , -0.06292349,  0.3503364 ,  0.13120133,
  -0.3550174 , -0.15781473, -0.36217928, -0.00200163,  0.23625536,
  -0.19095199,  0.400819  , -0.45617872, -0.4875336 , -0.23479629,
  -0.88627803,  0.22891052, -0.68697846],
 [ 0.77755547,  0.4014983 ,  0.35228765,  0.6161548 ,  0.3214273 ,
  -0.5287201 , -0.2589521 , -0.66861343,  0.10894746, -1.0911682 ,
  -0.02793319,  0.08295223,  0.48290756,  0.94715315,  0.80866647,
  -0.10676906,  1.355074  ,  0.36869878,  0.01426839,  0.7511757 ,
   0.19755523, -0.28126693,  0.70102674,  0.13584422,  0.3516464 ,
   0.1961916 , -0.13066843,  0.0400659 ,  0.31831107, -0.2071943 ,
   0.24107711, -0.7773291 ,  0.26960152,  1.3727391 ,  0.586169  ,
   1.1226287 ,  0.49526677, -0.09001782,  0.32859668,  0.23444393,
   0.6652309 ,  0.51224643, -0.27778864,  0.59186465,  0.29007623,
   1.1051875 ,  4.0811987 ,  0.3083013 ,  1.3243914 , -0.14735581,
  -0.06923987,  0.84496385,  1.5846524 ,  0.60969514,  0.68061125,
  -0.2520398 ,  0.92497593,  0.45136184, -0.26340047, -0.11611668,
   0.6186619 ,  1.6116825 , -0.0047456 ,  0.15718015,  0.04997096,
   0.04684476,  1.0612882 , -0.8112    ,  0.7220194 ,  0.6532245 ,
   0.07390385, -0.29011196,  0.09294683, -0.21914355,  0.4537251 ,
   0.50524616, -0.6712218 ,  0.3212826 ,  1.2013439 ,  0.07860219,
   1.3192539 , -0.17549206,  0.36823225,  0.57998484, -0.12640196,
   0.18862808,  0.9058689 , -0.1516316 ,  1.3964245 , -0.3562637 ,
  -0.2721737 , -0.6488761 , -0.40801737,  0.32572418,  0.90802145,
   0.10643566,  0.1064382 ,  0.64854187, -0.1049739 , -0.77492243,
   1.8224729 , -0.02002003, -0.04900986,  0.18953763, -0.23910703,
  -0.20165472, -1.1611545 ,  0.12153482, -0.17626378, -0.9474138 ,
   0.16494066, -0.21632016,  0.07797926, -0.2804884 ,  0.6345607 ,
   0.4627512 , -0.48853347,  0.16478375, -0.05338877,  0.18301289,
   0.10095775,  0.2600481 ,  1.1205508 ,  0.16030647,  0.04933423,
   0.60275805,  0.24226502,  0.03183626],
 [ 0.1419247 , -0.14063463,  0.40608057, -0.21075109, -0.49788335,
   0.3455302 ,  0.39516687, -0.16683301, -0.25945026, -1.3195394 ,
   0.42126203, -0.4062733 , -0.42125174,  0.19621295, -0.2560227 ,
   0.16818184, -0.22320701, -0.02639135, -0.06093886, -0.2783462 ,
   0.40082216, -0.12640716,  0.47305003,  0.1519522 ,  0.2293813 ,
   0.5776358 , -0.31161284, -0.52751   , -1.2336134 , -0.8969171 ,
   0.2330518 ,  0.48603523,  0.04672838,  0.52846366,  0.19690625,
   0.7300844 ,  0.17436968,  0.05093207, -0.65424913,  0.28569582,
  -0.29410723,  0.8513945 ,  0.20728663, -0.12816198, -0.5281074 ,
   0.28621027, -1.2246704 ,  0.7910432 ,  0.2126322 ,  0.2691966 ,
   0.1405286 , -0.54308826,  0.49644282,  0.16087537,  0.34808242,
   1.2222328 , -0.4396537 ,  0.7095432 , -0.3852729 ,  0.00667204,
  -0.33668542, -0.4520281 , -0.6214747 ,  0.16536444,  0.64075506,
   2.7940748 ,  1.4057999 ,  0.40466407,  0.03124757, -0.1783353 ,
   0.9198476 , -0.60061747,  0.15622117,  0.1335006 ,  0.11418714,
   0.2577796 ,  1.336524  ,  0.1933789 , -1.1375362 , -0.01464068,
   0.13006139, -0.25911605,  0.15227814,  0.90871197, -0.42819732,
   0.1003812 ,  0.41650724, -0.08277216, -0.01814681, -0.7810301 ,
  -1.0556178 , -1.0244956 , -0.16172972,  1.1191751 ,  0.08099864,
  -0.25485238,  1.1152061 ,  0.11843296, -0.27719092, -1.2301792 ,
  -0.49089363,  0.40932947,  0.23339422,  0.37088576,  0.33218288,
   0.15417342, -0.29588968, -0.71342164, -0.32192785,  0.96908426,
  -0.5751605 ,  0.62884164, -0.33993343,  0.6106407 , -0.26633748,
   1.5383712 ,  0.2565403 , -0.18449773,  0.24382661,  1.4959303 ,
   1.114474  , -0.5305488 , -0.98654866,  0.14153852, -0.12033314,
  -0.08454828,  0.26957685,  0.0110715 ],
 [ 0.6873673 ,  1.0039383 , -0.6437731 , -0.8126027 ,  0.6253158 ,
  -0.9800504 ,  0.50467247,  1.834791  , -0.99829346, -0.03239196,
   0.55822504, -0.13720575,  0.18791337, -0.31682935,  0.75477153,
   1.0971557 ,  0.7831162 ,  0.50359136,  0.05421742,  0.23022509,
   0.05214795, -0.28533784,  0.29101762, -0.22427958, -0.03478982,
  -0.41040272,  0.5731055 ,  0.32563046, -1.2467449 ,  0.2094065 ,
   1.0192652 , -0.6754773 ,  1.235506  ,  0.28328973,  0.38626823,
   0.06452588,  0.77679175,  0.06212466,  1.5325472 , -1.0046477 ,
   1.3553028 , -0.3924355 ,  0.13232575,  0.08966438, -0.18172187,
  -1.3376551 ,  0.77846277,  1.4572121 ,  0.1700624 ,  0.09890883,
   1.0881823 , -0.22321706,  0.16364524,  0.5983466 , -0.14751582,
  -0.49390218,  0.29392952,  0.27052176, -0.65699774,  0.38856542,
   1.5811728 , -0.03971645, -0.25170183, -0.40596226, -0.739702  ,
  -0.42063823, -0.13782918, -0.13405213,  0.26482686,  0.8365918 ,
   0.01997322, -0.17334825, -0.06364969,  1.9322326 , -0.6858132 ,
   0.20998481,  0.70132023,  0.48803672,  0.26925382,  0.07358588,
   0.99052227,  0.31108484,  0.40675834,  0.22855517,  0.4103911 ,
   0.11322505,  0.27178183, -0.74598545, -0.34045002,  0.14142807,
  -0.6289159 ,  0.06960428,  0.08760233,  0.20879218,  0.32814544,
  -0.00263088, -0.63810736, -0.03691893, -0.30068475,  1.9425757 ,
   0.46162507, -0.2162241 ,  1.6533706 , -0.05849476, -0.38867033,
   0.8801147 ,  1.3551037 , -0.3394704 , -0.589458  ,  0.8639442 ,
  -0.6824934 ,  0.8245685 ,  0.5251385 , -0.68844634, -0.15265796,
  -2.4883478 , -0.33090904,  0.7390065 ,  0.1133385 ,  0.7237499 ,
  -0.35920373,  0.36930233, -0.04780763, -0.32510164,  0.36391672,
   0.14022896,  0.6092818 , -0.02251437],
 [-0.19235066, -0.00578505, -0.61786836, -0.39165026,  0.51249504,
   0.64975166,  0.59928375, -0.11566485,  0.3358713 , -0.3186517 ,
   0.83264774,  0.31124687,  0.85512465,  1.1901668 ,  0.37443957,
  -0.09653867,  0.30211818, -0.24929465, -1.1901133 ,  0.0981237 ,
   0.17769593,  0.14163679,  0.43073946, -0.17292954, -0.09629922,
  -0.16999833,  0.2914481 ,  0.50470376, -0.26431125, -0.14751899,
   0.03805495,  0.67484903, -0.11602504, -0.7434092 ,  0.6361917 ,
  -1.6031796 ,  0.387574  ,  0.19041605,  0.39187273, -0.82576895,
   0.45228255,  0.8845011 ,  0.13956453, -0.04509645,  0.06665442,
  -0.17347293,  0.26960167,  1.3406544 , -1.0214851 ,  0.24894649,
   0.04110058, -0.17476958,  0.53233266, -0.08178981, -0.11336467,
   0.25175446,  0.36352277,  1.4604398 , -0.44079277,  0.05486803,
  -0.0613372 , -0.46726274, -0.6469697 , -0.05405212,  0.24387074,
   2.6784832 ,  0.8409035 , -0.9713384 ,  0.8579152 ,  0.24310662,
  -0.19030587,  0.01536343,  0.5635399 , -1.7410555 ,  0.5013797 ,
  -0.04610522,  2.337004  ,  0.25005007,  0.13857515, -0.0346645 ,
   0.09098518,  0.21788372,  0.90190256,  0.22216816,  0.26892   ,
   0.75766885,  0.11950817,  0.0240074 ,  1.3016031 , -0.13290516,
  -0.017872  , -0.45361724,  0.05897724,  2.284858  ,  0.22331421,
  -0.18215647,  0.15281866,  0.6449388 ,  0.07295945, -0.18403351,
   1.2172827 ,  0.48765695, -0.66175973,  1.9037611 ,  0.00716829,
  -0.24452311, -1.1473888 ,  0.10300755, -1.012702  ,  0.21454728,
   0.12836568, -1.0392541 , -0.69331753,  0.61221856,  0.65983343,
  -0.48641202, -0.42202747,  0.21235685,  1.306031  ,  0.13610068,
   0.32956928,  0.40930524,  0.6427777 ,  0.539536  ,  0.40345183,
  -0.01431343, -0.59501517, -0.13356924],
 [ 0.45183668, -0.2870997 ,  0.23194493,  0.7192629 ,  0.7304247 ,
   0.50905734, -0.45163852,  0.23164742,  0.1993064 , -0.62403136,
  -0.09295265, -0.00298329, -0.6168091 ,  0.4738836 ,  0.06817427,
   0.47638246, -0.10029434, -0.15480895,  1.0729047 , -0.23365483,
  -0.23668706, -0.42473638,  0.52214044, -0.1419742 , -0.4500017 ,
   0.06131078, -0.59495234,  0.7669816 ,  0.67345816,  0.05003243,
  -0.45211023, -0.7985671 , -0.14659603, -0.50666946,  0.3165311 ,
  -0.6851613 ,  0.25829595,  0.01061248, -0.613909  ,  0.02558803,
   0.47369963, -1.9721484 , -0.28823665, -0.21159743, -0.24278873,
   0.44662583,  0.03645426, -0.17950127,  1.0515172 , -0.9221539 ,
  -0.82147133,  0.15293923, -0.35659257,  0.11243553,  0.17915098,
  -0.11452724,  0.6364059 , -0.2636331 , -0.1466999 , -0.002674  ,
  -0.6775802 , -0.32527065,  0.264291  ,  0.18746346, -0.19758661,
   0.1575687 ,  0.5543283 , -0.24249503, -0.34691063, -0.28430295,
  -1.1968558 , -0.46643585, -0.88194203,  0.0977744 ,  0.06050669,
   0.04799584, -0.6556124 , -0.02294669,  0.5504304 , -0.42588937,
   0.15548535, -0.0618977 ,  0.4022384 , -0.5494308 ,  0.29619235,
  -0.54635274,  0.52238935, -0.29844102, -0.13210103, -0.4485876 ,
  -0.5019748 , -0.7293143 , -0.5753359 , -0.0195525 , -0.07457559,
  -0.1260985 ,  0.5068361 ,  0.22814555, -0.21276197, -0.3042283 ,
  -1.6268066 , -0.3065124 ,  0.06329378, -0.07821188,  0.05490277,
   0.42018458, -0.5832572 ,  0.27301195,  0.14629653, -0.3168337 ,
  -0.42414567,  0.39049795, -0.49021408,  0.5142029 , -0.08067536,
   0.8478885 , -0.6774026 , -0.2632102 , -0.31987464, -0.5594532 ,
  -0.10412244,  0.02213424, -0.01137802, -0.26126078, -0.23912711,
   0.0017371 ,  0.07500701,  0.41883704],
 [-0.38465595,  0.01978221, -0.5120865 , -0.5986468 ,  1.717487  ,
   0.05987997,  0.4402865 , -0.25729048, -0.08192629,  0.57105833,
   0.5241068 ,  0.10993824,  0.89790106,  0.5436044 , -0.5834304 ,
   0.28340122,  0.63717747,  2.4362328 , -0.13778675,  0.13154198,
  -0.07652983,  0.19386442,  0.8051486 ,  0.05297585,  0.41641384,
   0.4320677 ,  0.1607724 , -0.47969916,  0.43259922,  0.7125644 ,
  -0.15580578, -0.2656241 , -0.3166661 , -0.26903573, -0.29536712,
   2.7453194 ,  0.33182335,  0.31798053,  1.4697579 , -0.18698658,
   1.015842  ,  2.2606082 ,  0.5452185 ,  0.06842543, -0.05580909,
   0.75554156,  2.596477  ,  1.5751177 ,  0.21599166,  0.6584884 ,
   0.9875542 , -0.65863   ,  0.67013913,  0.52353877, -0.3558823 ,
   1.9215369 ,  1.1615264 ,  0.67677873,  0.43639988, -0.41492295,
  -0.04819242,  1.1534544 , -0.3555842 ,  0.0966213 ,  1.343535  ,
   3.4500923 , -2.6700127 , -0.09722486,  0.5087258 , -0.09265187,
   0.44382167,  0.5189451 ,  0.20670839,  0.09759234, -0.13234851,
  -0.43449634,  0.23960695,  0.52211165, -0.44706738,  0.16816302,
   0.5747925 , -0.7304914 ,  0.4512259 ,  0.8765207 , -0.10323358,
   1.5392594 , -0.6619702 , -0.4177834 ,  0.5008894 , -0.24890284,
  -0.41319817, -1.4399322 , -0.10122544,  1.1060582 ,  0.3046281 ,
   0.71468693,  0.03145034,  0.3944179 , -0.5329665 ,  0.14673199,
   1.14133   ,  0.43591404,  0.21648015,  0.6185544 ,  0.07063366,
   0.58473724, -0.95593816,  0.1394004 , -0.89009887,  0.78348154,
   0.6150344 , -1.6835328 , -0.45897284,  0.31948116,  0.44610035,
  -0.10405884, -0.45506978,  0.3285174 , -0.29337433,  1.5551993 ,
  -0.10314049,  0.14806598,  0.7603707 ,  0.76056844,  0.17679009,
   0.20053451,  0.12217043,  0.40253204],
 [ 0.69883496, -0.7606835 , -0.05944803,  0.6099601 ,  0.2846715 ,
   0.3097309 ,  0.6065271 ,  0.22897467, -0.3093097 ,  0.13832013,
  -0.50123703,  0.46418968,  1.3258132 ,  0.9273968 ,  0.26987752,
   0.25602844,  0.7568087 ,  0.09901384, -0.4324716 , -0.16166079,
  -1.0465857 , -0.78970516,  0.2043742 ,  0.34030497,  0.36739427,
   0.6983395 ,  0.14614874,  0.73456895, -0.09388989, -0.7967326 ,
  -0.6967141 , -0.09430356, -0.47113693,  1.0166713 ,  0.36821604,
  -1.3261364 ,  0.22920603, -0.27695572,  0.26034772,  0.1912245 ,
  -0.4299645 ,  0.1213473 ,  0.9368302 , -0.25504795,  0.20119545,
  -0.34062353,  0.56166273,  0.28443015,  0.39192146,  0.40702572,
   0.61900145,  0.09682184,  1.1278502 , -0.2557113 ,  1.0551993 ,
   0.54263085,  1.0616782 ,  0.84078956,  0.34724563,  0.6836587 ,
   0.07243661,  0.8539411 ,  0.07238989,  0.04430454,  0.6075494 ,
  -1.1853919 ,  0.02575377, -3.2295382 , -1.6149915 ,  0.10620765,
  -0.10825181,  0.15452379, -0.09089378,  0.1217617 ,  0.4238011 ,
   1.297864  , -0.1691473 , -1.4362638 ,  0.1963465 ,  0.1254101 ,
   0.872635  , -0.09652373,  0.6511607 ,  0.8479202 , -0.13412349,
  -0.5327787 , -0.37690336,  0.04840448, -0.06134905,  0.7442975 ,
  -0.6917369 , -0.45417565,  0.02914134,  0.32191807, -0.07052854,
   0.23436016,  0.45279828,  0.18108544, -0.01592455, -0.22367227,
   0.5038273 ,  0.28193215, -0.27863863,  0.9718301 ,  0.01354992,
  -0.8300703 ,  0.15785567, -0.06573109, -0.18465738, -0.00808824,
  -0.9089507 ,  0.13287969,  0.29933974,  0.59682673, -0.00771519,
   0.36999094, -0.5080078 ,  0.08278638,  0.9026071 ,  0.18896294,
   0.38792443,  0.17883061,  0.2977778 , -0.2472884 ,  0.08210038,
   0.7451652 , -0.49427745, -0.23916139],
 [-0.17924926, -0.65527064, -0.43883237, -0.30010006, -1.0462288 ,
   0.40863898,  0.07650976, -0.06218641, -0.40591803, -0.15969129,
  -0.171892  , -0.02997849, -0.11648254, -0.0035016 , -0.37351146,
   0.37312967,  0.2745423 , -0.38311982, -1.4193904 , -0.062768  ,
   0.41297832,  0.31726995,  0.49315852, -0.5524515 ,  0.09982302,
   0.24299134, -0.470778  ,  0.14764857, -0.829295  ,  0.8767753 ,
  -0.71523595, -0.14854069,  0.4835802 ,  1.8968387 , -0.18997754,
  -0.06095617, -0.12325565, -0.29653016, -0.33684176, -0.21044987,
   0.6017084 , -0.22360034, -0.7127934 ,  0.12242645, -0.60996443,
   0.12468668,  0.73320127, -0.21316327, -2.0572107 , -1.5041145 ,
  -0.29624072, -0.7465747 , -0.8091925 , -0.6547527 , -0.21596996,
  -0.2846766 ,  0.04278275, -0.27148488, -0.4405229 , -0.8892996 ,
   0.7029038 , -0.03990826, -0.10795163, -0.11350661,  0.58361   ,
  -0.9165053 ,  0.0080881 , -0.37131014, -0.52114296,  0.31566027,
  -1.5224667 ,  0.63622296,  0.06386874,  0.32350603,  0.63001513,
   0.82030857,  1.1225834 ,  0.12218659, -1.0928279 ,  0.18088272,
  -0.33322346, -0.28916577, -0.32583904, -0.5284461 , -0.42311418,
   1.6290078 ,  0.17397171,  0.01246781, -0.33443522,  0.02938521,
  -0.0443182 ,  0.05483051, -0.27850512,  0.75488293, -0.8608206 ,
   0.07576642, -0.5384947 ,  0.00685235,  0.22235258, -0.83656746,
   0.6576563 , -0.05842428,  0.21556804,  0.6962616 ,  0.26649004,
  -0.4635816 , -0.5620801 , -0.27951133, -0.07464048, -0.07589246,
   0.6225988 , -2.3608153 , -0.4165984 , -0.06874146,  0.1990007 ,
   0.9225035 ,  0.4801365 ,  0.23358238, -0.26914018, -0.24783319,
   0.72338116,  0.00833859,  0.23380147, -0.47885942,  0.12978849,
  -0.11469252,  0.27751634, -0.07288712]])
    b2 = jnp.array([ 0.09769175, -0.12291728, -0.34042785, -0.17632829, -0.7344313 ,
  0.10988799, -0.20412716, -0.30413464, -0.03843142,  0.3676505 ,
 -0.09497325,  0.06932303,  0.33904374, -0.04700841,  0.631277  ,
  0.44620416, -1.6505188 , -1.0399599 ,  0.01708574, -0.4493682 ,
 -0.2732188 , -0.14335434, -1.0359335 ,  0.18183532, -0.22154835,
 -0.25778994, -0.61189044, -0.11665822, -1.3444922 ,  0.20834094,
 -0.65737844,  0.7985877 , -0.28858256, -0.4139807 , -0.2254307 ,
 -0.24717958,  0.09940699,  0.2468201 , -0.82596064, -0.7029487 ,
 -0.6443829 , -0.4515846 ,  0.22913776, -0.56000966, -0.12293912,
 -0.62634915, -0.57854295, -0.5263929 ,  0.23190723, -0.25858057,
  0.29033622,  0.46585837, -1.4162378 , -0.04564173, -2.4323204 ,
  0.02109577, -2.3120356 , -0.02978089, -0.04577754, -0.06506875,
 -0.468934  , -1.8048799 ,  0.05815032,  0.12276746, -0.11120178,
 -0.5092649 , -0.2537352 , -0.4613592 ,  0.03569265, -0.35886413,
 -0.18114759, -0.10945092, -0.3101323 ,  0.53701425, -0.09529908,
  0.52533364, -0.5817883 , -0.03986457, -0.54368204,  0.6141126 ,
 -1.8706151 ,  0.24751152, -0.2218712 ,  0.3220662 , -0.21720189,
 -0.16908121,  0.45718083,  0.21216825, -0.39727876, -0.06853646,
 -0.46085358, -0.00120944,  0.3644559 , -0.07044046, -0.6616947 ,
 -0.23783627, -0.28051808, -0.32575023, -0.12760441, -0.11771484,
 -0.6692813 ,  0.01266471, -1.018449  , -0.8261298 , -0.13276419,
 -0.19043452, -0.3285805 ,  0.7902487 , -0.20649365, -0.28615367,
 -0.5149687 , -0.08002198,  0.10479198, -0.41376474, -0.22735502,
 -0.6490746 , -0.39129674, -0.12487444, -0.8892508 , -0.89749026,
 -0.20096268,  0.04142992, -0.4714573 , -0.12642509,  0.0359032 ,
 -0.28498828, -0.29604536,  0.16412273])
    W3 = jnp.array([[-0.13540989, -0.08429768, -0.4540025 ,  0.44681695,  0.13879591,
   0.19960016,  0.00196397,  0.26621425, -0.04731092,  0.02462257,
   0.06714883, -0.31484735,  0.09580266,  0.3334353 , -0.9077771 ,
  -0.67545575, -0.6539194 , -0.17231126, -0.14777032,  0.10022922,
  -0.11999197,  0.17652635,  0.23536633, -0.11708727, -0.2518568 ,
   0.18862046,  0.06750397, -0.03361501, -0.01314581, -1.0104197 ,
  -0.39550188, -0.1660874 , -0.6395547 ,  0.06302315,  0.03565674,
  -0.03191706, -0.15428822,  0.21393554,  0.60763615, -0.44170123,
  -0.39031214, -0.2453389 , -0.04455627,  0.41094235, -0.13756923,
   0.06133348, -0.9883698 ,  0.14030145, -0.39412987, -0.364413  ,
   0.36431438, -0.3879612 , -0.8171772 ,  0.20102803, -0.02501706,
  -0.6188078 ,  0.3705093 , -0.01232322, -0.09235947,  0.5342411 ,
  -0.04084721,  0.2876527 ,  0.2665685 , -0.41132534],
 [-0.7395727 , -0.40021354, -0.4409798 ,  2.5077858 ,  0.35745144,
   0.2224819 , -0.44978377,  1.6400837 ,  0.36552018,  1.8882596 ,
   1.055155  ,  0.94469947,  0.92284197,  1.6085742 ,  0.84212416,
   0.94554776,  0.9782206 ,  1.9967278 , -0.08331259,  0.860774  ,
   0.59694767,  1.1120503 ,  0.8986935 , -0.16267392,  0.3809266 ,
   0.0186497 ,  0.5934806 , -0.4455637 , -0.47076184,  0.04007856,
  -0.5875895 ,  1.3973495 ,  0.28163028,  0.2884083 , -0.973402  ,
  -0.95324516,  2.4004285 ,  0.13555862,  0.3728242 ,  2.2177272 ,
  -0.1637905 ,  0.37783453,  1.1954997 ,  0.6325031 , -0.623656  ,
   0.72007585,  1.8158783 ,  1.2904248 , -0.25531024,  0.02738523,
   0.2837381 ,  1.7197059 , -2.1829662 ,  1.2050098 , -0.65472096,
   1.2014242 ,  0.1784905 , -0.57858646,  1.1381273 ,  0.15083902,
   0.30609074,  0.14639108,  0.7183038 ,  0.8904334 ],
 [-0.72859544, -0.8099867 , -1.8081988 ,  1.2132729 ,  0.8639777 ,
  -0.7259052 ,  0.3001178 , -0.65088904,  0.06549986,  1.11168   ,
  -1.9515411 ,  0.6556902 ,  1.6117319 ,  1.9028695 ,  1.0410371 ,
   0.70484185, -0.13602787,  0.3757756 ,  1.0584247 ,  0.6576444 ,
   0.58263457,  1.457202  ,  1.0787017 , -0.29181933, -0.31759378,
   0.8586505 ,  0.2449519 , -1.1241351 ,  0.09534998,  0.16569869,
   0.14481926,  0.5556885 ,  0.29858628, -0.05310893,  0.3700731 ,
  -0.5206215 ,  1.1747695 , -0.4838691 , -0.51583266,  0.43350926,
  -0.9363534 , -0.62087166,  0.7909766 ,  0.99622184, -0.8970112 ,
  -0.92949384,  0.94046533, -0.03197548, -0.8854954 ,  0.76118964,
   0.76274574,  1.1254824 , -0.75506294,  2.8243358 , -1.0895126 ,
   1.4923842 , -1.1585528 , -0.71349937,  0.13531752,  0.37361985,
  -0.13498645,  0.1089676 , -0.52862084, -0.15391359],
 [-0.3375143 , -0.519494  , -0.19196801, -0.08634499,  1.2360678 ,
   0.35448018, -0.5579453 , -0.03685473,  0.57679623,  1.3416641 ,
   1.8650303 ,  0.45142543, -0.29003337,  0.20736958,  0.2128629 ,
   1.0942433 ,  0.09843752, -0.07857838,  0.06613655,  0.742919  ,
  -0.48700118,  0.41809767, -0.00263995, -0.41391465,  0.21247275,
   0.21044803,  0.49193025, -0.06751637,  0.8874541 ,  0.4282589 ,
   0.08619339, -0.09018452,  0.17128886,  1.0859385 ,  0.74237394,
  -0.14546071,  0.0398832 ,  1.2054151 , -0.50233686,  0.28174537,
   0.15057094, -0.10225224,  0.4817087 , -0.31775275, -0.02683898,
   0.45020303, -1.309595  ,  0.23867732,  0.15737563,  0.36465135,
   0.24878258,  0.03533968,  0.15342093,  0.25888905,  0.42968667,
   0.05736667,  0.76071155,  0.23962456,  0.3515655 ,  0.21816115,
   0.16281475, -0.45733634,  0.11641968, -0.5808386 ],
 [ 1.5617516 , -0.09596212, -1.5166081 ,  1.5224359 ,  0.43331626,
   0.83390963,  0.28830945,  0.87237537, -0.26380065,  1.5171679 ,
   0.09014688,  0.59674716,  0.2564755 ,  2.4674685 ,  1.0062859 ,
   0.38330483, -0.37200084,  1.1145425 , -0.08641425,  1.3269564 ,
   1.6551162 ,  1.068744  ,  0.93015397, -0.42496955, -2.3201323 ,
   0.66138667,  0.27127787,  0.19695303,  0.7283431 ,  1.2901452 ,
  -0.53280187,  0.5933341 ,  0.45507085, -0.13656202,  0.62738657,
  -1.1431386 ,  3.4844406 , -0.46518198, -0.4322304 ,  0.869743  ,
  -0.5155587 ,  0.77269554,  0.5464933 ,  1.2201853 ,  0.10346595,
   1.1986535 , -0.20030552,  1.0881732 , -0.6840437 ,  0.8451136 ,
   0.18409468,  0.48644075,  0.08535463, -0.21413961,  0.4495132 ,
   2.3600392 , -1.091112  ,  0.00395147,  0.05617707,  0.18879367,
   0.2984501 ,  0.9572348 , -0.12864111,  1.3420136 ],
 [-1.0682201 , -0.07906821,  0.11281113,  0.11330868, -0.01892839,
   0.00381993, -0.20071359, -0.0699652 , -0.39516625, -0.5366134 ,
  -0.36699215,  0.88043714, -0.05704165, -0.01392595,  0.10711837,
   0.51030505, -0.34500173, -0.58305615, -0.8338663 , -0.35394388,
  -0.95977384, -0.308844  , -0.07064163, -0.6189151 , -0.08078724,
  -0.21167785, -0.42295897, -0.01795272,  0.60643405, -0.41467512,
   0.03285581, -0.03002878,  0.14026016,  0.27690786, -0.6339213 ,
   0.07931099, -0.33662984,  0.01751795,  0.37096253,  0.57271427,
  -0.06559528, -0.48316133, -0.11418277, -0.0890266 , -0.34773332,
  -0.3396354 ,  0.16478121, -0.40046883, -1.0937409 ,  0.552736  ,
  -0.27187985,  0.659641  , -1.6477883 , -0.6990792 , -0.41926575,
  -0.25093532,  0.45154807, -0.7718661 ,  0.3528931 , -0.3235442 ,
   0.16376674, -0.9823532 , -0.48303726,  0.13904095],
 [-0.65207326, -0.90094   , -2.11436   ,  0.4416761 ,  0.16865167,
   1.5469164 , -0.3261436 , -0.07931343, -0.647246  ,  2.2558744 ,
   0.20977952,  0.10458676,  0.7671679 ,  0.761515  ,  1.6684587 ,
   0.20769894,  0.5187174 , -0.12915248, -0.31356993,  0.36491725,
   1.6968999 ,  0.8768912 ,  0.31694356, -0.24697873, -1.9356576 ,
   1.0179112 ,  0.3162392 ,  0.2478361 ,  0.59307307,  0.92199737,
   0.33052057, -0.06318384, -0.88323   , -0.33652422, -0.00197229,
  -0.528992  ,  2.0039926 , -0.03682865,  0.25729808, -0.53465253,
  -0.5137392 ,  0.07213982, -0.1675241 ,  0.6814877 , -0.1877902 ,
   0.01774924, -0.09081281,  0.2337693 , -1.1033146 , -0.18816093,
   0.1250846 , -0.3672103 , -0.8437296 ,  0.92428964,  0.39934507,
   0.49364308,  0.36636642, -0.33408716,  0.2956513 ,  0.6970705 ,
  -0.33787572,  0.6909622 ,  0.25362927,  0.38851714],
 [ 0.14240249, -1.2654537 , -0.04675884,  0.94004416,  0.34098756,
   0.15878065, -0.4113715 ,  0.06596633, -0.16790193, -0.10961257,
  -0.37565103,  0.3634344 ,  0.63578033,  0.22678219,  0.63323337,
   1.1037631 ,  0.31498772,  0.37731662, -0.07617066,  0.47901687,
   0.24177553,  0.43194577,  0.40664613,  0.36419055,  0.14386104,
   0.12464502, -0.25985304, -0.05492009,  0.5080418 , -0.44421953,
  -0.35862035,  0.27840033,  0.02242298,  0.00281053, -0.43355218,
   0.03965669,  1.6067302 , -0.12845786, -0.27117723,  0.50788367,
   0.26313698, -0.04911374,  0.1161551 , -0.41291988, -0.97011536,
   0.37398335, -1.4005927 ,  0.52278835, -0.38522977, -0.14597848,
   0.21829067,  0.41384184,  0.1999951 ,  0.9374394 , -0.9699106 ,
   0.69280124, -0.00820762, -0.31897   ,  0.9710042 ,  0.1723731 ,
  -0.92567015, -0.1495092 , -0.49423888,  0.08253978],
 [-0.7520152 , -0.08728751,  0.19714385,  0.12918398,  0.73168045,
   0.07518277,  0.12154166,  0.2700434 ,  0.58251905,  0.17075104,
   0.23379156,  0.49543127,  0.37746602,  1.5495933 ,  0.2791504 ,
   0.08687364, -0.4480352 , -0.26779523,  0.00009025,  0.07632239,
  -1.2946218 , -0.08058099,  1.115431  , -0.306059  ,  0.2340242 ,
   0.30537388,  1.3581345 ,  1.0830683 ,  0.36174598,  0.09890791,
  -1.7584696 ,  0.5684042 ,  0.05930821, -0.00146652,  0.84903646,
  -0.9156161 , -0.33664465, -0.04290896,  1.7563438 ,  0.29697376,
   0.2825082 ,  0.08547702,  0.21079093,  0.2887609 ,  0.17071256,
   0.26607674, -0.68383306,  0.12707925, -0.30276552,  0.26451096,
   0.0970821 , -0.16861059,  0.6109127 ,  1.2674929 , -0.15713926,
   0.56789964, -0.4800851 ,  0.05572058,  0.7100854 ,  0.26517618,
   0.00543537,  0.52712613, -0.9163546 ,  0.5040269 ],
 [ 0.0523264 ,  0.03437271, -0.13279063, -0.55925393, -0.88748276,
  -0.9987841 , -0.15102783, -0.28451788,  0.05039666, -0.07305486,
  -0.48028716, -0.76142246,  0.19574681, -0.8478427 , -0.33333236,
  -0.8066867 , -0.35651156, -0.7888987 ,  0.5799455 , -0.4679268 ,
  -0.38022816, -0.9231634 , -0.08865437, -0.39408663, -0.38589066,
  -0.302399  , -0.40742385,  0.05391776,  0.47635663, -1.1821867 ,
   0.1541275 , -1.0735424 , -0.66554254, -0.5261368 ,  0.44012296,
  -0.01562876, -0.22046989,  0.03341841,  0.43024665,  0.41297182,
   0.18438512,  0.24877098, -0.29949227, -0.31243092,  0.24266544,
  -0.588004  ,  0.9507885 ,  0.4036222 ,  0.01256335,  0.09030983,
   0.317001  ,  0.23248276,  0.02239041,  0.17508489,  0.07361414,
  -1.1088967 , -0.40608224, -0.10675671, -0.8319587 ,  0.37035167,
  -0.30491006, -0.68785673, -1.2882942 ,  0.35276088],
 [ 0.9733036 ,  0.68029195, -0.6942278 , -0.70930153, -0.8009028 ,
   0.8057617 ,  0.9248722 ,  0.8300339 , -0.20680763, -0.82202625,
  -1.0427173 ,  0.89517885,  0.28615844,  0.54595333, -0.23303977,
  -0.02859229, -0.28314173,  0.06571046,  0.26313522, -0.69606674,
   0.06826298,  0.3215173 , -0.30859667, -0.9506741 , -3.061112  ,
   0.08635454,  0.8432267 ,  0.68045336, -0.14863671,  0.14996429,
   0.7047684 , -0.35551563,  0.33955988,  0.33293417, -0.66067743,
   0.27926332, -1.2874731 , -1.0354674 ,  1.2482266 ,  0.22853455,
  -0.40201354,  0.99960303, -0.04849599,  0.1392452 ,  0.04085146,
   0.41118953,  1.1760653 ,  0.33447114,  0.10325895,  0.01260027,
   0.1193741 ,  0.52973133,  0.35801688, -0.26491567,  0.44689465,
  -0.6014176 ,  0.96149176,  0.27691153,  0.05835961,  0.4924893 ,
   0.22443369,  0.4804265 ,  0.2553404 ,  0.20795757],
 [-0.6474216 ,  0.0527257 , -0.6186293 ,  1.0093144 ,  0.9182557 ,
   0.25381416, -0.5725046 ,  0.49901074,  0.54558665,  0.6213569 ,
   1.8161281 ,  0.04249868,  0.624302  ,  0.2619813 ,  0.5522855 ,
   0.33308113,  0.38873607,  1.2225134 , -0.3251253 ,  0.12144814,
   0.35187066, -0.19220507,  0.65620697, -0.10843738, -0.12061112,
   0.36749578,  0.67685723, -0.31524426,  0.6928956 ,  0.7200828 ,
   0.04519803,  0.6284258 ,  0.37854818,  0.72435606, -0.05121605,
  -0.09353761,  1.4125974 , -0.11445889, -0.3500467 , -0.33949053,
  -0.8929537 ,  0.84624934,  0.8209507 , -1.1419334 , -0.48105878,
  -0.7681831 , -0.5591948 ,  0.9238385 , -0.633293  , -0.22298203,
   0.5054219 , -0.5682823 ,  0.5617449 ,  0.44468397, -0.69381213,
   0.33634606, -0.25301167, -0.2862323 ,  0.02189165,  1.095845  ,
   1.170584  , -0.40616924,  0.41260192,  1.0519277 ],
 [ 0.38730034, -0.9027384 ,  0.18200482,  0.9746317 , -0.8385101 ,
  -1.2908174 ,  0.06260678,  0.91011053, -1.2203269 , -2.1080573 ,
   1.763777  ,  0.25596303,  0.30302322, -0.39222264, -0.51688457,
  -0.48215857, -0.54018307, -0.37776616, -0.68054414, -1.716357  ,
   0.11655075, -1.4489073 , -1.338774  , -0.25030693, -1.3638643 ,
  -0.5088195 ,  0.06466793, -0.79821473, -0.7053624 , -1.9969467 ,
   1.7911164 , -1.0590646 , -0.1906483 , -0.39755508,  0.6331477 ,
   0.2706588 ,  0.07885153, -1.0172025 ,  0.3855815 ,  2.0391424 ,
   0.53324026,  0.14761075,  0.4621623 ,  1.322848  , -0.9976885 ,
  -0.48666444,  1.4743075 , -0.40178415,  0.7838515 , -0.62217563,
  -1.136936  ,  0.9216943 ,  1.4830232 ,  2.2918296 ,  0.41591513,
  -1.548202  ,  0.31101772,  1.7738571 , -1.6739637 , -0.33310965,
   1.0516288 , -1.211124  , -0.4691822 ,  0.25936416],
 [-0.1267118 , -0.31412372, -0.60055083, -0.22345848,  0.5250091 ,
   0.10347332,  0.19454737, -1.7459193 , -0.43898505, -0.36871904,
   0.52346134, -0.43360874,  0.02727302, -2.5258644 ,  0.30555987,
  -0.4764197 ,  0.5018515 , -0.95089376, -0.08299588,  0.33631933,
  -0.9734563 , -0.6220703 ,  0.22050972, -0.26379555, -0.43434632,
  -0.40549952,  0.26930845,  0.37533674,  0.03892737, -0.44563034,
   0.39759794, -0.12472705,  0.73672396,  0.45413858, -0.56520754,
   0.53433305,  0.19588602,  0.2377694 , -0.23437071,  0.3931452 ,
  -0.05717356,  1.466512  ,  0.16837372,  0.5703124 , -0.90688807,
  -0.2812793 ,  0.5926473 , -0.3875986 , -0.63894737, -0.24339111,
  -0.12710813,  0.6860872 ,  0.39529967,  0.9977012 ,  0.03823355,
  -0.522253  ,  0.25352684,  1.3798614 , -1.0253751 , -0.28877953,
  -0.31782442, -1.0815711 , -0.4062466 , -0.5838414 ],
 [-0.7988049 , -0.61770564,  0.3599696 , -1.3057204 ,  1.1169709 ,
   0.30989438, -0.30107296, -0.9138001 , -0.11258436,  0.9636159 ,
  -0.67011964, -0.12717259, -0.43466675,  0.6218083 , -0.26973045,
   0.10583843, -0.5349321 , -1.0154868 , -0.58131313, -0.3789588 ,
  -0.41793957, -0.17425612, -0.15696062, -0.5454953 ,  0.07064389,
  -0.40721983,  0.9545787 , -0.28050795,  0.24843433,  0.12871754,
  -0.4854096 ,  0.57137537, -0.4547244 , -0.6184212 , -1.1390061 ,
  -0.23507826, -0.35086098, -0.58189744,  0.23568062,  1.2794669 ,
  -0.39763707,  0.14282013, -0.45522997,  0.15224703, -0.5571158 ,
  -0.22803846,  0.00169665,  0.73816687, -0.00614928, -0.10528664,
  -0.3994739 ,  0.8790138 ,  1.7875798 , -0.9213566 ,  0.400307  ,
  -0.54378575,  0.18533814, -0.03280128,  1.5294205 , -0.24078351,
   1.8189933 , -0.9663329 , -0.16792196,  0.19007096],
 [ 0.07858577, -0.14775051,  0.03454272, -0.43134895, -0.15455766,
  -0.66729265,  0.01146606, -0.98516333, -0.81238174, -0.8473412 ,
  -0.6766325 , -0.620343  , -0.2335786 ,  0.10988881, -0.24480495,
  -0.03320043, -1.0823566 , -0.8378858 , -0.2911819 , -0.3922278 ,
  -0.44328165,  0.0293913 ,  0.12920734, -0.09847514,  0.04149562,
  -0.8452556 ,  0.1359193 , -0.6669927 , -0.26679957,  0.10931922,
  -0.4738189 , -0.66637015, -0.3380643 ,  0.03606695, -0.4046063 ,
  -0.62066126, -0.54840326, -0.23560227, -0.27624893, -0.47420794,
  -0.12559809, -0.23674889, -0.21796413, -1.0075346 , -0.18992937,
  -0.80473405,  0.01520899,  0.16929229, -0.1444984 , -0.25602698,
  -0.07135287,  0.5187628 , -0.2907126 , -0.71105224,  0.00612168,
  -1.0746112 , -0.18863228,  0.22589427, -1.8789967 , -0.0458419 ,
  -0.55107874, -0.96452683, -0.01710654,  0.02054108],
 [-0.44959885, -0.79325956,  0.00465837, -0.15785238, -0.00288959,
  -0.47458318, -0.34530973, -1.0824635 , -1.2278221 , -1.6213728 ,
   0.850202  ,  0.29300725,  0.52945906, -1.7491701 , -0.68582666,
  -0.09382359, -0.70292205,  0.18490379, -0.25677544,  0.20701382,
  -0.41003978,  0.17704035, -0.5558748 ,  0.39077908,  0.31156325,
  -0.02355356, -0.41244498,  1.1568378 , -0.691535  , -0.9826873 ,
  -0.08080211, -0.23531029, -0.19770353, -0.8582429 ,  0.06817168,
  -0.4272733 ,  0.07891823, -0.53509635, -0.3443395 ,  0.63002086,
  -2.0744932 , -0.8216522 ,  0.12419239, -0.5096111 , -0.41262728,
  -0.8777459 , -0.2468117 ,  0.31067947, -0.01498542, -0.48501986,
   0.06511845, -0.8776201 ,  0.34348494,  1.9711647 , -0.66971505,
  -0.39286125,  1.0518396 ,  0.14457263, -1.3551869 ,  0.4121849 ,
  -0.32968277, -0.21006453,  0.6776539 , -0.09255797],
 [ 0.45057458,  0.55784905, -0.4064667 ,  1.3666643 , -1.0005647 ,
  -1.7869762 , -1.4086343 ,  1.0382733 ,  2.013771  , -2.4478946 ,
   0.8217487 , -0.4673467 ,  2.1295226 ,  0.1524633 , -0.61315966,
  -2.9776082 ,  0.10918988, -0.23091501, -0.24685054,  0.3611841 ,
  -0.36892092, -0.5004133 ,  1.9480882 , -0.64786136, -0.10017154,
   0.8262876 , -0.7205093 , -0.9214759 , -0.6691419 ,  0.9291687 ,
  -0.09373579, -0.15737201,  1.2723622 ,  1.3911396 ,  0.37373778,
  -0.46043134,  2.293088  ,  0.5279676 ,  0.0631799 , -0.13939719,
  -0.80353886, -1.5897903 ,  0.5896301 , -0.28316128, -0.3292281 ,
  -0.2267221 , -0.36264664, -1.8537629 , -0.17726968,  1.7737142 ,
   1.8506484 , -0.5211391 ,  4.635479  ,  3.1913643 ,  0.08127084,
   0.61809075, -1.2475886 ,  0.39193487, -0.9814648 , -1.0210952 ,
   2.377181  , -0.29305992, -0.8813699 , -0.31515858],
 [ 0.9557011 ,  0.07607498, -1.7091507 ,  0.8216593 , -0.05005129,
  -0.35845482,  0.10584962,  0.15889476, -0.11413605,  0.15541346,
   0.5154671 ,  0.5456039 ,  0.12157867,  1.0543346 ,  0.64968175,
   1.332051  ,  0.16021565, -0.21280648,  0.05196624, -0.5199229 ,
   0.7810833 ,  0.14867239,  0.36193237,  0.19523567, -2.0569766 ,
  -0.04915217,  0.4667279 , -1.161372  ,  1.0253376 ,  1.6408683 ,
   0.08965118, -0.95135856,  0.35534117, -0.14773887,  1.1341809 ,
  -0.14959633,  0.5377164 , -0.05476898, -0.3025509 , -0.8179159 ,
  -0.29527286,  1.7206602 ,  0.43195686, -0.00645167,  0.55119884,
  -0.40742108, -0.7725772 ,  0.3249464 , -0.46430466,  0.41572985,
  -0.20191893,  0.21197616,  0.22838046,  0.3664974 , -0.12461254,
   1.180731  ,  0.29305503,  0.9100542 , -0.00899684,  0.39982724,
   0.384942  ,  0.48979768,  1.0364085 , -0.01875194],
 [-0.02879672, -0.02484631, -0.14172989,  0.38776284,  0.8954408 ,
  -0.2601723 ,  1.2759876 ,  0.45979688,  0.71016794, -0.12905402,
   0.6306173 ,  0.8491909 ,  0.17485887,  0.28332365,  0.4215366 ,
   0.5123809 ,  0.04503525,  0.21601926,  0.7712124 ,  0.58012056,
  -0.04686994,  0.25994912,  0.38136575, -0.37992823,  1.2858425 ,
   0.5043601 , -0.13459328, -0.13413487,  0.16088648, -0.0475322 ,
   0.94411844,  0.5874081 , -0.01794941, -0.5696581 ,  0.89204234,
  -0.11673015,  0.7546106 ,  0.44150144,  0.06102636,  0.42915702,
  -1.5964372 ,  0.57403207,  0.65920323, -0.3622516 ,  0.49629694,
   0.12482111,  0.47920796,  0.14483559, -0.2858636 ,  0.18651064,
   0.23238784,  0.5618618 ,  0.3250202 , -0.59029883, -1.051517  ,
   0.30820614,  0.22852686,  0.44440395,  1.2401074 ,  0.40647283,
   0.09938644, -0.18156083,  0.7639723 ,  0.30530104],
 [ 0.19620945,  0.30192372, -0.56371444,  0.14798884, -0.05168363,
  -0.5969197 ,  0.12679514, -0.1489895 ,  1.4753377 , -0.06174201,
   0.7971656 ,  0.29920667,  0.99942964, -0.4892708 , -0.43032908,
   0.41524386, -0.28351584,  0.97805977,  0.31149524, -0.29590473,
  -0.7172373 ,  0.0784378 , -0.01324572, -0.46439275,  1.4574748 ,
  -0.5617414 , -0.26733944, -1.2985059 ,  0.5781826 , -0.16804229,
   0.9064012 , -0.1297097 ,  0.13808174,  0.2726387 ,  1.0776285 ,
   0.12744687,  0.6576206 ,  0.53488904, -1.177274  ,  0.2967102 ,
  -0.75841135, -1.7988002 ,  0.56510144, -0.55003345, -1.0105729 ,
  -0.38306093, -0.6239712 ,  1.0397756 , -0.5070309 , -0.3664483 ,
   0.77755123, -0.7522483 ,  2.2512565 ,  2.3928053 , -0.2532794 ,
  -1.861699  , -0.40930566, -0.84004486, -0.3876686 , -0.6684415 ,
   1.6259304 , -0.0732752 ,  1.3590255 , -0.43306577],
 [ 0.9130405 ,  0.14620593, -1.0900608 ,  0.76255715,  0.27850357,
   0.18859166, -0.03866802,  0.6918693 ,  0.3489338 ,  1.1277813 ,
   0.78054214,  0.59405154,  0.6946188 , -0.19443369,  0.5437484 ,
   0.6427364 ,  0.45991093,  0.98261255, -0.57993203,  0.06704854,
   0.24348494,  1.1098888 ,  0.250146  , -0.18423493,  0.8349801 ,
   0.21975139,  0.78393555, -0.7630967 ,  0.9752266 ,  1.2308828 ,
  -0.6439146 ,  0.26918855, -0.0254709 , -0.2977053 ,  0.3636494 ,
   0.03389828,  0.73481685,  0.15144292, -0.31131136, -0.14113824,
  -1.1929904 ,  0.58215   ,  0.15239272,  0.13651152, -0.69799745,
   0.81386954, -0.04519306,  0.84400135, -0.5498822 ,  0.136447  ,
   0.21836318, -0.31303093,  0.70559084,  0.8338769 , -0.13260087,
   1.0010682 ,  0.5765484 , -0.00417046,  0.2853703 ,  0.9480502 ,
  -0.33215094,  0.42835993, -0.13958633,  0.21876417],
 [-0.07394066, -1.0577782 , -0.01373343, -1.0962658 , -0.52723694,
  -1.1421843 , -1.6983912 ,  0.12357633, -0.16410235,  0.09787684,
  -0.10352173,  0.19662987,  0.16816294, -0.8929707 , -0.12582791,
  -1.2134354 , -0.6049171 , -0.21914092,  0.31233317,  0.12763865,
  -1.1275015 , -0.39578277, -0.44720256, -0.3788872 ,  0.06638292,
  -0.44691047, -0.24670903, -0.15506816, -0.3650971 , -1.0455424 ,
  -1.2521164 , -1.0168436 , -0.09987773, -0.52951646, -0.18548945,
  -0.12844437,  0.16289935, -0.27357453,  0.13986212, -0.02090725,
  -0.01951073,  0.25630343, -0.4310016 ,  0.0852863 ,  0.00938343,
   0.24000974,  0.2871309 , -1.0057508 , -0.23881464, -1.3386196 ,
   0.13292679, -0.8857396 , -0.26813516, -0.23291108, -0.7621434 ,
   0.02376446, -0.10800697, -0.5873961 , -0.91914874, -0.45954302,
  -1.0413817 , -0.07971748,  0.13751924,  0.3172194 ],
 [-0.04087142, -1.2033774 , -0.07495333,  0.7802647 ,  0.7132174 ,
   0.26490164, -1.0424058 ,  0.6335106 ,  0.9818694 ,  1.9533291 ,
   3.0669146 ,  0.21958324,  0.8670863 ,  0.34221873,  0.43150556,
   0.67989475, -0.19930628,  1.2259643 , -0.914764  ,  0.13459185,
   0.35455057,  0.4009396 ,  0.43684283, -1.0153404 ,  0.08301814,
  -0.3964718 ,  1.7011466 , -0.15231177,  0.97396624,  0.14897767,
  -0.29044008,  0.27896744, -0.30669388, -0.49547625, -0.5271202 ,
  -0.03175919,  1.1474605 , -0.02473213,  0.07358622,  0.54458606,
  -1.9159861 , -0.6617159 ,  0.03636061,  0.50447655, -1.5081642 ,
  -0.5617882 , -0.36920238,  1.0918003 , -1.6402096 , -0.15665539,
   0.9361238 ,  0.96708333,  0.04666644,  0.27691358, -1.2846911 ,
   2.0957341 ,  0.35209084, -0.6623197 ,  0.93426514,  1.0491925 ,
   0.21478777, -0.9762045 ,  0.64739877,  1.6680313 ],
 [-0.31893998,  0.1167643 , -0.05944777,  0.888474  ,  0.78699344,
  -0.04748525, -0.9891489 ,  0.92887264,  1.3231786 ,  1.3646427 ,
   1.1613889 ,  1.1890527 ,  0.8327574 ,  1.1025664 ,  0.8799921 ,
   1.1894655 ,  0.5026802 ,  0.9436427 , -0.273863  ,  0.07621013,
   0.679145  ,  0.7856471 ,  0.7015123 , -0.5260999 ,  0.09432738,
   0.03481966,  0.793324  , -0.40735483,  1.2428854 ,  1.8291999 ,
   1.419307  ,  0.8198054 ,  0.29908416,  0.7524084 ,  0.6383967 ,
  -0.6960703 , -0.7037606 ,  0.6363059 , -0.23999867, -0.88044703,
  -0.9844865 , -0.2475938 ,  1.5127013 ,  0.00132054, -0.9221911 ,
  -0.18270087,  0.05110887,  1.5103455 , -0.25531855,  0.2738621 ,
   0.52729404,  0.32105547, -0.04472814,  0.0338241 ,  1.785451  ,
   0.733285  , -0.14001413, -0.739132  , -0.20420985,  0.57797235,
   0.19493027, -0.41509372,  0.5133198 ,  1.3046783 ],
 [ 0.32690904,  0.47086936, -2.0538063 ,  0.21981104,  0.05570398,
  -0.71709114,  0.21577685, -0.37767795,  0.463297  ,  0.8167211 ,
   0.3082808 ,  0.47599477,  0.46575224,  0.6673066 ,  1.7916166 ,
   0.79221386,  0.38074604, -0.16536546,  0.44033912,  0.75381505,
  -0.10366812,  1.3143102 ,  0.7882622 ,  0.13897069, -0.53534335,
   0.5558732 ,  0.24109572,  0.41843212,  0.6446483 ,  1.3953551 ,
   0.7595047 ,  0.07503362,  0.48078784,  0.05903371,  1.184107  ,
   0.46805057,  0.96257305,  0.51375216, -0.02082588,  1.0305055 ,
  -1.5147495 ,  1.5813354 ,  0.90871227,  1.2948042 , -0.00737672,
  -0.04345635,  1.2719407 ,  0.01845662, -0.90735334,  0.63859016,
   0.36863428,  0.8186206 ,  0.13841686, -0.6051483 ,  0.7691189 ,
   0.11075798, -0.5758052 ,  1.4989414 , -0.32849455,  0.5423084 ,
   1.2365745 ,  0.7895405 ,  0.64581645, -0.42166793],
 [ 0.2563185 ,  0.33517766, -0.32834402,  0.01612667,  0.1823337 ,
   0.04988235, -0.17758115,  0.03430087,  0.27280596,  0.40066934,
   0.3070773 ,  0.07610349, -0.05717089,  0.48618594, -0.33320448,
  -0.09721082,  0.01549527, -0.09466284, -0.39735332,  0.0407442 ,
   0.18859151, -0.23409607,  0.8135714 , -0.02689355,  0.43703046,
   0.0829678 ,  0.37007576,  0.04680798,  0.11343567,  0.05757631,
  -0.17389283, -0.10338135, -0.28533006,  0.23758328,  0.13982369,
   0.04642519, -0.11677051,  0.06271197,  0.31801322,  0.3836842 ,
   0.02323464, -0.2583954 , -0.01527053,  0.03569046,  0.02138873,
   0.3556709 , -0.13291577,  0.55451655,  0.02706844, -0.24031384,
  -0.11775013,  0.28589752,  0.31545627,  0.15228933,  0.06858086,
   0.25733232,  0.46275142, -0.10510668,  0.04874133,  0.2072772 ,
   0.3547676 , -0.09527165,  0.5174569 ,  0.37139845],
 [-0.65630037,  0.40845308,  0.34527868,  0.85966355, -0.3268608 ,
  -0.55731314,  0.42361075, -0.13186988,  0.16963351,  0.06558762,
  -0.6673976 ,  0.6197049 , -0.08398312,  1.6785629 , -0.48040563,
   0.15084937, -0.32765707,  0.73445153,  0.17189723,  0.49913543,
  -0.20839204,  0.39217353,  0.6616345 , -0.1947761 ,  0.34858173,
   0.16873083,  0.79713285, -0.41208884, -0.00987277,  0.00270387,
  -0.39260638,  0.87060255, -0.10996281, -0.31752133, -1.0084175 ,
  -0.20452969, -0.68814826, -0.35244942, -0.3340775 ,  1.4710977 ,
  -1.221078  , -0.8221428 ,  0.6408831 ,  0.7334574 , -0.03146382,
   0.05801067,  0.9551841 , -0.2256291 , -0.25614533,  0.45454496,
   0.1764714 ,  1.0263093 , -0.01488261,  0.9799467 , -0.19772623,
   0.7374728 , -1.3668915 ,  0.01179407,  0.05932305,  0.27435136,
   0.13155204,  0.02469724,  0.12705846,  0.28956187],
 [-1.8724359 , -0.32321474,  0.00671394, -0.309357  ,  0.44231257,
  -1.1128377 , -0.05875359, -0.42300573,  0.23950234, -0.34593812,
  -0.04173502,  0.06654079, -0.1717595 , -0.19310467, -0.06799741,
  -0.13614051, -1.1783907 , -0.1198425 , -0.05688376,  0.6795593 ,
  -0.23914978,  0.58526874, -0.22778602, -0.05061075,  0.12845999,
  -0.36204934, -0.09868509, -0.14817552, -0.89131945,  0.29831305,
  -0.48963445,  0.14615455,  0.12577556, -0.53128326,  0.17070861,
   0.7446385 , -0.26943928,  0.39338055,  0.30852085,  0.35535038,
   0.26178122, -0.2290932 ,  0.08115491,  0.1897753 , -0.37257618,
  -0.20112975, -1.527897  ,  0.02995975, -0.3575614 ,  0.13068087,
  -0.1761562 , -0.39841416,  0.16929752, -0.18727967,  0.11052656,
  -0.896077  ,  0.84191793, -0.141774  , -0.03830529, -0.02506819,
   0.9211208 ,  0.503974  ,  0.03767062, -0.3533666 ],
 [-0.20677799, -0.19766104, -1.0822475 , -0.9448561 , -0.00322631,
   0.10455905, -0.2506642 ,  0.20939636, -0.41400847, -0.8167981 ,
  -0.10797998, -0.6488353 ,  0.28072226,  0.52652574, -0.17860381,
  -0.32450485,  0.42254537,  0.04823832, -0.06648975,  0.05297983,
  -0.63467693, -1.0126855 , -0.26176164, -0.710568  ,  0.08866745,
  -0.12777251,  0.6476979 ,  0.16091295, -0.56258273, -0.70190436,
   0.14610521, -0.00608611,  0.2020397 ,  0.149754  ,  0.27122   ,
   0.0339423 , -0.69265676,  0.01837851,  0.401797  ,  0.5727231 ,
  -0.516452  ,  0.3631947 , -0.40769288, -0.368637  ,  0.14136794,
  -0.05253097,  0.34276858, -0.697285  , -0.41256005, -0.16505744,
  -0.01120867,  0.5447233 ,  0.6516891 ,  0.39455354,  0.14004126,
   0.2981601 ,  0.2305878 , -0.19176704, -0.16412449, -0.10279178,
   0.20777121,  0.02138659,  0.79151136, -0.57301694],
 [-0.30819967, -0.5999825 , -1.4459373 ,  1.826789  , -0.18617706,
   0.7732857 , -0.04476428,  1.2123973 ,  0.33228412,  1.4551443 ,
   0.65962684,  1.1717448 ,  1.5815674 ,  1.0953496 ,  0.4642752 ,
   1.8411022 ,  0.6377296 ,  0.81756747,  0.76111823, -0.17702462,
   0.41379267,  0.80734867,  0.3803401 , -0.5384098 , -0.5515058 ,
   0.50542885,  0.6032731 ,  0.17256239, -0.7702658 ,  0.30276263,
   0.11457103,  2.447614  , -0.10749801, -1.2730104 ,  0.47398946,
  -0.7204805 ,  0.57590383,  0.21349403,  0.38748488,  0.73461604,
  -0.59656614, -0.5914693 ,  1.2555121 ,  0.9321983 , -0.38657653,
   0.9083155 ,  1.893757  ,  1.2694991 , -1.0432428 ,  1.6380814 ,
   0.7968594 ,  0.1275223 ,  0.4314009 ,  2.6449647 , -2.1412072 ,
   0.31017885,  0.3053701 ,  1.3530216 ,  1.5609429 ,  0.36863172,
   0.9549694 ,  1.6750795 ,  1.1734346 , -0.02449386],
 [ 0.605386  , -0.0469464 ,  0.3095027 , -0.18161345,  0.5901442 ,
  -0.21686682, -0.3307509 ,  0.07144287, -0.00426546,  0.20879178,
  -0.95192385, -0.8080127 , -0.3685426 ,  0.9813793 ,  0.27564645,
  -0.0864771 ,  0.09593832,  0.5898508 ,  0.01498389, -0.50231105,
  -0.33094546,  0.2900203 , -0.18583441,  0.26251245, -0.09878395,
  -0.6262028 , -0.73917174, -0.810591  ,  0.27421537, -0.29853943,
   0.2560306 ,  0.69726986, -0.07470717, -0.01385403, -0.22404844,
  -0.0266043 ,  0.16720998,  0.07604807, -0.48043856,  0.27398947,
  -0.19451259, -0.14505573, -0.29749015, -1.174899  , -0.18912567,
   0.22419687,  0.9392903 ,  0.24228042, -0.13158432, -0.45407757,
  -0.14482878,  0.5493851 ,  0.12222611, -0.7597946 ,  0.11976575,
  -0.9403068 ,  1.034866  , -0.00725865,  0.12926976, -0.49730355,
   0.67502046, -0.25079396,  0.11523671,  0.08847649],
 [ 0.3396804 , -0.24830674, -1.2379924 ,  0.16643128,  0.27586496,
   0.58150166,  0.607321  ,  0.94263583,  1.0952816 , -1.2786397 ,
   2.4810336 ,  1.0508854 ,  0.47442463,  0.33946046, -0.09624051,
  -0.04726255,  1.2398261 ,  1.2270133 , -0.5380113 ,  0.7711837 ,
  -0.7179314 , -0.28537825,  0.89464873,  0.36742267, -0.8738331 ,
   0.49388742,  1.1223552 ,  0.66361517,  0.4936239 ,  0.0596551 ,
  -0.52802706,  0.02997389, -0.06879518,  1.2772796 ,  1.0634763 ,
  -0.00102728,  0.49566287,  0.65874165,  0.7440924 , -0.04108439,
  -0.54161114,  0.1562788 ,  0.04021161,  1.7882128 , -0.5156103 ,
  -0.59937555, -0.22646149, -0.08034456, -0.9482809 ,  0.5770799 ,
   0.10416152, -0.02489491,  0.38594693,  0.59817106,  0.93817073,
   0.8740908 , -0.2694444 ,  0.11451284,  0.9822389 ,  0.68159944,
   0.41874614,  0.8965427 ,  0.4267358 , -0.00417569],
 [ 0.02618784, -0.41873878, -0.51725566,  1.2105802 ,  0.02420934,
   1.2229781 , -0.45290846,  0.8526715 , -0.68445116,  0.50978094,
  -1.2956239 ,  0.2175925 ,  0.16660619,  0.43662348,  0.5772803 ,
   1.2891601 ,  0.00717507,  0.4395381 ,  0.4669784 , -0.11423064,
   0.6263324 ,  0.28645682,  0.5181848 , -0.04606065, -4.2124233 ,
   0.48767173, -0.09753214,  0.86640775,  0.32138774,  0.00822688,
   0.04739791, -0.16522865,  0.74928844,  0.31531253, -1.3115423 ,
  -0.11617109,  0.47681418, -0.19470297, -0.05575079, -0.60010624,
  -0.6961607 ,  0.8085702 ,  0.52179414, -0.46312916, -0.22944555,
   1.1207469 , -1.7397276 ,  0.1478428 , -0.6443768 ,  0.54455626,
   0.05092349,  0.22962414, -2.2533607 ,  1.1123402 , -0.38943645,
   0.3926813 ,  0.42979404,  0.4140577 ,  0.340699  ,  0.01665604,
   0.05317157,  0.39065617, -0.0644186 ,  0.15364209],
 [-0.09058858, -0.4479842 , -1.6048961 ,  1.0192039 ,  0.19463138,
   0.9160243 , -0.91070473,  1.23415   ,  1.4926037 ,  0.29825035,
   1.437223  ,  1.1357076 ,  0.38585895, -0.5220424 ,  0.9617578 ,
   1.7943957 ,  0.04140341,  1.1408001 , -0.00718692,  0.09185579,
   0.4669628 ,  1.9034181 ,  0.7468566 , -0.77845144, -1.4970849 ,
   0.7377012 ,  1.6751294 ,  1.1718462 ,  0.5668392 ,  2.0546093 ,
  -0.39738148,  1.2905194 , -1.6431278 , -0.05774029,  1.0554929 ,
   0.66648704,  0.35818553, -0.43546024, -0.3314669 ,  0.65288955,
  -0.26646265,  1.661502  ,  2.0681205 ,  0.91178626, -1.0537838 ,
  -0.26550895,  0.33947155,  0.3731115 , -0.76078284,  0.02273103,
   0.5247299 , -0.40095863,  0.7239117 ,  0.11836257,  0.8559496 ,
   0.7272982 ,  0.06327782, -0.6687568 ,  0.1639723 ,  0.42777222,
   0.54314744,  0.00118166,  1.0846006 ,  1.070433  ],
 [ 0.19822425,  0.73979014, -0.19223963, -0.71275437,  0.2653088 ,
   0.13670418, -0.15637891, -0.7212953 ,  0.44642267,  0.35853428,
   0.3261567 , -0.47208872,  0.18263073, -0.65333736, -0.6281498 ,
  -0.87095696, -0.05142735, -1.9556316 ,  0.36594793,  0.2808761 ,
  -0.0215592 ,  0.11610026, -0.32431388, -0.42452234, -0.39820608,
  -0.0003529 , -0.39687738,  0.39380094,  0.41283873, -0.8025424 ,
  -0.20498312,  0.22197545,  1.5037237 ,  0.06483897,  0.4579054 ,
   0.7606223 , -0.23481801, -0.6192091 , -2.0388217 ,  0.07457405,
   0.29666027, -0.14852324, -0.40885076, -0.01663327,  0.5751604 ,
   0.5384648 ,  0.5371899 ,  0.15417561,  0.21979536,  0.27370483,
   0.20749298,  0.3105402 ,  0.16944201,  0.12126429,  0.47569424,
  -0.5851009 ,  0.22271906,  0.64193976,  0.92112476,  0.6754901 ,
  -1.5090208 ,  0.25996733,  0.3089139 , -0.45971912],
 [-0.49001452,  0.20653774, -1.5210809 , -0.019765  ,  0.73572487,
  -0.09271821, -0.37715465,  0.17186566,  1.2441648 ,  0.4075355 ,
   0.49408123, -0.01836646,  0.4878798 ,  0.02391889,  0.5424008 ,
   0.49734676,  0.06607871, -0.5153102 , -0.33775282, -0.22035372,
  -0.6724999 ,  0.6967095 ,  0.52608573, -0.6015833 , -0.09847211,
   0.03568131, -0.7379071 , -0.86482435,  0.70727503,  0.30989188,
   0.3994985 ,  0.18427968,  0.7776046 ,  0.55625904,  0.45323175,
  -1.0465086 , -1.0766501 , -0.16862817, -0.16023453, -0.413944  ,
  -0.79153407, -0.29110533,  0.04636007, -0.45198047, -1.0226244 ,
  -0.40323412,  0.10847916, -0.40964505, -1.6715981 , -0.18594487,
   0.87961745,  0.8360219 , -0.24545695,  0.9524955 , -0.80496347,
   1.0040374 ,  0.0946461 , -0.9119744 , -0.40685025,  0.6919647 ,
  -0.36427063, -0.13855368,  0.03391975,  0.21687552],
 [-0.2670738 , -1.0579975 , -0.9177993 ,  2.1163816 ,  1.0528314 ,
   0.69331443, -0.9176876 ,  0.15976825,  1.7276859 ,  1.5883749 ,
   2.7057726 ,  1.0035377 ,  1.0506659 ,  2.0576355 ,  0.8667396 ,
   1.0877414 , -0.01693225,  0.6592212 , -0.24642475,  0.7607191 ,
   0.11874876,  0.6893868 ,  1.7845573 , -0.09046473, -0.24200147,
   0.21068968,  0.87131214, -0.41040537,  1.0995072 , -0.54271203,
  -1.0191003 ,  1.0131485 ,  0.5468624 ,  1.0166081 ,  0.57073706,
  -0.36664286,  3.390646  , -0.04803886,  0.6461887 ,  1.0125512 ,
  -0.58837146, -0.4975976 ,  1.0383655 , -0.340458  , -1.0332791 ,
   0.47740194, -0.7671715 ,  1.0672741 , -0.5738441 ,  0.19288231,
   0.37251827,  1.8362077 ,  0.128732  ,  0.95098054,  0.2118203 ,
   1.0050153 , -0.53369063, -0.58433276,  0.4946061 ,  1.8213127 ,
   0.6065837 ,  0.07069918, -0.71274984,  1.2746887 ],
 [ 0.1300574 , -0.69634694, -0.696481  ,  0.7499864 ,  0.51025033,
   0.6991009 ,  0.84575963, -0.551443  ,  0.25670788, -0.27606565,
   0.48142362, -0.20524278,  0.04007909,  0.37121147,  0.5936932 ,
   0.24693356,  0.40563655, -0.27967262,  0.07529794,  0.5022215 ,
   0.47675505, -0.03135675,  0.7636707 ,  1.6772933 , -2.1368148 ,
  -0.00203894, -0.258178  , -0.6753502 ,  1.0679097 , -0.06383168,
   0.1137081 , -0.5432554 ,  0.3172333 ,  0.4874759 ,  1.6634735 ,
  -0.41873854, -0.46479234, -0.19690849,  0.1299023 ,  0.08449597,
   0.31437007,  0.60999256,  0.4785977 ,  0.04778104,  0.91547036,
   0.4525176 , -0.27275315,  0.88442945, -0.13185368,  1.3000766 ,
   0.094662  ,  0.35853872, -0.6786715 , -0.63926065,  2.5317378 ,
   0.46203914, -0.02277575, -0.2976753 , -0.15523568,  0.61518395,
  -0.00734385,  0.33091193,  0.05073703, -0.26333597],
 [-0.06710873,  1.3143079 , -0.6613792 ,  0.24524723,  0.10199156,
  -0.48349375, -0.16451259,  0.0890214 , -0.37604877, -0.5194154 ,
  -0.7108873 ,  0.05978601, -0.64325166, -0.8769156 ,  0.4301827 ,
   0.29604203,  0.7229649 , -0.10909413, -0.7194354 , -0.41688138,
   0.02951427,  0.25833872, -0.36726975, -0.853116  ,  0.08113704,
  -0.50557506,  0.5822136 ,  1.3421618 ,  0.01380436, -0.3723764 ,
  -0.4667433 , -0.67362   , -0.10106257,  0.66754663,  0.31547496,
  -0.8770755 , -1.5293891 , -0.34409723,  0.15195636,  0.42531797,
  -1.6334336 , -0.88876945,  0.2065608 ,  0.3859217 , -0.78641623,
  -0.74209636,  0.08564695, -0.49115372, -0.84464043, -0.04101082,
  -0.5371554 , -1.0630616 ,  0.17729451, -1.5243177 , -0.16550219,
  -0.38357365,  0.03208974, -0.7548928 ,  0.00513818, -0.32602143,
  -0.22038445,  0.3717086 , -1.0833864 , -1.6134764 ],
 [ 1.2231089 ,  0.15380964, -0.3811757 , -1.6703792 , -0.22427754,
  -0.5158584 ,  0.6694099 , -0.20929937,  1.623755  ,  2.7398698 ,
  -0.8224446 ,  0.15094148,  0.95845246,  1.0768609 ,  0.34316638,
  -0.941546  ,  0.4996698 , -0.44237244, -0.21849844,  0.30300424,
  -0.6770444 ,  0.607383  ,  1.0993096 ,  0.14379883, -1.478486  ,
  -0.11975305, -0.02802722, -1.5649934 ,  1.2058945 ,  1.5913075 ,
   0.46895817,  0.09993105,  0.32949615,  0.32733312,  0.36912137,
  -0.5013615 ,  1.496474  ,  1.0963032 ,  0.03350998,  0.45224777,
  -0.42111972, -0.55513847,  0.4337616 ,  0.48195335, -0.22194202,
   0.17535436, -0.2880834 ,  0.20378655, -0.19460653,  0.3977262 ,
   0.35328123,  1.1425931 ,  1.0550337 ,  0.75916564,  1.5031407 ,
   0.5564947 ,  0.5727583 , -0.90926623,  0.19648963,  0.62311935,
  -0.32957968,  0.22240713,  0.62638295,  1.0311836 ],
 [ 0.96898645,  0.78117406, -0.07313138,  1.0078543 ,  2.3891025 ,
   0.65413404, -0.16596451,  1.1363302 ,  0.877258  ,  2.1303089 ,
   6.009898  , -0.11778481,  1.6311903 , -2.5637167 , -1.6336347 ,
  -1.9171951 ,  0.43061736, -2.454447  ,  1.1475886 , -1.179775  ,
   0.6699089 , -0.8302617 ,  1.1901989 ,  0.02768881,  0.6045643 ,
  -1.1974436 ,  0.63275003,  0.20770282,  0.3685177 , -0.8452053 ,
  -1.9801708 ,  0.18399629,  1.423167  ,  0.10800403, -1.3942889 ,
  -1.1161703 ,  0.7104614 ,  0.57672846,  0.41941145,  0.4491316 ,
  -0.69489944,  2.0539927 ,  0.6763325 , -0.9663249 , -0.08659142,
  -0.2361844 ,  2.5282452 ,  0.71054685,  0.11791787,  0.42517558,
   0.03867377, -1.1341236 ,  7.0424886 ,  3.5799096 ,  0.9540615 ,
  -3.2648022 , -1.4206057 , -0.16112936, -0.32225642,  0.44041958,
  -0.24585634,  0.3901031 , -0.5226708 ,  0.29031506],
 [ 2.136424  , -0.07490091,  1.4844406 , -1.0245454 , -1.8521044 ,
  -0.45889938,  0.36040804,  1.0857433 , -1.3383024 ,  0.723482  ,
   1.66192   , -0.9300307 , -0.44262195, -2.4756725 , -0.527825  ,
   0.19579448, -0.571051  , -2.2509809 , -1.4102771 , -1.689717  ,
   0.02450793, -1.331618  , -1.7987393 , -0.26127842, -0.3234907 ,
  -1.9159547 , -0.11595179, -0.8423292 , -1.0887281 ,  0.78306043,
  -0.51011765, -2.523748  , -0.67996746, -0.44333842, -0.43305066,
  -0.71601754, -1.0242488 ,  0.03791955, -0.9689688 , -2.3221662 ,
   1.5332046 , -0.08630773, -0.41472226, -1.7638962 ,  0.57382315,
   0.52756184,  1.6963053 , -2.4197986 , -0.60623574, -0.7650245 ,
  -1.063467  ,  0.93186533,  0.77040666,  1.081335  ,  0.63767344,
  -0.918235  , -0.3107004 , -0.55423087, -1.9067403 , -0.9775968 ,
   0.5436174 , -0.6989414 , -0.15860447, -2.1201727 ],
 [ 0.50571924,  0.25570366, -0.4999974 ,  0.2373697 ,  0.07973257,
  -0.24055627,  0.72957546,  0.03999398,  0.40711132,  1.0409126 ,
   0.28694052, -0.10609412,  0.7864949 ,  0.42714214,  0.2188159 ,
   0.45459083,  0.605914  , -0.16530983,  0.5033059 ,  0.49502948,
   0.2632628 ,  0.5335261 , -0.02621434, -0.04761335, -1.3462981 ,
   0.3959418 , -0.3472401 , -0.27311453, -0.35400686,  0.90585995,
  -0.30804154,  0.6080172 , -0.05803764,  0.38760468,  0.4152072 ,
   0.23622318,  0.32417345, -0.28221667, -0.1488474 ,  0.09019478,
   0.9538728 ,  0.98159033,  0.4808053 ,  0.11071429,  0.44294184,
   0.10003298,  0.6964158 ,  0.06070219, -0.44864023,  0.21108328,
  -0.59100676,  0.5434017 , -0.6462208 ,  0.05686509,  0.01809907,
   0.58914095, -0.17805435,  0.6065943 ,  0.30602893, -0.19760427,
  -0.16470107,  0.35891345,  0.0336445 ,  0.03422923],
 [ 0.6072351 ,  0.02217538, -0.08197997,  1.8946128 ,  0.86250514,
   0.3800662 ,  0.53516906,  1.6856847 ,  0.37478572,  1.3367499 ,
   2.022865  ,  1.0658078 ,  0.84460276,  1.3813233 ,  1.1110386 ,
   0.34114274,  0.55188257,  1.9873042 ,  0.39749154,  0.36096188,
   1.3649917 ,  0.79883003,  0.8687381 , -0.3779751 ,  0.04767549,
   0.68734306,  0.9653317 ,  0.30685475,  0.73774034, -0.26224053,
   0.32654324,  0.0962005 , -0.37756076, -0.45478764,  1.3514619 ,
   0.20503119,  1.6275511 , -0.3640095 ,  0.19137423,  0.7307539 ,
  -0.64692897,  0.4818917 ,  1.0914838 ,  0.47764584, -0.5319736 ,
  -0.25627926,  0.6381245 ,  1.5262606 , -0.81691   ,  0.34818032,
   0.11697787,  0.9241493 ,  0.10319495,  0.9806445 , -0.1584848 ,
   1.6442648 ,  0.6962867 ,  0.27254385,  0.72162384,  1.6434873 ,
   1.0328394 ,  0.23747072,  0.62517446,  1.9938173 ],
 [-0.3848035 , -0.04971685,  0.11466315,  0.06997779,  0.47367477,
   0.4117628 ,  1.1678631 ,  0.12831876,  0.28767884,  0.6053094 ,
   0.02365866,  0.11521356,  0.61952287,  0.2672707 ,  0.53614   ,
  -0.04410882,  0.38104755, -0.29773358,  0.72198075,  0.8019163 ,
   0.21412557,  0.11518373,  1.1052921 ,  0.49981704, -1.8993883 ,
   0.22519241, -0.00988427,  0.35953736,  0.07802142, -0.771175  ,
  -0.30294165,  0.07413171,  0.45038125, -0.2844273 , -1.7999583 ,
   0.10736869,  0.7922413 , -0.569748  , -0.00608235,  0.06383716,
  -0.8845655 ,  0.6078228 ,  0.3115572 ,  0.3668531 ,  0.95602316,
  -0.07147753,  0.26717094,  0.33463392, -0.10727549, -0.08991394,
  -0.25873753,  0.06967349,  0.026669  , -0.0799686 , -0.9915633 ,
  -0.04306743,  0.18067297,  0.003744  ,  0.28409266,  0.2867545 ,
   0.14332476,  0.52789235,  0.06422957,  0.4186259 ],
 [-0.6001598 ,  0.1544494 , -0.22401015, -0.7665015 , -0.42245325,
  -1.2554387 , -0.30218   ,  1.0080554 ,  0.05486092, -0.8445367 ,
   0.05193113,  0.78692764,  0.2271379 ,  0.69230264, -1.1125536 ,
  -1.1388851 ,  1.0140733 , -0.4240299 , -0.51835656, -0.18931788,
  -0.89017105, -1.4655877 , -0.136042  ,  0.92808187, -1.4808564 ,
  -0.3932039 , -0.30163503,  0.8611613 ,  0.5322709 , -0.455791  ,
   0.6470642 , -0.9224825 ,  0.33046785,  0.4469428 ,  1.2155633 ,
   0.48051116, -0.0037025 ,  0.25422505,  0.14243196,  1.1028522 ,
   0.12399432,  0.40097767, -0.20505744, -1.3338358 ,  1.1400448 ,
   0.5441889 ,  0.29435802, -1.2393198 , -0.82824606,  0.34267992,
   0.23830236, -1.1158603 , -0.5853518 ,  0.67455155,  0.34712592,
   0.29647022, -0.8376902 , -0.5764574 ,  0.2314199 ,  0.13424596,
   1.3656385 ,  0.20801848, -0.9540796 , -0.69279116],
 [-0.81985235,  0.26437312, -1.0300548 ,  0.85992104, -0.52850664,
   0.4169506 ,  0.38578847, -0.1315796 ,  0.34268424,  0.41792226,
  -0.9621037 ,  0.6174235 ,  0.6435559 , -0.50919145,  0.17983018,
   0.42005175,  0.42780328,  0.6308403 ,  1.595243  ,  0.28276327,
   0.85662496,  0.6640145 ,  1.2187002 ,  0.13853487, -0.37956274,
   0.49201125,  0.6248562 ,  0.00255031,  1.3674238 , -0.64605963,
  -0.03894292,  0.8969071 , -0.0307101 ,  0.30551618,  1.8533521 ,
  -0.3571212 ,  0.81862235, -0.4064982 ,  0.5027952 ,  0.12841873,
  -1.2045088 ,  1.1457528 ,  0.96151453,  0.08263791,  0.25165042,
   0.48881721,  0.95528084,  0.29170856, -0.40940604,  1.1108897 ,
   0.14419329,  0.04603579,  0.23466152,  1.5317627 , -0.3198906 ,
   1.2190466 , -0.17074242,  0.49137244,  0.47260308, -0.0929376 ,
   0.6177935 ,  0.8100413 ,  1.2460766 , -0.24930684],
 [ 0.6162615 , -0.64541745, -0.53970367, -0.12604494,  0.758616  ,
   0.6354467 , -0.06602826,  0.89776385, -0.75192165,  0.40081504,
  -0.05432574,  0.12764353, -0.04124793,  2.115405  , -1.4348556 ,
  -0.83145577,  0.19740126,  0.42137772, -1.2586087 , -0.78892225,
  -0.08261509,  0.5427381 ,  0.6514124 , -0.5045546 , -0.9771802 ,
  -0.292914  ,  0.09648218, -0.43843573,  0.9690916 , -0.37934667,
  -0.97328615,  0.8485275 ,  1.7289127 ,  0.54156786, -0.9288762 ,
  -0.81668603, -1.7553017 ,  0.7529359 , -0.43809682, -0.9921343 ,
  -0.00206265, -1.9440984 , -0.00374438, -0.5785824 ,  0.08142398,
  -0.08780497, -0.08410567,  0.627119  ,  0.15741348, -0.9683988 ,
  -0.08216801,  1.0098506 ,  0.36803964,  0.6021136 , -0.22816025,
  -1.1339468 ,  0.56109565, -0.48266804,  0.97314525,  1.4678795 ,
  -0.6184849 ,  1.3057528 ,  0.09297246,  0.39773935],
 [ 0.68763417, -0.5721882 , -3.0410092 ,  0.88220143,  0.41966128,
   0.55507886,  2.3902512 ,  0.02317627,  0.6191465 ,  1.8382902 ,
  -0.7820694 ,  1.0937843 ,  0.23856083,  1.1557428 ,  0.8403301 ,
   1.3726597 ,  1.1482329 ,  1.6441542 ,  0.80546016,  0.9361991 ,
   1.0707452 ,  1.5078577 ,  1.3561898 , -0.7249931 , -0.9634314 ,
  -0.12154324,  1.0511698 ,  2.1717105 , -0.509404  , -1.2647804 ,
  -0.9914355 ,  2.6283512 , -0.7533384 , -0.04888048,  1.7388816 ,
   1.919719  , -0.50946754,  0.18516788,  1.3204739 ,  2.014142  ,
  -0.34305248,  2.6297643 ,  1.3653028 ,  0.5910905 , -1.0404955 ,
   0.16983429,  1.264121  ,  1.4873788 , -0.57801056,  1.2567604 ,
   0.19616868,  1.2590755 , -0.28503558, -1.2675062 ,  0.38185742,
   0.7570992 ,  0.98428375,  2.1577325 ,  2.097542  ,  0.3486359 ,
   0.35634002,  0.6558344 , -1.1033411 ,  1.91173   ],
 [-0.30935767, -0.83541787,  0.30256936,  0.18289551,  0.972404  ,
   0.01895754,  0.19986495, -0.08167762,  0.894079  ,  0.81391805,
  -0.14093365,  0.61817044,  0.30902204, -0.07907117,  0.6617849 ,
   0.15039094,  0.7465595 ,  1.19143   , -0.9937509 , -0.23797454,
  -0.5718194 , -0.85867226, -0.43392065, -0.43571147, -0.02479113,
  -0.5030564 ,  0.6551832 , -1.3127185 ,  0.8583651 , -0.27416432,
   0.36822975, -0.36957672,  0.50167125,  0.8230477 ,  0.03290981,
  -0.21959642,  1.1188465 ,  0.70443165,  0.39335752,  2.0962563 ,
  -0.1970749 , -1.7756696 ,  0.04145357,  0.4403426 , -1.0654932 ,
   0.00644558, -0.66410863,  0.3863078 , -0.9348444 , -0.07130038,
   0.04335807,  1.1898079 , -0.4632943 ,  0.56624705,  0.25951037,
   0.6382314 , -0.03299949, -1.0223695 , -0.08029076, -0.84260744,
   0.5011682 , -0.09978538, -0.4337748 ,  0.78538907],
 [ 0.42923462,  0.26592267,  0.06492516, -0.60509473,  1.8719025 ,
  -0.15362768,  0.43616575, -0.58729684, -0.09632006,  1.1761584 ,
   0.35102743,  0.00127093,  0.7113927 ,  0.8831336 , -0.06072652,
   0.75972915,  0.37241933, -0.865845  , -0.15861759,  0.6886319 ,
   0.59536946, -0.88670397,  1.0184315 , -0.75997555,  1.6432433 ,
   0.3601889 , -1.2795002 ,  0.6240881 , -0.53671   ,  1.1480751 ,
  -0.6215227 , -0.59509856,  0.5603638 ,  0.92963463,  0.51633   ,
  -0.28721353,  0.4230187 ,  0.6074897 ,  0.19434436,  0.87233895,
  -0.49777445, -0.7644237 , -0.6701986 , -0.8600945 , -0.6163023 ,
  -0.38335627, -0.9534397 ,  0.3783889 , -0.95305645,  0.60167515,
  -0.545506  ,  1.9985001 ,  1.4801852 ,  1.1005218 , -0.21058224,
  -0.6781074 ,  0.4485296 ,  0.52160805, -0.18915533,  0.14477898,
   1.2349042 , -1.4547921 ,  0.8032941 , -0.11349461],
 [ 1.3098042 ,  0.13479266, -0.17676301, -0.47987756,  0.12602894,
   0.48065013,  0.0450599 , -2.9424202 , -0.59974235, -0.6563584 ,
   1.527823  , -0.5140024 , -0.3443915 , -1.6073883 , -0.02581206,
   0.26637053,  1.1546304 , -0.21620914,  0.22165139, -0.02723525,
  -0.76370627, -0.1267625 , -0.5870631 , -0.7832384 , -0.01875409,
  -0.06790501,  0.00697535, -0.3920465 ,  0.33601534, -0.25102717,
   0.32594267, -0.666109  , -0.32324666,  0.16908723,  0.39635375,
   1.2250862 , -0.19541723, -1.1451447 ,  1.1180215 , -3.8677258 ,
   0.30270714,  0.04845034, -1.0234991 , -0.5326175 ,  1.5652462 ,
   0.8187812 , -3.0628202 , -1.8707768 ,  0.59551924, -0.48269743,
   0.03406416, -1.6832876 ,  1.4166718 ,  1.1380591 ,  1.711395  ,
   0.04894767, -0.33918303,  1.9907157 , -0.24744666,  0.73976773,
   1.1512282 , -0.358423  ,  3.4102259 , -0.3340091 ],
 [-0.83177555, -0.04557897, -1.2192236 ,  0.9761811 , -0.172366  ,
   0.9754174 ,  1.8018101 ,  1.0904505 ,  1.0529357 , -0.06167357,
   1.0485834 ,  1.618672  ,  0.2139654 ,  1.1691084 ,  1.110008  ,
   1.505535  ,  0.34523562, -0.01872745,  0.76350343,  0.54775524,
   1.2709948 ,  1.4708236 ,  1.3032386 ,  0.52418333, -0.7701008 ,
   0.5622649 ,  1.0762112 ,  0.16190024,  1.3722943 ,  0.21606992,
  -0.05701618, -0.10956612,  0.8175245 ,  0.96825236,  2.058312  ,
   0.00395069,  0.02784722,  0.5296854 ,  0.7369284 ,  0.23801114,
   0.16305156,  0.7802917 ,  0.9458683 , -0.1412082 ,  0.44571936,
   0.6078586 ,  0.44200835,  0.38255706,  0.45847237,  0.96948254,
   0.35846943, -0.04874871,  0.99579763,  0.49477375,  1.1774284 ,
   1.0409153 ,  0.36480483,  0.81814003,  0.57445544,  1.4372308 ,
   0.49814326,  0.30027297,  0.22037426,  0.4567034 ],
 [-0.43517172, -0.8181257 ,  0.91275454, -0.665781  , -0.7974504 ,
  -0.80284923,  1.1888556 , -0.5160825 , -1.1622932 , -0.01222571,
  -2.019349  ,  0.01904463, -1.2752972 ,  0.77200544, -0.882076  ,
  -0.87520194, -0.3208979 , -0.40592164, -0.5180816 , -0.53234   ,
  -0.23644668, -0.62454545, -0.5167614 ,  0.153336  ,  0.21374474,
  -0.6248521 , -0.5965252 , -0.9780817 , -0.21714737, -0.4753746 ,
  -0.8817093 , -0.4616652 , -0.09671735, -0.4229118 , -1.1938369 ,
  -0.60211223, -0.8570599 ,  0.35089275, -0.66182244, -1.0036861 ,
  -1.07262   , -2.5819023 , -0.45822144, -1.3419703 ,  0.40154332,
  -0.4167491 , -0.53943527, -0.6529742 , -0.5604449 , -0.5280932 ,
  -1.0702498 ,  0.811233  ,  1.1614478 , -1.1906929 ,  0.5682145 ,
  -0.33804694, -0.28862363, -0.45379695, -0.5114664 , -0.17750518,
   1.0350748 , -0.3212305 , -0.53086925, -0.07241168],
 [-1.1260515 , -0.348597  , -1.2583979 ,  1.2641536 ,  0.27768496,
   1.0714927 ,  0.38662747,  0.64397275,  0.8648864 ,  0.8474577 ,
   0.9998033 ,  1.1313481 ,  0.6486522 ,  1.6157756 ,  0.6446482 ,
   0.38104337,  0.16055411,  1.3165168 ,  0.25988045,  0.32927322,
  -0.22073597,  1.1396502 ,  0.79764456, -0.40797296, -0.01606499,
   0.21272546,  0.5056902 ,  0.20131043,  1.0623806 ,  0.7873563 ,
   0.08682334,  1.4729486 , -0.8226238 ,  0.48864946, -0.6709741 ,
  -0.14476264,  0.7424972 ,  0.32800293,  0.5606732 ,  1.5495251 ,
  -1.0567571 ,  0.75549144,  0.65893614,  0.2443114 , -1.1134807 ,
  -0.6045309 ,  0.34220478,  1.4868463 , -0.7103696 ,  0.16763309,
   0.23199119,  1.2376908 ,  0.4460902 ,  1.3007747 ,  0.24724053,
   1.2202362 , -0.01388407, -0.5602359 ,  1.3584244 ,  0.7465688 ,
   1.3002733 ,  0.31910777,  0.06619643,  0.3663811 ],
 [-0.06260625, -1.0585729 ,  0.6514267 , -0.0863595 , -0.9009488 ,
  -1.1667728 , -0.01954388,  0.5364528 , -3.3860695 ,  0.11853363,
   1.891163  , -0.137061  , -0.01519881,  0.1537416 ,  0.22056258,
  -0.17354783,  0.25043878,  0.4085175 ,  0.6042787 , -0.05412986,
  -0.05086955, -0.21901791,  0.18975078,  0.81918985,  0.386644  ,
   0.00863593,  0.5643006 , -1.2259669 , -0.218669  , -0.8018949 ,
  -0.38150403,  0.11222333,  0.18479826,  0.13809133, -0.41913542,
   0.6362218 ,  0.6647579 ,  0.3711558 , -0.90378183, -0.07358765,
   2.014324  , -7.7996583 , -0.05836308,  0.27704924, -0.46455446,
   0.6589946 , -1.4104253 ,  0.24042511,  1.5910205 ,  0.6553273 ,
  -1.4725132 ,  0.5224184 ,  0.35408884, -1.1236933 , -0.6796086 ,
  -0.5320895 , -0.52253073,  0.9335967 ,  0.75463176,  0.47544625,
   0.6080104 ,  0.76915675, -0.6467754 , -0.2922643 ],
 [-0.51673114, -0.1614707 ,  1.1845181 ,  0.11687932, -0.5876106 ,
  -0.28939196,  0.5279343 ,  0.11530153, -0.28543782, -1.3513296 ,
  -0.86235255,  0.4427696 , -0.78633976, -1.0695323 , -1.4409965 ,
   0.17754927,  0.46482566,  0.05895732,  0.13612884, -1.0535063 ,
  -1.0902818 , -0.71334136,  0.2121572 , -0.19263461,  1.3643707 ,
  -0.63260823, -0.38115007, -0.07829165, -2.270298  ,  1.6806178 ,
   0.7507732 , -0.43994674, -0.13404539,  0.19117694, -1.016891  ,
  -0.40227506, -0.37494504,  0.3902855 , -1.6379745 , -1.5935574 ,
   0.06028125, -0.30045205,  0.07709616, -0.53550804,  0.08891474,
  -1.0441926 , -0.633722  , -1.1153424 ,  0.83102006, -0.77687067,
  -0.62323654,  0.39459047, -1.2172718 , -0.8345477 ,  0.2822405 ,
  -0.56231576,  0.2594733 , -0.04170179,  0.48334453, -0.9030083 ,
   0.2378279 , -0.27961951,  0.8258852 , -0.7680002 ],
 [ 0.7575861 ,  0.46126878, -1.4555004 ,  1.4495158 , -0.28398904,
   0.43442786,  1.0786679 , -0.84122926,  0.34403703, -1.0525293 ,
  -0.76758564,  0.71336204, -0.07546329,  0.9669857 ,  0.45292497,
   0.9548043 ,  0.1448596 ,  0.33718687,  0.96620196,  0.8783637 ,
   1.4289175 ,  0.28238988,  0.532935  ,  0.8565624 , -0.8867165 ,
   0.77616304, -0.3355184 ,  2.0111444 ,  1.270253  ,  0.75737995,
   0.804167  , -0.08312679,  0.34367475,  0.11491521,  1.4433424 ,
   1.1080955 ,  2.7280886 , -0.46421698, -0.00283067,  1.3894883 ,
  -0.01938539,  0.88843256,  1.1553282 , -0.6613369 ,  0.59795785,
  -0.28446862, -0.06511629,  0.6199674 ,  0.5254381 ,  1.1804976 ,
  -0.2988304 ,  0.1292196 ,  0.80524635,  1.8338482 ,  1.352641  ,
   0.13268922, -0.7057606 ,  0.57703394,  0.19270442,  0.6853082 ,
   1.360269  ,  1.5794414 , -0.01314147, -0.23226666],
 [-0.23585065, -0.91891885, -0.34326267,  0.20310266,  0.7105922 ,
   0.02119658, -0.17181946,  0.55272603, -0.17816268, -0.11410169,
   0.82404536,  1.1098272 , -0.28021014,  0.58178353,  0.3702633 ,
   0.42269212,  0.21392056, -1.5755653 , -0.45702374, -0.3311108 ,
   0.26049936, -0.17162332,  0.38563693, -0.22258122, -0.19595562,
  -0.34720492,  0.26480842, -0.3924249 ,  0.00460079,  0.33061904,
  -1.0319858 ,  0.2212937 ,  0.75166315, -0.22435471, -0.8069323 ,
  -1.1489851 ,  0.32629046, -0.71524924, -0.23464096,  0.923152  ,
  -0.41501614,  0.460896  ,  0.5687042 , -0.21504745, -0.72159207,
  -0.55046254, -1.1861043 , -0.07013366, -0.80530024,  0.11627475,
  -0.10480063, -0.07029345,  0.49324116, -0.31922913, -0.31700698,
  -0.86287874,  0.7531867 ,  0.5162726 ,  1.3209273 ,  0.24916928,
   0.20001054,  0.39066908,  0.15671216,  0.8079409 ],
 [-0.5169646 ,  0.1272327 , -1.3501145 ,  2.034534  , -0.35997188,
  -0.96864504,  0.21096686,  2.2806878 ,  0.6112442 ,  0.09275037,
  -0.46874985,  0.47070816,  0.6441224 ,  1.4615752 ,  1.4246024 ,
   1.0879176 ,  0.67774445,  1.1896064 ,  0.8389791 , -0.00789479,
   0.997583  ,  1.1928607 ,  1.5722804 ,  0.00712141, -1.329239  ,
   0.32308558,  0.7994355 ,  0.6402924 , -0.482242  ,  1.381301  ,
  -0.20345104,  0.8472962 , -1.2844857 ,  0.49274725, -0.30948544,
   0.39907628,  2.0013046 , -0.36572143, -0.3481739 , -0.0249366 ,
  -0.5833433 ,  1.1520617 ,  1.0095644 ,  0.76318824, -0.06821387,
  -0.62801075,  0.9264144 ,  0.5817106 , -0.34471804,  0.44472116,
   0.06812362, -0.750523  , -1.344474  ,  1.118654  ,  0.4183095 ,
   2.2968726 ,  0.16619362,  0.59031093,  1.0068293 ,  1.3923401 ,
  -0.17282516,  0.8995362 ,  0.18223934,  0.75069594],
 [ 0.76499796, -0.6061953 , -0.96630853,  0.18566482, -0.21734445,
  -0.6428915 ,  0.34942764,  1.7069944 ,  0.28039142, -1.154435  ,
  -0.6098721 , -0.05425224,  0.29100978,  0.42966902,  0.47434577,
  -0.6557675 , -0.10050457,  1.7457467 ,  0.04254321,  0.33271274,
   0.18297851,  0.6810997 , -0.05020916,  2.9617286 , -0.12155583,
  -0.35758802,  1.3202941 ,  0.17549327,  0.1036545 , -0.7265869 ,
  -0.45763525, -0.0366535 ,  0.1876432 ,  0.8504063 , -0.17881791,
  -0.08158757,  1.1019553 ,  1.6575412 , -0.5961653 , -0.39013144,
   0.60629237, -0.28568795, -0.6350128 ,  0.6372519 , -2.0548012 ,
   0.3064713 ,  2.0320935 , -1.0758594 ,  0.03815871, -0.81043863,
   0.09786855, -0.45730963, -0.32441124, -0.63625085,  0.79754573,
   0.30944517, -0.0633973 ,  1.0203546 , -0.264827  , -0.26722816,
   1.1885077 ,  0.45301768,  0.00346277,  0.3917679 ],
 [-1.1485796 , -0.25598043,  0.13642682, -0.4743772 , -0.13108191,
  -0.06587326, -0.21746317, -0.4386994 , -0.67015433, -0.22766072,
  -0.28075007, -0.25030652,  0.19478188, -0.23956627, -0.35479233,
  -0.96493036,  0.6083871 , -0.49047408, -0.46337622, -1.1752088 ,
  -1.1779234 , -0.55558634,  0.218685  , -0.23404399,  0.18174559,
  -0.7401024 , -0.1476553 , -1.3014637 , -0.3928006 ,  0.12760499,
  -0.11444019, -0.57612026,  0.51981515, -0.06059959, -0.3945416 ,
  -0.4944841 ,  0.5146807 ,  0.08522542, -1.47896   , -0.56158155,
  -0.38642764, -0.8382069 , -0.30587187, -0.24483761, -0.13623925,
  -0.1669822 , -0.4054391 , -0.27700233, -0.5853571 , -0.15625982,
  -0.23109785,  0.35194045, -0.51607656, -0.31881455, -0.36479184,
  -0.08129366, -0.5876236 , -0.7598841 ,  0.07456324,  0.16222641,
  -0.3269253 , -0.7245404 , -0.22005609,  0.23809688],
 [ 0.5761077 , -0.0155294 , -0.15665662,  1.0428263 ,  0.34539652,
   0.16064648,  0.09779755,  0.63819426,  0.5229526 ,  0.8668268 ,
   2.9610336 ,  0.28984848,  0.9747944 ,  1.5756958 ,  0.8644761 ,
   0.5736009 ,  0.49891755,  0.9910222 ,  0.49637383,  0.79686576,
   0.2518733 ,  0.45194885, -0.08430723, -0.19275919,  0.10332333,
   0.5319983 , -0.6517449 , -0.5186427 , -0.01325426,  0.5992331 ,
   0.04711502,  0.17277926, -0.45233595,  0.63181174,  0.375792  ,
  -0.2675504 ,  1.3129832 , -0.16895394,  0.71391857,  1.560755  ,
  -0.45113155, -0.14854631,  0.6442715 , -0.5188294 , -0.36227244,
   0.04328441,  0.9184315 ,  0.27520156, -0.4462015 ,  0.20600398,
   0.22231457,  0.46650586,  0.599873  ,  1.589515  ,  0.44381893,
  -0.5390909 ,  0.27987316, -1.3553889 ,  0.03103871, -0.01443743,
   0.3558101 ,  0.07012878, -0.92229694,  0.72080755],
 [-0.45033902,  0.18344185, -0.36517608,  1.1972764 ,  1.0086397 ,
   1.228977  , -1.0127414 , -0.21758609,  1.2654434 ,  1.1178857 ,
   0.21164194,  1.4268011 ,  0.91357833,  1.1107718 ,  1.477125  ,
   1.1510149 ,  0.07763547,  1.4386252 , -0.3140241 ,  0.3896948 ,
   0.54880804,  0.59308463,  0.7944294 , -0.5029972 ,  0.0793144 ,
   0.6731261 ,  0.93102247,  0.8819578 ,  1.5137541 ,  0.96794075,
   0.36390376,  0.8401754 , -0.93607944, -0.15209699, -0.02223817,
   1.2554796 ,  0.36061063, -0.39103523,  0.05651012,  1.1707175 ,
  -0.6854133 ,  1.5294819 ,  1.4478151 ,  0.8098507 , -0.53364176,
  -0.83977807,  0.11438292,  0.44861075, -1.0970175 , -0.01359705,
   0.6918709 ,  0.34228134,  0.5624439 ,  0.7846324 , -0.17995887,
  -0.39600238,  0.8063603 , -0.98531395,  0.99430037,  0.7303617 ,
   1.8890237 ,  0.46849856,  0.52740866,  1.5563792 ],
 [ 0.02616634,  0.4757061 , -1.329961  ,  1.1833588 ,  2.1201682 ,
   0.4579877 ,  0.04876213,  1.1578496 ,  1.9206378 ,  0.9186245 ,
   0.75048554, -0.0428645 ,  0.95752597,  0.30286252, -0.43277147,
   0.78794104,  1.0138367 , -1.1760689 , -0.11785746, -0.31381157,
   0.01676213, -0.75364244,  0.6381117 ,  0.45856473, -0.457788  ,
  -0.3578636 ,  0.84358114,  0.12743245,  0.3124623 ,  0.16933633,
   0.39908844,  0.47500107, -0.37460214,  0.20696145,  0.6947089 ,
  -0.3105045 , -0.45836386, -0.43708804,  1.0772744 , -0.4092453 ,
   0.7770741 ,  1.5420996 ,  0.86138654,  0.6520297 ,  1.0060997 ,
  -0.6512121 ,  0.12771091,  0.01434273, -0.3289215 ,  0.2494709 ,
  -0.2950305 , -0.02344291,  1.2959247 ,  2.265968  ,  0.6853456 ,
   0.5869111 ,  0.04416198,  0.44610623,  0.3944466 , -1.0060436 ,
   0.6967983 , -0.30665013, -0.52479666,  0.72736394],
 [-1.593683  , -2.12061   , -0.32762524,  1.0216429 ,  0.649349  ,
  -0.01688968, -0.43491313,  0.15272507, -0.5209635 , -0.5902286 ,
   2.0436223 ,  0.18965486, -0.4960395 ,  2.2605875 , -0.21368185,
  -0.18741533, -0.84233165, -0.02959369,  1.3647474 ,  0.45500028,
   0.51575625,  0.63613   , -0.02359869,  0.3147978 , -0.1299918 ,
   0.32770303, -0.5682951 ,  0.8894047 ,  0.557423  , -0.94600314,
   1.4932904 ,  0.83023596,  0.5374967 , -0.37080362,  0.8026938 ,
   0.15244573,  1.36161   , -1.1729494 ,  1.517776  ,  1.4640025 ,
  -0.06169002,  2.2872515 , -0.2093264 , -0.07955394,  0.04006092,
   0.4105179 ,  0.673684  ,  0.7682912 ,  0.31167147, -0.22853173,
   0.01782157,  1.145508  , -0.24696532,  0.2162929 , -0.9000515 ,
  -0.25770953,  0.64096576,  0.12989934, -0.50391996,  0.28191486,
  -0.17892666,  0.31298816,  0.15864392,  1.420491  ],
 [ 0.22928493,  0.6993232 , -0.6848745 , -0.00586874,  0.6005871 ,
   0.49104586,  0.30469674,  0.47667116,  0.23242992,  0.56586885,
   1.402574  , -0.19214928, -0.60295033,  1.1569054 ,  0.34470555,
   0.63687485, -0.8036938 , -0.1927757 ,  0.02696895,  0.3372142 ,
   0.01288231,  0.22149405, -0.03303113,  0.1639862 ,  2.194718  ,
   0.53945315,  0.02983128,  0.16995874,  0.48995   ,  0.00380017,
   0.0277949 , -0.11113469, -0.14746101, -0.02127663, -0.6323928 ,
   0.00036138, -0.47140065,  1.2860638 ,  0.3262738 ,  0.7871291 ,
   0.18554065,  0.74617004,  0.2563502 , -0.2658017 , -0.9357762 ,
   0.06051844,  0.9129556 ,  0.60442585,  0.02816275,  0.00507839,
   0.07684927, -0.32072398, -0.7784732 ,  0.11461411,  0.2269625 ,
   0.04964324,  0.4210513 ,  0.15255833,  0.07693405,  0.23916233,
   0.66342175, -0.07741494, -0.21072167, -0.16708507],
 [-0.09372074,  0.6307599 , -0.17850423, -1.3230023 , -0.21760356,
  -1.4200599 ,  0.46663508, -0.11517189,  2.5303607 , -0.89484054,
   1.4815444 , -0.3478035 ,  0.6684549 ,  1.366333  , -1.317425  ,
  -0.6057439 , -0.5328063 , -0.07916421, -0.64670867, -0.9808033 ,
  -0.5980619 , -0.91121846, -0.1066739 ,  0.37605777, -0.75798446,
  -0.5936543 ,  0.35244867, -1.939537  ,  0.23850933, -0.22653717,
  -0.27750844, -0.00650688, -0.02216107, -0.3148624 , -1.7300127 ,
  -0.47090074,  0.3006235 , -0.21191715, -1.916161  , -0.38295424,
  -0.0204233 ,  1.5950236 , -1.5006914 , -0.074932  , -0.33089876,
  -0.9289839 ,  1.0336267 , -0.01629622, -0.37954015,  0.08657526,
   0.11113798, -3.158807  ,  4.048515  ,  0.4993267 ,  0.7438421 ,
  -0.37346837,  0.50872654, -0.3227115 ,  0.05733104,  0.8687143 ,
  -0.53580594, -1.4430287 ,  0.10591135,  0.28514418],
 [ 0.09345304,  0.27734697, -2.1033106 ,  1.8324916 ,  0.70516384,
   0.32609963,  2.0488393 ,  2.2670107 ,  1.4898936 ,  3.8475857 ,
   1.0983921 ,  2.2831256 ,  1.4548452 ,  3.3749835 ,  1.432428  ,
   3.1236334 ,  0.6515203 ,  2.109657  , -0.2862295 ,  1.3983208 ,
   0.96638435,  1.5043801 ,  1.3413669 , -0.56419224, -0.02544798,
   0.7685333 ,  1.4164813 ,  0.4424189 ,  1.1960645 ,  0.68109924,
   0.08545577,  1.9691308 , -0.47574723, -0.06968032, -0.35744342,
  -0.58674675,  5.0577173 ,  0.05673991,  1.4986342 ,  1.9793686 ,
  -1.1184673 ,  0.63443255,  1.806231  ,  1.6572572 , -0.72992766,
   0.38876495,  0.6688608 ,  1.6700512 , -1.1004568 ,  0.87879384,
   0.9020263 ,  1.7155383 , -1.0344884 ,  0.87112474,  0.97785836,
   2.5919135 ,  1.173955  ,  0.5453993 ,  1.9702291 ,  1.7214134 ,
   0.6523697 ,  1.2160958 ,  1.342338  ,  2.0817082 ],
 [-0.64073104, -0.4759465 , -0.7596571 ,  0.8422437 ,  0.39127997,
   0.33602646,  0.6548242 , -0.38673186,  0.7807912 ,  0.4444823 ,
   0.2501557 , -0.01427281,  0.14288563,  0.608084  ,  0.3301345 ,
   0.97796535,  0.09136216, -0.12551454, -1.1058651 , -0.05523844,
   0.41448444,  0.2440711 ,  0.44547254, -0.23745903, -2.3532588 ,
   0.15793678,  0.16814747,  0.02751558,  0.28892297, -0.84548336,
   0.3712924 ,  0.72092307,  0.54016584,  0.24020734, -0.06916111,
  -0.87106043, -0.12322759,  0.02676614,  0.06055087, -0.408153  ,
  -1.5264732 ,  0.69814235,  0.11607443,  0.09756183,  0.10788701,
  -0.3573996 , -0.48337376,  0.14144059,  0.22934122,  0.29225105,
   0.21776874, -0.06767229, -0.6052015 , -0.38287595,  0.34390545,
  -0.8228177 ,  1.089799  , -0.3605512 ,  0.9141014 ,  0.18407075,
  -0.04252306, -0.89423084,  0.20440128,  0.5793001 ],
 [-0.2775946 ,  0.39262232, -0.60878   ,  1.1413356 ,  0.5249237 ,
  -0.5139464 , -0.43349007,  1.1661961 ,  1.1575941 ,  0.6383415 ,
   0.58503497,  0.56678206,  0.26683947,  0.87459743,  0.51804066,
   0.62799317,  0.13055131, -0.26814616,  0.15797584,  0.6361047 ,
   0.8978216 ,  0.42318746,  1.4595472 ,  0.29454362,  1.5825903 ,
   0.28883386, -0.12595813, -0.27409282,  0.41979176,  0.5610758 ,
  -0.00839841,  0.906507  ,  0.38628405,  0.6827838 ,  1.7013342 ,
   1.3347887 ,  2.5571263 ,  0.3893211 , -0.5310547 ,  0.37394097,
   0.23993872,  0.72381115,  0.84692645, -0.7736529 ,  0.19176316,
   0.45605317, -0.29863322,  1.3148141 ,  0.24000932,  0.45319867,
   0.28467095, -0.17190422,  0.21541844,  1.2248533 ,  0.67027265,
  -0.5916081 , -0.7078611 ,  0.606239  ,  0.00985214,  0.578743  ,
  -0.34501538,  0.09345526,  0.20035978,  1.195263  ],
 [-0.11487195, -0.18362628,  0.12286772, -0.27353358, -0.5123415 ,
  -0.40051946, -0.2831705 ,  0.22940083, -0.18432605, -0.03481506,
  -0.15693802, -0.5784559 , -0.60276973, -0.4697014 , -0.32455063,
  -0.46892205,  0.12960322, -0.05733234, -0.6632557 , -1.1029795 ,
   0.22888802, -0.46961373, -0.43634784, -0.45229208,  0.21164197,
  -0.5268143 ,  0.31163386,  0.00960911, -0.5804037 , -0.2810904 ,
  -0.59438574, -0.29346102, -0.8441964 , -0.25296155, -0.12195432,
  -0.17705059, -0.6868648 , -0.0091777 , -1.1746099 , -0.2915977 ,
  -0.01317125, -0.02060179, -0.46622178,  0.52752143, -0.04824445,
  -0.81502736, -0.07687207, -0.48819113, -0.19461426, -0.17027673,
  -0.21500932, -1.3397268 , -0.18056123,  0.53136235, -0.11079112,
  -0.8861584 , -0.3636919 ,  0.09536799, -0.23897623, -0.3842664 ,
  -0.6140299 , -1.1309265 , -0.6906667 , -0.4264211 ],
 [-0.6334661 , -0.6883265 , -0.0102769 , -0.08743338, -0.7159349 ,
  -0.5989538 , -0.06959818, -0.87567323, -0.21234949, -0.14355032,
  -0.2531731 , -0.47554815,  0.05030549, -0.6834375 , -0.4442513 ,
  -0.42664427, -0.13333242, -0.53774256,  0.00633011, -0.41909158,
  -0.54084814, -1.0952424 ,  0.1000071 , -0.29731178,  0.1543326 ,
  -0.62653136, -0.73173165, -1.0037196 , -0.53377575,  0.00226391,
  -0.5487239 ,  0.29860517, -0.65472955, -0.7600509 , -0.21417797,
  -0.545929  ,  0.06552443, -0.69717973,  0.02187637, -0.49769387,
  -0.22957513,  0.14575459, -0.40019593, -0.73923314, -0.42233634,
  -0.17803234, -0.55042243, -0.39498746, -0.5647795 ,  0.5997088 ,
  -0.19166379, -0.01245408, -0.41196516,  0.33459315, -0.44199163,
  -0.50882334, -0.44453079, -0.2550022 , -0.5453714 , -0.24156617,
   0.04287576, -1.2108706 , -1.0403311 , -0.16218337],
 [ 0.16013683, -0.12895636, -1.770923  ,  2.29329   ,  0.8337886 ,
  -0.12033655, -0.73096174,  1.6707674 ,  1.1733457 ,  1.3182126 ,
   0.796388  , -0.2675213 ,  1.0005438 , -0.26182476,  1.4027234 ,
   1.3564811 ,  0.8046209 , -0.0359242 ,  0.02618147,  0.04336787,
   1.1003541 ,  0.7267269 ,  0.9138348 ,  0.88017535, -0.85718405,
   1.9988114 , -0.31746745, -0.17080174,  0.7765236 ,  1.595639  ,
  -0.3016853 ,  0.7955933 , -0.23387621,  0.03270335,  0.1156227 ,
   0.4420856 , -0.10877433, -1.2031112 , -0.9661278 ,  0.6048165 ,
  -0.34808323,  1.5804825 ,  1.3322368 , -0.7799764 ,  0.5296294 ,
   0.21788588,  0.6005264 ,  1.021662  ,  0.13583888,  0.9391907 ,
   0.12517554,  0.4855195 , -0.3234039 ,  0.6556862 ,  1.9146612 ,
   0.65036184, -0.18589206, -0.02289335,  0.3005348 ,  0.7723748 ,
  -1.6906401 ,  0.8639055 ,  0.08520718,  1.1752526 ],
 [ 1.4428102 , -0.89220613,  1.801816  , -0.53019273,  0.02386494,
  -0.63798213, -0.24717444,  0.7941441 , -1.8789612 , -0.0038463 ,
   1.1826094 ,  0.01214596, -1.4972583 ,  0.739331  , -1.8939722 ,
  -1.3036219 ,  0.10426946, -2.699271  , -1.1069056 , -1.175127  ,
  -0.4583818 , -3.2083347 , -2.0752823 , -1.4476895 ,  0.7899341 ,
  -0.8538062 , -0.24775589, -2.3843718 , -0.6755549 , -0.99222857,
  -0.46547705, -2.2392697 , -0.9211718 , -0.39940062, -1.2731122 ,
  -0.04383409, -0.784836  , -0.10055067, -0.49293295, -0.03825375,
   0.3906816 , -1.5164461 , -0.82995623, -0.18111528, -0.6302647 ,
  -1.7393712 , -0.23967084, -1.215063  ,  0.01650197, -0.59399307,
  -1.834818  ,  1.7891308 , -0.11999957, -1.6367776 ,  0.16726795,
  -1.0658104 , -0.10820333,  0.15018746, -1.4537379 , -1.6741304 ,
   2.5032218 , -1.4775275 , -0.81292593, -0.77578425],
 [ 0.5176847 ,  0.10148519,  0.12929098, -1.0607089 ,  1.1160202 ,
  -0.5969318 ,  0.79077446, -0.5162965 ,  1.2713745 ,  0.5998277 ,
  -0.893479  , -0.21516325,  0.02603833, -0.40639487,  0.38174677,
  -0.3845424 ,  0.16810265, -0.14667924, -0.05528328,  0.11580328,
  -2.2850034 ,  0.01743998, -1.707808  ,  0.6832543 ,  3.1668587 ,
   0.21618629,  1.0372431 ,  0.7078152 , -0.74079555,  0.09184733,
   0.6143269 ,  0.23888949, -0.5471736 ,  0.99686205, -1.7671353 ,
  -1.0191385 ,  0.3264724 ,  0.5011554 ,  0.43437254,  0.733782  ,
   0.19734971, -0.22135234,  0.51443833,  0.05504538,  0.5131601 ,
  -0.3367    ,  2.5376654 ,  1.1915168 ,  1.4719218 , -1.0431657 ,
  -0.16128387,  1.1431508 , -0.68476635, -0.00032501,  0.16782168,
  -0.8076462 , -0.27363184,  0.21119672, -0.5082104 ,  0.1218186 ,
   0.11160932, -0.8180461 ,  1.629002  , -0.16443737],
 [-0.519648  ,  0.6664834 ,  0.47591463, -0.49995816, -0.48523158,
   0.09688143, -0.4347373 , -0.27779922,  0.97649074, -1.0160524 ,
   0.17577864,  0.14472277,  0.7745353 ,  0.7494982 ,  0.02368227,
  -0.16501649,  0.62093943, -0.81411207, -0.6583544 , -1.2140275 ,
  -0.26703507,  0.14254619, -0.02802599,  0.01399592, -1.9871587 ,
   0.46632895, -0.04385994, -1.2177814 ,  1.441977  ,  0.34153554,
   0.10429654, -0.3783649 , -0.03357864, -0.9045979 ,  1.1186303 ,
  -0.5383981 ,  0.5975963 ,  0.71338046, -0.60725445, -0.46541688,
  -0.2706129 ,  0.95359945,  0.12927215, -0.28097358,  0.25765803,
   0.38643044, -0.67945874,  0.48653573, -0.00471297,  0.0513008 ,
   0.1570013 ,  0.3872522 ,  0.5876468 ,  0.8416806 ,  0.4545563 ,
   1.3476982 ,  0.16666214, -0.32573342,  0.29077667,  0.10716732,
  -0.23813874, -0.7009327 ,  0.4289142 , -0.27383056],
 [-0.24629901, -0.14119959, -0.9355621 ,  0.15264724, -0.21016887,
  -0.32038   , -0.3105532 , -0.17288134, -0.08480635,  1.0815196 ,
  -1.270374  ,  0.5231785 , -0.20934817,  0.07159649, -0.2899932 ,
   0.32145157,  0.26800743,  0.1650391 , -0.52933425, -0.68719816,
  -0.10520443,  0.34443927,  0.35374242, -0.37338096, -2.788228  ,
  -0.35188955,  0.12450715, -0.06436998,  0.6258727 ,  0.36357802,
  -0.19572838,  0.7676958 ,  0.31221667, -0.40033692,  0.04850477,
  -1.0965545 , -0.01592472, -0.42188105,  0.40372303, -0.21714081,
  -1.0039713 , -0.06354323,  0.31681785, -0.5796349 , -0.42117625,
  -0.4490868 ,  0.18214045,  0.6660719 , -0.6667994 , -0.81871194,
  -0.07143142,  0.072647  ,  0.91295314,  0.43704602, -0.93692386,
   0.8186801 ,  0.40991506,  0.10656408, -0.33827844,  0.65160036,
   0.20989548,  0.19269858,  0.4599635 ,  0.2713498 ],
 [-0.6675083 , -0.00900006,  0.1620699 , -0.5967054 , -1.1832491 ,
  -0.50428027, -0.09024002,  0.08513776, -0.30601677,  0.23078999,
  -2.4944265 , -0.23853377, -0.37346905,  0.04652913, -0.45208228,
  -0.01441727, -0.10520324, -0.24686942, -0.25198993, -0.5521652 ,
  -0.26537666,  0.22342677,  0.09084558, -0.07594479,  0.33812463,
  -0.20014885,  0.23407532, -0.3870739 , -1.155355  ,  0.35919344,
  -0.02051219, -0.11240204, -0.37746984, -0.25748625, -0.40232018,
   0.13183877, -0.24428928, -0.25714856, -0.1400142 ,  0.07050642,
   0.04033167, -0.29421166, -0.13787244,  0.02775974, -0.08113965,
   0.11447995,  0.05324435, -0.86876607,  0.01762587, -0.49509442,
  -0.11459473, -0.34195507, -0.9611901 , -0.6535823 , -0.08876154,
  -0.37908056, -0.15385729,  0.4885602 ,  0.12342539, -0.550173  ,
  -0.75448376, -0.18547103, -0.33290872, -0.3121708 ],
 [-0.49308473, -0.38294515, -0.2438425 , -0.09785177,  0.10052649,
  -0.02102738, -1.7393693 , -0.23290807, -0.33853045, -2.166257  ,
  -0.26760036, -0.28422022,  0.23526579, -0.6466112 ,  0.05220385,
  -0.8589067 , -0.2507871 ,  0.26914814,  0.2231222 , -0.44763425,
  -0.49558598, -0.41379595, -1.2902234 , -0.5619228 , -0.0701463 ,
  -0.03259714, -0.4885296 , -0.11640139, -0.23561858,  0.9918281 ,
   0.01258942, -0.45717257, -0.15341008, -0.44795522, -0.20788488,
   0.01344433, -0.06368655, -1.093104  , -0.26057136,  0.13704738,
  -0.15883851, -1.7169034 ,  0.2484092 , -0.04243931, -0.5430397 ,
  -0.514481  , -0.60602564, -0.23057696, -1.3349311 ,  0.46889088,
  -0.5196724 , -0.05904949, -2.2778847 ,  0.6181186 ,  0.00954329,
  -0.65466595,  0.06650016,  0.6255654 , -0.39380777, -0.22553039,
  -0.37858492,  0.03743534, -0.23519383, -0.7990115 ],
 [-0.22585385, -1.2181741 , -0.32450756,  0.4699892 ,  0.08802915,
   0.7136871 , -0.43153697,  0.37164322,  1.139705  ,  0.48118737,
   0.5978211 ,  0.4906701 ,  0.4676799 ,  1.5798504 , -0.01588357,
  -0.43610182, -0.188045  ,  0.6256925 , -0.25321102,  0.10891092,
   0.6707731 ,  0.45637667,  0.1799041 ,  0.3221835 , -0.2747354 ,
   0.07518031, -0.11611234, -0.7963694 ,  0.12546799,  0.1930053 ,
  -0.92288655,  1.0609173 ,  0.16135034, -0.37475398,  0.1586819 ,
  -0.40585038,  1.6435416 ,  0.10604513,  0.03491202,  2.015845  ,
  -0.56220853, -0.85612583, -0.02057271, -0.51060134, -0.35814238,
   0.10379699, -0.25936013,  1.184658  , -0.8269856 ,  0.10456049,
   0.26531002,  0.266591  , -1.021351  ,  1.2991216 ,  0.3181804 ,
   1.2638406 , -0.36567393, -0.29981428,  0.7563775 ,  0.00854036,
   0.1955049 , -0.23871036,  0.18777858,  0.7887567 ],
 [ 0.3077586 ,  1.4421096 , -0.94778323,  0.04335629,  0.36627772,
   0.12631032, -0.16986506, -0.46097606,  0.60833997,  1.067506  ,
   0.6877634 ,  0.23567228,  1.1426985 ,  0.0439711 ,  0.42788   ,
   0.30752447,  0.97082233,  0.41010877,  0.58368903,  0.8872767 ,
  -0.17439471,  0.5782749 ,  0.6721105 , -0.16131508, -2.3959165 ,
   0.73088455,  1.0402251 ,  0.8547259 ,  0.07850211,  0.9325583 ,
   0.34135553,  0.71204334,  0.8607335 ,  0.40126133,  0.73749316,
  -0.09882826,  1.0888913 ,  0.65111464, -0.27452484,  0.10473479,
   0.14305787, -0.07272767,  1.147067  , -0.540101  , -0.7672282 ,
   0.55296063,  0.8673863 ,  0.51662284, -0.01189736,  0.37060195,
   0.22052816,  0.6385613 ,  0.5691296 ,  1.2568464 , -0.03671716,
   1.1834849 , -0.75235283,  0.31817368,  0.510949  ,  0.36358094,
   0.07387761, -0.2233059 ,  0.6866882 ,  0.64540786],
 [-0.06961291,  0.11073016,  0.18856706,  0.1493328 , -0.08066788,
   0.308251  ,  0.74397296,  0.096488  ,  0.3271784 , -1.6198375 ,
  -0.9779166 , -0.120241  , -0.22597124,  0.13489087, -1.1748774 ,
  -0.47611946,  0.41723022, -0.30690593, -0.3353033 , -0.55876374,
   0.14806357, -1.5608376 , -0.12140775,  0.38974443,  1.6255169 ,
  -0.37842622, -0.29810023,  0.6450199 , -1.2620628 , -0.6933955 ,
   0.12413616, -0.996063  , -0.2198525 , -0.6044407 ,  0.5007846 ,
  -0.68257314, -0.6030394 ,  0.40173945, -0.36017793,  0.47770265,
   1.0522764 , -0.73490155, -0.3248064 ,  0.7133418 ,  0.07209104,
   0.41745183,  0.40109518, -0.6801146 ,  0.32842037, -1.1277877 ,
  -0.61253726,  0.20462169,  0.25652206, -0.14051045, -0.06021131,
  -0.6876526 ,  0.02255746, -0.5319858 , -1.8372906 , -0.9344912 ,
   0.83731186, -1.540248  ,  1.5893066 , -1.3667175 ],
 [ 0.5648107 , -0.43888932,  0.658218  ,  0.26965782, -0.3269388 ,
   0.21742351, -0.00516116, -0.13036382, -0.3733597 , -0.4699712 ,
  -0.99420285, -0.14996968, -0.07998772,  0.21385886,  0.05159982,
  -0.2103627 , -0.40994745,  0.15344797, -1.0862693 ,  0.33169755,
  -0.28363138,  0.36222234,  0.26975834, -0.3205347 ,  1.7369574 ,
   0.10577693,  0.01540274,  0.46486378,  0.2714964 , -0.25264743,
  -0.02843646, -0.04084384, -0.18325588, -0.49966878,  0.3257019 ,
  -0.5930736 ,  0.31235102, -0.35180348,  0.40129146,  0.6735248 ,
  -0.07175119, -0.40499058,  0.31893504,  0.31953615, -0.41760424,
   0.05814924,  0.44755659, -0.08412261, -0.4139996 , -0.14182775,
   0.17477556,  0.30330497,  0.0209935 ,  0.09439319, -0.912808  ,
   0.70405185, -0.02767006,  0.2794728 ,  0.5866111 , -0.21073884,
   0.6329119 ,  0.4137952 , -0.5816874 , -0.36302784],
 [ 0.11277723,  1.0331565 ,  0.8443723 ,  1.0404449 ,  0.85592985,
  -0.8217808 , -0.7556826 ,  0.7269211 ,  0.5300654 ,  0.5044755 ,
   1.3336323 ,  0.40817714,  0.39175445,  0.5081958 , -0.06065205,
  -0.13439755,  1.3190788 , -0.3016683 ,  0.02659251, -0.2710481 ,
  -1.0665015 ,  0.19209084,  0.48971143, -0.15686585,  0.7898521 ,
   0.18092573,  1.0225126 , -0.14042419,  0.61729354, -0.81637204,
  -0.12267584,  0.6255292 ,  0.10723899,  0.19863774, -2.6042178 ,
  -0.45600224,  0.00675281, -0.7155858 ,  0.3366431 ,  0.3140981 ,
   1.0225055 ,  1.959232  ,  0.38433734, -0.14754865, -0.6395464 ,
  -0.3730607 , -0.12108152,  0.02224887, -0.26431534,  0.13526376,
  -0.47045103,  0.42904887, -0.93110603, -1.6219057 ,  0.8678107 ,
   0.8047934 ,  0.6461437 ,  0.0665662 ,  1.0301789 ,  0.170672  ,
   0.98958254,  0.61408424,  0.08723564,  0.24339464],
 [ 0.21604887, -0.30036348, -0.2933628 ,  0.07933351, -0.17427519,
   0.1084471 ,  0.70305127,  0.3339146 ,  0.09800444,  0.7637302 ,
   0.03201772, -0.01212402, -0.39111075, -0.2194673 , -0.28458872,
   1.0964719 ,  0.36497125, -0.27892223, -0.8085057 , -0.12211044,
   0.2695173 , -0.68802196, -0.14976427, -0.82543343,  0.06596196,
   0.0435441 , -0.1853948 ,  0.9635164 , -0.84890354,  0.22459364,
   0.23602132, -0.71497303,  0.549169  , -0.49053922,  0.2246788 ,
   0.10522376, -0.11148676, -0.1848484 ,  0.49418452,  0.86303097,
  -0.8071107 ,  0.9276763 , -0.1692305 ,  0.8864658 , -0.24736734,
   0.19766007,  0.34014878,  0.3312993 , -0.4561602 , -0.0707446 ,
  -0.3835134 ,  0.3991944 ,  1.4677612 , -0.45399514,  0.6371734 ,
  -0.2975486 ,  1.3626789 , -0.6036647 , -0.29828066, -0.34339508,
   0.9419485 ,  0.0039031 ,  0.14278315,  0.24903366],
 [-2.393945  , -0.89430493,  0.34598273, -0.07149112, -0.14569806,
  -0.05704889, -0.07045823, -0.6193982 , -0.16797712, -0.35910156,
   0.04239132, -0.8347495 , -0.09065329,  0.19733585, -0.8930966 ,
  -0.6422872 ,  0.63205487, -1.3305637 , -0.87823987, -0.18072157,
  -0.36043024, -0.58369607,  0.18463671, -0.7320493 ,  0.07278549,
  -0.3454778 , -2.0297468 , -1.9228171 , -0.63694173, -0.67656595,
  -0.32229683,  0.36292052,  0.15285306, -0.69090796,  0.06963092,
  -0.17304625,  0.12868278, -0.26010153,  0.05510764, -0.64319474,
  -0.01171859, -0.33397937, -0.69225526, -0.7924836 , -0.12862697,
  -0.27395976, -1.9767247 , -0.809218  , -0.62276363,  0.31915414,
  -0.6359433 ,  0.44295654,  0.06260613, -0.02983488, -0.4737437 ,
  -0.16794935, -0.5812264 , -1.6705878 ,  0.15528233, -0.77051187,
  -0.02320494, -0.8647963 , -0.515022  , -0.12988617],
 [ 0.24417493,  0.6315626 , -1.7498465 ,  0.15845074, -0.5781493 ,
  -0.017213  , -0.01908832, -0.20863695,  0.1352737 , -0.10871091,
  -0.09636949, -0.16827348,  0.33791056, -0.14144847,  0.367071  ,
  -0.34443775, -0.03748862, -0.06119753,  0.3152079 ,  0.26985693,
   0.04727051,  0.25370103, -0.25406128,  0.22804478,  0.7671742 ,
   0.40929493,  0.16598263,  0.36459285, -0.02184533, -0.96724534,
  -0.8443115 ,  0.13907717, -0.24222566, -0.79316586,  0.7392169 ,
  -0.90227026,  0.8418384 , -0.27798027, -0.01624463,  0.482919  ,
  -0.42505223, -0.06951448,  0.27199203, -0.33960247,  0.58120227,
   0.1573301 ,  0.28941065,  0.06916152, -0.6705964 , -0.65936786,
   0.40162465, -0.13640372,  0.5527013 ,  0.63210493, -1.4742452 ,
  -0.15971255,  0.19194365, -0.42508402,  0.21836656,  0.11009288,
   0.2620884 ,  0.27819574, -0.01499975, -0.2472218 ],
 [-0.12897551,  0.77424717, -2.3809948 , -0.11813692,  0.28179047,
  -0.36009443,  0.44671354, -0.47757524,  0.5009011 ,  0.29856417,
   0.8831441 ,  0.17548467,  0.97779584,  0.668786  ,  0.37576866,
   0.73043996,  0.31949642,  0.3625123 ,  0.713054  ,  0.26508105,
   0.55593085,  0.5561766 ,  0.43560827,  0.41192186,  1.4612992 ,
  -0.1656716 , -0.02906762,  0.44508132, -0.07264578,  1.0972363 ,
   0.96176213,  0.16805823, -0.03274946,  0.91037935,  1.8889493 ,
   1.1928139 ,  0.32411084,  0.5187865 , -1.0318044 ,  0.03254435,
   0.07664458,  1.0159988 ,  0.7528155 ,  0.05796005, -0.02765272,
  -0.00418764, -0.05064028, -0.5471431 ,  0.33329308,  0.42474848,
   0.24166523,  0.04773186, -0.23955826,  0.8500889 ,  0.460099  ,
   0.28232113,  0.15151824, -0.5147278 ,  0.21309333, -0.23552817,
   0.40994442,  0.41411516,  0.24120611,  0.67647004],
 [ 0.24393217,  0.8273156 , -0.01607116, -0.9845768 , -1.0902208 ,
  -0.09128956,  1.5218426 , -0.51478136,  0.6534394 , -1.3890578 ,
  -0.5751458 , -0.25698933, -0.6835332 , -0.12185428,  0.12263967,
   0.18502928, -0.05810544,  0.09838872, -0.5514953 , -0.856164  ,
   0.5722619 , -0.5900922 , -0.4165431 , -0.38630188,  3.287445  ,
  -0.45666662, -0.81048995,  0.04564775,  0.69790244, -0.2708792 ,
   0.7167112 ,  0.12707034, -0.6498923 ,  0.39170554,  0.15904176,
   0.5390869 , -1.0961299 , -0.37767538,  0.18468356,  0.01708032,
  -0.6397996 , -0.41465575, -0.6895858 , -1.1013477 , -0.23880778,
  -0.7101874 ,  0.47792113, -0.14204596,  0.09557833, -0.42990777,
  -0.39006853, -0.25176436,  0.69937027, -0.5342828 ,  1.8930578 ,
  -0.59022933,  0.74912906,  0.33602694,  0.04516676, -0.43181074,
   0.24101856, -0.64116573, -0.24670735, -0.3227838 ],
 [ 0.51156026,  0.5326543 , -1.2653731 ,  2.335542  ,  0.15161218,
   0.60532504,  0.33094987,  1.050878  ,  1.1304227 ,  0.7260518 ,
   0.03499493,  1.2867281 ,  0.8582314 ,  1.7726029 ,  0.21375707,
   1.0241679 ,  0.8159027 ,  1.4645154 ,  0.5898904 ,  0.20998284,
   0.5403477 ,  0.36541903,  0.83689064,  0.5998456 , -0.5852471 ,
  -0.29738683,  1.015903  ,  1.2303151 ,  0.80476606,  1.2599151 ,
   0.4093175 ,  0.14758924,  0.70596164,  0.28218588,  0.12561987,
  -0.5844506 ,  1.8597243 ,  0.6641741 , -0.14425772,  1.6458141 ,
   0.5137151 , -0.566478  ,  1.3466824 ,  0.46837032,  0.38088357,
  -0.3643628 ,  0.31319326,  1.510533  , -0.34682712,  0.6648625 ,
  -0.15726623,  1.1628766 ,  0.9731421 ,  0.4926579 , -0.2227547 ,
   3.1543665 , -1.2925813 ,  0.530354  , -0.24369691,  1.5996069 ,
   0.22421603,  0.6586986 ,  1.1692274 ,  0.10200804],
 [-0.5479047 , -0.18877701,  0.17903061, -0.2588143 , -0.5902544 ,
  -0.52600104,  0.38395154, -0.5877845 , -0.61304116, -0.40325725,
  -0.4712032 , -0.6659789 , -0.29232275, -0.7071272 , -0.4190515 ,
  -0.30458236, -1.4042466 , -0.74138266, -0.28368574, -0.6553918 ,
  -0.566775  , -0.58695924, -0.4834265 ,  0.06476545,  0.10353731,
  -0.55741185, -0.8101427 , -1.4287442 , -0.10579631, -0.23050444,
   0.14158973, -0.49343896, -0.47909728, -0.12824835,  0.06076093,
  -0.18521164, -0.5738624 , -0.4179128 , -0.9315186 , -0.927287  ,
  -0.00146486, -0.3642751 , -0.54000497, -0.7547489 ,  0.05482238,
  -0.7615413 , -0.88163847, -0.5095826 ,  0.05209246, -0.391556  ,
  -0.39329794, -0.8824776 , -0.56985724,  0.29013348,  0.1237288 ,
  -0.56969666, -2.7334344 , -0.72046137, -0.49356484, -0.22455417,
  -1.3402067 , -0.6691401 , -0.61297584, -0.61427647],
 [-1.2038379 ,  0.12114315, -0.09295255,  0.37767583,  0.7928824 ,
  -0.16516161,  0.7517778 , -0.5995087 ,  0.44702122, -0.2201297 ,
   1.9430975 ,  0.08831538, -0.21540171, -2.083416  ,  0.6244408 ,
   0.8146728 , -0.4722573 , -0.34960002,  0.6476308 ,  1.009183  ,
  -0.6700291 ,  0.9017565 ,  1.3291396 , -0.06152465, -0.25436848,
   0.641628  ,  0.2726269 ,  0.82178247,  1.100548  ,  1.3796654 ,
  -0.11108633,  0.56813025,  0.10778426,  0.45659694, -0.22023875,
   0.6604005 , -1.1623197 ,  0.36189818,  0.71020544, -0.5966072 ,
   0.42667204,  0.8962969 ,  0.9047172 , -0.7134913 , -0.06552991,
   0.19759384, -0.48934042,  0.17636403,  0.9388188 ,  0.4447617 ,
  -0.02819879, -0.50571424,  0.94548386, -0.08447296,  1.3320092 ,
  -0.4441615 , -0.26915818, -0.5569298 ,  1.3752788 ,  0.40264103,
   0.1628934 , -0.11636103, -0.4173691 ,  0.19250512],
 [-0.0210976 ,  0.29730597, -0.10800816,  0.39639613,  0.34029564,
  -0.15435325, -1.3692803 ,  0.04869544,  0.35033402,  0.8776579 ,
   0.4358263 ,  0.29660296,  1.1538101 ,  0.2456602 ,  0.30747727,
   0.3713579 ,  0.3729985 ,  0.52058136, -0.04285196, -0.4058532 ,
   0.30578753,  0.5346492 ,  1.4410483 , -0.81767774, -0.13887426,
  -0.2507512 ,  0.64671904, -0.58787775, -0.03600193,  0.32057324,
   0.2771665 ,  0.6418736 , -0.0035989 , -0.31231567, -0.26858228,
  -0.5002893 ,  0.54808164,  0.15798497,  0.18054362,  0.17725085,
  -0.3424304 ,  0.19849372,  0.38763598, -0.43243256, -1.0293642 ,
   0.02839744,  0.51047987,  0.70889854, -0.23415887,  0.22353314,
   0.5513832 ,  0.26183343,  0.3945806 ,  0.3958126 , -0.7301358 ,
   1.1041304 ,  0.5371023 ,  0.22096701, -0.00693326, -0.09265297,
  -0.05005307, -0.36794573,  0.6543423 ,  0.33142203],
 [-0.6314703 ,  0.21380314, -0.32578516,  1.7230848 ,  1.2012217 ,
  -0.08814429,  0.67788064, -0.2406782 ,  1.4503824 ,  1.101868  ,
   0.9625148 ,  1.2094351 ,  1.219558  ,  1.0900809 ,  0.7402566 ,
   0.24817173,  0.5553209 ,  1.213656  ,  0.5393197 ,  0.61917406,
   0.15046135,  0.8173571 ,  0.8754215 , -0.22909385,  0.27490544,
   0.17176378,  0.750429  , -0.5225004 ,  1.7525382 ,  0.01865661,
   0.545727  ,  1.4114825 , -0.547823  ,  0.34834674, -0.29063943,
   0.29636383,  0.75631815,  0.10807487,  0.32045138,  0.2904678 ,
  -0.26296297,  0.24419445,  0.8564176 ,  0.6084189 , -0.73774415,
  -0.44278175,  0.9118762 ,  2.2152917 , -0.7919819 ,  0.07805704,
   0.5808334 ,  0.489503  ,  0.5993491 ,  2.0835395 , -0.8618267 ,
   0.561997  , -0.1644014 ,  0.10651154,  0.7759275 ,  1.188454  ,
   0.11269282,  0.30826214, -0.04142131,  1.3425993 ],
 [ 0.53089297,  0.2969817 , -0.03790041,  0.84567815, -0.7110382 ,
  -0.01508395,  0.32448784,  0.41956165,  0.6227812 , -0.10649864,
   0.9803816 , -0.3277269 , -0.127698  ,  0.17874292,  0.07652657,
   0.08771183,  0.37958556, -0.0445555 , -0.30253115, -0.11867846,
   0.11189879, -0.46052414,  0.44728658,  0.316247  , -2.4545817 ,
  -0.16168931,  0.17627649,  0.1760839 ,  0.20396504, -0.54906046,
  -0.00877643, -0.10357515,  0.55551237,  0.5170752 ,  0.65299517,
   0.24007943,  0.15294334,  0.18923882,  1.2739396 ,  0.8794528 ,
   0.23422115,  0.6865765 , -0.20449631,  0.01562596, -0.09229667,
   0.47532347,  0.3728917 ,  0.07184207, -0.3475138 ,  0.41130963,
  -0.51326734,  1.0629793 , -0.31397834, -0.42803648,  0.68690825,
   0.43992922,  0.81292987,  0.01103855,  0.441484  ,  0.1306665 ,
   0.2950825 , -0.3882232 , -0.02634773,  0.09005969],
 [-0.34674177,  0.03779767, -0.27721167, -0.06285597,  0.7589311 ,
  -0.86184907, -0.05845238, -0.331848  ,  1.3509946 ,  0.6175653 ,
   0.65585214,  0.43549472,  1.0093901 ,  0.23295572,  0.08865915,
   0.20144081,  0.40579984,  0.25447437, -0.16629659,  0.19272518,
   0.12665938,  1.2056003 ,  0.47018355, -0.577214  , -0.35375425,
  -0.05264672,  0.17123291, -0.34553266,  0.42960802,  0.2610838 ,
   1.201417  ,  0.46664134,  0.623658  ,  0.0792824 , -0.808567  ,
  -0.5558146 ,  0.27336   , -0.00308586, -0.8840349 , -0.5370299 ,
  -0.6014183 ,  0.47977978,  0.31965834,  0.4224806 , -0.7643124 ,
  -0.2030876 ,  0.45472875,  0.15342087, -0.5783849 , -0.3128056 ,
   0.58280784, -1.1110126 ,  1.2437929 ,  1.0865918 ,  0.5258955 ,
   0.57971203, -0.31559268,  0.2936303 , -0.20152459,  0.49412784,
   0.0082622 , -0.54804426,  0.57126933,  0.52459216],
 [ 0.07645708,  0.1740442 , -0.10074081,  0.45107025,  0.7051395 ,
   0.1669849 , -0.61800367,  0.4620548 ,  0.04317226,  0.69930226,
   0.96428174,  0.47438684,  0.61332375,  0.39355397,  0.10578775,
   0.2536021 ,  0.26337245,  0.42178413, -0.587974  , -0.39389807,
   1.0405681 , -0.00390638, -0.42218965,  0.12268876,  0.0217768 ,
  -0.81523305,  0.26959908,  0.00025878,  0.30607498, -0.36534277,
   0.15718667, -0.48944858,  0.06280243, -1.3870643 ,  0.25641888,
  -0.36431736,  0.20748286, -0.8277064 ,  0.36807638,  0.6638551 ,
  -1.3696626 , -0.7355835 ,  0.3232189 ,  0.7604912 , -0.24167247,
   0.6224822 ,  0.00386567,  0.50299174, -0.64951986, -0.7717874 ,
   1.0678302 , -0.24410756, -0.5085518 , -0.43464524, -0.61952823,
   2.3191452 ,  0.90663826, -0.7490262 ,  0.0216127 ,  0.05866245,
  -0.01483351, -0.22385183, -0.14790201, -0.44691667],
 [ 0.7647043 ,  0.6020682 , -1.3563516 ,  0.3987515 ,  0.44265744,
   0.37747934,  0.65429807,  0.5634052 , -0.01113948,  0.36659116,
   0.4540989 ,  0.22380717,  0.6916353 ,  1.3496728 ,  0.02828212,
   0.36897185,  0.5103064 ,  0.35548657, -0.19991484,  0.6935061 ,
   0.59537846,  1.1338221 ,  0.9999945 , -0.08865684, -0.76190394,
   0.48270842,  1.2462642 ,  1.0917337 ,  1.0889492 , -0.27021265,
  -0.36414552,  0.92259395,  0.03106493,  0.26089308,  0.50867   ,
  -0.42816287,  0.24334027,  0.5140277 ,  0.91896355,  0.9443402 ,
  -1.2026287 , -0.48611233,  0.3247258 , -0.29790947, -0.79951125,
  -0.40929818,  0.6894944 ,  0.8597868 , -0.65182805,  0.46084332,
   0.22945502,  0.7383728 , -0.07681362,  0.83048695,  0.68469745,
   0.8009073 ,  0.08774576,  0.74022657,  0.6110486 ,  0.5145303 ,
   0.20673339,  0.8686519 ,  0.97084713,  0.3596422 ],
 [-0.7562325 ,  0.2845013 , -0.4865739 ,  0.2397491 , -0.34238777,
   0.5913217 , -0.60236704,  0.25675368,  0.3333973 , -0.0399983 ,
  -1.1529974 , -0.5542652 ,  0.64304465,  0.9291289 , -0.42727345,
   0.01369958,  0.41877192, -0.51422524,  0.18996306, -0.40701964,
  -0.3481934 , -0.13105471, -0.10335146, -0.20603456,  0.31542474,
  -0.0067918 , -0.13017526, -1.401617  ,  0.09450852,  0.10399754,
   0.7837482 , -0.4999314 , -0.77291846, -0.04488152, -1.6529838 ,
   0.45574817,  0.22974287,  0.8608568 ,  1.4094801 , -0.3381376 ,
  -0.19938043, -4.4074497 ,  0.2888105 ,  0.08069667, -0.28430343,
  -0.7368437 , -3.8279068 , -0.88709885,  0.81376106, -0.14130476,
   0.2861657 , -0.9432475 ,  1.0017774 ,  0.8742446 ,  0.31174135,
  -0.59203136, -0.2071368 ,  0.16491415, -0.13580763,  1.0593882 ,
   0.6303197 , -0.6707065 , -0.0154854 ,  0.10511999],
 [-1.9422895 , -0.1763306 , -0.07684309,  1.0997834 ,  0.63764316,
  -0.01217373, -0.04478646, -0.97905016, -0.0191058 ,  0.5105799 ,
   1.0245391 ,  1.310707  ,  1.2481622 , -0.00813972,  0.3278287 ,
   0.5604519 ,  0.43505663,  1.3131583 ,  0.7779316 ,  1.9151546 ,
   2.0992103 ,  1.3762854 ,  1.5079123 ,  0.33598784,  0.08500516,
   0.94353956,  0.9439085 , -1.0104837 , -0.68316174,  1.3972008 ,
  -0.2727325 ,  1.1925008 ,  0.30784467, -0.036897  ,  0.2509223 ,
  -0.8502823 ,  1.5273085 , -0.86162734,  0.5805385 ,  1.0601835 ,
   0.12447523, -0.7797291 ,  0.96721095, -0.61219656,  0.49611947,
   1.2325528 , -2.1477652 , -0.01728771, -0.58048254,  0.53305584,
   0.8142049 ,  0.30302608, -2.0339997 ,  0.08346987, -0.6859297 ,
  -1.882642  , -0.56838876, -0.82203734,  0.81784993,  0.7616869 ,
   1.9332039 ,  0.8292048 ,  0.2427297 ,  0.62121165],
 [ 0.21247014, -0.18759832, -0.32468626, -0.77381665,  0.01776481,
   0.51995873, -0.13071142, -0.86578894, -0.7265119 , -0.61814344,
  -0.42981374, -0.96525085, -0.6894686 ,  0.04980134, -0.22289722,
  -0.2060005 ,  0.01326073,  0.03174141, -0.23491016, -0.48424944,
   0.374496  , -0.59834963, -0.14230533,  0.09904251,  0.07148731,
  -0.24367467, -0.97434694,  0.49492636,  0.47932974, -0.9659952 ,
  -0.7878363 ,  0.15622133,  0.16050068,  0.11026184, -0.87647   ,
  -0.40690255, -1.315544  , -0.70947313, -0.19752342, -0.6722172 ,
  -0.16205998, -1.0471617 , -0.42731538,  0.21153688, -0.00663796,
   0.26457593, -1.6771572 ,  0.27484548, -0.29215613, -0.5002002 ,
  -0.66403824, -0.66816235, -1.406388  , -0.06739817,  0.09886794,
  -0.91165686, -1.1167927 ,  0.41878447,  0.5298644 , -0.73369485,
  -0.4429506 , -1.2283586 ,  0.30064848, -0.35527074],
 [-0.21921201, -0.49811006, -0.13215262,  0.39193073, -0.67092055,
  -0.28604883, -0.03985979, -0.14326094, -0.48461854, -0.07831986,
  -1.626085  ,  0.55633533, -0.14540322, -0.7211591 , -0.12709062,
   0.32871157,  0.46586132, -0.33480516, -0.26869237, -0.5197218 ,
  -1.1437304 ,  0.37026876, -0.3283283 , -0.68474734, -0.02976089,
  -0.1596502 ,  0.39713907, -0.18286501,  0.47224408, -0.31122008,
   0.33184156,  0.38835683, -0.079557  , -0.3260764 , -0.36063603,
  -0.29091963, -0.17788371, -1.115636  , -0.5792879 , -1.842772  ,
   0.13944374, -1.1928097 , -0.38865134,  0.80980456, -1.2360005 ,
   0.05978165, -2.1910954 , -0.5640788 , -0.21348697,  0.06593236,
  -0.55777234, -0.84680146, -0.55317694, -0.30878377, -0.11709329,
  -0.5286646 , -0.50085276,  0.41111815, -0.6090611 , -0.4068161 ,
  -0.6988909 , -1.0754277 , -0.1217065 , -0.55773425],
 [ 0.896643  ,  0.68975896, -0.12158139,  0.98062444,  0.17877084,
   0.32787922,  1.1627035 , -0.04208357,  1.0334208 ,  1.2833438 ,
   0.28089473,  1.7593929 , -0.16915734,  1.676488  ,  0.7192689 ,
   1.0102854 ,  0.65733826,  1.6628407 ,  0.42148197,  1.0044851 ,
   0.8817238 ,  1.5896626 ,  1.3245901 , -0.00043861,  0.54292417,
   0.37415475,  1.5971807 ,  1.3961827 ,  0.8682564 ,  0.37347275,
   1.8808845 ,  1.7899716 , -0.0545782 ,  0.5722439 ,  0.5859013 ,
   0.48028132,  1.0473155 ,  0.49087432,  0.40287957,  0.5875414 ,
  -1.0817026 ,  0.73492014,  1.5559231 ,  0.7499337 ,  0.38210684,
   0.02738572,  1.2924381 ,  2.269762  , -0.74267304,  0.14496917,
   0.21230029,  0.7682895 ,  0.72149265,  0.6861924 ,  0.30878273,
   0.16452943, -0.53980887, -0.02442504, -0.11320976,  0.56608534,
   0.36237425,  0.5168339 ,  0.79009473,  1.0679568 ],
 [-0.0828291 , -0.18453306, -1.7195879 ,  0.1140295 , -0.4352212 ,
   0.20670629,  1.4457849 ,  0.19374229, -0.12409779, -0.21399587,
  -0.1014055 , -0.11246437, -0.32742986,  0.52318245,  0.59666866,
   0.4515953 ,  0.21704218,  0.25070804, -0.17541783, -0.00950476,
  -0.34889776,  0.13532011, -0.07658033,  0.5060371 , -0.67675525,
   0.18246706,  0.13024682, -0.14656448, -0.65693814, -0.5554339 ,
   0.10348623,  0.04353319, -0.6709151 ,  0.11604612,  0.09293134,
  -1.40242   , -0.23633505,  0.11616798,  0.16477306,  0.43702894,
   0.52110684, -0.3263815 , -0.14251295, -0.14422612,  0.4526658 ,
  -0.2155986 ,  0.623057  , -0.26310128, -0.5303149 , -0.52547413,
  -0.31373933, -0.18725103,  0.40391573,  0.11006492, -0.68200284,
   0.39347616,  0.39059615, -0.42382422,  0.4980865 ,  0.00847682,
   0.41181958, -0.26884204,  0.02678786, -0.20558663],
 [-1.3144425 , -0.45365554, -0.37398285,  0.6144625 , -1.0188717 ,
   0.04893857, -1.4506106 , -0.26636845,  0.06780906, -0.28607625,
  -0.36214426, -0.5371572 ,  0.01856541,  1.1312308 , -0.23658878,
  -0.39842352, -0.16021536,  0.05228498, -1.1635195 , -0.33229846,
   0.0522686 , -0.31374273,  0.31785047,  0.18343408,  0.23791759,
  -0.8302223 ,  0.2713907 , -1.0863669 , -0.21503595, -0.9094286 ,
  -1.0074362 ,  0.2692547 , -0.2624424 , -0.7360708 , -0.7868843 ,
  -0.7218491 , -0.6151645 , -0.7365985 ,  0.05039628,  0.9073126 ,
  -1.8974786 , -0.5894218 , -0.40611443,  0.46533188, -1.0114025 ,
  -0.13407922, -1.8304611 , -0.19066703, -1.0529715 ,  0.19456288,
  -0.2587233 ,  0.93716675, -1.2764971 , -0.15898292, -0.30687442,
   0.8043257 ,  1.2815989 , -0.9195735 , -0.40748286, -0.30381307,
  -1.1272848 ,  0.1107694 , -1.2808498 , -0.15545706],
 [-0.6952645 , -0.34657067, -0.45668185, -0.00273659,  0.05749701,
  -0.7237065 , -0.40099773, -0.41543174, -0.03641606, -0.02166493,
   0.05050074, -0.6623154 ,  0.48913378, -0.43206888, -0.05334252,
   0.08309057, -1.9041812 , -0.34762445, -0.27933866, -0.05640466,
  -0.5101993 , -0.18556195, -0.09630522, -0.3567858 , -0.03348608,
  -0.2080271 , -0.95452434, -0.5421234 , -0.65753156, -0.08933662,
  -0.77661   , -0.18463513, -0.34684172, -0.3722744 , -0.79586864,
  -0.85633147,  0.20150273, -0.45120585, -0.7518805 , -0.6247078 ,
  -1.0211321 , -0.45194843, -0.4891609 , -0.46502674, -0.89629805,
  -0.466306  , -0.7139416 , -0.41322502, -0.98529863, -0.25035888,
   0.4233739 , -0.5204321 , -0.18787187,  0.22628908, -0.96996737,
  -0.15591685, -0.28188005, -0.6558479 , -0.14529406, -0.5824763 ,
  -0.33721942, -0.516122  , -1.3578657 , -0.42420653],
 [-0.20557441, -0.27002162,  0.14930807, -0.9765206 , -0.02025815,
  -0.07850961, -0.42359117, -0.96843624,  0.11135971,  0.00992704,
   0.32104397,  0.30815226,  0.05442697,  0.7444969 , -0.1625276 ,
  -0.35136554,  0.5438746 , -0.08861475, -0.14207909,  0.24710622,
   0.02033385,  0.20661171,  0.25884143,  0.04626539,  0.3270502 ,
   0.31503904, -0.133633  , -0.33142614, -0.34822404,  0.05699683,
  -0.48947957,  0.43428493, -0.73484224, -0.41074777, -0.07516902,
  -0.2910514 , -0.00794907, -0.09131946,  0.33160973,  1.0369501 ,
  -0.17830408, -0.24213557, -0.02474419, -0.6182001 , -0.16504622,
  -0.29775944, -0.00842941, -1.3252524 ,  0.0301057 , -0.09112027,
  -0.16449223,  0.7891271 ,  0.19077703, -0.7311844 , -0.23488525,
   0.32939765,  0.51166326, -0.23089461, -0.6145515 , -0.31708086,
  -0.14191687, -0.55206394, -0.3567079 ,  0.8900051 ],
 [ 0.38942942,  2.245201  , -2.0813012 ,  0.5085478 ,  0.6502007 ,
  -0.3813116 , -1.0921292 , -0.13485211,  0.3497573 ,  0.9815134 ,
   0.4473558 ,  0.08528494, -0.40978616, -0.15290992,  1.391019  ,
   0.55398   , -0.03917966, -0.85843664,  0.5019007 ,  1.1578866 ,
   0.2561787 ,  1.2488904 ,  1.6794013 ,  0.518812  , -1.4394475 ,
   0.59847313, -0.12789503, -0.74023   ,  0.80351335,  0.18482046,
  -0.8557831 ,  0.499826  ,  0.24164905,  0.7248805 , -0.14741811,
   0.28612846,  1.3908844 ,  0.34775966,  0.9442157 , -0.43212932,
   1.2674294 , -0.07025379,  0.58705854, -0.33284032,  0.21347746,
   0.00369951,  1.4759755 ,  0.4859636 ,  0.12517346,  0.49210936,
   0.21300337,  0.5877475 , -0.42632043,  1.2160217 , -0.6357643 ,
   1.6924015 ,  0.3857082 ,  0.49198732,  0.7772673 ,  0.68973416,
  -0.37339604,  1.117068  ,  0.10918217,  0.42975542],
 [ 1.2860895 , -0.29422325,  0.02243607, -0.4645225 ,  2.8167334 ,
  -0.46393692,  0.56662625,  0.57290345, -1.2016636 , -0.6257567 ,
  -0.3624726 ,  0.06507071,  0.16811152,  2.2411387 , -0.98811656,
  -1.3362563 , -0.48364317,  0.02040415,  0.36093065, -0.7360328 ,
  -0.18363054,  0.23868403,  0.2916563 , -0.8142721 ,  0.9188416 ,
  -0.02349168, -0.37385228, -0.28639182,  0.6767765 , -0.3880928 ,
  -0.3837912 ,  0.11044986,  0.42623484,  0.7380578 ,  0.17066175,
  -0.05796972,  1.0230585 ,  0.18455249, -1.0513791 , -0.55542666,
   0.26177272, -1.6905597 , -0.15308145, -0.00434201,  0.04077528,
  -0.1282552 ,  1.3821852 ,  0.111176  , -0.6077288 , -0.07940081,
  -0.05664098, -0.45212182, -0.949291  ,  0.6092017 , -0.48953173,
   1.7605195 , -0.47615087, -0.5191213 ,  0.29885116, -0.15959284,
   0.15085202,  0.44282535,  0.07959564,  0.505431  ],
 [-0.94221747,  0.219197  , -0.24634276,  0.8042861 , -0.1515628 ,
  -0.17857419,  0.8479198 ,  1.5547149 ,  0.12086538,  0.8373147 ,
   0.8797113 ,  1.0116633 ,  0.7845366 ,  0.19939938,  0.66256946,
   0.22646344,  0.537456  ,  0.4908861 ,  0.25653762,  0.5875127 ,
   0.92624474,  2.2308254 ,  0.68650454,  0.2084056 , -0.39141095,
  -0.12580445,  0.52650636, -0.6237187 , -0.19026187,  3.0335941 ,
   0.6900172 ,  0.86058205,  0.41003498,  0.08847935,  0.49016115,
   0.30598053,  1.6188011 ,  1.1862514 ,  0.23750292,  2.0347118 ,
  -1.0822536 , -0.6099062 ,  0.9823365 ,  1.325109  , -0.30148414,
   0.49460834,  1.5570475 ,  1.553417  , -0.46123415,  0.34596094,
   0.556789  ,  1.892794  ,  0.5554028 ,  1.4654844 , -0.22675076,
   0.5348189 , -0.6481029 , -0.63975513, -0.15614554,  0.4444681 ,
   1.6000142 , -0.03031106,  0.3871653 ,  0.09574426],
 [-0.34997195,  0.02855344, -0.26804194, -1.1460387 , -0.12776455,
  -0.46057674, -1.7150191 , -0.6733689 ,  0.26962128,  0.08089729,
   0.79343235, -0.04713241, -0.2572924 , -0.06199823, -0.7686167 ,
  -1.1300522 , -0.86892384,  1.4662907 , -1.8672903 , -1.6170541 ,
   1.1512038 , -0.8936806 , -0.442151  , -0.6503336 ,  0.05840851,
  -1.262956  ,  1.734235  , -0.97145087, -0.5447435 , -0.56597275,
  -0.8409091 , -0.6431177 , -0.19589344, -1.3236102 , -1.6749864 ,
  -1.4733549 , -1.2248092 , -0.9381418 , -0.24113744,  1.7318655 ,
  -0.9186289 , -2.0851831 , -0.97939724,  0.21106614, -1.0580271 ,
  -0.5553553 , -0.23258859,  0.31346154, -1.2736387 , -0.98282194,
   0.513564  , -0.1478536 , -0.47237787,  0.12973836, -1.6346654 ,
   0.13477674,  0.9279032 , -1.1963154 , -0.10903963, -0.1984401 ,
  -0.03573642, -0.26966825, -0.5520798 , -0.10667833],
 [ 0.21651992, -0.40967265, -0.33014303,  0.40590322,  0.05937617,
   0.09577062,  0.69323176,  0.08871067, -0.12480638,  1.2709714 ,
   0.6228124 ,  0.16908917, -0.06781091,  0.29320002, -0.01634311,
   0.3910819 ,  0.61398643, -0.16258092, -0.27731815, -0.05848847,
  -0.12563787,  0.46998417,  0.61193496,  0.27746427, -1.7891932 ,
  -0.353721  , -0.16591437,  0.35804194, -0.36269632, -0.45462832,
  -0.28617924,  0.73915005, -0.6101049 , -0.34108302,  0.2913366 ,
  -0.0071077 ,  0.77759093, -0.21210173,  0.24357402,  0.10874738,
   0.30336642,  0.83143544,  0.07798082, -0.25896293,  0.49362198,
   0.04470809, -0.27180746,  0.08264567,  0.52459455, -0.11093122,
   0.26196778,  0.18123817,  0.23140107,  1.3197898 ,  0.28687933,
   0.7860249 , -0.16840717,  0.15445393,  0.01600349,  0.02392107,
   0.19620712,  0.11354409, -0.27550754, -0.49398118],
 [ 0.69034964,  1.0078636 , -0.00507278,  0.5654075 , -0.26213482,
   0.456015  , -0.08028815,  1.8094622 ,  0.7643208 ,  1.2923069 ,
   0.9676222 ,  0.07309201,  0.49763635,  0.19590186,  1.2197608 ,
   1.0687631 ,  1.1755451 ,  0.03452777,  0.01468717,  0.29242414,
   0.85613877,  0.4127688 ,  0.35477188, -0.14366181, -0.24240123,
   0.4479159 , -0.0982163 ,  0.1893685 ,  1.0195887 ,  1.4022936 ,
  -0.23217477,  0.20646328, -0.7281689 ,  0.76257116,  1.0317641 ,
   0.14212635,  0.5355762 ,  0.38351622,  0.23685293,  0.7354132 ,
   0.4923343 ,  0.1016468 ,  1.0773275 ,  0.31203473,  0.12722161,
  -0.13710861,  1.23081   ,  0.32867527, -0.16980673, -0.35610703,
   0.26245347,  0.9810528 ,  0.05396161,  2.2514246 ,  0.03561493,
   0.53758526,  0.54204315, -0.5806886 ,  0.90052974,  0.38420004,
  -0.7673275 ,  1.428854  ,  0.29347116,  0.44651943],
 [ 0.0653532 ,  0.03901089,  0.02929532,  0.7343102 , -0.16020083,
  -0.47268647, -0.10117002,  0.50057966,  0.24553837,  1.3862348 ,
   1.2625477 ,  0.17436822,  0.8102323 ,  1.4972274 ,  0.38478467,
   0.39764833,  0.16315314,  1.2323934 , -0.3693388 , -0.60145366,
   0.41073945, -0.34253824, -0.21457106, -0.852911  ,  0.13251352,
  -0.12019076,  1.1785554 , -0.2399701 ,  0.20710056,  0.8194199 ,
  -1.0289481 , -0.5898091 , -0.27657443, -0.78140813, -0.23566417,
  -0.6824749 ,  0.72925526,  0.95695835, -1.1453705 ,  0.71135193,
  -0.00573969, -0.81248015, -0.13111894,  0.65566874, -0.550841  ,
  -0.6834002 ,  0.7412382 ,  1.0141333 , -0.97128636,  0.13037775,
   0.38356367,  0.20042631, -0.25262505, -0.83053905, -1.0716908 ,
   0.38084227,  0.57262415, -0.6744989 ,  0.32166287,  0.15531147,
   0.10093488, -0.5808611 ,  0.47087583, -0.1413839 ],
 [-0.72092974,  0.15189026, -0.7444314 ,  0.81357586,  0.27150136,
   0.8199477 , -0.3027542 ,  0.659479  ,  0.21253824,  0.23780206,
   1.1160243 ,  0.68977153,  0.43824106,  0.20510374,  0.44370505,
   0.0456372 ,  0.17953564,  0.7527179 ,  0.35473776,  0.8895288 ,
   0.578888  ,  0.74985605,  0.30479878, -0.05030917, -2.1516564 ,
   0.13703059,  0.15936673,  1.377492  ,  0.28947955,  0.4255813 ,
  -1.415349  ,  0.53124225,  0.33147174, -0.13197082,  1.3693572 ,
   0.02019414,  1.0628461 ,  0.18602094,  1.2327098 ,  1.0333328 ,
   0.2495234 ,  0.49766126,  0.21373484,  0.87905246, -0.47563162,
   0.6236633 ,  1.222162  ,  0.37147963, -0.49385104, -0.36326125,
   0.02192886,  1.033714  ,  1.762648  , -0.83676434, -0.5223137 ,
   0.6143845 ,  0.4882617 ,  0.18477255,  0.73589396,  0.23012954,
   0.64372873, -0.05103619, -0.12489265,  0.3881704 ],
 [ 0.07723592, -0.0372228 , -0.11888498,  0.6242658 ,  0.085035  ,
   0.36902916,  0.35846302,  0.00686472,  0.16363464, -0.36004505,
   0.32618362, -0.21560769, -0.02078826,  0.13085718,  0.34605774,
   0.35838878, -0.03960843, -0.20115195, -0.21490082, -0.22137555,
  -0.07614361, -0.5001602 , -0.03447857, -0.05910877,  0.02267032,
  -0.11092754, -0.40464228, -0.25125793,  0.05950893, -0.11690724,
   0.01236708, -0.26231736,  0.2961123 ,  0.2993753 , -0.0504307 ,
   0.04325734, -0.15618783,  0.3274584 ,  0.09076132,  0.3629237 ,
  -0.28558347, -0.3985847 , -0.2414275 ,  0.26914155, -0.35027972,
   0.09172359, -0.02763033, -0.19421308, -0.172817  , -0.11666764,
  -0.3096261 ,  0.29521325,  0.915614  ,  0.04961335,  0.05151133,
  -0.22116199,  0.62182176, -0.26044697,  0.07382925, -0.25019738,
   0.77317077, -0.29225495, -0.14160757, -0.10949049],
 [-0.33747217, -0.42098624, -4.570299  , -0.14168842, -0.45549726,
   0.2373772 ,  0.24328767, -0.02994695, -0.13469097,  0.8710244 ,
   1.0144974 , -0.1865578 ,  0.6065671 ,  0.22022578,  0.19821681,
   0.00634473, -0.18309389, -0.03313394, -0.03722979,  1.0066566 ,
  -0.02892574,  0.69467646, -0.20348443, -0.81794393,  1.8432355 ,
  -0.16224146,  0.03199522,  0.268743  ,  0.58953744,  0.7064262 ,
   0.29550496,  0.24393837, -0.4031646 , -0.16860774, -0.03864361,
  -0.24005714,  0.22786483, -0.04200365, -0.03736502,  0.6169134 ,
   0.36617145, -0.336191  , -0.282369  ,  0.12814708, -0.20695838,
   0.6263413 ,  0.29732993, -0.41141284, -0.07557643, -0.15386169,
   0.16944073, -0.3107029 , -0.1612722 ,  0.43221962,  0.37013775,
  -0.15213418, -0.09239273, -0.02919178, -0.04515974, -0.0303361 ,
  -0.2776825 ,  0.6296488 ,  0.11217057,  0.10022855],
 [-0.8351163 , -0.15115447, -0.19145787,  0.44881678, -0.14620814,
   0.24145314,  0.68724144,  0.5700744 , -0.07749209, -0.27031332,
  -0.8647534 , -0.2931835 ,  0.30444688, -0.6679057 ,  0.07820527,
   0.5975561 , -0.07347113, -0.7183093 ,  0.87620026, -0.03806528,
  -0.6033688 ,  0.0500927 ,  0.00570988,  1.6009136 , -0.13833624,
  -0.21426514,  0.3372144 ,  1.8695766 , -0.58112687, -0.09261254,
   0.53202146,  0.19346656, -0.2353234 ,  0.4101768 ,  0.6465493 ,
  -0.23575883,  0.00774884, -0.93336135,  2.0194302 ,  0.5087675 ,
  -0.3343258 ,  2.3247178 , -0.45686847,  1.0327214 ,  0.6913156 ,
   0.0988448 ,  0.20084587,  0.14457677,  0.5523256 , -0.32067406,
   0.28317595, -0.08336572,  0.59945893, -0.18742213,  0.08032037,
  -0.27950704,  0.0638662 ,  1.0307641 , -1.3549715 , -0.64169115,
   0.32178608,  0.646727  ,  0.87553656, -0.45319256],
 [ 0.6181688 ,  1.7419906 , -1.7394431 ,  0.7909053 ,  0.37631798,
  -0.04580368,  1.5124981 ,  0.76897216,  0.7505491 ,  1.4275657 ,
   0.40614802,  0.25749874,  0.9117701 ,  0.5292836 ,  0.799529  ,
   1.1278086 ,  0.28982055, -0.32744586,  0.16707036,  0.6053532 ,
   0.28489828,  1.4832151 ,  1.4140311 ,  0.3193105 , -1.6893482 ,
   0.3780594 ,  0.73920953,  0.39973226,  0.62801373,  2.0871067 ,
   0.22144845,  1.4950936 , -0.44650388,  0.3161371 ,  0.43619403,
   0.44467586,  0.33706617, -0.06692744,  0.09349959,  0.04350894,
   0.3021839 ,  0.9402844 ,  0.99566454,  0.2951485 ,  0.53058153,
   0.00044867,  0.6247999 ,  0.76713765, -0.01949725,  0.04259383,
   0.39452827,  0.06870381,  1.490739  ,  0.79490113,  0.02340494,
   0.5836114 , -0.684894  , -0.13121423,  0.56909686,  0.6628713 ,
   0.20885336,  0.3703347 ,  0.3746184 ,  1.0253565 ],
 [ 0.2476496 , -1.1830713 ,  0.09393036,  0.02757662, -0.2516676 ,
   0.46831024, -0.36795676, -0.7927108 ,  0.23364225,  0.68498516,
   1.2956853 ,  0.5280917 , -0.3324207 ,  1.068886  ,  0.29575813,
   0.19287781,  0.00129361,  0.44504833,  0.01352877, -0.2082311 ,
  -0.43127966, -0.36886996,  0.56416357, -0.02034316,  0.52935994,
  -0.08011661, -0.07056868,  0.27192047,  0.32338393, -0.04038164,
  -0.11363484,  0.26201457, -0.14840972,  0.5715719 ,  0.54704833,
   0.24596137,  1.0955391 ,  0.49023873, -0.3096531 ,  0.7089632 ,
  -0.71160746, -0.74074376, -0.08903196, -0.5124341 , -1.4725144 ,
   0.12333258, -0.47708824,  0.02428715, -1.4324098 ,  0.10065243,
   0.08861344,  0.17741834,  0.19695845,  0.31427035, -0.09475937,
  -0.2462119 ,  0.5757003 , -0.5822504 , -0.00761436, -0.1289582 ,
   1.0090995 , -0.7834633 ,  0.09692495,  0.02374823],
 [ 0.21757035,  0.21861233,  0.07136381,  0.1711281 ,  0.09325549,
  -0.22200441,  0.92606467, -0.3609798 ,  0.51906586,  0.5068394 ,
   0.6093019 ,  0.61704576,  0.49322313,  0.09722687, -0.3387717 ,
   0.09644472,  0.6036942 ,  0.62089276, -0.5914612 ,  0.23757851,
   0.49952087,  0.8945509 , -0.04784663,  0.3712639 ,  2.0960858 ,
  -0.37164694,  0.78440034,  0.8407189 ,  0.02839513,  0.40273342,
   0.14674489,  1.1344163 , -0.13338092,  0.14786799, -0.32043543,
  -0.27922642,  0.18857655, -0.0699132 ,  0.10517389,  0.49716264,
   0.5194736 ,  0.5404929 ,  0.61691445,  0.2528117 , -0.22636263,
  -0.19493897, -0.0236054 ,  0.33179432, -0.09451962, -0.1143244 ,
  -0.19359457,  0.28206614, -0.00467482,  0.3497232 ,  0.77549213,
   1.2391921 , -0.5000266 , -0.04599769, -0.41510433,  0.3868045 ,
  -0.15950222, -0.08949078,  0.29924   ,  0.1437198 ],
 [-0.0468705 , -0.6809727 , -0.5261507 ,  0.263555  ,  0.0860653 ,
  -0.0915275 , -0.40309268, -0.8443861 ,  0.87670225,  0.9390062 ,
   3.187145  ,  1.007602  ,  0.77067006, -0.25241205, -0.27549914,
   0.54968446, -0.10968679,  2.0341635 ,  0.43800473,  0.25060213,
   0.35301417, -0.11688346,  0.19595066, -0.4913493 ,  0.2680218 ,
  -0.09272427,  0.37675023, -0.6887753 ,  0.5582485 ,  0.76520663,
   0.1376793 ,  0.49184042,  0.4457588 , -1.4855663 ,  0.22248803,
   0.0395303 ,  1.3882222 ,  0.19598731, -0.91564876,  0.3972456 ,
  -0.69208187, -1.0420058 ,  0.57295763,  0.02667695, -0.8863977 ,
  -0.08020163,  0.11987254,  1.8159512 , -1.089826  , -0.00473037,
   0.14202988, -1.0868374 ,  1.8772454 ,  1.5479842 , -0.04111576,
  -0.14580123, -0.57519996, -0.7736313 ,  0.01984658,  0.3264351 ,
   1.0022235 ,  0.14807846, -0.04413529,  0.82857466],
 [-0.265674  , -0.17131472, -0.0257187 ,  0.02198399,  0.1615994 ,
   0.264347  ,  0.32433543,  0.04660306, -0.3885293 , -0.05792128,
  -0.0183691 ,  0.33674154, -0.40899447,  0.1389838 , -0.23295757,
  -0.06687367,  0.48034212, -0.55731314, -0.6869061 , -0.15568012,
  -0.01286722, -0.39914873, -0.506555  , -0.30372444, -0.08445662,
  -0.3522656 ,  0.06412769, -0.02837609,  0.19273117, -0.20042829,
   0.22701076, -0.31866714,  0.5072319 ,  0.6407038 , -0.51400393,
  -0.0816785 , -0.14537854,  0.39045897,  0.00481668,  0.19257362,
  -0.53760046, -0.1039927 , -0.03464213,  0.3623632 , -0.76656824,
  -0.31778198, -0.490125  , -0.2592671 , -0.4619059 , -0.6361386 ,
  -0.5157159 ,  0.02554864,  0.5190595 ,  0.25200886, -0.08895317,
   0.5121858 ,  0.5886946 ,  0.09863628, -0.40917754, -0.24941678,
   0.86545694, -0.31919435,  0.19024405,  0.07407721],
 [ 0.24866852, -0.02467575, -0.0622928 ,  0.60287297, -0.40121865,
  -0.04745087,  0.05379629,  0.41094223,  0.46200576, -0.4805995 ,
   0.34724405, -0.24553895,  0.18582246,  0.26895443, -0.05153331,
   0.03872312,  0.02169278,  0.08624893,  0.12995648, -0.56396013,
   0.16585672,  0.17189322,  0.12878905, -0.28316075,  0.36796105,
   0.03292594, -0.24865644,  0.22719942,  0.07625581, -0.44261298,
   0.23052484, -0.02272124, -0.02345758,  0.2039147 ,  0.32318717,
  -0.17207462,  0.23787256, -0.14354798,  0.16807973,  0.1763627 ,
  -0.2865312 , -0.05328189,  0.29871327,  0.14482948, -0.4196577 ,
   0.15018737,  0.36658594,  0.09585723,  0.02906338, -0.07829031,
  -0.27018914,  0.31197888,  0.11104421,  0.60179234, -0.08796411,
  -0.11720363,  0.7303606 , -0.17663625, -0.1965223 , -0.17794864,
   0.3493209 ,  0.17100516, -0.02212376, -0.03153528],
 [-0.08262836, -0.13127461,  0.41743988,  0.21688034, -0.3283567 ,
  -0.12821203,  0.24158572,  0.35010916, -0.25382495,  0.44023997,
   0.07474002,  0.36500674,  0.25703534,  0.50588906,  0.18862274,
  -0.9046815 ,  0.39220348,  0.23724262,  0.47765756,  0.10322703,
  -0.24956828,  0.2094136 ,  0.13480379,  0.21690649,  0.6312123 ,
   0.17741546,  0.1770385 , -0.8921045 , -0.5144981 ,  0.5727092 ,
  -0.19137514,  0.22289318, -0.28996244, -0.17145242,  0.29739308,
  -0.11826695, -0.02399857,  0.39263397, -0.0252096 ,  0.23418212,
  -0.20971608, -1.0450717 ,  0.126786  ,  0.3111845 , -0.23405777,
   0.03997282,  0.27716947,  0.29974103, -0.3370457 , -0.05670923,
   0.022502  ,  0.64381546, -0.13651676,  0.49775147,  0.08019117,
   0.7886865 ,  0.46809494, -0.04379825,  0.47907597, -0.27649862,
  -0.3493105 , -0.3199417 , -0.61944896, -0.1357949 ],
 [-0.50247794, -0.48514852, -0.52444947, -0.10746925,  0.86187136,
  -0.4799068 ,  0.8003279 , -0.16588332,  0.55772364, -0.00664709,
   0.98165774, -0.10253827,  0.44264567,  0.02799409, -0.04033372,
  -0.13084216,  0.39706996,  0.47310406,  0.16694412,  0.17583942,
  -0.41215262,  0.44297904,  0.91365254,  0.22251612, -0.4212896 ,
  -0.39294556,  0.15484278, -1.0406723 ,  0.49881205, -0.31403205,
   0.7169659 , -0.09795258,  0.71820307,  0.4756222 , -0.24821696,
   0.14878449,  0.41772923, -0.4843437 , -1.0409333 , -0.6914664 ,
  -1.0910981 ,  0.236578  ,  0.06138638, -0.843271  ,  0.3066029 ,
  -0.10983342,  0.25002587,  0.5897196 , -0.18340778, -0.11976646,
   0.21558355,  0.01029072,  0.20274797,  1.9417136 ,  0.5849594 ,
  -0.6525352 , -0.28857926, -0.14851706, -0.5335119 , -0.08857103,
   0.11908081, -0.8434434 ,  0.12310313,  0.20776765]])
    b3 = jnp.array([-0.84330314, -0.13662359,  0.05954956, -0.25311974, -0.74522763,
 -0.35909855, -0.86706483, -0.82602507, -0.7197158 , -0.10119968,
 -0.91804147, -0.76997226,  0.53604776, -1.5167543 , -0.31340045,
 -0.36934024, -1.4625914 , -1.1463696 , -0.09428879, -0.46677637,
  0.02049542, -0.6308798 , -0.90632623, -0.21746626, -0.41191685,
 -0.44823477, -0.77672565, -0.83292794, -0.16833971, -0.5666128 ,
 -0.40781748, -1.007936  , -0.17965966, -0.25081655, -0.3861154 ,
 -0.33062935, -0.35253274, -0.23269184, -0.7940749 , -1.671918  ,
 -0.9192582 , -0.39842126, -0.5624515 , -0.45877975, -0.14354382,
 -0.18888251, -1.1053859 , -0.92445946, -0.31      , -0.11435425,
  0.8744508 , -1.8332474 , -1.2229084 , -0.49509755, -0.4490817 ,
 -0.06985246, -1.6351303 , -0.5631838 , -1.2994795 , -0.8192315 ,
 -2.0291488 , -0.08433966, -1.6142024 , -1.0383419 ])
    W4 = jnp.array([[-0.17435744,  0.2560676 , -0.13906659, -0.06744406, -0.11796279,
   0.55773574, -0.4085432 ,  0.5136023 ,  0.76104826, -0.31901363,
  -0.4823193 ,  0.80190825, -0.12694268, -0.80157024,  0.30502218,
   0.66584563,  0.7509659 ,  0.46105886,  0.29897648, -0.41168988,
  -0.06755753,  0.9359036 ,  0.4645031 , -0.17962028,  0.80009764,
  -0.0254063 ,  0.20822433,  0.52449566,  0.898503  ,  0.58091974,
   0.742062  ,  0.6499796 , -0.24293306,  0.93608123,  0.7053605 ,
  -0.38941795, -0.15664154, -0.6249352 ,  0.01521295, -0.3356753 ,
  -0.39354938, -0.614982  ,  1.0738274 , -0.00673604, -0.27497822,
   0.6470649 ,  0.4249674 , -0.4781515 , -0.04178052, -0.24375024,
  -0.01144503,  0.33728626,  0.52417266,  0.9362572 ,  0.22471322,
   0.14561707,  0.9663059 , -0.00119102,  0.9896421 ,  1.2481289 ,
   0.29707888, -0.06858756,  0.1569207 ,  0.58391225],
 [ 0.2112626 ,  0.28056827, -0.12502897, -0.20092817,  0.10066283,
   0.5886061 ,  0.140735  ,  0.75756645,  0.2705986 ,  0.15784785,
  -0.00696543,  0.9769861 ,  0.00686458,  0.47325313,  0.48361152,
   0.41105184,  0.35895833,  0.3702749 ,  0.24350944,  0.12483874,
   0.43614858,  0.70915973,  0.5030731 , -0.01774832,  0.17753844,
   0.7703794 ,  0.29291028,  1.2950621 ,  1.3765808 , -0.05946794,
   0.97702336,  1.203691  , -0.11889178,  1.0362598 ,  0.9826964 ,
   0.04913158, -0.00074165,  0.5318128 , -0.15762669,  0.17472164,
   0.40936613,  0.7652587 ,  1.1143363 ,  0.12750362,  0.4277639 ,
   0.43853706,  0.27967456, -0.0891344 , -0.0737123 , -0.06282333,
   0.5607391 ,  0.11067957,  1.5532374 ,  1.0103844 ,  0.7971037 ,
  -0.01494463, -0.4106458 ,  0.10640657,  0.41530037,  1.1875129 ,
  -0.2870486 , -0.09151812, -0.31675318, -0.01262442],
 [ 0.00164844,  0.29977658,  0.2238051 ,  0.23605399, -0.2389128 ,
  -0.49274805,  0.02139238, -0.00693188, -1.1984192 , -0.01178398,
   0.0523454 , -0.24103472,  0.04032056,  0.02337342, -0.08216573,
  -0.22147202, -0.07309049,  0.01916102, -0.21537912, -0.01315439,
   0.02717398, -0.01095056,  0.08691224, -0.17410213, -1.2263745 ,
  -0.1854325 , -0.2038361 ,  0.03747683,  0.01337422, -0.7547253 ,
  -1.1912013 , -0.8195191 ,  0.02608377, -0.9585788 ,  0.50999355,
  -0.3522339 , -0.1375655 , -0.02006047, -0.3092852 , -0.00276468,
  -0.00300324,  0.00818634,  0.45635507, -0.71018714,  0.00438679,
   0.03631264, -0.24631731, -0.16125055,  0.15559983, -0.00075269,
  -0.165063  , -0.6421671 , -0.4120071 , -0.529366  , -0.01044748,
  -0.19133136, -0.20877114, -0.07809363, -0.27015787, -0.12130838,
  -0.21542196, -0.25995868, -0.15869232, -0.17773758],
 [-0.39657745,  0.10639371,  0.03483151,  0.8320888 , -0.10216019,
   0.14781027, -0.22819321,  0.2070679 ,  0.3038933 , -0.19306852,
   0.27555737,  0.566688  , -0.3380215 ,  0.03018465,  0.18597837,
   0.01636096,  0.09279341,  1.1277484 ,  0.23031078,  0.21780412,
  -0.27315414,  1.3351752 ,  0.07265903,  0.07366141,  0.41040245,
  -0.3177086 ,  0.66271967,  1.2738541 ,  0.5279968 ,  0.36923385,
   0.88100654,  0.5302919 , -0.13891913,  0.73125196,  0.11642517,
   0.1260935 ,  0.13806283,  0.09404356,  0.10396639, -0.21290363,
  -0.40486503, -0.04449629,  1.4005651 , -0.04227027,  0.7267301 ,
  -0.00860116,  0.09743364, -0.441879  ,  0.02925461,  0.16286407,
   0.9246631 ,  0.6982155 ,  1.271213  ,  0.6153844 ,  1.498888  ,
   0.5233014 , -0.25012308, -0.20736828,  0.99223065,  0.82226914,
   0.12490853,  0.11913192,  0.12292761,  0.6083314 ],
 [ 0.17142135, -0.14428875,  0.15420048,  0.47347292,  0.1949119 ,
   0.31764325,  0.04619056,  0.43058786,  1.3750527 ,  0.10390875,
   0.49713087,  0.9262061 ,  0.21030962,  0.3171692 ,  0.55732596,
   0.21184109, -0.01665494,  0.7013701 ,  0.288783  ,  0.45314625,
   0.17224155,  0.2492129 ,  0.45983908,  0.3235863 ,  0.28393993,
   0.07745067,  0.5931084 ,  0.3468036 ,  0.21321484,  1.2160678 ,
   0.53977054,  0.5767342 ,  0.12799677,  0.7470045 ,  0.33739385,
  -0.12435402,  0.50362676,  0.32573757,  0.3998839 ,  0.11879675,
   0.21471915,  0.31562376,  0.7812236 ,  0.5841966 ,  0.27527553,
   0.8541648 ,  0.44183254,  0.2346797 ,  0.29734758,  0.2569586 ,
   0.6808453 ,  0.4457548 ,  0.9876716 ,  0.66686463, -0.09018741,
   0.7892979 , -0.0062933 ,  0.51209307,  0.0404197 ,  0.5310416 ,
   0.41924834,  0.32275212,  0.7000228 ,  1.7023919 ],
 [ 0.18246503,  0.0079954 ,  0.22203296,  0.05126004, -0.16191608,
   0.1123796 ,  0.21427946,  0.17542516, -0.019676  , -0.01664854,
  -0.06611189, -0.27316138, -0.08492808,  0.44498825,  0.2576566 ,
   0.45597017, -0.31464535, -0.10611998,  0.24814379, -0.4049142 ,
  -0.04753926,  0.26966718, -0.03152115, -0.2853227 , -0.04538109,
   0.25508276,  0.37739915,  0.7886179 , -0.02194018,  0.5259587 ,
  -0.23282836,  0.28947443, -0.05612119,  0.20231973,  0.7169592 ,
  -0.434747  , -0.04626771,  0.00791676, -0.30277976, -0.03839643,
   0.09584651, -0.23938642,  0.01607541, -0.11197477,  0.06074925,
   0.33928266,  0.60240597,  0.09047242,  0.22601868,  0.23956166,
   0.22601712,  0.35290828,  0.28867242,  0.21566585,  0.71275616,
   0.33827883,  0.02779534, -0.15067168, -0.09950949,  0.08453873,
   0.01591027, -0.18227586,  0.18406528, -0.3530773 ],
 [ 0.04736573,  0.43011856,  0.4505371 ,  0.38203093, -0.39385822,
  -0.04382699,  0.09168955,  0.10186937,  0.30935785, -0.03159518,
   0.24564882,  1.0326453 ,  0.1713736 ,  0.16979167,  0.21447834,
  -0.43852115, -0.17282158,  0.43271947, -0.10592929, -0.17337474,
   0.01898316,  0.42449227,  0.14262383, -0.11537051, -0.12052484,
  -0.09381047,  0.50622886,  0.38189435,  0.18900464,  0.03417029,
  -0.13301735, -0.05963215,  0.0608681 , -0.13194956, -0.7294069 ,
   0.02835041, -0.60663086, -0.0132683 ,  0.04550776,  0.09149665,
   0.03083929, -0.09923624,  0.44646758, -0.16801931, -0.22991821,
   0.16122727, -0.14828952, -0.2339647 ,  0.55231255,  0.00942428,
   0.38434473, -0.80550414,  0.20574653,  0.37979132, -0.32938677,
   0.381353  , -0.09314855, -0.25933456,  0.16807385, -0.54253983,
  -0.15537323, -0.34380665, -0.22647889,  0.72479534],
 [ 0.15802212,  0.53944904, -0.05601375,  0.5590941 ,  0.09947168,
   1.1708565 , -0.23166282,  0.47696903,  0.23612691, -0.35128102,
   0.02260981,  1.6234766 ,  0.08037254, -0.4030181 , -0.49801245,
   0.39792326, -0.04813378,  0.5642017 ,  0.19678535, -0.33105984,
   0.08301819,  1.4487606 ,  0.03320906, -0.28604615,  0.7299259 ,
  -0.11707637,  0.58356017,  0.1179079 ,  0.34653008,  0.51296794,
   0.7660368 ,  0.27030388,  0.41361335,  1.1838812 ,  1.3140242 ,
  -0.1279393 , -0.52056843, -0.27063593,  0.07719272,  0.02269726,
   0.2562503 , -0.0766204 ,  1.4330186 , -0.12944381, -0.0812574 ,
  -0.16818894,  0.87652034, -0.4533142 , -0.05795236,  0.14157507,
   1.185568  ,  0.7068751 ,  0.05530085,  1.1047752 , -0.4571478 ,
   0.44841883,  0.3641822 ,  0.08792184, -0.82532257, -0.05415295,
   0.21542294, -0.04684991,  0.13993172,  0.37262565],
 [-0.11343397,  0.15714747,  0.08886995,  0.19427364, -0.05700147,
   0.28563735,  0.2291902 ,  0.87455463,  0.53024507,  0.11500207,
   0.6256201 ,  1.5739329 ,  0.08047094,  0.5644694 ,  0.62483597,
   0.3287098 , -0.5664742 ,  0.33943313,  0.33847776,  0.68421024,
   0.18967366,  0.3396454 ,  0.49427834,  0.23396207,  0.64111006,
   0.00737969,  0.5316976 ,  1.3530256 ,  1.0153795 ,  1.0292307 ,
   0.4453547 ,  0.9706451 ,  0.3589019 ,  0.47794023,  1.156601  ,
   0.06650074,  0.2882867 ,  0.39439678,  0.06887464,  0.1967041 ,
   0.10577431,  0.10043218,  1.4745611 , -0.02390818,  0.6814281 ,
   0.6831323 ,  0.3445972 , -0.16614468,  0.1149559 ,  0.16597153,
   0.43743992,  0.2169353 ,  0.9048311 ,  0.68178624,  0.4616121 ,
   0.5452791 ,  0.5581654 , -0.02464153,  0.6905418 ,  0.5854745 ,
   0.34063503,  0.22090875,  0.17371415,  0.37298846],
 [-0.15528032,  0.02394989,  0.40427777,  0.4088576 ,  0.25222862,
  -0.68589646,  0.10131186,  0.5387161 , -0.28511524, -0.06765554,
  -0.23071034,  0.34150392, -0.44296485, -0.42134905,  0.07996506,
  -0.29596394, -0.15513279,  0.16410631, -0.10064441,  0.00122252,
  -0.40909582,  0.3526753 ,  0.06768243, -0.27690458, -0.5586497 ,
  -0.14821473,  0.5216357 ,  0.6363653 ,  0.19839579,  0.40081197,
   0.00731045, -0.25308773,  0.2433759 , -0.49688378,  1.0124333 ,
   0.33080032,  0.17608877, -0.10308906,  0.20479542,  0.11360938,
   0.13397859,  0.03852504,  0.4169636 , -0.16109438,  0.32356274,
  -0.20520945,  0.10568271, -0.40986013,  0.37591013,  0.14286256,
   0.5815916 ,  0.6890073 ,  1.177556  , -0.70801246, -0.89017963,
   0.30643818, -0.1434286 , -0.25016573, -0.9543108 , -0.60512674,
  -0.14393972,  0.14232674,  0.08833261,  0.40002725],
 [-0.20076156, -0.5605089 ,  0.266268  ,  0.8695525 ,  0.0767634 ,
  -1.095082  ,  0.2498596 ,  0.47021738, -0.3941648 , -0.1347457 ,
   0.06017122, -0.11233462,  0.05541004,  0.44814903,  0.48013726,
  -0.05108113, -0.9591014 ,  1.1257687 ,  0.03598738,  0.39860582,
   0.08456972,  0.17237951,  0.05261846, -0.03281925, -0.75741076,
   0.10830962,  0.3121523 ,  0.65346897,  0.4015635 ,  0.8650479 ,
  -0.7940589 ,  0.40277   , -0.14587021, -0.22181042,  1.0972778 ,
   0.24653652, -0.02502796,  0.01113185, -0.07148144, -0.3315353 ,
  -0.3450653 , -0.38434762, -1.0835743 , -0.35812882,  0.4208604 ,
   0.37576926,  0.44138753,  0.0757482 ,  0.2788432 ,  0.25977856,
   0.00433437, -0.70799345,  0.78044593, -0.36579123,  0.09443594,
   0.5221108 ,  0.09969998,  0.02850926, -0.19997607,  0.22226495,
   0.13786882,  0.23035765, -0.09855703,  0.7001365 ],
 [ 0.15706934, -0.33533266,  0.43828565,  0.5535059 , -0.01470156,
   0.11442224,  0.230303  ,  0.20989761,  0.3124272 ,  0.11915066,
  -0.05470616,  0.09285647,  0.09749797, -0.08227016,  0.71801597,
  -0.0670019 ,  0.26698568,  0.07449261, -0.00658054,  0.12742469,
   0.10926112, -0.16305956,  0.3969273 , -0.01996404,  0.31249273,
   0.12049279,  0.09990953,  0.96056736,  0.49758694,  1.0749362 ,
   0.5045196 ,  0.7002342 ,  0.24057999,  0.8261717 ,  0.8240868 ,
  -0.15022014, -0.00450222,  0.18883833,  0.09220994,  0.20543973,
   0.19824156,  0.20011774,  0.6470177 , -0.2765322 ,  0.25872728,
   0.64555305,  0.17298014, -0.11140286,  0.34959596,  0.14558072,
   0.0621701 ,  0.56974375,  0.40265518,  0.97146076,  0.43840054,
   0.3074057 ,  0.20342395, -0.00071268,  1.480483  ,  1.4960324 ,
   0.07753248,  0.08521968,  0.05609415,  0.9928    ],
 [-1.0970436 ,  0.18353884, -0.95873374, -0.2067866 , -0.64929265,
  -0.5976239 , -0.6496258 , -0.77159154, -0.40847284, -1.3116335 ,
  -0.92731416, -1.0878437 , -0.8865759 ,  0.23683281, -1.3937187 ,
  -0.64372915, -0.04887867, -0.7459205 , -0.58265615, -0.6087648 ,
  -0.7542909 , -0.6444963 , -0.6593964 , -0.48921213, -0.48724258,
  -0.94233185, -0.9234004 , -1.052683  , -0.94180155, -1.0947248 ,
  -0.57456845, -0.5045359 , -0.39385334, -0.44337347, -0.7720271 ,
  -0.22564599, -0.67052114, -0.85504067, -0.14557584, -0.7913406 ,
  -0.9498463 , -0.7895858 , -0.63845897, -1.1558532 , -0.61824524,
  -0.591393  , -1.1364555 , -0.82310706, -1.1255233 , -1.5987328 ,
  -0.36729982, -0.66006637, -0.02573186, -0.45411548, -0.2510405 ,
  -1.0937765 , -0.15004289, -1.1777449 , -0.18115653, -0.29793876,
  -0.42611113, -0.6026353 , -1.4917213 ,  0.07720542],
 [ 0.528231  , -1.0642985 ,  0.75744337,  0.09612468,  0.11666524,
   0.2648853 ,  0.16978323,  0.6235486 ,  0.7599741 ,  0.2651546 ,
   0.18185097,  0.8950805 ,  0.5935781 ,  2.2388637 ,  1.0159465 ,
   0.16835429, -0.38892904,  0.27094382,  0.21605842,  0.26900667,
   0.8848689 , -0.11268411,  0.46763828,  0.21536954, -0.05376821,
   0.39942503, -0.45606926,  0.7565961 ,  0.28327778, -0.12130179,
   0.41525123,  0.74547344,  0.2607692 ,  0.71347433,  0.1765766 ,
   0.45263934,  0.2692495 ,  0.39645195,  0.5659226 ,  0.68928146,
   0.3404469 ,  0.78871274,  0.17297263,  0.14013243,  0.4558743 ,
   0.4914698 ,  0.63821936,  0.34542233,  0.3517956 ,  0.12921268,
   0.05151955, -0.17033729,  0.77931875,  1.049215  ,  0.05777399,
  -0.02354304, -0.4688707 ,  0.45759815,  0.8928765 ,  0.76408637,
  -0.13003013,  0.43260366,  0.13411862, -0.02872157],
 [-0.10990713,  0.37240633,  0.45288566,  1.2236402 ,  0.1981998 ,
   0.09119845, -0.07280998,  0.41875637,  0.5447254 ,  0.11175664,
   0.19736317,  1.1089438 ,  0.04522393, -0.25186548,  0.60705024,
   0.15840967,  0.3876477 ,  1.3042009 ,  0.01059236,  0.06745408,
   0.29095048,  1.457708  ,  0.3964273 ,  0.01688807,  0.39878726,
   0.1554587 ,  1.1846117 ,  1.0003828 ,  0.8716489 ,  0.5756095 ,
   0.8538285 ,  1.3156558 , -0.2808215 ,  1.1987244 ,  1.2924951 ,
   0.3499357 ,  0.01109181, -0.01216673,  0.50731206, -0.10834086,
  -0.03917747,  0.11618411,  1.5924475 , -0.08497699,  0.6850082 ,
   0.26830345,  0.40035108,  0.17732619,  0.4628322 ,  0.36396396,
   0.96466756,  0.4774711 ,  0.86280507,  1.2005736 ,  2.3513227 ,
   0.9490188 ,  0.1926536 ,  0.17869721,  1.3311644 ,  0.7285129 ,
  -0.08658698,  0.21528508,  0.564017  ,  1.0910573 ],
 [-0.16213867,  0.16157699,  0.15912664,  0.7348098 , -0.00684676,
   0.91749084,  0.08190465,  1.2795205 ,  0.65099436,  0.14567016,
   0.37145147,  1.182806  , -0.05404802, -0.37634802,  0.92622066,
   0.14471486,  0.32369632,  1.0083894 ,  0.18893826,  0.54400766,
  -0.06108473,  0.71475005,  0.36617073, -0.13298427,  0.5818503 ,
   0.01445548,  0.58395517,  1.8274078 ,  1.5796738 ,  0.8025039 ,
   0.8161274 ,  1.2715541 ,  0.05625692,  1.4538766 ,  0.73540735,
  -0.03767174,  0.2921268 ,  0.24640845,  0.17573608, -0.22225396,
   0.00189406,  0.09811299,  1.8198175 ,  0.593982  ,  0.72803485,
   0.51297146,  0.10136363, -0.1583864 ,  0.17966251,  0.07989115,
   1.0789436 ,  0.5321809 ,  1.7394552 ,  1.2329601 ,  1.1706539 ,
   0.50837785,  0.18086866, -0.16419509,  1.3775104 ,  1.331254  ,
   0.21304412,  0.4234344 ,  0.04746805,  1.2419074 ],
 [-0.2997825 ,  0.10960245, -0.6811546 ,  0.6292901 ,  0.26218975,
   0.21932285, -0.24750611,  0.9126786 ,  0.3155614 , -0.4653528 ,
   0.1740319 ,  1.3764205 , -0.67354906,  0.0951829 , -0.37811694,
   0.30566663, -0.04026518,  0.6066816 ,  0.40887585,  0.22916684,
  -0.65796024,  0.41695026, -0.7589838 ,  0.16180436,  0.64556193,
  -0.04996956,  0.5103772 ,  0.0286494 ,  0.15951625,  0.44052982,
  -0.37586907,  0.47303155,  0.53031737,  0.01495402,  0.8637831 ,
   0.32088867, -0.29879814, -0.42085597,  0.35387015, -0.15362692,
  -0.2449487 , -0.5794984 ,  0.3915978 ,  0.75738674,  0.24707492,
  -0.07131328, -0.02179079,  0.4138852 , -0.74731195, -1.2846842 ,
   0.60618114,  0.33586586,  0.21014297,  0.19049664, -1.4124402 ,
   0.36425367,  0.80599076, -0.30108732,  0.61971843,  0.49315077,
   0.17886515,  0.43683416, -0.5054079 ,  0.28127128],
 [ 0.3878293 ,  0.31828344,  0.67460716,  0.44997472, -0.07050914,
  -0.05732906,  0.1361667 ,  0.09239668,  0.17975849,  0.5814175 ,
  -0.03952171,  0.24199426,  0.60765165, -0.19904579,  0.67079216,
  -0.17093366,  0.21502164,  0.41618422, -0.23077622, -0.09155339,
   0.5985336 ,  0.69093144,  0.7881272 , -0.30193722, -0.16347314,
   0.09607443,  0.29368928,  0.93178946,  0.4329495 ,  0.55036265,
   0.54412556,  1.08355   ,  0.09098181,  0.77900296,  0.4910239 ,
  -0.3744301 , -0.31459627,  0.07474987, -0.15522927,  0.03524664,
   0.16433781, -0.01264352,  1.6405011 , -0.43275553,  0.5730818 ,
   1.0513372 ,  0.37298492,  0.00441985,  0.6689241 ,  0.5097872 ,
   0.42742664, -0.15812919,  0.4677612 ,  0.7954123 ,  1.5714855 ,
   0.5993526 ,  0.47835055,  0.5233078 ,  1.4492373 ,  1.0624695 ,
   0.10159575, -0.15701005,  0.78322315,  0.93968695],
 [ 0.47754532, -0.12499071,  0.6339938 ,  0.8316169 ,  0.32238367,
  -0.06617182,  0.6336119 ,  0.8274274 ,  0.22799346,  0.64836276,
   0.05280832,  0.55183065,  0.50364655,  0.08031431,  0.64091927,
  -0.1754677 ,  0.18994315,  1.0964255 ,  0.22105609,  0.5386972 ,
   0.80870247, -0.02638605,  0.63858926,  0.8104832 , -0.04020121,
   0.13656652,  0.8710447 ,  0.80229694,  1.014492  ,  0.48294532,
  -0.4833362 ,  0.52264357,  0.8479951 ,  0.14025973,  1.2684472 ,
   0.4142156 ,  0.5376432 ,  0.8625861 , -0.1917287 ,  0.7193931 ,
   0.40052703,  0.74353147,  1.0080721 ,  0.0765567 ,  0.78492767,
   0.51705503, -0.36910263,  0.31382614,  0.81495637,  0.72112083,
   1.1273777 ,  1.2163453 ,  1.0797642 ,  0.03692789,  1.127049  ,
   0.683743  ,  0.20705041,  0.56807894,  1.5420496 ,  2.1871061 ,
   0.6183358 ,  0.632953  ,  0.6106691 ,  0.47498623],
 [ 0.27054808,  0.59219724,  0.64185923,  1.6145691 ,  0.11723683,
   0.20253018,  0.06735477,  1.1614623 ,  0.39238432,  0.3566796 ,
   0.18264721,  1.1409068 ,  0.38436648,  0.16056734,  0.6642338 ,
  -0.17225073, -0.64181274,  1.6673789 , -0.14966641,  0.31428078,
   0.75062096,  1.050056  ,  0.6862128 ,  0.6088783 ,  0.04320934,
   0.11747347,  1.3578522 ,  0.7481482 ,  0.91153926,  0.54492193,
   0.05682153,  0.71963996,  0.4575427 ,  0.52595687,  1.5663388 ,
   0.12930372,  0.7206961 , -0.03835167, -0.11645721,  0.2481335 ,
   0.04475102,  0.10808497,  1.3959111 , -0.1141387 ,  0.49447277,
   0.6530747 ,  0.15299903,  0.47239068,  0.8461879 ,  0.70287126,
   1.8422925 ,  0.7505929 ,  0.7892046 ,  0.6473137 ,  1.5357926 ,
   1.2204478 , -0.15088001,  0.36851954,  1.3069811 ,  1.4817734 ,
   0.5060361 , -0.01432292,  0.8313552 ,  1.2307931 ],
 [-0.18673962,  0.21681334,  0.00709222,  0.32398942,  0.1809794 ,
  -0.13998842,  0.14907643,  0.22954333,  0.48228958, -0.18119194,
   0.00852791,  0.49840674, -0.525486  ,  0.0667974 ,  0.5135665 ,
   0.00636144,  0.37444752,  0.05597463, -0.23472187, -0.08769894,
  -0.45227093,  0.68900496, -0.07219335, -0.07440463,  0.0673646 ,
   0.01299342,  0.40823692,  1.1553807 ,  1.0283586 ,  0.60275996,
   0.62190396,  0.87032276, -0.6029445 ,  0.6576832 ,  0.21508898,
   0.14392166,  0.33513916,  0.17413086,  0.33387357, -0.03003042,
  -0.2266555 ,  0.05852333,  0.49349684, -0.1905219 ,  0.21126282,
   0.23364809,  0.12936215,  0.40502724,  0.16488612, -0.22854759,
   0.03797899,  0.23568481,  0.2806535 ,  0.6661496 ,  0.5118724 ,
   0.19678701, -0.15502849, -0.27658436,  0.5648727 ,  0.35241613,
  -0.34687862, -0.04505002, -0.31720176,  1.0296242 ],
 [ 0.41690478,  0.7969691 ,  0.27753013,  0.6527486 ,  0.0621094 ,
   0.3669365 ,  0.07210126,  1.0032477 ,  0.6380457 ,  0.30491292,
  -0.00965394,  1.6636999 ,  0.2793715 , -0.1696683 ,  0.9621856 ,
   0.02864284, -0.13736986,  1.0748222 , -0.20906068,  0.02409422,
   0.2755773 ,  1.6633743 ,  0.87939084,  0.17048158,  0.6678832 ,
   0.16896762,  1.4126318 ,  1.0354829 ,  1.2520725 ,  1.0624775 ,
   1.1854793 ,  1.4216572 , -0.1545859 ,  0.8865712 ,  0.90914243,
   0.2965703 ,  0.10652915,  0.19494306,  0.3700358 ,  0.1144111 ,
   0.09971527,  0.04323031,  2.1387436 , -0.2055001 ,  0.37153137,
   1.1195142 ,  0.4968956 ,  0.23978841,  0.5162826 ,  0.57107174,
   1.1644807 ,  1.0080843 ,  1.6592747 ,  1.1684883 ,  2.446996  ,
   1.1747555 ,  0.10281071,  0.32775965,  1.4461085 ,  0.6652837 ,
  -0.2717409 , -0.19061898,  0.44107428,  1.1749357 ],
 [ 0.12955624,  0.25451362,  0.41242936,  1.4803102 , -0.11593592,
   0.4132936 ,  0.31218877,  0.6399999 ,  0.0146444 ,  0.08815112,
   0.2936176 ,  1.2786958 ,  0.06689207,  0.30904198,  0.3576735 ,
  -0.00637455, -0.00460969,  1.3978968 ,  0.13039365,  0.0157094 ,
   0.44909984,  0.9202358 ,  0.43143487,  0.10982677,  0.10253467,
   0.26073676,  1.1871293 ,  0.7181573 ,  0.9598097 ,  0.08175561,
   0.09250035,  0.06602495, -0.0120392 ,  0.6118931 ,  1.2649812 ,
  -0.09757035,  0.22889036,  0.08428424, -0.5642446 ,  0.03012975,
  -0.02095522,  0.10055605,  0.48709425,  0.25592202,  0.7070522 ,
   0.2695757 , -0.03228009,  0.35940212,  0.23448482,  0.246341  ,
   1.0842632 , -0.5003734 ,  1.7145697 ,  0.13884783,  0.9089956 ,
   1.0441741 ,  0.38631973,  0.37717918,  0.5997877 ,  0.8406676 ,
   0.3380861 , -0.03288677,  0.6471139 ,  1.3631557 ],
 [ 0.2309965 , -0.01621264,  0.22731028,  0.03433854, -0.19935735,
   0.4384631 ,  0.5089467 ,  0.7091204 ,  0.42856005,  0.23791279,
  -0.27751032,  1.0672209 ,  0.17969465,  0.11599255,  0.14221178,
  -0.49563262,  0.2692284 ,  0.8553103 , -0.18496934,  0.20357034,
   0.35947648,  0.40307862,  0.7795779 ,  0.33565474,  0.15245719,
   0.37063962,  0.76325977,  0.37904534,  0.0791507 ,  0.9546736 ,
   0.575889  ,  0.6556995 ,  0.1882419 ,  0.17461556,  1.2676253 ,
   0.2832064 , -0.33018762,  0.24816701,  0.34359902, -0.03938703,
   0.2251653 ,  0.1434972 ,  0.62413955, -0.13030465, -0.01098036,
   0.33107835, -0.1446377 ,  0.13543439,  0.6451458 ,  0.44910383,
   0.797257  ,  0.73008543,  0.7915112 ,  0.7606989 , -0.20379508,
   0.6975558 , -0.24530624,  0.15623282,  1.6230218 ,  1.755603  ,
  -0.13297209, -0.21020837,  0.51087064,  0.04665656],
 [ 0.00995861, -0.00898315, -0.16988398, -0.1884753 , -0.03695195,
  -0.5334081 , -0.02012255, -0.00693261, -0.40825248, -0.00306301,
  -0.6167924 , -0.08927847, -0.02873361, -0.29568315, -0.03762825,
  -0.01095501, -0.03813408,  0.00544771, -0.00625565, -0.00347292,
  -0.00997079,  0.00576658, -0.09054662, -0.00673933, -0.6681044 ,
  -0.00302545, -0.06726098, -0.12314294, -0.07848691, -1.5448259 ,
  -0.06779196, -0.06590761, -0.03712023, -0.04588341, -0.82290375,
  -0.10612377, -0.07371115, -0.01005984, -0.10046663, -0.04252754,
   0.00262303, -0.0395999 ,  0.4869048 , -0.0736794 , -0.04065446,
  -0.0936176 , -0.03054564, -0.09491501, -0.01444478,  0.00485511,
  -0.06740839, -1.7866323 , -0.69652843, -0.07682305, -0.05004143,
  -0.06240992, -0.07387065,  0.00154172, -0.06739461, -0.2586203 ,
  -0.07359818, -0.03487739, -0.07291065, -1.1598073 ],
 [ 0.28430936,  0.48803112,  0.35704464,  0.8062735 ,  0.3995328 ,
   0.6204043 ,  0.32859734,  0.69384456,  0.21636878,  0.2120927 ,
   0.29301485,  0.44530177,  0.4154311 ,  0.0155893 ,  0.8938916 ,
  -0.00562611,  0.45476264,  0.64328957,  0.09922144,  0.28013617,
   0.26287642,  0.70813733,  0.6078686 ,  0.14923006,  0.25485402,
   0.34666175,  0.90547466,  1.1959478 ,  1.3841357 ,  1.0828912 ,
   0.4926308 ,  0.8750222 ,  0.06431998,  0.82754743,  1.3243369 ,
   0.08504618,  0.33654803,  0.32426572, -0.05635164,  0.15488623,
   0.14657077, -0.02138281,  1.5365988 ,  0.00625523,  0.36499983,
   0.7167768 ,  0.47068822,  0.08218421,  0.42969945,  0.39588544,
   0.47205377,  0.86887765,  0.81058407,  0.964952  ,  1.4459457 ,
   0.71458566,  0.12882142,  0.2713723 ,  1.0182507 ,  1.0855765 ,
   0.24483563,  0.37958875,  0.2928955 ,  0.68590987],
 [ 0.22597244,  0.25388098,  0.41730207,  0.954587  , -0.05325224,
   0.5738131 , -0.15352577,  0.40241542,  0.4059006 ,  0.485573  ,
  -0.5074984 ,  0.2834748 ,  0.376371  ,  0.03950679,  0.66720414,
   0.04944849,  0.03267253,  0.9862693 , -0.23934823, -0.204615  ,
   0.13820243,  0.95789415,  0.60201687, -0.14209132,  0.76049066,
   0.18097574,  0.7356292 ,  0.66520756,  0.663678  ,  0.07979804,
   1.293694  ,  0.82519317, -0.29307437,  0.84363246,  0.72717077,
   0.06486522, -0.0431097 , -0.34987068,  0.23792143, -0.04520899,
   0.18978827,  0.03408796,  1.7593344 , -0.3536833 ,  0.3706586 ,
   0.6435893 ,  0.60338753, -0.01775294,  0.41855177,  0.47043458,
   0.31829378,  0.17132387,  0.4576051 ,  1.0794238 ,  2.1804771 ,
   0.6108171 ,  0.41727558,  0.59360313,  1.2081244 ,  1.4408855 ,
   0.04766845, -0.11385554,  0.5950764 ,  0.6290058 ],
 [ 0.20786831,  0.8413093 ,  0.05438812,  0.21892   , -0.5976307 ,
   0.8252259 ,  0.10348966,  0.6235176 ,  0.48323593,  0.322399  ,
  -0.7670571 ,  0.78206575, -0.13129678,  0.66890085,  0.19070333,
  -0.0774975 ,  0.07908856,  0.23774594, -0.41877702, -0.6213858 ,
  -0.2207893 ,  0.01589483,  0.19425288, -0.6602112 ,  0.64209074,
   0.10927393, -0.01421372,  0.51350725,  0.82338494,  0.68114877,
   0.34490773,  0.3869751 , -0.5372749 ,  0.12973472,  0.40434083,
  -0.2627855 , -0.74710673, -0.07030628, -0.03793801, -0.1256979 ,
  -0.21244878, -0.4387338 ,  1.5538719 ,  0.04497829, -0.11260685,
   0.7278954 ,  0.11481383,  0.06369747,  0.27501664, -0.08760722,
   0.3454131 ,  0.35144675,  0.3956816 ,  0.42512906,  0.85317665,
   0.01498761, -0.580317  ,  0.35790306,  0.8387392 ,  1.7411308 ,
  -0.7542822 , -0.21105017, -0.14536723,  0.48170546],
 [ 0.21838056,  0.15433306, -0.0531431 ,  0.1394913 ,  0.14485668,
  -0.06486634,  0.29559317,  0.6270926 ,  0.16293542, -0.08118265,
   0.0273988 ,  0.7671525 ,  0.34911463,  0.12561098,  0.7694049 ,
  -0.05726549,  0.10049479,  0.32454216,  0.17971231,  0.52746737,
   0.2289193 ,  0.28163382,  0.33090204,  0.5461725 ,  0.40770623,
   0.04129666,  0.3159838 ,  1.3818344 ,  1.4300308 ,  0.8795411 ,
   0.20564438,  0.92888695,  0.41792953,  0.11095006,  0.49659857,
  -0.01911918,  0.10609318,  0.10521641, -0.02439536,  0.31944805,
   0.11440527,  0.13461286,  1.0427089 ,  0.05791436,  0.23765296,
   0.54219   , -0.06761637, -0.09881728,  0.09677941,  0.12313514,
   0.46905446,  0.53387934,  0.37274897,  0.53632706,  1.2893828 ,
   0.1577638 , -0.15138581,  0.4732335 ,  1.0241998 ,  0.62808233,
   0.44228932,  0.19107167,  0.22103387,  0.6372364 ],
 [ 0.05911824, -0.10920618,  0.14235383,  0.16341068,  0.25015023,
   0.04282478,  0.33786544,  0.53182703,  0.994467  ,  0.02305396,
   0.08625709,  1.1535486 , -0.01502147, -0.28495023,  0.7258288 ,
   0.24222371,  0.16593705,  0.2469371 ,  0.37359315,  0.30940795,
  -0.12737966,  1.0451899 ,  0.543675  ,  0.40105757,  0.35984758,
   0.4439657 ,  0.807364  ,  0.30676132,  0.58521444,  0.43102318,
   0.7639463 ,  1.419621  , -0.08539165,  1.3203108 ,  0.08604408,
   0.33714965,  0.05274599,  0.44310737,  0.380579  , -0.20391104,
   0.11120015,  0.2424687 ,  0.92389965,  0.11758212,  0.25365207,
   0.7081739 ,  0.37665424,  0.2412055 ,  0.26618314,  0.22770247,
   0.7076005 ,  0.3733366 ,  1.7404891 ,  1.2124325 ,  1.1202147 ,
   0.6904281 ,  0.22812122,  0.25888968,  0.2105825 ,  0.22396775,
  -0.06071058, -0.05186402,  0.425657  ,  1.8368808 ],
 [ 0.16023837, -0.06843945,  0.27045548,  0.4925816 , -0.1478903 ,
  -0.04748981,  0.09587437,  0.16550614,  0.20279697,  0.13738732,
  -0.46592763,  1.0153881 , -0.12698576,  0.17796263,  0.05600359,
  -0.03130552,  0.09500602, -0.7596124 , -0.24456888, -0.4082541 ,
  -0.13198501, -0.5657956 , -0.04216576,  0.05092406,  0.29369196,
   0.04217416,  0.38701078,  0.39556307,  0.48616448,  0.2935213 ,
  -0.03578395, -0.03138146,  0.1015454 ,  0.28871116,  0.6047394 ,
   0.00697435, -0.46527213,  0.3893296 , -0.34858787, -0.0135033 ,
   0.08531404, -0.20001344,  0.48518395, -0.30017596, -0.14058255,
   0.3715411 , -0.09141959, -0.1700237 ,  0.46767822,  0.3245002 ,
  -0.23179552,  0.7193164 ,  0.29653192,  0.2318328 ,  0.89027977,
   0.2093713 , -0.87162703, -0.1325815 ,  0.6937161 ,  0.45933515,
  -0.06642124, -0.31430104, -0.03004904, -0.17265774],
 [ 0.55705124,  0.2936609 ,  0.3709807 ,  0.24516997, -0.17647421,
  -0.00805244,  0.21623681,  1.1871588 ,  0.39707994,  0.5245848 ,
   0.17424037,  1.1944557 ,  0.35050887,  0.24287419,  1.2568202 ,
   0.10713281, -0.07040904,  0.88997453, -0.09400082,  0.17713632,
   0.59290105,  0.6868093 ,  0.94536877,  0.3798627 ,  0.43189704,
   0.23396336,  1.0345591 ,  1.1710043 ,  0.8918536 ,  1.303434  ,
   0.8684246 ,  0.8811526 , -0.08909445,  0.796947  ,  0.9345099 ,
  -0.12608244,  0.18879187,  0.34074783,  0.26152933,  0.03549217,
   0.09601635,  0.23286839,  2.2790377 , -0.17621897,  0.718909  ,
   1.2088714 ,  0.67501384,  0.24649872,  0.60601026,  0.6176288 ,
   0.99847245,  0.72402287,  1.6298505 ,  0.7541408 ,  1.0471387 ,
   0.74324346, -0.6250399 ,  0.5795569 ,  1.5706677 ,  0.9093677 ,
  -0.03680039, -0.26399672,  0.5677193 ,  1.4534155 ],
 [-0.01971989, -0.02066544,  0.35900638,  0.20549595, -0.2597859 ,
   0.66282773,  0.23604965,  1.0537211 ,  0.2517265 ,  0.10034573,
  -0.21411261,  1.4512695 ,  0.3796565 , -0.67285955,  1.0923712 ,
   0.03960132, -0.26578984,  1.0358181 , -0.12842171,  0.1043697 ,
   0.19137684,  0.71835   ,  0.39771158,  0.11365397,  0.03628154,
   0.01209299,  0.69157493,  0.79298025,  0.6270931 ,  1.3891101 ,
   0.39522642,  0.7457346 , -0.17183293,  0.91944325,  0.68202704,
   0.0698573 ,  0.16935372,  0.47737953, -0.01564662,  0.4624141 ,
   0.20512591,  0.40339878,  0.21427988, -0.13120247,  0.41525927,
   0.4932933 , -0.15828532,  0.19951677,  0.33602244,  0.42892656,
   1.2027907 ,  0.25711808,  0.62338126,  1.0573574 ,  0.62979543,
   0.38022798, -0.6238228 ,  0.21969992,  0.54310983,  1.0776857 ,
  -0.06217473, -0.13974157,  0.3475057 ,  1.1832181 ],
 [-0.00364584,  0.3033346 ,  0.2741975 ,  0.22683612,  0.3590831 ,
   0.10611626,  0.25610948,  0.68534774,  0.48407504, -0.04534008,
   0.21835436,  0.8685624 ,  0.14899625,  0.0994283 ,  0.9288295 ,
   0.29455492,  0.26877022,  0.4308395 ,  0.25456196,  0.3999012 ,
   0.3376762 ,  1.7066652 ,  0.38354912,  0.16379291, -0.03935851,
   0.48465148,  0.39613557,  1.0807588 ,  0.71948874,  0.20208114,
   1.4157509 ,  1.3024225 ,  0.21901774,  1.2715783 ,  0.24009177,
  -0.1233912 ,  0.2500504 ,  0.27165005, -0.11797062,  0.318898  ,
   0.28219613,  0.75139177,  0.89159614,  0.13727438,  0.6066803 ,
   0.3456326 ,  0.6696993 ,  0.25767675,  0.32097194,  0.02925099,
   0.8214507 , -0.6345038 ,  0.51636356,  1.246041  ,  0.9024087 ,
   0.35894495, -0.2080633 ,  0.18476517,  1.4621662 ,  0.13945109,
  -0.08296894, -0.14385992,  0.03514507,  0.16834581],
 [ 0.23300965,  0.22398314,  0.32990396, -0.33447516, -0.39859477,
   0.18030217,  0.17586239, -0.05944359,  0.4852944 ,  0.21678388,
   0.09186714,  0.67570305,  0.17125331,  0.4929156 ,  0.1087823 ,
   0.05212266, -0.39570314,  0.40915522,  0.06838217,  0.07910968,
   0.13642982,  0.62943405,  0.3049805 ,  0.05141467,  0.4940411 ,
   0.19522257,  0.47395268,  0.21210447,  0.1546249 ,  0.43454337,
   0.75644344,  0.3328134 , -0.09866128,  0.8784695 ,  0.88711405,
  -0.26651594, -0.01720531,  0.2974602 , -0.1929154 ,  0.2134518 ,
  -0.01059779, -0.07880016,  0.7627953 , -0.1259773 ,  0.2912683 ,
   0.17361203,  0.43167922, -0.2561634 ,  0.3323342 ,  0.15052146,
   0.605751  ,  0.5989315 ,  0.9613659 ,  0.8144759 , -0.2415633 ,
   0.27692756, -0.60686326,  0.14369658,  0.98843324, -0.7571357 ,
  -0.18710433, -0.3754578 ,  0.0689325 , -0.37477782],
 [ 0.00624303,  0.09468264,  0.36969176,  0.0391277 , -0.2788418 ,
   1.4259148 ,  0.08482026,  0.40652913,  0.93924093,  0.29077807,
  -0.5993655 ,  0.07343477,  0.05514004,  0.44409025,  0.689204  ,
   0.16255066,  0.5632822 ,  0.03799736, -0.249096  , -0.49916664,
  -0.06718341,  0.00113276,  0.6448336 , -0.655927  ,  0.95598614,
   0.04172967,  0.33156207,  0.67071253,  1.0950714 ,  0.38360772,
   1.1432046 ,  0.7964397 , -0.38888758,  1.263393  ,  0.73612577,
  -0.07329208, -0.8215196 , -0.2585515 ,  0.05314139,  0.04471518,
  -0.16070586, -0.333779  ,  0.88776785, -0.28948537, -0.25521156,
   0.7420494 ,  0.54916954, -0.6466246 ,  0.36430192,  0.01759674,
  -0.11041053,  1.0276089 ,  1.2723541 ,  0.852526  ,  0.49507064,
   0.16944917,  0.2846773 , -0.00521203,  1.0563021 ,  1.1299874 ,
  -0.53759664, -0.25299966, -0.09496722,  0.44045618],
 [ 0.47416624,  0.25552493,  0.6936844 ,  0.9510034 , -0.33612236,
  -0.52347046, -0.03048547,  0.48984405,  0.3039907 ,  0.31912524,
  -0.6648812 ,  0.15432169,  0.28682995,  0.31203783,  0.34686345,
  -1.1543943 ,  0.17536147,  1.1430291 , -0.82186455, -0.64252216,
   0.20578064, -0.1934263 ,  0.7633363 , -0.18701103, -0.0416648 ,
  -0.5959767 ,  0.43968698, -0.20583877, -0.18083282, -0.18959005,
  -0.40258265,  0.22163863, -0.29464635, -0.31061196,  0.00343906,
  -0.19245414, -0.43956363, -0.41559798, -0.17319624, -0.63731563,
  -0.05633901, -0.43896502,  0.25696146, -0.54864204,  0.465142  ,
   0.67490315, -0.06888263, -0.2341074 ,  0.7193837 ,  0.72997385,
   0.557695  , -0.28681815,  0.36836538, -0.15811226,  0.27061066,
   0.53317165, -0.40516785,  0.3265245 ,  0.67317486,  0.6015057 ,
  -0.28665605, -1.024163  ,  0.7009605 ,  0.9679915 ],
 [ 0.19377302,  0.11177593,  0.21934941, -0.02007265, -0.20474826,
   0.55467206,  0.10607562,  0.9525423 ,  0.4010257 ,  0.22227816,
  -0.57004875,  1.3040313 ,  0.2652748 ,  0.30132502,  0.5276724 ,
   0.08378245,  0.320173  ,  0.5169512 ,  0.13164166,  0.1725553 ,
   0.3454261 ,  0.3950084 ,  0.0808877 ,  0.01436515,  0.18444867,
   0.06570762,  0.48425922,  0.7646371 ,  0.6979909 ,  0.98728895,
   0.60237384,  1.2098478 ,  0.25787276,  0.63167834,  0.54844373,
   0.05590387,  0.05355346,  0.19586687, -0.04590665,  0.15167487,
   0.0451274 ,  0.01370323,  0.9130378 , -0.00643361,  0.08960757,
   0.2190938 ,  0.20354754, -0.15891409,  0.2172235 ,  0.3151055 ,
   0.4752213 ,  0.8899224 ,  1.1967847 ,  0.71638477, -0.15944365,
   0.41184357,  0.15198074,  0.17105298,  1.0898542 ,  1.0604522 ,
   0.08655441, -0.05254069,  0.0478588 ,  1.0600106 ],
 [-0.15671217,  0.4945648 ,  0.13652796,  0.5938879 , -0.17172572,
  -0.68253   ,  0.2981738 ,  0.39674646, -0.05609779,  0.12660849,
  -0.34502256,  0.8791858 ,  0.10800626,  0.4598704 ,  0.41267577,
   0.00560046, -0.11110397,  0.26884073, -0.08691165, -0.16686735,
  -0.10421489,  0.39196354,  0.06663325,  0.17762354, -0.4776276 ,
   0.43255496,  0.38460338,  0.9350795 ,  0.60309273,  0.4399694 ,
   0.29499367,  0.32422426, -0.82943773, -0.29590234,  0.57592005,
   0.06706104, -0.21176113,  0.01816714,  0.30150247,  0.09249539,
  -0.02699599, -0.20049636,  1.0130142 , -0.16902734,  0.22305124,
   0.51677096,  0.07288049,  0.2472441 ,  0.27585855, -0.21259184,
   0.5790303 ,  0.37358722,  0.9367327 , -0.33263817,  1.1705818 ,
   0.33289385, -0.06606682,  0.10635998,  0.6484711 ,  0.38406247,
  -0.5452727 , -0.09933128, -0.39017156, -0.13099127],
 [ 0.48045814, -0.03692601,  0.39806676,  0.5959581 , -0.5300984 ,
   0.12556231,  0.11824707,  0.65370655,  0.36467952,  0.17354953,
  -0.4244048 ,  0.3888393 ,  0.36940417,  1.2151792 ,  0.432694  ,
  -0.09036648, -0.10472205,  0.95508415, -0.30802473, -0.19974674,
   0.4164365 ,  0.1742941 ,  0.41925946, -0.5502104 ,  0.54272765,
   0.3537276 ,  0.2753298 ,  0.691722  ,  0.68620765,  0.808437  ,
   0.1184348 ,  0.21361528, -0.5302796 ,  0.20555362,  0.34347787,
   0.45526966, -0.23493238, -0.17151149,  0.56067175, -0.6768523 ,
   0.09526335, -0.37572604,  0.05943882,  0.10082526,  0.77912176,
   0.7273752 ,  0.58924186, -0.23732017,  0.39763272,  0.09451108,
  -0.63249713, -0.08822813,  0.22200243,  0.24977143,  0.2110574 ,
   0.33371186,  0.00411526,  0.4014537 ,  0.15806656,  0.6180919 ,
  -0.8389826 ,  0.08023411,  0.44048712, -0.10874735],
 [-0.44316894,  0.15033899, -0.05713081,  0.13809507,  0.00311785,
   1.1288575 , -0.95929605,  0.4136884 ,  1.319324  , -0.25989187,
   0.5583022 ,  0.6881475 , -0.57656354, -0.09263996,  0.64552337,
   0.61212707,  0.39863685,  0.3884175 ,  0.38592184,  0.3575021 ,
  -0.02122308,  1.0261314 , -0.13770068,  0.17425297,  0.83964264,
  -0.74278355,  0.33973676,  0.7042277 ,  0.81858087,  0.06000526,
   1.3330576 ,  1.286268  ,  0.78238535,  1.5943794 ,  0.05293529,
   0.31794563,  0.7473928 , -0.19606869,  0.20024292, -0.11419261,
  -0.3557182 ,  0.13936067,  0.24926096,  0.16267385,  0.51282084,
   0.18012413,  1.0177442 , -1.0231673 , -0.27653345, -0.03500289,
   0.46683323,  0.30232057,  0.83195895,  1.5610029 ,  0.80130076,
  -0.00539114,  0.69285905, -0.6061925 ,  1.4944972 ,  0.9964484 ,
   0.71189946,  0.5473964 , -0.33387613,  0.39807984],
 [ 0.1294816 ,  0.3546122 ,  0.26513484,  0.9761809 , -0.16860487,
   0.16344044, -0.01853152,  0.6434863 ,  0.15601045,  0.16830243,
   0.01103675,  1.0391854 ,  0.13512346,  1.9643412 ,  0.8801946 ,
   0.2070309 , -0.07930882,  0.8455851 , -0.12994091, -0.39773571,
  -0.04220447,  0.48888004,  0.26301673, -0.36846885, -0.05925379,
   0.18656895,  0.26254383,  1.7695831 ,  1.2308775 ,  0.29902148,
   0.15460137,  1.0634764 , -0.24907237,  0.42657456,  0.44140384,
  -0.52634895, -0.6385639 , -0.1326039 , -0.03919021,  0.2705847 ,
   0.03116868, -0.3688816 ,  0.97069484, -0.1585119 ,  0.39546904,
   0.82191616,  0.2794835 , -0.16689165,  0.13012795, -0.14587253,
   0.34739468,  0.2763361 ,  1.0055887 ,  0.3372724 ,  0.82747865,
   0.20096508, -0.07523132,  0.4388923 ,  0.6675452 ,  0.9457941 ,
  -0.20134914, -0.08171909, -0.07719094,  0.74753356],
 [ 0.05540695,  0.35475838,  0.04466308,  0.5803647 , -0.03358778,
   0.4945579 , -0.01788076,  0.21505234,  0.30675772,  0.30271974,
   0.12431656, -0.04046544, -0.11484618,  0.1456984 ,  0.2754559 ,
   0.11506899,  0.6596618 ,  0.40077826, -0.06177015,  0.11543783,
  -0.09844685,  0.50982696,  0.22800814,  0.15915051,  0.46444562,
  -0.23878905,  0.51626045,  0.9811653 ,  0.5871956 ,  0.81350183,
   0.66472685,  0.4141707 ,  0.14385843,  0.8600461 ,  0.63928556,
  -0.40631214,  0.07621342,  0.16789289, -0.26900762,  0.05482522,
   0.00108983,  0.20039122,  0.559961  ,  0.02021709,  0.52707887,
   0.29040137,  0.01808117, -0.23409791,  0.27484438,  0.3647419 ,
   0.6741883 ,  0.45020992,  0.4178585 ,  0.7816975 ,  0.73641664,
   0.55335635,  0.31039482, -0.16039754,  0.4235269 ,  0.56298065,
   0.06091775, -0.11560878,  0.04101128,  0.8041598 ],
 [-0.40492982,  0.58287764, -0.177789  ,  0.83633643, -0.00097683,
   1.1583275 , -0.47639164, -0.12034287,  0.67545325, -0.10953167,
   0.03968034,  0.4087918 , -0.19980592, -0.2605993 ,  0.07297166,
   0.7276445 , -0.10168562,  0.2389125 ,  0.48742703, -0.43348324,
  -0.16193177,  0.2180203 ,  0.22898202,  0.19459474,  0.690841  ,
  -0.43625683,  0.3162023 ,  0.267437  ,  0.24152449,  0.42762098,
   1.337428  ,  0.7927073 , -0.01818601,  1.2486362 ,  0.6431263 ,
  -0.230041  ,  0.13130508, -0.29114413, -0.15216266,  0.22485909,
  -0.42162487,  0.12743385,  0.74926156,  0.14442451, -0.1415773 ,
   0.2484047 ,  1.0399902 , -0.65072376, -0.2579265 , -0.10679369,
   0.35581458,  1.3854976 ,  0.27661726,  1.0463873 , -0.18311957,
   0.17454992,  0.11237652, -0.1283435 , -0.02305574,  0.16669182,
   0.24441914,  0.09727278, -0.01361776,  0.5851093 ],
 [ 0.27354145,  0.5282987 ,  0.09016973,  0.11468125, -0.32995108,
  -0.12562607,  0.27681208, -0.21397819, -0.54195744,  0.16645189,
  -0.04735934, -0.01064063,  0.22842844,  0.12534405, -0.40238574,
  -0.54315925, -0.46261546, -0.09315921, -0.44836387, -0.05582529,
   0.28765798, -1.4427124 ,  0.1273065 ,  0.0513741 , -1.016275  ,
   0.30714348, -0.04069064, -0.03854358, -0.08980291,  0.08989855,
  -0.18565491, -0.06579526, -0.22365676,  0.33837527,  0.05269746,
  -0.4780204 , -0.06352407,  0.16672456, -0.12322964, -0.12358879,
   0.33721438,  0.02461912,  0.23676209, -0.31828064,  0.26014045,
  -0.0399364 , -0.46277687,  0.60050106, -0.01334912, -0.04986055,
   0.7356556 , -0.55519086,  0.51809174, -0.8935324 , -0.34178457,
  -0.44310766, -0.7516139 ,  0.47761312, -0.4057426 , -0.3732479 ,
  -0.0540972 , -0.16901335,  0.45967212,  0.16551077],
 [ 0.13843305,  0.27751613,  0.3365747 ,  0.03863859, -0.245835  ,
   0.39667875,  0.1026807 ,  0.5369819 ,  0.18768936,  0.15685533,
  -0.00540811,  1.0082895 ,  0.1846779 ,  0.17713289,  0.23341285,
  -0.26166043, -0.15060435,  0.64899105, -0.05345935,  0.0211784 ,
   0.42450756,  1.0595744 ,  0.30857188,  0.00433788,  0.02908269,
   0.09379411,  0.6385184 ,  0.34498957,  0.11653797,  0.9450066 ,
   0.4380757 ,  0.4963022 ,  0.07184528,  1.1747645 ,  1.0268067 ,
  -0.3865765 , -0.20688711,  0.2735469 , -0.3052642 ,  0.02852383,
   0.08222608,  0.09845054,  0.77264184, -0.25544232, -0.09173078,
   0.45551348,  0.12999767, -0.26355356,  0.31155086,  0.297783  ,
   0.57957584,  0.8157342 ,  0.19900213,  1.2732849 , -0.03280266,
   0.35894418, -0.2358233 ,  0.05998436,  1.2345793 ,  0.32825333,
  -0.01717102, -0.3044852 ,  0.21222195,  0.39227483],
 [-0.36024442,  0.2664325 , -0.15797277,  0.4080025 , -0.5481407 ,
  -0.20833841, -0.36329913,  0.5231798 ,  0.25671825, -0.42541286,
  -0.5026525 ,  0.85454077, -0.2493511 ,  0.8875629 ,  0.7228158 ,
   0.4542187 ,  0.13666978,  1.0999799 , -0.30947268, -0.19030875,
  -0.01137749,  1.1123557 ,  0.1734234 , -0.7901978 ,  0.4144752 ,
  -0.283192  ,  0.7378073 ,  0.70947754,  0.5114471 , -0.04128777,
   0.4429917 ,  0.04718898, -0.45098948,  0.16411388,  0.16465634,
   0.11739026, -0.23971461, -0.6029004 ,  0.5991484 , -0.619716  ,
  -0.25791067, -0.32994035,  0.7396861 , -0.16443259,  0.13758841,
   0.5787787 ,  0.55419934, -0.5050986 , -0.08682113,  0.02769887,
   1.0059578 ,  0.26362374,  0.46348682,  0.10422061,  1.1618636 ,
   0.55492914,  0.18752499, -0.21378408,  0.49672505,  0.42607707,
  -0.47238263, -0.6349538 ,  0.14627881,  0.05565886],
 [ 0.13574141, -0.11665141,  0.18913642,  0.8128416 , -0.36673656,
   0.43723252, -0.07706817,  0.72508454,  0.26488855, -0.03072407,
  -0.18561321,  0.67868125,  0.25685802,  0.34348115,  0.59425795,
   0.33563498, -0.03292141,  1.1382848 ,  0.1341517 , -0.04188884,
   0.50510603,  0.16519286, -0.08160388,  0.07574695,  0.8851728 ,
   0.03004637,  0.19698836,  0.8956606 ,  0.47806126,  0.2691629 ,
   0.18865809,  0.32896188, -0.3926799 ,  0.17051427,  0.39011106,
   0.15462226,  0.19857016, -0.07169136, -0.14836602, -0.45397067,
  -0.00052901,  0.14178787,  0.0616814 , -0.03511576,  0.8746996 ,
   0.23075408, -0.02293782, -0.07569112,  0.289135  ,  0.19528429,
   0.9885615 ,  0.04077401,  0.70638055,  0.28103343,  1.360888  ,
   0.11226398, -0.49500856,  0.25608665,  0.79138744,  0.9016113 ,
  -0.3667613 , -0.33377945,  0.55417645,  0.48777813],
 [-0.11682466,  0.5531336 ,  0.21994378,  0.28230536, -0.374045  ,
   0.08611821, -0.16980283,  0.7097904 ,  0.4663058 ,  0.18283066,
   0.36034718,  1.0018282 ,  0.14477518,  0.02800404, -0.06480548,
   0.52521145, -0.20363678,  0.25411963, -0.06696853, -0.50709134,
   0.30434448,  0.03482676,  0.41917363, -0.4398268 ,  0.563495  ,
  -0.02343375,  1.0424415 ,  0.35408944,  0.46404922,  0.06296545,
   0.6013125 ,  0.02826957, -0.32631457, -0.30081213,  0.7706632 ,
   0.31948763, -0.10175756, -0.16847776, -0.2126511 , -0.23056798,
  -0.2542568 , -0.68006176, -0.01214469, -0.65697587,  0.47172144,
   0.02366448, -0.11488865, -0.23353115,  0.31595716,  0.17924865,
   0.9580001 ,  0.33915588,  0.91745585, -0.08365444,  0.63035554,
   0.6085151 ,  0.35405895,  0.20500755,  0.13447125,  0.84886605,
   0.1260852 , -0.39881912,  0.49203384, -0.07264447],
 [ 0.11864755,  0.02915198,  0.48126456,  1.2128724 , -0.01159931,
   0.69251937,  0.23270819,  0.66195464,  0.83974284,  0.21776046,
   0.23841594,  0.9984421 ,  0.13855462,  0.03815736,  0.623527  ,
   0.10770757,  0.08561832,  0.97884744,  0.17862342,  0.17126663,
   0.2549899 ,  0.13707899,  0.57343954,  0.26691785,  0.6380216 ,
   0.02976116,  0.5298627 ,  1.20428   ,  1.3067703 ,  0.77133256,
   0.1143999 ,  0.7899304 ,  0.27280316,  0.38357744,  1.1275295 ,
  -0.3399685 ,  0.42638913,  0.21507381, -0.01623513,  0.33244127,
  -0.00354456,  0.16723204,  1.0502177 ,  0.18086174,  0.6013752 ,
   0.56119174,  0.37562776, -0.265161  ,  0.7155434 ,  0.40733337,
   0.53060365,  1.0109919 ,  1.653389  ,  0.9591989 ,  0.8931037 ,
   0.58539206,  0.08089422,  0.27516407,  1.6394981 ,  1.006216  ,
   0.35639557,  0.31029657,  0.41133115,  0.58191437],
 [-0.56930494, -0.38361016, -0.7396468 , -1.0349646 , -0.29198173,
  -0.95996183, -0.4294153 , -1.1056457 , -0.6020707 , -0.6883491 ,
  -0.695496  , -1.4287337 , -0.52294624,  0.1325389 , -1.2353678 ,
  -0.42866436, -0.0339398 , -0.73255336, -0.4278792 , -0.2165834 ,
  -0.5697423 , -1.2472991 , -0.5712466 , -0.36830917, -0.737704  ,
  -0.586171  , -1.6028932 , -1.6149138 , -1.5927352 , -1.6352918 ,
  -1.0181868 , -0.97849953, -0.20348994, -0.8479281 , -0.38352373,
  -0.24419703, -0.0355472 , -0.35433114, -0.12826733, -0.23580872,
  -0.31003764, -0.35908344, -1.4539499 , -0.6203399 , -0.49635914,
  -0.7129255 , -0.9057755 , -0.39907286, -0.8063106 , -1.1753379 ,
  -0.7890025 , -1.1669099 , -2.044396  , -1.2086247 , -1.6060411 ,
  -1.3175293 , -0.23028736, -0.67767155, -1.3979652 , -1.2024802 ,
  -0.20026399, -0.18997593, -1.2143315 , -1.0636883 ],
 [ 0.08532846, -0.06925591,  0.14363456,  0.46069473, -0.5633421 ,
   1.0565572 ,  0.14275098, -0.41926458,  0.37174958, -0.28872576,
  -0.47808635,  0.27513462,  0.4081769 ,  0.969728  ,  0.85597605,
  -0.0434553 , -0.65553975, -0.14611714,  0.18844391, -0.2317011 ,
   0.6047874 ,  0.22462484,  0.46916854, -0.07617826,  0.5051933 ,
   0.09234597,  0.02211483,  1.2641283 ,  0.91166645, -0.2857104 ,
   0.87190056,  0.2256776 , -0.18477531,  1.2250233 ,  0.04644274,
   0.4258308 , -0.23046288,  0.35002118,  0.78530204,  0.07717177,
   0.10305417,  0.51630527,  0.55913526,  0.5320308 ,  0.50229526,
   0.7938382 , -0.04881832, -0.10683358,  0.5060212 ,  0.06945742,
  -0.43150872,  0.13883792, -0.12601998,  1.6361006 , -0.23828894,
   0.11776511,  0.8509423 ,  0.2379411 ,  0.43261   , -0.03357439,
   0.16447814,  0.19929081,  0.13639435,  0.27865228],
 [-0.18726876,  0.3403925 ,  0.24098459,  0.29476652, -1.6151732 ,
  -1.1893287 , -0.07420027, -0.5405113 , -1.1347538 , -0.4659242 ,
  -1.0605973 , -1.4852923 , -0.09315101,  0.29094043, -0.23025504,
  -1.0365497 , -1.7008528 , -0.66402334, -0.8675765 , -0.10915208,
   0.04559811, -0.3319407 , -0.03487253, -0.19215223, -1.515373  ,
  -0.6295194 , -1.6896758 , -0.0770425 , -0.12386189,  0.08866734,
  -1.162278  , -0.8913074 , -0.14532667, -0.6526848 , -0.1941232 ,
  -1.0407704 ,  0.26313323, -0.09739999, -2.1444542 , -0.30980796,
  -0.0401602 , -0.09900758,  1.0033168 , -1.3050022 , -0.08047079,
  -0.20647584, -1.7553029 , -0.37950462, -0.05786606, -0.06814958,
  -1.3134995 , -0.02669942, -0.16875535, -1.6020075 , -0.04047348,
  -1.3328838 , -0.27823746, -0.61941105, -1.0904233 , -1.3703187 ,
  -0.5488661 , -1.4027805 , -0.5053186 ,  0.20058613],
 [ 0.05404621, -0.37077668, -0.03325899, -0.13084926, -0.44725338,
   0.12826599, -0.71835333, -0.14004306,  0.04752423, -0.06301873,
  -0.67464155,  0.7354934 , -0.04758253,  0.39959472,  0.4822887 ,
   0.24847284, -0.65157104, -0.04132784, -0.2069646 , -0.9944452 ,
  -0.04963691, -0.48165625, -0.05212706, -0.12477615,  0.21438815,
   0.05797993,  0.5893739 ,  0.10046827, -0.07447805,  0.17473039,
  -0.2121523 ,  0.04035596, -0.9721772 ,  0.34850362,  0.11486802,
  -0.35788926, -0.22571291, -0.19677666, -0.63729477, -0.2873213 ,
  -0.13046132, -0.21771944, -0.07887332, -0.29313502,  0.15802631,
   0.17024583, -0.4986632 , -0.18761776,  0.10779668,  0.09253877,
   0.5139393 , -0.12728524, -0.6179285 ,  0.20048283, -0.02079238,
   1.1876786 , -0.16946846,  0.19021451,  1.1867452 ,  1.125623  ,
   0.04236802, -0.08124219,  0.41049397,  0.25914112],
 [ 0.06684148,  0.3210591 , -0.23869412,  0.12047094, -0.0386869 ,
   0.26482543, -0.10887162,  0.14237271,  0.45089406, -0.11297325,
  -0.4854518 , -0.2914802 , -0.11632086,  0.09830146,  0.49075052,
  -0.37968355,  0.34307158,  0.19806278, -0.6256726 ,  0.24704982,
   0.0807316 , -0.43375155,  0.07971428, -0.24126294,  0.27300113,
  -0.28153917,  0.30600762,  0.31616646,  0.2755897 , -0.05934688,
   0.2954697 ,  0.7504407 ,  0.24995558,  0.3613938 , -0.31242788,
   0.1521159 ,  0.41915765, -0.05320292,  0.21002077,  0.05496106,
   0.08158604,  0.07107868,  0.760462  , -0.6998036 , -0.1305441 ,
   0.10383818,  0.05268971, -1.165109  , -0.06067889,  0.11409431,
   0.57092583, -0.66412455, -0.23296376,  0.34274247, -0.5720487 ,
   0.30339566,  0.33999738,  0.0535196 ,  1.071822  ,  0.31067944,
  -0.02119911,  0.32193574, -0.09643464, -0.3874399 ],
 [-0.46922004, -0.27332148,  0.12343791,  0.06118266,  0.01687832,
   0.09674194, -0.41131204,  0.39222762,  0.5821303 ,  0.3216399 ,
   0.34369642,  0.69978493, -0.28024432,  0.26575038,  0.55711955,
   0.40222064,  0.25827745, -0.04131689,  0.10950751, -0.3886861 ,
   0.16084339,  0.64767015,  0.07560889, -0.29474902,  0.00769215,
  -0.21584673,  0.91219264,  0.3021459 ,  0.42194104,  1.6300026 ,
  -0.00852063,  0.7282659 ,  0.09582587,  0.5177051 ,  0.26679316,
   0.36054242, -0.7457076 , -0.33828682,  0.2061811 ,  0.07959052,
  -0.64487004, -0.2054056 ,  0.3605859 ,  0.03173747, -0.49918064,
   0.4382116 ,  0.22998181,  0.23751001,  0.17779964,  0.5245923 ,
   0.45645595,  0.6345556 ,  0.4973044 ,  0.6460363 ,  0.8603594 ,
   0.6884211 ,  0.1456184 , -0.42434183,  0.58047426,  1.5460452 ,
  -0.23676115,  0.11564942,  0.4477428 ,  1.4488449 ],
 [ 1.0149146 ,  0.18156934,  0.8321812 ,  1.3034905 ,  0.7529512 ,
   0.2526064 ,  0.32889554,  0.5036674 , -0.07023912,  0.22993673,
  -0.21241206, -0.00528619,  0.7257466 , -0.26285943,  0.15317479,
   0.5884778 , -0.54092336,  1.0353935 ,  0.31212062, -0.21900955,
   0.9848623 ,  0.8336    ,  0.57097197, -0.2842936 ,  0.34224212,
   0.34563172,  0.4876327 ,  0.41968378,  0.18758073,  1.6618952 ,
   0.08732621,  0.6014361 , -0.3293208 ,  0.64976335,  1.3836912 ,
   0.7567854 , -0.29843494, -0.2104605 ,  0.5314343 , -0.44765872,
   0.29239503,  0.2681713 , -0.30367243, -0.34473133,  0.6189645 ,
   0.6062272 ,  0.61170244,  0.6478523 ,  0.821135  ,  0.57460624,
   1.4122585 , -0.6156699 , -0.1811899 ,  0.46628067,  0.68786585,
   0.8082206 , -0.01939268,  1.0865365 ,  0.41607335,  0.01750198,
  -0.16139789, -0.18085492,  0.7951528 , -0.06037139],
 [ 0.09094386,  0.3543686 ,  0.13020901,  0.13954079,  0.23706385,
   0.78502876, -0.18245964,  0.6677625 ,  0.2109165 ,  0.22750302,
  -0.08228347,  0.7350207 ,  0.0846084 ,  0.4770058 ,  0.8377633 ,
   0.25831798,  0.6508005 ,  0.49976897, -0.05032941, -0.66028535,
  -0.41121444, -0.29296988,  0.6459531 , -0.3587239 ,  0.584641  ,
   0.17200223, -0.00015423,  0.6332417 ,  0.842663  ,  0.31349313,
   0.82561773,  0.71448624, -0.5338409 ,  0.7645152 , -0.02842869,
  -0.00101761, -0.6604299 , -0.54378146, -0.25410163, -0.3211634 ,
  -0.06814382, -0.41763428,  0.5966784 ,  0.40006337,  0.01313119,
   0.8906029 ,  0.05980213, -0.35527238,  0.46565726,  0.05558459,
   0.32813552,  0.49041006,  0.80275697,  0.74987316,  0.5729854 ,
   0.10090951, -0.03736174,  0.20126857,  0.74336   ,  1.3784983 ,
  -0.37707078, -0.11706988,  0.13701737,  0.36679876],
 [ 0.7949629 ,  0.31985605,  0.6709409 ,  0.7842187 , -0.00571796,
  -0.02676827,  0.23061341,  0.70357025,  0.4616415 ,  0.7212571 ,
   0.01764664,  1.5734376 ,  0.52144593,  0.40176845,  1.2225372 ,
  -0.1041737 , -0.25833356,  1.4849054 , -0.0125685 ,  0.2891747 ,
   0.8623497 ,  1.2921139 ,  1.0806353 ,  0.4399657 ,  0.19103396,
   0.5021852 ,  0.7873251 ,  0.14573933,  0.5509041 ,  1.4889945 ,
   0.7111291 ,  0.820703  , -0.00534912,  0.49742967,  0.539702  ,
  -0.52091646,  0.77704436,  0.30331448,  0.16529316, -0.05064823,
   0.51733184,  0.74943   ,  0.2154828 ,  0.29640025,  1.1128027 ,
   1.0416286 ,  1.1705819 ,  0.6097887 ,  0.8109623 ,  0.7066679 ,
   1.4545422 ,  0.2751882 ,  0.48679683,  0.5723722 ,  2.3335648 ,
   0.82439315, -0.61315906,  0.9857353 ,  0.5113451 ,  0.25332913,
   0.23062083, -0.48297718,  1.2671415 ,  0.78570455],
 [ 0.12652053,  0.00863551,  0.21159717,  0.7562037 ,  0.29839572,
   0.6354447 , -0.02917131,  0.9056687 , -0.05125718,  0.24898322,
  -0.10281365,  0.72335243,  0.22055016, -0.5013843 ,  0.46208876,
  -0.01953437,  0.55598295,  1.1513605 , -0.15353036,  0.07702132,
   0.11604081,  0.879262  ,  0.28356147,  0.3087061 ,  0.38664868,
   0.19736253,  0.4722712 ,  1.2175142 ,  1.1696962 , -0.34501168,
   0.5114362 ,  0.32353386, -0.11785096,  0.66164976,  0.8609046 ,
   0.19608684,  0.42836797,  0.30303866, -0.133132  ,  0.3627914 ,
  -0.02614495,  0.29819164,  1.4682928 ,  0.34916875,  0.38725862,
  -0.00080884,  0.00122103,  0.06625958,  0.41419956,  0.23041192,
   0.66007143,  0.359967  ,  1.4396814 ,  0.6146885 ,  1.4871464 ,
   0.47298604, -0.06666652,  0.20207508,  0.5677182 ,  1.4945576 ,
   0.1566274 ,  0.3100401 ,  0.41813904,  0.4882877 ],
 [-0.36672756, -0.28650126, -0.11279906, -1.442275  ,  0.24160428,
   0.10114587, -0.1386709 , -0.5485563 , -0.28816438, -0.6697406 ,
  -0.3697755 , -0.30792615, -0.08395795,  0.641061  , -0.08243654,
  -0.434784  ,  0.12525746, -0.7664744 , -0.6732827 , -0.52349097,
  -0.3219939 , -1.1201442 ,  0.36215538, -0.10570962, -0.2276211 ,
  -0.36664325, -0.3428963 , -0.14203179, -0.11870344,  0.33052164,
  -0.3347787 ,  0.2601382 , -0.6324187 , -0.02176224, -0.11980521,
  -0.00590941, -0.3388752 , -0.17411105, -0.5016525 ,  0.00141938,
  -0.26362303, -0.2197068 ,  0.7243195 , -0.41508955, -0.6651521 ,
   0.06710143, -1.047123  , -0.51865023, -0.22995333, -0.59731615,
  -1.0507269 , -0.02857865, -1.2073572 , -0.17742442, -1.3434256 ,
  -0.26117048, -0.29003003, -0.02182907,  0.03032787, -0.23903854,
  -0.9017871 , -0.5973541 , -0.56498045, -0.68735224],
 [-0.38119388,  0.620567  , -0.1752968 ,  0.3451982 ,  0.26927498,
  -0.6192053 ,  0.1260378 ,  0.51440454, -0.6184119 , -0.12055361,
   0.26984662,  0.5282982 ,  0.18058947,  0.13716589,  0.77858335,
  -0.11299927, -0.07594313,  0.7245508 ,  0.17250754,  0.1554903 ,
  -0.14800125,  0.08236001,  0.43863976,  0.3539882 , -0.4207867 ,
   0.03519405,  0.48919603,  0.7375035 ,  0.5828216 ,  0.5579558 ,
  -0.36186984,  0.90575004,  0.13444534,  0.01667099,  0.38049164,
  -0.17510475,  0.0089238 ,  0.15231144,  0.1519713 ,  0.35444647,
   0.03328402,  0.20304087,  0.32823828,  0.2935448 ,  0.1939573 ,
   0.6903026 , -0.2580557 ,  0.03857576,  0.02509209, -0.29439527,
   0.8973729 ,  0.39447516,  0.5982068 , -0.032314  ,  1.1597251 ,
   0.2962635 , -0.766589  , -0.0875401 ,  0.45680144,  0.19638191,
   0.15426484,  0.22437651, -0.1333542 ,  0.6047615 ],
 [-0.01401246,  0.08812129, -0.01467358,  0.60387975,  0.655046  ,
   0.1353321 , -0.5710116 ,  0.4641477 ,  0.40781888,  0.10358647,
  -0.45922643, -0.05199346, -0.09431642, -0.39251733,  0.29486215,
  -0.21681689,  0.92479527,  0.52769697, -0.6601974 , -0.1832335 ,
   0.06907918,  0.32409438,  0.18453379, -0.53237486,  0.7987887 ,
  -0.49897203,  0.30661282,  0.5662109 ,  0.5220952 ,  0.27620173,
   0.34761488,  0.28901288, -0.09516823,  0.02079004,  0.24802282,
   0.44558418, -0.00741464, -0.30191743, -0.39780876,  0.06615103,
  -0.36249378, -0.4646086 ,  0.63326585,  0.02985943,  0.5059641 ,
   0.30191588,  0.31598496, -0.8767468 ,  0.01861459,  0.12973529,
   0.13977335,  0.35848364,  0.4329712 ,  0.3116755 ,  0.2898133 ,
   0.21827258,  0.10346389, -0.17886807,  0.17731805,  0.4233958 ,
   0.21896374,  0.24496754,  0.0430319 ,  0.6686074 ],
 [ 0.12343284, -0.02277377,  0.17749752,  0.31494376, -0.16978128,
  -0.33369133,  0.3188298 ,  0.18678197,  0.6517639 ,  0.2618973 ,
  -0.06955757,  0.44835344, -0.0134311 , -0.09694277,  1.2017094 ,
   0.2729894 ,  0.12505129, -0.09410158,  0.21385501,  0.06783969,
  -0.01995174,  0.09397394,  0.44943342,  0.14117251,  0.15979819,
   0.4176091 ,  0.07457798,  0.9771736 ,  0.32935527,  1.1745565 ,
   0.47686005,  0.99325085, -0.2428646 ,  0.55100685,  0.32350263,
  -0.17616889,  0.09116847,  0.05977833,  0.35942268, -0.22561285,
   0.1593113 ,  0.07055333,  1.3545921 ,  0.17594676,  0.5310683 ,
   1.0989429 ,  0.3696315 ,  0.24007064,  0.40623352, -0.068221  ,
   0.14262456,  0.36035743,  0.06150356,  0.5474644 ,  0.8358607 ,
   0.20740175,  0.08514871,  0.26714468,  0.7054435 ,  0.02987694,
  -0.21010596,  0.00383842,  0.24878226,  0.5911078 ]])
    b4 = jnp.array([-0.9058946 ,  0.32687533, -0.6929519 , -0.8027441 ,  0.04900447,
 -1.6243806 , -0.6608856 , -0.61808586, -1.381121  , -0.6864674 ,
 -0.04656301, -0.23799121, -0.79332006, -0.21145448,  0.18512556,
 -0.36740634,  0.06906458, -0.81571317, -0.43825084, -0.66411096,
 -0.816127  , -0.8288048 , -0.1497331 , -0.41132972, -0.98646086,
 -0.5285957 , -0.8874982 ,  0.3115572 ,  0.07492431, -1.1541378 ,
 -0.44622698, -0.7392172 , -0.680802  , -1.3916881 , -0.82018524,
  0.16400634, -0.5626477 , -0.50972724,  0.02610517,  0.18414725,
 -0.7774067 , -0.4882229 ,  0.3782688 ,  0.22285981, -0.50064975,
  0.08364799, -0.05428135, -1.0426248 , -0.625178  , -0.8287983 ,
 -1.0362575 , -0.3696127 , -0.89935267, -0.75714123, -0.06567267,
 -1.0087959 , -0.54031736, -0.7228628 , -0.83302945, -0.97180074,
 -0.43463355, -0.01162718, -0.87551266, -1.24942   ])
    W5 = jnp.array([[ 0.09373646, -0.29432434, -0.07693493,  0.01584284,  0.02543655,
  -0.03553972,  0.1450611 , -0.12375909,  0.46731564,  0.10457546,
  -0.03179228, -0.17945682, -0.00829576, -0.0503164 ,  0.05475668,
  -0.03018523,  0.04264623,  0.01907245,  0.10565021, -0.0670141 ,
  -0.02016284, -0.12958586, -0.07269667,  0.0209719 , -0.01643806,
  -0.02215671,  0.554439  ,  0.06121888, -0.10048053, -0.6746074 ,
   0.11281964, -0.09686121],
 [-0.07481191,  0.11526795, -0.17093532,  0.06273243, -0.02625501,
  -0.02213324,  0.16432749,  0.16280524,  0.23855728, -0.01144338,
  -0.02437522,  0.0862077 , -0.03101278,  0.02612902,  0.03880431,
   0.13581401, -0.00551964,  0.06355937,  0.04522624,  0.10384455,
  -0.00064146,  0.05953252, -0.02394617,  0.03903516,  0.05145992,
  -0.01968347, -0.42971173, -0.35801697,  0.03335173,  0.20991378,
   0.08906921,  0.0234144 ],
 [-0.03847228, -0.20133564,  0.3991671 ,  0.02105014,  0.01634498,
  -0.04423168, -0.57639116, -0.17477265, -0.43872166, -0.01381949,
   0.12129669, -0.2557003 , -0.1592147 , -0.12736341,  0.29792234,
  -0.33524638,  0.04574918, -0.12809019, -0.03189085, -0.37655258,
   0.1068604 ,  0.01660083, -0.20866714, -0.07896675,  0.19523664,
  -0.14123634,  0.06712931,  0.05019532, -0.08450448, -0.54293674,
  -0.20686449, -0.25854942],
 [ 0.4011705 , -0.3563021 ,  0.23721616, -0.1853494 ,  0.41910326,
   0.06190537,  0.4602604 , -0.43660936, -0.85993165,  0.357968  ,
  -0.2066118 , -0.24683094, -0.08077972,  0.22246937, -0.5957196 ,
  -0.24944991, -0.04707215, -0.34481692, -0.5485255 ,  0.3015405 ,
  -0.18657513, -0.2978005 ,  0.1793798 , -0.1026159 , -0.41645443,
   0.15629366, -0.26851982, -0.256433  , -0.33788866, -0.7265409 ,
  -0.31973135, -0.8669147 ],
 [ 0.10097558,  0.02834394, -0.0825821 , -0.13480937, -0.03129326,
  -0.01142609,  0.00424717, -0.06569781,  0.30460662, -0.0391815 ,
   0.00977469, -0.01015859, -0.13182388, -0.0130821 , -0.13200264,
   0.03973539,  0.00328404, -0.17582008, -0.24742703, -0.1333757 ,
  -0.01995998, -0.19208458, -0.01376422, -0.23803417, -0.14873916,
   0.01115036,  0.32268953,  0.00808305, -0.13176751, -0.06236886,
   0.05068338, -0.10756871],
 [-0.0795456 ,  0.22922169, -0.01583878, -0.09759893, -0.03895869,
  -0.32299858,  0.1612673 ,  0.22269611,  0.643113  , -0.05925502,
  -0.47987613, -0.38755006, -0.2763874 , -0.1636447 ,  0.05104168,
  -0.19200699,  0.07720479, -0.21049304,  0.06053492, -0.1200481 ,
  -0.1138579 , -0.39310065, -0.2835972 , -0.1629357 , -0.3918965 ,
  -0.02266746,  0.15187111, -0.50599664,  0.40887398,  0.09428044,
  -0.11517304, -0.1555334 ],
 [-0.03607084, -0.18366726,  0.13322042, -0.03243887, -0.07300518,
  -0.10740107,  0.34506476, -0.05997602,  0.05806764, -0.04636532,
   0.07127447,  0.16504653, -0.03932972, -0.0426546 , -0.24843596,
   0.25736424,  0.02168217,  0.03646242,  0.16213152,  0.24921496,
  -0.06997111, -0.24687111, -0.06031071, -0.03253951, -0.14869794,
  -0.1189103 ,  0.52126366,  0.10716135, -0.06282587, -0.59894675,
   0.11034149,  0.20823812],
 [ 0.03936436, -0.05131853, -0.1438527 ,  0.12805691,  0.12083177,
   0.08745869, -0.35688615, -0.16794531,  0.54798037,  0.06306807,
   0.08365969,  0.21260288, -0.07477278,  0.07379456,  0.07264445,
   0.22528632, -0.08368979,  0.12030965,  0.13607532, -0.24490087,
   0.25515082,  0.10964857,  0.18669933,  0.04912068,  0.1061913 ,
   0.31177387,  0.45592573, -0.00553145, -0.0819186 , -0.2993315 ,
   0.21678461, -0.03759532],
 [-0.10126597,  0.30552953,  0.4420829 ,  0.3793782 , -0.20196997,
  -0.2538377 , -0.26063815,  0.36558032,  0.36734045, -0.05580857,
   0.6816097 ,  0.1883003 , -0.23867734,  0.07061668,  0.45440477,
  -0.15141833,  0.00686568,  0.04715696,  0.5915047 ,  0.12082285,
  -0.01688678,  0.13978598, -0.14204174,  0.26077   ,  0.04080541,
  -0.05847008,  0.6425369 , -0.2551085 ,  0.37640697, -0.03653382,
   0.23194775,  0.45156443],
 [-0.12764877, -0.28513318, -0.02717201, -0.0226285 , -0.08858021,
  -0.03570861, -0.001535  , -0.12252747,  0.48944056, -0.11584754,
   0.11397959, -0.00112515, -0.08929307, -0.09376828,  0.08578621,
   0.02499025,  0.01113804, -0.1435456 , -0.05062766, -0.05565654,
  -0.01151654, -0.0655771 , -0.00264167, -0.06362986, -0.11225946,
   0.0232717 ,  0.55532116,  0.16165583, -0.14816386, -0.6045368 ,
   0.04019829, -0.1339241 ],
 [-0.00788118,  0.04495212,  0.5821667 , -0.3349787 ,  0.00198799,
   0.01037948, -0.17965789, -0.00456067, -0.5813155 ,  0.06360263,
   0.13880739, -0.09106775, -0.00219348, -0.06846508, -0.27695736,
  -0.24468513,  0.00772546, -0.25566024, -0.04945189, -0.08564092,
   0.00596497, -0.3060734 ,  0.03908602, -0.20290825, -0.2807404 ,
   0.03849241,  0.40808624,  0.6348517 ,  0.12958257, -0.02494194,
  -0.19155954, -0.06598645],
 [-0.07509235, -0.15614702,  0.00899633,  0.25725582,  0.25524223,
  -0.02046764, -0.15361695, -0.14720818,  0.8309563 ,  0.20503078,
   0.0371871 , -0.03709442, -0.12736091,  0.16900623,  0.13187216,
   0.03401962, -0.00527441,  0.0567879 ,  0.11766105, -0.35133487,
   0.07457501,  0.35746527,  0.1637297 ,  0.19285692,  0.19820605,
   0.13883963,  0.0569296 , -0.2964944 ,  0.01679252, -0.37513837,
   0.04110776,  0.06533547],
 [ 0.02546962, -0.17329453,  0.1356675 , -0.06070804, -0.05930689,
  -0.02617825,  0.12306964, -0.06574854, -0.15419105,  0.09948792,
   0.01446443,  0.11894681, -0.04941194, -0.10866308, -0.21836396,
   0.16919982, -0.10534994,  0.2124082 ,  0.02681663,  0.13936134,
  -0.0799686 , -0.0207747 , -0.05147965,  0.06584723, -0.03896057,
   0.0065534 ,  0.2619529 ,  0.06127642, -0.05751181, -0.42364722,
   0.09896686,  0.21100552],
 [ 0.00240206, -0.17064354, -0.3022406 , -0.00461536,  0.00070727,
   0.00068038, -0.28864938, -0.16022548, -0.88085955, -0.00023419,
  -0.00104246, -0.37015262, -0.00221356, -0.00328042, -0.00138044,
  -0.2987121 ,  0.00044654, -0.2976476 , -0.20204751, -0.30961365,
  -0.00061341, -0.00779802,  0.0008707 , -0.004831  , -0.00456054,
   0.00153549, -0.30368578, -0.193792  , -0.15573782, -0.6253438 ,
  -0.29330716, -0.31301108],
 [ 0.43339026, -0.05104205, -0.03232478, -0.075771  ,  0.2485627 ,
   0.18158266,  0.06209979, -0.03587641,  0.15009993, -0.05288746,
   0.0145259 ,  0.13196257,  0.13382058,  0.1461082 , -0.2694374 ,
   0.01986113,  0.03274386, -0.2880629 , -0.33265713, -0.12777327,
   0.05615506, -0.16541381,  0.01244161, -0.11487151, -0.11967778,
   0.08741622,  0.84135365, -0.07426488, -0.02823996, -0.19416411,
  -0.16391511, -0.307363  ],
 [ 0.07425259, -0.22960113, -0.00357497,  0.13385685,  0.04468377,
   0.13325359, -0.07708206, -0.16028921,  0.44228292, -0.01427322,
  -0.02272044,  0.11005637,  0.02647117, -0.00528151,  0.1931235 ,
   0.10774896,  0.00678966,  0.1839204 ,  0.3366686 ,  0.12048384,
  -0.01760417,  0.17708358, -0.08704147,  0.05878699,  0.22110167,
  -0.03587968,  0.5971686 , -0.09637001, -0.08254045, -0.31615564,
   0.08203424,  0.35664785],
 [-0.01102048, -0.21348241, -0.26726446,  0.08472462,  0.01026044,
   0.00237832, -0.38925532, -0.04891031,  0.50908107,  0.00027463,
   0.14526744,  0.02880181,  0.02016989,  0.02135094,  0.15350328,
  -0.22637667, -0.00680986, -0.5217989 , -0.33349076, -0.314115  ,
   0.03076456,  0.15775476,  0.00255907,  0.0839479 ,  0.16608587,
  -0.02138523,  0.36812267,  0.1912602 , -0.220107  , -0.02742822,
  -0.1860485 , -0.62021315],
 [ 0.71080065, -0.40254202, -0.17681666,  0.17777392,  0.42506307,
   0.29219538,  0.6712258 , -0.20440124, -0.46629053,  0.20980391,
   0.00678478,  0.63927364,  0.0548583 ,  0.16436292,  0.08041985,
   0.63121915,  0.03876827,  0.10465305,  0.10007484,  0.59893316,
  -0.02851604,  0.21558672,  0.56429964,  0.12709704,  0.16616467,
   0.37629342, -0.00724626, -0.19816518, -0.28179002, -0.5022378 ,
   0.45643932, -0.17532581],
 [-0.00242984, -0.13466701,  0.23938704,  0.03863667,  0.02472697,
  -0.02247919,  0.09766933, -0.09356027,  0.34543675, -0.08243982,
   0.01963093,  0.18073833,  0.00112157, -0.09695613,  0.05566764,
   0.10712955,  0.02796686,  0.1318142 ,  0.30029854,  0.24074516,
   0.06941742,  0.00804659,  0.00452931,  0.08223582, -0.01277558,
  -0.03339444,  0.5489781 , -0.23278417,  0.00370599, -0.42739743,
   0.16574861,  0.29815024],
 [ 0.00948425, -0.15956965,  0.00026796, -0.099594  , -0.09556539,
  -0.05905214,  0.08963924, -0.1205349 ,  0.36439863,  0.05147021,
  -0.0423254 ,  0.20372131, -0.10145572, -0.04712242, -0.15475315,
   0.09694823,  0.15095276, -0.16290514, -0.18952519,  0.3003511 ,
  -0.04492146, -0.06826195, -0.02042393, -0.02928656, -0.2842844 ,
   0.19100527,  0.5908675 ,  0.01478188,  0.0726456 , -0.5469083 ,
  -0.02468858,  0.09569051],
 [-0.02643955, -0.06950556,  0.10489181, -0.05923236, -0.07634918,
   0.02149787,  0.03374737, -0.0504487 , -0.2488689 ,  0.04306474,
  -0.0887795 ,  0.05125969, -0.07241479, -0.00930643, -0.09935375,
   0.12474035,  0.05326708,  0.26343593, -0.0289159 , -0.03205102,
   0.13187417,  0.12450512,  0.10411638,  0.0497154 , -0.05282494,
  -0.03038791,  0.19366515,  0.08420568,  0.13638407, -0.39758462,
  -0.02729677,  0.01275346],
 [ 0.027337  , -0.32862836, -0.19798082,  0.27481878,  0.2473468 ,
  -0.1454405 , -0.00071801, -0.14616935,  1.1203808 ,  0.4890385 ,
  -0.03761252,  0.52235293, -0.26435712, -0.17756188,  0.10829453,
   0.13811682,  0.01363983, -0.04463844,  0.20784615,  0.25526777,
   0.02186367,  0.11669161,  0.21843418,  0.24379222,  0.17509995,
   0.39967808,  0.60917765, -0.07816563, -0.08473508, -0.49822032,
   0.10059517, -0.18209799],
 [-0.0911924 , -0.0005773 , -0.04052873,  0.00925087, -0.01592517,
   0.05078613,  0.05151213,  0.05276069,  0.04710701, -0.04719859,
   0.1962337 ,  0.1229571 , -0.0239212 ,  0.09688848,  0.49730867,
  -0.08386138, -0.03217131,  0.02072101,  0.19909403,  0.24897395,
  -0.04189098, -0.06417762, -0.02383438, -0.16991623,  0.16183831,
   0.00197936, -0.22264574,  0.02462527,  0.00423612, -0.2535469 ,
   0.00400532,  0.17043698],
 [-0.08292373, -0.18803123,  0.10822531, -0.11634395,  0.05202327,
  -0.01743684, -0.06125281,  0.00995858,  0.19947577,  0.02481344,
  -0.08825167,  0.05106068,  0.04254084,  0.00242748,  0.12018207,
   0.09880782, -0.07504047, -0.000905  ,  0.00598175, -0.05343452,
  -0.07066528,  0.04479877,  0.07369106, -0.11725072,  0.18608499,
  -0.042644  ,  0.3247013 , -0.15869242,  0.06600278, -0.4965211 ,
   0.15484989, -0.02026711],
 [ 0.168465  ,  0.43015215,  0.01928724,  0.23475869, -0.08859048,
  -0.10701527, -0.18767475,  0.61415803,  0.13652691, -0.0607215 ,
  -0.25376552, -0.1378756 ,  0.08149461, -0.11776984,  0.5968713 ,
  -0.26813006, -0.01988092, -0.04501979, -0.06280395,  0.14915752,
   0.12630317,  0.31900656, -0.13186683, -0.11744867,  0.7508027 ,
  -0.25165978,  0.4034503 , -0.2429625 ,  0.6099252 ,  0.23713604,
  -0.07806043, -0.05635923],
 [-0.04310384, -0.08591831, -0.06959559,  0.0343937 ,  0.09435537,
   0.11327992, -0.10970103, -0.04357817,  0.26856747, -0.02099032,
   0.00740728, -0.0203389 , -0.02655283,  0.08368573,  0.14532462,
  -0.10526333,  0.01579103, -0.01582651,  0.01095154, -0.09391548,
   0.02482857, -0.06274256,  0.1181281 , -0.06604511,  0.02430168,
   0.1868492 ,  0.42848793,  0.00263618,  0.1443658 , -0.40697744,
   0.0781953 , -0.08981891],
 [ 0.14472403, -0.14736782, -0.10348341,  0.1539339 ,  0.04423762,
   0.07633909,  0.11472037, -0.00387108,  0.2797847 , -0.01396939,
  -0.21534161,  0.14180955,  0.07720585, -0.09432023,  0.24288177,
   0.21620363,  0.03438001,  0.31101015,  0.34046292,  0.19277525,
  -0.03065093,  0.50870836,  0.11917777,  0.35822   ,  0.2137712 ,
  -0.07040956,  0.18466833, -0.31038937, -0.00881562, -0.3754741 ,
   0.29215515,  0.5486676 ],
 [ 0.36851072, -0.21331954, -0.478574  ,  0.44834968,  0.09552497,
   0.3782497 , -0.4055112 , -0.1863969 ,  0.01634122, -0.2562333 ,
  -0.20269345,  0.22406425,  0.3428969 ,  0.39765644,  0.28230664,
   0.2583832 ,  0.07251617,  0.27408767,  0.26453015, -0.26503333,
  -0.02172316,  0.47781414, -0.1564661 ,  0.50775677,  0.32127026,
  -0.1147785 ,  0.36203638, -0.11950713, -0.2962767 , -0.43358526,
   0.13147809,  0.41986698],
 [ 0.07714214, -0.12424889, -0.29501465,  0.25271863,  0.05439125,
   0.12175141,  0.187132  , -0.01299279,  0.14761461,  0.0960815 ,
   0.06863467,  0.1434237 ,  0.16354378,  0.09596893,  0.16708542,
   0.06691483, -0.08545117,  0.16397567,  0.28473872,  0.20785537,
   0.08776099,  0.20515496, -0.01599493,  0.24133182,  0.18365626,
   0.01606154,  0.15741098, -0.34495136,  0.02591726, -0.37619936,
   0.16676565,  0.30342686],
 [-0.2389817 ,  0.4384117 ,  0.16373062,  0.3725152 , -0.01617949,
  -0.15728076,  0.18886012,  0.4092562 , -1.5776737 , -0.04612629,
  -0.3574201 ,  0.23625991, -0.0291193 , -0.17863277, -0.25987837,
  -1.128432  , -0.20026624, -0.60166687, -0.01334252,  0.59205836,
  -0.12577124,  0.7940165 , -0.2643589 ,  0.8386441 ,  0.28920373,
  -0.4162292 , -0.4053195 ,  0.80469704,  0.6956481 ,  0.4096606 ,
  -1.495605  , -0.3493771 ],
 [-0.06562304,  0.3154829 , -0.2294568 ,  0.12319128,  0.15931007,
  -0.06331257, -0.594577  ,  0.20751148,  0.44038373,  0.1596454 ,
  -0.0978748 , -0.0353414 , -0.11309765, -0.01293465, -0.08081041,
  -0.17074916, -0.09199888, -0.36384827, -0.65254956, -0.6611782 ,
  -0.04315163, -0.37221807,  0.04178314, -0.16838811, -0.41338575,
  -0.04592611,  0.19328047, -0.09285624,  0.49946624, -0.04717472,
  -0.11100629, -0.4948107 ],
 [-0.00873686,  0.038234  ,  0.06670674,  0.21968392, -0.04477157,
   0.05849224, -0.6256132 ,  0.13156308,  0.2094196 ,  0.22900334,
  -0.07100032,  0.21807627,  0.08363256,  0.16682231,  0.2492851 ,
   0.16120675,  0.05121142, -0.02369958,  0.10950027, -0.4905066 ,
  -0.16448265, -0.04274867,  0.26461163, -0.03632831,  0.10438683,
   0.06707012,  0.22523525, -0.5075662 ,  0.22486801,  0.06288285,
   0.05513775,  0.2880152 ],
 [-0.03534518, -0.15343598,  0.73004323,  0.10040794,  0.07244736,
  -0.05549844,  0.40745482, -0.14419399,  0.26713705,  0.04424332,
   0.02362901,  0.39715502,  0.06562674,  0.04676117,  0.19256914,
   0.35408416, -0.07236131,  0.5533152 ,  0.5858267 ,  0.5931894 ,
   0.01049617,  0.20696515,  0.06932006,  0.12275624,  0.19334169,
  -0.05297646,  0.3933946 ,  0.09166235,  0.01917006, -0.43229902,
   0.3903917 ,  0.7071595 ],
 [-0.19124828,  0.27271035,  0.02561059,  0.29680336, -0.16572165,
  -0.22649239, -0.11305499,  0.32756358,  0.35136738,  0.08760855,
  -0.19061694,  0.49691075, -0.31674471, -0.21707006, -0.06469793,
   0.03601103,  0.02324296, -0.08643115,  0.19995305,  0.03676152,
   0.02911306,  0.07051853, -0.1561543 ,  0.09632121, -0.28707436,
  -0.07412766,  0.07249375, -0.27751058,  0.3014242 ,  0.09806594,
  -0.02486427,  0.23880108],
 [ 0.61454046,  0.00260672,  0.2311225 , -0.17561318,  0.5557594 ,
   0.136839  ,  0.03163439,  0.0945702 , -0.8669202 ,  0.25817075,
   0.0599804 ,  0.0167    , -0.08022079,  0.03219945, -0.15094064,
   0.12242775,  0.10908265,  0.35323694,  0.42356005, -0.10205328,
  -0.00361753,  0.08592704,  0.27973434,  0.05275511, -0.16027717,
   0.17414884, -0.2969526 ,  0.63389313,  0.23336281, -0.21861996,
   0.21934813,  0.49747744],
 [-0.08674793, -0.08681975,  0.21002448, -0.05528463,  0.00693998,
  -0.00142952,  0.40297675,  0.00091194,  0.3609057 ,  0.06848256,
  -0.00030163,  0.18807599, -0.05363193, -0.0417062 , -0.16659829,
   0.4090474 ,  0.01113043,  0.4323945 ,  0.2876754 ,  0.26549968,
   0.04649762, -0.09462488,  0.10494885, -0.09912392, -0.04925422,
   0.07255039, -0.17684713, -0.08297655, -0.13888517,  0.04775969,
   0.23983161,  0.45897254],
 [-0.04976591, -0.15585129,  0.29563484, -0.10142937, -0.08880441,
   0.06081167,  0.00803152, -0.17474186,  0.36139366, -0.03031097,
   0.06977507,  0.00659634, -0.00603883, -0.08135432, -0.11656141,
  -0.20010184, -0.03829918, -0.03203795,  0.0171981 ,  0.04531528,
  -0.0034376 , -0.2821976 ,  0.11570758, -0.21064669, -0.15289983,
  -0.00862298,  0.42538232, -0.01981024, -0.03360141, -0.5160703 ,
  -0.07384919,  0.09352955],
 [ 0.01185691, -0.19060785, -0.0501283 , -0.18081726,  0.09037344,
   0.13134886, -0.05218925, -0.07026865,  0.28358182,  0.07849211,
   0.14143181, -0.24637061,  0.00658165,  0.12733358, -0.4160298 ,
  -0.01977249,  0.01644478, -0.20664687, -0.23808573, -0.21787247,
  -0.00216114, -0.14468007,  0.04165632, -0.23045795, -0.25564185,
   0.0374239 ,  0.47746152, -0.09127363, -0.05504974, -0.4928716 ,
  -0.03809918, -0.2565488 ],
 [ 0.06090965, -0.08901814,  0.0825015 , -0.10926963, -0.01079372,
  -0.00219556,  0.02709676, -0.07340379,  0.6987561 , -0.0457253 ,
  -0.2323104 , -0.39752832,  0.03566165, -0.00727516, -0.17108789,
  -0.23637936,  0.00229915,  0.21844578,  0.12220484,  0.03485516,
  -0.07583515, -0.26132512, -0.05345249, -0.15333578, -0.24807622,
  -0.01712336, -0.2736301 , -0.6018971 ,  0.06327318, -0.02922382,
  -0.22385564,  0.37052545],
 [-0.04507609,  0.01877483,  0.13834466, -0.00182819,  0.01974682,
   0.04570778,  0.39979142,  0.07222942,  0.11557329, -0.02858942,
   0.06868105,  0.13105878,  0.03031727,  0.0670993 , -0.08961977,
   0.1669448 ,  0.01729056,  0.11359043,  0.23635356,  0.2463747 ,
  -0.00181207, -0.10516178, -0.01000131,  0.01060655, -0.12646294,
  -0.00629027,  0.57972986, -0.3903987 , -0.13545597, -0.08091475,
   0.15202515,  0.07270694],
 [ 0.01995953, -0.25631773, -0.02892466, -0.05160629, -0.01488431,
   0.03939895,  0.20431048, -0.26077563,  0.6145034 , -0.09632812,
  -0.08172461,  0.08743086, -0.03691423,  0.05494434, -0.11596204,
   0.14339624, -0.11151137,  0.05378966, -0.11222953,  0.1159751 ,
  -0.09359568, -0.0422531 ,  0.00203631, -0.00754208, -0.14705142,
  -0.03303627,  0.54686546, -0.08633029, -0.14833933, -0.58748996,
   0.07178029, -0.0735745 ],
 [ 0.07078438, -0.04408546, -0.07014857, -0.08181579, -0.00811109,
  -0.13127242,  0.09332802, -0.12994507,  0.28353044,  0.04679275,
   0.06144939, -0.01917182, -0.0461046 ,  0.04424863,  0.06068646,
  -0.02461649,  0.04200576, -0.09252694, -0.09413642, -0.05065993,
   0.10046809, -0.05301272, -0.05357192, -0.07215934, -0.04737324,
  -0.02416777,  0.6174615 ,  0.10237331, -0.03703265, -0.49877992,
  -0.08695786,  0.11420567],
 [-1.6077646 ,  0.46627846, -0.04324371, -0.00419383, -1.574088  ,
  -1.3752472 ,  0.00692294,  0.31831887,  0.32888243, -1.0305116 ,
   0.00871521,  0.02872532, -1.1704432 , -1.470525  ,  0.00848928,
   0.01857488,  0.00192549, -0.03169803, -0.0244723 , -0.01354535,
   0.00288797, -0.01533535, -1.04643   , -0.01037141, -0.00931722,
  -1.0128351 ,  0.22997376, -0.01190468,  0.3457111 ,  0.6278228 ,
   0.01326536, -0.05416708],
 [-0.01098804, -0.1427147 , -0.14396767,  0.1990017 , -0.00693105,
  -0.01130075, -0.07028183,  0.0048066 ,  0.68169147, -0.02823303,
  -0.08888432,  0.01355839, -0.06844708, -0.01618063,  0.16165711,
  -0.01453786, -0.00768575, -0.02437683, -0.11542502,  0.07384513,
  -0.01244431,  0.15066054, -0.00288769,  0.0535352 ,  0.22498073,
   0.00996366,  0.4766599 , -0.14550722,  0.0389724 , -0.19150963,
  -0.11988308, -0.01922508],
 [-0.06907128, -0.16274701,  0.071628  ,  0.07825939,  0.0082493 ,
   0.02153384,  0.36473352,  0.02205427,  0.29190633,  0.2008197 ,
  -0.05630418,  0.1485118 ,  0.25797924,  0.08525196, -0.05388075,
   0.1355147 , -0.11906776,  0.1926734 ,  0.10148598,  0.28854072,
   0.00798489, -0.02823721,  0.18037048,  0.05675595,  0.0705706 ,
   0.13495909,  0.38544214,  0.01415539,  0.11022895, -0.42059523,
   0.06593846,  0.26446432],
 [-0.21818429, -0.065575  , -0.56519413, -0.00106317, -0.0200951 ,
   0.03256189,  0.22745033, -0.11562287,  0.11205104, -0.07173974,
  -0.16444649,  0.0188137 ,  0.06403063,  0.06074254, -0.13610806,
  -0.06859671, -0.01515766, -0.03838465, -0.04300407,  0.2939803 ,
  -0.0931384 ,  0.00352848, -0.18970954, -0.06482861, -0.10924638,
  -0.1067945 ,  0.0159084 , -0.08151645, -0.02067257, -0.08839524,
  -0.06332887,  0.08678983],
 [ 0.06334697, -0.12770586, -0.17281233,  0.00400289, -0.04867957,
  -0.16594726, -0.04253467, -0.01344938,  0.42014444, -0.14078167,
   0.0046675 , -0.00422105, -0.17556632, -0.014966  ,  0.00896438,
  -0.01593641, -0.02725484,  0.16199227, -0.11377895, -0.05471113,
  -0.04524704, -0.03210369, -0.08939935,  0.01550012, -0.04446506,
  -0.13220297,  0.8338433 , -0.30401886,  0.00438995, -0.2463713 ,
  -0.05255314,  0.10941935],
 [ 0.0331951 , -0.22183572,  0.37910178,  0.12865019,  0.05466224,
  -0.01440347, -0.08223947,  0.01743291,  0.48700324, -0.02181765,
  -0.26750407,  0.00197917,  0.06614027, -0.12521437,  0.06752937,
  -0.14025776,  0.00594568,  0.13059787,  0.06876165,  0.03728735,
   0.0200009 , -0.08652846, -0.06601083,  0.00085206,  0.09985229,
   0.00331472,  0.18744849,  0.14051202,  0.10173707, -0.4854193 ,
  -0.29280168,  0.10618265],
 [-0.00199015, -0.04921474,  0.33689854, -0.05123723,  0.03623686,
   0.01808639, -0.4036419 , -0.01990036, -0.20227902,  0.09035955,
   0.02995463, -0.23118214,  0.09595265,  0.06218635, -0.02745525,
  -0.32709926,  0.01619087, -0.14283454, -0.07428053, -0.20492896,
   0.04567437, -0.04687453,  0.11822176, -0.00027477, -0.02887681,
   0.00288263,  0.38380328, -0.18611442,  0.13795057, -0.36740044,
  -0.24334368, -0.12012859],
 [-0.1519952 , -0.36080402, -0.16748935, -0.13187401, -0.0078418 ,
   0.03079316,  0.10536237, -0.281313  ,  0.3630281 ,  0.02010261,
  -0.02975025,  0.01700593, -0.03333752, -0.1418164 ,  0.12775172,
  -0.08036629,  0.01785355, -0.12642108,  0.05107697,  0.02239097,
  -0.04665941, -0.28259268,  0.11557061, -0.20722984, -0.02964713,
   0.07417144,  0.47481278,  0.05174686, -0.15969929, -0.5531078 ,
   0.12368369, -0.18539588],
 [ 0.14761963, -0.15734811, -0.63647723, -0.12925765, -0.00792165,
  -0.10925934,  0.12382041, -0.07998448,  0.40630618,  0.40291998,
   0.07483883,  0.19405767,  0.03449603,  0.13860625, -0.04564117,
  -0.12578055,  0.01660612, -0.4429865 , -0.4404476 ,  0.03600973,
  -0.26104975, -0.48140538,  0.5088393 , -0.30085644, -0.14310516,
   0.32005948,  0.05927234, -0.535814  , -0.11388704, -0.5439357 ,
  -0.26664048, -0.2966654 ],
 [-0.1777904 ,  0.42797205,  0.4566211 , -0.0887313 , -0.23855422,
  -0.00679262,  0.1389475 ,  0.37916747, -1.6724573 ,  0.03522712,
  -0.10478961, -0.00091397,  0.06453238, -0.28424537, -0.3101842 ,
  -0.13770883,  0.12831266, -0.06122641, -0.21182975,  0.08210311,
   0.12704997,  0.3189515 ,  0.18453078,  0.23336071, -0.01405207,
   0.15739354,  0.23391068,  0.23090105,  0.47579786,  0.21491112,
  -0.36542234, -0.12958209],
 [ 0.06135964, -0.05146096, -0.1106101 , -0.37541568, -0.01020953,
  -0.437336  ,  0.11501287, -0.02008243, -0.6339416 , -0.31607857,
   1.0771285 , -0.60040665, -0.6687626 , -0.24100456, -0.86639726,
  -0.91660476, -0.14855078, -1.4598645 , -0.6139617 ,  0.31756905,
   0.08266172, -0.1742532 , -0.0757224 ,  0.52184457, -0.2103291 ,
  -0.23675933, -0.14838792,  0.15358628,  0.15250543, -0.14170645,
  -1.2254844 , -1.0613061 ],
 [-0.18223895,  0.11185812,  0.00139517,  0.4002358 , -0.02017323,
  -0.30879614, -0.2108494 ,  0.15460083,  0.9658894 ,  0.11232759,
  -0.29334822,  0.16029961, -0.27835074, -0.20466657,  0.14705352,
   0.09949083,  0.00645474, -0.08475633,  0.3399641 ,  0.1354756 ,
  -0.15821284,  0.03589886,  0.1368204 ,  0.38024226,  0.15401724,
   0.19447675, -0.04043113, -0.6935264 ,  0.25203562, -0.09718336,
   0.08667403,  0.6950308 ],
 [ 0.525113  , -0.52338165, -0.40948257, -0.36461693,  0.67610365,
   0.6598666 ,  0.14927855, -0.28234887,  0.26440355,  0.07596748,
   0.1412447 , -0.3813945 ,  0.31895688,  0.5863932 ,  0.05914096,
  -0.02249263,  0.07129146, -0.34979916, -0.37220153, -0.47040838,
  -0.00663882, -0.1991143 ,  0.3042262 , -0.34917668, -0.18057191,
   0.3903431 , -0.01436405, -0.22774175, -0.24601834, -0.6057695 ,
   0.15248986, -0.6166762 ],
 [ 0.16689765,  0.03580084,  0.24913158,  0.3468019 ,  0.07005915,
   0.08722457,  0.11893636,  0.0768066 ,  0.17479134, -0.0252428 ,
  -0.0273295 ,  0.04723946,  0.00055749,  0.01594471,  0.16988826,
   0.20042528, -0.02048585,  0.5167388 ,  0.5027247 ,  0.3886876 ,
  -0.01023305,  0.47800758, -0.06862749,  0.5518005 ,  0.2574508 ,
  -0.00848254,  0.28359213, -0.10266152,  0.1525702 , -0.27196303,
   0.17524274,  0.7445416 ],
 [ 0.0760666 , -0.00106366,  0.7124227 ,  0.20146471,  0.00227192,
   0.01918538,  0.61308867,  0.08718965,  0.38484415, -0.01271802,
   0.10855946,  0.40317634,  0.0076259 ,  0.06295022,  0.20882374,
   0.6655973 , -0.0209835 ,  1.2270173 ,  1.0872339 ,  0.65463567,
   0.03828657,  0.5068692 , -0.02543038,  0.3629716 ,  0.22842291,
  -0.07429861,  0.07532062,  0.47654685,  0.06852961, -0.30951086,
   0.84192187,  1.13782   ],
 [ 0.05425581, -0.10525372, -0.08657026, -0.15103059,  0.01946576,
   0.03239654, -0.12637147, -0.0178704 ,  0.30911988, -0.09315188,
  -0.00410465, -0.10149132,  0.01610267, -0.01248016, -0.16256636,
   0.07586934,  0.08949598, -0.14026222, -0.17792256, -0.24538594,
   0.08787257, -0.12557288,  0.11117146, -0.09451919, -0.13184392,
   0.10918686,  0.51273   , -0.02644573,  0.07943854, -0.45435557,
   0.12758957, -0.18661655],
 [-0.13739207,  0.1824432 ,  0.1577861 ,  0.2625775 , -0.18829206,
  -0.24565965, -0.40737796,  0.21946576,  0.1363118 ,  0.05718884,
  -0.28032687,  0.53559273, -0.37893412, -0.42892098, -0.08851499,
   0.44963872,  0.0346645 ,  0.17614412,  0.18787892, -0.08755981,
  -0.08768041, -0.29902115,  0.1344811 ,  0.03861897, -0.1616023 ,
   0.15187164,  0.11186491, -0.23843965,  0.30862665, -0.00528042,
   0.20723394,  0.18122762],
 [-0.35723162,  0.18217608,  0.616828  , -0.24735758, -0.2977228 ,
  -0.40709567,  0.98871005,  0.25775072,  0.0245236 ,  0.17096739,
  -0.52393997,  0.4783085 , -0.44138175, -0.25997877, -0.96229595,
   0.6534597 , -0.01435372,  0.506436  ,  0.29883373,  0.65310353,
   0.12175356, -0.69809955,  0.09850815, -0.21353307, -0.74961185,
   0.15410614, -0.7346698 , -0.0524316 ,  0.40741497,  0.0059496 ,
   0.6608013 ,  0.22275987],
 [-0.01973238,  0.03494603,  0.5262741 ,  0.12549612, -0.04177873,
  -0.00158206,  0.46387273,  0.00491325,  0.2125642 , -0.02732118,
  -0.07392196,  0.30058122, -0.00827009,  0.04756051, -0.01810238,
   0.4109462 ,  0.01071244,  0.45873544,  0.35552195,  0.5882646 ,
   0.02496885,  0.01636334, -0.10874464,  0.11116503,  0.11234371,
   0.02558592,  0.38705018,  0.3529855 ,  0.0344396 , -0.3968197 ,
   0.35499913,  0.82534724],
 [-0.06012744, -0.04839467,  0.10791235, -0.14500429,  0.04759474,
  -0.02468618,  0.05407177, -0.05150427,  0.34103513,  0.00421094,
   0.00365349,  0.02779216,  0.12793946, -0.02336623, -0.01431776,
   0.02439512,  0.02296533,  0.10225993,  0.1164009 ,  0.03665571,
   0.01001242, -0.10405924, -0.05162436, -0.01702207, -0.09953764,
  -0.01338285,  0.61611164, -0.01415609, -0.0759929 , -0.09826576,
   0.07594681,  0.16107251],
 [ 0.02822078, -0.07899215,  0.17711239,  0.04978858, -0.06863002,
  -0.04128632,  0.04589863,  0.04488108,  0.5143478 , -0.00853916,
  -0.0241153 , -0.04939031,  0.02357513, -0.05418321, -0.27516243,
   0.08529363, -0.04206523,  0.07164103, -0.02554093,  0.03175871,
  -0.04138558,  0.229472  , -0.06347856,  0.18281224, -0.07875878,
  -0.00383393,  0.00264038, -0.07358799,  0.1745439 , -0.34099075,
  -0.10054812,  0.24984792],
 [ 0.11152804,  0.4557636 ,  0.84932303,  0.61294883, -0.10365763,
  -0.25165337, -0.35404932,  0.5622106 , -1.2909315 ,  0.03987352,
   0.8049299 ,  0.40938577, -0.31406635, -0.27758253, -0.27258143,
  -1.2477539 , -0.10696864, -0.56828785,  0.28785387,  0.56918675,
  -0.12435094,  1.0846    ,  0.04718252,  1.015265  ,  0.41472024,
   0.06925178,  0.6655212 ,  0.8061525 ,  0.68375266,  0.4119901 ,
  -1.4220874 ,  0.38456514]])
    b5 = jnp.array([-0.01163196,  0.33890554, -0.3824479 , -0.74075985, -0.0035078 ,
 -0.00386348, -0.49023822,  0.2833634 , -0.96138704, -0.00540846,
 -0.7480957 , -0.5793471 , -0.00463537,  0.00393785, -0.7456158 ,
 -0.5296091 , -0.00202509, -0.41367808, -0.45154923, -0.42586827,
  0.00270197, -0.73329765, -0.00614027, -0.736525  , -0.74394035,
 -0.00738524, -0.4056412 , -0.73545146,  0.20465788,  0.60969305,
 -0.5302713 , -0.28546265])
    W6 = jnp.array([[ 0.00219193],
 [-0.18533635],
 [ 0.49728268],
 [ 0.47427925],
 [ 0.00073047],
 [-0.00059285],
 [ 0.46024415],
 [-0.1082835 ],
 [ 0.49245784],
 [-0.01125484],
 [ 0.39419237],
 [ 0.5463636 ],
 [-0.02964754],
 [-0.02045851],
 [ 0.4708536 ],
 [ 0.47879687],
 [ 0.00001175],
 [ 0.4741732 ],
 [ 0.33992353],
 [ 0.43204546],
 [-0.00000674],
 [ 0.33500683],
 [-0.00355704],
 [ 0.36059245],
 [ 0.33549404],
 [-0.00020603],
 [ 0.37705225],
 [ 0.3674953 ],
 [-0.1147306 ],
 [-0.37126017],
 [ 0.56784123],
 [ 0.4685979 ]])
    b6 = jnp.array([-0.42517364])

    x = jnp.array([beta_st, log_pc] + list(log_ps_nsat))
    x = (x - mean_X) / scale_X

    h1 = jnn.gelu(jnp.dot(x, W1) + b1)
    h2 = jnn.gelu(jnp.dot(h1, W2) + b2)
    h3 = jnn.gelu(jnp.dot(h2, W3) + b3)
    h4 = jnn.gelu(jnp.dot(h3, W4) + b4)
    h5 = jnn.gelu(jnp.dot(h4, W5) + b5)
    out_log = jnp.dot(h5, W6) + b6

    return jnp.exp(out_log[0])
