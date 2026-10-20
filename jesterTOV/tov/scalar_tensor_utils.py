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
    """Weighted log-Huber central-field surrogate; exact GELU."""
    mean_X = jnp.array([-14.54287538652345 , -10.752340278915986, -14.636147955284493,
 -14.433487580822428, -14.220294473204056, -13.997446234410035,
 -13.769427319457987, -13.547335586047108, -13.324996402013475,
 -13.100861404978875, -12.876535685112868, -12.634618861795794,
 -12.358539374132327, -11.95026091943837 , -11.509049291978169,
 -11.093321767668884, -10.687727645019537, -10.326165125834994,
 -10.01267854206837 ,  -9.803698821831963,  -9.70006517270932 ,
  -9.60056744465621 ])
    scale_X = jnp.array([8.662297643069087  , 1.6336428980815176 , 0.00247053338811789,
 0.00169017452050849, 0.00211802113173794, 0.00227443613149136,
 0.00373159180202785, 0.00237494646527216, 0.00223952983582551,
 0.00303703073796803, 0.00196436322909407, 0.00308490267661302,
 0.01289816132533392, 0.13098429453603594, 0.2741039306585021 ,
 0.3939019051203402 , 0.5331361869647873 , 0.6613396028515693 ,
 0.7469822050288989 , 0.7639808955703924 , 0.7138464640096275 ,
 0.6764850439240477 ])
    W1 = jnp.array([[   0.3235401511192322 ,    0.09647341072559357,    0.4414794445037842 ,
     0.8700332641601562 ,   -0.07817842066287994,   -0.2786369025707245 ,
     0.2736990451812744 ,    1.5797244310379028 ,   -0.10401910543441772,
    -0.1356429159641266 ,   -0.27288341522216797,    1.0111722946166992 ,
    -0.8944127559661865 ,    0.16562886536121368,   -0.00805030763149261,
     0.01025124825537205,   -0.01135518588125706,   -0.06774888932704926,
    -1.4954173564910889 ,   -0.7554870247840881 ,   -0.6454723477363586 ,
    -0.05570025742053986,   -0.17265605926513672,    0.02032917179167271,
     0.4212918281555176 ,   -0.6838701367378235 ,   -0.00041919591603801,
     0.0082193985581398 ,   -0.03316446393728256,   -0.6120107173919678 ,
    -0.9202948212623596 ,    0.05215113610029221,    0.08883798122406006,
    -0.514837384223938  ,   -0.29807668924331665,   -0.19787746667861938,
     0.30029866099357605,   -0.42292144894599915,    2.058427333831787  ,
    -0.18644459545612335,    2.0216188430786133 ,    0.35399124026298523,
     0.0255623571574688 ,   -0.07881700247526169,   -0.13553285598754883,
    -0.2622988522052765 ,    0.3134237229824066 ,    0.6008093953132629 ,
     2.298854351043701  ,    0.04773155599832535,   -0.21108685433864594,
    -0.08801910281181335,    0.04183637723326683,   -0.03267627209424973,
    -0.20496737957000732,   -0.1687784045934677 ,   -0.0350828617811203 ,
    -0.15753409266471863,   -0.09786444902420044,   -0.01461850292980671,
     0.9019192457199097 ,   -0.17128875851631165,   -0.12294027954339981,
     0.819306492805481  ,   -0.07874542474746704,    1.7722235918045044 ,
    -0.09036281704902649,   -0.00832100585103035,    0.3310766816139221 ,
     0.5781238675117493 ,    2.4682998657226562 ,    0.3715590536594391 ,
    -0.06779088824987411,   -0.06899934262037277,   -0.5716758966445923 ,
    -0.39959463477134705,   -0.33664971590042114,    1.1894372701644897 ,
     0.8499658703804016 ,    0.9642254710197449 ,   -1.1115834712982178 ,
     0.03539396822452545,    0.6633527278900146 ,   -0.00441805273294449,
    -0.744830310344696  ,   -0.04409921914339066,   -0.07795627415180206,
    -1.3528515100479126 ,   -0.06394115835428238,   -0.07481912523508072,
     2.1020731925964355 ,   -0.15140822529792786,   -0.00633761845529079,
    -0.6368257403373718 ,    0.31759366393089294,   -0.7605140209197998 ,
     0.11041595041751862,    3.0415236949920654 ,   -0.3851177990436554 ,
    -0.00325741153210402,   -0.23606650531291962,   -0.07244314253330231,
     1.2613393068313599 ,    0.07115396112203598,   -0.5575333833694458 ,
    -0.10490630567073822,    0.03616776317358017,   -0.7014011740684509 ,
    -0.214799165725708  ,    0.3378877639770508 ,   -0.22175630927085876,
    -0.12699998915195465,    0.8991041779518127 ,    1.0574926137924194 ,
     1.0135581493377686 ,   -0.24796175956726074,   -1.120591402053833  ,
    -0.02053960040211678,    2.7247841358184814 ,    0.9223350286483765 ,
    -0.01251574512571096,   -0.2786213457584381 ,   -0.00369421811774373,
    -0.26814863085746765,    0.13922615349292755,   -0.4962942600250244 ,
    -0.01636706106364727,   -0.7582881450653076 ],
 [   1.5680242776870728 ,    0.6433872580528259 ,   -0.42174020409584045,
    -0.6427375674247742 ,   -1.3000775575637817 ,   -3.5953149795532227 ,
    -0.9948808550834656 ,   -0.2871391773223877 ,   -1.9014488458633423 ,
     1.332202434539795  ,   -0.46973633766174316,    0.24685095250606537,
    -0.5541859865188599 ,   -0.03622083365917206,    0.5550100207328796 ,
    -1.1087816953659058 ,   -1.981948971748352  ,    0.49110960960388184,
    -0.10065975785255432,    1.0213125944137573 ,    0.06521390378475189,
    -1.3695188760757446 ,    1.5463758707046509 ,   -0.3092871308326721 ,
    -0.6612340807914734 ,   -0.29386138916015625,    0.05325577035546303,
    -1.3410110473632812 ,   -2.71087384223938   ,   -2.5783681869506836 ,
    -1.8080724477767944 ,   -1.7965691089630127 ,   -2.9038279056549072 ,
     0.8458805084228516 ,    0.886478841304779  ,    0.5444660186767578 ,
    -1.322633981704712  ,   -0.6217834949493408 ,    0.294048547744751  ,
     0.40689313411712646,   -2.6398818492889404 ,   -0.6047950387001038 ,
     0.36124151945114136,   -0.38153892755508423,   -0.9241989254951477 ,
    -0.8376589417457581 ,    0.19292095303535461,   -0.11613889783620834,
    -0.16459135711193085,    0.3129216432571411 ,   -0.03854730352759361,
     1.330000638961792  ,    0.6312711238861084 ,   -0.03046780824661255,
    -2.078655242919922  ,    0.10295756161212921,   -0.10349023342132568,
    -1.990501880645752  ,    0.7369651198387146 ,    0.00102631165646017,
    -1.8339122533798218 ,    1.2410653829574585 ,    0.3850359320640564 ,
    -1.611498475074768  ,    0.12012180685997009,    0.16443856060504913,
    -1.3346918821334839 ,   -0.06419132649898529,   -2.2948124408721924 ,
    -0.6801518201828003 ,   -2.623107671737671  ,    0.713826060295105  ,
    -0.19458653032779694,   -1.538517713546753  ,   -1.5285437107086182 ,
    -0.0908784419298172 ,   -0.7332558035850525 ,   -5.264030456542969  ,
    -0.7757327556610107 ,   -0.8267545104026794 ,   -0.01538168638944626,
    -2.2274482250213623 ,    1.1455363035202026 ,   -0.9350182414054871 ,
    -0.6648001074790955 ,   -0.06067362055182457,   -0.6492646336555481 ,
    -0.9574983716011047 ,    0.7838760614395142 ,    0.24525000154972076,
    -0.2794913053512573 ,    0.4234665334224701 ,   -0.6717074513435364 ,
    -0.3022783100605011 ,    0.7366883158683777 ,    1.351266622543335  ,
    -2.239933729171753  ,    0.00069172214716673,   -1.2586606740951538 ,
     0.01526173483580351,    0.00499854562804103,    0.10434257984161377,
    -1.3508330583572388 ,    0.8759322166442871 ,   -1.475502371788025  ,
     0.7740755081176758 ,   -2.8849611282348633 ,    0.44864600896835327,
     0.06764014810323715,   -0.10273536294698715,   -0.33964595198631287,
     1.1453746557235718 ,   -1.6715365648269653 ,    0.4695022404193878 ,
    -0.7614939212799072 ,   -0.36778947710990906,    0.08087021857500076,
    -0.54168701171875   ,   -0.25591278076171875,   -0.04057894647121429,
     0.00010212138295174,   -0.532202959060669  ,    0.21247465908527374,
     0.46867305040359497,    0.02751837484538555,   -0.4657539129257202 ,
    -0.036445502191782  ,   -1.7708508968353271 ],
 [  -0.2452215850353241 ,   -0.48729023337364197,   -1.2959165573120117 ,
     1.56047523021698   ,   -0.5405785441398621 ,   -0.02608188986778259,
    -0.5877805352210999 ,   -0.3942735195159912 ,   -0.03107999265193939,
    -0.3536767363548279 ,   -0.15043330192565918,    0.04380914568901062,
     0.1648532748222351 ,   -2.273130416870117  ,    2.7844135761260986 ,
     0.26446789503097534,    0.7691447734832764 ,    1.2757171392440796 ,
     0.1855621188879013 ,   -0.00078538060188293,    0.5246874094009399 ,
    -0.12185728549957275,    0.31142348051071167,    1.1815342903137207 ,
    -2.8928728103637695 ,   -0.30431005358695984,    1.8795833587646484 ,
    -1.1201822757720947 ,    0.36080920696258545,    0.02654685080051422,
    -0.1374850571155548 ,    3.1435980796813965 ,    0.11074519157409668,
     0.2591915428638458 ,   -0.4372558891773224 ,    1.7952824831008911 ,
     0.12243558466434479,    0.15674695372581482,    0.4649786353111267 ,
     0.3767858147621155 ,    0.22056668996810913,    1.8785383701324463 ,
    -3.513139486312866  ,   -2.421257257461548  ,   -0.4319056272506714 ,
    -5.240960597991943  ,    1.6713829040527344 ,   -0.00642943382263184,
    -0.9751543402671814 ,   -0.554079532623291  ,    1.1696500778198242 ,
     1.1929739713668823 ,    0.18803828954696655,   -1.1896274089813232 ,
     2.5095765590667725 ,    0.28991591930389404,    1.082788109779358  ,
    -0.01890384405851364,   -0.5066268444061279 ,   -3.2358498573303223 ,
    -0.32734233140945435,    0.7902933359146118 ,   -1.400593876838684  ,
     2.9777398109436035 ,   -1.6169464588165283 ,   -0.32030627131462097,
    -0.05123719573020935,    1.3885525465011597 ,   -0.329449325799942  ,
     0.5242802500724792 ,   -0.09094177931547165,   -0.15913930535316467,
     2.8181984424591064 ,   -0.8320783376693726 ,    3.4922001361846924 ,
    -0.6573691964149475 ,    1.0674045085906982 ,    0.00193321704864502,
    -1.505121111869812  ,   -0.13485193252563477,    1.2964341640472412 ,
     0.6363633871078491 ,    0.4244168996810913 ,    1.9892700910568237 ,
    -0.7016652822494507 ,   -1.5737978219985962 ,    1.7341853380203247 ,
     0.10037021338939667,    0.7161504626274109 ,   -1.9606801271438599 ,
     0.16237613558769226,    0.13147839903831482,    0.6039003133773804 ,
     2.073664426803589  ,    1.3461077213287354 ,    0.11287671327590942,
    -0.00432631373405457,    0.03060111403465271,   -2.3459832668304443 ,
     2.1800832748413086 ,    0.08782696723937988,    2.1348161697387695 ,
    -0.06056010723114014,    0.438870370388031  ,    0.27558574080467224,
     1.7024877071380615 ,    0.0919257402420044 ,   -0.20119456946849823,
     1.157353162765503  ,    0.5727182626724243 ,   -2.4453039169311523 ,
     0.17355692386627197,    1.6369860172271729 ,    0.06472647190093994,
     0.8972316980361938 ,    1.0962644815444946 ,    0.3605363965034485 ,
     0.5458550453186035 ,    0.07489720731973648,   -0.08833800256252289,
    -4.560243606567383  ,   -0.7079702615737915 ,   -2.205958127975464  ,
     0.83107590675354   ,    1.137789011001587  ,    0.9870117902755737 ,
     8.315164566040039  ,    2.899717330932617  ],
 [   0.00028103590011597,    0.7461975812911987 ,    1.1389950513839722 ,
     1.0780646800994873 ,    0.7746051549911499 ,    0.1119391918182373 ,
     1.0646368265151978 ,    0.3851497769355774 ,   -0.09530174732208252,
     0.7353774905204773 ,    0.5558468103408813 ,    0.530955970287323  ,
     0.9369126558303833 ,    3.0968308448791504 ,   -5.055638313293457  ,
     0.15872621536254883,   -1.419974446296692  ,   -1.246889352798462  ,
     0.28279322385787964,    0.07747006416320801,   -0.8235236406326294 ,
     0.790787935256958  ,   -0.19544732570648193,   -0.2666003108024597 ,
     2.0895113945007324 ,    0.7623236179351807 ,   -6.302937030792236  ,
     1.429389238357544  ,   -0.47494685649871826,   -0.24279841780662537,
     0.2656933069229126 ,   -1.3965587615966797 ,    0.10245352983474731,
    -0.2714400291442871 ,    0.48984819650650024,   -1.9044526815414429 ,
    -3.4569451808929443 ,    0.4192337691783905 ,   -0.09750479459762573,
    -0.6170522570610046 ,   -0.42450404167175293,   -3.6938016414642334 ,
     5.669314861297607  ,    6.704859733581543  ,    0.01516234874725342,
     5.271999359130859  ,    3.1677181720733643 ,    2.6639199256896973 ,
     0.8519004583358765 ,    1.4856935739517212 ,   -1.3857524394989014 ,
    -1.8825968503952026 ,   -0.2577667236328125 ,    1.0987443923950195 ,
    -1.177019715309143  ,   -0.5314933061599731 ,    5.737965106964111  ,
    -0.22783762216567993,    0.45612943172454834,    7.816483974456787  ,
     0.02261894941329956,   -1.6105989217758179 ,    3.7335400581359863 ,
     0.1919494867324829 ,    0.6663545966148376 ,   -0.6674503087997437 ,
     0.6366640329360962 ,   -2.5440750122070312 ,    0.27142012119293213,
    -0.7354933023452759 ,    0.0366329625248909 ,    0.14923042058944702,
    -4.248227119445801  ,    0.5173201560974121 ,   -2.133674144744873  ,
     1.2860312461853027 ,   -0.9597911834716797 ,    0.16633367538452148,
     1.1784175634384155 ,    0.3177211284637451 ,   -2.828735828399658  ,
    -0.3260284662246704 ,   -0.30517396330833435,   -4.427184104919434  ,
     1.7213363647460938 ,    2.745394706726074  ,    0.7600034475326538 ,
    -0.5973278284072876 ,   -0.232161283493042  ,    4.739258289337158  ,
     0.5044716596603394 ,   -0.8086121678352356 ,   -0.17652368545532227,
    -2.4710540771484375 ,   -1.9621567726135254 ,   -0.8834546804428101 ,
     0.06417977809906006,   -0.04398900270462036,    3.306384801864624  ,
    -2.2902510166168213 ,    2.1188392639160156 ,   -2.2877752780914307 ,
    -0.07122474908828735,   -0.42021864652633667,   -1.1825048923492432 ,
    -2.4345107078552246 ,    0.02281856536865234,    0.11145565658807755,
    -1.526382565498352  ,    0.8807355761528015 ,    0.8139220476150513 ,
    -0.1098138689994812 ,   -0.8436017036437988 ,    1.0467209815979004 ,
     3.3418829441070557 ,   -2.053788661956787  ,   -0.3123432695865631 ,
    -0.7161558866500854 ,   -0.3241058588027954 ,   -2.586402177810669  ,
     9.695926666259766  ,   -0.836473822593689  ,   -1.3869566917419434 ,
    -1.8812767267227173 ,   -0.9329414963722229 ,   -1.7269285917282104 ,
    -8.710051536560059  ,   -2.1149444580078125 ],
 [  -0.00790856778621674,   -0.5170864462852478 ,   -0.8266687989234924 ,
    -2.5312960147857666 ,   -0.29277971386909485,    0.00059128552675247,
    -0.4883902370929718 ,    0.461965411901474  ,   -0.15949636697769165,
    -0.10650072991847992,    0.280374139547348  ,   -0.31382712721824646,
    -0.8746001720428467 ,   -1.185620665550232  ,    1.543508529663086  ,
    -0.3495378792285919 ,    1.8847845792770386 ,    2.0244579315185547 ,
    -0.46068111062049866,    0.03665529191493988,    0.14956584572792053,
    -0.4797045886516571 ,   -0.07450643181800842,   -0.12674033641815186,
     0.02472634986042976,   -0.0131102055311203 ,    2.2858214378356934 ,
    -0.06010785698890686,    0.17167788743972778,    0.18221929669380188,
    -0.00370979309082031,   -2.957932949066162  ,   -0.08394689857959747,
    -0.0788525715470314 ,   -0.16496986150741577,    0.7445077896118164 ,
     1.828925371170044  ,   -0.36430343985557556,    0.14526164531707764,
    -0.03690484166145325,   -0.06873971223831177,    1.4632949829101562 ,
    -0.442336767911911  ,   -4.1743597984313965 ,    0.30148425698280334,
    -2.2164034843444824 ,   -5.590530872344971  ,   -1.4204556941986084 ,
    -0.1268521100282669 ,   -1.0485285520553589 ,    0.8773926496505737 ,
     0.9879087805747986 ,    0.13040325045585632,   -0.7524778246879578 ,
    -0.12820056080818176,   -0.08254548907279968,   -1.189617395401001  ,
     0.13878409564495087,    0.08432991057634354,   -6.393768310546875  ,
     0.2218303680419922 ,    0.02593335881829262,   -0.31090185046195984,
    -1.0224385261535645 ,    0.09316650032997131,    0.20207683742046356,
     0.5847983360290527 ,    0.8377769589424133 ,   -0.10550669580698013,
     0.21536105871200562,   -0.04965049028396606,   -0.09703505039215088,
     0.22964805364608765,   -0.4581391215324402 ,    2.1762046813964844 ,
    -1.072770357131958  ,    0.09187587350606918,    0.0873221755027771 ,
    -1.1720702648162842 ,    0.13165268301963806,    0.7816266417503357 ,
     0.48644256591796875,   -0.00913570821285248,    0.1814914047718048 ,
    -0.7992640733718872 ,   -2.195129871368408  ,    2.234663248062134  ,
     0.10115846991539001,   -0.4974241256713867 ,    1.4631593227386475 ,
    -0.4831829369068146 ,    1.0714809894561768 ,   -0.3639521896839142 ,
     0.6024593114852905 ,    1.2907638549804688 ,    0.4215947687625885 ,
     0.3403628170490265 ,   -0.01738378405570984,    0.19143541157245636,
     2.6640818119049072 ,    0.0371689461171627 ,   -0.0360240638256073 ,
    -0.12337999045848846,    1.6555521488189697 ,    0.5666109323501587 ,
     0.5635499954223633 ,   -0.27876144647598267,    0.7622988224029541 ,
     1.0265905857086182 ,    0.8433055281639099 ,   -1.224703311920166  ,
    -0.14043329656124115,    0.2700737714767456 ,   -0.02167989313602448,
    -1.5090365409851074 ,    2.068721055984497  ,    0.1741829812526703 ,
     0.5030797719955444 ,    0.22838573157787323,    7.496615886688232  ,
    -6.268997669219971  ,    0.02639730274677277,    2.6998305320739746 ,
     0.18654684722423553,    0.8029439449310303 ,    0.3598041534423828 ,
     2.4196388721466064 ,   -3.1774044036865234 ],
 [   0.42395874857902527,    1.6089673042297363 ,    1.7071782350540161 ,
     2.4831953048706055 ,   -0.24563530087471008,    0.14831706881523132,
    -1.070040225982666  ,   -0.6220162510871887 ,    0.6735644936561584 ,
    -0.6820261478424072 ,   -1.7449229955673218 ,    0.7291725277900696 ,
     1.3283870220184326 ,   -0.09205424785614014,    0.6867890954017639 ,
    -0.8161550164222717 ,   -2.5054478645324707 ,   -3.9263439178466797 ,
     0.885004997253418  ,    0.1601763367652893 ,   -0.15620064735412598,
     0.07778677344322205,    0.3630130887031555 ,    0.8154280185699463 ,
     1.1371562480926514 ,   -0.44246286153793335,   -0.9296029210090637 ,
     0.14223311841487885,   -0.45718657970428467,   -0.01721149682998657,
    -0.16525812447071075,    0.06918233633041382,    0.09465330839157104,
     0.15232400596141815,    0.9674553275108337 ,   -1.0049198865890503 ,
    -1.043323278427124  ,   -0.5608755350112915 ,   -0.05205166339874268,
     0.2909308075904846 ,    0.12590286135673523,    1.040832281112671  ,
     5.1071929931640625 ,    6.558914661407471  ,   -1.1282116174697876 ,
     1.9744164943695068 ,    7.2650885581970215 ,    0.3398905396461487 ,
     0.22542311251163483,   -0.08253967761993408,    0.3780221939086914 ,
    -1.115070104598999  ,   -0.287962406873703  ,    4.43782901763916   ,
     0.6543586254119873 ,    0.20446395874023438,   -1.0511624813079834 ,
    -0.18727853894233704,    0.05363589525222778,   -0.43033355474472046,
    -0.3974597752094269 ,    0.9633393287658691 ,   -0.9889333248138428 ,
    -1.3825385570526123 ,    0.4905461072921753 ,    0.609906792640686  ,
    -1.5138920545578003 ,    1.7195104360580444 ,   -0.00648360699415207,
    -0.4577297866344452 ,    0.01584481075406075,    0.912080705165863  ,
     2.7390294075012207 ,    5.001191139221191  ,   -3.158698081970215  ,
     0.6525367498397827 ,    1.6887223720550537 ,   -0.11842229962348938,
    -0.278668075799942  ,    0.4741555154323578 ,   -0.38195881247520447,
    -0.08293524384498596,   -0.04292598366737366,   -0.5769320130348206 ,
     2.2634177207946777 ,    1.3243591785430908 ,   -0.15538270771503448,
     0.06642024219036102,   -2.1744306087493896 ,   -0.9514248371124268 ,
     0.11763796210289001,    0.4747708737850189 ,    0.20905107259750366,
    -0.9444682002067566 ,   -1.0554769039154053 ,   -0.01419216394424438,
    -1.6261311769485474 ,   -0.13152547180652618,   -0.08043241500854492,
    -3.8381540775299072 ,    1.9045817852020264 ,    1.4087378978729248 ,
     0.5557319521903992 ,    0.5700299143791199 ,   -1.0691304206848145 ,
    -1.1574385166168213 ,    0.7512133121490479 ,   -0.6384609937667847 ,
    -1.6110117435455322 ,   -6.102899074554443  ,    3.04606556892395   ,
     0.2900981903076172 ,   -1.4120163917541504 ,   -1.4974851608276367 ,
    -3.06478214263916   ,   -1.761113166809082  ,   -0.6683914661407471 ,
    -0.6932569742202759 ,    0.13359040021896362,   -5.452574729919434  ,
     3.6160361766815186 ,    0.7604013681411743 ,   -1.048736333847046  ,
     1.459652066230774  ,   -1.9479625225067139 ,   -0.6114994883537292 ,
    -3.8984508514404297 ,   -4.462495803833008  ],
 [  -0.8976972103118896 ,   -3.8828821182250977 ,   -0.5617057085037231 ,
    -1.978355884552002  ,    1.0812541246414185 ,   -0.6406317353248596 ,
     9.258519172668457  ,   -0.467376172542572  ,   -0.954342246055603  ,
     1.7318603992462158 ,    6.803610801696777  ,   -2.692744255065918  ,
    -3.564633846282959  ,    1.7363923788070679 ,   -0.3692997694015503 ,
     2.656562328338623  ,    4.4235405921936035 ,    4.876733779907227  ,
    -2.075746536254883  ,   -0.7941617369651794 ,    1.0740675926208496 ,
    -0.897022008895874  ,   -1.3239933252334595 ,   -6.745207786560059  ,
    -3.4879302978515625 ,    0.7560297250747681 ,    9.633009910583496  ,
    -2.707080364227295  ,    2.269911766052246  ,   -0.2994522452354431 ,
    -0.01137131452560425,    4.159092426300049  ,   -0.646672248840332  ,
     0.9162195920944214 ,   -2.8884012699127197 ,    1.1221376657485962 ,
    11.162957191467285  ,    2.0164849758148193 ,   -0.8049917221069336 ,
    -0.32050883769989014,    0.8734855055809021 ,   -3.0793521404266357 ,
   -25.51923370361328   ,  -16.842483520507812  ,    1.5686125755310059 ,
    -4.289252758026123  ,  -16.641145706176758  ,   -4.563662528991699  ,
    -1.5148557424545288 ,    2.074638843536377  ,   -1.5848016738891602 ,
     2.4889047145843506 ,    1.0682268142700195 ,   -7.179279804229736  ,
    -3.231586456298828  ,   -0.23328077793121338,   -6.380327224731445  ,
     0.38300377130508423,   -0.5760365724563599 ,    6.251152038574219  ,
     1.478346347808838  ,   -1.0951273441314697 ,   -7.055246829986572  ,
    -3.404444456100464  ,   -0.12642168998718262,    0.05812717229127884,
     4.922403812408447  ,   -3.90903377532959   ,   -0.3655422031879425 ,
     1.134068250656128  ,    0.07734248787164688,   -2.6279921531677246 ,
    -2.5739221572875977 ,   -7.93509578704834   ,   -0.833352267742157  ,
     0.864890456199646  ,   -3.4841129779815674 ,   -0.5868914127349854 ,
     6.096321105957031  ,   -3.551703453063965  ,    4.15030574798584   ,
    -1.3770549297332764 ,   -0.05991625785827637,    7.137805938720703  ,
    -5.632571697235107  ,   -0.804082989692688  ,  -12.347792625427246  ,
     1.1636388301849365 ,    5.881124973297119  ,  -13.799886703491211  ,
    -0.337234228849411  ,   -0.48846736550331116,   -0.37330687046051025,
     4.4264421463012695 ,    1.4579225778579712 ,   -0.13817507028579712,
     3.7481188774108887 ,    1.0029007196426392 ,   -3.0810089111328125 ,
     2.5297956466674805 ,  -10.686920166015625  ,   -1.3067470788955688 ,
    -0.740493893623352  ,   -3.0147502422332764 ,    3.5593719482421875 ,
     3.3685503005981445 ,   -1.3953200578689575 ,   -1.4480133056640625 ,
     4.330580711364746  ,    7.281882286071777  ,   -2.6327691078186035 ,
    -1.4308068752288818 ,    5.383328437805176  ,    0.5250736474990845 ,
     3.067572593688965  ,    3.735837936401367  ,    1.2384041547775269 ,
     1.269491195678711  ,   -0.8319560885429382 ,    6.110828399658203  ,
   -11.213041305541992  ,    1.5447287559509277 ,    1.7506612539291382 ,
    -0.9136919975280762 ,    1.2045339345932007 ,    2.7504138946533203 ,
    10.893366813659668  ,   16.49112319946289   ],
 [  -0.03347145766019821,    0.39481261372566223,   -1.2274600267410278 ,
     1.946260690689087  ,   -0.5010912418365479 ,    0.21292060613632202,
    -2.0532689094543457 ,    0.07825610041618347,    0.08082261681556702,
    -0.56122225522995   ,   -1.6944512128829956 ,    1.1134939193725586 ,
     1.2646543979644775 ,   -1.1493080854415894 ,   -0.5601609945297241 ,
    -0.12089093029499054,   -0.08473048359155655,   -0.17457571625709534,
     0.5456883907318115 ,    0.26813963055610657,   -0.25974273681640625,
     0.36123812198638916,    0.31400349736213684,    2.380852699279785  ,
    -1.891372561454773  ,    0.08366011083126068,   -7.627353191375732  ,
     1.080033540725708  ,   -0.2358337938785553 ,   -0.3859149217605591 ,
    -0.06622583419084549,   -0.09159666299819946,    0.3503536283969879 ,
    -0.6482428908348083 ,    0.6146378517150879 ,    0.5496656894683838 ,
    -3.762150287628174  ,   -0.01738501340150833,    0.2231200635433197 ,
    -0.11786088347434998,   -0.4933706521987915 ,   -0.3188509941101074 ,
     6.587680339813232  ,    2.750000238418579  ,   -0.8284464478492737 ,
     0.5447240471839905 ,    6.9033613204956055 ,    1.0151466131210327 ,
     0.2433406412601471 ,   -1.143019437789917  ,   -1.2370344400405884 ,
    -0.6606395244598389 ,   -0.5912477970123291 ,   -0.2983550429344177 ,
     3.709559440612793  ,    0.5984439849853516 ,    1.0157723426818848 ,
     0.01584404334425926,    0.32200145721435547,   -1.2456412315368652 ,
    -0.8561777472496033 ,   -0.0799308642745018 ,    3.50993275642395   ,
     2.528315305709839  ,   -0.9571285843849182 ,   -1.2305715084075928 ,
     3.0008652210235596 ,   -0.37608516216278076,    0.4343375563621521 ,
    -0.11735890805721283,    0.0437600389122963 ,    0.6874784231185913 ,
    -0.13499122858047485,   -1.6641485691070557 ,   -2.3553550243377686 ,
     0.3682876229286194 ,   -0.28159692883491516,    0.3617839515209198 ,
    -3.7571425437927246 ,   -1.5955262184143066 ,   -1.8159745931625366 ,
     0.8407436013221741 ,    0.3428932726383209 ,   -1.888297200202942  ,
     1.0902255773544312 ,    0.15385697782039642,    1.5367777347564697 ,
    -0.5477384328842163 ,   -0.32230710983276367,    4.281829357147217  ,
     0.9772104024887085 ,   -1.7674789428710938 ,    0.5766496062278748 ,
    -1.050500750541687  ,   -0.6281573176383972 ,   -0.1174127608537674 ,
    -1.0723949670791626 ,   -0.06678207218647003,    1.2369816303253174 ,
     2.892186403274536  ,    0.18649333715438843,   -1.3332030773162842 ,
     0.00834240764379501,   -1.834358811378479  ,   -0.38282570242881775,
     0.42280399799346924,    0.227630615234375  ,   -0.21644327044487   ,
    -0.8042402267456055 ,    0.9032220840454102 ,   -2.739574909210205  ,
     0.5873903632164001 ,   -0.04803364723920822,    1.155501365661621  ,
    -0.5713555812835693 ,   -0.9868568181991577 ,   -0.03853748366236687,
    -0.1586175560951233 ,   -0.19252415001392365,   -9.244372367858887  ,
     2.7631726264953613 ,   -3.3713619709014893 ,   -3.6858432292938232 ,
    -2.923393964767456  ,    1.262028455734253  ,   -1.3547111749649048 ,
    -1.845154881477356  ,   -6.5367865562438965 ],
 [   0.47970789670944214,    0.45887401700019836,    0.2986290752887726 ,
    -3.8507983684539795 ,   -0.31864988803863525,   -0.10350126028060913,
     0.8410935401916504 ,    0.6852285861968994 ,   -0.30171382427215576,
    -0.7492612600326538 ,   -2.32593035697937   ,   -1.8262441158294678 ,
    -0.6598262786865234 ,   -0.27618324756622314,    6.086432456970215  ,
     0.6374988555908203 ,   -0.9649244546890259 ,   -0.9332505464553833 ,
    -0.25922128558158875,   -0.1664329618215561 ,    0.7639838457107544 ,
     0.0277675986289978 ,    0.2508309483528137 ,    3.2275047302246094 ,
     4.6016035079956055 ,   -1.1608054637908936 ,    2.1450376510620117 ,
    -0.27747681736946106,    0.0366094708442688 ,    0.5048626661300659 ,
     0.01917289197444916,    0.0761924758553505 ,   -0.19383463263511658,
     0.7023777961730957 ,    0.31683817505836487,   -0.3585551381111145 ,
    -0.32225772738456726,   -0.6887220740318298 ,   -0.11648266017436981,
     0.3274018466472626 ,   -0.01739484071731567,    2.6151504516601562 ,
    -5.689798831939697  ,   -0.6394864916801453 ,    1.6990872621536255 ,
     1.166212797164917  ,   -9.139459609985352  ,   -0.1380903124809265 ,
     0.398912250995636  ,    0.4009307026863098 ,   -0.3732268214225769 ,
     1.061120629310608  ,    0.00031185150146484,   -2.776764154434204  ,
    -5.442553997039795  ,   -2.1075377464294434 ,   -4.048598766326904  ,
     0.12598475813865662,   -0.28223204612731934,    3.160656213760376  ,
     0.7875717878341675 ,    1.1169867515563965 ,   -0.6402589082717896 ,
    -0.7625478506088257 ,    0.6333804130554199 ,    1.1633390188217163 ,
    -3.128376007080078  ,    1.5180699825286865 ,   -0.34903469681739807,
     0.3750699460506439 ,    0.06204405799508095,   -0.5748740434646606 ,
     2.683424711227417  ,   -1.335121512413025  ,   -0.32897427678108215,
    -1.3250012397766113 ,   -0.07521200180053711,   -0.2798302173614502 ,
     3.6525230407714844 ,    1.027528166770935  ,    1.5690045356750488 ,
    -0.23321688175201416,   -0.092354416847229  ,    3.235903263092041  ,
    -0.07666081190109253,   -0.5857954621315002 ,   -0.2970115840435028 ,
     0.5477067232131958 ,    3.7215616703033447 ,   -0.10351498425006866,
    -1.208775281906128  ,    2.200427770614624  ,   -0.5555639266967773 ,
    -1.385541558265686  ,    0.21734490990638733,    1.1442533731460571 ,
     0.3598366379737854 ,    0.13228066265583038,   -1.2735917568206787 ,
    -2.614379405975342  ,   -1.2196108102798462 ,   -0.07748937606811523,
    -0.3141377568244934 ,    0.1481296718120575 ,    0.27377215027809143,
    -0.0299755334854126 ,    0.15136629343032837,   -0.5587660074234009 ,
    -0.6101148128509521 ,    3.7738139629364014 ,    4.112598419189453  ,
     0.08826252818107605,   -1.7530193328857422 ,   -0.6785570383071899 ,
     1.7592215538024902 ,   -0.4305209219455719 ,    0.41653677821159363,
    -0.2034716159105301 ,   -0.05191577970981598,    4.5314226150512695 ,
     0.3949975371360779 ,    2.355278253555298  ,    3.6705400943756104 ,
     3.577195405960083  ,   -0.9570576548576355 ,    2.0802290439605713 ,
    -3.0788750648498535 ,    7.860677719116211  ],
 [  -1.0899198055267334 ,   -1.5015884637832642 ,   -0.8340943455696106 ,
     5.178399085998535  ,    1.73180091381073   ,   -0.20306169986724854,
    -7.029736042022705  ,   -0.824680507183075  ,    0.5728323459625244 ,
     2.381829261779785  ,    2.103830099105835  ,    2.771527051925659  ,
     0.4922093152999878 ,    1.8271182775497437 ,  -11.987536430358887  ,
    -2.645822525024414  ,    0.30456867814064026,    0.7244429588317871 ,
     0.5633387565612793 ,   -0.01436734199523926,   -1.1229937076568604 ,
     0.14385253190994263,    0.10865841805934906,   -6.476247310638428  ,
    -4.476109981536865  ,    1.458850622177124  ,    3.789522409439087  ,
     0.1374596357345581 ,   -1.1857929229736328 ,    0.2965583801269531 ,
     0.40000009536743164,    0.7306798696517944 ,    0.00364071130752563,
    -1.1726642847061157 ,   -1.1357089281082153 ,    0.16153305768966675,
    -5.638279914855957  ,    0.4937790632247925 ,    0.3900771737098694 ,
    -0.21078872680664062,    0.3646388351917267 ,   -2.105299234390259  ,
    14.57878589630127   ,    3.738517999649048  ,   -1.8417083024978638 ,
    -3.1289291381835938 ,   17.248640060424805  ,    3.9655139446258545 ,
    -0.3074598014354706 ,   -0.24695512652397156,    3.2203614711761475 ,
    -2.0677874088287354 ,    1.3076159954071045 ,    5.416195392608643  ,
     9.609910011291504  ,    4.805259704589844  ,    5.32611608505249   ,
    -0.2381211221218109 ,   -0.11931878328323364,  -12.144466400146484  ,
    -0.8832102417945862 ,   -1.7273268699645996 ,    1.0750900506973267 ,
     2.9855499267578125 ,    0.4863028824329376 ,   -0.3817569613456726 ,
    -4.703862190246582  ,   -0.5079296827316284 ,    0.3137638568878174 ,
    -0.16064122319221497,   -0.13855722546577454,    1.6704933643341064 ,
    -2.2648816108703613 ,    6.58427619934082   ,    8.482665061950684  ,
     0.9463831186294556 ,    1.892615795135498  ,    0.42852121591567993,
    -7.363584518432617  ,    4.318756103515625  ,   -2.294466972351074  ,
    -0.7644456624984741 ,   -0.2956053614616394 ,   -6.289297580718994  ,
    -1.4355883598327637 ,    0.6946237087249756 ,    8.184985160827637  ,
    -0.8588367700576782 ,   -9.500041007995605  ,    2.514591693878174  ,
     0.6127630472183228 ,   -2.758511781692505  ,    0.5997982025146484 ,
     0.7523391246795654 ,    0.8280443549156189 ,   -2.141695976257324  ,
     0.4138614535331726 ,   -1.070975661277771  ,    0.6787092685699463 ,
     1.1239287853240967 ,    7.476535797119141  ,    3.7488276958465576 ,
     0.7801759839057922 ,    3.1833345890045166 ,   -2.5736660957336426 ,
    -1.8453423976898193 ,   -0.6400575637817383 ,    2.8665366172790527 ,
     0.6322875022888184 ,   -9.814506530761719  ,   -2.2863948345184326 ,
    -0.2850649356842041 ,   -2.0592615604400635 ,    0.17857620120048523,
    -2.833726406097412  ,   -1.1701173782348633 ,   -0.37495166063308716,
     0.5370883941650391 ,    1.1188874244689941 ,    2.0039286613464355 ,
     0.2342630922794342 ,    1.101975440979004  ,   -0.3334001302719116 ,
    -1.9036678075790405 ,    0.9144284725189209 ,   -0.8776493072509766 ,
     6.743927955627441  ,   -8.351829528808594  ],
 [  -0.08396752178668976,    0.19288751482963562,    0.1874220073223114 ,
    -0.6211490035057068 ,   -0.4452037215232849 ,    0.2193111926317215 ,
     2.31152081489563   ,   -0.3041772246360779 ,    0.21163322031497955,
    -0.39005616307258606,    1.3049790859222412 ,    0.23032964766025543,
     0.4723120927810669 ,   -0.4908321797847748 ,   -0.8046450614929199 ,
     0.26383134722709656,    0.6345317959785461 ,    0.7745265364646912 ,
    -0.03517762944102287,    0.14726604521274567,   -0.13608600199222565,
    -0.6986996531486511 ,   -0.08004029095172882,   -0.8919271230697632 ,
    -0.4152423143386841 ,    0.4966048002243042 ,   -2.357856273651123  ,
     0.29983463883399963,    0.23538672924041748,   -0.2797923684120178 ,
    -0.24266019463539124,   -1.335088849067688  ,    0.06603510677814484,
     0.18773289024829865,    0.4737235903739929 ,    0.4506838321685791 ,
     2.1661384105682373 ,    0.19265736639499664,    0.30783340334892273,
    -0.43142905831336975,   -0.10201390087604523,   -1.3946185111999512 ,
    -2.4179606437683105 ,    0.4069463014602661 ,   -1.0752276182174683 ,
     0.18787111341953278,   -2.255321741104126  ,   -0.2564062476158142 ,
    -0.38774171471595764,   -0.00590807944536209,   -0.27897292375564575,
    -0.00004909001290798,   -0.17584092915058136,   -0.19852006435394287,
    -1.6980395317077637 ,   -1.3763173818588257 ,    0.16883161664009094,
    -0.2166995406150818 ,    0.1602460741996765 ,    1.619201898574829  ,
    -0.09276510030031204,    0.28744208812713623,   -1.2047306299209595 ,
     0.02275441773235798,   -0.6857008934020996 ,   -0.1942926049232483 ,
     2.65045428276062   ,   -0.7340714335441589 ,   -0.1900937557220459 ,
    -0.4219357967376709 ,   -0.10718328505754471,   -0.513964056968689  ,
    -1.7976981401443481 ,    0.05940915644168854,   -1.2665972709655762 ,
     0.4900904893875122 ,    0.2782381474971771 ,   -0.00466367555782199,
     1.4775075912475586 ,   -1.7905620336532593 ,    0.06871700286865234,
     0.5870100259780884 ,    0.20670706033706665,   -1.3418158292770386 ,
     1.4811722040176392 ,    0.22380417585372925,   -1.0584195852279663 ,
     0.03012131527066231,   -0.44676974415779114,   -0.6329673528671265 ,
     0.19998908042907715,    2.645787000656128  ,   -0.07969430088996887,
     1.0513536930084229 ,   -0.16434359550476074,    0.10473625361919403,
    -0.6300894021987915 ,    0.412134051322937  ,    0.8925917148590088 ,
    -0.6089721322059631 ,    0.3731852173805237 ,   -1.0894036293029785 ,
     0.1691940575838089 ,    0.6647002696990967 ,   -0.29380589723587036,
    -1.2702800035476685 ,    0.5488677024841309 ,   -0.41215822100639343,
     0.37830713391304016,   -0.6260316371917725 ,   -1.3328728675842285 ,
    -0.20023725926876068,    1.1908785104751587 ,   -0.5447713136672974 ,
    -0.187419593334198  ,    0.7929299473762512 ,   -0.48551711440086365,
    -0.12520217895507812,   -0.41938188672065735,    1.8932338953018188 ,
    -0.6348758339881897 ,   -1.7157577276229858 ,   -1.1482352018356323 ,
    -0.3434118330478668 ,   -0.4566943049430847 ,   -1.255569338798523  ,
     0.8569751381874084 ,   -0.09016278386116028],
 [   0.36318516731262207,    0.550136148929596  ,    0.19164890050888062,
     1.1179708242416382 ,   -0.02565495669841766,   -0.12186628580093384,
    -0.5564429759979248 ,    0.08877335488796234,   -0.15124820172786713,
     0.0183069258928299 ,   -0.9941940903663635 ,   -0.2588575780391693 ,
    -0.36836695671081543,    0.26720669865608215,    2.2350430488586426 ,
     0.26915061473846436,   -0.8775067925453186 ,   -0.9708864092826843 ,
    -0.08142846822738647,   -0.09038969874382019,    0.09010007977485657,
     0.8244034647941589 ,   -0.16687528789043427,    1.6870697736740112 ,
     0.5993766784667969 ,   -0.5444991588592529 ,    2.3474791049957275 ,
    -0.7331647276878357 ,    0.03426024317741394,    0.08803421258926392,
     0.10526005923748016,    2.5151021480560303 ,    0.00261811912059784,
     0.01832513138651848,   -0.31400537490844727,   -0.7184846997261047 ,
    -0.38596296310424805,   -0.1634654551744461 ,   -0.5200101733207703 ,
     0.5895557403564453 ,    0.15832623839378357,    1.183947205543518  ,
    -0.691798985004425  ,   -1.7916569709777832 ,    1.2199747562408447 ,
     0.2147853523492813 ,    0.5836466550827026 ,   -1.6812913417816162 ,
     0.556429922580719  ,    0.5562282800674438 ,   -0.13636785745620728,
     0.2746906876564026 ,   -0.22930322587490082,    0.6433669328689575 ,
    -0.24139444530010223,    0.7028515338897705 ,    0.8470017910003662 ,
     0.16946570575237274,   -0.1849673092365265 ,    3.0098588466644287 ,
     0.22060281038284302,   -0.03309614211320877,    0.23996716737747192,
    -0.6815212368965149 ,    0.3335557281970978 ,    0.11277927458286285,
    -2.7948389053344727 ,    0.6416320204734802 ,    0.11829942464828491,
     0.21754443645477295,    0.09421591460704803,    0.10401006788015366,
     0.6075795888900757 ,   -1.238641381263733  ,    0.07256390154361725,
    -0.45494937896728516,   -0.09801200032234192,   -0.1758415251970291 ,
    -0.10141351073980331,    0.7974498867988586 ,    0.24074873328208923,
    -0.5410212874412537 ,   -0.14931181073188782,    2.1523900032043457 ,
    -0.9822160601615906 ,   -0.08431357145309448,   -0.9414371848106384 ,
     0.11147044599056244,    1.8024238348007202 ,   -1.7429012060165405 ,
    -0.02629629522562027,   -3.323841094970703  ,    0.00556305050849915,
    -0.3444281220436096 ,   -0.3584561347961426 ,    0.08861197531223297,
     0.37077444791793823,   -0.28112560510635376,   -0.728366494178772  ,
     0.18526886403560638,   -1.5184822082519531 ,    0.7568638920783997 ,
    -0.3282461166381836 ,   -1.6383064985275269 ,    1.011971354484558  ,
     1.822860598564148  ,   -0.29735103249549866,   -0.09426838904619217,
    -0.5529891848564148 ,    1.420125126838684  ,    0.8293673992156982 ,
     0.29123610258102417,    0.01944203674793243,    0.5285407304763794 ,
     1.5497468709945679 ,   -0.40993404388427734,    0.30616188049316406,
    -0.02665559947490692,    0.2931717336177826 ,   -3.2129173278808594 ,
     1.4402461051940918 ,    1.0725911855697632 ,   -0.42683959007263184,
     0.29879051446914673,    0.2548113465309143 ,    0.506536602973938  ,
    -1.1860371828079224 ,    1.646393060684204  ],
 [  -0.01528321951627731,    0.14639407396316528,   -0.5534680485725403 ,
    -0.6264464259147644 ,   -0.3561345040798187 ,    0.04585195332765579,
    -0.42026486992836   ,   -0.07850920408964157,   -0.00571259111166   ,
     0.09422245621681213,    0.47006872296333313,    0.3959500193595886 ,
     0.01717604696750641,    0.01465141773223877,   -1.4317636489868164 ,
    -0.19823579490184784,    0.14345897734165192,    0.9168333411216736 ,
    -0.01747038960456848,    0.07873878628015518,   -0.2917892038822174 ,
    -0.01012467592954636,   -0.20022758841514587,   -0.5156986117362976 ,
    -0.758526086807251  ,    0.1871841549873352 ,    0.615420937538147  ,
     0.2761959135532379 ,   -0.11550472676753998,    0.01717835664749146,
    -0.01917778700590134,   -0.26811864972114563,    0.0323975682258606 ,
    -0.27306443452835083,    0.08590943366289139,    0.5426793694496155 ,
     0.8831222057342529 ,    0.05417782813310623,   -0.10305160284042358,
     0.06294393539428711,   -0.04638699442148209,   -0.9988120198249817 ,
     0.6332533359527588 ,   -2.241391658782959  ,    0.1337362676858902 ,
     0.5057920217514038 ,   -0.20578515529632568,   -0.6016774773597717 ,
     0.12223517149686813,    0.35421082377433777,   -0.1494104564189911 ,
    -0.10155988484621048,   -0.17520563304424286,    0.11022113263607025,
    -0.5064370036125183 ,    0.00749759376049042,   -0.09625668823719025,
     0.01629343256354332,    0.07746250182390213,   -0.13031741976737976,
     0.00817586481571198,    0.01374807953834534,    0.08186918497085571,
    -0.18379905819892883,   -0.11021165549755096,   -0.07068011909723282,
     0.09540742635726929,   -0.625637948513031  ,    0.19731955230236053,
    -0.1474178582429886 ,   -0.015700314193964  ,   -0.16928011178970337,
    -2.421721935272217  ,   -1.1977812051773071 ,    0.395043283700943  ,
    -0.12496671825647354,   -0.1270417720079422 ,   -0.01368825882673264,
    -0.9644255042076111 ,   -0.4524829387664795 ,   -0.14383406937122345,
    -0.06690672785043716,    0.06064211577177048,   -0.7596452832221985 ,
    -0.01253627985715866,   -0.2551679015159607 ,   -1.0519353151321411 ,
    -0.10875110328197479,   -0.6012365818023682 ,   -0.39510852098464966,
     0.1800011843442917 ,   -0.4292212426662445 ,   -0.02075804769992828,
     0.5609755516052246 ,   -0.08894219994544983,   -0.08571308851242065,
    -0.0122455358505249 ,    0.03851103037595749,   -0.1298011839389801 ,
     0.05965717509388924,   -0.20556966960430145,   -0.7369186878204346 ,
    -0.04014419764280319,   -0.5517755746841431 ,    0.20185384154319763,
    -0.13946986198425293,   -0.00634834170341492,    0.26129814982414246,
     0.09824012964963913,    0.2602686285972595 ,   -0.4796296954154968 ,
    -0.04397881776094437,    0.08767114579677582,   -0.08025069534778595,
    -0.27544620633125305,    0.03538445383310318,   -0.2676326036453247 ,
    -0.05261106789112091,   -0.04560799151659012,   -0.23540252447128296,
    -0.5852747559547424 ,   -0.7723773717880249 ,   -0.2551229000091553 ,
    -0.12045768648386002,    0.9244406223297119 ,   -0.4766629934310913 ,
    -0.6095235347747803 ,    0.02447481453418732],
 [   0.6647496223449707 ,    1.6155098676681519 ,   -0.05415183305740356,
     0.6098604202270508 ,   -1.7436712980270386 ,   -0.19609813392162323,
     1.4779072999954224 ,    0.8446525931358337 ,    0.09743829071521759,
    -0.11408907175064087,   -2.630404472351074  ,   -1.2404730319976807 ,
     0.19093941152095795,    0.6927627325057983 ,    5.37514591217041   ,
     0.31624943017959595,    1.471552848815918  ,   -3.5663886070251465 ,
    -1.4356894493103027 ,   -0.07839419692754745,    0.4036751389503479 ,
     0.6106295585632324 ,   -0.1032647043466568 ,    5.133640766143799  ,
     0.6060779094696045 ,   -1.5806763172149658 ,   -0.2519727647304535 ,
     0.9505418539047241 ,    0.4628896713256836 ,    0.00497591495513916,
     0.1943701058626175 ,   -2.410926342010498  ,   -0.2546710968017578 ,
     0.5775545835494995 ,    0.7490752935409546 ,   -1.9908027648925781 ,
    -1.6927447319030762 ,   -0.4742511510848999 ,   -0.03025578707456589,
     2.0103392601013184 ,   -0.04269468784332275,    3.172267436981201  ,
     0.7561013698577881 ,   -3.7299509048461914 ,    0.5695863366127014 ,
    -3.2776575088500977 ,   -4.842337131500244  ,   -0.01846849918365479,
     0.2188345193862915 ,    4.256022930145264  ,    1.2375327348709106 ,
    -0.494709849357605  ,   -0.3128519654273987 ,    0.03119447827339172,
     0.14820924401283264,    0.4923103451728821 ,    0.28277331590652466,
     0.22395174205303192,   -0.8461072444915771 ,   -0.4350396394729614 ,
     0.29679107666015625,    1.4051003456115723 ,    1.1542772054672241 ,
    -0.7384998798370361 ,    0.7353867888450623 ,    1.3124821186065674 ,
    -0.08886135369539261,    2.33241868019104   ,    0.32125210762023926,
    -0.2615329325199127 ,    0.141115203499794  ,    0.8853040337562561 ,
    -0.17642530798912048,   -2.4722094535827637 ,   -4.453181266784668  ,
     0.8741805553436279 ,    0.5999792814254761 ,   -0.13103975355625153,
    -0.3332974314689636 ,   -3.3526864051818848 ,    0.5429116487503052 ,
    -0.19472052156925201,   -0.26979008316993713,    0.817668080329895  ,
     1.4445557594299316 ,   -1.3619438409805298 ,    1.66538405418396   ,
     0.5263237953186035 ,    0.9704486727714539 ,   -2.124756097793579  ,
    -0.707608699798584  ,   -0.583978533744812  ,   -0.2104763686656952 ,
    -4.1909499168396    ,    0.6243227124214172 ,   -0.21678215265274048,
     0.422671914100647  ,    0.50192791223526   ,    1.0141359567642212 ,
    -0.36525896191596985,   -5.641028881072998  ,   -0.11892712116241455,
     0.41574931144714355,   -0.03871399164199829,    0.02293819189071655,
     0.46167218685150146,   -0.00661873817443848,   -2.036083698272705  ,
    -0.29957178235054016,    1.1276745796203613 ,    6.493596076965332  ,
    -0.17150381207466125,   -1.231815218925476  ,   -0.4212331771850586 ,
     0.5856832265853882 ,   -3.273616075515747  ,    1.6464372873306274 ,
    -0.687356173992157  ,   -0.6277501583099365 ,    1.167054295539856  ,
     0.4042758643627167 ,   -0.19498932361602783,    0.00272959470748901,
     3.711071252822876  ,   -5.176805019378662  ,    3.096795082092285  ,
     0.12865294516086578,    1.7542781829833984 ],
 [  -1.1943806409835815 ,   -3.076231002807617  ,    0.855756402015686  ,
    -0.9887986183166504 ,    3.965031147003174  ,    0.13193607330322266,
    -1.582885503768921  ,   -1.131364107131958  ,   -0.5273181200027466 ,
     0.2320462465286255 ,    3.718001365661621  ,    0.5029299259185791 ,
    -0.9653196334838867 ,   -0.951790452003479  ,   -5.117095947265625  ,
    -0.1331397294998169 ,   -2.9132204055786133 ,    4.148289203643799  ,
     3.1954150199890137 ,   -0.10282416641712189,   -0.14482206106185913,
    -1.1027655601501465 ,    0.7619784474372864 ,   -7.846709728240967  ,
    -0.34515368938446045,    3.467027187347412  ,    1.7608777284622192 ,
    -1.6579885482788086 ,   -0.7831846475601196 ,   -0.06040143966674805,
    -0.2672812342643738 ,    5.781099796295166  ,    0.24796640872955322,
    -0.7724006175994873 ,   -1.653513789176941  ,    1.4370476007461548 ,
    -0.46012812852859497,    2.919689178466797  ,   -0.04503140226006508,
    -3.190325975418091  ,   -0.09719344973564148,   -3.771580457687378  ,
    -2.1938860416412354 ,   11.729155540466309  ,   -1.1273831129074097 ,
     3.4631550312042236 ,    9.325172424316406  ,    2.034170389175415  ,
    -0.2270602285861969 ,   -5.99577522277832   ,    0.1316627860069275 ,
     0.45558199286460876,    0.569632887840271  ,   -0.09071628749370575,
     1.0454175472259521 ,   -0.8562620282173157 ,   -0.8295181393623352 ,
    -0.23537757992744446,    1.3259540796279907 ,    1.5393062829971313 ,
    -0.12810589373111725,   -1.9191796779632568 ,   -3.3924720287323    ,
     1.5790355205535889 ,   -0.8222806453704834 ,   -2.617832660675049  ,
     0.211252361536026  ,   -2.527353286743164  ,   -0.9299725294113159 ,
     0.8053961396217346 ,   -0.2293805181980133 ,   -0.5495014190673828 ,
     6.2080769538879395 ,    6.487279891967773  ,    5.655138969421387  ,
    -0.42240309715270996,   -0.3186584711074829 ,   -0.2808964252471924 ,
     1.5945637226104736 ,    7.462237358093262  ,   -1.2831218242645264 ,
     1.104575514793396  ,    0.3348058760166168 ,    0.20395857095718384,
    -2.0161330699920654 ,    2.926208019256592  ,   -2.015244483947754  ,
    -0.7872260808944702 ,   -2.1261277198791504 ,    6.203639030456543  ,
     0.7186256647109985 ,    1.8784022331237793 ,   -0.05218887329101562,
     5.758935451507568  ,   -0.7318843603134155 ,    0.19511696696281433,
    -0.7483326196670532 ,   -0.8932542204856873 ,   -1.7907698154449463 ,
     0.2495240867137909 ,   11.297487258911133  ,    1.3582241535186768 ,
    -0.2798498570919037 ,    1.331872582435608  ,   -1.2985788583755493 ,
    -0.5368643403053284 ,   -0.1595228910446167 ,    2.4834628105163574 ,
    -0.00586998462677002,   -2.6407389640808105 ,   -8.298770904541016  ,
     0.5315163731575012 ,    2.0134894847869873 ,    0.37940654158592224,
    -0.05505681782960892,    6.279545783996582  ,   -2.0156197547912598 ,
     1.2796285152435303 ,    1.104001522064209  ,   -0.3698655068874359 ,
    -0.9119664430618286 ,    1.2308980226516724 ,    1.0584920644760132 ,
    -4.246017932891846  ,    6.659580230712891  ,   -5.042074680328369  ,
    -0.33123499155044556,   -3.427778482437134  ],
 [   0.33847811818122864,    1.0903773307800293 ,   -1.0888240337371826 ,
    -1.0665948390960693 ,   -1.3075591325759888 ,    0.05584651231765747,
    -0.01855792105197906,    0.7115103602409363 ,    0.26285913586616516,
    -0.7492589354515076 ,   -3.4058077335357666 ,    0.4357609450817108 ,
     2.5099024772644043 ,   -0.94040846824646   ,    2.588696002960205  ,
    -0.20565369725227356,    4.277703285217285  ,   -2.2337851524353027 ,
    -1.0856051445007324 ,    0.15919363498687744,    1.0881845951080322 ,
     1.7794442176818848 ,   -0.43596935272216797,    5.858611583709717  ,
     0.5619353652000427 ,   -0.8420908451080322 ,   -2.9621076583862305 ,
     2.1545073986053467 ,   -0.10501699149608612,   -0.29682838916778564,
     0.15100087225437164,   -4.269713878631592  ,    0.18536579608917236,
     1.0486650466918945 ,    0.7822737097740173 ,   -1.8761935234069824 ,
     3.354889154434204  ,   -1.7308428287506104 ,    0.3854270279407501 ,
     1.5384581089019775 ,   -0.09305194765329361,    3.414022207260132  ,
     1.0830243825912476 ,   -7.380395889282227  ,    0.49978911876678467,
     0.6846059560775757 ,   -6.661763668060303  ,   -2.1803486347198486 ,
    -0.5357905030250549 ,    1.3677610158920288 ,   -2.3551065921783447 ,
     0.9429978728294373 ,    0.5730035305023193 ,   -1.5803825855255127 ,
    -1.0219111442565918 ,    0.602562665939331  ,   -0.6711333990097046 ,
     0.21295800805091858,   -0.03856953978538513,   -2.4704647064208984 ,
    -0.5755000710487366 ,    1.8610413074493408 ,    6.747512340545654  ,
    -1.6242115497589111 ,    0.2271188497543335 ,    1.9819130897521973 ,
    -0.8778902888298035 ,    1.1291395425796509 ,    0.9471434354782104 ,
    -0.49758899211883545,    0.157578244805336  ,   -0.5145609378814697 ,
    -3.4898626804351807 ,   -2.2785301208496094 ,   -2.7120749950408936 ,
    -0.06435084342956543,    2.9520528316497803 ,   -0.21609097719192505,
    -3.608880043029785  ,   -4.546652793884277  ,    1.4126834869384766 ,
    -2.069040536880493  ,   -0.18959683179855347,   -0.5541895627975464 ,
     1.437196135520935  ,   -1.0788160562515259 ,    3.19704532623291   ,
     0.13167543709278107,    4.743734359741211  ,   -5.221441268920898  ,
    -0.39632266759872437,   -1.082101821899414  ,    0.3024013936519623 ,
    -2.9991352558135986 ,    0.26138854026794434,    0.6281231045722961 ,
     0.20231181383132935,    0.6313499808311462 ,    0.9622744917869568 ,
     2.3479104042053223 ,   -5.270387649536133  ,   -0.8268194794654846 ,
     0.11642550677061081,   -0.06095799803733826,    3.6140339374542236 ,
     1.0636663436889648 ,   -0.04076233506202698,   -0.7484192848205566 ,
     0.6707019805908203 ,    2.773332118988037  ,    4.409179210662842  ,
    -0.35508614778518677,   -1.4080978631973267 ,   -0.5381560921669006 ,
     0.5292221307754517 ,   -3.2843806743621826 ,    1.5379902124404907 ,
    -0.6765910983085632 ,   -0.5161669850349426 ,   -2.496609926223755  ,
     1.3266743421554565 ,   -0.7053700685501099 ,   -1.4306507110595703 ,
     1.0365904569625854 ,   -3.324643850326538  ,    4.4168596267700195 ,
     1.1822984218597412 ,    1.355240821838379  ],
 [  -0.6027992963790894 ,    0.614965558052063  ,    1.063166618347168  ,
     1.6191322803497314 ,    2.026919364929199  ,   -1.112725853919983  ,
     1.0239583253860474 ,    0.09106684476137161,   -0.5711606740951538 ,
    -0.01969450712203979,    4.685976505279541  ,   -0.9834225177764893 ,
    -6.878261089324951  ,    1.5512136220932007 ,   -1.6119084358215332 ,
     0.8620693683624268 ,   -5.380301475524902  ,    1.8729500770568848 ,
    -0.33222496509552   ,   -0.14454632997512817,   -1.194545030593872  ,
    -2.908036470413208  ,   -1.519541621208191  ,   -2.8291521072387695 ,
     3.7661702632904053 ,   -0.5381706953048706 ,    1.0327558517456055 ,
    -1.2625577449798584 ,    0.8547144532203674 ,    0.3090673089027405 ,
    -0.1394573599100113 ,    3.166501045227051  ,   -1.1681997776031494 ,
     0.356941282749176  ,    0.20122867822647095,    1.9680125713348389 ,
    -3.96862530708313   ,   -0.3749644458293915 ,   -0.4837849736213684 ,
    -1.1151145696640015 ,   -1.08220636844635   ,   -2.0538368225097656 ,
     3.427828311920166  ,   -0.21748414635658264,    0.5185155272483826 ,
    -1.2795841693878174 ,    4.463624000549316  ,    2.0878705978393555 ,
     0.01872282475233078,   -0.4759502708911896 ,    2.598449468612671  ,
    -1.1224933862686157 ,   -0.3852599561214447 ,    5.405606746673584  ,
    -0.9675291180610657 ,   -0.8508310914039612 ,    4.587936878204346  ,
    -0.281997948884964  ,   -2.2631521224975586 ,    1.0057382583618164 ,
     0.5307237505912781 ,   -2.399845600128174  ,   -7.371856689453125  ,
     3.475184202194214  ,   -0.7693246006965637 ,   -0.6586071252822876 ,
     1.4023107290267944 ,   -0.8840633034706116 ,   -1.9953393936157227 ,
     0.7492138147354126 ,   -0.2799849808216095 ,    0.4118163287639618 ,
     3.318727493286133  ,   -0.7027002573013306 ,    1.320672869682312  ,
    -0.7475444078445435 ,   -3.0323259830474854 ,    0.28408923745155334,
     5.3392653465271    ,    0.7886338233947754 ,   -1.0713448524475098 ,
     1.158648133277893  ,    0.3448634147644043 ,    0.8847085237503052 ,
    -2.8909614086151123 ,   -1.244318962097168  ,   -2.475386142730713  ,
     0.8627288937568665 ,   -2.266091823577881  ,    5.41682767868042   ,
     1.2403429746627808 ,   -0.4459131360054016 ,   -0.3856133818626404 ,
     0.6060316562652588 ,    0.582941472530365  ,   -0.8082111477851868 ,
    -0.12461599707603455,   -0.9671890139579773 ,   -0.38596951961517334,
    -6.304233074188232  ,    0.6141509413719177 ,    0.9340847134590149 ,
    -0.20578788220882416,   -2.8764171600341797 ,   -1.5329934358596802 ,
    -1.9193464517593384 ,    1.4679136276245117 ,   -1.4789247512817383 ,
    -0.6074175834655762 ,   -2.926063299179077  ,   -4.111827850341797  ,
     0.2664823532104492 ,    0.5559325218200684 ,    2.14644455909729   ,
    -3.5030548572540283 ,   -1.8790969848632812 ,   -0.0272662341594696 ,
     1.1472834348678589 ,    0.616363525390625  ,    4.506833553314209  ,
    -3.5056042671203613 ,   -1.1972965002059937 ,   -2.826093912124634  ,
     1.1721487045288086 ,    0.11157868802547455,   -0.2838858962059021 ,
    -4.0784454345703125 ,    0.359161376953125  ],
 [   0.24113500118255615,   -0.41638925671577454,    0.3166479170322418 ,
     0.4220280349254608 ,   -1.9144361019134521 ,    1.0683363676071167 ,
    -2.6036314964294434 ,    0.08185197412967682,    0.06541471183300018,
    -0.16205647587776184,   -3.004940986633301  ,    4.127397537231445  ,
    10.196610450744629  ,   -1.3922712802886963 ,    4.82520866394043   ,
     2.576011896133423  ,    2.4062228202819824 ,   -1.7243341207504272 ,
    -0.9457671642303467 ,    0.24234621226787567,    0.5417872667312622 ,
     9.695991516113281  ,    2.116257429122925  ,    1.7804356813430786 ,
    -0.5682341456413269 ,    0.02522021532058716,   -1.1587380170822144 ,
     0.10250867158174515,   -0.23886750638484955,   -0.5166988968849182 ,
    -0.0575183741748333 ,   -3.038804054260254  ,   -0.0428340882062912 ,
    -3.0025129318237305 ,   -3.6663460731506348 ,   -0.32005947828292847,
     2.428068161010742  ,   -0.23405346274375916,    0.8318076133728027 ,
    -0.4470495879650116 ,    0.5888670682907104 ,    3.0695905685424805 ,
    -4.070613861083984  ,    3.326474666595459  ,   -0.12109793722629547,
     0.885909378528595  ,   -2.9430932998657227 ,   -1.8821457624435425 ,
    -0.6800967454910278 ,    0.07257810235023499,   -1.6190259456634521 ,
     3.989067316055298  ,   -0.9829806089401245 ,   -0.09910235553979874,
    -0.05041392147541046,    0.5993903279304504 ,   -3.0966694355010986 ,
    -0.08184565603733063,    1.8358467817306519 ,    3.3523144721984863 ,
    -0.2407645285129547 ,    1.615002155303955  ,    6.310291290283203  ,
    -0.5114279389381409 ,    1.2294341325759888 ,   -0.8501707315444946 ,
    -0.9317920207977295 ,    1.940962553024292  ,    1.0334776639938354 ,
    -0.5030885934829712 ,    0.01898062601685524,    3.566800117492676  ,
     0.8513102531433105 ,    1.5005260705947876 ,    0.6822769045829773 ,
    -0.36737942695617676,    1.4348880052566528 ,   -0.04862651228904724,
    -1.8589587211608887 ,   -1.2222366333007812 ,    0.7003756165504456 ,
    -0.18736566603183746,   -0.5040388107299805 ,    1.506533145904541  ,
     5.421274662017822  ,    1.211844801902771  ,    5.676671504974365  ,
     0.08449395000934601,   -0.19986911118030548,   -1.1379772424697876 ,
    -0.40875008702278137,    5.100986480712891  ,   -2.2036876678466797 ,
    -0.7292762994766235 ,    0.4106948673725128 ,    0.25502288341522217,
    -0.6295986771583557 ,    0.7184681296348572 ,   -0.15542158484458923,
     5.916509628295898  ,    2.260382890701294  ,    1.2826623916625977 ,
    -0.03070699423551559,    3.3402392864227295 ,    0.9204635620117188 ,
    -0.8658682107925415 ,   -1.4864977598190308 ,    2.5334887504577637 ,
    -0.42941519618034363,    2.324774980545044  ,    2.607755661010742  ,
    -0.7694746255874634 ,   -0.877130925655365  ,   -1.2640485763549805 ,
     3.673903465270996  ,    0.07465316355228424,   -0.7137635350227356 ,
    -1.120774745941162  ,   -0.5972124338150024 ,   -4.213562488555908  ,
     2.2113029956817627 ,    0.45497509837150574,    1.1447466611862183 ,
    -0.37609848380088806,   -1.6324270963668823 ,   -0.30920737981796265,
     4.588871479034424  ,   -1.2200727462768555 ],
 [   0.9439290761947632 ,   -1.8480584621429443 ,   -0.9316684007644653 ,
    -1.3509968519210815 ,    0.5307757258415222 ,   -0.00320115685462952,
     4.0190110206604    ,   -1.133208990097046  ,    0.42653870582580566,
     1.7868759632110596 ,   -0.4278905391693115 ,    0.4437193274497986 ,
    -1.0868921279907227 ,   -0.18981671333312988,   -4.890749931335449  ,
    -0.6713176965713501 ,    0.9248709082603455 ,    1.029062271118164  ,
    -0.46498095989227295,    0.6737598776817322 ,   -0.19393879175186157,
    -4.502044677734375  ,   -0.9691625237464905 ,    1.7999968528747559 ,
     2.554873466491699  ,   -0.3955157995223999 ,   -0.5292476415634155 ,
    -0.00328540802001953,    1.5880215167999268 ,    2.2332491874694824 ,
    -0.11332577466964722,    3.062892198562622  ,    0.7716695070266724 ,
     1.2538036108016968 ,    2.0101029872894287 ,   -0.7337554693222046 ,
     0.10627143085002899,    0.24069595336914062,   -0.3060499429702759 ,
     5.056976795196533  ,   -0.18386268615722656,   -0.3378129005432129 ,
     3.709017038345337  ,   -2.871476650238037  ,   -1.3524065017700195 ,
     0.40629714727401733,    0.4474812150001526 ,    1.788688063621521  ,
     0.8113790154457092 ,   -1.6073065996170044 ,   -0.41895508766174316,
    -3.7118613719940186 ,    1.2734129428863525 ,   -3.2668445110321045 ,
    -2.364816665649414  ,   -5.205628395080566  ,    0.7257134318351746 ,
    -0.68482506275177   ,   -0.17222213745117188,   -2.789684772491455  ,
    -0.4725567102432251 ,    1.1252318620681763 ,   -1.9029902219772339 ,
    -0.3677123486995697 ,    1.7484147548675537 ,    1.2180098295211792 ,
     0.1451737880706787 ,   -1.6704158782958984 ,    3.0325682163238525 ,
    -0.6119042634963989 ,   -0.12874242663383484,   -5.221405029296875  ,
    -0.5518890619277954 ,   -3.175659656524658  ,   -0.9044980406761169 ,
     4.666261672973633  ,   -0.9658599495887756 ,    0.08372581005096436,
    -1.725138545036316  ,    0.38912346959114075,    2.6599555015563965 ,
    -1.8795086145401    ,    0.8947250843048096 ,   -1.3678568601608276 ,
    -2.935791015625     ,   -1.0786247253417969 ,   -5.037256717681885  ,
    -1.5650169849395752 ,   -0.2994163930416107 ,   -1.8929831981658936 ,
    -0.45926815271377563,   -2.1579790115356445 ,    3.370300054550171  ,
     0.3635951578617096 ,   -0.7185401916503906 ,    0.6735578775405884 ,
     0.7790338397026062 ,    0.1995564103126526 ,   -0.3961448669433594 ,
     6.758083343505859  ,   -2.4664392471313477 ,   -0.635779619216919  ,
     0.3133525848388672 ,   -1.3285956382751465 ,   -0.8668310642242432 ,
     1.3504302501678467 ,    0.5935624837875366 ,   -1.5622153282165527 ,
     0.4145668148994446 ,   -2.329251527786255  ,   -0.5524320006370544 ,
     1.2546930313110352 ,    1.5787068605422974 ,   -0.6846985816955566 ,
     1.2473721504211426 ,    0.34553080797195435,    1.0376827716827393 ,
    -2.2027108669281006 ,   -0.00505302846431732,    1.8995784521102905 ,
    -1.8263349533081055 ,   -1.006705403327942  ,    2.366244077682495  ,
     0.18978317081928253,   -6.019230842590332  ,   -0.19508886337280273,
    -1.1539634466171265 ,    1.9402661323547363 ],
 [  -0.9632101058959961 ,   13.98794174194336   ,  -66.6108627319336    ,
    20.306066513061523  ,   61.96879577636719   ,    4.615771293640137  ,
    29.94545555114746   ,   -5.052845001220703  ,    3.614248275756836  ,
   115.7752456665039    ,  140.41087341308594   ,   52.82991027832031   ,
   -15.40138053894043   ,  145.91546630859375   , -173.51858520507812   ,
   -55.91150665283203   ,    3.637772560119629  ,   50.755615234375     ,
   -30.41927146911621   ,   19.174964904785156  ,  -60.708831787109375  ,
  -124.10702514648438   ,   -7.81304931640625   ,  -63.1380615234375    ,
  -168.76800537109375   ,   84.03216552734375   ,   14.547526359558105  ,
   -32.68640899658203   ,    5.135706424713135  ,  -14.040443420410156  ,
    -2.8071489334106445 ,  -19.251285552978516  ,   13.621236801147461  ,
   -56.632728576660156  ,  -10.566415786743164  ,  -12.302526473999023  ,
   -20.082454681396484  ,   50.19675064086914   ,   -2.5686793327331543 ,
   -26.895851135253906  ,   12.198263168334961  , -114.40180969238281   ,
   -15.511680603027344  ,  -55.54429626464844   ,   -2.881389617919922  ,
    53.76790237426758   ,   -1.7002344131469727 ,   62.60518264770508   ,
    22.444961547851562  ,  -49.1593017578125    ,  -11.686796188354492  ,
   -24.95383071899414   ,  116.7605209350586    ,  -19.19089126586914   ,
    28.119354248046875  ,   97.23171997070312   ,   -7.567384719848633  ,
    -4.546114921569824  ,   60.33082580566406   ,   -2.302295684814453  ,
    -8.688804626464844  ,  -71.16078186035156   ,  -65.49302673339844   ,
   -48.81852340698242   ,   -3.389904022216797  ,  -24.55057716369629   ,
     3.6034860610961914 , -133.3779754638672    ,   42.84089279174805   ,
   -39.19232177734375   ,    0.5356557369232178 ,  -49.983253479003906  ,
  -107.73373413085938   ,   15.936756134033203  ,   47.174495697021484  ,
  -127.41604614257812   ,  -86.14427185058594   ,    6.63154411315918   ,
    -2.360443592071533  ,    6.032286643981934  ,  -10.826026916503906  ,
     2.557351589202881  ,   20.597217559814453  ,  -88.95402526855469   ,
   -10.783321380615234  ,   55.57981872558594   ,  -71.68777465820312   ,
     2.6902711391448975 ,  -27.345745086669922  ,  -36.11329650878906   ,
    16.91690444946289   ,  -10.40919303894043   ,   49.747066497802734  ,
     9.063253402709961  ,   12.466694831848145  ,   15.672181129455566  ,
   105.89556884765625   ,    6.662642478942871  ,  -26.08316421508789   ,
   -19.969635009765625  ,  -24.482433319091797  , -107.1537094116211    ,
   -17.14089584350586   ,  -53.68439483642578   ,  -18.266197204589844  ,
    13.450613975524902  ,  -28.96001434326172   ,  -60.69255065917969   ,
    93.97457885742188   ,  -68.47323608398438   ,  -21.2471866607666    ,
     0.1383037567138672 ,    5.400506019592285  ,    1.4338035583496094 ,
   -24.901445388793945  ,  -21.576698303222656  ,  -52.93232345581055   ,
     7.4236650466918945 ,   -6.450289249420166  ,   -4.722324371337891  ,
    17.21319580078125   ,   -8.907939910888672  ,   18.19790267944336   ,
   -34.312740325927734  ,   33.214317321777344  , -115.3875503540039    ,
     3.2951669692993164 ,   17.60546112060547   ],
 [  -3.2548484802246094 ,  -30.493104934692383  ,  130.41966247558594   ,
   -34.54368591308594   , -104.55203247070312   ,  -12.165519714355469  ,
   -58.085243225097656  ,    8.099761962890625  ,   -7.804285049438477  ,
  -196.8974609375       , -226.4805145263672    , -110.36767578125      ,
    11.829538345336914  , -233.6387481689453    ,  259.36663818359375   ,
   100.0036392211914    ,    0.38361167907714844,  -81.13873291015625   ,
    61.8962287902832    ,  -39.3520393371582    ,  110.81681060791016   ,
   225.98597717285156   ,   15.685064315795898  ,   66.48664093017578   ,
   285.9241027832031    , -147.5376434326172    ,  -17.7462158203125    ,
    55.94214630126953   ,  -20.51806640625      ,   28.06157684326172   ,
     5.785219192504883  ,   25.458711624145508  ,  -29.245819091796875  ,
   116.42877197265625   ,   17.757566452026367  ,   20.369497299194336  ,
    32.14830017089844   ,  -83.752685546875     ,    0.9352407455444336 ,
    38.23802947998047   ,  -22.012441635131836  ,  174.35662841796875   ,
    18.343841552734375  ,   91.08059692382812   ,   -0.5300827026367188 ,
   -91.81036376953125   ,   18.83957290649414   , -107.9011001586914    ,
   -45.151344299316406  ,  113.65594482421875   ,   29.51114273071289   ,
    40.828155517578125  , -200.68370056152344   ,   27.842342376708984  ,
   -46.152442932128906  , -146.0960235595703    ,    8.70317268371582   ,
     6.85931396484375   , -111.41254425048828   ,    2.668153762817383  ,
    17.641965866088867  ,  120.95262908935547   ,  126.20552825927734   ,
    77.25962829589844   ,   16.131175994873047  ,   43.703033447265625  ,
    -3.9099435806274414 ,  226.30892944335938   ,  -89.7385482788086    ,
    73.1829833984375    ,   -1.8733134269714355 ,  102.81503295898438   ,
   151.59024047851562   ,  -27.76433753967285   ,  -88.20951080322266   ,
   262.83941650390625   ,  147.19036865234375   ,  -11.447124481201172  ,
     3.584076404571533  ,  -11.502707481384277  ,    6.0984954833984375 ,
    -5.70494270324707   ,  -42.52846145629883   ,  154.4132080078125    ,
     4.7550506591796875 ,  -88.20769500732422   ,  114.92748260498047   ,
    -4.996600151062012  ,   35.33114242553711   ,   54.48678207397461   ,
   -30.499746322631836  ,    3.021951913833618  , -103.78143310546875   ,
   -14.357003211975098  ,  -25.100204467773438  ,  -35.88386535644531   ,
  -183.51995849609375   ,  -11.772982597351074  ,   66.62046813964844   ,
    14.462451934814453  ,   39.09626007080078   ,  195.60885620117188   ,
    30.64389419555664   ,  105.87276458740234   ,   26.127714157104492  ,
   -28.519824981689453  ,   49.86925506591797   ,  129.54513549804688   ,
  -166.67330932617188   ,  129.32882690429688   ,   37.288055419921875  ,
    -6.627218246459961  ,  -15.979656219482422  ,    4.988847732543945  ,
    42.17365264892578   ,   35.86724853515625   ,  106.03460693359375   ,
    -5.274869918823242  ,   10.772054672241211  ,   28.462158203125     ,
   -22.435604095458984  ,   29.32415771484375   ,  -27.868574142456055  ,
    44.54492950439453   ,  -45.011924743652344  ,  226.5327911376953    ,
    -5.489826202392578  ,  -34.780208587646484  ],
 [   4.040661811828613  ,   16.407814025878906  ,  -65.62834930419922   ,
    15.55793285369873   ,   44.42875671386719   ,    7.546943664550781  ,
    27.585994720458984  ,   -3.071277141571045  ,    3.3150954246520996 ,
    85.3355712890625    ,   92.92913055419922   ,   55.92280960083008   ,
     0.08769893646240234,   95.34581756591797   ,  -99.63416290283203   ,
   -45.83177185058594   ,   -3.2563157081604004 ,   32.3161735534668    ,
   -30.798192977905273  ,   20.768898010253906  ,  -52.802879333496094  ,
  -107.95185852050781   ,   -7.776060581207275  ,  -15.63589859008789   ,
  -129.16305541992188   ,   66.5887680053711    ,    6.2100114822387695 ,
   -23.97587776184082   ,   14.290338516235352  ,  -15.114559173583984  ,
    -3.4177770614624023 ,   -9.45026683807373   ,   16.633508682250977  ,
   -59.99365997314453   ,   -6.867533206939697  ,   -7.6453752517700195 ,
   -14.257600784301758  ,   35.3724365234375    ,    1.5577728748321533 ,
   -14.821257591247559  ,    9.752501487731934  ,  -70.0743637084961    ,
    -6.437264442443848  ,  -38.95370864868164   ,    6.058708190917969  ,
    39.40177917480469   ,  -16.034292221069336  ,   46.73675537109375   ,
    23.849777221679688  ,  -61.30323028564453   ,  -16.871871948242188  ,
   -17.250038146972656  ,   87.04915618896484   ,  -10.342391967773438  ,
    22.19959259033203   ,   57.42500686645508   ,   -2.703807830810547  ,
    -2.4171485900878906 ,   52.62714767456055   ,   -0.6660623550415039 ,
    -9.602639198303223  ,  -54.34926223754883   ,  -63.39440155029297   ,
   -31.694210052490234  ,  -12.503532409667969  ,  -20.394899368286133  ,
     0.7120418548583984 , -100.17389678955078   ,   46.048370361328125  ,
   -35.64031982421875   ,    1.153637170791626  ,  -51.73323059082031   ,
   -56.519752502441406  ,   14.060781478881836  ,   42.254425048828125  ,
  -139.3788299560547    ,  -64.9444351196289    ,    5.55820369720459   ,
    -0.48406365513801575,    6.0716657638549805 ,    2.2770509719848633 ,
     4.1053667068481445 ,   23.020864486694336  ,  -71.45805358886719   ,
     4.833491802215576  ,   36.077049255371094  ,  -48.06439971923828   ,
     2.7381865978240967 ,  -11.483840942382812  ,  -21.663850784301758  ,
    14.969947814941406  ,    4.710315704345703  ,   55.961822509765625  ,
     5.724251747131348  ,   13.406953811645508  ,   20.211528778076172  ,
    81.37295532226562   ,    5.065033912658691  ,  -37.62613296508789   ,
    -3.621103286743164  ,  -16.962852478027344  ,  -93.32746887207031   ,
   -15.244779586791992  ,  -53.23879623413086   ,   -9.20907211303711   ,
    14.351068496704102  ,  -21.24855613708496   ,  -68.91795349121094   ,
    76.64108276367188   ,  -62.587032318115234  ,  -17.05025291442871   ,
     6.338324546813965  ,   10.101388931274414  ,   -5.731000900268555  ,
   -20.193374633789062  ,  -13.651235580444336  ,  -55.006195068359375  ,
    -0.5211896896362305 ,   -4.54081392288208   ,  -21.810306549072266  ,
     8.421244621276855  ,  -16.691102981567383  ,   10.748647689819336  ,
   -14.601533889770508  ,   20.013916015625     , -113.09635925292969   ,
     2.1946792602539062 ,   17.412500381469727  ]])
    b1 = jnp.array([-1.9981708526611328 , -1.6332927942276    , -0.7837842106819153 ,
 -1.2841345071792603 , -0.22536703944206238,  1.5577454566955566 ,
  0.0112126087769866 , -1.0890254974365234 ,  2.3364293575286865 ,
 -1.6100143194198608 , -0.5058126449584961 , -1.1532697677612305 ,
 -1.1711174249649048 , -0.986660361289978  , -1.0603426694869995 ,
 -0.7673891186714172 ,  0.9711342453956604 , -1.686197280883789  ,
 -0.6267324686050415 , -0.11543142795562744, -0.15506738424301147,
  0.07384883612394333, -2.3810510635375977 , -0.584926187992096  ,
 -0.13055391609668732, -1.5643311738967896 , -1.1396386623382568 ,
  0.7028676271438599 ,  1.3668030500411987 ,  1.8314541578292847 ,
  1.8212546110153198 ,  0.560804545879364  ,  2.084618330001831  ,
 -1.3334075212478638 , -1.8827825784683228 , -1.3430355787277222 ,
 -0.6639325618743896 , -0.19711561501026154, -2.1934657096862793 ,
 -1.9146571159362793 ,  0.5995593070983887 , -0.6245555877685547 ,
 -1.2384817600250244 , -0.4546106159687042 , -1.2707064151763916 ,
 -1.4121613502502441 , -2.012434720993042  , -1.1298449039459229 ,
 -1.4590330123901367 , -1.9786823987960815 , -1.9676017761230469 ,
 -2.104036808013916  , -0.6538645029067993 , -0.5398136377334595 ,
  0.8506576418876648 , -1.5070226192474365 , -0.8807231187820435 ,
  3.0029735565185547 , -1.2939884662628174 , -1.1407546997070312 ,
  0.9870569109916687 , -1.486853837966919  , -1.5212823152542114 ,
  0.10222257673740387, -1.9796921014785767 , -1.732766032218933  ,
  1.3619003295898438 , -0.465619832277298  ,  0.8898173570632935 ,
 -0.32781746983528137, -0.17644469439983368, -1.8053739070892334 ,
 -0.7677116990089417 , -0.7559933662414551 ,  0.5082131624221802 ,
 -1.3680188655853271 ,  0.0637621283531189 ,  0.17589980363845825,
 -1.7871642112731934 , -1.1149866580963135 , -1.7986794710159302 ,
  1.808502197265625  , -1.7040257453918457 , -0.6817774772644043 ,
 -2.233429193496704  , -0.4659684896469116 , -0.05540827661752701,
  0.47758424282073975, -1.7192835807800293 , -0.9225650429725647 ,
 -1.9348278045654297 , -2.275402307510376  , -0.8846752643585205 ,
 -0.8487991690635681 , -1.9797157049179077 , -1.7821707725524902 ,
  1.4392794370651245 , -1.9962862730026245 , -0.5116490721702576 ,
 -0.6993059515953064 , -0.7057291269302368 , -0.6428607106208801 ,
  0.00132823246531188, -1.8475309610366821 ,  1.385801911354065  ,
 -1.9635483026504517 ,  1.2109825611114502 , -1.0952457189559937 ,
 -1.4253730773925781 , -1.7612807750701904 , -1.0441572666168213 ,
 -2.520479679107666  , -0.4630501866340637 , -1.2859045267105103 ,
 -0.9431197643280029 , -0.43532294034957886, -1.7979942560195923 ,
 -0.26406624913215637, -1.9611167907714844 , -2.3272793292999268 ,
 -0.8760437965393066 , -2.0038564205169678 , -1.1317310333251953 ,
 -1.906333088874817  , -1.6712706089019775 , -1.028365135192871  ,
 -0.9681423306465149 ,  0.9127869606018066 ])
    W2 = jnp.array([[-0.388274222612381  ,  0.02269302308559418, -0.7100057005882263 ,
  -0.03364372998476028,  0.07275530695915222,  0.45546966791152954,
   0.11253683269023895, -0.03423355892300606,  0.29513150453567505,
   0.4374788701534271 ,  0.08887126296758652, -1.028020977973938  ,
   0.18083153665065765, -0.16395872831344604, -0.23499280214309692,
   0.9633663296699524 , -0.7162154316902161 ,  0.06554887443780899,
   0.4931509494781494 ,  0.04394914209842682,  0.22251616418361664,
   0.2970075011253357 ,  0.00256169168278575,  0.4526393413543701 ,
  -0.461009681224823  , -0.30900809168815613,  0.47548770904541016,
  -0.06864909827709198, -0.39456477761268616, -0.3471493721008301 ,
  -0.34621819853782654, -0.6733869910240173 ,  0.7411023378372192 ,
  -1.0272901058197021 , -0.10515163838863373,  0.00225885910913348,
  -0.00403476925566792,  0.09707299619913101,  0.8180258870124817 ,
  -0.11640284955501556, -0.8282004594802856 , -0.9039372205734253 ,
   1.0433045625686646 ,  0.9436569809913635 , -0.1368202120065689 ,
  -0.06098888069391251,  0.15658743679523468, -0.15119661390781403,
  -0.7002399563789368 , -0.21370495855808258, -0.3126577138900757 ,
  -1.9083131551742554 , -0.8884238004684448 , -0.03058355487883091,
  -0.39797163009643555, -0.7243683338165283 , -0.200716033577919  ,
  -0.2171734720468521 , -1.6748806238174438 , -0.03197398036718369,
   0.48122549057006836,  0.22067831456661224, -0.03167092427611351,
  -0.6000992655754089 , -0.614676833152771  ,  1.104818344116211  ,
  -0.03572700917720795, -1.4400620460510254 , -0.2401358038187027 ,
   0.22351017594337463, -0.4188484847545624 ,  0.12922964990139008,
  -0.24254891276359558,  0.5743710398674011 , -0.3336082994937897 ,
  -0.5146571397781372 , -2.40879487991333   ,  0.3408815860748291 ,
  -0.45297935605049133, -0.8865652680397034 ,  0.532198429107666  ,
   0.00769833056256175, -0.37622812390327454,  0.9384912252426147 ,
   0.41192835569381714,  0.2575543224811554 , -0.21628063917160034,
  -0.866565465927124  , -1.7260489463806152 ,  0.8200095295906067 ,
   0.42670178413391113, -0.5571765303611755 , -1.0929450988769531 ,
  -0.5981283783912659 , -0.4100033640861511 ,  0.05650391429662704,
  -0.5700812339782715 , -0.2796962261199951 , -0.19209370017051697,
  -0.9502902626991272 , -1.1880290508270264 , -0.5599217414855957 ,
  -0.8168908357620239 ,  0.4463626444339752 ,  0.22120454907417297,
   0.00768398214131594, -0.2553655207157135 , -0.1442752480506897 ,
  -0.49405843019485474,  0.578636884689331  , -0.5265151262283325 ,
   0.6205168962478638 ,  0.6086626648902893 , -0.12709327042102814,
  -0.04155402630567551,  0.16480277478694916,  0.23425795137882233,
  -0.47820228338241577, -0.8881114721298218 ,  0.0464916042983532 ,
  -0.31422877311706543, -0.767349898815155  , -0.00626274198293686,
  -0.6661577224731445 ,  0.3367091715335846 , -2.155156373977661  ,
   1.1140955686569214 , -1.6906182765960693 ],
 [-1.5798146724700928 , -0.1289331614971161 ,  0.03557549044489861,
  -0.09931162744760513,  0.26933279633522034,  0.24358326196670532,
  -0.16781838238239288, -0.16500896215438843,  0.19419455528259277,
   0.21965646743774414,  0.1965743452310562 , -0.7798380851745605 ,
  -0.3166287839412689 , -0.03838446363806725, -0.2634390592575073 ,
  -0.1069839596748352 , -0.3471488356590271 ,  0.6705085635185242 ,
   0.3271319270133972 ,  0.11850529164075851,  0.13556739687919617,
   0.13568808138370514, -0.21492953598499298, -0.9537772536277771 ,
   0.19804172217845917, -0.0842236652970314 , -0.20392724871635437,
   0.6432764530181885 ,  0.34108972549438477,  0.03127686679363251,
  -0.4513334035873413 , -0.5250188112258911 , -0.40111973881721497,
   0.3151511549949646 , -0.3167971968650818 ,  0.8464155197143555 ,
  -0.21010999381542206,  0.07208061963319778, -1.010701298713684  ,
  -0.5929632782936096 , -1.4361162185668945 , -1.464966058731079  ,
   0.05145217105746269, -0.416841596364975  , -0.28521156311035156,
  -0.7796643376350403 , -1.443933129310608  ,  0.3137223720550537 ,
  -0.42459434270858765, -0.5752776861190796 , -1.6919875144958496 ,
  -0.9573063254356384 , -0.7236484289169312 ,  0.35713842511177063,
  -1.1776037216186523 ,  0.44848868250846863,  0.2621036469936371 ,
   0.11413834989070892, -0.620806097984314  , -0.5817602276802063 ,
  -0.02505308762192726, -0.14024333655834198, -0.4709740877151489 ,
  -0.6894363760948181 , -1.5309449434280396 , -1.5757246017456055 ,
  -0.8970630764961243 ,  0.10948143154382706,  0.10450728237628937,
  -0.2836146354675293 , -0.05654763430356979, -0.1542184054851532 ,
   0.7443773150444031 ,  0.914126455783844  ,  0.3738965690135956 ,
  -0.34303584694862366,  0.03129507973790169,  1.0206619501113892 ,
   0.11794279515743256,  0.21730753779411316, -0.05016902461647987,
  -0.09257130324840546, -0.15707877278327942, -0.18844686448574066,
   0.01762227527797222, -0.2536706328392029 ,  0.03862268477678299,
   0.5181081891059875 ,  0.2952950596809387 , -1.4711796045303345 ,
  -1.2216047048568726 , -0.3049018383026123 , -0.5591252446174622 ,
  -0.5922027826309204 , -0.3596527874469757 ,  0.8492732048034668 ,
   0.6051295399665833 ,  0.5100470185279846 ,  0.38124293088912964,
  -0.3796582818031311 , -0.6837531924247742 ,  0.5095630884170532 ,
   0.33672982454299927, -0.5107237100601196 , -0.6010832190513611 ,
   0.26213520765304565,  0.2881845235824585 , -1.6570945978164673 ,
  -0.9079027771949768 , -0.0867055132985115 , -0.1515781134366989 ,
  -0.6969629526138306 , -1.039654016494751  ,  0.6768553256988525 ,
   0.22086995840072632, -0.8423513174057007 , -2.123300313949585  ,
  -1.157322883605957  ,  0.6852682828903198 , -0.44782954454421997,
   0.04119884967803955,  0.24382859468460083, -0.1938725858926773 ,
   0.23916226625442505,  0.18499919772148132,  0.6368284821510315 ,
  -0.11783210188150406,  0.05797941610217094],
 [ 0.13521340489387512,  1.5139745473861694 , -0.26945969462394714,
   0.4821368157863617 ,  0.05391977354884148, -0.12855683267116547,
   0.2784138023853302 , -0.33044251799583435,  0.2888040244579315 ,
   0.5309808850288391 , -0.33510446548461914, -0.05892191827297211,
  -0.2516637444496155 ,  0.6003645658493042 , -0.20922498404979706,
  -0.2473047524690628 ,  1.0375542640686035 ,  0.661850094795227  ,
   0.5560911893844604 , -0.6433552503585815 , -0.8238161206245422 ,
   0.5126591920852661 ,  0.15036481618881226,  1.3006497621536255 ,
   0.446607768535614  , -0.02165250852704048,  0.25561121106147766,
  -1.1726456880569458 ,  0.5883937478065491 ,  0.4593522846698761 ,
   0.15371756255626678,  0.3062037527561188 , -0.17181140184402466,
  -0.03857021406292915,  0.26494401693344116,  1.4755984544754028 ,
   1.9207781553268433 , -0.24117136001586914,  0.8488891124725342 ,
  -0.3007581830024719 ,  1.4015740156173706 , -0.9179046750068665 ,
   1.13303542137146   ,  1.0354077816009521 ,  1.7350857257843018 ,
   0.5266887545585632 ,  1.9388327598571777 , -0.7755470871925354 ,
   0.7169515490531921 ,  0.7488058805465698 , -0.5652969479560852 ,
   0.41292569041252136,  1.6265908479690552 , -0.42113298177719116,
   1.0112476348876953 , -0.13113300502300262, -0.4417558014392853 ,
   0.4681253433227539 ,  0.5292928218841553 ,  0.3572259247303009 ,
   0.2654469907283783 ,  0.7517948746681213 ,  0.1306767761707306 ,
   0.8391206860542297 ,  0.11963681876659393,  1.1762874126434326 ,
  -0.8397907018661499 , -0.4086362421512604 , -0.3946664035320282 ,
   1.232424020767212  ,  0.7317490577697754 , -0.8611737489700317 ,
   0.34837958216667175,  0.8047443628311157 ,  1.0307400226593018 ,
   1.2555609941482544 ,  1.4642155170440674 ,  0.06968256086111069,
   2.100295066833496  , -0.00278180930763483, -0.02346882224082947,
   0.19372119009494781,  0.05484027042984962, -0.8547704815864563 ,
   0.0271790586411953 ,  0.18643292784690857,  0.1816626936197281 ,
   0.28212040662765503,  0.7095914483070374 ,  0.7577656507492065 ,
   1.881615161895752  , -0.8511888384819031 ,  0.02564970031380653,
  -0.2810850441455841 ,  0.6250196099281311 ,  0.11738339811563492,
   0.12964685261249542,  0.6391614079475403 ,  0.17700400948524475,
  -0.01606211066246033, -0.0477670319378376 ,  0.7594351172447205 ,
   0.4574027955532074 ,  1.2144743204116821 , -0.5972089171409607 ,
  -0.4270695745944977 ,  0.2395475208759308 ,  0.44035574793815613,
   1.104596734046936  ,  1.2456295490264893 ,  0.08335475623607635,
   0.27891644835472107,  0.13147015869617462,  0.6610783934593201 ,
  -0.3648012578487396 ,  1.1683934926986694 ,  0.3318291902542114 ,
  -0.3294888436794281 , -0.5718095302581787 ,  0.3834713101387024 ,
  -0.40692681074142456,  0.17362713813781738,  0.08861424773931503,
   0.8164385557174683 ,  0.2102920413017273 ,  0.332326740026474  ,
   0.4338837265968323 , -0.12070868909358978],
 [-0.7242889404296875 , -0.5076386332511902 ,  0.02724766358733177,
   0.37729668617248535, -0.04495024681091309,  0.3201981484889984 ,
   0.45957353711128235, -0.2312401831150055 , -0.02542754448950291,
   0.2030697911977768 ,  0.03429674729704857, -0.11778859794139862,
   0.06854564696550369,  0.18221746385097504,  0.3580896556377411 ,
  -0.10922147333621979,  0.9526957869529724 ,  0.22564585506916046,
  -1.3328431844711304 ,  0.565468430519104  , -0.18228107690811157,
  -0.04865284636616707,  0.3519625961780548 , -0.48045018315315247,
  -0.14138659834861755, -0.05607252940535545, -0.15524406731128693,
  -0.15606051683425903,  0.18398720026016235, -0.44101837277412415,
  -0.6002837419509888 , -0.25795063376426697,  0.01108986325562   ,
  -0.0582854300737381 ,  0.15887479484081268,  0.6535170674324036 ,
  -0.23298212885856628,  0.393292099237442  , -0.15795113146305084,
  -0.2948402166366577 , -0.0343630276620388 , -0.1753382384777069 ,
   0.6845370531082153 ,  2.5336434841156006 , -0.01693687587976456,
   0.13224640488624573, -0.4311041533946991 , -0.38518255949020386,
  -0.4088258445262909 ,  0.5345835089683533 ,  0.35738852620124817,
  -0.21200469136238098, -0.47861576080322266, -0.18647684156894684,
  -0.4439164400100708 ,  0.08903145045042038,  0.6366568207740784 ,
   0.4717530906200409 ,  0.15493980050086975,  0.383608341217041  ,
  -0.939588189125061  ,  0.09334303438663483, -0.24528101086616516,
  -0.5044163465499878 ,  0.6534507870674133 , -0.6222480535507202 ,
  -0.6752006411552429 , -0.20833681523799896, -0.46232807636260986,
  -0.3194313645362854 , -0.3084646165370941 ,  0.38589176535606384,
   0.4754469692707062 ,  0.21661987900733948,  0.19228115677833557,
   0.5962097644805908 ,  0.11081306636333466,  0.5541921257972717 ,
   0.5412458181381226 ,  0.41129404306411743,  0.44364070892333984,
  -1.1293011903762817 ,  0.1725481152534485 , -0.05784400552511215,
  -0.06705943495035172,  0.05487070977687836, -0.07264690101146698,
  -0.02871661633253098,  0.33519935607910156, -0.23003026843070984,
   0.19982296228408813, -0.14626401662826538,  0.41029444336891174,
   0.13930433988571167,  0.01541336067020893,  0.11363757401704788,
  -0.28157687187194824,  0.3586709499359131 , -0.9457197785377502 ,
  -0.3004084825515747 , -0.6143112182617188 , -0.20096954703330994,
   0.09456691890954971,  0.07390212267637253,  0.3023553192615509 ,
   0.2985730469226837 ,  0.1645151823759079 , -0.48969584703445435,
   0.988835871219635  , -0.2389371544122696 ,  0.2668079137802124 ,
   1.2566883563995361 ,  0.09141530096530914,  0.13431386649608612,
   0.03377736359834671, -0.09830071032047272,  0.48589029908180237,
   0.6563875675201416 , -0.04756811633706093,  0.02087352238595486,
   0.6789409518241882 ,  0.2131628394126892 ,  0.05506464838981628,
   0.21872879564762115, -0.2854481041431427 ,  0.1653175801038742 ,
   0.300077348947525  ,  0.9847127199172974 ],
 [-0.00581681728363037,  0.9574531316757202 , -0.1415753960609436 ,
  -0.9070876836776733 ,  0.3897533714771271 , -0.7171717882156372 ,
   1.1251695156097412 ,  0.5519862771034241 ,  0.831372082233429  ,
  -0.08382416516542435, -0.02291557937860489,  0.03439195081591606,
   0.2934289574623108 ,  1.5478419065475464 , -0.14734576642513275,
  -0.07565117627382278,  0.4525889754295349 , -0.2551425099372864 ,
  -1.6050529479980469 ,  0.18597003817558289,  0.3596598505973816 ,
  -0.26350101828575134,  1.8134046792984009 ,  0.08512364327907562,
  -0.03791798278689384,  0.817821204662323  ,  2.449024200439453  ,
  -0.4286702871322632 ,  1.1129502058029175 , -1.569968819618225  ,
   0.02675545401871204,  0.28627461194992065,  0.9554084539413452 ,
  -0.8388630747795105 ,  0.08761781454086304, -0.2180176079273224 ,
   0.04026054963469505,  0.01528388354927301, -0.694039523601532  ,
  -0.05489886179566383, -1.9706028699874878 ,  0.2997549772262573 ,
  -0.6265251636505127 , -1.4056869745254517 , -2.3760435581207275 ,
   0.45500093698501587, -0.5421738028526306 ,  0.10496063530445099,
  -0.14491309225559235,  0.175113245844841  ,  0.7513394951820374 ,
   0.04119585826992989, -2.107889413833618  , -0.03955065086483955,
   0.13226211071014404,  0.25716322660446167,  0.62391597032547   ,
  -0.00243777059949934,  0.06442274898290634,  0.03033945336937904,
   0.9021342396736145 ,  1.0438019037246704 ,  1.4595849514007568 ,
  -0.04778373241424561, -0.13727034628391266,  0.131175696849823  ,
  -1.6279138326644897 , -1.523266077041626  ,  1.1958476305007935 ,
   1.4285273551940918 ,  0.2064352184534073 ,  0.31823083758354187,
   0.4847846031188965 , -0.24642662703990936,  0.05711224675178528,
  -0.7198691368103027 ,  1.2397725582122803 , -0.45399364829063416,
   0.02489054203033447,  0.640602707862854  ,  0.04950201883912086,
  -0.4812198281288147 ,  0.2815640866756439 ,  0.13647207617759705,
  -0.11228246986865997,  0.3687184453010559 ,  0.7498414516448975 ,
  -2.8128600120544434 , -0.16495655477046967, -0.24765543639659882,
  -0.11888300627470016,  0.3194458484649658 , -1.6099575757980347 ,
  -1.1989295482635498 , -0.6995019316673279 ,  0.22240175306797028,
   1.1580268144607544 , -0.45046383142471313,  0.01757499575614929,
  -0.0048759737983346 , -0.1824483424425125 ,  0.6155290603637695 ,
  -0.3927941620349884 , -0.08909666538238525,  0.08612760156393051,
   1.01725172996521   ,  0.1803286224603653 , -0.5932390093803406 ,
   0.39615270495414734,  0.20422431826591492, -2.135185480117798  ,
   1.1789830923080444 ,  0.26365378499031067, -0.8355596661567688 ,
   0.03427783027291298,  0.5978456139564514 , -1.4771186113357544 ,
   0.4809284508228302 ,  0.02145013771951199,  0.09792987257242203,
  -0.08749603480100632,  0.6882911920547485 , -0.25442659854888916,
   0.4958135783672333 ,  1.0024124383926392 , -1.3713014125823975 ,
  -0.4736137390136719 ,  1.3239980936050415 ],
 [ 1.122283935546875  ,  0.7652469277381897 ,  0.5665379166603088 ,
  -0.4202263653278351 ,  1.3150537014007568 , -0.8694930076599121 ,
  -0.9538620710372925 ,  0.6693185567855835 , -1.8859872817993164 ,
  -3.604935884475708  ,  0.561804473400116  ,  0.4157554507255554 ,
   0.4141758382320404 , -0.14973783493041992, -1.9105823040008545 ,
   2.3643815517425537 , -0.5055831670761108 , -1.3280117511749268 ,
  -0.9751507639884949 ,  3.4128739833831787 ,  0.2503283619880676 ,
  -0.21337518095970154, -0.6855741739273071 ,  0.31007638573646545,
   0.8776095509529114 ,  0.9763081073760986 , -0.8813657164573669 ,
  -1.9687185287475586 , -1.7566044330596924 ,  1.028601884841919  ,
   0.510287344455719  ,  0.9425667524337769 ,  1.1913731098175049 ,
   0.7134711742401123 , -0.2820259630680084 , -0.41639336943626404,
  -0.6395155787467957 , -1.3266810178756714 ,  3.596953868865967  ,
   0.02934623323380947,  2.6035513877868652 ,  0.13507281243801117,
   2.534921646118164  ,  3.4832704067230225 , -0.39943376183509827,
  -1.320395827293396  , -0.846098780632019  , -0.7616708278656006 ,
   0.5759614109992981 , -0.5027220845222473 ,  1.9790453910827637 ,
  -0.31270286440849304, -0.9361076951026917 ,  0.5532365441322327 ,
   0.48106491565704346, -3.208305597305298  , -1.8823596239089966 ,
   0.04012782126665115, -0.22445517778396606, -1.2709218263626099 ,
  -0.6246234178543091 ,  1.0348620414733887 ,  2.0582275390625    ,
   0.05946266651153564, -0.5622117519378662 , -3.190929889678955  ,
   1.7732043266296387 ,  1.3622658252716064 , -0.10510604828596115,
  -0.5709137320518494 , -0.5997976064682007 , -2.4687113761901855 ,
   0.37288978695869446, -0.241413414478302  , -2.599754810333252  ,
   0.6554092168807983 , -0.46644479036331177,  0.9549417495727539 ,
  -2.4713125228881836 ,  0.1450064480304718 ,  4.5187668800354    ,
  -0.4976644515991211 , -0.24218998849391937,  1.0335395336151123 ,
   3.3586955070495605 , -0.31319504976272583, -1.5603830814361572 ,
  -1.6573015451431274 , -1.4897016286849976 ,  1.2364459037780762 ,
   0.758507251739502  ,  0.1295381337404251 , -0.43940815329551697,
   0.10899291932582855, -0.06405942887067795,  2.1887502670288086 ,
  -1.602726936340332  ,  0.2352127879858017 ,  0.49051031470298767,
   0.5632241368293762 , -0.6570529937744141 , -1.608276605606079  ,
   1.2025730609893799 ,  0.2234448492527008 , -6.724046230316162  ,
  -1.2097049951553345 , -0.5632204413414001 , -1.7639148235321045 ,
  -0.46963685750961304,  2.264708995819092  , -1.686832070350647  ,
   3.873436689376831  , -0.7851632833480835 , -0.10001718252897263,
  -2.401047945022583  , -1.696947455406189  , -0.14281977713108063,
  -0.09852564334869385,  0.5921716094017029 ,  1.9060713052749634 ,
   1.4619970321655273 , -1.1653006076812744 , -1.6716408729553223 ,
  -5.088164806365967  , -0.08233026415109634,  0.4886575937271118 ,
  -0.00349045498296618, -0.22375020384788513],
 [-0.4562685489654541 , -1.3742343187332153 ,  0.10846772789955139,
  -0.5839873552322388 ,  0.40248188376426697, -0.579729437828064  ,
  -0.5476230382919312 , -0.02961575984954834, -0.5994855165481567 ,
   0.04358550533652306, -0.10824272036552429,  0.02447597682476044,
  -0.6494547128677368 , -1.846216082572937  , -0.6319664716720581 ,
   0.08245538920164108, -0.296185702085495  ,  0.14812685549259186,
  -0.06334619969129562,  0.10133615881204605, -0.04067657887935638,
   0.06486054509878159, -0.45537474751472473,  0.15966378152370453,
  -0.08443087339401245, -0.6727126240730286 , -0.4540611803531647 ,
  -0.7416161894798279 , -0.49071118235588074,  0.3002047836780548 ,
   0.10945317149162292, -0.2556188702583313 , -0.03210718557238579,
  -0.27071353793144226, -0.158900186419487  , -0.9397394061088562 ,
  -0.5021921992301941 , -0.01178471092134714, -0.12410136312246323,
  -0.19647343456745148,  0.20799870789051056, -0.02681424096226692,
   0.2522624731063843 , -0.18573269248008728,  0.41037237644195557,
  -0.2702566683292389 ,  0.5358185768127441 ,  0.5146235227584839 ,
   0.04867829754948616, -0.8609794974327087 ,  0.31060099601745605,
   0.1782086342573166 ,  0.5738196969032288 , -0.04885050281882286,
   0.4734624922275543 ,  0.08232120424509048, -0.2736063599586487 ,
  -0.11675918102264404, -0.36439499258995056,  0.25130346417427063,
  -0.4695151448249817 , -1.1679275035858154 , -0.4253864884376526 ,
   0.11720821261405945, -0.796379566192627  ,  0.10720415413379669,
  -0.10711273550987244, -0.09180545806884766, -0.6872050762176514 ,
  -1.0506824254989624 ,  0.05185548961162567, -0.21851304173469543,
  -0.20120783150196075, -0.19482658803462982,  0.11280487477779388,
  -0.6848440170288086 , -0.6839842796325684 , -0.6314424872398376 ,
  -0.18893548846244812,  0.03888380900025368, -0.568385899066925  ,
   0.5521775484085083 , -0.23906491696834564,  0.14847011864185333,
   0.1529669463634491 , -0.4364165663719177 , -0.38268229365348816,
  -0.01244195085018873, -0.16481433808803558, -0.0366048738360405 ,
  -0.23750413954257965, -0.3976089060306549 , -0.4220340847969055 ,
  -0.02672270871698856,  0.15037879347801208,  0.45787808299064636,
  -0.9746279716491699 , -0.9940633773803711 , -0.6958559155464172 ,
  -0.09901764988899231,  0.03644222021102905,  0.5611826181411743 ,
   0.03396183624863625, -0.02077777497470379,  0.6253451704978943 ,
   0.22699275612831116, -0.04977671429514885, -0.9076195955276489 ,
  -0.16813065111637115, -0.29275745153427124, -0.3160375952720642 ,
   0.323686808347702  ,  0.16835929453372955, -1.3672821521759033 ,
  -0.09033423662185669, -0.17656546831130981, -0.11490551382303238,
  -0.3257454037666321 ,  0.0925535187125206 , -0.19749575853347778,
  -0.8759494423866272 , -0.07585076987743378, -0.4163251519203186 ,
  -0.10053686052560806, -0.29397711157798767,  0.05057555437088013,
  -0.8308607339859009 ,  0.6855382919311523 ],
 [-0.7129914164543152 , -0.7253903746604919 , -0.2586168646812439 ,
   0.46744680404663086, -0.9167096614837646 ,  0.2511045038700104 ,
   0.28509262204170227,  0.8251499533653259 , -0.4002223014831543 ,
  -0.5373897552490234 , -0.3550310432910919 ,  0.7104635238647461 ,
  -0.32636725902557373, -0.40086686611175537, -0.5327789187431335 ,
  -0.1642502397298813 ,  0.6037082076072693 , -0.5114185214042664 ,
  -2.5265142917633057 , -0.29378148913383484,  0.2716728746891022 ,
   1.4251654148101807 ,  0.37181195616722107, -0.9128975868225098 ,
  -0.3979083001613617 , -0.03301366418600082, -0.00019894626166206,
   0.590522050857544  ,  0.13138651847839355, -0.3124750554561615 ,
   0.18058642745018005, -0.02864412777125835,  0.8307713866233826 ,
   0.6663116216659546 , -0.32115858793258667, -0.07156490534543991,
  -0.10118284821510315,  0.3857739567756653 ,  0.307754784822464  ,
   0.18654698133468628, -0.537943422794342  , -0.36436793208122253,
   0.13244853913784027,  2.7440710067749023 ,  0.45927494764328003,
   0.05989193543791771,  0.01700458116829395, -0.4933341145515442 ,
   0.6559385657310486 ,  0.2512052357196808 ,  0.33042627573013306,
   0.04969028010964394, -1.508532166481018  ,  0.07093670964241028,
  -1.0130060911178589 ,  0.4433532953262329 , -0.2430914044380188 ,
   0.2967138886451721 , -0.37429279088974   ,  0.45705896615982056,
  -0.31579288840293884,  0.7615958452224731 , -0.5413325428962708 ,
  -3.0399422645568848 , -1.5573070049285889 ,  0.02688981965184212,
   0.29984572529792786,  0.1225234866142273 , -0.5180310606956482 ,
   0.06849449872970581, -0.5323100686073303 , -0.01710686087608337,
  -1.3724194765090942 , -0.18222709000110626,  0.3738280236721039 ,
   2.7619519233703613 , -0.1214815080165863 , -0.8911991715431213 ,
   0.7599507570266724 ,  0.7600566744804382 ,  0.49872300028800964,
  -1.695846676826477  , -0.00554264197126031, -1.597330927848816  ,
   0.29990139603614807, -0.13984525203704834, -0.06413313001394272,
  -0.633041501045227  , -0.03988857939839363,  1.3623814582824707 ,
  -0.7026242613792419 ,  0.7576618194580078 , -1.6336034536361694 ,
   0.3170507550239563 ,  0.22037500143051147,  0.27864885330200195,
  -0.3237108290195465 ,  0.33266690373420715, -0.6941089034080505 ,
   0.7710973024368286 , -1.1297577619552612 ,  0.5498963594436646 ,
   0.2774820327758789 ,  0.5885194540023804 ,  0.2747022807598114 ,
  -1.1834125518798828 , -0.5919124484062195 ,  0.41691145300865173,
   0.784616231918335  ,  0.21286924183368683, -0.02231191284954548,
   0.07859209924936295,  0.6446875929832458 , -0.00551473069936037,
   0.02410121634602547, -0.42118996381759644,  0.8191272020339966 ,
  -0.44617319107055664, -0.2860373854637146 ,  0.3620683550834656 ,
  -1.201991319656372  ,  0.01916094683110714, -0.12969852983951569,
   0.5656543374061584 ,  1.3536067008972168 , -0.19449959695339203,
  -0.1982712596654892 , -0.16511185467243195],
 [-0.5926932096481323 , -0.518112301826477  ,  0.7110032439231873 ,
  -0.12804025411605835, -0.6928232312202454 ,  0.9384890794754028 ,
  -0.7337369918823242 ,  0.05339246243238449,  0.6080928444862366 ,
  -0.08535294234752655, -0.7720493078231812 ,  0.35778218507766724,
  -0.2373138964176178 , -0.5034282207489014 ,  0.18011462688446045,
  -0.6799806952476501 , -0.1537255197763443 ,  0.2153410166501999 ,
   0.03235173225402832, -0.5393269658088684 , -0.09978394955396652,
  -0.21791879832744598, -0.05791260674595833, -0.26734471321105957,
   0.2418852150440216 , -1.0964069366455078 , -0.11162455379962921,
  -0.10824096947908401, -0.40995046496391296, -0.17131738364696503,
   0.4433269202709198 ,  0.09675624966621399,  0.08918797224760056,
  -0.0597023218870163 , -0.08475293964147568, -0.00372002436779439,
  -0.25009703636169434, -0.0644063875079155 , -0.3570743501186371 ,
  -0.6751295924186707 , -0.31792545318603516,  0.21758706867694855,
   0.12234941124916077, -0.04662607982754707, -0.23561695218086243,
   0.23653215169906616, -0.2214924842119217 ,  0.11645971983671188,
  -0.01072624046355486, -0.15785399079322815,  0.43576687574386597,
  -0.23891960084438324,  0.7335373759269714 , -0.14622896909713745,
   0.41805362701416016, -0.6796936392784119 , -0.24372439086437225,
  -0.40260767936706543, -0.27009499073028564,  0.04653388634324074,
   0.02974953874945641, -0.7229784727096558 ,  0.00510759279131889,
  -0.03781041875481606, -0.3924984931945801 ,  0.42372193932533264,
  -0.06373634189367294,  0.13053074479103088,  0.07155627012252808,
   0.3876301646232605 , -0.7669872641563416 , -0.05374192446470261,
  -0.0901552364230156 ,  0.7261275053024292 , -0.7830041646957397 ,
  -0.25164467096328735, -0.8010846376419067 ,  0.2576051950454712 ,
   0.3546942174434662 ,  0.2593609690666199 , -1.1309545040130615 ,
   0.2885251045227051 , -0.7999659180641174 ,  0.284032940864563  ,
  -0.8386068344116211 , -0.18799734115600586,  0.3911553919315338 ,
  -0.575770378112793  , -0.06190266460180283, -0.11614121496677399,
  -1.1913625001907349 ,  0.40210989117622375,  0.340740829706192  ,
   0.4110960066318512 ,  0.33754870295524597,  0.16993115842342377,
  -0.8019837141036987 , -0.23125436902046204, -0.00573827140033245,
   0.26266103982925415, -1.613638162612915  , -0.41406309604644775,
   0.34286701679229736, -0.5734289884567261 , -0.1824101209640503 ,
   0.35842159390449524, -0.8622134327888489 ,  0.3153192698955536 ,
   0.7373026609420776 , -0.9792523980140686 ,  0.19986112415790558,
   0.30510225892066956, -0.5255959630012512 , -0.36710575222969055,
  -0.10532444715499878, -0.33618247509002686,  0.0995364785194397 ,
   0.4161324203014374 , -2.2878291606903076 , -0.47739890217781067,
   0.16100957989692688, -0.0979582741856575 , -0.05018362775444984,
  -0.877227783203125  , -0.761909008026123  ,  0.6049936413764954 ,
  -0.04705506190657616,  0.2378702163696289 ],
 [-1.2202202081680298 , -0.31087034940719604, -0.4711016118526459 ,
  -0.7156559228897095 ,  0.514601469039917  ,  0.05290417745709419,
  -0.2764664888381958 ,  0.6210426092147827 , -1.398697018623352  ,
  -0.03134796768426895, -0.09097693115472794, -0.6755020022392273 ,
   0.4886247515678406 ,  0.17684206366539001, -0.12023461610078812,
   0.6598766446113586 , -1.2839033603668213 , -0.9125084280967712 ,
   0.3177700936794281 , -0.51526939868927   , -0.00017074980132747,
   0.9641945958137512 , -0.4584082067012787 , -1.0629194974899292 ,
  -0.15062342584133148, -0.2567066252231598 , -0.4847531020641327 ,
   0.00636398978531361, -0.41580361127853394, -0.00004147770232521,
  -0.08573715388774872, -0.2936529815196991 , -0.7654075622558594 ,
  -0.5387527942657471 , -0.14539699256420135, -1.0354536771774292 ,
  -0.33651667833328247, -0.01383462641388178,  1.6514071226119995 ,
  -0.11734356731176376,  1.2276276350021362 ,  0.7143741846084595 ,
   0.5480289459228516 ,  0.348945289850235  , -0.11519281566143036,
  -1.4249088764190674 ,  0.7582194805145264 , -0.14806753396987915,
  -1.3759477138519287 , -0.3562468886375427 , -0.4517040252685547 ,
  -0.5189902782440186 , -0.9192754626274109 , -0.12282547354698181,
  -1.764840841293335  , -0.7022219896316528 ,  0.562214195728302  ,
  -0.5702787637710571 ,  0.05961038544774055,  0.09285319596529007,
   0.1285935938358307 , -0.06263356655836105, -0.24490399658679962,
  -0.01756045408546925, -0.4442785978317261 , -0.7866665720939636 ,
   0.9648995995521545 , -0.22740492224693298, -0.31568536162376404,
  -0.8052634596824646 , -0.01940796710550785, -0.8217169046401978 ,
  -0.2670046091079712 ,  0.29221487045288086, -0.41643163561820984,
   0.8714297413825989 ,  0.26454877853393555, -1.7522386312484741 ,
   0.10700829327106476, -0.37558209896087646,  0.8097962141036987 ,
   0.1848231852054596 , -0.6033024787902832 , -0.9902597069740295 ,
   0.1461542397737503 , -1.0195536613464355 , -0.4982476532459259 ,
  -0.04779592528939247, -0.34553295373916626, -0.3892277777194977 ,
  -0.18284884095191956, -0.7505978941917419 , -0.6573633551597595 ,
  -0.5326545238494873 , -1.07626473903656   , -0.05770908668637276,
  -0.23495328426361084, -0.5360584855079651 ,  0.11071006208658218,
  -0.22826074063777924, -0.9372464418411255 , -0.6547616720199585 ,
  -0.31395813822746277, -0.38408738374710083,  0.33204734325408936,
  -0.375801146030426  , -0.29570379853248596,  0.8242871165275574 ,
  -2.312220811843872  ,  0.9255418181419373 , -0.9116249680519104 ,
  -0.6619547009468079 ,  0.19947510957717896, -0.3197020888328552 ,
  -0.7821193933486938 , -0.7172161340713501 , -1.4226421117782593 ,
  -1.662394404411316  , -0.6544384956359863 , -0.13043490052223206,
  -0.20666222274303436,  0.32301580905914307,  0.7771672606468201 ,
  -0.5891207456588745 ,  0.00938838161528111, -0.6413829326629639 ,
  -0.30299144983291626,  0.5731175541877747 ],
 [ 0.075656957924366  ,  0.3089521825313568 ,  0.0676184743642807 ,
   0.46527203917503357, -1.441308856010437  , -0.5206621289253235 ,
   0.9071127772331238 ,  0.09487669169902802, -1.248592495918274  ,
   0.2833527624607086 , -0.1173788532614708 ,  0.1554812341928482 ,
   0.06795956194400787,  0.5928745865821838 , -3.527614116668701  ,
   0.03123572841286659,  0.29527997970581055,  0.4868195056915283 ,
  -0.1445929855108261 , -0.00822533946484327,  0.488354355096817  ,
  -0.07061243802309036, -0.29798513650894165,  0.40358662605285645,
  -0.4147481620311737 , -2.021724224090576  , -1.7510396242141724 ,
  -0.227637380361557  , -0.20360669493675232,  0.668476402759552  ,
   0.0024460784625262 , -2.2202210426330566 , -0.14859309792518616,
   0.5827014446258545 , -0.1327504813671112 ,  0.2196313887834549 ,
   0.7582151293754578 , -0.08539401739835739, -1.6412103176116943 ,
   0.15170475840568542, -1.2003806829452515 ,  0.5069296360015869 ,
   0.22939680516719818,  1.270971655845642  , -0.5859671235084534 ,
  -0.5936071872711182 ,  0.15611089766025543, -0.19067268073558807,
   0.02598742581903934, -2.255403757095337  , -0.39008739590644836,
  -0.17088814079761505, -0.8290472626686096 , -0.0170313622802496 ,
   0.02399265207350254,  0.09210548549890518, -0.49581727385520935,
   0.05423294380307198, -0.46895918250083923, -0.7909514904022217 ,
  -0.14904002845287323,  0.01347872987389565, -0.9910632967948914 ,
   0.09984764456748962, -0.27545827627182007, -1.243445873260498  ,
   0.16166138648986816,  0.7444502115249634 , -2.9678070545196533 ,
  -0.46444204449653625, -0.2804296314716339 ,  0.31071630120277405,
  -0.01995627768337727, -0.02013673074543476,  1.071068286895752  ,
   0.930978536605835  , -0.2898237705230713 , -0.6443741917610168 ,
  -0.18750697374343872, -0.10480960458517075,  0.06832469999790192,
   0.02036656625568867, -0.13896135985851288, -0.06528718769550323,
   0.20393909513950348, -0.12721969187259674,  0.3388659954071045 ,
  -0.03077344410121441,  0.03419890627264977, -0.0805547758936882 ,
  -0.8938829898834229 ,  0.5763764977455139 , -0.313858300447464  ,
   0.40220606327056885,  0.10788039863109589,  0.7719828486442566 ,
   0.1598910391330719 , -0.002350103110075  , -0.3022245168685913 ,
   0.10034774988889694,  0.5629023909568787 , -1.854154348373413  ,
  -0.10133659094572067,  0.5945534110069275 ,  0.03955893218517303,
  -0.18036210536956787, -0.2529083788394928 , -0.12754222750663757,
  -0.8005295395851135 , -0.11158639937639236, -1.9597091674804688 ,
  -0.8495070934295654 , -0.6064446568489075 , -0.8391892313957214 ,
   0.11131379753351212,  1.4908920526504517 , -0.3173162341117859 ,
  -1.3125139474868774 , -0.15647059679031372,  0.4663892388343811 ,
  -0.09344536066055298,  0.657985508441925  ,  1.2429206371307373 ,
   0.5439211130142212 ,  0.01943052373826504, -0.05119302123785019,
   0.9033899307250977 ,  0.5873335003852844 ],
 [-0.947420597076416  ,  0.4031102955341339 , -0.23148635029792786,
  -0.14452174305915833,  0.7986694574356079 , -0.9347078800201416 ,
  -0.292450875043869  ,  0.44964492321014404,  0.2098734825849533 ,
   0.11689169704914093,  0.10960522294044495, -0.18119119107723236,
   0.17535367608070374, -0.18970613181591034, -0.23610587418079376,
   0.31520020961761475, -1.0583010911941528 , -0.19068783521652222,
   0.5764338374137878 , -0.18931692838668823,  0.0875534936785698 ,
  -0.01375543139874935, -0.4631548225879669 , -0.6587573289871216 ,
  -0.03312195837497711, -0.8438413739204407 ,  0.07563719153404236,
  -0.4253653287887573 ,  0.17306604981422424, -0.152191162109375  ,
  -0.13029968738555908, -0.9694817066192627 , -0.6389756202697754 ,
  -0.45713675022125244,  0.15184716880321503, -1.6284946203231812 ,
  -0.25984448194503784, -0.03306206688284874,  0.14525899291038513,
  -0.08864585310220718, -0.6634843349456787 ,  0.20596425235271454,
  -0.4009167551994324 , -0.2837343215942383 , -0.1963805854320526 ,
  -0.9094169735908508 , -0.5339173078536987 ,  0.6099153161048889 ,
  -0.21153205633163452,  0.09592441469430923, -1.1279207468032837 ,
  -0.26815882325172424, -0.541018545627594  ,  0.07466492056846619,
   0.5408440232276917 , -0.9512185454368591 , -1.6207553148269653 ,
   0.18874743580818176, -0.06639545410871506,  0.284081369638443  ,
  -0.35696834325790405,  0.26695218682289124, -0.3848564922809601 ,
  -2.083496332168579  , -0.01477332785725594, -0.10440199822187424,
   0.05357730761170387, -0.5037418603897095 , -0.7812502980232239 ,
  -0.5581737756729126 ,  0.5265706777572632 , -0.23388706147670746,
   0.40055057406425476, -0.7169870138168335 , -0.43562930822372437,
  -0.5000994205474854 , -0.9368712902069092 , -0.9727417230606079 ,
  -0.19173040986061096,  0.4252788722515106 ,  0.22210417687892914,
   0.01381105370819569, -0.1870666742324829 ,  0.339065283536911  ,
   0.28467729687690735, -0.8374004364013672 , -0.19755634665489197,
  -0.19610869884490967, -0.4927852749824524 , -0.14986202120780945,
   0.06575781852006912, -0.47104892134666443, -0.9202129244804382 ,
  -0.22247371077537537, -0.09115693718194962, -0.6488704085350037 ,
  -0.09210853278636932,  0.5660191178321838 , -0.5894763469696045 ,
  -0.24143749475479126,  0.7550581097602844 , -0.6409600377082825 ,
  -0.17823611199855804,  0.9531823992729187 , -0.5490512847900391 ,
  -0.9821112751960754 ,  0.1882072389125824 , -1.6922301054000854 ,
  -1.286285161972046  ,  0.31925252079963684,  0.14301155507564545,
  -0.4313436448574066 , -0.3053174614906311 ,  0.07610941678285599,
  -0.08100630342960358,  0.7225373387336731 , -0.20055757462978363,
  -0.3556627035140991 ,  0.5174351930618286 , -0.12445630133152008,
  -0.4108774960041046 , -2.5370330810546875 , -0.30003708600997925,
  -0.4198789894580841 ,  0.20042066276073456, -0.3360201418399811 ,
  -0.3128916621208191 , -1.466443419456482  ],
 [ 0.16006805002689362,  0.26584556698799133, -0.02465524710714817,
  -0.27256256341934204, -0.507227897644043  , -0.10013698786497116,
  -0.10747068375349045,  0.3328958749771118 , -0.5173954963684082 ,
  -0.0083292406052351 , -0.4779386818408966 , -0.0893421396613121 ,
  -0.07697300612926483, -0.08091235905885696,  0.11414828896522522,
   0.1893218457698822 ,  0.09963434934616089,  0.0882740169763565 ,
   0.23926618695259094,  0.06499995291233063,  0.16832032799720764,
   0.2903677821159363 , -0.8936456441879272 , -0.28928858041763306,
  -0.07334458827972412,  0.00821774918586016, -1.1030960083007812 ,
  -0.8976916074752808 , -0.189456045627594  , -1.3181947469711304 ,
  -0.1042642816901207 ,  0.4892021715641022 , -0.20152856409549713,
   0.4346004128456116 ,  0.475262314081192  ,  0.4348996579647064 ,
  -0.3724687695503235 ,  0.21093043684959412, -0.29019638895988464,
   0.03733773902058601, -0.18327435851097107,  0.15939202904701233,
  -0.24836543202400208,  0.02103373967111111, -0.2113896608352661 ,
  -0.02413281053304672,  0.0849536880850792 , -1.0471254587173462 ,
  -0.21672919392585754, -0.9413044452667236 ,  1.0547094345092773 ,
   0.42542460560798645,  0.20624369382858276, -0.19859793782234192,
  -0.05878089368343353, -1.2094440460205078 , -0.03087784722447395,
  -0.4656522274017334 ,  0.03535392507910728, -0.6634125709533691 ,
  -2.181828737258911  , -0.14926941692829132, -0.6195028424263    ,
   0.241978719830513  , -0.07444754987955093, -0.10813811421394348,
  -0.5492364168167114 , -0.5750769376754761 , -0.8207910060882568 ,
  -0.22560298442840576,  0.13260866701602936,  0.1479390263557434 ,
  -0.39360830187797546,  0.517798900604248  ,  0.5558719038963318 ,
   0.4474866986274719 , -0.00906980875879526,  0.14776203036308289,
  -0.3010881245136261 , -0.07923250645399094,  1.2860524654388428 ,
  -0.44324350357055664,  0.11026068776845932, -0.9047191739082336 ,
  -0.2391061782836914 , -0.3489932417869568 , -0.48936018347740173,
   0.00316447578370571, -0.47082263231277466,  0.2361646443605423 ,
  -0.6815186738967896 , -0.5335100889205933 ,  0.21485403180122375,
  -0.3842354714870453 ,  0.5463569760322571 , -0.35189011693000793,
  -0.35467445850372314, -0.06280919164419174, -0.10472580045461655,
  -0.07407256215810776,  0.02137824147939682,  0.6854990720748901 ,
   0.13068442046642303,  0.06389980018138885,  0.3200559616088867 ,
   0.2416321486234665 , -0.9003387689590454 ,  0.10308495908975601,
   0.1375327706336975 ,  0.3300655782222748 ,  0.39635562896728516,
   0.3634832203388214 ,  0.1353854238986969 ,  0.01864137500524521,
   0.02177563682198524, -0.485281765460968  , -0.07845979183912277,
   0.4351861774921417 , -0.17253251373767853,  0.03973834216594696,
  -0.14903461933135986, -1.103270411491394  , -0.0807526707649231 ,
  -2.7770800590515137 , -0.10454905033111572, -0.00116272631566972,
   0.26457324624061584, -0.3980792760848999 ],
 [ 0.16376903653144836,  0.4067072570323944 ,  0.19397278130054474,
   0.35962438583374023,  0.09699966758489609,  0.6785345077514648 ,
  -0.03966057673096657,  0.1639479249715805 , -0.35953670740127563,
   0.31445634365081787,  0.34899330139160156, -0.15571220219135284,
   0.11905918270349503, -0.33913716673851013, -0.48525646328926086,
   0.13138844072818756,  0.780341625213623  ,  0.6320037841796875 ,
   0.00712002860382199,  0.06458639353513718, -1.8892686367034912 ,
  -0.33284178376197815, -1.1425951719284058 , -1.1661992073059082 ,
  -0.7812477350234985 , -0.0359480045735836 , -0.9978324174880981 ,
   0.1991177350282669 , -0.8673542141914368 ,  0.8949446678161621 ,
  -0.0502169206738472 ,  0.37214329838752747, -0.6439805626869202 ,
   0.01063348725438118,  0.30423328280448914, -0.49817341566085815,
   0.7017055153846741 , -0.0188821405172348 , -0.9749928712844849 ,
  -0.6045451760292053 ,  0.7740886807441711 ,  0.9169877767562866 ,
  -0.05135532468557358, -0.2999385595321655 , -0.8997439742088318 ,
  -0.817608654499054  , -0.3294861912727356 , -1.2088154554367065 ,
   0.33794108033180237, -0.5224946141242981 ,  0.46200814843177795,
   0.5138714909553528 , -0.48199522495269775, -0.01527481526136398,
   0.4459368884563446 , -0.2146647870540619 , -1.345598816871643  ,
  -0.7787730693817139 , -0.43742144107818604, -0.5161536931991577 ,
  -0.5303660035133362 ,  0.02488549612462521, -0.9132068753242493 ,
   0.1394849568605423 ,  0.27832454442977905, -0.6204831600189209 ,
   1.0174142122268677 ,  0.7300640344619751 ,  0.10091128200292587,
  -1.1567645072937012 , -1.1148768663406372 ,  0.5780202746391296 ,
  -0.388616681098938  , -0.22522443532943726,  0.16709409654140472,
   0.47463974356651306, -1.546093225479126  , -1.6642343997955322 ,
   0.5555537939071655 , -0.7547754049301147 ,  0.6013663411140442 ,
   0.6859194040298462 , -0.7410492897033691 , -0.6603137850761414 ,
   0.01585031487047672, -0.9177840948104858 , -0.30338984727859497,
  -0.545589804649353  ,  0.03396685421466827, -0.31070131063461304,
  -0.3450576364994049 , -0.24267370998859406,  0.1735963225364685 ,
   0.2757333815097809 , -0.24049951136112213,  0.7885205745697021 ,
   0.131754532456398  , -0.19627511501312256, -0.5624659657478333 ,
  -0.02823281288146973, -0.44488534331321716, -1.3147295713424683 ,
  -0.0879695862531662 ,  0.4885002076625824 ,  0.2753559350967407 ,
  -2.055241346359253  ,  0.7490403056144714 ,  1.041372537612915  ,
  -1.1028459072113037 ,  0.4344036877155304 ,  0.29965946078300476,
   0.24086244404315948, -0.6105542182922363 , -0.1708901822566986 ,
  -0.04922854155302048, -0.6357661485671997 , -0.07570647448301315,
  -0.6872156262397766 , -0.16087974607944489,  0.14621469378471375,
   0.6057370901107788 ,  0.5895832180976868 ,  1.5170754194259644 ,
  -0.12290561944246292, -0.4766993820667267 ,  0.2634177505970001 ,
   0.01142988167703152, -2.2713353633880615 ],
 [ 0.17199991643428802, -1.2266085147857666 ,  0.00940660294145346,
   0.2699841260910034 , -0.1323351263999939 , -0.2523173987865448 ,
  -0.8497046828269958 ,  0.06379753351211548, -0.385078489780426  ,
  -0.13600248098373413, -0.18069176375865936,  0.00228859484195709,
  -0.20136316120624542, -1.117637276649475  ,  0.3631262481212616 ,
   0.06362581998109818, -0.3039092719554901 ,  0.00524915056303144,
   0.7402340769767761 , -0.30921489000320435,  0.6202083230018616 ,
  -0.00279372814111412,  1.4839779138565063 , -0.6974652409553528 ,
   0.6683147549629211 ,  0.6596625447273254 ,  0.57706618309021   ,
  -0.48311638832092285,  0.3266911506652832 ,  0.08296017348766327,
  -0.02538576535880566,  0.07942323386669159,  1.4351853132247925 ,
  -0.2796175181865692 ,  0.01516586728394032,  0.7280588150024414 ,
  -0.05429554730653763, -0.00973103661090136, -0.20412932336330414,
   0.1863642781972885 , -1.8837600946426392 , -3.0052404403686523 ,
  -0.4555358588695526 , -0.00614535575732589,  0.44216057658195496,
   0.8920472860336304 , -1.6641807556152344 ,  0.3719916045665741 ,
   0.01214119698852301, -0.13218384981155396, -0.2182655930519104 ,
  -0.31749874353408813, -0.15049678087234497, -0.07574199885129929,
  -0.35040774941444397,  0.18507465720176697, -1.6825366020202637 ,
   0.04513281583786011, -0.02156064659357071,  0.5441078543663025 ,
   0.08074752986431122, -0.6435112357139587 ,  1.6294625997543335 ,
  -0.3471592664718628 , -0.16605646908283234,  0.24200217425823212,
  -2.070035696029663  , -0.3134060204029083 , -0.8533960580825806 ,
   0.03573236614465714,  0.18143799901008606,  0.0112814512103796 ,
  -0.16699495911598206, -0.40352603793144226, -0.31364306807518005,
  -1.2509831190109253 , -1.5765661001205444 ,  0.37336891889572144,
  -1.0790117979049683 , -0.18580980598926544, -0.5531423091888428 ,
  -0.2852827310562134 ,  0.16975535452365875, -0.11503491550683975,
   0.06761391460895538,  0.2047962248325348 , -0.15026140213012695,
  -1.8183103799819946 , -0.04222200810909271, -0.1357048898935318 ,
  -1.1476900577545166 ,  0.14062154293060303, -1.0296255350112915 ,
   0.00823925249278545,  0.20634488761425018,  0.16720153391361237,
   0.25554993748664856, -0.43250665068626404,  0.26391416788101196,
   0.06438655406236649, -0.2364618480205536 , -0.43995240330696106,
  -0.0124358581379056 , -0.29098308086395264, -0.00845170393586159,
  -0.7588474750518799 , -0.33884531259536743, -1.824257254600525  ,
   0.7582647800445557 , -0.21733754873275757, -0.1343051791191101 ,
  -0.4585966467857361 ,  0.4809863567352295 , -0.01866144873201847,
   0.09976622462272644, -0.10618700087070465, -0.07845747470855713,
   0.272421658039093  ,  0.3949403762817383 , -0.09013980627059937,
   0.02633360587060452,  0.05134880542755127, -0.580164909362793  ,
   0.00484704598784447, -0.530938982963562  , -0.00339841633103788,
  -0.14494842290878296,  0.067341648042202  ],
 [ 0.5639392137527466 , -0.03801648691296577, -0.2182440161705017 ,
   0.30403760075569153,  0.7755406498908997 , -0.20951293408870697,
  -0.6148366332054138 ,  0.3442015051841736 , -0.3648472726345062 ,
   0.23544207215309143,  0.9682113528251648 ,  0.10829761624336243,
   0.22992439568042755, -1.3656997680664062 ,  0.17334291338920593,
   0.4216103255748749 ,  0.24278773367404938,  0.0559055395424366 ,
  -0.10780265927314758,  0.444949209690094  , -0.3522682189941406 ,
   0.19867181777954102, -0.0277419202029705 , -0.4410906136035919 ,
  -1.4177979230880737 , -1.090131163597107  , -1.3954157829284668 ,
  -1.088183879852295  ,  0.11691077053546906,  0.5617254376411438 ,
   0.09229078888893127,  0.16796420514583588,  0.6370924711227417 ,
   0.35229483246803284,  0.24187332391738892, -1.2319680452346802 ,
  -0.8810017704963684 ,  0.1657458245754242 , -0.32836997509002686,
  -0.326742023229599  , -0.5019426345825195 , -1.615633249282837  ,
  -0.5053985118865967 ,  0.29799383878707886, -0.04077311977744102,
   0.3869076669216156 , -0.9904554486274719 ,  0.05194719508290291,
   0.37054193019866943, -0.6201108694076538 , -0.04614977538585663,
  -0.41496169567108154, -0.3420916199684143 ,  0.17044979333877563,
   0.6374191641807556 ,  0.03478056192398071,  0.5335744619369507 ,
   0.0724392905831337 , -0.1273832768201828 ,  1.73273503780365   ,
  -0.47477054595947266, -0.43562600016593933, -0.10739237815141678,
   1.3901654481887817 , -0.07243426144123077, -0.16137830913066864,
  -0.5186394453048706 , -0.1562318205833435 , -0.9324301481246948 ,
  -0.13692982494831085, -0.6946926116943359 , -0.8325076699256897 ,
  -0.3750547170639038 , -0.7970762848854065 , -0.3063848316669464 ,
   0.16112101078033447, -0.27988409996032715, -0.9488856792449951 ,
  -0.17289507389068604, -0.5789569616317749 ,  0.08215489238500595,
  -1.3381956815719604 , -0.08733943849802017, -1.8319916725158691 ,
   0.14477479457855225,  0.24766865372657776,  0.2717786729335785 ,
   0.23552553355693817, -0.6404869556427002 , -0.00389484572224319,
  -1.390217900276184  , -0.4971776306629181 , -0.24379442632198334,
   0.591917097568512  , -0.6788824200630188 , -0.937454879283905  ,
  -0.8052987456321716 ,  1.154581904411316  , -0.4221747815608978 ,
   0.00222820369526744, -0.3824441134929657 , -0.9153833389282227 ,
  -0.08897007256746292, -0.6505580544471741 , -0.11406172066926956,
  -0.07824814319610596, -0.42251336574554443,  0.4294149875640869 ,
  -0.29624783992767334,  0.5708132386207581 ,  0.2386779934167862 ,
   0.8947566747665405 ,  0.21545200049877167, -1.7043694257736206 ,
  -0.6459561586380005 ,  0.27904823422431946,  0.5068295001983643 ,
   0.6335494518280029 , -1.016903281211853  , -0.554233193397522  ,
  -1.704300045967102  ,  0.07311131805181503, -1.1523621082305908 ,
  -0.7836427688598633 , -0.05125168710947037,  0.6830430626869202 ,
  -0.07317554950714111,  0.37184271216392517],
 [ 0.02216915600001812,  1.9803177118301392 , -0.19675317406654358,
   0.6625530123710632 ,  0.5308767557144165 ,  0.06721015274524689,
  -0.779133677482605  , -0.15566641092300415,  0.6369799375534058 ,
  -0.2616875171661377 , -0.929835855960846  ,  0.13384582102298737,
  -0.1251147985458374 ,  1.1465506553649902 , -0.02367629297077656,
   0.2249348759651184 , -0.13268744945526123, -0.4904908239841461 ,
   0.05352787673473358,  0.13692380487918854, -0.412590354681015  ,
  -0.5891135334968567 ,  0.685260534286499  ,  0.9113097190856934 ,
   0.15221957862377167,  1.0386171340942383 , -0.49645566940307617,
   0.6790772080421448 , -1.0526673793792725 , -1.2848005294799805 ,
  -0.07824160903692245, -1.413367748260498  ,  0.3998134136199951 ,
  -0.21562893688678741,  0.3725964426994324 , -1.114725112915039  ,
   0.764153778553009  , -0.05684786289930344, -0.3396686017513275 ,
   0.530601441860199  ,  0.9002594351768494 ,  0.00120682106353343,
   0.40002402663230896,  0.9062705636024475 ,  0.06583529710769653,
   0.1396915763616562 ,  1.1319305896759033 , -0.9983141422271729 ,
  -0.5920670628547668 , -0.00188294355757535,  0.0486128106713295 ,
  -0.1653393656015396 , -0.5727623701095581 , -0.49526458978652954,
   1.1666451692581177 , -1.1639081239700317 , -0.2408212125301361 ,
   0.3143813908100128 , -0.11442239582538605, -0.7722563743591309 ,
  -0.07844863831996918, -0.01903435029089451,  0.4478308856487274 ,
  -0.24607093632221222,  0.15233686566352844,  1.680213212966919  ,
   0.6791029572486877 , -1.3308054208755493 ,  0.21006157994270325,
   0.22271011769771576,  0.12247996032238007, -1.4696887731552124 ,
   0.4959951639175415 ,  0.8268843293190002 ,  0.23699797689914703,
  -0.26816052198410034, -0.11434479802846909, -0.6062285304069519 ,
   0.7093798518180847 , -0.33168378472328186, -0.2229786217212677 ,
   0.7174219489097595 ,  0.34037554264068604,  0.7465134859085083 ,
  -0.10466298460960388,  0.739228367805481  , -0.43160873651504517,
  -1.0060358047485352 ,  0.09297254681587219, -0.17646753787994385,
   1.4756282567977905 , -0.543459415435791  , -0.5780095458030701 ,
  -1.0240181684494019 , -0.9559988379478455 , -1.5818498134613037 ,
   0.8402873873710632 , -0.11209841817617416, -0.16264241933822632,
  -0.15030023455619812, -0.1134677603840828 , -1.1501953601837158 ,
  -0.1378893256187439 ,  0.17223286628723145, -0.7412548065185547 ,
  -0.2643742561340332 , -0.5879210233688354 , -1.2164833545684814 ,
  -0.4961971044540405 , -0.21814166009426117,  0.49051839113235474,
  -0.6677882075309753 ,  1.4088799953460693 ,  0.5525991320610046 ,
  -0.08524392545223236,  1.1995465755462646 , -0.41597118973731995,
  -1.6066417694091797 ,  0.14245019853115082,  0.47045817971229553,
   0.28533223271369934,  0.4280855357646942 ,  0.317984938621521  ,
   0.20270821452140808,  0.07633142918348312, -0.16638922691345215,
   0.158311128616333  ,  1.0880014896392822 ],
 [ 0.01187693793326616,  0.22746409475803375,  0.03572497889399529,
  -1.0206847190856934 , -2.1187448501586914 , -0.4906470775604248 ,
  -0.14265874028205872, -0.03304633125662804,  0.07451264560222626,
   0.44939473271369934, -0.01899636909365654, -0.40453648567199707,
  -0.12284619361162186,  0.4745778441429138 , -0.19035343825817108,
  -0.15062114596366882, -1.0867646932601929 ,  0.2787497043609619 ,
  -0.8464035391807556 , -0.3627898395061493 ,  0.3413759469985962 ,
   0.4190670847892761 , -0.5041411519050598 ,  1.2178305387496948 ,
   0.7186317443847656 , -1.0501933097839355 , -0.19272807240486145,
   0.7200959324836731 , -0.098129503428936  ,  0.4354032874107361 ,
  -0.19797581434249878, -0.4924700856208801 ,  0.7500805854797363 ,
   0.6374630331993103 ,  0.11528919637203217,  0.6863013505935669 ,
  -2.5736021995544434 , -0.05573185160756111, -0.36108383536338806,
  -0.5317057371139526 , -0.9287326335906982 ,  0.20596161484718323,
   0.3612500727176666 ,  0.7982490062713623 ,  0.18403038382530212,
  -0.10359994322061539, -0.8537354469299316 ,  0.01504788082093   ,
  -0.7695412039756775 , -1.2809580564498901 , -1.7000281810760498 ,
  -1.0179407596588135 ,  0.186207115650177  ,  0.7855934500694275 ,
   0.14521408081054688, -0.7169716358184814 , -0.4937509000301361 ,
   0.15936993062496185, -1.1572518348693848 , -0.15500137209892273,
   0.06274684518575668, -0.25171929597854614,  0.06979109346866608,
   0.46627530455589294, -3.2528254985809326 , -0.4459390342235565 ,
  -0.29738718271255493, -0.187534898519516  , -1.2106728553771973 ,
   0.7211036086082458 ,  0.08812128007411957,  0.00046700879465789,
   0.09006895869970322, -0.3535800278186798 ,  0.7555809020996094 ,
   0.00583585537970066,  0.166362002491951  , -1.6355024576187134 ,
  -0.40861302614212036,  0.7734125852584839 ,  0.12759950757026672,
  -0.8361297249794006 , -0.1411084532737732 ,  0.10190960019826889,
   0.03666117787361145,  0.5314419269561768 ,  0.3425252437591553 ,
   0.39224809408187866, -0.45109641551971436, -0.5322364568710327 ,
  -1.2266998291015625 , -0.15859481692314148,  0.05037307366728783,
  -0.7008204460144043 , -0.04242182523012161, -0.02948224544525146,
  -0.8473795056343079 , -0.2602517902851105 , -1.6805890798568726 ,
   0.005635482724756  , -0.29636791348457336,  0.7075960636138916 ,
  -0.0793527364730835 , -0.9229406118392944 , -0.681107759475708  ,
   0.622377336025238  , -0.9212428331375122 ,  0.498273640871048  ,
  -0.9321374297142029 , -0.1651000678539276 ,  0.753635823726654  ,
  -0.45505762100219727,  1.5119622945785522 ,  0.21645934879779816,
  -0.2098999172449112 , -1.5405791997909546 , -0.18709267675876617,
  -0.06243897974491119, -0.5020841360092163 ,  0.2968241572380066 ,
  -0.1792617291212082 ,  0.5091453194618225 ,  0.08533480018377304,
   0.09028124809265137, -0.574696958065033  ,  0.15647578239440918,
   0.5181113481521606 , -0.338556170463562  ],
 [ 0.5861279368400574 ,  0.254854679107666  ,  0.18012414872646332,
   0.40985727310180664, -0.30174481868743896, -1.5881199836730957 ,
  -0.27123966813087463,  0.15930868685245514, -0.23983632028102875,
  -0.07361627370119095, -0.9318119287490845 , -0.18722404539585114,
  -0.23995864391326904, -0.2900593876838684 , -0.04988064244389534,
   0.06378121674060822,  0.7079715728759766 ,  0.5302433371543884 ,
   0.21594150364398956, -0.842243492603302  , -0.1969480812549591 ,
  -0.08104301244020462, -0.6610599756240845 , -0.740010142326355  ,
  -0.6484160423278809 ,  0.6059221625328064 ,  0.6663183569908142 ,
   0.05523521453142166, -0.19439780712127686, -0.07956428080797195,
  -0.34451037645339966, -0.2570461630821228 , -1.0238925218582153 ,
   0.05915766954421997,  0.1011834368109703 ,  0.06641900539398193,
  -0.5803244709968567 ,  0.21819055080413818,  0.30659300088882446,
   0.05713725090026855, -0.12459623068571091,  0.16404522955417633,
  -0.36110058426856995,  0.1562604457139969 , -0.6659978628158569 ,
   0.10489393025636673,  0.10754236578941345, -0.5836867690086365 ,
  -0.6488041877746582 , -0.7839337587356567 ,  0.5700744986534119 ,
  -0.6166589260101318 ,  0.6081957221031189 , -0.20125988125801086,
  -0.48304513096809387,  0.27482402324676514,  0.5620254278182983 ,
   0.09614711254835129, -0.39647507667541504,  0.1824570894241333 ,
  -0.587816596031189  , -0.6115572452545166 , -0.68194180727005   ,
   0.41472071409225464, -0.16780662536621094, -0.2836310863494873 ,
  -0.3624474108219147 ,  0.4398430287837982 , -2.240885019302368  ,
  -1.2273560762405396 , -0.33228427171707153, -0.07378538697957993,
   0.24492089450359344,  0.19405154883861542,  0.8452337980270386 ,
  -0.09936100989580154,  0.44987449049949646,  0.3186369836330414 ,
  -0.773983359336853  ,  0.13289028406143188, -0.5755377411842346 ,
  -2.172268867492676  , -0.0930730402469635 , -0.40894678235054016,
  -0.16180028021335602,  0.5273727178573608 , -0.9578202962875366 ,
  -0.02084881626069546,  0.1490320861339569 , -0.14439764618873596,
  -0.1589762419462204 ,  0.37644845247268677,  0.21167802810668945,
   0.22601571679115295, -0.7492520213127136 ,  0.4926449656486511 ,
  -1.507010817527771  , -1.4536452293395996 ,  0.2811814844608307 ,
  -0.38766464591026306, -0.23563970625400543,  0.5594906806945801 ,
   0.02001065760850906, -1.4990826845169067 , -0.1203097477555275 ,
  -1.5915038585662842 , -0.8718969821929932 , -1.0889427661895752 ,
   0.47081658244132996, -0.01770496927201748, -0.8416900634765625 ,
  -0.01127273123711348, -0.0828809067606926 , -0.11252616345882416,
  -0.03565709292888641,  0.6718155741691589 , -0.40340283513069153,
   0.03770056366920471,  0.05258412286639214,  0.1724003255367279 ,
   0.1789158433675766 , -0.3945949077606201 , -0.00360667589120567,
   0.0009347340092063 , -0.283192902803421  ,  0.2753762900829315 ,
   0.140798881649971  , -1.2633922100067139 ],
 [-1.1234667301177979 ,  0.3276270925998688 , -0.19402948021888733,
  -0.7213897109031677 , -0.999416172504425  , -0.13445818424224854,
  -0.36061665415763855, -0.9311866760253906 , -0.9394177198410034 ,
  -0.7570432424545288 , -0.5074625611305237 , -0.5565590262413025 ,
  -1.4215872287750244 ,  0.06025455892086029, -0.4318888187408447 ,
   0.7204339504241943 , -0.35595086216926575, -0.22587518393993378,
  -0.38061049580574036, -0.03663744777441025,  0.44606146216392517,
   0.5627568364143372 , -0.16167286038398743,  0.07676277309656143,
  -1.0940775871276855 , -0.5140513777732849 , -0.8319116234779358 ,
  -0.995902955532074  , -0.7904715538024902 , -1.1382787227630615 ,
  -1.068596363067627  , -0.8478962182998657 , -0.0429941825568676 ,
  -0.23587816953659058, -1.3119544982910156 , -0.5067336559295654 ,
   0.38702765107154846,  0.7645593881607056 ,  0.10660452395677567,
  -0.2835023105144501 , -0.27730023860931396,  0.04606685414910316,
  -0.4165448248386383 ,  0.24558883905410767, -0.25750845670700073,
  -0.06311759352684021,  0.22517290711402893, -0.2982318103313446 ,
  -0.33795952796936035, -1.4756951332092285 , -1.2896485328674316 ,
  -0.1359424889087677 ,  0.3074270784854889 , -1.3099584579467773 ,
  -0.2863001823425293 , -0.35402631759643555,  0.04073766618967056,
  -1.2513327598571777 , -0.34161439538002014, -1.9215677976608276 ,
  -1.516741394996643  , -0.3051794469356537 , -0.13132980465888977,
  -1.1398521661758423 , -0.39892736077308655, -0.19465972483158112,
  -0.3301033675670624 , -0.1811387538909912 ,  0.09861712902784348,
  -0.2423134297132492 , -0.6376909613609314 , -0.8616203665733337 ,
  -0.8678312301635742 , -0.49448615312576294, -1.9760584831237793 ,
   0.8107619285583496 , -0.3699451684951782 , -0.08291584253311157,
   0.20758651196956635, -0.5553905963897705 , -0.6536486148834229 ,
  -0.292990505695343  , -0.22304852306842804, -0.04533755406737328,
  -0.5870092511177063 , -0.7431055903434753 , -1.0108741521835327 ,
  -0.4517136216163635 , -0.22517824172973633, -2.8258204460144043 ,
   0.40669354796409607, -0.22850753366947174, -0.08032716810703278,
  -0.8899314999580383 , -0.4346059262752533 , -0.43545880913734436,
  -1.1866501569747925 , -0.07896512746810913, -0.7702335715293884 ,
  -2.0694968700408936 , -0.2333008050918579 , -0.7435064911842346 ,
  -0.670976996421814  , -0.31184884905815125, -0.02516983635723591,
  -0.9603076577186584 , -0.7066170573234558 , -0.39742937684059143,
  -0.02258390001952648,  0.3343431353569031 , -0.8627187013626099 ,
   0.33123257756233215, -0.16339267790317535,  0.22888681292533875,
  -0.6098964214324951 , -0.4716019332408905 , -1.2984142303466797 ,
  -0.23970013856887817, -0.381715327501297  , -1.4617109298706055 ,
   0.3182133734226227 , -1.1572455167770386 , -0.7892855405807495 ,
  -0.9799851179122925 ,  0.3977041244506836 , -0.3644784688949585 ,
  -0.8807001113891602 , -0.1543848067522049 ],
 [-1.082494854927063  ,  0.11734876781702042,  0.8825708031654358 ,
  -0.5151112079620361 , -1.6424263715744019 , -0.6744120717048645 ,
  -0.24070331454277039, -0.9957436323165894 , -1.1191658973693848 ,
  -1.1184996366500854 , -1.322030782699585  ,  0.4881516695022583 ,
   0.06884225457906723, -0.0640769749879837 , -0.11218979954719543,
  -0.2789161205291748 , -0.08438195288181305, -1.077175259590149  ,
  -1.6720727682113647 ,  0.14398007094860077,  0.32242509722709656,
  -0.9253357648849487 ,  0.3532365560531616 , -2.1115329265594482 ,
  -0.2623307704925537 , -0.576472282409668  , -0.41807493567466736,
  -0.41751816868782043, -0.35548195242881775, -0.16324883699417114,
   0.7350617051124573 , -0.537505567073822  ,  0.11042167246341705,
   0.09897451847791672, -0.8275029063224792 , -0.8408141136169434 ,
  -0.40579405426979065, -0.4989584982395172 , -0.2303469181060791 ,
   0.7300595641136169 ,  0.0340799055993557 ,  0.6046996712684631 ,
   0.07093778252601624,  0.7331669926643372 , -1.836323618888855  ,
   0.42332330346107483, -0.4855130612850189 , -0.9299749135971069 ,
  -0.05457896739244461, -1.1427375078201294 , -0.27344000339508057,
  -0.8953609466552734 ,  0.1707872897386551 ,  0.11397797614336014,
  -0.592820405960083  , -0.21412333846092224,  0.2217269390821457 ,
   0.46839144825935364,  0.5427601933479309 , -1.1338814496994019 ,
  -0.03689554706215858, -2.3666088581085205 , -1.112532377243042  ,
  -0.14662623405456543,  0.663966953754425  ,  0.23127518594264984,
  -0.11057716608047485, -0.6599861979484558 ,  0.3552609384059906 ,
  -1.0611064434051514 ,  0.25997233390808105, -1.0862315893173218 ,
  -0.6380453705787659 , -0.21944890916347504, -0.26929834485054016,
  -0.0661143884062767 , -0.6578153371810913 ,  0.05404432862997055,
  -0.8596853613853455 , -0.657051682472229  , -0.46021923422813416,
  -0.9992544054985046 ,  0.7050452828407288 , -0.19738076627254486,
  -0.8668383359909058 ,  0.5248762965202332 , -1.1854302883148193 ,
  -0.10328315943479538, -0.45965704321861267,  0.6501128077507019 ,
  -0.8735519647598267 , -0.01985683664679527,  0.30787381529808044,
   0.00677158683538437, -0.01800655759871006, -0.4150397479534149 ,
  -0.1827365905046463 , -0.1488608717918396 , -0.6481894254684448 ,
   0.7215951681137085 ,  0.01350918784737587,  0.14782413840293884,
  -0.11157587170600891, -0.29174935817718506, -0.8548965454101562 ,
   0.20653554797172546, -0.18706731498241425,  0.02926846221089363,
   0.18633615970611572, -0.32999709248542786,  0.11049266904592514,
  -0.3953648507595062 , -0.5794110298156738 , -0.08238743990659714,
  -0.24337641894817352,  0.2564046084880829 , -0.0591132640838623 ,
   0.2371203601360321 , -1.1359483003616333 , -0.27334654331207275,
   0.01637818105518818, -0.1783054918050766 ,  0.01841054856777191,
  -0.5651720762252808 , -0.4118788242340088 ,  0.2469872236251831 ,
  -0.3662358820438385 , -0.9701192378997803 ],
 [-0.2086179256439209 ,  0.48456403613090515, -0.20097851753234863,
  -0.2175808846950531 ,  0.01553869154304266,  0.12538352608680725,
  -1.203820824623108  ,  0.2204395979642868 ,  0.7313459515571594 ,
   0.29648399353027344,  0.3495312035083771 , -0.04557307809591293,
  -0.02344591170549393,  1.164174199104309  , -0.0167468786239624 ,
   0.24648039042949677, -1.1783257722854614 , -0.39224421977996826,
  -0.8394190073013306 ,  0.8327386379241943 ,  0.2642219066619873 ,
   0.40896931290626526, -1.058068871498108  ,  0.42817336320877075,
   0.10429443418979645,  0.6381480693817139 ,  0.5573019981384277 ,
  -1.2011255025863647 , -0.9707706570625305 ,  0.3059401512145996 ,
  -0.21100881695747375, -0.30743393301963806, -1.6714162826538086 ,
  -0.06127998232841492,  0.6986528038978577 ,  0.23860250413417816,
   0.1990891844034195 ,  0.02817987464368343, -0.7469240427017212 ,
  -0.23687691986560822,  0.32574447989463806, -0.11518334597349167,
  -0.7388735413551331 ,  0.29281023144721985, -1.5870776176452637 ,
  -0.3252826929092407 ,  2.7853429317474365 , -0.31594088673591614,
  -0.3446035385131836 , -0.5705127716064453 , -1.114058256149292  ,
  -0.2365872710943222 , -1.274464726448059  , -0.0661865845322609 ,
   0.553848147392273  , -1.3126879930496216 , -0.26789039373397827,
  -0.16969721019268036,  0.2375767081975937 , -0.1059064120054245 ,
  -0.43958985805511475, -0.5294869542121887 , -0.1532156616449356 ,
  -0.24006205797195435,  0.32066571712493896, -0.8738228678703308 ,
   0.4928162395954132 , -0.845497190952301  , -0.07929021865129471,
   0.6867508292198181 , -0.9668627977371216 , -0.7515648007392883 ,
   0.19711031019687653, -0.11584767699241638, -0.6547220945358276 ,
  -0.44588786363601685, -0.6917684674263    , -0.2843267023563385 ,
  -1.1764862537384033 ,  0.3270002007484436 ,  0.3356415629386902 ,
  -1.3832721710205078 , -0.01420108042657375, -0.6292322874069214 ,
   0.1706811636686325 , -1.4846343994140625 , -0.49358728528022766,
  -0.5959677696228027 ,  0.19437792897224426, -0.2097787708044052 ,
   2.314094066619873  ,  0.9963914155960083 ,  0.3776476979255676 ,
  -0.4752424359321594 , -0.536940336227417  ,  0.13460570573806763,
  -0.8617125153541565 , -0.8159672021865845 , -0.954340934753418  ,
  -0.17084862291812897,  0.6025253534317017 , -1.0419795513153076 ,
   0.12521888315677643,  0.21041734516620636, -0.15413253009319305,
   0.2825928032398224 , -0.14791874587535858,  0.03671162202954292,
   0.5530636310577393 ,  0.5399636030197144 , -0.09940066188573837,
  -0.8318170309066772 , -1.7725136280059814 , -2.683605670928955  ,
   0.0395183265209198 ,  0.00034255790524185, -0.5274019837379456 ,
  -0.8526001572608948 ,  0.6381215453147888 ,  0.03866345062851906,
  -0.45487335324287415, -1.1343505382537842 , -0.16914846003055573,
  -0.11934055387973785,  0.2757588326931    , -0.1244262233376503 ,
  -0.30624693632125854, -2.2752106189727783 ],
 [-0.8962900042533875 ,  0.3470202386379242 ,  0.08935195952653885,
  -0.20201486349105835,  0.12555959820747375,  0.3733474612236023 ,
   0.1110856905579567 , -0.8361021876335144 , -0.01631101593375206,
  -0.40094754099845886, -0.09907900542020798, -0.5755380988121033 ,
   0.581806480884552  ,  0.33610817790031433,  0.29634392261505127,
  -0.24017950892448425, -0.02505187876522541, -0.5475113391876221 ,
  -0.10790355503559113,  0.19291919469833374,  0.2536733150482178 ,
   0.02006338536739349,  0.5466151833534241 , -0.23992933332920074,
   0.3789769411087036 ,  0.10511517524719238,  0.05963469296693802,
   0.9944354891777039 , -0.24800844490528107, -0.3673880100250244 ,
   0.33818286657333374, -0.18239200115203857,  0.2381151020526886 ,
  -0.11696420609951019, -1.3892014026641846 , -1.309725046157837  ,
   0.21316498517990112, -0.10942273586988449,  0.5880407094955444 ,
  -0.04009296000003815,  0.19043824076652527,  0.3471875488758087 ,
   0.876236617565155  ,  0.9898335337638855 , -0.7455543279647827 ,
   0.6403639316558838 ,  0.16841106116771698, -0.7223661541938782 ,
   0.7762238383293152 , -0.22604350745677948, -0.48662084341049194,
  -0.9249927997589111 ,  0.2351936399936676 ,  0.36978375911712646,
  -1.3295124769210815 , -0.02053563296794891,  0.15197977423667908,
  -0.03610491007566452,  0.13411220908164978,  0.1459680050611496 ,
   0.5100374817848206 ,  0.62558513879776   ,  0.07820198684930801,
   0.33361518383026123,  0.615149736404419  ,  1.0557053089141846 ,
  -0.20860227942466736, -0.32525017857551575,  0.3486931324005127 ,
   0.02928232401609421,  0.01320511568337679, -0.2651354968547821 ,
   0.6286534667015076 , -0.35871800780296326,  0.767127513885498  ,
   0.17036105692386627,  0.09785415977239609,  0.17925304174423218,
  -0.1262069046497345 , -0.2491305023431778 ,  0.14528919756412506,
   0.1240457072854042 ,  0.2284923493862152 ,  0.5612534880638123 ,
   0.19653485715389252,  0.7646843194961548 ,  0.1981884390115738 ,
  -0.4057127833366394 ,  0.12384224683046341,  0.20098425447940826,
  -0.3181126117706299 , -0.5796180963516235 , -0.16448260843753815,
  -1.1085199117660522 ,  0.10140542685985565,  0.9963420629501343 ,
   0.04209134727716446, -0.5772177577018738 , -0.804328441619873  ,
  -0.38919129967689514, -0.5100454688072205 ,  0.02983853220939636,
  -0.8856299519538879 , -0.12091477960348129,  0.12038439512252808,
   0.10346481204032898,  0.3915978968143463 ,  0.690981924533844  ,
   0.9562993049621582 ,  0.0632941797375679 , -1.4897246360778809 ,
   0.35269004106521606, -0.3678758442401886 , -0.1778198778629303 ,
   0.682391345500946  ,  0.6910326480865479 ,  0.3733210563659668 ,
  -0.6304872632026672 , -1.3876336812973022 ,  0.3244546353816986 ,
   0.17773061990737915,  0.15597376227378845,  0.173021137714386  ,
   0.2861434817314148 ,  0.16280680894851685, -0.9651243686676025 ,
  -0.4595141112804413 ,  0.23535294830799103],
 [ 0.03542333468794823, -1.1095616817474365 ,  0.06763989478349686,
  -0.1839359700679779 ,  0.5296386480331421 ,  0.20897804200649261,
  -0.6431230902671814 , -0.14493617415428162, -1.0835363864898682 ,
   0.08153391629457474, -0.08143038302659988,  0.05612876266241074,
   0.1456729918718338 , -0.7006916403770447 , -0.16414885222911835,
  -0.05249299854040146,  0.2886171340942383 ,  0.0398586131632328 ,
   0.2565573453903198 , -1.286464810371399  ,  0.06294075399637222,
  -0.03869810700416565,  0.7525194883346558 , -1.7261756658554077 ,
  -0.24985043704509735,  0.10194595158100128, -0.46223458647727966,
   0.4203387200832367 , -0.16305319964885712,  0.22258979082107544,
   0.03407581150531769, -0.2568941116333008 , -0.5727066397666931 ,
   0.1255922168493271 , -0.02164620719850063, -0.27867501974105835,
   0.2517952620983124 ,  0.02678760513663292, -0.15070457756519318,
  -0.03980719298124313, -2.0856125354766846 , -2.5530810356140137 ,
  -0.00645763846114278, -0.4019390642642975 , -0.5710733532905579 ,
  -0.2595391869544983 , -2.369776725769043  , -0.04552124068140984,
   0.1964336484670639 , -0.6668974161148071 , -0.1638711839914322 ,
   0.26600220799446106, -0.578831136226654  ,  0.08053047209978104,
  -1.318534016609192  , -0.6587485671043396 , -0.00971547700464725,
  -0.1777300387620926 , -0.2967914640903473 , -0.43133673071861267,
  -0.4714902639389038 ,  0.7039238810539246 ,  0.8846318125724792 ,
   0.18901005387306213, -0.2985300123691559 , -2.602876663208008  ,
  -1.5396453142166138 ,  0.7914267182350159 ,  0.1360282152891159 ,
  -1.1028401851654053 , -0.17164020240306854, -0.08036673069000244,
   0.06973862648010254,  0.10262210667133331, -0.06086374074220657,
   0.7171626091003418 , -0.5738242864608765 , -0.5160999298095703 ,
   0.34823256731033325, -0.1914232075214386 , -0.34702616930007935,
   0.4941079020500183 ,  0.2603447139263153 , -0.4760218858718872 ,
  -0.00567829422652721,  0.05648243427276611, -0.17402903735637665,
   0.04037869349122047,  0.05102433636784554, -0.06818478554487228,
  -1.091848373413086  ,  0.15112419426441193, -0.3566642105579376 ,
   0.15621238946914673,  0.04295824095606804, -0.09866922348737717,
   0.33871665596961975, -0.2871973216533661 , -0.24884729087352753,
   0.10811351984739304, -0.6494914293289185 ,  0.17783449590206146,
  -0.11412113159894943, -1.50827956199646   ,  0.20167464017868042,
  -0.06282368302345276, -0.2533601224422455 , -2.203364849090576  ,
  -0.03357525169849396, -0.11209326237440109,  0.12500610947608948,
  -0.3257511258125305 , -0.535078227519989  , -0.36277344822883606,
  -0.14803482592105865, -0.1568915992975235 ,  0.05098767578601837,
  -0.35347676277160645, -0.7509733438491821 , -0.08882801979780197,
  -0.43772462010383606,  0.12111873924732208,  0.40997380018234253,
   0.05757986381649971,  0.0917777344584465 ,  0.00518346764147282,
   0.10134927928447723,  0.195184126496315  ],
 [-0.4013018310070038 , -1.6883933544158936 , -0.09719251841306686,
  -0.18443404138088226, -0.25136956572532654, -1.2626432180404663 ,
  -2.0332021713256836 , -0.12854322791099548, -0.315125972032547  ,
   0.09798280894756317,  0.37176400423049927,  0.01096337102353573,
  -0.16663947701454163, -0.5938864350318909 , -0.7467803955078125 ,
   0.13474678993225098,  0.3997190296649933 ,  0.5227730870246887 ,
  -0.02018373645842075, -0.17410773038864136,  0.6055073142051697 ,
  -0.2349984496831894 , -0.2570500671863556 , -1.2656545639038086 ,
  -0.06476335972547531, -1.5579668283462524 , -0.5948787331581116 ,
  -0.9291649460792542 , -0.5877756476402283 , -0.15844982862472534,
   0.09649287909269333,  0.24567224085330963,  0.16083994507789612,
   0.102696493268013  , -0.24470922350883484, -0.6747518181800842 ,
   0.28956902027130127, -0.01955450139939785, -0.6515249013900757 ,
   0.23744750022888184, -1.2971980571746826 , -0.46282798051834106,
  -1.164404034614563  , -0.9739845991134644 , -0.5833368897438049 ,
   0.44116678833961487, -2.402987480163574  ,  0.6954053640365601 ,
  -0.6057494878768921 , -0.2999374270439148 , -0.2514118552207947 ,
  -0.20339302718639374, -0.4562089443206787 ,  0.1123410016298294 ,
  -0.3518046438694    , -0.24333354830741882,  0.18438473343849182,
   0.08759839832782745, -0.06164403259754181, -0.71162348985672   ,
  -0.16460436582565308, -0.5674023628234863 , -0.7838579416275024 ,
  -2.555405378341675  ,  0.03402027115225792,  0.0640786662697792 ,
  -1.534930944442749  , -0.4035142660140991 , -0.3171187937259674 ,
  -1.4430177211761475 ,  0.6404148936271667 , -0.3729715049266815 ,
  -0.01627011597156525, -0.5330561399459839 , -0.16580206155776978,
  -0.33246153593063354, -0.44776105880737305,  0.884463906288147  ,
  -0.36979448795318604, -0.3002500534057617 , -0.4859538674354553 ,
  -0.5310075283050537 ,  0.09831955283880234,  0.6526283025741577 ,
  -0.04372098296880722,  0.4451535940170288 , -0.7368555068969727 ,
  -0.04594260454177856, -0.25585296750068665, -0.04544621706008911,
  -1.8205100297927856 , -0.9221026301383972 , -0.6888202428817749 ,
   0.13931091129779816,  0.14210636913776398, -0.08673065900802612,
  -0.692096471786499  ,  0.40568462014198303,  0.01140417158603668,
  -0.08553515374660492,  0.22395959496498108,  0.3148355484008789 ,
   0.22148877382278442, -1.0301905870437622 ,  0.2227780520915985 ,
  -0.8805545568466187 , -0.24215374886989594, -1.5820108652114868 ,
  -0.8984562158584595 , -0.18113775551319122, -0.21170389652252197,
  -0.22608041763305664,  0.44176530838012695, -0.18154504895210266,
  -0.26069170236587524, -0.558211088180542  ,  0.23959767818450928,
   0.46005555987358093,  0.10293842107057571, -0.1232067197561264 ,
  -0.6309679746627808 , -0.13967208564281464, -1.5185986757278442 ,
  -0.41282087564468384,  0.09995075315237045, -0.158816859126091  ,
  -0.30343514680862427, -0.46559402346611023],
 [-1.8162940740585327 , -0.824427604675293  , -0.0778295248746872 ,
   0.32524698972702026, -1.7010459899902344 ,  1.1107596158981323 ,
   0.544061005115509  , -0.02013329789042473,  1.6532909870147705 ,
  -1.1557114124298096 , -0.11277920752763748,  0.28457605838775635,
  -0.8849497437477112 , -0.5835349559783936 ,  0.0146516552194953 ,
  -0.08554239571094513,  1.2232600450515747 ,  0.7267691493034363 ,
  -0.3705560564994812 , -0.8154972791671753 ,  0.8805603384971619 ,
   0.1292690485715866 ,  0.1624733954668045 , -0.46661603450775146,
   1.2230833768844604 ,  0.8853563070297241 ,  0.8655316233634949 ,
  -0.8482257127761841 ,  0.21076571941375732,  0.2519834339618683 ,
   0.4861752986907959 ,  1.4566127061843872 ,  1.9784315824508667 ,
  -0.9937164187431335 ,  0.09984384477138519,  2.140604019165039  ,
   0.7614706754684448 ,  0.2273600697517395 ,  0.5546953678131104 ,
  -0.17347292602062225, -0.5797074437141418 , -0.7774388790130615 ,
  -0.8238601088523865 , -0.6857275366783142 , -0.29679134488105774,
   1.0436629056930542 , -0.17504332959651947,  0.6238552927970886 ,
   0.30541929602622986,  0.8522777557373047 ,  1.2016137838363647 ,
   0.5457348823547363 , -0.8312607407569885 , -0.5483219027519226 ,
   0.2677619159221649 , -0.524483859539032  , -0.05056346580386162,
   0.4642890691757202 ,  0.31759506464004517,  1.0578656196594238 ,
   1.2928822040557861 , -0.41546839475631714, -0.7616584897041321 ,
  -0.9088684320449829 ,  0.8626636266708374 ,  3.9880645275115967 ,
   0.7729314565658569 ,  0.5160504579544067 ,  0.46511176228523254,
   0.7033081650733948 ,  0.22947454452514648,  0.27623221278190613,
  -0.03865135461091995,  0.673430860042572  ,  0.4310707449913025 ,
  -0.17709925770759583, -0.3128158450126648 ,  1.5250319242477417 ,
   0.585961639881134  , -0.04986515641212463, -0.6085204482078552 ,
   0.12586796283721924,  0.4464832544326782 ,  0.4545969069004059 ,
   0.10031137615442276,  0.8540408611297607 ,  0.19605574011802673,
  -1.0175598859786987 , -0.26626890897750854,  1.128963589668274  ,
  -0.28708645701408386, -0.42069655656814575, -0.02036122605204582,
   0.3013596832752228 ,  1.3201539516448975 , -0.15467298030853271,
   1.0886054039001465 , -1.3964946269989014 , -0.6050668358802795 ,
   0.37435123324394226,  0.29447856545448303, -0.34096238017082214,
   0.39358025789260864, -0.9177015423774719 , -0.09289639443159103,
   0.7461785078048706 ,  0.6388000845909119 , -0.4161721467971802 ,
   1.5260043144226074 ,  0.30012568831443787,  0.4549750089645386 ,
   0.9875070452690125 ,  0.9440025687217712 , -0.13417713344097137,
  -0.5623830556869507 , -0.8864961266517639 ,  1.0077168941497803 ,
   0.20602844655513763,  0.6243804097175598 , -0.12788555026054382,
   0.5994662046432495 ,  0.558795690536499  , -0.1376066654920578 ,
   0.3253806233406067 ,  0.34608858823776245,  0.7043685913085938 ,
  -0.05741634964942932,  0.6715924739837646 ],
 [ 0.8539233803749084 ,  0.07843206822872162,  0.09330829977989197,
   0.21492144465446472, -0.1683112233877182 ,  0.10692945122718811,
   0.03236028179526329,  0.12678967416286469, -0.3483971357345581 ,
   0.20795249938964844, -0.16263672709465027, -0.13577592372894287,
  -0.29325178265571594,  0.06440620869398117, -0.06289254128932953,
   0.02108784392476082, -0.2022915631532669 ,  0.3683459758758545 ,
   0.08157450705766678, -0.05992556735873222,  0.09022199362516403,
  -0.06433995813131332, -0.03230693936347961,  0.05395713448524475,
  -0.3429282009601593 , -0.12389693409204483,  0.2989181578159332 ,
   0.23269501328468323,  0.17714565992355347,  0.13052280247211456,
   0.11585713922977448,  0.3699924051761627 , -0.0214171651750803 ,
  -0.10160969942808151,  0.00274768867529929, -0.1242966279387474 ,
  -0.11533164232969284,  0.05618546530604362,  0.01957546174526215,
  -0.24628543853759766, -0.06712613999843597, -0.00138763547874987,
  -0.21922104060649872,  0.00272454111836851,  0.02105827815830708,
   0.02078686840832233, -0.0149637758731842 , -0.03238890320062637,
   0.06692658364772797, -0.3960285186767578 ,  0.8001128435134888 ,
   0.2178986370563507 , -0.09304598718881607,  0.2792520523071289 ,
   0.02479725517332554, -0.1151021271944046 , -0.03833751380443573,
   0.0054641580209136 , -0.08707556873559952,  0.07066358625888824,
   0.03383589908480644, -0.11034957319498062, -0.04621922224760056,
   0.09151032567024231, -0.00641915528103709,  0.0689762532711029 ,
   0.07442446798086166,  0.16780522465705872,  0.09414424747228622,
   0.07902191579341888,  0.04757620021700859, -0.3190757632255554 ,
   0.03311141207814217,  0.11409206688404083, -0.14449048042297363,
   0.04726098105311394,  0.34187954664230347, -0.00999819114804268,
   0.08548318594694138, -0.09063123911619186, -0.13871023058891296,
   0.01332968194037676,  0.06119177490472794, -0.02934691682457924,
  -0.09997954964637756,  0.01342930179089308,  0.2288961261510849 ,
  -0.05703318119049072,  0.0709901750087738 , -0.01540892012417316,
   0.00000382380994779,  0.05177905410528183,  0.12215259671211243,
  -0.03908038139343262,  0.1977296769618988 , -0.01981677301228046,
  -0.11902561783790588, -0.2597291171550751 , -0.11566510051488876,
   0.16100835800170898,  0.0036420130636543 , -0.091976098716259  ,
  -0.01912974752485752,  0.01625743694603443, -0.379302442073822  ,
   0.26516982913017273, -0.21929214894771576, -0.11093011498451233,
  -0.02501381374895573,  0.12545645236968994, -0.5972849130630493 ,
   0.0937468484044075 ,  0.02057958953082561, -0.01825197786092758,
   0.04414284974336624, -0.03835143521428108, -0.2696274518966675 ,
   0.10740969330072403, -0.08037757873535156,  0.33422160148620605,
  -0.04766153171658516, -0.02082838304340839, -0.01935428753495216,
  -0.6180847883224487 , -0.0398436039686203 , -0.01731272041797638,
  -0.24677376449108124,  0.06388699263334274],
 [-0.04967503994703293, -0.9418565034866333 , -0.5708526968955994 ,
  -0.8165953159332275 ,  0.5217429995536804 ,  0.01753045432269573,
  -0.46866849064826965,  0.22196105122566223,  0.22965916991233826,
   0.1361231505870819 ,  0.27096086740493774, -0.3567584455013275 ,
  -1.0803004503250122 ,  0.8127568364143372 ,  0.658142626285553  ,
   0.25441789627075195, -0.14067840576171875, -0.40784692764282227,
   0.43146225810050964,  0.00475176516920328, -0.04979106411337852,
  -0.3698490858078003 , -0.8724444508552551 , -0.4786878526210785 ,
  -0.7295100688934326 , -0.5215290188789368 ,  0.7412430644035339 ,
   0.5647780299186707 , -0.17904730141162872,  0.5445351004600525 ,
  -0.3516729772090912 , -1.1541301012039185 ,  0.05833448842167854,
   0.606065034866333  ,  0.4143584072589874 , -0.5306718945503235 ,
   0.15393352508544922,  0.12453905493021011, -0.5631775259971619 ,
  -0.9191136956214905 , -1.7986177206039429 , -0.5174365043640137 ,
  -1.1853768825531006 ,  0.15772278606891632, -0.07893472909927368,
  -0.8787603974342346 , -0.5675248503684998 ,  0.4936426877975464 ,
  -0.05381718277931213, -0.5824244618415833 , -0.6314976215362549 ,
   0.03481324762105942,  0.34904709458351135, -0.3383176326751709 ,
  -0.24558348953723907,  0.36354318261146545, -0.0374731570482254 ,
  -0.0116916261613369 , -1.0210003852844238 ,  0.7764267325401306 ,
  -1.5284149646759033 , -0.480037122964859  , -0.4364370107650757 ,
  -0.03404496982693672, -0.05822183191776276, -1.0416101217269897 ,
  -1.0337138175964355 , -0.0443723089993    , -1.1126854419708252 ,
  -0.4934541583061218 ,  0.0205870047211647 , -0.0234188511967659 ,
   0.09657877683639526, -1.8701177835464478 , -0.07863128930330276,
  -0.3166012465953827 ,  0.4886474013328552 , -0.9050647020339966 ,
  -0.3735063672065735 ,  0.339514821767807  , -0.25253593921661377,
  -0.18368932604789734, -1.306470513343811  , -0.2557166814804077 ,
   0.6533600091934204 , -1.4673317670822144 ,  0.10127244144678116,
  -0.24635952711105347,  0.35539817810058594, -0.6591036319732666 ,
  -0.7176867723464966 , -0.6639073491096497 , -0.31847816705703735,
  -1.9038053750991821 , -1.5910323858261108 ,  0.3321981728076935 ,
  -0.02315052784979343, -2.071881055831909  , -0.04893426969647408,
  -0.5212599039077759 ,  0.2318640947341919 , -0.6134742498397827 ,
  -0.5124991536140442 , -0.349624365568161  ,  0.1482391357421875 ,
   0.20216527581214905, -0.26426172256469727, -0.2818998396396637 ,
  -0.05458414927124977,  0.07859711349010468, -0.7632936239242554 ,
  -0.06668364256620407, -0.21887707710266113, -0.5378291010856628 ,
  -0.05615640059113503, -1.3768261671066284 , -1.4445606470108032 ,
   0.12495899200439453,  0.45504242181777954,  0.03651844710111618,
  -0.6060143709182739 ,  0.7740247845649719 ,  1.103161334991455  ,
  -0.1457451432943344 ,  0.19014008343219757, -0.9472872018814087 ,
  -0.8619288206100464 ,  0.7061706781387329 ],
 [-0.2208426594734192 ,  0.42762476205825806,  0.41374778747558594,
   0.12801602482795715,  0.05586099624633789,  0.57851243019104   ,
  -0.7857771515846252 ,  0.3682142496109009 , -1.027743935585022  ,
  -0.13783617317676544, -2.271230936050415  , -0.17086951434612274,
   0.4906544089317322 ,  0.6243684887886047 , -0.4575589895248413 ,
  -0.22852861881256104,  0.10312426835298538,  0.5630324482917786 ,
   0.25780296325683594, -1.1837785243988037 ,  0.4887225329875946 ,
   0.3231126666069031 ,  0.3383695185184479 , -2.6067211627960205 ,
  -0.8521929979324341 ,  0.27464139461517334, -1.6626496315002441 ,
  -0.3785422146320343 , -1.412036657333374  , -2.633575916290283  ,
  -0.06163316965103149,  0.5646860003471375 ,  2.0651211738586426 ,
  -3.298274278640747  , -0.7323774099349976 , -1.845951795578003  ,
   0.0861935168504715 , -0.3212801218032837 ,  0.7735334634780884 ,
  -0.21985042095184326,  1.9965416193008423 ,  0.11732310801744461,
  -0.08821406960487366,  0.281502366065979  , -1.7394262552261353 ,
  -0.22816844284534454,  1.022206425666809  ,  0.30776122212409973,
   0.03201278671622276,  0.24475227296352386,  0.28677210211753845,
  -0.44445037841796875, -0.2746717631816864 , -0.02933667600154877,
  -1.9182437658309937 , -2.4798858165740967 ,  1.3392432928085327 ,
  -0.4358583092689514 ,  0.37337878346443176, -0.6126899719238281 ,
   0.5433217287063599 ,  0.43098270893096924, -0.7057138681411743 ,
   0.08181268721818924, -0.2613801658153534 ,  2.010948419570923  ,
   0.7367899417877197 ,  0.19594870507717133,  0.7468680739402771 ,
  -1.9358140230178833 ,  0.12567773461341858,  0.5122491717338562 ,
   0.31031128764152527,  0.03997574001550674, -0.10984136909246445,
  -0.4784109592437744 , -2.2498421669006348 ,  0.4159245193004608 ,
  -0.6158319115638733 ,  0.21986649930477142,  0.8707379102706909 ,
  -0.2748729884624481 ,  0.00382394716143608,  3.2671918869018555 ,
  -0.5044272541999817 ,  1.1249370574951172 , -0.2083083540201187 ,
  -2.749768018722534  ,  0.3747362792491913 , -0.25379127264022827,
   1.0777446031570435 , -0.44176602363586426, -0.23227271437644958,
   0.06053662300109863, -0.2795030176639557 ,  0.9397026300430298 ,
  -0.11008814722299576,  0.6178774237632751 , -1.2491647005081177 ,
   0.10357123613357544, -0.6524741053581238 , -0.3397813141345978 ,
  -0.13727636635303497,  0.20798557996749878, -0.862060546875     ,
  -1.3241989612579346 , -0.3639049530029297 , -0.6860238909721375 ,
   0.7369282245635986 , -0.09262343496084213, -0.9996292591094971 ,
  -1.330925703048706  , -0.19563166797161102,  0.7871137261390686 ,
  -0.7125341892242432 ,  0.07225042581558228, -0.5825814604759216 ,
  -2.4746148586273193 , -1.226349115371704  , -0.02774039097130299,
   2.931673765182495  , -1.4498488903045654 , -0.01308752689510584,
  -0.43048521876335144,  1.332615852355957  , -0.20321190357208252,
  -0.21406883001327515, -1.2579035758972168 ],
 [ 0.23717334866523743, -1.0687605142593384 ,  0.2710797190666199 ,
  -0.29790759086608887,  0.08761016279459   , -0.8861396908760071 ,
  -0.13725097477436066,  0.602436900138855  , -0.7224985361099243 ,
  -0.49889442324638367,  0.58527010679245   , -0.12203095108270645,
  -0.9115247130393982 , -2.1821396350860596 , -0.1321459859609604 ,
  -0.23267100751399994,  0.7574517726898193 ,  0.12059003859758377,
   0.01430749148130417, -0.2154540717601776 , -1.1342442035675049 ,
   0.17027387022972107, -0.34958454966545105, -0.24023905396461487,
   0.9784965515136719 , -1.7116503715515137 , -1.0893937349319458 ,
  -0.722320556640625  , -0.08316711336374283, -0.8526272773742676 ,
  -0.3799285292625427 , -0.34938308596611023, -0.5349240303039551 ,
  -1.5165972709655762 ,  1.0263841152191162 , -0.702988862991333  ,
  -0.2570672035217285 ,  0.34159842133522034, -1.4630663394927979 ,
  -0.33633875846862793, -2.538266181945801  , -2.5796878337860107 ,
  -0.8299872279167175 , -2.9449000358581543 , -0.7118816375732422 ,
  -0.43901678919792175, -2.131890058517456  , -0.8838902711868286 ,
  -0.0334467813372612 , -0.38016918301582336, -1.9938743114471436 ,
   0.31139105558395386, -0.9013068079948425 , -0.19999492168426514,
   1.2907483577728271 , -0.10664596408605576, -1.4924801588058472 ,
  -0.27103087306022644, -0.5884881615638733 , -0.3155147433280945 ,
   0.46707695722579956, -0.4760201871395111 ,  0.28744834661483765,
   0.2746759355068207 ,  0.16506168246269226,  0.02356439456343651,
  -1.4159621000289917 , -0.29505208134651184, -0.8090900778770447 ,
   0.26699918508529663, -1.4945955276489258 , -0.06100090965628624,
   0.04169006645679474, -1.127137541770935  ,  0.1711438149213791 ,
  -1.009992241859436  , -2.2599315643310547 , -1.4449183940887451 ,
  -0.7198525667190552 , -0.39136216044425964, -0.3182162642478943 ,
   0.05169786512851715,  0.1108362078666687 ,  1.8878936767578125 ,
  -0.6503918170928955 , -0.1362060010433197 , -0.3797067701816559 ,
  -1.476527214050293  , -0.32789891958236694, -0.337990403175354  ,
  -0.10674624890089035,  0.12410666048526764, -0.10164312273263931,
  -1.2026598453521729 , -0.13682232797145844, -1.7443294525146484 ,
  -0.4866342842578888 ,  0.6293027400970459 , -1.4037045240402222 ,
   0.0438896082341671 , -0.6418346166610718 , -0.8599667549133301 ,
   0.7016153335571289 , -1.3067635297775269 ,  0.6160556674003601 ,
   0.2083323746919632 , -0.27849826216697693, -0.19576279819011688,
  -1.8353650569915771 , -0.18709395825862885, -0.14490550756454468,
  -0.7067971229553223 ,  0.6309683918952942 , -0.2790946066379547 ,
   0.723820686340332  , -1.3044387102127075 ,  0.02273801527917385,
  -0.6206058859825134 , -1.3640159368515015 , -0.4394634962081909 ,
  -2.9894256591796875 , -0.22688114643096924, -0.20236603915691376,
   0.33446332812309265,  0.02159001678228378,  0.7408008575439453 ,
   0.08514170348644257, -1.9189702272415161 ],
 [ 0.582309901714325  ,  0.02702171914279461,  1.1612786054611206 ,
   0.18083824217319489,  0.6235799193382263 , -0.00640350021421909,
   0.9448235630989075 ,  0.13853153586387634, -0.5906514525413513 ,
  -3.9261178970336914 ,  1.0860131978988647 ,  0.3832273483276367 ,
   0.722744345664978  ,  0.5060010552406311 ,  0.33783549070358276,
   0.5570884943008423 , -0.21767161786556244, -0.9942519664764404 ,
  -0.9202958941459656 ,  0.6116071939468384 ,  0.10066977143287659,
  -0.7093053460121155 , -0.00713245570659637,  0.7220210433006287 ,
   0.6536411046981812 , -1.147409439086914  ,  0.3215824365615845 ,
  -1.530778408050537  , -0.8840807676315308 ,  0.01537199132144451,
   0.8551509976387024 ,  0.5550724267959595 ,  0.08932086825370789,
  -2.333554267883301  , -0.5265432000160217 , -0.7575008869171143 ,
  -0.5647645592689514 , -1.0401283502578735 ,  0.7282442450523376 ,
  -0.03560514375567436, -0.8150519132614136 , -0.2351692020893097 ,
   1.556634783744812  , -0.6389554738998413 ,  0.5106930136680603 ,
  -0.43070581555366516, -0.02435849048197269,  0.3054350018501282 ,
  -0.7850680351257324 ,  1.376354694366455  , -0.5255788564682007 ,
   0.3043252229690552 , -0.4355683922767639 ,  0.529346227645874  ,
   0.2664939761161804 ,  0.5301725268363953 ,  0.11761384457349777,
  -1.1194980144500732 ,  0.4977209270000458 ,  0.458820641040802  ,
   0.01000879518687725,  0.1173926368355751 ,  0.27677106857299805,
   0.1883215755224228 ,  0.21014665067195892, -0.39123648405075073,
   0.2745473086833954 , -0.01257786061614752,  0.19682781398296356,
   0.07778622955083847, -0.14347533881664276, -1.7355024814605713 ,
  -0.12945249676704407,  0.39662113785743713,  0.0982130840420723 ,
   0.6267858147621155 , -0.7022305130958557 ,  0.16580915451049805,
   0.19935572147369385,  0.360451877117157  ,  1.4500696659088135 ,
  -1.2265020608901978 , -0.0450267493724823 , -0.32575881481170654,
   2.3458540439605713 ,  0.04882184788584709, -2.6511495113372803 ,
   0.31147027015686035, -0.16305115818977356,  0.5138245224952698 ,
  -0.2648143768310547 ,  0.4700131416320801 , -0.775675356388092  ,
   0.24958953261375427, -0.38119688630104065,  0.715021550655365  ,
  -0.5726345181465149 ,  0.200600728392601  , -0.5247049331665039 ,
  -0.1846248358488083 , -0.2734545171260834 ,  0.7518634796142578 ,
   0.4304327666759491 , -0.12263362109661102, -0.40807488560676575,
  -0.27729395031929016, -0.06798692792654037,  0.10933726280927658,
  -2.1184513568878174 , -0.3231499195098877 , -0.6815309524536133 ,
   1.3089054822921753 ,  0.09403669834136963, -0.3113876283168793 ,
  -0.373699814081192  ,  0.9887425899505615 , -0.18967071175575256,
  -0.502213716506958  ,  0.4128948152065277 ,  1.6398550271987915 ,
  -0.3599189221858978 , -0.5437387228012085 ,  0.13037574291229248,
  -2.2495486736297607 , -1.460498571395874  , -1.8200616836547852 ,
   0.6593717932701111 ,  0.4619138538837433 ],
 [-0.03699242696166039,  0.08888179063796997,  0.03575544059276581,
  -0.25952425599098206, -0.4124279022216797 ,  0.5785645842552185 ,
  -1.842926025390625  ,  0.05872882157564163, -1.0415666103363037 ,
  -0.15745019912719727,  0.33500224351882935,  0.05395318940281868,
  -0.1613740175962448 ,  0.3684995174407959 , -0.11949155479669571,
  -0.03997126221656799, -0.5904988646507263 , -0.40283235907554626,
  -0.6137971878051758 , -0.854834258556366  , -0.15536481142044067,
   0.9810221195220947 ,  0.5955027341842651 ,  0.4503900706768036 ,
   0.02191492728888988, -0.10674093663692474,  1.3861600160598755 ,
   0.7966517210006714 , -0.01212336402386427,  0.9894318580627441 ,
   0.05627911537885666, -0.5750504732131958 , -0.7885759472846985 ,
  -0.7684689164161682 ,  0.12325605005025864, -2.441521644592285  ,
   0.31888309121131897, -0.04714863002300262, -0.2618556618690491 ,
  -0.31382790207862854,  1.0285346508026123 , -0.25448015332221985,
   0.33758699893951416,  1.3280600309371948 ,  0.04968861490488052,
   0.4667787551879883 ,  1.3783084154129028 , -0.6477758884429932 ,
  -0.05403194576501846,  0.34232038259506226, -1.4893929958343506 ,
   0.3028876483440399 , -0.4880560040473938 , -0.09738288074731827,
  -1.062802791595459  , -1.0664823055267334 ,  0.22134849429130554,
  -0.02501804381608963,  0.51835697889328   , -0.32976001501083374,
  -0.16488203406333923,  0.14549566805362701,  0.36495181918144226,
  -0.07490609586238861,  0.2212551236152649 ,  0.879433810710907  ,
   0.32244327664375305, -1.0276174545288086 ,  0.2101079821586609 ,
  -0.4607350826263428 , -0.9356376528739929 , -0.22023507952690125,
  -0.13067209720611572, -0.1980244666337967 , -0.43386656045913696,
  -0.0974370539188385 , -0.9133642315864563 , -0.6629922389984131 ,
  -0.48254165053367615, -0.00256353593431413,  0.9429031014442444 ,
   1.5302836894989014 , -0.23995524644851685,  0.9441213011741638 ,
  -0.09070111811161041, -0.25104331970214844, -0.8208966851234436 ,
  -0.7493707537651062 ,  0.11659800261259079, -0.01911015994846821,
  -0.10530887544155121, -0.01438862085342407, -0.4255498945713043 ,
  -0.02839009836316109, -0.2575368285179138 , -0.464354544878006  ,
  -0.6823445558547974 , -2.4603397846221924 , -0.5336244702339172 ,
   0.12954483926296234, -0.19287046790122986, -0.8268070816993713 ,
  -0.2070654183626175 , -0.7279012799263    , -0.34551090002059937,
  -0.42452436685562134, -1.1937001943588257 , -0.720416784286499  ,
  -0.027207275852561  , -0.12650109827518463, -0.3101418912410736 ,
  -0.10544469207525253, -0.5172640681266785 , -0.08327804505825043,
  -0.02507711388170719,  0.33592209219932556, -0.06605150550603867,
  -1.7033085823059082 , -0.6578707098960876 ,  0.2760997414588928 ,
  -0.34977981448173523,  1.1212217807769775 , -0.09114566445350647,
   0.24241410195827484,  0.20940491557121277, -0.10805146396160126,
   0.3736605942249298 , -1.2321584224700928 ],
 [-0.25963568687438965, -0.3156927227973938 ,  1.4529985189437866 ,
  -1.0491749048233032 ,  0.8947267532348633 , -0.2840753197669983 ,
  -1.1173604726791382 ,  0.39544379711151123, -1.9853758811950684 ,
   0.9662401080131531 , -1.3153513669967651 ,  0.1859450787305832 ,
   0.5552419424057007 ,  0.6316779851913452 ,  0.07596756517887115,
  -1.9205329418182373 , -1.7388226985931396 ,  0.5722081065177917 ,
   0.46914273500442505, -4.315001010894775  ,  0.01213926263153553,
  -1.4341098070144653 ,  1.081871747970581  , -1.3339109420776367 ,
  -1.3062007427215576 ,  0.12134400755167007, -0.24401512742042542,
  -0.5270131826400757 , -1.3744741678237915 , -1.7953248023986816 ,
   0.7380704283714294 , -0.8156646490097046 ,  0.824248731136322  ,
   0.3717188835144043 ,  1.146540641784668  , -3.758061647415161  ,
  -1.1580687761306763 , -0.04533471539616585, -2.779578447341919  ,
   0.3287127614021301 , -2.095825433731079  ,  1.760714054107666  ,
  -0.9019446969032288 , -1.866409182548523  , -0.6679216623306274 ,
   1.1672792434692383 ,  1.0147086381912231 , -0.5786010026931763 ,
  -0.01265110820531845,  0.12444108724594116, -0.3777005970478058 ,
  -0.33287692070007324,  0.9306069612503052 ,  0.18563774228096008,
  -1.6436058282852173 , -0.35643577575683594,  0.01967204548418522,
   0.21477986872196198, -0.10897216945886612, -0.7566866278648376 ,
  -0.04871476069092751,  0.2679717540740967 ,  0.11985082179307938,
  -0.33453139662742615,  0.3125585615634918 ,  0.02632865309715271,
  -0.18296043574810028, -1.6636114120483398 ,  0.2139839231967926 ,
  -1.4198386669158936 , -1.238588809967041  ,  0.2919401526451111 ,
   0.484296053647995  , -3.2254841327667236 , -0.7031545639038086 ,
  -0.8918184638023376 , -0.48881563544273376, -0.17887479066848755,
  -0.18394170701503754,  0.7689687013626099 , -3.502575159072876  ,
   0.7555992007255554 ,  0.00540606305003166, -0.6678152680397034 ,
  -1.8609164953231812 ,  0.3880605399608612 , -0.13150349259376526,
   0.5763559937477112 , -0.1956394612789154 ,  0.43714648485183716,
  -1.1324522495269775 , -0.4162885844707489 , -0.13584481179714203,
  -1.0032083988189697 , -0.532576858997345  , -2.422281265258789  ,
  -0.9156184792518616 , -0.04969711974263191, -1.4826208353042603 ,
   0.5357570648193359 ,  0.2899630665779114 , -1.1213610172271729 ,
   1.199986219406128  , -0.16407473385334015, -1.2730556726455688 ,
   0.19800829887390137,  0.2123587280511856 , -0.2317618429660797 ,
   0.40582939982414246, -2.6666035652160645 ,  0.815203845500946  ,
  -1.8449797630310059 , -0.7492333650588989 ,  0.6117352247238159 ,
   0.15243855118751526, -0.9399545788764954 ,  0.08072633296251297,
  -2.476195812225342  , -0.5034155249595642 , -0.1345382034778595 ,
  -1.4239273071289062 ,  0.3215552270412445 , -1.018453598022461  ,
  -0.2564687132835388 ,  0.2707538306713104 , -0.05764725059270859,
  -0.08942928910255432, -3.8177852630615234 ],
 [ 0.24986764788627625, -0.8565276265144348 ,  0.10028853267431259,
  -0.36174941062927246,  0.7692609429359436 , -1.9768481254577637 ,
  -0.4816856384277344 , -0.2569001317024231 , -0.501844584941864  ,
  -0.11272719502449036, -0.4005456268787384 ,  0.12060942500829697,
   0.04584651440382004, -0.809855043888092  ,  0.12971104681491852,
  -0.01881290785968304, -0.6353498101234436 ,  0.14590781927108765,
  -0.4719178080558777 , -0.2508685290813446 , -0.13608068227767944,
  -0.09504444152116776, -0.760200023651123  , -0.14146184921264648,
  -1.0016449689865112 , -0.4508121609687805 , -0.25737038254737854,
  -0.13160502910614014,  0.14410366117954254, -0.43695688247680664,
  -0.1281796246767044 ,  0.23247568309307098, -0.40256810188293457,
  -0.19339001178741455, -0.0776086300611496 , -0.16894234716892242,
  -0.42145073413848877,  0.12080776691436768,  0.2525556683540344 ,
  -0.32471710443496704, -0.5146046876907349 ,  0.08249078691005707,
   0.2215968370437622 , -0.00285558984614909,  0.2023584395647049 ,
  -0.34224313497543335, -0.9671119451522827 , -0.6111978888511658 ,
  -1.3152813911437988 , -0.0279870368540287 , -0.20393122732639313,
   0.27144530415534973,  0.01085588056594133,  0.01196249574422836,
  -1.214100956916809  , -0.2039995789527893 , -0.02290159463882446,
   0.260627418756485  ,  0.30217716097831726, -0.2870330810546875 ,
  -0.43102720379829407,  0.5948957800865173 , -0.27205055952072144,
  -0.8038845658302307 ,  0.15630608797073364,  0.3413679599761963 ,
   0.33076924085617065, -0.9800562858581543 ,  0.29089558124542236,
  -0.49847909808158875,  0.21504555642604828,  0.4723581075668335 ,
  -0.14618922770023346,  0.29857900738716125,  0.2643303871154785 ,
   0.15394583344459534, -0.3733336925506592 ,  0.24681256711483002,
  -0.10239566117525101, -0.22105954587459564, -0.12646576762199402,
  -0.5507627129554749 ,  0.7148463129997253 ,  0.2372768223285675 ,
  -0.25197383761405945,  0.3611159026622772 , -0.04197941347956657,
  -0.6758618950843811 ,  0.29324987530708313,  0.16376005113124847,
   0.04503889009356499, -0.20722024142742157,  0.06185602769255638,
   0.5449675917625427 , -0.35896939039230347,  0.35104191303253174,
   0.07610556483268738,  0.5435340404510498 ,  0.23302869498729706,
  -0.10914763063192368, -0.17502287030220032, -1.4066741466522217 ,
  -0.0076907011680305 , -0.21382786333560944, -0.9635112285614014 ,
   0.03998621180653572, -0.20959404110908508, -0.1722853183746338 ,
   0.2942437529563904 ,  0.02546108514070511,  0.35872897505760193,
   0.36941632628440857,  0.42420658469200134, -0.122862309217453  ,
  -0.6298695206642151 , -0.642218828201294  ,  0.7274975776672363 ,
   0.01605267077684402, -0.15423454344272614, -0.3547373414039612 ,
  -0.07304095476865768, -0.254112184047699  , -0.0754789337515831 ,
  -0.02862496674060822, -0.0596451498568058 ,  0.4554729759693146 ,
  -0.7289243936538696 , -0.83928382396698   ],
 [-0.18920980393886566,  0.02947216294705868, -0.3513107895851135 ,
  -1.2986505031585693 ,  0.49292680621147156, -0.1462937444448471 ,
  -0.577783465385437  , -0.7561256289482117 ,  0.12087815999984741,
   0.15657879412174225,  0.3581473231315613 ,  0.1818220466375351 ,
   0.24556367099285126, -0.5080587863922119 , -0.6117549538612366 ,
   0.28019070625305176, -1.3404301404953003 , -0.7125368118286133 ,
   0.21071504056453705, -0.2633567154407501 ,  0.0908658429980278 ,
   0.02359882369637489, -0.1172085553407669 , -0.256556898355484  ,
  -0.49288055300712585,  0.40423473715782166, -0.0743299126625061 ,
   0.4892033338546753 ,  0.5774015188217163 ,  0.00705461855977774,
   0.3100367784500122 , -0.48964378237724304,  1.018794059753418  ,
  -0.42686739563941956, -0.11395510286092758, -0.1943369358778    ,
  -0.36967596411705017, -0.4147610068321228 , -0.2071906179189682 ,
   0.05589846149086952, -0.7225620150566101 , -1.1179845333099365 ,
   0.19104449450969696,  0.47306570410728455, -0.04891526326537132,
  -0.5361031293869019 , -0.19256916642189026,  0.7435551285743713 ,
  -0.2004425823688507 , -0.39892104268074036, -0.44453340768814087,
  -0.23802323639392853, -0.14475783705711365, -0.06903687864542007,
  -0.2419697791337967 ,  0.18828625977039337, -0.5149251818656921 ,
  -0.35947471857070923, -0.5158439874649048 , -0.120127372443676  ,
   0.07585983723402023,  0.5689216256141663 ,  0.15024639666080475,
   0.5389909148216248 , -0.30447933077812195,  0.6851847767829895 ,
  -0.6804715394973755 , -0.7855220437049866 ,  0.3256188631057739 ,
   0.6539238095283508 , -0.26391103863716125, -0.293933629989624  ,
   0.7582052946090698 ,  0.9965066313743591 , -0.09294774383306503,
   0.8222091794013977 ,  0.48543399572372437,  0.0162893682718277 ,
   0.5833549499511719 ,  0.25844520330429077, -0.9146177172660828 ,
  -1.3300011157989502 , -0.44825607538223267, -0.37522274255752563,
  -0.32562533020973206, -0.8782956004142761 , -0.1283305436372757 ,
   0.9826433658599854 ,  0.18280570209026337,  0.02984478510916233,
   0.5224046111106873 , -0.3408903181552887 , -2.3189873695373535 ,
  -0.49220359325408936,  0.01938092894852161,  0.37792736291885376,
  -0.8807424306869507 , -0.10477278381586075, -0.09859184920787811,
   0.10998444259166718,  0.13654455542564392, -0.5086521506309509 ,
   0.18859294056892395, -0.1696658432483673 , -1.572397232055664  ,
   0.28363850712776184,  0.15117375552654266, -0.7665954828262329 ,
   0.591679573059082  ,  0.25993612408638   ,  0.09599122405052185,
   0.2068362534046173 , -0.7353605031967163 , -0.02866609767079353,
  -0.724823534488678  , -0.02563095651566982,  0.03074086643755436,
   0.3230113685131073 ,  0.1316794604063034 ,  0.1646699160337448 ,
   0.15128174424171448, -0.04621145501732826, -0.44448122382164   ,
  -0.9817328453063965 ,  0.26742324233055115,  0.4122471809387207 ,
  -0.03849266842007637,  1.2208032608032227 ],
 [-1.1123905181884766 ,  0.6775137782096863 , -0.1539084017276764 ,
   0.5207135081291199 ,  0.10653440654277802, -1.4595059156417847 ,
  -0.20254065096378326, -0.06375891715288162, -0.9669502377510071 ,
  -0.17968657612800598, -0.8197900056838989 , -0.2628975212574005 ,
   0.05891980975866318,  0.22018717229366302,  0.09878775477409363,
   0.20734143257141113,  0.3374759554862976 , -0.27405688166618347,
  -0.7789333462715149 ,  0.12400241196155548, -0.02121314778923988,
   0.01734825037419796, -2.088499069213867  ,  0.29295554757118225,
  -0.2555050253868103 , -0.06792560964822769, -0.14249061048030853,
   0.3057098090648651 ,  0.7447667121887207 ,  0.534508228302002  ,
  -0.29718244075775146,  0.9703484773635864 , -1.9737671613693237 ,
  -0.04003196954727173, -0.06722411513328552,  0.9239389896392822 ,
  -1.0692832469940186 ,  0.1031540259718895 ,  0.10865988582372665,
   0.07745058834552765, -0.33519741892814636, -0.04510565474629402,
  -0.04834123328328133,  0.23058101534843445,  0.22257515788078308,
  -0.5976362228393555 ,  0.04951774701476097,  0.3350329101085663 ,
  -0.28185999393463135, -0.8046093583106995 ,  0.48342204093933105,
  -0.30934539437294006, -0.1827050745487213 ,  0.03913292288780212,
   0.2615661323070526 , -0.3949088156223297 , -1.0224223136901855 ,
   0.05119255930185318, -0.18562252819538116, -0.5896745324134827 ,
   0.10130833089351654,  0.9735154509544373 , -1.0826687812805176 ,
  -1.6633285284042358 ,  0.34953126311302185, -0.11203540116548538,
  -0.8325231075286865 ,  1.020080804824829  ,  0.24368272721767426,
   1.0788276195526123 , -0.43863746523857117,  0.24718064069747925,
   0.6064080595970154 ,  0.3407348692417145 , -0.04425982013344765,
  -0.4984075129032135 , -1.9203144311904907 ,  0.8857384324073792 ,
  -0.646526575088501  ,  0.1468515247106552 , -0.08730363845825195,
   0.3464297354221344 , -0.13868765532970428, -0.15506160259246826,
  -0.0099364323541522 , -0.5988354682922363 ,  0.37375539541244507,
   0.18192100524902344,  0.39924952387809753, -0.5283506512641907 ,
   0.3809747099876404 ,  0.00424294453114271,  0.6146189570426941 ,
   0.31642529368400574, -0.09121310710906982,  0.37147101759910583,
   0.28829652070999146,  0.17834976315498352,  0.43266698718070984,
  -0.5957877039909363 , -0.44731593132019043,  0.3845531940460205 ,
  -0.23529267311096191, -0.29109492897987366, -0.4868123531341553 ,
   0.07196681201457977, -0.04833690822124481,  0.23756089806556702,
   0.6124595999717712 ,  0.1226385086774826 ,  0.02126412838697433,
   0.25938117504119873, -0.2999430000782013 ,  0.3406684100627899 ,
  -0.19347314536571503,  0.3277937173843384 , -0.8826231956481934 ,
   0.07950997352600098, -0.41690829396247864,  0.29494744539260864,
   0.4734092950820923 , -0.6484121680259705 , -0.6500807404518127 ,
   0.29262545704841614,  0.20734471082687378,  0.14002567529678345,
  -0.9817572832107544 , -0.2466139793395996 ],
 [ 0.34984347224235535, -0.7497917413711548 , -0.264666348695755  ,
  -0.11338850855827332,  1.2369922399520874 ,  0.23465192317962646,
  -0.02602323703467846,  0.5136700868606567 , -0.5869999527931213 ,
  -0.06218894943594933, -0.17184409499168396,  0.37819328904151917,
   0.28008902072906494, -1.003824234008789  , -0.13172584772109985,
  -0.17750844359397888,  1.0207209587097168 , -0.8897075057029724 ,
   0.8222207427024841 , -0.5180989503860474 , -0.2567259967327118 ,
   0.9438722133636475 ,  0.04311973229050636, -0.6553506851196289 ,
  -0.9965658187866211 , -0.13934554159641266, -0.18604248762130737,
  -0.02394094504415989, -1.6418973207473755 , -0.7118805646896362 ,
   0.07959522306919098,  0.48339444398880005,  0.8045947551727295 ,
  -0.38307175040245056,  0.02833806909620762, -0.6095806956291199 ,
   1.065606713294983  ,  0.139816552400589  ,  1.8894941806793213 ,
   0.07281266152858734,  0.6874904036521912 , -0.5421971678733826 ,
   0.10394459962844849,  0.48221394419670105, -0.01268054638057947,
   0.15391269326210022,  1.3479068279266357 ,  0.6417189836502075 ,
   0.7050905227661133 ,  0.2898193299770355 , -0.06976523995399475,
   0.1479327529668808 ,  0.16836923360824585, -0.20634810626506805,
   1.0141657590866089 , -0.6349523663520813 ,  1.2980282306671143 ,
  -0.5353763103485107 , -0.22181454300880432,  0.5338976979255676 ,
   0.5584530234336853 , -0.5447726845741272 , -0.05832343921065331,
   0.03851913660764694,  0.9665564894676208 ,  0.10830900073051453,
  -0.06618987768888474, -0.27904874086380005, -0.1373928338289261 ,
  -0.09009073674678802, -0.45838141441345215, -0.20469507575035095,
  -0.2289752960205078 ,  0.38913753628730774,  0.347532719373703  ,
   0.8504775762557983 ,  0.35238051414489746,  0.12807193398475647,
  -0.7192896604537964 , -0.26457318663597107,  0.9602954387664795 ,
   0.1070626974105835 , -0.02736867219209671, -1.4013195037841797 ,
   0.2065216451883316 , -0.22006019949913025, -0.13954663276672363,
  -0.43397268652915955,  0.26869091391563416, -0.3240815997123718 ,
   0.2407049983739853 ,  0.0381740927696228 , -0.20329804718494415,
  -0.03920014947652817,  0.434349924325943  , -1.0065670013427734 ,
  -0.6481636166572571 ,  0.8177052140235901 , -0.00202978099696338,
   0.24932502210140228,  0.5218704342842102 , -0.27224403619766235,
   0.037653598934412  ,  0.239049032330513  ,  1.134576439857483  ,
   0.28833943605422974, -0.0331353135406971 , -0.2587433159351349 ,
   0.5670400261878967 ,  0.00443727243691683,  0.5424640774726868 ,
  -0.04643423855304718,  1.4364428520202637 ,  0.5849820971488953 ,
  -0.00148586556315422,  0.8700136542320251 , -0.11409863084554672,
  -0.00821861438453197,  0.09980881214141846,  0.9264222383499146 ,
   1.3926414251327515 , -0.5485244989395142 , -0.39367902278900146,
  -0.06445944309234619, -0.6752545833587646 , -0.8182059526443481 ,
   0.3496765196323395 ,  0.7224012017250061 ],
 [ 0.06770343333482742, -0.39037641882896423,  0.11772332340478897,
   0.43692541122436523, -0.6482018232345581 , -0.6822358965873718 ,
  -0.21898867189884186, -0.8466306328773499 , -0.06047302857041359,
  -0.3267895579338074 , -0.0598917119204998 , -0.39185118675231934,
  -0.2839697599411011 ,  0.05817379429936409, -0.6348528265953064 ,
  -0.16712792217731476,  0.6606746912002563 , -0.30266815423965454,
  -0.1900460422039032 , -1.0739294290542603 , -0.39524877071380615,
  -0.07093323767185211, -0.0588582418859005 , -0.295385479927063  ,
   0.6248359084129333 , -0.299809068441391  , -0.8431492447853088 ,
  -0.09669534116983414, -0.6997186541557312 , -0.13785666227340698,
  -0.5324456095695496 , -0.2761382758617401 , -0.3415105938911438 ,
  -0.9230026602745056 , -0.2669453024864197 ,  1.0694483518600464 ,
   0.29313230514526367,  0.06629825383424759,  0.781679630279541  ,
  -0.01881652511656284,  0.0820348858833313 ,  0.4950741231441498 ,
   0.6851727962493896 ,  0.26915115118026733,  0.5294264554977417 ,
  -0.100957952439785  ,  0.16692186892032623, -0.03094891272485256,
  -0.6338598132133484 ,  1.2438087463378906 , -0.10415409505367279,
   0.20662535727024078, -0.7327280044555664 , -0.0094046238809824 ,
   0.21348755061626434,  0.12176371365785599, -0.5400151610374451 ,
   0.30918294191360474,  0.0976586863398552 ,  0.616229236125946  ,
   0.28191888332366943,  0.5375530123710632 , -0.16127590835094452,
  -0.40011414885520935,  0.40564289689064026, -0.41385218501091003,
  -0.10692889988422394,  0.5670032501220703 ,  0.5517894625663757 ,
   0.96527498960495   ,  0.10604875534772873,  0.7998645901679993 ,
   0.9260921478271484 , -0.6604914665222168 , -0.9295390844345093 ,
  -0.343470960855484  , -0.3061504065990448 ,  1.3023347854614258 ,
  -0.11286099255084991, -0.0207088328897953 ,  0.3492298126220703 ,
   0.22140195965766907, -0.03245866671204567,  0.7392438054084778 ,
  -0.627454400062561  , -0.20277009904384613, -0.41428127884864807,
  -0.31076082587242126,  0.00690623791888356, -0.39839300513267517,
  -0.547758162021637  , -0.22999468445777893, -0.12499476224184036,
   0.0943814218044281 ,  0.32777509093284607,  0.10667385905981064,
  -0.28996387124061584, -0.3707016110420227 ,  0.4944065511226654 ,
   0.05768154934048653, -0.7430400848388672 ,  0.5530896782875061 ,
   0.11343293637037277, -0.14524060487747192,  0.5457049012184143 ,
  -0.5098922848701477 ,  0.51340651512146   ,  0.06268637627363205,
   0.6380558013916016 ,  0.06532704830169678,  0.4629819393157959 ,
   0.26512452960014343,  0.28378531336784363,  0.3268965184688568 ,
   0.2425113022327423 ,  1.0239002704620361 ,  0.3078362047672272 ,
   0.5779105424880981 ,  0.05820537731051445, -0.13746921718120575,
   0.07705729454755783, -0.25749489665031433,  0.13840684294700623,
   0.04049049317836761, -0.2288392335176468 ,  0.03513109683990479,
   0.9490118622779846 ,  0.09698078781366348],
 [ 0.26215484738349915,  0.3279797434806824 , -0.1821214258670807 ,
   0.2599179148674011 , -0.2773994505405426 , -1.288133978843689  ,
  -0.8377670049667358 , -0.2158239781856537 , -0.02702672965824604,
  -0.4311779737472534 , -0.10110165923833847, -1.2550524473190308 ,
  -0.03288766369223595,  0.6090821027755737 , -0.23045042157173157,
   0.3100987374782562 ,  0.7209067344665527 , -0.2974597215652466 ,
  -0.21925786137580872,  0.5284618139266968 ,  0.26631224155426025,
   0.12110723555088043,  0.7657519578933716 ,  0.28872403502464294,
  -1.4477840662002563 , -0.6837950348854065 , -0.5087943077087402 ,
  -0.20710395276546478,  0.32708245515823364, -1.520239233970642  ,
  -0.07670115679502487, -0.7882928252220154 ,  0.01478500012308359,
   0.12200624495744705,  0.1693054884672165 , -0.2383587658405304 ,
   1.2852602005004883 , -1.3382879495620728 , -0.8405216336250305 ,
  -0.1717347502708435 ,  0.2909988760948181 ,  0.4358137547969818 ,
  -0.6962805390357971 , -0.32426339387893677, -0.4258728623390198 ,
  -0.06216105818748474,  0.4216758906841278 ,  0.3489309549331665 ,
  -0.06522542983293533,  0.48877212405204773,  0.24749556183815002,
  -0.05078417807817459,  0.3691542148590088 ,  0.21687069535255432,
   0.7831358313560486 ,  0.09839662909507751, -0.8578457236289978 ,
   0.9674327969551086 ,  0.3398441672325134 ,  0.7201166749000549 ,
  -0.5812023878097534 , -1.4505212306976318 ,  0.85854572057724   ,
   2.02586030960083   ,  0.27021151781082153, -1.3359805345535278 ,
  -0.08813420683145523,  0.7288573980331421 ,  0.03584799543023109,
  -0.6032285690307617 ,  0.0898912325501442 ,  0.4725363552570343 ,
  -0.43499520421028137, -1.179461121559143  ,  0.7439049482345581 ,
   2.1452438831329346 ,  0.32867535948753357,  0.6857585310935974 ,
   0.273098349571228  , -0.01382563635706902, -0.4486905634403229 ,
  -2.07828688621521   ,  0.7271102666854858 , -0.9769243597984314 ,
   0.735706627368927  , -0.5785189270973206 , -0.3078185021877289 ,
  -0.5549429059028625 ,  0.4486527442932129 ,  0.4171708822250366 ,
   1.2242511510849    , -0.23773044347763062,  2.3587167263031006 ,
   0.2085542231798172 ,  0.280235230922699  ,  0.40512821078300476,
   0.28257137537002563, -0.8127058148384094 , -0.15705469250679016,
   0.20828260481357574, -0.561705470085144  ,  0.3984755575656891 ,
  -0.9572202563285828 , -1.2078571319580078 ,  0.44541850686073303,
   0.4379529356956482 , -1.087142825126648  ,  0.5589079856872559 ,
   1.042076587677002  ,  0.31116795539855957,  0.09174361079931259,
   0.7739187479019165 , -0.8020885586738586 ,  0.1546930968761444 ,
   0.19278112053871155,  0.5728822946548462 , -0.33107542991638184,
   1.022051215171814  , -0.35492393374443054,  0.8942363858222961 ,
   2.054002285003662  ,  0.45777398347854614, -0.23223896324634552,
  -0.0573745109140873 , -1.7887873649597168 ,  0.01997589692473412,
   1.16232430934906   , -1.311823844909668  ],
 [-0.4143116772174835 , -1.171758770942688  , -0.06182357296347618,
  -0.3348008692264557 ,  0.3910558223724365 , -0.4827907085418701 ,
   0.15539421141147614, -0.49263930320739746, -1.9006948471069336 ,
   0.2255190759897232 , -0.45643752813339233,  0.3616240620613098 ,
  -0.5872797966003418 , -0.2778565287590027 , -0.10305042564868927,
   0.19497880339622498, -0.365496426820755  , -0.4759691655635834 ,
   0.07040625065565109, -0.3151741921901703 ,  0.04196970909833908,
  -0.32862746715545654,  0.8550193309783936 , -0.10036630183458328,
   1.6028259992599487 , -0.565169632434845  ,  0.2602385878562927 ,
   0.43441250920295715, -1.5261119604110718 ,  0.44087764620780945,
  -0.04906357079744339, -0.750924825668335  , -0.8262805342674255 ,
   0.572138786315918  ,  0.5528528690338135 , -1.1221290826797485 ,
  -0.596596896648407  , -0.06519916653633118,  0.6170495748519897 ,
   0.1912066638469696 , -0.195023775100708  , -0.2284354418516159 ,
  -1.3484970331192017 ,  0.3380259573459625 ,  0.3229803144931793 ,
   0.49168503284454346, -0.6308889389038086 ,  0.4209596812725067 ,
  -0.35639768838882446,  0.9664263725280762 , -0.3398509621620178 ,
  -0.48095735907554626, -0.10173114389181137,  0.4054293930530548 ,
  -0.1276160478591919 ,  0.5205545425415039 , -0.18947173655033112,
   0.409902423620224  , -0.8703653812408447 ,  0.5925695300102234 ,
  -0.5112121105194092 ,  0.6377424001693726 ,  0.4247864782810211 ,
   1.4166442155838013 , -0.632928192615509  ,  0.3515678346157074 ,
   0.11560263484716415, -0.09749265760183334,  0.0960174947977066 ,
  -0.02045015431940556, -0.7601685523986816 , -0.4520440697669983 ,
   0.11517506092786789, -0.5660363435745239 ,  0.7903608083724976 ,
   0.11759237200021744,  0.1848205327987671 , -0.5370744466781616 ,
   0.8871534466743469 , -0.23763251304626465,  0.05179161578416824,
   0.01972344890236855,  0.6326251029968262 , -0.33779963850975037,
   0.10167674720287323,  0.09411907196044922, -0.1543220579624176 ,
  -0.17951932549476624,  0.7580188512802124 , -0.18820884823799133,
  -0.932800829410553  , -0.4504876136779785 ,  1.1055811643600464 ,
   0.3602336049079895 ,  0.388136088848114  , -0.22831125557422638,
   0.760026752948761  , -0.00066498731030151, -0.5760234594345093 ,
   0.03331534564495087, -0.3285220265388489 , -1.5625832080841064 ,
  -0.00300779868848622, -0.5182173252105713 ,  0.5692046880722046 ,
  -0.06008565425872803,  0.37645208835601807,  0.24474874138832092,
  -0.16423514485359192,  0.39654603600502014,  0.5992639064788818 ,
   0.1510031521320343 , -0.7081282138824463 , -0.3107989430427551 ,
   0.0853552296757698 , -1.069953203201294  ,  0.4351542294025421 ,
  -0.12302248179912567, -1.1166726350784302 ,  0.2516580820083618 ,
   0.00617575692012906,  0.19651371240615845, -1.29715096950531   ,
   0.1720624566078186 , -0.11276883631944656,  0.3653923273086548 ,
  -1.0816839933395386 , -1.3806275129318237 ],
 [-0.3284784257411957 , -1.3673052787780762 ,  0.05741346254944801,
   0.36673831939697266,  0.513277530670166  , -0.36505603790283203,
  -0.32186222076416016,  0.6640291810035706 , -0.7448334693908691 ,
  -0.35798531770706177,  1.2499200105667114 ,  0.4719591736793518 ,
   0.2067210078239441 ,  0.05086897313594818,  0.3219652771949768 ,
   0.06476341187953949,  0.06007370352745056,  0.8525996804237366 ,
  -0.45885202288627625,  0.6888931393623352 ,  0.0345633402466774 ,
   0.5224437117576599 , -0.03435172885656357,  0.618049681186676  ,
  -0.05394186079502106, -0.6949109435081482 , -0.5705768465995789 ,
   0.404802531003952  , -0.4814777076244354 , -0.7139921188354492 ,
   0.4726727306842804 ,  0.42412862181663513,  0.5339813828468323 ,
   0.7852770686149597 ,  0.31950974464416504,  0.4592239558696747 ,
  -1.1672594547271729 ,  0.10325480997562408, -0.02314973063766956,
   0.280052125453949  , -1.2156071662902832 ,  0.37209099531173706,
   1.1646435260772705 , -2.952815294265747  , -0.8943912386894226 ,
   0.03110005334019661, -1.988947868347168  ,  0.4411003589630127 ,
  -0.6814191341400146 ,  0.0299376230686903 , -0.28115156292915344,
  -0.01889959722757339, -0.8103387951850891 , -0.08576459437608719,
   0.7859737277030945 ,  1.1776076555252075 ,  0.29957684874534607,
   0.5260136127471924 , -0.22055138647556305, -0.7413122653961182 ,
  -0.8151498436927795 , -0.4115687906742096 , -0.4990062117576599 ,
   0.4139745533466339 ,  0.3325076103210449 , -0.08198205381631851,
  -0.05978373065590858,  0.7296198010444641 ,  0.35074833035469055,
  -0.6561217904090881 ,  1.129901647567749  ,  0.23236294090747833,
  -0.6590651273727417 , -0.7806830406188965 , -0.11624456942081451,
  -1.0058852434158325 , -1.1638962030410767 ,  0.40744152665138245,
   0.6989760994911194 ,  0.15694265067577362,  0.23103080689907074,
  -0.4926303029060364 ,  0.1607457548379898 ,  1.273442268371582  ,
   0.2032664567232132 , -0.5314817428588867 ,  0.9327617287635803 ,
  -0.7937265634536743 ,  1.14115309715271   ,  0.5399993062019348 ,
   0.22610951960086823, -0.8769865036010742 , -1.518280029296875  ,
   0.00889874063432217,  0.6397690773010254 , -0.6089017987251282 ,
   1.0609582662582397 , -0.945896327495575  , -0.04165759310126305,
   1.041957974433899  , -0.1889193058013916 ,  0.6082801222801208 ,
   0.31789809465408325,  0.6641697287559509 ,  0.2722032070159912 ,
   0.4542325735092163 ,  1.4401670694351196 , -0.37041667103767395,
  -1.2105517387390137 , -0.1291581094264984 , -0.35281410813331604,
   0.09490562230348587,  0.828755259513855  , -0.9883250594139099 ,
  -0.6346707344055176 , -1.4598166942596436 , -0.00002596954072942,
   0.4673534333705902 , -0.5505626797676086 ,  0.30895525217056274,
   1.5737971067428589 ,  0.577815055847168  ,  0.43994805216789246,
  -0.46230781078338623, -0.568741500377655  ,  0.145778626203537  ,
   0.4404089152812958 , -0.3617016077041626 ],
 [ 0.06749063730239868,  1.4933236837387085 , -0.11378654092550278,
  -0.43154487013816833,  0.14273138344287872,  0.611885666847229  ,
  -0.8159951567649841 ,  0.19512005150318146,  0.5374932289123535 ,
   0.1328275054693222 ,  0.3609948456287384 , -0.16845422983169556,
   0.13404008746147156,  3.2360522747039795 , -0.4832911789417267 ,
  -0.03907611221075058, -2.0236074924468994 ,  0.00395692372694612,
   0.3812422752380371 , -0.06017924472689629, -1.0397752523422241 ,
   0.13179147243499756, -2.393376350402832  , -0.3759132921695709 ,
  -0.15764345228672028, -1.4177212715148926 , -0.5573067665100098 ,
   0.6261810064315796 , -0.7209102511405945 , -1.1480082273483276 ,
   0.01516996417194605, -0.09633708745241165, -2.1743111610412598 ,
  -0.08017445355653763,  0.21495850384235382, -0.684880793094635  ,
  -0.04616935551166534,  0.00330000254325569,  1.0344665050506592 ,
  -0.2166241705417633 ,  0.463380366563797  ,  0.7418991923332214 ,
   0.23168276250362396,  0.12252581119537354, -0.2750784158706665 ,
  -1.8844534158706665 ,  2.525268077850342  , -0.8367694616317749 ,
  -0.32304343581199646,  0.45519015192985535,  0.23366686701774597,
  -0.00907861720770597,  0.2602299451828003 , -0.08442574739456177,
   0.24404488503932953,  0.6195038557052612 ,  0.2856069505214691 ,
   0.4149271845817566 , -0.11907603591680527,  0.04013850167393684,
  -0.1405019462108612 ,  1.398974895477295  , -1.087454080581665  ,
   1.1002815961837769 ,  0.06982368975877762, -1.6851131916046143 ,
   0.5248547196388245 , -0.13234864175319672,  1.1342629194259644 ,
  -0.3933805525302887 , -0.4948422908782959 ,  0.464335173368454  ,
   0.40112894773483276,  0.6494962573051453 ,  1.0602850914001465 ,
   0.6812200546264648 ,  1.340741515159607  , -1.0566439628601074 ,
  -0.34018558263778687,  0.15238957107067108,  1.4069095849990845 ,
   1.4119960069656372 , -0.8302893042564392 , -0.2766571044921875 ,
   0.01379762403666973, -0.6171155571937561 , -0.03708354756236076,
  -0.36411523818969727, -0.19734898209571838,  0.4195082485675812 ,
   2.57379150390625   ,  0.1621393859386444 ,  0.48641008138656616,
  -0.9834773540496826 , -1.4465842247009277 , -0.24916981160640717,
  -0.7545262575149536 ,  0.7455684542655945 ,  0.5956292152404785 ,
  -0.3372761011123657 ,  0.3582734167575836 , -0.7864025831222534 ,
   0.10840114951133728,  2.1901674270629883 ,  0.18017719686031342,
   0.13216736912727356,  0.30328619480133057,  0.6475932002067566 ,
   0.2142355591058731 ,  0.3591027855873108 ,  0.156294047832489  ,
   0.8796349763870239 , -0.6096151471138    , -0.5807402729988098 ,
   0.04081980511546135,  1.8608884811401367 , -0.4052734076976776 ,
  -1.017128586769104  , -1.7509818077087402 ,  0.48289260268211365,
   0.6719376444816589 , -0.845742404460907  ,  0.495432585477829  ,
  -0.2980218529701233 ,  1.7944923639297485 , -0.08948639035224915,
   0.47664088010787964,  1.3777233362197876 ],
 [ 0.2532601058483124 , -0.04769653081893921,  0.03558246791362762,
  -0.4107581675052643 , -0.7272307276725769 , -0.07808701694011688,
  -0.08794056624174118,  0.02845456264913082,  0.327385276556015  ,
  -0.04512255266308784,  0.02151757106184959,  0.00116981170140207,
  -0.1421562135219574 , -0.02688685059547424, -0.4030669927597046 ,
   0.00788482930511236,  0.4220052659511566 , -0.021390525624156  ,
  -0.04642851650714874, -0.00811809673905373,  0.01714704930782318,
  -0.02695910446345806,  0.03052672557532787, -0.15122614800930023,
  -0.10938682407140732, -0.18259593844413757, -0.0179906077682972 ,
  -0.154708594083786  ,  0.22656528651714325,  0.0214986652135849 ,
  -0.0756419450044632 ,  0.2814807891845703 , -0.04481211304664612,
  -0.0131560917943716 , -0.10022922605276108,  0.24429365992546082,
   0.22628991305828094, -0.05766192823648453,  0.15067948400974274,
  -0.18702024221420288,  0.00213592080399394, -0.01053461525589228,
   0.13785405457019806,  0.02459620870649815,  0.1794469952583313 ,
  -0.0125166941434145 , -0.03227399289608002, -0.16327525675296783,
  -0.10849957913160324, -0.4881718158721924 , -0.05961402505636215,
  -0.16352076828479767, -0.02875603176653385, -0.0385458804666996 ,
   0.08621960133314133, -0.20616964995861053, -0.15375962853431702,
   0.3882822096347809 ,  0.1566883772611618 , -0.6951715350151062 ,
   0.423269122838974  ,  0.06899935007095337,  0.01475018635392189,
   0.1857958436012268 , -0.11267341673374176, -0.06660261750221252,
  -0.03620966151356697,  0.211484894156456  ,  0.3223535418510437 ,
  -0.029243228957057  , -0.03688178583979607, -0.21814939379692078,
   0.1911439299583435 , -0.03785906359553337, -0.6622485518455505 ,
  -0.20474939048290253, -0.13245625793933868, -0.01079784519970417,
  -0.4576393663883209 ,  0.511748194694519  , -0.2596418559551239 ,
   0.19603349268436432,  0.13874252140522003,  0.13099981844425201,
   0.02654771134257317,  0.06652919948101044, -0.12177985161542892,
   0.00723344087600708,  0.15437766909599304,  0.02260801754891872,
  -0.08718782663345337,  0.1017521321773529 , -0.08935902267694473,
  -0.1716178059577942 , -0.033787801861763  , -0.09715715050697327,
   0.3601861596107483 , -0.08505512773990631, -0.26075196266174316,
   0.05660621821880341, -0.04001926630735397, -0.03792262449860573,
  -0.01889319717884064, -0.05217424035072327,  0.19220604002475739,
  -0.06069114804267883,  0.10556842386722565, -0.07573693245649338,
  -0.08603796362876892, -0.10417517274618149, -0.23693054914474487,
   0.01961428299546242,  0.16397038102149963, -0.0584847629070282 ,
   0.16704466938972473,  0.03157710656523705,  0.02341428026556969,
   0.03171152248978615, -0.00558215659111738,  0.9761714935302734 ,
  -0.1471153199672699 ,  0.0207060668617487 ,  0.01780194602906704,
  -0.3827858865261078 , -0.05708899348974228,  0.3111158013343811 ,
   0.06232776120305061, -0.0540061742067337 ],
 [-0.21287886798381805, -0.14864005148410797,  0.22612731158733368,
  -0.5196180939674377 ,  0.9127188920974731 , -0.19946201145648956,
  -0.7379657030105591 , -0.03083574399352074, -1.0285142660140991 ,
   0.21711115539073944, -0.40538638830184937, -0.01439262926578522,
   0.176864355802536  , -0.4620589017868042 , -0.12077447772026062,
  -0.02547020837664604, -0.362573504447937  ,  0.27673420310020447,
  -0.38898012042045593,  0.369403213262558  ,  0.2370019406080246 ,
  -0.4136238098144531 , -0.00137143197935075, -0.7972182035446167 ,
  -0.2327779233455658 , -0.2805033326148987 , -0.01982279680669308,
  -0.5249035954475403 , -0.3982890844345093 , -0.12068052589893341,
  -0.13649870455265045, -0.6259831190109253 ,  0.10151154547929764,
   0.36916014552116394, -0.07273422181606293,  0.05570075660943985,
  -1.3220621347427368 ,  0.10652891546487808,  0.48487600684165955,
  -0.08618339896202087,  0.6368847489356995 ,  0.04380162432789803,
  -0.3453402817249298 , -0.08558467030525208,  0.6120581030845642 ,
  -0.08350038528442383, -0.7214679718017578 ,  0.17056207358837128,
  -0.11688759922981262, -0.01128592062741518, -0.278486043214798  ,
  -0.24458061158657074, -0.5182365775108337 , -0.11454048752784729,
  -0.48543375730514526, -0.27413496375083923, -0.49567514657974243,
  -0.10151635855436325,  0.14148087799549103,  0.02720729634165764,
   0.29772114753723145, -0.9001094102859497 ,  0.02524700202047825,
  -0.08217927068471909,  0.08931873738765717,  0.15122033655643463,
  -0.1882173866033554 , -0.8277885913848877 , -0.10286818444728851,
  -0.8270562887191772 ,  0.26520058512687683,  0.16586779057979584,
  -0.05489598959684372, -0.07435910403728485, -0.27390924096107483,
  -0.0126367611810565 , -0.27275028824806213,  0.18828953802585602,
  -0.7845439910888672 , -0.18062399327754974, -0.4195769131183624 ,
   0.6266736388206482 ,  0.07952239364385605, -0.2608000934123993 ,
   0.00029919072403572,  0.2934674918651581 , -0.2468952238559723 ,
  -0.2549701929092407 , -0.07583600282669067, -0.20952142775058746,
  -0.09199876338243484,  0.23085980117321014, -0.3304418623447418 ,
   0.5938100218772888 ,  0.33300378918647766,  0.03174198791384697,
  -0.4371633529663086 ,  0.07532774657011032,  0.1925617754459381 ,
   0.06874147802591324, -0.11761873960494995,  0.6399869918823242 ,
  -0.09555911272764206, -0.30572664737701416,  0.24185548722743988,
   0.1178571879863739 , -0.20201805233955383, -0.4740326404571533 ,
   0.02678268775343895, -0.1205870732665062 ,  0.07354681938886642,
  -0.19928593933582306,  0.5215989947319031 ,  0.04262354969978333,
   0.06526452302932739, -1.0290982723236084 ,  0.34945279359817505,
   0.6812535524368286 ,  0.18462885916233063,  0.0503811240196228 ,
  -0.1866794377565384 ,  0.12137667089700699, -0.06514535844326019,
   0.28305867314338684, -0.30061471462249756,  0.06295891851186752,
  -0.37723448872566223,  0.20350490510463715],
 [-0.47704842686653137, -1.3376049995422363 , -0.6407002210617065 ,
  -0.49824240803718567, -1.7684882879257202 ,  0.5008420944213867 ,
  -0.5176920890808105 ,  0.05539647117257118, -0.41475117206573486,
  -0.4622439742088318 ,  0.5363932847976685 , -0.07969659566879272,
   0.35526764392852783,  0.5906193852424622 , -0.24133750796318054,
   0.8331730365753174 ,  0.4104730486869812 , -0.42840391397476196,
  -0.2866593599319458 ,  0.7971112728118896 , -0.5566132664680481 ,
   0.15299083292484283, -0.5375391244888306 ,  0.03593531623482704,
   0.05485048145055771, -0.01019058283418417,  0.07197799533605576,
  -0.551302969455719  , -1.364395022392273  ,  0.6434410810470581 ,
  -0.07982707768678665,  0.6302749514579773 , -1.0410205125808716 ,
   0.8606197834014893 , -0.15595175325870514,  0.28858956694602966,
  -0.37017735838890076, -0.07109721004962921, -0.349249929189682  ,
   0.20964257419109344, -1.603864073753357  , -2.192598581314087  ,
  -1.0231066942214966 ,  0.4739759862422943 ,  0.38101691007614136,
   0.9725108742713928 , -0.8789509534835815 , -0.7135671377182007 ,
   0.46539852023124695, -0.4937533140182495 ,  0.1688210815191269 ,
  -0.31560322642326355,  0.06212111935019493,  0.15393000841140747,
   0.4841989576816559 ,  0.10775058716535568, -0.35254421830177307,
   0.2429094761610031 ,  0.02722537331283092, -1.6644014120101929 ,
  -0.48936033248901367, -0.33261585235595703, -0.9953605532646179 ,
  -2.773205280303955  , -1.6069403886795044 , -1.3472388982772827 ,
  -2.282461166381836  , -0.03097381815314293,  0.6723441481590271 ,
  -0.24325969815254211, -0.3738964796066284 , -0.575071394443512  ,
   0.03725295141339302, -0.28104305267333984,  0.12627434730529785,
   0.19478337466716766,  0.40135136246681213, -1.488695502281189  ,
  -0.4062454104423523 ,  0.22083888947963715, -0.19326837360858917,
   0.6724961996078491 ,  0.33558014035224915, -2.7594451904296875 ,
   0.6283202171325684 ,  0.38686829805374146,  0.34933149814605713,
  -0.02754801698029041, -0.8582656979560852 , -0.2612612545490265 ,
  -1.3717974424362183 ,  0.05876703560352325, -0.2195092886686325 ,
   0.75853031873703   , -0.2610171139240265 , -0.08126349002122879,
  -0.7427797913551331 , -1.8954052925109863 , -0.28557437658309937,
  -0.02543542720377445, -0.32232120633125305,  0.14244084060192108,
  -0.4943452477455139 , -1.339045763015747  , -0.3685169517993927 ,
   0.3809325397014618 ,  0.2240031510591507 , -1.2577085494995117 ,
  -0.974551260471344  ,  1.422491192817688  , -0.2500573992729187 ,
   0.8509216904640198 , -0.8548673987388611 , -0.4118621349334717 ,
  -0.8303396701812744 , -0.414936363697052  ,  0.4098062217235565 ,
   0.945700466632843  , -0.30611324310302734,  0.11810850352048874,
  -0.6053904294967651 ,  0.00062306592008099,  1.369088888168335  ,
   0.00777319306507707, -0.00703683448955417,  0.5100548267364502 ,
  -0.5181061625480652 , -1.1593934297561646 ],
 [-0.07291398197412491, -1.377002477645874  , -0.1585504561662674 ,
  -0.7885609269142151 ,  0.43387120962142944, -0.4398837387561798 ,
   0.27388083934783936, -0.09552200883626938, -0.2150590866804123 ,
  -0.312325656414032  , -0.05518421903252602,  0.1076304093003273 ,
  -0.5942062735557556 , -0.5833638310432434 , -1.0878373384475708 ,
  -0.02387993782758713, -0.7892497777938843 , -0.562070369720459  ,
   1.15193510055542   ,  0.5956811904907227 , -0.05588631704449654,
  -0.01177171710878611,  2.396367073059082  ,  0.9294558167457581 ,
   0.3515000641345978 , -0.6115239858627319 , -0.8935802578926086 ,
   0.5490874648094177 , -0.3075326979160309 ,  0.2911611795425415 ,
  -0.09173139184713364,  1.4298309087753296 ,  0.38849955797195435,
  -0.3804137408733368 ,  0.24330465495586395, -1.2071669101715088 ,
   0.5155640244483948 ,  0.04414201900362968,  0.902572751045227  ,
   0.19763566553592682,  0.5822324156761169 , -0.18468832969665527,
   0.07996372133493423,  0.42111173272132874,  0.00780697353184223,
  -0.5899930596351624 ,  1.0930325984954834 , -0.45132210850715637,
  -0.3141370713710785 ,  0.9312444925308228 ,  0.6805471181869507 ,
  -0.4010215699672699 ,  0.8746650815010071 , -0.06377877295017242,
  -0.6179181933403015 , -0.09450045228004456,  0.16112610697746277,
  -0.1190311536192894 ,  0.13659068942070007, -0.7664924263954163 ,
  -0.03256497904658318, -0.00268015102483332,  1.7461113929748535 ,
  -0.13240733742713928,  0.27391552925109863,  0.6652893424034119 ,
   0.5496009588241577 , -1.8428319692611694 ,  1.6764720678329468 ,
  -0.6104220151901245 ,  0.3812505006790161 ,  0.335407018661499  ,
   0.29228299856185913,  0.11173282563686371, -0.08009035885334015,
  -0.3872179687023163 , -1.209107518196106  ,  0.6797518730163574 ,
  -0.7928115129470825 , -0.5305771827697754 , -0.11379984021186829,
  -0.893733024597168  ,  0.2751481235027313 ,  0.7251999378204346 ,
   0.16547361016273499,  0.09505446255207062, -0.02443554438650608,
  -0.3634794354438782 ,  0.12033841013908386,  0.02699417993426323,
   1.3553351163864136 ,  0.06292298436164856, -1.1007543802261353 ,
   0.420911967754364  , -0.4288713335990906 , -0.21191950142383575,
  -0.04499585926532745,  0.9817571640014648 ,  0.9565396904945374 ,
   0.04668548330664635,  0.557672917842865  , -0.34419775009155273,
   0.1053631380200386 , -0.3005209267139435 ,  0.2096238136291504 ,
  -1.1012604236602783 ,  1.1447831392288208 , -0.5089138746261597 ,
   0.41642457246780396,  0.05212682113051414,  0.7138404250144958 ,
  -0.0885700136423111 ,  0.00096142169786617, -0.758341908454895  ,
   0.21812710165977478,  0.3876178562641144 ,  0.4250986874103546 ,
  -0.13486206531524658,  0.1553371697664261 ,  0.00742053613066673,
  -0.4418538808822632 , -0.37452587485313416,  0.5983104109764099 ,
   0.34934791922569275, -0.46915969252586365, -0.4749753475189209 ,
   0.0224581565707922 , -1.199047565460205  ],
 [ 0.5846970677375793 , -0.2988698184490204 ,  0.02111935615539551,
  -0.17188507318496704, -0.1266307532787323 , -0.21861934661865234,
   0.13476680219173431,  0.04671595990657806, -0.42580118775367737,
   0.14256978034973145, -0.02458340674638748, -0.04187487438321114,
   0.43123915791511536, -0.383757621049881  , -0.09325098246335983,
  -0.02532735094428062,  0.47877880930900574,  0.04160352051258087,
  -0.254158616065979  ,  0.1279907077550888 , -0.09969189018011093,
   0.09765072911977768,  0.01540208887308836, -0.44017457962036133,
  -1.8316376209259033 , -0.1097310408949852 ,  0.2919914722442627 ,
  -0.5971450209617615 , -0.6085739731788635 ,  0.27797144651412964,
   0.03521935269236565,  0.23272870481014252, -0.20220039784908295,
  -0.7344580292701721 ,  0.06023825705051422, -0.06190180778503418,
   0.3580796718597412 ,  0.02389762923121452,  0.15365305542945862,
   0.19813717901706696, -0.1144009381532669 , -0.10901541262865067,
  -0.2187950313091278 , -0.3615666925907135 , -0.5103007555007935 ,
  -0.15120236575603485, -0.4715532660484314 , -0.07931272685527802,
   0.4356040358543396 , -1.0191235542297363 , -0.8011466860771179 ,
   0.04576823487877846,  0.26411136984825134,  0.24182112514972687,
   0.28176233172416687, -0.06450454890727997,  0.08588364720344543,
  -0.450229674577713  ,  0.747319757938385  ,  0.46112802624702454,
  -0.06021561101078987, -0.3650638461112976 , -0.31207776069641113,
   0.3021742105484009 ,  0.02037076465785503, -0.01683834567666054,
  -0.45257338881492615, -0.5628056526184082 , -1.8168880939483643 ,
   0.44447699189186096,  0.41465672850608826,  0.09744521230459213,
   0.04086288437247276,  0.03422193229198456,  0.7417994141578674 ,
  -0.06871174275875092, -0.1406508833169937 ,  0.13419389724731445,
   0.3703237771987915 , -0.4327087700366974 ,  0.11851654201745987,
  -0.18976742029190063, -0.3091166317462921 , -0.15104274451732635,
   0.00508092204108834, -0.13749194145202637,  0.3756115734577179 ,
   0.21800820529460907, -0.08087994158267975,  0.1795920580625534 ,
  -0.0377228632569313 , -0.1842471808195114 ,  0.7181360125541687 ,
  -0.696593165397644  ,  0.35339075326919556, -0.00554423872381449,
  -0.508138120174408  ,  0.1196541041135788 , -0.6140867471694946 ,
  -0.06612925231456757,  0.3936297595500946 , -0.1506146490573883 ,
   0.0088599743321538 , -0.3159039616584778 ,  0.14628757536411285,
  -0.15437884628772736,  0.00944342836737633, -0.8605562448501587 ,
  -0.43254178762435913, -0.0320391058921814 ,  0.1287698745727539 ,
   0.04335370659828186,  0.06683138757944107,  0.21372878551483154,
  -0.00445289071649313, -0.23009592294692993, -0.05644318461418152,
   0.1594981998205185 , -0.03656037151813507,  0.4375375509262085 ,
   0.3673657476902008 , -0.23530006408691406, -0.5020542740821838 ,
   0.19975346326828003,  0.4239859879016876 ,  0.16339713335037231,
   0.09018062800168991, -0.232306107878685  ],
 [ 0.17032353579998016,  0.514970600605011  , -0.03368423134088516,
  -0.22868742048740387,  0.5469573736190796 , -0.35891193151474   ,
   1.3027199506759644 , -0.04444060474634171, -0.9799546003341675 ,
   0.19126252830028534, -0.2702447175979614 , -0.05687055364251137,
   0.1505875438451767 , -0.5600903630256653 ,  0.31879621744155884,
   0.04317788779735565,  0.5591714382171631 ,  0.15496110916137695,
  -0.19948656857013702, -0.5436376333236694 ,  0.04637650772929192,
  -0.00300647038966417, -1.0750515460968018 ,  0.8227331042289734 ,
   0.05155855417251587, -0.38522058725357056, -0.49785491824150085,
  -0.3051188588142395 , -1.0882956981658936 ,  0.6146947741508484 ,
   0.29971230030059814, -0.3075425922870636 ,  0.12108730524778366,
  -1.1886268854141235 , -0.0360245406627655 ,  0.5680094957351685 ,
  -0.9728354811668396 ,  0.08124697208404541, -0.4121675491333008 ,
   0.23690858483314514,  0.20549896359443665,  0.18049736320972443,
   0.16464528441429138,  0.670775830745697  ,  0.5091659426689148 ,
  -0.23042313754558563,  0.44759953022003174,  0.01176529005169868,
  -0.2660188674926758 , -0.10614919662475586,  0.41749393939971924,
   0.13266584277153015, -0.8892121911048889 ,  0.04029488563537598,
   0.4750925898551941 , -0.05707381665706635, -0.20222048461437225,
   0.3129526972770691 , -0.24924227595329285, -0.0682578906416893 ,
   0.10215194523334503, -0.27274078130722046, -0.5204648375511169 ,
  -1.0786086320877075 ,  0.6854169368743896 , -0.8501782417297363 ,
   0.4391651451587677 , -0.43310168385505676, -2.0084242820739746 ,
   0.6575295925140381 , -0.1443312019109726 , -0.18265676498413086,
  -0.07139674574136734,  0.02588115818798542,  0.1188519075512886 ,
  -1.951776385307312  , -1.9202271699905396 ,  0.1686837375164032 ,
   0.17945967614650726,  0.03490980342030525,  0.16571183502674103,
  -0.2983553111553192 , -0.10878124088048935,  0.8139600157737732 ,
   0.04038183763623238,  0.01137363165616989, -0.37457752227783203,
   0.08848502486944199, -0.3255113959312439 ,  0.2575438916683197 ,
   0.41221025586128235,  0.7391244173049927 , -0.4918111562728882 ,
   0.3437080979347229 ,  0.06099365279078484,  0.5461560487747192 ,
  -1.1920613050460815 ,  0.982710063457489  ,  0.12296421825885773,
  -0.00107701041270047,  0.21800750494003296, -0.4090673625469208 ,
   0.03512376174330711, -0.06538443267345428,  0.6318705081939697 ,
   0.3324780762195587 , -0.2632730305194855 , -0.2096218764781952 ,
   0.47769489884376526, -0.45781034231185913,  0.3083452880382538 ,
  -0.0515499971807003 ,  0.1082024797797203 , -0.3814891278743744 ,
  -0.7867814302444458 ,  0.1795034408569336 , -0.09224026650190353,
   0.16098135709762573,  0.03124540112912655,  0.438055157661438  ,
   0.3583992123603821 , -0.11765694618225098, -0.17305144667625427,
   0.1876569390296936 , -2.2353622913360596 , -0.4052116274833679 ,
   0.45650818943977356,  1.1806730031967163 ],
 [ 0.5280448794364929 , -0.07347579300403595, -0.0684143677353859 ,
   0.3116627335548401 , -0.00431699119508266, -1.3257519006729126 ,
  -1.4110503196716309 , -0.2755945324897766 , -0.2821820378303528 ,
  -0.2502880096435547 ,  0.8519651889801025 , -0.00663725286722183,
   0.05087864398956299,  0.1667928248643875 , -0.2580621838569641 ,
  -0.06981649249792099,  0.7148703932762146 ,  0.43358343839645386,
  -2.06016206741333   ,  0.8290398716926575 , -0.580467939376831  ,
  -0.5406213402748108 ,  0.07283581793308258,  0.2558136284351349 ,
   0.2251695692539215 , -0.4910969138145447 , -0.8341315388679504 ,
  -0.5164075493812561 ,  0.40542998909950256,  0.7660451531410217 ,
   0.3036234974861145 , -0.3320496678352356 , -0.6995378732681274 ,
  -1.5756621360778809 , -0.4120805859565735 , -0.9916476607322693 ,
  -1.7488291263580322 , -0.4032369554042816 ,  0.27768778800964355,
  -0.11987840384244919,  0.5356079339981079 ,  0.02472531981766224,
  -0.27898067235946655, -1.2736085653305054 , -0.6271705627441406 ,
  -0.4561569094657898 , -0.5376186966896057 , -0.08690673857927322,
   0.0294732041656971 ,  0.8048197031021118 , -0.5157446265220642 ,
  -0.38318899273872375, -1.80411958694458   ,  0.577610433101654  ,
   0.13316850364208221, -0.21658246219158173,  1.0317890644073486 ,
   0.5148410797119141 , -0.0908818244934082 ,  0.6186405420303345 ,
  -0.06271452456712723, -0.6044740676879883 , -0.4167484939098358 ,
   0.05747437849640846, -0.6538351774215698 , -0.09880001842975616,
  -0.6325439810752869 , -0.0810142382979393 ,  0.617301344871521  ,
  -1.3523670434951782 , -0.3527032732963562 ,  0.1780809760093689 ,
  -0.3286270797252655 , -0.8924188017845154 ,  0.644767701625824  ,
  -0.773219883441925  ,  0.02121751941740513, -0.3293742835521698 ,
  -0.5924673676490784 , -0.12386376410722733, -0.5023192167282104 ,
  -1.9398695230484009 , -0.18002960085868835, -0.1382763534784317 ,
   1.1041799783706665 , -0.2864566743373871 , -0.20375590026378632,
  -0.12411247938871384, -0.7314172983169556 ,  0.3953531086444855 ,
  -0.6387107968330383 , -0.27508166432380676,  0.6173186898231506 ,
  -0.21178850531578064,  0.06843257695436478, -0.3114449083805084 ,
   0.3085598945617676 , -1.1542998552322388 ,  0.48384812474250793,
   0.4396107494831085 ,  0.23796962201595306, -0.5568219423294067 ,
  -0.31881511211395264, -0.20629264414310455, -0.01223208755254745,
  -0.07857675850391388, -0.41941601037979126, -0.3314116895198822 ,
   0.6843993067741394 , -0.13376642763614655,  0.18749387562274933,
   0.7094392776489258 , -1.590998888015747  ,  0.11751735210418701,
  -0.16524410247802734,  1.051007628440857  ,  0.01632772944867611,
  -0.11986273527145386, -0.08101126551628113,  0.3594614565372467 ,
   0.00304776849225163, -1.1595149040222168 ,  0.18210838735103607,
   0.3540153205394745 , -1.705505609512329  ,  0.5149840116500854 ,
  -1.3897770643234253 , -0.7107161283493042 ],
 [-1.7726575136184692 , -0.2691481411457062 , -0.1694306582212448 ,
  -0.13049057126045227, -0.469547301530838  , -2.691998243331909  ,
  -0.5962007641792297 ,  0.09110991656780243,  0.08878283947706223,
  -0.3781546950340271 , -0.6105377674102783 , -0.2952709496021271 ,
   0.04524576291441917,  0.6899703741073608 , -0.1680944561958313 ,
  -0.0542193204164505 , -0.6625218987464905 , -0.9939260482788086 ,
  -0.3935885727405548 , -0.2501091957092285 ,  0.7247946858406067 ,
  -0.6949055194854736 , -0.26085883378982544, -0.09473896771669388,
   0.9947864413261414 ,  1.3932849168777466 ,  0.5280370712280273 ,
   1.2577844858169556 ,  0.4649607539176941 ,  0.9143579006195068 ,
   0.233046293258667  ,  0.3223167657852173 , -0.9878541231155396 ,
   0.17536716163158417,  0.13817289471626282, -0.4915284514427185 ,
   0.772255003452301  , -0.15958085656166077,  0.6004563570022583 ,
  -0.39386552572250366, -0.41507846117019653,  0.0117486622184515 ,
  -0.07031102478504181, -0.2438059151172638 ,  0.15956172347068787,
   0.5120207071304321 , -0.5874149799346924 , -0.01074767671525478,
  -0.1163371130824089 ,  0.7010785341262817 , -0.306173712015152  ,
   0.0976417064666748 ,  0.10115247964859009, -0.13277938961982727,
   0.6989116668701172 , -0.9969587922096252 , -0.1264379322528839 ,
   0.4905230700969696 ,  0.14809225499629974,  0.4787884056568146 ,
   0.527459442615509  ,  0.29313886165618896,  0.5237920880317688 ,
   0.4145274758338928 ,  1.0415388345718384 ,  0.630243182182312  ,
  -0.25985395908355713,  0.7776721119880676 ,  0.9489215016365051 ,
  -0.5041608810424805 ,  0.5098972320556641 , -0.7129683494567871 ,
   0.14118821918964386,  0.14237824082374573,  0.1648755967617035 ,
   0.11735709011554718,  1.1704171895980835 ,  0.5147181153297424 ,
   0.14048703014850616, -0.09458662569522858, -0.6294581294059753 ,
   0.9253484010696411 , -0.08693211525678635,  0.4127802848815918 ,
   0.02904525958001614,  0.05148309469223022, -0.27514779567718506,
  -0.2121702879667282 , -0.03667062148451805,  0.5083762407302856 ,
   0.01113584823906422,  0.33208370208740234,  0.30009785294532776,
   0.3431112468242645 , -0.1449335217475891 ,  0.02329961396753788,
  -1.6212631464004517 ,  0.32375645637512207, -0.02970910258591175,
  -0.35226917266845703,  0.2714460790157318 ,  0.35418736934661865,
  -0.09377213567495346,  0.7905773520469666 ,  0.3647300601005554 ,
  -0.4383452832698822 , -0.1317710280418396 ,  0.791897177696228  ,
   0.6771985292434692 ,  0.28816458582878113,  0.43144363164901733,
   0.220401331782341  ,  0.16773471236228943, -1.3041452169418335 ,
  -0.01345804426819086, -0.4971425235271454 ,  0.2955849766731262 ,
  -0.1776323765516281 ,  0.20683825016021729, -0.49194878339767456,
  -0.4978523552417755 ,  1.1738027334213257 ,  0.21421681344509125,
  -0.9616214632987976 ,  0.05336567386984825, -2.295949935913086  ,
  -1.0590929985046387 , -1.0461689233779907 ],
 [ 0.7019758820533752 ,  0.355405330657959  , -0.17126131057739258,
   0.5192331075668335 , -0.6459457874298096 ,  0.22408747673034668,
  -0.2653615176677704 ,  0.23495464026927948,  0.7331622242927551 ,
  -0.06298215687274933, -0.443172812461853  , -0.12033779174089432,
  -0.10082376003265381, -0.16880753636360168, -0.29671695828437805,
   0.03794743120670319,  1.3636140823364258 , -0.14183273911476135,
   0.14969515800476074, -0.09137425571680069, -0.6243528127670288 ,
  -0.06965097039937973,  0.3427574336528778 , -0.02284753136336803,
  -0.16829583048820496, -0.5617521405220032 , -0.14618779718875885,
   0.4002726078033447 , -0.5961551666259766 ,  0.5087488889694214 ,
  -0.04178201034665108, -1.4822862148284912 , -0.7065480351448059 ,
   1.3355302810668945 ,  0.31863537430763245,  0.271511435508728  ,
  -0.06224849447607994, -0.0891144722700119 ,  0.2258855253458023 ,
  -0.4171961545944214 ,  0.32627764344215393, -0.7140527367591858 ,
  -0.3750801384449005 , -1.603029489517212  ,  0.05531106516718864,
   0.13475529849529266,  0.4693758487701416 , -0.1771765500307083 ,
  -0.2744646966457367 ,  0.6586076617240906 , -0.43134456872940063,
   0.20658724009990692,  0.24380503594875336, -0.20608288049697876,
  -0.4591272175312042 ,  0.24820394814014435, -0.1494225561618805 ,
   0.14812816679477692,  0.19412222504615784, -0.999643087387085  ,
  -0.01369157619774342,  0.19343382120132446, -0.5067955255508423 ,
   0.28825339674949646, -0.4523475170135498 , -1.2084202766418457 ,
  -0.16123512387275696,  0.05483971536159515, -0.09137865155935287,
  -0.8002797365188599 , -0.5016704797744751 ,  0.2345127910375595 ,
  -0.8459087610244751 , -0.20984403789043427, -0.9867942333221436 ,
  -0.21679532527923584, -0.31921514868736267,  0.2184334546327591 ,
   0.6496071815490723 , -0.43633246421813965, -0.09519065916538239,
   0.34315213561058044,  0.20023280382156372, -0.878812313079834  ,
   0.131316676735878  , -0.7921657562255859 ,  0.07051905244588852,
   0.4559619128704071 ,  0.02437875606119633,  0.12141568958759308,
   0.09884054213762283, -0.4437023103237152 , -0.3956204950809479 ,
  -0.24506552517414093,  0.25004711747169495, -0.17198485136032104,
   0.5512797832489014 , -0.749469518661499  , -0.11667057871818542,
  -0.1030636876821518 , -0.01181891560554504,  0.5867015719413757 ,
  -0.00048799643991515,  0.5423693656921387 ,  0.3019140958786011 ,
   0.600660502910614  ,  1.0329822301864624 , -0.24837170541286469,
  -0.12158187478780746,  0.0418144203722477 ,  0.16121715307235718,
  -0.4755629599094391 , -0.5038875341415405 , -0.4740484952926636 ,
   0.02326770685613155, -0.19048801064491272, -0.06137893348932266,
  -0.5946635007858276 , -0.5145929455757141 , -0.30278828740119934,
  -0.00977559760212898,  0.088527150452137  ,  0.00682957330718637,
  -0.09779790788888931,  0.35220852494239807,  0.09286858141422272,
  -0.9237993359565735 ,  0.01117025129497051],
 [ 0.3283196985721588 ,  0.3021646738052368 ,  0.13607671856880188,
   0.2737627625465393 ,  0.5008922815322876 ,  0.3073824644088745 ,
   0.4857780635356903 ,  0.45642808079719543,  0.40385594964027405,
  -0.28225913643836975,  0.0836879163980484 , -0.2501363754272461 ,
  -0.1601298600435257 ,  0.1633238047361374 , -0.43391895294189453,
  -0.2735064625740051 ,  0.5480409264564514 ,  0.42344871163368225,
   0.01686890050768852, -0.21613603830337524,  0.00712270010262728,
   0.19378551840782166,  0.3833421766757965 , -0.3133731782436371 ,
   0.36172032356262207, -0.02129559777677059, -0.04183821007609367,
   0.5593315362930298 ,  0.2535759508609772 , -0.09227707982063293,
  -0.01443897280842066, -0.5618512034416199 ,  0.27088049054145813,
  -0.42683646082878113, -0.7934020757675171 , -0.19785870611667633,
  -0.06343431770801544, -0.22134147584438324,  0.13300850987434387,
   0.20253683626651764, -0.00764887314289808,  0.20302528142929077,
   0.5154922008514404 ,  0.7076236605644226 , -1.002328872680664  ,
   0.07482092082500458,  0.18016214668750763, -0.04311984404921532,
   1.036332607269287  , -0.8652228713035583 , -0.5208916664123535 ,
  -0.6220874786376953 , -0.14521647989749908,  0.5814428329467773 ,
  -0.5640690922737122 , -0.3223153054714203 ,  0.06239605695009232,
  -0.3635962903499603 , -0.1821596473455429 ,  0.5121816992759705 ,
   0.16058161854743958,  0.26036980748176575,  0.3371928334236145 ,
   0.638741135597229  ,  0.17926909029483795,  0.5626658797264099 ,
  -0.4299207925796509 , -0.36809903383255005, -0.14386531710624695,
   0.2492314875125885 ,  0.01276299450546503, -0.43917667865753174,
   0.36584895849227905,  1.0268505811691284 ,  1.0360881090164185 ,
   0.19631733000278473,  0.01567558199167252, -0.01449938025325537,
  -0.2091754972934723 , -1.123172402381897  ,  0.32387080788612366,
  -1.101762294769287  ,  0.4071713387966156 ,  0.49231910705566406,
  -0.02190616726875305,  0.6182449460029602 , -0.04952578991651535,
  -0.5937742590904236 , -0.3910658657550812 ,  0.1628832072019577 ,
  -0.44010990858078003, -0.6034581661224365 , -0.5844091176986694 ,
  -0.47849664092063904, -0.05541353672742844,  0.7207342982292175 ,
  -0.31953170895576477, -1.3150544166564941 , -0.3216323256492615 ,
   0.02201163396239281, -0.37800031900405884,  0.43333104252815247,
  -0.12931035459041595,  0.2658868730068207 ,  1.0405311584472656 ,
  -0.0348040945827961 , -0.789675772190094  ,  0.6547136902809143 ,
   0.674287736415863  , -0.5266242027282715 , -0.3943887948989868 ,
  -0.11230475455522537, -0.12872403860092163, -0.10883957147598267,
   0.08255601674318314,  0.38273265957832336,  0.08739876002073288,
  -0.2495286911725998 , -0.561355710029602  , -0.15633705258369446,
   0.04357481747865677,  0.27651700377464294, -0.00716554978862405,
  -0.7076401710510254 , -0.07090356945991516, -0.03056907467544079,
   0.3608744740486145 ,  0.45570623874664307],
 [-0.7329556941986084 , -0.6632335782051086 , -0.3125593960285187 ,
  -0.2820749282836914 , -0.32076603174209595, -0.26723283529281616,
   0.41590383648872375, -0.4973872900009155 , -0.7297731041908264 ,
   0.5303969979286194 ,  0.4574744403362274 , -0.7108802795410156 ,
   0.08102501183748245,  0.238010436296463  , -0.06097051501274109,
   0.3569394648075104 , -0.03463205322623253,  0.0283182580024004 ,
  -0.257160484790802  ,  0.24412256479263306, -0.9021131992340088 ,
   0.06253596395254135, -0.7325724363327026 , -0.4984360337257385 ,
  -0.14768335223197937, -0.11051741242408752, -0.3789306879043579 ,
   0.4253052771091461 , -0.7125464081764221 ,  0.41141921281814575,
  -0.28939804434776306, -0.09672261774539948, -1.2744207382202148 ,
  -0.21179498732089996,  0.4939880669116974 ,  0.671769917011261  ,
   0.7681811451911926 ,  0.07566969096660614, -0.603344738483429  ,
  -0.9654460549354553 , -0.20829857885837555, -0.4984061121940613 ,
   0.26078906655311584, -2.0544962882995605 , -0.7761266231536865 ,
  -1.9682716131210327 , -0.5307148098945618 , -0.3972969353199005 ,
  -0.40158459544181824, -0.5592023730278015 , -0.2702617049217224 ,
  -0.4129045903682709 , -1.1605850458145142 ,  0.00630917446687818,
  -0.8161484599113464 , -0.7850438356399536 , -0.7027692198753357 ,
  -0.11118195950984955, -0.856281578540802  , -0.28969141840934753,
  -0.41679060459136963, -0.1426360160112381 , -0.6651594042778015 ,
   0.1812688410282135 , -0.03434493020176888, -0.7524297833442688 ,
   0.4285370111465454 ,  0.21626821160316467, -0.8487392067909241 ,
   0.04528241977095604, -1.5571209192276    , -0.17355293035507202,
  -0.4307829737663269 ,  0.078310027718544  , -0.5692923069000244 ,
  -0.09534677863121033, -0.07441740483045578, -0.4314744472503662 ,
   0.11142195016145706, -0.6531572341918945 ,  0.35016965866088867,
   0.00122918491251767, -1.672013521194458  , -2.617969274520874  ,
   0.1763598918914795 , -1.7721556425094604 , -0.2910667359828949 ,
  -0.6121373772621155 ,  0.3684140145778656 , -1.0867538452148438 ,
  -1.5169137716293335 ,  0.00800227653235197, -0.0252370648086071 ,
  -0.06692192703485489, -0.6957817077636719 ,  0.5727112889289856 ,
  -0.1379324197769165 , -0.2103426605463028 , -0.00314005324617028,
  -0.30652114748954773, -0.7263919115066528 , -0.23450852930545807,
  -0.04581453278660774, -0.41247496008872986, -0.40461236238479614,
  -1.8947776556015015 , -0.18228957056999207, -0.07394267618656158,
  -0.15386582911014557,  0.5841464996337891 ,  0.06321296095848083,
  -0.5747479796409607 , -0.2448980212211609 , -1.1548303365707397 ,
  -0.3413254916667938 , -1.0439255237579346 , -0.9879490733146667 ,
  -0.6902523040771484 , -0.8931971192359924 , -0.3232572674751282 ,
  -0.5287513732910156 ,  0.6695201992988586 ,  0.6422811150550842 ,
   0.01861859485507011, -0.1731588989496231 ,  0.34934201836586   ,
   0.5488433241844177 , -0.4774809181690216 ],
 [ 0.3067222833633423 , -0.07483548671007156,  0.06289251893758774,
  -0.26498883962631226, -0.24782374501228333,  0.02000033296644688,
  -0.43758270144462585,  0.24249477684497833, -0.28783297538757324,
   0.1349794864654541 , -0.05081330984830856,  0.03498093038797379,
  -0.06716341525316238, -0.1409873515367508 , -0.46067866683006287,
   0.09132720530033112,  0.02272014692425728,  0.2901363670825958 ,
   0.07186708599328995,  0.1511509269475937 , -0.0223385151475668 ,
  -0.01861985214054585,  0.12273117899894714, -0.00243144179694355,
  -0.32396233081817627, -0.0680575892329216 , -0.12861281633377075,
   0.3320714235305786 ,  0.2042994499206543 ,  0.04604452103376389,
   0.06640887260437012,  0.18052276968955994, -0.03910000249743462,
  -0.07091058790683746, -0.18174666166305542,  0.11944388598203659,
   0.1452886164188385 ,  0.02049726061522961,  0.03904597461223602,
   0.04071370884776115,  0.03548097610473633, -0.05670469254255295,
   0.04180465266108513,  0.0027101484593004 , -0.03845662251114845,
  -0.25991949439048767, -0.1153310090303421 , -0.20541201531887054,
  -0.3213158845901489 ,  0.7430609464645386 ,  0.5094743371009827 ,
   0.07231171429157257,  0.15098635852336884, -0.47543150186538696,
   0.00020850468717981, -0.16484305262565613, -0.03747421875596046,
  -0.08357443660497665, -0.05892293155193329, -0.6519482731819153 ,
   0.19949278235435486,  0.0932261273264885 ,  0.05656022951006889,
  -0.16884149610996246, -0.17612774670124054, -0.05780842900276184,
  -0.02034885063767433,  0.1607321798801422 ,  0.41149336099624634,
  -0.11934658139944077, -0.01363808661699295, -0.665932297706604  ,
  -0.5042834281921387 ,  0.24449820816516876, -0.4706798195838928 ,
  -0.00042733139707707, -0.05187994241714478, -0.06107654795050621,
  -0.44093722105026245,  0.3258298337459564 ,  0.08879704028367996,
   0.12589804828166962,  0.1676977127790451 ,  0.03616541624069214,
  -0.02238433994352818,  0.03881840780377388, -0.18834038078784943,
  -0.08506670594215393, -0.48525041341781616,  0.08529088646173477,
  -0.02129214815795422, -0.18880417943000793,  0.02585143223404884,
  -0.15464039146900177,  0.17884461581707   ,  0.13208582997322083,
  -0.258698970079422  , -0.3197292387485504 , -0.16706638038158417,
   0.3030097186565399 ,  0.0149970380589366 , -0.2621043026447296 ,
  -0.13187478482723236,  0.00313400803133845,  0.2412015050649643 ,
   0.04760420694947243, -0.08121602237224579, -0.10652793943881989,
  -0.05810300260782242, -0.20465567708015442, -0.15886636078357697,
  -0.01365936174988747, -0.13491572439670563, -0.03379399329423904,
  -0.18829399347305298, -0.06806536763906479, -0.3002590537071228 ,
  -0.06856481730937958, -0.04998371377587318,  0.39263010025024414,
  -0.1559097170829773 , -0.17103292047977448, -0.16302120685577393,
   0.11623401194810867, -0.273652046918869  , -0.0667792335152626 ,
   0.26349321007728577,  0.06260943412780762],
 [-0.6775289177894592 , -0.5211640000343323 ,  0.4395242929458618 ,
  -0.07461784780025482,  0.21867968142032623, -1.3728737831115723 ,
  -0.8930982947349548 , -0.43538782000541687, -0.33334386348724365,
   0.12354525923728943, -0.43039873242378235, -0.08734714239835739,
  -0.35498350858688354, -1.5585427284240723 ,  0.15998674929141998,
  -0.2582615613937378 , -0.16673710942268372,  0.5293275117874146 ,
  -0.7599371075630188 , -0.22636757791042328, -0.7907827496528625 ,
   0.10300777107477188, -0.5758059024810791 , -0.38149717450141907,
  -0.28118693828582764, -0.06923455744981766, -0.26839667558670044,
  -1.148565411567688  ,  0.12387216836214066, -0.5708034634590149 ,
  -0.14246326684951782, -0.19666972756385803, -1.443289041519165  ,
  -0.930590033531189  , -0.0534307137131691 , -0.5064035654067993 ,
  -1.4389277696609497 ,  0.41810545325279236, -0.5751070380210876 ,
  -0.0605507418513298 , -0.996525228023529  , -0.05457384139299393,
  -2.8843202590942383 , -2.6002578735351562 , -0.7343558669090271 ,
  -0.6022823452949524 , -0.5514368414878845 , -0.303459107875824  ,
   0.44289085268974304, -0.19718404114246368, -0.08058127015829086,
  -0.28188854455947876, -0.48351386189460754, -0.15552984178066254,
  -0.10237887501716614, -0.5334529280662537 , -1.7411930561065674 ,
  -0.05121063441038132,  0.2834867835044861 , -0.15871530771255493,
  -0.20587040483951569, -0.07110338658094406, -0.37419360876083374,
  -0.5877062082290649 , -0.364940345287323  , -0.6642944812774658 ,
   0.26001429557800293, -0.134791299700737  , -0.5835133790969849 ,
  -0.4748755395412445 ,  0.6016867756843567 , -0.18358692526817322,
  -0.22353243827819824, -2.490288019180298  ,  0.04264335334300995,
  -0.00823545083403587, -0.6397252678871155 , -0.19815762341022491,
  -0.4163905382156372 , -0.21121850609779358, -0.8279558420181274 ,
   0.26726430654525757, -0.4011084735393524 , -1.2197980880737305 ,
  -0.3160150647163391 , -0.3218092620372772 , -0.00049655523616821,
   0.19178634881973267,  0.2570417523384094 , -0.09378480166196823,
   0.38747018575668335, -0.8967932462692261 , -0.5523163080215454 ,
   0.19346512854099274, -0.39362916350364685,  0.30327099561691284,
  -0.0708267092704773 ,  0.07310666143894196,  0.102078877389431  ,
   0.05007536709308624,  1.5645751953125    , -0.11244139820337296,
   0.5659919381141663 , -0.5457135438919067 , -1.5649482011795044 ,
  -0.6928615570068359 ,  0.3410775065422058 , -1.071485161781311  ,
  -0.6109848022460938 , -0.2546105682849884 , -0.3897858262062073 ,
  -1.079706072807312  ,  0.15206120908260345, -0.510136067867279  ,
  -0.07496775686740875, -0.00396917760372162, -0.19714514911174774,
  -0.28567782044410706,  0.868505597114563  , -0.26777219772338867,
  -1.7167893648147583 , -1.4376167058944702 , -0.8150161504745483 ,
   0.09466902166604996, -0.06500101089477539, -0.06988693028688431,
  -0.8784947395324707 , -0.5245230197906494 ],
 [ 0.05779170989990234, -0.10518451035022736,  0.04025458544492722,
   0.17907150089740753,  0.4066687524318695 , -0.5648045539855957 ,
   0.26392605900764465,  0.09256390482187271,  0.38858258724212646,
   0.0237431488931179 ,  0.3060328960418701 , -0.1871030479669571 ,
  -0.1513081192970276 , -0.38323909044265747,  0.07517574727535248,
  -0.11265094578266144,  0.07817929238080978,  0.30348047614097595,
   0.09483213722705841, -0.10873754322528839, -1.1574419736862183 ,
  -0.08821742236614227, -0.3145263195037842 , -1.2538048028945923 ,
   0.519516110420227  , -0.43688586354255676, -0.9146538972854614 ,
  -0.0497409999370575 , -0.28026729822158813,  0.34290066361427307,
  -0.07518307119607925,  0.8087024092674255 , -0.9054721593856812 ,
  -0.5871365666389465 , -0.1937026083469391 , -1.0849117040634155 ,
   0.6653359532356262 ,  0.00155574921518564, -0.7077418565750122 ,
  -0.4842565059661865 ,  0.02623721025884151, -0.9229826927185059 ,
   0.16462865471839905, -0.1652674674987793 , -0.9371680021286011 ,
   0.08865772932767868, -1.1526352167129517 ,  0.6700950860977173 ,
  -0.23871950805187225,  0.6676613688468933 , -0.28754934668540955,
   0.25430935621261597, -0.6571487188339233 ,  0.08497220277786255,
   0.1838524043560028 ,  0.7032654881477356 , -0.608744740486145  ,
   0.13109411299228668, -0.5864094495773315 , -0.19299836456775665,
  -0.1379140466451645 , -0.471600741147995  , -1.188462257385254  ,
  -0.3965340554714203 ,  0.33899086713790894,  0.2382984757423401 ,
  -0.3998701870441437 ,  0.1932395100593567 , -0.43338197469711304,
  -0.3974758982658386 , -0.530857264995575  , -0.3700251579284668 ,
   0.11946631222963333, -1.17318594455719   , -0.19025132060050964,
  -0.08336389064788818, -0.49045780301094055,  0.9310091137886047 ,
   0.01877538301050663, -0.01115274801850319, -0.13457435369491577,
   1.0505322217941284 , -0.03792928531765938, -0.30333882570266724,
   0.09945935755968094, -0.28377455472946167,  0.1271546185016632 ,
  -0.2177300900220871 ,  0.01198101416230202,  0.11559915542602539,
   0.16253261268138885,  0.18164946138858795,  1.8239175081253052 ,
  -1.416098952293396  , -0.5624242424964905 ,  0.497642457485199  ,
   0.6054062247276306 , -0.09854521602392197, -0.36600303649902344,
  -0.06580385565757751,  0.17892101407051086,  0.6407544016838074 ,
   0.08207175880670547,  0.7411453127861023 , -0.23334762454032898,
  -0.7713935375213623 , -0.0303841233253479 ,  0.01874051429331303,
  -1.9385364055633545 ,  0.00660865847021341,  0.08500764518976212,
   0.2982637882232666 , -0.5704134702682495 , -0.07852935791015625,
   0.13844673335552216,  1.5054293870925903 , -0.35314446687698364,
   0.13386264443397522,  0.7499186396598816 ,  0.11120579391717911,
  -0.25886374711990356,  0.04560424759984016,  0.24159136414527893,
   0.5965291261672974 , -0.21281401813030243, -0.0188074167817831 ,
   0.1371370404958725 , -0.7356023192405701 ],
 [ 1.002082347869873  , -0.02180718071758747,  0.169660747051239  ,
  -0.06906086951494217, -0.04819405823945999, -0.19317953288555145,
   0.2575031518936157 ,  0.4388545751571655 , -0.4175087511539459 ,
  -0.24968206882476807, -0.11816378682851791,  0.09343147277832031,
   0.3892892301082611 , -0.01528128329664469,  0.42015886306762695,
  -0.09918171167373657,  0.5550923943519592 ,  0.3328022360801697 ,
  -0.04112847149372101,  0.3316812813282013 , -0.20420320332050323,
  -0.17716726660728455, -0.06482387334108353, -0.01830872520804405,
   0.10025929659605026, -0.09597652405500412,  0.44935640692710876,
   0.5219177007675171 , -0.16592049598693848,  0.11665292084217072,
   0.18003343045711517, -0.6080325841903687 , -0.14080877602100372,
   0.2739618718624115 , -0.30260857939720154,  0.5637480020523071 ,
   0.4588688015937805 , -0.09747637808322906,  0.08407928794622421,
  -0.04208627343177795, -0.0574975349009037 , -0.04693670943379402,
  -0.04878808930516243, -0.00719441752880812, -0.3231065273284912 ,
   0.07841961085796356,  0.16220620274543762,  0.0804438591003418 ,
   0.5279077887535095 ,  0.9698514938354492 ,  0.1181349828839302 ,
   0.2374650239944458 , -0.01031752489507198,  0.11219387501478195,
  -0.2766832709312439 ,  0.15954464673995972,  0.08990415930747986,
   0.04768098145723343, -0.1621677130460739 ,  0.14447252452373505,
   0.1021769568324089 ,  0.10319969058036804, -0.05120237544178963,
  -0.22805088758468628, -0.23996223509311676, -0.08518309146165848,
  -0.11223524808883667,  0.2475024163722992 , -0.11961320042610168,
  -0.1298038512468338 , -0.2412404716014862 ,  0.38559436798095703,
   0.02046107687056065, -0.3964497148990631 ,  0.08616839349269867,
  -0.00223541585728526, -0.00275688734836876, -0.04319386184215546,
   0.4590212106704712 , -0.5658856630325317 ,  0.16156932711601257,
   0.30988380312919617, -0.21686239540576935,  0.08369007706642151,
   0.01372306700795889,  0.22210684418678284, -0.22586093842983246,
  -0.06355275213718414,  0.4618208408355713 ,  0.5529863238334656 ,
   0.1237361878156662 ,  0.27143043279647827,  0.07566357403993607,
  -0.4037923514842987 , -0.1973225474357605 ,  0.233724907040596  ,
   0.10428844392299652,  0.20221461355686188,  0.8246628642082214 ,
  -0.10615978389978409, -0.1366451531648636 , -0.0864056721329689 ,
  -0.00993299577385187, -0.13137665390968323,  0.27482718229293823,
   0.00077175669139251,  0.08207035064697266, -0.00897762086242437,
  -0.07076505571603775, -0.06579175591468811, -0.7757516503334045 ,
  -0.05691130459308624,  0.16546589136123657,  0.00229837326332927,
   0.04040949419140816,  0.0462847575545311 , -0.5155250430107117 ,
  -0.04559467360377312,  0.01123940572142601,  0.26645615696907043,
   0.02313181571662426, -0.11707510054111481,  0.02597837336361408,
   0.23121732473373413, -0.3968052566051483 , -0.03261572122573853,
   0.19338534772396088, -0.01444066409021616],
 [-0.08071507513523102, -1.2039061784744263 ,  1.6464861631393433 ,
   0.14938785135746002, -0.63637775182724   , -1.0143600702285767 ,
  -0.9300302267074585 ,  0.09500932693481445, -0.41742682456970215,
   0.49964582920074463, -1.464490294456482  ,  0.732452929019928  ,
   0.28534069657325745, -1.7534748315811157 , -0.42866742610931396,
  -1.8691989183425903 , -0.22896485030651093, -0.363066166639328  ,
  -0.6434007287025452 , -3.0577945709228516 , -0.820228099822998  ,
  -0.8359859585762024 , -0.12531082332134247, -1.3837292194366455 ,
   0.1379704773426056 , -0.3924494981765747 , -0.6120277047157288 ,
   0.12351308017969131, -0.5945552587509155 , -0.27403929829597473,
   0.6359293460845947 , -0.5359573364257812 , -0.8794777393341064 ,
  -0.7657298445701599 ,  0.6904941201210022 , -2.375558376312256  ,
  -1.005677580833435  ,  0.1858271360397339 , -0.6193469166755676 ,
   0.118932344019413  , -0.27495449781417847, -0.37453991174697876,
  -1.8152064085006714 , -1.4515432119369507 ,  0.10709500312805176,
  -0.45135846734046936, -0.8303022980690002 , -1.4286924600601196 ,
  -0.4634166955947876 ,  0.06410110741853714,  0.27978605031967163,
   0.0396806038916111 , -0.6715128421783447 ,  0.39384350180625916,
  -1.7249267101287842 , -0.515332043170929  ,  1.0720137357711792 ,
  -0.03604074195027351,  0.5246179103851318 ,  0.04328301548957825,
  -0.3490239679813385 , -0.36460331082344055, -0.05891343206167221,
   0.43365803360939026, -0.32272180914878845,  0.22707533836364746,
  -0.5938631892204285 ,  0.1097945123910904 , -0.472307026386261  ,
  -0.36713162064552307, -0.5710600018501282 ,  0.25263476371765137,
  -0.508957028388977  , -1.693716049194336  , -0.22383002936840057,
  -0.24725663661956787, -0.646037220954895  , -0.3791782259941101 ,
  -0.9452350735664368 , -0.11412632465362549, -1.9810905456542969 ,
  -1.008644938468933  ,  0.17935428023338318,  0.3747089207172394 ,
  -2.8572094440460205 ,  0.02359224110841751, -0.4716390371322632 ,
  -0.5026965141296387 , -0.04849166050553322, -0.05989115312695503,
  -0.8065550327301025 , -0.3928556740283966 , -1.1844285726547241 ,
   0.9032445549964905 , -0.09415033459663391, -0.9000425338745117 ,
   0.1461588442325592 ,  0.16299960017204285, -0.4078883230686188 ,
   0.19236303865909576,  1.0130687952041626 , -0.520244836807251  ,
   1.1856237649917603 , -1.9624536037445068 ,  0.3382114768028259 ,
  -0.5019350051879883 ,  0.34099939465522766,  0.12171027809381485,
  -1.4986660480499268 , -2.395871639251709  ,  0.6124835014343262 ,
  -0.4809015989303589 , -0.6844619512557983 , -0.7811756730079651 ,
  -0.1652883142232895 , -0.6796090602874756 ,  0.24224863946437836,
  -1.248018741607666  ,  0.10763731598854065, -0.7682053446769714 ,
  -0.32580074667930603, -0.343650221824646  , -0.26402175426483154,
   0.6546595692634583 , -0.973503589630127  ,  0.1900622844696045 ,
  -0.4451099932193756 , -1.2827773094177246 ],
 [ 0.80091792345047   ,  0.2163846641778946 , -0.6538980603218079 ,
  -0.4232330024242401 ,  0.5180588364601135 , -1.2567670345306396 ,
  -0.3346993625164032 ,  0.4792560040950775 , -1.338194489479065  ,
  -0.3032050132751465 ,  0.3405403792858124 , -0.6707959175109863 ,
  -0.7972658276557922 ,  0.0476977676153183 , -0.2289261668920517 ,
   0.6153919100761414 , -0.5159891247749329 ,  0.552091658115387  ,
   0.19605377316474915,  0.07900331914424896, -0.06525509804487228,
   0.23617243766784668,  0.32021772861480713,  0.1936517059803009 ,
   0.1800125688314438 ,  0.14091864228248596,  0.6757778525352478 ,
   0.17113769054412842,  0.01845167391002178,  0.12233352661132812,
  -0.30268022418022156, -1.9979051351547241 , -0.6179335713386536 ,
   0.00918240658938885, -0.12383513152599335, -1.488382339477539  ,
   0.4087052047252655 , -0.4314001202583313 , -0.24453207850456238,
  -0.47995609045028687, -0.17699798941612244, -0.7577764987945557 ,
  -0.3623793423175812 , -0.06135949119925499, -0.2037443369626999 ,
  -1.6909438371658325 ,  0.3071548044681549 ,  0.10479351878166199,
  -0.8474434614181519 ,  0.03968783840537071, -1.075525164604187  ,
  -1.672878623008728  , -0.29613664746284485, -0.2448624223470688 ,
   0.21858477592468262, -0.918940544128418  , -0.7089086174964905 ,
  -0.05078538507223129, -0.17312218248844147,  0.09685517847537994,
  -0.51673424243927   ,  0.14908047020435333,  0.7531549334526062 ,
   0.317394495010376  , -0.8842204809188843 , -2.9211723804473877 ,
   0.4712185561656952 , -0.07403452694416046,  0.1224341168999672 ,
  -1.497278094291687  , -0.8410856127738953 ,  0.03275937214493752,
   0.5956120491027832 ,  0.41008222103118896,  0.4087086319923401 ,
   0.200966939330101  ,  0.3008277714252472 , -3.1521692276000977 ,
   0.5888999700546265 , -1.5482616424560547 , -0.39270517230033875,
  -0.2325851321220398 , -1.2057863473892212 , -0.10705754160881042,
   0.05013161897659302, -1.1981866359710693 ,  0.00568565540015697,
  -0.8009481430053711 , -0.09659560024738312, -0.814876139163971  ,
  -0.13431623578071594, -0.38157230615615845, -0.9121502041816711 ,
   0.1418944001197815 ,  0.01365374587476254,  0.3461477756500244 ,
  -0.22749634087085724, -0.06709979474544525, -0.0840211808681488 ,
  -0.5233580470085144 ,  0.14189262688159943,  0.43068400025367737,
  -0.3864479660987854 ,  0.4940735101699829 , -0.6390541791915894 ,
  -0.7756282687187195 , -0.8541626334190369 , -0.05995512753725052,
  -0.8676748275756836 ,  0.3548855781555176 , -0.3863801956176758 ,
   0.587178647518158  , -0.3629958927631378 , -2.204272508621216  ,
  -0.37667229771614075, -0.21828781068325043, -2.1197614669799805 ,
   0.01356165204197168, -0.18211208283901215,  0.2590934634208679 ,
   0.21637773513793945,  0.20321659743785858, -0.4071609079837799 ,
  -0.15841685235500336,  0.7723620533943176 , -0.0479794479906559 ,
  -0.2089104950428009 ,  0.24155445396900177],
 [-0.03397449851036072, -0.0455995537340641 , -0.08605959266424179,
   0.48935917019844055,  0.01363204605877399, -0.0331832580268383 ,
  -0.3559591770172119 , -0.02435195073485374, -0.29354220628738403,
  -0.23311547935009003,  0.1066015288233757 , -0.07022683322429657,
   0.10776911675930023, -0.15265776216983795,  0.09928687661886215,
   0.00502978870645165,  0.00288664945401251,  0.04982219263911247,
  -0.04805641621351242, -0.04323676228523254,  0.03795592114329338,
  -0.18383345007896423,  0.00752856908366084,  0.04581386223435402,
   0.05921277776360512, -0.2608881890773773 , -0.16027818620204926,
  -0.3378378450870514 ,  0.13136810064315796,  0.05394576117396355,
   0.04066152125597   , -0.16261078417301178, -0.10346347838640213,
  -0.11420824378728867,  0.09931613504886627, -0.22691629827022552,
  -0.01410320587456226, -0.12026675790548325,  0.05908515304327011,
  -0.02354331873357296, -0.05490407720208168, -0.10016794502735138,
   0.04589783772826195,  0.00839791819453239,  0.05877504125237465,
  -0.05638399347662926,  0.02155940793454647,  0.08922389894723892,
   0.27076345682144165, -0.14715617895126343,  0.1845216304063797 ,
  -0.23440377414226532,  0.04995246976613998, -0.18372288346290588,
   0.11657083034515381, -0.09160210192203522, -0.01335076708346605,
   0.6499764323234558 ,  0.00227684318087995,  0.4172563850879669 ,
   0.14548394083976746,  0.12265343964099884,  0.01543265115469694,
   0.39364033937454224,  0.00955813378095627,  0.0248993095010519 ,
  -0.03292552754282951,  0.07974090427160263,  0.08848322182893753,
   0.16036535799503326,  0.18718722462654114,  0.04148397594690323,
  -0.08650431782007217,  0.28077560663223267, -0.39186495542526245,
   0.10447879880666733,  0.05557866021990776,  0.03340593352913857,
   0.18103794753551483, -0.07310574501752853, -0.02610825560986996,
  -0.13193731009960175,  0.01067184098064899, -0.00754926959052682,
  -0.08509856462478638,  0.06948982179164886, -0.13951532542705536,
   0.0431913249194622 , -0.18870724737644196,  0.10908861458301544,
  -0.16077443957328796, -0.14885291457176208, -0.0239975992590189 ,
  -0.06266044825315475, -0.214678093791008  , -0.08499470353126526,
  -0.2053729146718979 , -0.07966891676187515,  0.11553161591291428,
  -0.02523421682417393,  0.02792937308549881, -0.05782579258084297,
   0.03411883115768433,  0.09466324001550674,  0.3039362132549286 ,
  -0.07763056457042694,  0.015446487814188  ,  0.16721102595329285,
  -0.00302068516612053,  0.03149177134037018, -0.02538971602916718,
  -0.1733911633491516 ,  0.15513992309570312,  0.049491997808218  ,
   0.11139222979545593, -0.15175069868564606, -0.24689847230911255,
  -0.0287818182259798 ,  0.10172025114297867,  0.4357755780220032 ,
  -0.0125396465882659 , -0.08538812398910522, -0.16626067459583282,
  -0.2664034962654114 ,  0.17298342287540436, -0.1017395555973053 ,
   0.05601530894637108,  0.02633707039058208],
 [ 1.3213001489639282 , -0.24616634845733643,  0.8361387848854065 ,
   0.5180017948150635 ,  0.49649494886398315,  0.5916513800621033 ,
  -1.0318173170089722 ,  0.7270090579986572 ,  0.34024447202682495,
  -0.7011369466781616 ,  0.7879639863967896 ,  0.03971342369914055,
   0.7226240634918213 , -0.603578507900238  , -0.5016027688980103 ,
   0.4684599041938782 , -0.2826356589794159 , -0.22480101883411407,
   0.10882750153541565,  1.7605396509170532 ,  0.13764379918575287,
   1.0980093479156494 ,  0.38674595952033997,  0.6486861705780029 ,
   0.05317898094654083, -0.576785683631897  ,  0.1527555137872696 ,
   0.7923797965049744 , -0.39891448616981506,  0.6643711924552917 ,
   0.9277019500732422 ,  0.22325406968593597, -0.29588472843170166,
   0.08739786595106125,  0.4677421450614929 , -0.9398089051246643 ,
  -0.01007237378507853, -0.46170681715011597,  1.527918815612793  ,
   0.3599790334701538 ,  0.4359104037284851 , -0.09083136171102524,
   0.2531861960887909 ,  0.789794921875     , -1.0870392322540283 ,
   0.3953990936279297 ,  0.29823702573776245, -0.02062322571873665,
   0.19038759171962738,  0.9508568048477173 , -0.16101622581481934,
   0.00306525803171098,  0.40173542499542236, -0.0424468070268631 ,
   0.5356091856956482 , -0.38527941703796387, -0.343142032623291  ,
  -0.2866656482219696 ,  0.23517191410064697, -0.504889190196991  ,
   0.08899884670972824,  0.25468432903289795,  0.26681241393089294,
  -0.7681577205657959 ,  0.7815422415733337 , -0.6292812824249268 ,
   0.1626790314912796 ,  0.18182207643985748, -0.48267725110054016,
  -0.3783421814441681 , -0.5796025991439819 , -0.02234598621726036,
  -0.35362711548805237,  0.5827428698539734 ,  0.16753265261650085,
  -0.4890996217727661 , -0.23724700510501862,  0.4332555830478668 ,
   0.05000382289290428,  0.6852391958236694 , -0.8540123105049133 ,
   0.8217787146568298 ,  0.45082366466522217, -0.33379438519477844,
  -0.347201406955719  , -0.1291806995868683 , -0.247240349650383  ,
   0.4487977921962738 , -0.41608068346977234,  1.0911283493041992 ,
   0.5178696513175964 ,  0.587974488735199  ,  0.7039868831634521 ,
   0.8815247416496277 ,  0.5686535835266113 ,  0.16961610317230225,
  -0.6739597320556641 ,  0.2649095058441162 ,  0.18570080399513245,
   0.9646484851837158 , -2.1607351303100586 ,  0.526980996131897  ,
  -0.4972273111343384 ,  1.247746229171753  , -0.17780496180057526,
   0.1801723837852478 , -0.6715338230133057 ,  0.02664897404611111,
   1.2335400581359863 ,  1.6490918397903442 , -0.683259129524231  ,
  -0.0871828943490982 ,  0.21252231299877167,  0.29634910821914673,
  -0.617670476436615  , -0.04409483820199966,  0.9121218323707581 ,
   0.5278384685516357 , -0.1663801670074463 ,  0.5788928270339966 ,
  -1.3418216705322266 , -0.29444754123687744, -0.02212098240852356,
  -0.09541518241167068,  0.18075397610664368, -0.4767146408557892 ,
   0.2916593849658966 ,  0.519351065158844  ],
 [ 0.8921074271202087 , -0.3809622526168823 , -0.22085897624492645,
  -0.02789649926126003,  0.25744161009788513, -0.13177244365215302,
  -0.4778863787651062 ,  0.11803451180458069, -0.45064979791641235,
  -0.03253550082445145, -0.3328925669193268 , -0.1327226459980011 ,
   0.1895679235458374 , -0.59132319688797   ,  0.17244884371757507,
   0.09257496148347855,  1.3544142246246338 ,  0.34356439113616943,
  -1.0563477277755737 ,  0.00852221995592117, -0.17448943853378296,
  -0.16038241982460022,  0.33360275626182556, -0.9615450501441956 ,
  -0.02527224645018578, -0.14175568521022797,  0.33922669291496277,
  -0.4650137424468994 ,  0.5464798212051392 , -0.17981316149234772,
  -0.19802570343017578,  0.4451201856136322 ,  0.3075007200241089 ,
  -0.0612327829003334 , -0.3692362904548645 ,  0.4373359978199005 ,
  -0.03005850128829479, -0.05512196943163872, -0.395551472902298  ,
  -0.15764139592647552, -1.0891988277435303 , -0.3004298210144043 ,
   0.21141931414604187,  0.6109336018562317 , -2.321202039718628  ,
   0.8039330244064331 , -1.5869539976119995 ,  0.1470927596092224 ,
  -0.27281635999679565,  0.17891700565814972, -0.17939697206020355,
  -1.022437572479248  , -0.5616552233695984 , -0.01950020343065262,
  -0.04265909641981125,  0.05206242203712463, -1.0402021408081055 ,
  -0.15679717063903809, -0.00607755873352289, -1.2396785020828247 ,
   0.6290784478187561 , -0.3978683054447174 , -1.2031468152999878 ,
  -0.7585572600364685 ,  0.60600745677948   ,  0.8909338712692261 ,
  -0.6780960559844971 , -1.0817182064056396 , -0.17215870320796967,
  -0.08376476913690567, -0.36248138546943665, -0.20130625367164612,
   0.31571102142333984,  0.5313186049461365 , -0.47680190205574036,
  -0.480815052986145  , -0.8767579197883606 ,  0.29541149735450745,
  -0.00464498111978173, -0.9511904120445251 , -0.4563102126121521 ,
  -0.1867898404598236 ,  0.6078778505325317 ,  0.46775320172309875,
  -0.08951511234045029,  0.5834804177284241 , -0.38242781162261963,
   0.8547050356864929 , -0.10542749613523483, -0.603705108165741  ,
  -0.6019219756126404 , -0.450517863035202  , -0.05688447877764702,
  -0.6514858603477478 ,  0.2980072796344757 , -0.1267937421798706 ,
   0.20282979309558868,  0.8911402821540833 ,  0.05627529695630074,
  -0.15757708251476288, -0.6102554202079773 ,  0.5518849492073059 ,
  -0.23944953083992004, -1.0273774862289429 , -0.5193001627922058 ,
   0.18228890001773834, -0.374858558177948  ,  0.4525972306728363 ,
   0.4301425516605377 ,  0.22882123291492462, -0.33200109004974365,
  -0.14248797297477722, -0.27320006489753723,  0.2275729775428772 ,
  -0.5826671123504639 ,  0.5501980781555176 ,  0.4511723816394806 ,
  -0.21123486757278442, -0.2843051552772522 , -0.29462936520576477,
  -0.03928644210100174, -0.09220821410417557, -0.11113673448562622,
  -0.47622615098953247, -0.4516850709915161 ,  0.09173611551523209,
  -0.5176638960838318 , -0.972195029258728  ],
 [ 0.543930172920227  , -0.07413672655820847, -0.03298351913690567,
  -0.6405510902404785 ,  1.060193657875061  , -0.6350446343421936 ,
   0.07988734543323517, -0.09971095621585846,  0.3628065884113312 ,
   0.252163827419281  ,  0.21480603516101837,  0.1961643397808075 ,
   0.3630859851837158 ,  0.12748819589614868, -0.09402625262737274,
  -0.03916453570127487, -0.3343847095966339 , -0.08059893548488617,
   0.1613525003194809 , -0.22625373303890228,  0.16032502055168152,
   0.20960651338100433, -0.17323772609233856,  0.2631496489048004 ,
  -0.29468828439712524,  0.04082179069519043,  0.3195684552192688 ,
  -0.36323052644729614, -0.03056809678673744, -0.05727237090468407,
   0.2814849019050598 ,  0.19462557137012482, -0.21195480227470398,
  -0.1438456028699875 ,  0.02136911824345589,  1.065656065940857  ,
   0.23799243569374084,  0.02991664037108421,  0.05394276976585388,
  -0.08853904157876968, -0.22917801141738892,  0.21583692729473114,
   0.30292651057243347,  0.04872832074761391, -0.23354880511760712,
   0.02895352616906166, -0.05378400161862373,  0.04501210898160934,
  -0.4656420052051544 ,  0.09377752244472504,  0.2515033483505249 ,
  -0.2514018714427948 , -0.01686607487499714, -0.12001246213912964,
  -0.15689557790756226, -0.36002448201179504, -0.07698654383420944,
  -0.22684592008590698,  0.471282422542572  ,  0.08826617151498795,
  -0.34336763620376587,  0.1718619018793106 ,  0.13612964749336243,
  -0.20528832077980042,  0.3673959970474243 , -0.02989833243191242,
  -0.13936662673950195,  0.3044505715370178 ,  0.3754352033138275 ,
  -0.26783880591392517, -0.02665886655449867, -0.8906805515289307 ,
  -0.16671054065227509, -0.3586938679218292 ,  0.3724798858165741 ,
   0.31139299273490906,  0.02383372001349926,  0.14587725698947906,
   0.11723088473081589, -0.43687671422958374, -0.04236320033669472,
   0.16668105125427246,  0.0423126295208931 ,  0.18922372162342072,
   0.04507246613502502,  0.35145050287246704, -0.8809757232666016 ,
  -0.2310037910938263 , -0.3758630156517029 ,  0.3502011299133301 ,
  -0.11648115515708923, -1.084018588066101  ,  0.20734205842018127,
   0.23616276681423187, -0.0513518750667572 , -0.34118902683258057,
   0.2807716727256775 , -0.2546672821044922 ,  0.10350798070430756,
   0.13467656075954437,  0.06483756005764008,  0.5887738466262817 ,
   0.0603940524160862 ,  0.6747748255729675 , -0.4541379511356354 ,
  -0.4311715066432953 ,  0.10660260915756226, -0.14448948204517365,
  -0.00187269097659737, -0.15548907220363617, -0.4367671310901642 ,
   0.14584152400493622,  0.7355242967605591 , -0.5194799900054932 ,
  -0.09257397800683975, -0.6410565972328186 ,  0.29847586154937744,
   0.29940879344940186,  0.35385850071907043,  0.19181285798549652,
  -0.20197336375713348,  0.326469749212265  ,  0.12981726229190826,
  -1.1035959720611572 ,  0.15553681552410126,  0.06697695702314377,
   0.5926331877708435 , -0.03707931563258171],
 [ 0.17024534940719604,  0.6002206802368164 ,  0.09648361802101135,
  -1.5003985166549683 ,  0.3825267255306244 ,  0.15722763538360596,
   0.4156076908111572 ,  0.14972057938575745, -0.29207873344421387,
   0.00892974995076656, -2.2833948135375977 , -0.16309702396392822,
  -0.2183643877506256 , -0.0259630624204874 , -0.16185298562049866,
  -0.09584279358386993, -0.5517603754997253 ,  0.20651651918888092,
  -1.4172738790512085 , -0.02681183256208897, -1.1068594455718994 ,
  -0.06472918391227722,  0.24409793317317963,  0.6436323523521423 ,
   0.43033838272094727,  0.46878182888031006,  0.0820896103978157 ,
  -0.07519204914569855,  0.16126300394535065,  0.6979803442955017 ,
   0.1120188906788826 , -0.44947630167007446, -1.028184175491333  ,
  -1.5909004211425781 , -0.07653708010911942,  0.7262316346168518 ,
  -0.0252988412976265 , -0.1313445270061493 ,  0.5348319411277771 ,
   0.00198325491510332,  1.670108437538147  , -0.8741033673286438 ,
   0.2854897379875183 ,  0.2900526523590088 , -0.08482703566551208,
  -0.7679715156555176 ,  0.47034960985183716, -1.401760220527649  ,
   0.3798544704914093 ,  0.27400895953178406, -1.4762190580368042 ,
  -0.42197686433792114,  1.5063178539276123 ,  0.10388938337564468,
  -0.981661856174469  , -1.674468994140625  ,  0.8538068532943726 ,
  -0.12988968193531036, -0.08292072266340256,  0.5208614468574524 ,
  -0.15312714874744415,  0.98404461145401   ,  0.13161857426166534,
  -0.5170823931694031 , -0.30320462584495544,  0.3520585596561432 ,
  -0.3355048894882202 , -1.4018584489822388 ,  0.8118214011192322 ,
  -0.9340162873268127 , -0.4108971953392029 , -0.13671942055225372,
   0.00158638379070908, -0.7206840515136719 , -0.27712708711624146,
   0.16763250529766083, -0.6094494462013245 , -0.7546741366386414 ,
  -0.12155314534902573, -0.7249035239219666 , -0.9991757273674011 ,
  -0.3660682141780853 , -0.29768863320350647,  0.02461636438965797,
   0.18291279673576355,  0.6456975936889648 ,  0.24734589457511902,
  -1.8180205821990967 , -0.7059250473976135 , -0.13896746933460236,
   0.8051828145980835 , -0.3964764177799225 ,  0.7222357988357544 ,
   0.25299033522605896, -0.13582107424736023, -0.2053723931312561 ,
   0.06620927900075912, -1.8494077920913696 , -0.7741307020187378 ,
   0.04641997441649437, -0.9396013617515564 , -0.12677401304244995,
  -0.1837441325187683 ,  0.36870571970939636, -0.03697650507092476,
   0.27451011538505554, -1.4657944440841675 ,  1.496431589126587  ,
   0.9227346777915955 , -0.27077844738960266, -0.31257033348083496,
   0.606757402420044  , -0.6009986400604248 ,  0.6022352576255798 ,
   0.15403352677822113,  0.5602941513061523 ,  0.851314902305603  ,
  -1.4668283462524414 , -0.23898321390151978,  0.00662884535267949,
   1.681824803352356  ,  1.4803903102874756 , -0.3647894561290741 ,
  -0.18931178748607635, -0.43338772654533386,  0.47684717178344727,
  -0.2994340658187866 , -1.1641151905059814 ],
 [ 1.399007797241211  ,  1.7211207151412964 , -0.00525887263938785,
   0.5281602144241333 ,  0.1975991278886795 , -0.3491457402706146 ,
  -0.6358115673065186 ,  0.01226423308253288,  0.8026303648948669 ,
   0.5051400661468506 ,  0.5548067092895508 ,  0.08187729865312576,
   0.05132291093468666, -0.33186647295951843,  0.44476598501205444,
   0.11988750100135803, -0.6735314726829529 , -0.3196859657764435 ,
   0.18886809051036835,  0.21298745274543762,  0.14638788998126984,
   0.2381521314382553 ,  1.1097561120986938 ,  0.895675003528595  ,
   0.11889002472162247,  3.1432876586914062 , -1.07512366771698   ,
  -2.1214494705200195 , -0.6367886066436768 , -0.1474248170852661 ,
   0.4287812113761902 ,  1.1288667917251587 ,  0.5606971383094788 ,
   0.7160001993179321 ,  0.5624547600746155 , -0.9189625382423401 ,
  -0.4353182017803192 , -0.2438698410987854 ,  0.27590829133987427,
  -0.680650532245636  ,  0.04811153933405876, -0.02276362664997578,
  -0.659185528755188  ,  0.9160751104354858 ,  0.0608941987156868 ,
   0.19020751118659973, -0.14115987718105316, -0.4924718141555786 ,
   0.33251655101776123, -0.02274099364876747,  0.6878684163093567 ,
  -1.1304383277893066 , -0.87636399269104   ,  0.356025755405426  ,
   0.04220954328775406, -1.3768582344055176 ,  0.48058831691741943,
  -0.24868744611740112, -0.385884553194046  , -0.34175318479537964,
  -0.29120513796806335,  0.7840198874473572 ,  2.774907350540161  ,
  -0.15564818680286407,  0.5907157063484192 ,  0.589171290397644  ,
  -0.4056040644645691 , -0.44890475273132324,  0.2841740548610687 ,
   0.7251531481742859 , -0.21595065295696259, -0.3322214186191559 ,
  -0.48394501209259033, -1.8219650983810425 ,  0.31007349491119385,
   0.06817419081926346, -0.0492706224322319 , -0.6551030874252319 ,
   1.0799282789230347 , -0.31144678592681885, -0.1075541228055954 ,
  -0.240509495139122  ,  1.253835916519165  , -0.5328263640403748 ,
   0.09292063117027283,  0.17977577447891235, -1.0921834707260132 ,
   0.5694469809532166 ,  0.5111236572265625 ,  1.33560311794281   ,
  -1.116905689239502  , -0.280280739068985  , -0.5248679518699646 ,
  -0.3318476974964142 , -0.1752368062734604 , -0.7077082991600037 ,
   0.9810417890548706 , -2.0644936561584473 , -0.24253493547439575,
   0.23656177520751953, -0.03631523251533508, -0.8194081783294678 ,
   0.3151066303253174 ,  1.284269094467163  ,  0.24699890613555908,
  -0.41043129563331604,  0.34217941761016846, -0.9997609257698059 ,
   0.5775536894798279 ,  0.19869095087051392,  0.9644652009010315 ,
   0.7731025218963623 ,  0.4626895487308502 , -0.3132287859916687 ,
   0.36275115609169006,  0.9091776013374329 ,  0.3392881155014038 ,
   0.7129023671150208 , -0.00176380109041929, -0.1080198734998703 ,
  -1.3517682552337646 ,  0.1912928968667984 , -0.20024517178535461,
  -0.16001637279987335,  0.11628151684999466,  0.29574182629585266,
   0.5402703881263733 , -0.6974248290061951 ],
 [-0.02869816310703754,  0.6394240856170654 , -0.1266336888074875 ,
  -0.32704320549964905, -1.036089301109314  ,  0.13265463709831238,
   0.895568311214447  , -1.338179588317871  , -0.43145930767059326,
  -0.1605118364095688 , -0.41064316034317017,  0.1320967972278595 ,
   0.03890235722064972, -0.22789007425308228,  0.0958472192287445 ,
  -0.1265684962272644 ,  0.21908922493457794, -0.25345680117607117,
   0.25881102681159973,  0.1718112677335739 , -0.14135749638080597,
   0.22197136282920837,  0.3597539961338043 ,  1.0542806386947632 ,
  -0.4383167028427124 , -0.12254782766103745,  0.5011929869651794 ,
   0.6175893545150757 , -0.09297823160886765, -0.1156167984008789 ,
   0.6368029117584229 ,  0.7727853059768677 ,  0.1510571837425232 ,
   0.29567933082580566, -0.07863028347492218,  0.4499015212059021 ,
   1.1036776304244995 , -0.46864303946495056, -0.1274026483297348 ,
   0.29797884821891785, -0.9826238751411438 , -0.6270492076873779 ,
  -0.23257936537265778,  1.1740719079971313 , -0.07592525333166122,
   0.671718418598175  ,  0.21441705524921417, -0.3790050148963928 ,
   0.37430277466773987,  0.5492920279502869 ,  0.09039306640625   ,
  -0.19992166757583618, -0.47528645396232605,  0.17332831025123596,
  -0.05169768258929253, -0.12319261580705643,  1.4213926792144775 ,
   0.41058456897735596, -0.8333512544631958 ,  0.15360984206199646,
   0.24735024571418762,  0.2233428657054901 , -0.5496845841407776 ,
  -1.9455527067184448 , -2.6320197582244873 ,  0.180564746260643  ,
  -0.07628031075000763, -0.631310224533081  ,  1.2621947526931763 ,
   0.13058531284332275, -1.0899502038955688 , -0.20506668090820312,
  -0.4316796362400055 , -0.00243492005392909,  0.5521223545074463 ,
   1.5056520700454712 ,  0.8992538452148438 , -0.3325541317462921 ,
  -0.00014447564899456, -0.8138157725334167 ,  0.2009086161851883 ,
  -2.0845324993133545 ,  0.2752823829650879 , -1.5387650728225708 ,
   0.625045120716095  , -0.3007766604423523 , -0.45872020721435547,
   0.46921274065971375, -1.656813144683838  , -0.1449986696243286 ,
   0.07935941964387894,  0.22133289277553558, -0.5327433347702026 ,
  -0.2199971228837967 , -0.5627758502960205 ,  0.139806866645813  ,
   0.29146885871887207,  0.43576523661613464, -0.07701826095581055,
   0.15236158668994904, -1.1577647924423218 , -0.12364792823791504,
  -0.01625329442322254, -0.648725688457489  ,  0.6046128273010254 ,
  -1.7224154472351074 , -1.3650987148284912 ,  0.18817737698554993,
  -0.21635599434375763, -0.5003513097763062 ,  0.03298471122980118,
   0.46475866436958313, -0.02855612896382809,  0.9842169880867004 ,
  -0.2676466405391693 ,  0.311412513256073  ,  0.15758976340293884,
   1.1487871408462524 , -1.9632257223129272 ,  0.23301854729652405,
  -0.42326438426971436,  0.18577422201633453, -0.1965469866991043 ,
  -0.3496814966201782 ,  2.506688356399536  , -0.6716375350952148 ,
  -0.24016310274600983, -0.02121789008378983],
 [ 0.1086447685956955 , -0.02932321466505527,  0.19036391377449036,
   0.47272953391075134, -1.304972529411316  , -0.7313636541366577 ,
  -0.8927731513977051 ,  0.1631241738796234 , -0.46050402522087097,
  -0.05322810262441635, -0.22614344954490662, -0.12813140451908112,
  -0.10731389373540878, -0.6332752704620361 , -0.3933495879173279 ,
   0.02048264816403389, -0.5912864208221436 ,  0.1360393613576889 ,
  -0.7037690877914429 , -0.20122568309307098,  0.05708665028214455,
   0.27090466022491455, -0.24548770487308502,  0.08012684434652328,
   0.07052530348300934, -0.03858722373843193,  0.01073595508933067,
  -0.6538172960281372 ,  0.0581304207444191 ,  0.4146190285682678 ,
   0.0072323614731431 ,  0.42577192187309265, -0.2596147060394287 ,
  -0.46808165311813354, -0.20362119376659393, -0.10138338059186935,
  -0.6239842772483826 ,  0.01224117912352085, -0.8523713946342468 ,
   0.17738495767116547, -0.21756990253925323, -0.01872632093727589,
  -0.6727551817893982 , -1.9025053977966309 , -0.307243287563324  ,
  -0.34260737895965576, -0.4782574772834778 ,  0.21058501303195953,
  -0.08546075224876404, -0.00463857082650065,  0.1552935540676117 ,
   0.4942622482776642 , -0.5965659022331238 ,  0.16793224215507507,
  -0.03706606104969978, -0.7745770215988159 ,  0.06149248033761978,
  -0.2748745083808899 ,  0.2521416246891022 , -0.5715041160583496 ,
  -0.24991582334041595, -0.8491596579551697 , -0.4073803424835205 ,
  -0.05463390797376633, -0.3270544409751892 , -1.4217451810836792 ,
   0.037545345723629  , -0.29822659492492676, -0.3805646598339081 ,
  -0.28907057642936707, -0.3143080770969391 ,  0.02491528168320656,
  -0.06306782364845276, -1.2393571138381958 , -0.6660857796669006 ,
  -0.32462078332901   , -0.17796768248081207, -0.19461537897586823,
  -0.470647394657135  , -0.21991093456745148, -0.6471970677375793 ,
  -0.9920580387115479 ,  0.21208660304546356, -0.5006087422370911 ,
  -0.00955237913876772, -0.3917483687400818 , -0.6657402515411377 ,
  -0.05918718501925468,  0.14735080301761627,  0.02578089758753777,
  -0.6243451237678528 ,  0.04069967195391655,  0.678500771522522  ,
   0.5569837093353271 ,  0.2502140700817108 , -0.16045495867729187,
  -0.34878963232040405, -0.8461125493049622 ,  0.10098985582590103,
   0.0439000278711319 ,  0.31673428416252136, -0.20767216384410858,
   0.03990774229168892, -0.3834293782711029 , -0.06303390115499496,
  -0.4289137125015259 ,  0.06881822645664215,  0.30063411593437195,
  -0.14210978150367737, -0.5645171403884888 , -0.11393313854932785,
  -1.2114534378051758 , -1.0051562786102295 , -0.34804192185401917,
   0.07615484297275543, -0.4218338429927826 , -0.20795391499996185,
  -0.02258498035371304,  0.10632792860269547, -0.42619505524635315,
   0.14365975558757782, -0.6222038865089417 , -0.3008998930454254 ,
   0.16489440202713013, -0.38727787137031555,  0.1262361854314804 ,
  -0.01849761232733727, -0.48900455236434937],
 [-0.3652205169200897 ,  0.01825917325913906,  0.16883128881454468,
   0.7295520305633545 ,  1.0960688591003418 , -1.10665762424469   ,
  -0.27509376406669617, -0.13525253534317017, -0.3372180163860321 ,
   0.08285486698150635, -0.7052242159843445 ,  0.30072471499443054,
  -0.15059025585651398, -1.897803544998169  ,  0.6840873956680298 ,
   0.00450164126232266, -1.4497625827789307 , -0.08695502579212189,
  -1.3688514232635498 ,  0.2911483645439148 , -0.14733058214187622,
   0.2008693367242813 ,  0.7416178584098816 , -0.6492748260498047 ,
  -0.8410662412643433 , -0.20942869782447815,  0.28872746229171753,
  -0.43611812591552734,  0.45316845178604126,  0.4937405288219452 ,
  -0.07602930814027786, -0.7310104966163635 , -0.1358785182237625 ,
  -0.45647063851356506, -0.07507606595754623,  0.17272743582725525,
   0.4260385036468506 ,  0.10600564628839493, -0.8274915814399719 ,
  -0.4153435528278351 , -0.42863309383392334, -1.0235404968261719 ,
  -0.15686236321926117, -0.0219461414963007 , -2.8148720264434814 ,
   0.46940070390701294,  1.5690748691558838 , -0.3537890911102295 ,
   0.3858175277709961 ,  0.04117988422513008, -0.3005547821521759 ,
  -0.06959709525108337, -1.7920758724212646 ,  0.16778475046157837,
  -2.9351189136505127 , -0.20966137945652008, -2.685800313949585  ,
  -0.2099960893392563 ,  0.06571130454540253, -0.27641966938972473,
   0.6546159982681274 , -0.5515786409378052 , -0.07182678580284119,
  -0.23880872130393982,  0.16300947964191437,  0.3643878102302551 ,
   0.6059190630912781 , -1.2636555433273315 , -0.15632104873657227,
   0.8247767090797424 ,  0.35061129927635193, -0.8882039189338684 ,
  -0.40541794896125793, -0.1894347220659256 ,  0.0213662963360548 ,
   0.46674829721450806, -0.789594829082489  , -0.23143181204795837,
   0.9645773768424988 ,  0.16199713945388794,  0.00648546312004328,
   0.21823932230472565,  0.25108110904693604,  0.6279183030128479 ,
  -0.2940407693386078 ,  0.7753122448921204 ,  0.01121168397367001,
  -1.1693998575210571 ,  0.4836660325527191 , -0.0485256053507328 ,
   1.7658090591430664 , -0.4247037470340729 , -1.3542745113372803 ,
   0.31567877531051636,  0.5321885943412781 , -0.3108917772769928 ,
  -0.09642034024000168,  0.00552441272884607, -1.1675652265548706 ,
   0.1657712459564209 , -0.0455435998737812 , -0.08147861063480377,
  -0.1716686487197876 , -1.2962627410888672 , -0.8211457133293152 ,
   0.18017975986003876,  0.6573923826217651 , -0.4577632546424866 ,
   0.1163882166147232 ,  0.10393323749303818,  0.3358471691608429 ,
  -0.24400053918361664, -1.748509407043457  ,  0.7224075198173523 ,
   0.24040453135967255,  0.2155582457780838 ,  0.48617520928382874,
   0.5544267892837524 ,  0.7834551334381104 , -0.24399127066135406,
   0.6427578926086426 ,  0.2773916721343994 , -0.3257245421409607 ,
  -0.144510880112648  , -0.9026678204536438 ,  0.20052386820316315,
  -0.3462531268596649 ,  1.1435251235961914 ],
 [ 0.33477967977523804, -1.4418070316314697 ,  0.42049968242645264,
   0.9898495078086853 , -1.0026332139968872 ,  1.6144218444824219 ,
   1.0564541816711426 ,  0.08941863477230072,  0.35711610317230225,
   0.2583039104938507 , -1.2336105108261108 , -0.15049991011619568,
   0.14888669550418854,  1.490138053894043  ,  0.5295167565345764 ,
  -0.9468526840209961 , -0.10594622045755386, -0.0420255996286869 ,
   0.06844781339168549, -1.1893746852874756 ,  0.6445631980895996 ,
  -1.6424386501312256 ,  0.3133162260055542 , -0.21471749246120453,
   0.6858863234519958 ,  0.8383992314338684 , -0.4952153265476227 ,
   0.0958716943860054 , -0.4811288118362427 ,  0.6944852471351624 ,
   0.42880871891975403, -0.5784068703651428 ,  1.438738226890564  ,
   0.5725170969963074 , -0.400068998336792  , -1.067814826965332  ,
   1.4838649034500122 , -0.04416543245315552, -0.20679903030395508,
   0.40714332461357117,  0.2247401475906372 ,  0.3020094633102417 ,
   1.394085168838501  ,  2.1126749515533447 , -0.05855738744139671,
   0.6242242455482483 , -1.5319088697433472 , -0.36090439558029175,
   0.23552584648132324,  0.08278682082891464,  0.8393414616584778 ,
   0.1478753238916397 ,  0.7089157700538635 ,  0.15204136073589325,
  -0.21463178098201752, -1.3887299299240112 ,  2.10792875289917   ,
  -0.2605723738670349 , -0.4909772276878357 , -1.7512444257736206 ,
   0.3198195993900299 ,  0.6627718806266785 ,  0.2532206177711487 ,
  -0.02329025231301785,  0.462186336517334  , -0.930458664894104  ,
   0.09358058124780655,  0.00588937615975738,  0.5781877040863037 ,
   0.7641614675521851 ,  1.1385931968688965 , -0.8782510161399841 ,
  -0.35220494866371155, -1.167783498764038  , -0.3750503957271576 ,
  -0.3744128346443176 ,  0.6413227319717407 ,  0.62154221534729   ,
  -0.03141894191503525,  0.12134595960378647,  1.2709605693817139 ,
  -2.6434876918792725 ,  0.11460878700017929, -1.7649312019348145 ,
  -0.1646607220172882 ,  0.3194405138492584 ,  0.05289879441261292,
   1.4005498886108398 ,  0.5758229494094849 ,  0.5249541997909546 ,
  -0.456616073846817  ,  0.3970540761947632 ,  1.6193751096725464 ,
   0.00107418419793248, -0.2267822027206421 ,  1.4886399507522583 ,
   0.8488171696662903 , -0.09078965336084366,  0.15966270864009857,
   0.6061574220657349 , -0.1102907806634903 ,  0.89251708984375   ,
  -0.02722459472715855,  0.28173449635505676, -1.4572701454162598 ,
   0.198533833026886  ,  0.11641797423362732,  1.063109278678894  ,
  -0.02667643129825592, -0.9795733690261841 ,  0.25533434748649597,
  -0.25858891010284424, -0.3412860333919525 ,  0.46851789951324463,
   0.7126109600067139 , -0.884100615978241  ,  0.28144538402557373,
   1.3382632732391357 ,  0.9509751796722412 ,  0.22590482234954834,
   2.2001562118530273 , -0.9138060212135315 ,  0.0215474646538496 ,
   0.05174321308732033, -0.8903190493583679 ,  0.00444616191089153,
   0.7346622347831726 ,  1.6196303367614746 ],
 [-0.9538649916648865 ,  0.5059248805046082 ,  0.02474386058747768,
  -0.12110771238803864,  1.2107465267181396 , -0.3056432604789734 ,
   0.3043263852596283 ,  0.9211280941963196 ,  0.78838711977005   ,
  -0.3775665760040283 , -0.7628731727600098 , -0.0149060906842351 ,
   0.23316513001918793, -0.04380243271589279,  0.11197631061077118,
   0.1398906707763672 , -0.559312105178833  ,  0.5346757173538208 ,
   0.9724724888801575 ,  0.9130426645278931 , -0.31102070212364197,
   0.8641241788864136 ,  0.2273775190114975 ,  0.3571731150150299 ,
  -1.112188458442688  , -0.19741791486740112,  0.810299277305603  ,
  -0.3251108229160309 , -0.6860403418540955 ,  0.666935920715332  ,
   0.1326884627342224 , -0.43664833903312683, -0.43621793389320374,
  -0.4645315706729889 , -0.07422329485416412,  0.13573405146598816,
   1.0770183801651    ,  0.43658706545829773,  1.1543375253677368 ,
  -0.3636373281478882 ,  0.6223737001419067 , -0.5044733285903931 ,
  -0.6634829044342041 ,  1.0494256019592285 , -0.9988657832145691 ,
   0.27437272667884827, -0.2629064619541168 , -0.13803671300411224,
  -0.2345593273639679 ,  0.7770958542823792 , -0.457732230424881  ,
   0.4019041061401367 ,  1.3833045959472656 ,  1.0056148767471313 ,
   0.02764005213975906, -0.45820513367652893, -0.8096005916595459 ,
   1.142617106437683  , -1.5649064779281616 ,  0.08825787901878357,
   0.590783953666687  ,  0.3931240737438202 ,  0.22307415306568146,
   0.5666174292564392 , -1.4218500852584839 ,  1.0071265697479248 ,
   0.3149968683719635 , -0.26322031021118164, -0.56601881980896   ,
  -0.25034603476524353,  0.42850369215011597, -0.04257382825016975,
   0.9495071172714233 ,  0.3757026195526123 ,  0.9426975250244141 ,
   1.6019415855407715 ,  1.2924076318740845 , -2.9745681285858154 ,
   1.940580129623413  ,  0.30795323848724365, -0.19910570979118347,
   1.0330017805099487 ,  0.18921364843845367, -1.1389187574386597 ,
  -0.4563131630420685 ,  0.6803449988365173 , -0.5850720405578613 ,
   0.18716953694820404,  0.17108067870140076,  0.5307390689849854 ,
   0.7113461494445801 , -1.1411097049713135 ,  1.7407665252685547 ,
  -0.00434164982289076,  0.27177903056144714, -0.03043089434504509,
   0.15656813979148865, -0.01762238517403603, -0.4256133437156677 ,
   1.1572489738464355 , -1.8146862983703613 , -0.10280871391296387,
   0.09140221029520035,  1.0120564699172974 , -0.45411649346351624,
  -0.4171120822429657 ,  0.6878373026847839 ,  0.44549790024757385,
   1.4880861043930054 ,  0.5492018461227417 ,  0.16677413880825043,
   1.3498128652572632 , -0.3596174716949463 ,  0.5934814810752869 ,
   0.08073002845048904,  2.4597299098968506 ,  0.527232825756073  ,
   0.5043677687644958 , -2.8725826740264893 ,  0.8372215032577515 ,
   0.45363298058509827,  0.3354014754295349 , -0.03822624310851097,
  -0.330434113740921  ,  1.7147585153579712 ,  0.3443748354911804 ,
  -0.6508873105049133 ,  1.433854579925537  ],
 [ 0.7646847367286682 ,  0.4542708396911621 ,  0.5870189070701599 ,
   1.088066577911377  ,  0.3278545141220093 , -0.44920432567596436,
   1.2548128366470337 ,  0.3277313709259033 ,  0.6731992959976196 ,
  -0.6609339714050293 ,  1.4227447509765625 ,  0.417339026927948  ,
   0.2472577840089798 , -0.2934204936027527 ,  1.2069987058639526 ,
   0.445447713136673  ,  0.42399123311042786,  0.4315834939479828 ,
   3.1469368934631348 , -1.3881357908248901 ,  0.5874590873718262 ,
   0.7636259198188782 , -1.892561435699463  , -0.9215695261955261 ,
   0.6859028935432434 ,  0.47796550393104553, -0.15582235157489777,
   0.20220953226089478,  0.03813164308667183,  0.5886022448539734 ,
   0.9638146758079529 , -0.5377640128135681 , -1.7535120248794556 ,
  -0.5484525561332703 ,  0.7405763864517212 ,  0.9764085412025452 ,
   1.905625581741333  ,  0.40488773584365845,  2.0252368450164795 ,
   0.36134839057922363,  1.0463181734085083 , -1.0148569345474243 ,
  -0.6806728839874268 , -1.8151243925094604 ,  1.672863245010376  ,
  -1.2613627910614014 , -0.8367674350738525 ,  1.0013431310653687 ,
   0.351962685585022  ,  0.2632671892642975 ,  0.49806734919548035,
   0.05665329843759537,  0.9123298525810242 ,  0.13904231786727905,
  -0.9246156811714172 , -0.8303768038749695 , -0.5943219065666199 ,
   0.5914338827133179 , -0.2763250470161438 , -0.5577874183654785 ,
  -1.3514653444290161 ,  0.9086918830871582 , -0.839708149433136  ,
  -2.9950928688049316 , -0.8736135363578796 ,  1.4622420072555542 ,
   0.6969214081764221 , -0.5953933596611023 ,  0.03786728158593178,
   0.5768937468528748 ,  1.1370116472244263 ,  0.30889391899108887,
   0.4961075186729431 ,  0.0827041044831276 ,  0.9604653716087341 ,
  -2.3613102436065674 ,  0.7663229703903198 , -0.9318350553512573 ,
   1.1103477478027344 ,  0.5412383079528809 , -0.509995698928833  ,
   0.5624270439147949 , -0.8376274704933167 , -0.37691500782966614,
   0.5486878752708435 , -1.1208311319351196 ,  0.1810915172100067 ,
  -1.4142247438430786 ,  0.1933673918247223 ,  0.8530288934707642 ,
  -1.2411127090454102 ,  0.5200273990631104 , -0.12952862679958344,
  -0.08840686827898026, -0.30958232283592224,  1.451115369796753  ,
   0.42218393087387085,  0.4117913544178009 ,  0.9618465900421143 ,
   1.0045840740203857 , -0.525668740272522  ,  0.184584379196167  ,
   0.07538007944822311,  1.0361926555633545 ,  1.5873397588729858 ,
   0.9617658257484436 ,  0.5014021396636963 ,  0.22771677374839783,
  -0.6380774974822998 ,  2.3874351978302    ,  0.03190365433692932,
  -0.5249776244163513 ,  0.1674516201019287 ,  1.0174179077148438 ,
  -0.6739227175712585 ,  0.8099853992462158 , -0.4932652711868286 ,
  -0.12860895693302155, -0.26673299074172974,  0.7043819427490234 ,
   0.02468420937657356,  0.4696175456047058 , -0.5524650812149048 ,
  -0.15455776453018188,  0.2971746027469635 ,  0.4921962320804596 ,
   0.9441226720809937 ,  0.4569205045700073 ],
 [-0.1859438270330429 , -1.2915308475494385 , -0.09203283488750458,
  -0.7793261408805847 ,  0.9509056806564331 , -0.48108622431755066,
   0.1753094345331192 ,  0.4710903465747833 ,  0.31516703963279724,
  -0.4722450077533722 ,  0.29848530888557434, -0.8926687836647034 ,
   0.07850945740938187, -0.15604175627231598, -0.3734309673309326 ,
   0.10395273566246033, -0.822489857673645  ,  0.1429726928472519 ,
   0.26761752367019653,  0.35449907183647156,  0.07056114077568054,
  -0.25855669379234314, -0.3823237121105194 , -0.8379465937614441 ,
   0.5430437326431274 , -0.37868496775627136, -0.24177859723567963,
   0.9410249590873718 , -0.15968705713748932,  0.3319230377674103 ,
   0.15335917472839355, -0.474160373210907  , -0.32194414734840393,
   0.0454462543129921 , -0.3477671444416046 , -0.4930301308631897 ,
  -0.01028380822390318,  0.12310104072093964, -0.04394746199250221,
   0.2759070098400116 , -0.49734681844711304, -0.18668024241924286,
  -0.14650781452655792,  0.01539201382547617,  0.6536362767219543 ,
   0.5853453278541565 , -0.01745963841676712,  0.40431782603263855,
  -0.8349078297615051 ,  0.14237229526042938,  0.7471309900283813 ,
  -0.2072344869375229 , -0.8302242159843445 , -0.13248613476753235,
   0.08551808446645737,  0.4596629738807678 ,  0.31671246886253357,
   0.26170626282691956,  0.393421471118927  , -0.00157279917038977,
   0.0966014638543129 ,  1.114210844039917  , -0.1831451654434204 ,
   0.47560733556747437, -0.6799691319465637 ,  0.3494316637516022 ,
  -0.24680887162685394,  0.21660509705543518, -0.3326968252658844 ,
  -0.5625675320625305 ,  0.5853874087333679 , -0.2200203388929367 ,
   0.538510799407959  ,  0.8168209791183472 ,  0.29392820596694946,
   0.23201465606689453, -0.14656434953212738,  0.35008692741394043,
   0.02113370969891548,  0.02665665186941624, -0.05997171998023987,
  -0.19434113800525665, -0.2302536815404892 ,  0.3592855930328369 ,
  -0.1079121083021164 ,  0.13257260620594025,  0.34422293305397034,
   0.08065115660429001, -0.0783490464091301 ,  0.4682617485523224 ,
   0.1627296507358551 , -0.18586499989032745, -0.10805036127567291,
  -0.6633711457252502 , -0.11308958381414413, -0.28613048791885376,
  -1.0091761350631714 , -0.20243753492832184,  0.305966854095459  ,
  -0.36459842324256897, -0.02311555854976177, -0.21551473438739777,
  -0.09712084382772446, -0.4895428419113159 ,  0.13578034937381744,
  -0.3792719542980194 ,  0.21391645073890686, -0.38368093967437744,
  -0.2099854201078415 , -0.03125166893005371, -0.5169118642807007 ,
   0.06166762113571167,  0.49291276931762695, -0.594066858291626  ,
  -0.09667979180812836, -0.9215756058692932 , -0.2697565257549286 ,
  -0.11324209719896317, -0.43608713150024414,  0.5479034185409546 ,
   0.3679869771003723 ,  0.21360188722610474, -0.22325678169727325,
  -0.15705670416355133,  0.3577534258365631 ,  0.24209119379520416,
  -0.05501745268702507, -0.00085415260400623],
 [-0.01451045740395784, -3.9719808101654053 , -0.03892089053988457,
  -0.0289141871035099 , -0.3664710521697998 ,  0.1890517622232437 ,
  -2.0943942070007324 ,  0.0048640388995409 , -2.451564311981201  ,
  -0.1745859533548355 , -1.8144092559814453 ,  0.06996144354343414,
  -0.0492611937224865 , -2.8362691402435303 , -0.12263912707567215,
  -0.04194408655166626,  1.4458863735198975 , -0.22574929893016815,
  -0.27638933062553406,  0.12377111613750458,  0.5112558007240295 ,
  -0.11324594169855118,  0.466776043176651  , -0.22713182866573334,
   0.02046774886548519, -0.06732220947742462,  0.398386150598526  ,
  -0.16041050851345062, -0.24901095032691956, -0.2673289477825165 ,
  -0.02017624117434025,  0.5682147145271301 ,  0.11119404435157776,
   0.01176850032061338,  0.04940532147884369,  0.43232953548431396,
  -0.4528144598007202 , -0.02923521399497986, -0.4379238486289978 ,
   0.26549461483955383,  0.27333617210388184, -1.0260097980499268 ,
  -0.6809094548225403 , -0.3015948235988617 ,  0.48776447772979736,
   0.2836596369743347 ,  0.21877838671207428,  0.5116533041000366 ,
  -0.09082408249378204,  0.04533716663718224,  0.16322381794452667,
   0.0369439609348774 , -0.05940672382712364, -0.06467562168836594,
  -0.773427426815033  , -0.05290137603878975,  0.02488336339592934,
  -0.10321571677923203, -0.0985753983259201 , -0.6077699065208435 ,
   0.04818366467952728, -1.156923532485962  ,  0.5092686414718628 ,
  -0.5606244206428528 ,  0.01199785713106394, -0.20087669789791107,
  -0.24223114550113678,  0.04508670791983604,  0.13816401362419128,
  -1.8806357383728027 ,  0.6759498715400696 , -0.5542404651641846 ,
   0.00411158008500934, -0.29353541135787964, -0.08993774652481079,
  -0.3111352026462555 , -1.2079925537109375 ,  0.5515523552894592 ,
  -2.1352293491363525 , -0.26813119649887085, -0.19282014667987823,
  -1.5710655450820923 ,  0.13714544475078583, -0.4308941662311554 ,
   0.06231605261564255,  0.3819034695625305 ,  0.28678739070892334,
  -0.4514482021331787 ,  0.09789630770683289, -0.03939097747206688,
  -1.641435146331787  ,  0.10317906737327576,  0.5517884492874146 ,
   0.29711833596229553,  0.22053538262844086,  0.30523398518562317,
   0.513190507888794  ,  0.34990590810775757,  0.49271726608276367,
   0.06344839185476303,  0.7731500864028931 , -1.289622187614441  ,
   0.03339725732803345, -0.3749307096004486 ,  0.2360997498035431 ,
  -0.24997682869434357,  0.24408051371574402, -0.6233376264572144 ,
  -1.0980647802352905 ,  0.14523470401763916, -0.1489740014076233 ,
   0.27352046966552734,  0.35150572657585144, -0.6260013580322266 ,
  -0.01277794782072306, -1.2775897979736328 , -0.08345378935337067,
   0.11922808736562729, -0.4219275414943695 , -0.20519666373729706,
  -0.865717351436615  , -0.38815054297447205, -0.2521832585334778 ,
   0.08208297193050385, -0.5490714311599731 , -0.1311154067516327 ,
   0.07837436348199844, -1.0480579137802124 ],
 [ 0.07755210995674133,  1.0213830471038818 , -0.2473352551460266 ,
  -0.4310244619846344 , -0.5631592273712158 ,  0.47336092591285706,
  -0.9416425824165344 , -0.16330459713935852,  0.1641480028629303 ,
  -0.21962757408618927,  0.939157247543335  ,  0.16574956476688385,
  -0.42088860273361206,  0.22824274003505707, -1.5419870615005493 ,
   0.14541856944561005, -0.220701664686203  , -0.4757828712463379 ,
  -1.9108941555023193 ,  0.46110013127326965, -1.1126543283462524 ,
   0.17672477662563324,  0.2016185224056244 , -0.5848706960678101 ,
   0.4953136444091797 , -1.6659398078918457 , -0.44583582878112793,
   0.20232322812080383, -0.01459200866520405, -1.7851927280426025 ,
   0.17847368121147156,  0.3380817174911499 , -0.0466078482568264 ,
   0.05588189512491226,  0.0120629807934165 ,  0.8875784277915955 ,
   0.17461062967777252, -0.12542392313480377,  0.231013685464859  ,
  -0.20408757030963898,  1.405457615852356  , -0.4687463641166687 ,
  -0.5059679746627808 ,  0.540518045425415  ,  0.02538364380598068,
   0.6018210649490356 ,  0.7260236144065857 , -0.824030876159668  ,
   0.13772845268249512,  1.3577402830123901 ,  0.6936016082763672 ,
   0.18162764608860016, -1.2419750690460205 , -0.2853032946586609 ,
  -1.4455138444900513 , -0.1642022430896759 , -1.4387394189834595 ,
  -0.3302977383136749 , -0.215087428689003  , -1.2773463726043701 ,
  -0.2666069567203522 , -0.16920478641986847,  1.4311003684997559 ,
   1.020616054534912  , -0.3114178776741028 ,  0.6513401865959167 ,
   0.38573575019836426, -1.822167158126831  ,  0.32119929790496826,
   0.1978984922170639 , -0.8475231528282166 , -0.1183353066444397 ,
   0.22258134186267853,  1.0188779830932617 , -0.43853989243507385,
   0.0888524055480957 , -1.0748223066329956 ,  0.26974058151245117,
  -1.6551181077957153 ,  0.11901876330375671,  0.41305696964263916,
   1.0552641153335571 , -0.536338210105896  ,  1.3662059307098389 ,
   0.1687101423740387 ,  0.12387318164110184, -0.7476897835731506 ,
  -1.6728849411010742 ,  0.02696112543344498, -0.10681518912315369,
   0.40469104051589966,  0.4282999634742737 , -1.1155070066452026 ,
  -1.265913486480713  ,  0.10334612429141998, -0.45273342728614807,
   0.3065547049045563 , -1.475525140762329  , -1.1720786094665527 ,
   0.07707028090953827, -0.3416232168674469 , -0.03960886970162392,
   0.02604459784924984,  0.5245158076286316 ,  0.03300009295344353,
  -1.0706582069396973 , -0.35869789123535156,  0.5008092522621155 ,
  -0.8413011431694031 ,  0.3561750054359436 ,  0.07672759145498276,
  -1.4150773286819458 , -0.7935608625411987 ,  1.2751092910766602 ,
   0.01714191026985645,  0.5205662846565247 ,  0.21525287628173828,
  -0.2752113938331604 , -0.05237847194075584, -0.3020121455192566 ,
  -0.4403022825717926 , -1.944508671760559  , -0.6350194811820984 ,
   0.09128839522600174,  0.43241727352142334, -0.04895520955324173,
   0.0284928884357214 , -0.7487177848815918 ],
 [ 0.12275213748216629,  0.2785799503326416 , -0.04225454851984978,
  -0.3509163558483124 ,  0.7785571217536926 ,  0.9360241889953613 ,
  -0.7474338412284851 ,  0.06221408396959305, -1.1467419862747192 ,
   0.2251848578453064 , -0.24958178400993347,  0.16082388162612915,
  -0.06096192821860313, -1.3648370504379272 ,  0.01195989269763231,
  -0.02330766245722771,  0.8270013332366943 ,  0.10502507537603378,
  -0.8720913529396057 , -0.5349699854850769 ,  0.03840197995305061,
  -0.09366374462842941,  0.5581790208816528 ,  0.3067218065261841 ,
   0.3844832181930542 , -1.0672072172164917 , -1.1391133069992065 ,
   0.7497333884239197 ,  0.5176390409469604 , -0.8191051483154297 ,
  -0.0886835977435112 ,  0.39704909920692444, -0.29564860463142395,
  -0.7070116996765137 ,  0.10556787997484207, -1.8577897548675537 ,
  -1.1268500089645386 ,  0.10614784061908722, -0.18133288621902466,
   0.05840468406677246,  2.350616455078125  ,  0.5613617300987244 ,
  -0.7119837999343872 , -0.4449809193611145 ,  0.378630131483078  ,
   0.5430744290351868 , -1.3670157194137573 ,  0.03856546804308891,
  -0.06985760480165482,  0.27305829524993896, -0.8688749670982361 ,
   0.04453280195593834, -1.602040410041809  ,  0.17810048162937164,
  -0.33828628063201904,  0.421691358089447  , -0.00781418289989233,
  -0.05931883677840233, -0.10353906452655792,  0.3522234857082367 ,
   0.37598273158073425,  0.2844221591949463 , -0.6152527928352356 ,
   0.46004557609558105, -0.29203686118125916,  1.0675218105316162 ,
   0.21988198161125183,  0.09892666339874268,  0.5923916101455688 ,
   0.4489164352416992 ,  0.5744328498840332 ,  0.19437748193740845,
   0.10000944882631302, -0.48923051357269287, -1.1252466440200806 ,
   0.39177796244621277, -1.5151355266571045 ,  0.6607091426849365 ,
  -1.0212255716323853 , -0.1226343959569931 , -0.5482333898544312 ,
  -0.38503992557525635,  0.3804575204849243 , -0.10405450314283371,
  -0.4070952236652374 , -0.13658279180526733,  0.12318738549947739,
   0.39862173795700073,  0.00134239916224033, -0.19477859139442444,
   0.2524285614490509 , -0.3073437511920929 , -0.2990632653236389 ,
   0.04059779644012451,  0.0296439416706562 , -0.7616119384765625 ,
  -0.05146091058850288, -0.16537240147590637,  0.4280540347099304 ,
  -0.04204833507537842, -0.15440908074378967, -1.7533435821533203 ,
   0.2771895229816437 , -0.8802075982093811 ,  0.11486157774925232,
   0.04322361946105957,  0.6124485731124878 , -0.2271031141281128 ,
   0.5597425699234009 ,  0.2923368513584137 , -0.10423287004232407,
   0.2265072762966156 , -1.9670823812484741 ,  0.02346588857471943,
   0.07705696672201157, -0.12686465680599213,  0.1408945620059967 ,
   0.8405131101608276 ,  0.25288161635398865, -0.36015117168426514,
   0.33112069964408875,  0.6601842045783997 ,  0.7118489146232605 ,
   0.01149605866521597, -0.02889993600547314, -0.05538671091198921,
  -0.3909289538860321 ,  0.9446786642074585 ],
 [ 0.07780764251947403, -2.9164352416992188 ,  0.1829187124967575 ,
   0.5594044327735901 ,  0.07329535484313965,  1.6789751052856445 ,
  -2.265951633453369  ,  0.05282407626509666, -0.20012441277503967,
   0.10952868312597275, -0.4939333200454712 , -0.08903733640909195,
  -0.6795789003372192 ,  0.5443106293678284 ,  0.3520609736442566 ,
   0.02786239050328732, -0.5851877927780151 , -0.23686333000659943,
  -1.9971928596496582 , -0.10977791994810104,  0.15887629985809326,
   0.4539664685726166 ,  0.3223973214626312 , -0.6390478610992432 ,
   2.1314408779144287 , -0.3211110234260559 , -0.7588843107223511 ,
  -1.1776553392410278 ,  0.40900158882141113, -1.039758324623108  ,
  -0.1278037428855896 , -0.48327571153640747, -0.6887020468711853 ,
  -1.0798859596252441 , -0.05748412758111954, -0.9495055675506592 ,
   0.5433646440505981 ,  0.20454518496990204,  0.03085215948522091,
   0.16420204937458038, -1.1105023622512817 ,  0.2901860177516937 ,
   0.7743491530418396 ,  1.5373510122299194 , -1.297644853591919  ,
   0.18580745160579681,  1.8830691576004028 ,  0.22719211876392365,
  -0.680319607257843  , -0.0620044507086277 , -0.41038572788238525,
  -0.02043940871953964, -0.35853150486946106, -0.0093116108328104 ,
  -0.4167191982269287 ,  0.3568806052207947 ,  0.2621162533760071 ,
   0.39656776189804077, -0.01167622953653336, -2.243607997894287  ,
   0.0208392757922411 ,  0.2123020440340042 ,  0.2895238995552063 ,
  -1.5011707544326782 , -0.01456349156796932,  0.9999409914016724 ,
  -0.7996714115142822 , -1.052113652229309  , -0.09192512184381485,
  -0.7320933938026428 ,  0.3087121248245239 , -0.5855114459991455 ,
  -0.10749021917581558,  0.11103720217943192,  0.09449446946382523,
   0.10644114762544632, -1.5957950353622437 , -0.23854704201221466,
   0.7102404832839966 ,  0.5827886462211609 , -0.46461060643196106,
   0.40300384163856506,  0.09948521107435226, -0.6263841390609741 ,
   0.22127719223499298, -0.00716700591146946,  0.12389151006937027,
  -1.6881645917892456 , -0.16627375781536102, -0.1973903626203537 ,
   2.4348249435424805 ,  0.8101170063018799 ,  0.7122586965560913 ,
  -0.3361172676086426 ,  0.32673290371894836, -1.9107508659362793 ,
   1.431708574295044  , -0.544227659702301  , -0.6363025903701782 ,
   0.04905758798122406,  0.00998354982584715,  0.8304309248924255 ,
   0.10335054248571396,  0.7843689322471619 , -0.8342983722686768 ,
  -0.3091527223587036 , -0.811111330986023  , -0.28693076968193054,
   0.9541602730751038 ,  0.03169805556535721, -0.826205849647522  ,
   0.7223623991012573 ,  0.42521733045578003, -1.7146821022033691 ,
   0.3620058298110962 , -0.4043126702308655 , -0.21261723339557648,
   0.4770769774913788 ,  0.5540529489517212 , -0.3636569678783417 ,
  -2.0680594444274902 , -3.1575591564178467 , -0.6996740102767944 ,
  -0.7143135666847229 ,  0.3754691779613495 , -0.08050042390823364,
  -0.4627860188484192 , -2.169377565383911  ],
 [ 0.07048606127500534, -0.10025595128536224, -0.1306328922510147 ,
  -0.29807248711586   ,  0.5018215775489807 , -0.21525931358337402,
  -0.6080594062805176 , -0.08111274242401123,  0.00864811800420284,
   0.2420196235179901 ,  0.17548154294490814, -0.20575182139873505,
  -0.3469303548336029 ,  0.5065645575523376 , -0.2639358341693878 ,
  -0.06951958686113358, -0.30001720786094666, -0.05092522501945496,
   0.24918758869171143,  0.22922329604625702, -0.30979472398757935,
   0.10042958706617355, -1.2304399013519287 , -0.9277914762496948 ,
  -1.6575700044631958 , -0.33013784885406494, -0.1996728479862213 ,
   0.2677777111530304 ,  0.07874829322099686,  0.15377993881702423,
  -0.17174863815307617,  0.11819328367710114, -0.26679766178131104,
   0.7208539843559265 ,  0.2857200801372528 , -0.03034077398478985,
   0.0833192765712738 ,  0.13552378118038177,  0.881646990776062  ,
  -0.14372119307518005, -0.710547149181366  , -0.44070324301719666,
  -1.158531665802002  ,  0.7811213135719299 , -0.775410532951355  ,
  -1.5891923904418945 , -1.0308568477630615 ,  0.23358012735843658,
   0.21386519074440002, -0.3455394208431244 , -0.43327176570892334,
  -0.01489310991019011,  0.2561866343021393 , -0.171770840883255  ,
  -0.25669538974761963,  0.4011355936527252 , -0.6884242296218872 ,
   0.05167276412248611, -0.05310345068573952,  0.7058019042015076 ,
  -1.1019575595855713 , -0.14651629328727722, -0.5270839929580688 ,
  -0.4359617829322815 , -0.41205358505249023, -0.11261864006519318,
  -1.0703060626983643 ,  1.0008081197738647 , -0.6106411814689636 ,
  -0.0358150415122509 , -0.9765951633453369 ,  0.4492936432361603 ,
   0.1175759807229042 ,  0.2290797084569931 ,  0.01427697855979204,
  -0.1765086054801941 ,  0.4952069818973541 , -0.2450522631406784 ,
   1.0716160535812378 , -0.04837579652667046,  0.5090411305427551 ,
  -0.2219829112291336 , -0.9242442846298218 , -0.07458572089672089,
   0.11554794013500214, -1.8407046794891357 ,  0.567340075969696  ,
  -0.17883658409118652,  0.08137459307909012, -0.00263347453437746,
  -0.2895163595676422 ,  0.0640617087483406 , -0.687685489654541  ,
  -0.43192484974861145, -0.7707234621047974 , -0.2414264976978302 ,
  -0.2928481996059418 , -0.8111972212791443 , -0.4658009707927704 ,
   0.06887883692979813,  0.05330890044569969, -0.23690073192119598,
   0.15573067963123322,  0.02309224754571915,  0.11759936064481735,
  -0.38714247941970825, -0.36597076058387756, -0.25998443365097046,
  -1.141714334487915  ,  0.22869046032428741,  0.18873868882656097,
  -0.5065223574638367 ,  0.40216150879859924, -0.14301413297653198,
   0.35302358865737915, -1.001997709274292  , -0.7418927550315857 ,
  -0.2819420397281647 ,  0.1029515415430069 , -0.0316510871052742 ,
   0.42768019437789917,  0.5218045711517334 , -0.26783937215805054,
  -0.3662680387496948 ,  0.2819990813732147 , -0.04657062888145447,
  -0.452833354473114  , -0.594822347164154  ],
 [-0.7151083946228027 , -0.42294150590896606,  0.3355282247066498 ,
  -0.881303071975708  , -0.1264832466840744 , -0.21058274805545807,
   1.0238544940948486 ,  0.9235159158706665 , -1.2305123805999756 ,
   0.15485616028308868, -0.7978149056434631 , -0.41807013750076294,
  -1.6833738088607788 , -1.8517307043075562 ,  0.5008552074432373 ,
   1.043412208557129  ,  0.3195628821849823 ,  0.4635787606239319 ,
   1.6875286102294922 ,  0.9734405875205994 ,  1.2996578216552734 ,
  -0.42606908082962036, -0.9890927076339722 ,  1.062429428100586  ,
  -0.718543529510498  , -2.3976452350616455 ,  0.6223147511482239 ,
   0.3925549387931824 , -0.068025603890419  ,  1.1664079427719116 ,
   0.7735961675643921 ,  0.1612578183412552 , -1.166848063468933  ,
  -2.242687463760376  ,  0.20488031208515167, -0.01491690240800381,
   0.5860432386398315 , -0.3462136387825012 ,  0.26260921359062195,
  -0.592064619064331  , -0.5822335481643677 , -0.5265536308288574 ,
  -1.1692345142364502 ,  1.964021921157837  ,  0.902896523475647  ,
  -0.3737480640411377 , -1.3858000040054321 ,  0.1270298957824707 ,
   0.2283119261264801 , -1.4708366394042969 , -0.5825175046920776 ,
   0.06713999807834625,  0.22430701553821564, -0.17900924384593964,
  -0.3119848370552063 ,  1.2863136529922485 ,  0.9870328903198242 ,
   0.18193522095680237,  0.7314279079437256 , -0.07003399729728699,
   0.697662889957428  , -0.43491917848587036, -0.6235446333885193 ,
  -1.0556732416152954 , -0.5160701274871826 , -4.2183518409729    ,
   0.05877622961997986, -0.03765165060758591,  0.4351806044578552 ,
   1.2527471780776978 ,  0.5165422558784485 , -1.4574220180511475 ,
   1.4032121896743774 , -0.00134810130111873,  0.397029310464859  ,
   1.3128951787948608 , -1.421527624130249  ,  1.133422613143921  ,
  -0.6269701719284058 ,  0.21155960857868195, -0.5524224042892456 ,
   0.294696569442749  ,  0.29988497495651245,  0.2022363543510437 ,
  -0.4917350709438324 , -0.29051268100738525,  0.34733664989471436,
  -2.306856632232666  , -0.6223164796829224 ,  1.3794292211532593 ,
  -1.7900967597961426 ,  0.3510189652442932 ,  1.1280725002288818 ,
  -0.3078620731830597 , -0.5450811982154846 , -0.9407373666763306 ,
  -1.244370460510254  ,  1.0630435943603516 , -0.7906557321548462 ,
   1.4814358949661255 , -0.4012727439403534 , -1.3084784746170044 ,
   0.219262957572937  ,  1.7763005495071411 ,  0.3733522593975067 ,
  -2.5045433044433594 , -1.5476669073104858 , -0.38065585494041443,
  -2.4286162853240967 ,  1.2766568660736084 , -0.03057736158370972,
   0.15403403341770172,  0.35170191526412964, -2.156015634536743  ,
  -0.1045108363032341 ,  0.6453782320022583 , -0.6642792224884033 ,
  -2.7369935512542725 ,  1.375488042831421  ,  0.2575211524963379 ,
  -0.9966200590133667 , -0.3862997591495514 , -0.7997835278511047 ,
  -0.7166102528572083 , -0.3395180106163025 , -0.26500657200813293,
  -0.03169340640306473,  0.4554906487464905 ],
 [-0.5375884175300598 ,  0.5142768025398254 ,  0.22620102763175964,
   0.5813756585121155 , -0.7180044651031494 ,  0.7018755674362183 ,
   0.6881839632987976 ,  0.14439146220684052, -0.17440538108348846,
   0.5231899619102478 , -0.7007451057434082 , -0.34896162152290344,
  -0.20920494198799133,  0.12112997472286224,  0.3858683407306671 ,
   0.04278453812003136, -0.28900936245918274, -0.38793087005615234,
  -0.24217665195465088, -0.5662701725959778 ,  0.21333366632461548,
   0.8028700947761536 ,  0.4289275109767914 , -0.39598068594932556,
  -0.6352242231369019 , -1.2105326652526855 ,  0.33071157336235046,
  -0.7597177028656006 ,  1.1706573963165283 ,  0.6497347950935364 ,
  -0.10266274213790894,  0.3186599314212799 ,  0.2551177144050598 ,
   0.27207884192466736, -0.32871875166893005,  0.4182702600955963 ,
   1.2583582401275635 ,  0.1888456791639328 ,  0.8506215810775757 ,
   0.560208797454834  ,  0.891083836555481  , -0.3404569625854492 ,
  -1.0089818239212036 ,  3.0327346324920654 , -0.7104722261428833 ,
  -0.5462742447853088 ,  0.7866449952125549 , -0.6694850325584412 ,
  -0.05317830666899681, -0.4067586362361908 ,  1.0721713304519653 ,
   0.5759273171424866 ,  1.3572559356689453 , -0.10038489103317261,
  -0.23067106306552887,  0.0155677143484354 ,  0.3271547555923462 ,
   0.5552144646644592 ,  0.9236749410629272 ,  0.7130975127220154 ,
   0.39788973331451416,  1.135543704032898  ,  0.62590491771698   ,
   1.1678266525268555 ,  1.166775107383728  ,  0.04043319076299667,
  -0.04061495512723923,  0.11022311449050903, -0.20419123768806458,
   0.7479550838470459 ,  0.2309969663619995 , -0.8203965425491333 ,
   0.41223588585853577,  1.0364339351654053 ,  0.584907054901123  ,
   2.066859722137451  ,  0.800244927406311  , -0.03261113539338112,
   0.9295462369918823 ,  0.42831215262413025,  0.9141156077384949 ,
  -0.1355113983154297 ,  0.36182987689971924, -0.3107021450996399 ,
   0.00639843987300992,  0.2201969176530838 ,  0.11653938889503479,
  -0.18490062654018402, -0.08217080682516098, -0.0917418897151947 ,
   0.33767226338386536,  1.2222530841827393 ,  1.8780279159545898 ,
   0.05530986934900284, -0.33558908104896545,  0.08493304997682571,
  -0.11223608255386353,  0.7016746997833252 , -0.827637791633606  ,
  -0.1995871365070343 , -0.429396390914917  ,  0.12959356606006622,
  -0.1200849860906601 , -0.07446745038032532,  0.617620587348938  ,
   0.9062926173210144 ,  0.516200065612793  ,  0.6359562277793884 ,
  -0.05310693010687828,  0.3925206661224365 ,  0.558728039264679  ,
   0.752638041973114  ,  0.340448260307312  ,  0.20481841266155243,
  -0.13529722392559052, -0.21729037165641785, -0.8723398447036743 ,
   0.5782381892204285 , -0.17980581521987915, -0.2774401009082794 ,
   1.1494626998901367 ,  1.798669695854187  , -0.3151741027832031 ,
   0.30519118905067444,  0.41676825284957886, -0.47273150086402893,
   0.7209633588790894 , -0.5629996657371521 ],
 [-0.28869980573654175, -0.1416337639093399 , -0.18862564861774445,
  -0.18570676445960999, -0.7432681918144226 ,  2.08879017829895   ,
   0.14078295230865479, -0.2871437072753906 , -0.4034721255302429 ,
  -0.3358590304851532 , -1.4198397397994995 ,  0.6320052742958069 ,
  -0.01176886353641748,  0.26712241768836975, -0.09436305612325668,
  -0.0295504555106163 ,  0.34782448410987854, -0.4361982047557831 ,
  -0.11202678829431534, -0.11700914055109024, -0.5220362544059753 ,
   1.4381202459335327 ,  0.11105182766914368, -0.12192972749471664,
  -0.783295750617981  , -0.6972773671150208 ,  0.445427805185318  ,
  -0.08922877907752991, -0.36344480514526367,  0.7450737953186035 ,
  -0.08258027583360672,  0.4150104820728302 , -0.1719912886619568 ,
   1.351180911064148  ,  0.4670708477497101 ,  2.0942740440368652 ,
   1.5024778842926025 ,  0.2902897000312805 ,  1.2768031358718872 ,
  -0.03096082247793674,  0.3915623128414154 , -0.02079761773347855,
   0.16485695540905   ,  1.986907720565796  , -0.03382059186697006,
   0.7274467349052429 ,  0.01953853666782379,  0.00018446380272508,
  -1.1167601346969604 , -0.27108654379844666,  0.7372147440910339 ,
   0.20886515080928802,  0.1702861785888672 ,  0.4421596825122833 ,
  -0.6928665637969971 ,  0.3694020211696625 , -0.7029972672462463 ,
  -0.11964564770460129,  0.1738874912261963 ,  0.7038980722427368 ,
   1.1387434005737305 , -0.23672239482402802,  0.7625111937522888 ,
   0.66781085729599   ,  1.4968442916870117 ,  0.4456096589565277 ,
  -0.6932134628295898 ,  1.0208379030227661 ,  1.156611442565918  ,
   0.5158820748329163 , -0.2483280897140503 ,  0.14412057399749756,
  -0.14292317628860474,  0.594811737537384  ,  0.1252812147140503 ,
   3.23095703125      ,  0.05390610173344612, -0.2226044088602066 ,
   1.354780673980713  ,  0.18491250276565552,  0.04917651042342186,
   0.7889021635055542 ,  0.6924188137054443 , -0.10271494835615158,
  -0.03177645429968834,  0.35467642545700073, -0.05180017277598381,
   0.1877174973487854 , -0.17479005455970764,  0.2317713052034378 ,
  -0.6243228316307068 ,  0.2570972740650177 ,  2.266960620880127  ,
  -0.6564978957176208 ,  0.9762672185897827 ,  0.17053258419036865,
   0.23841515183448792, -0.1520707756280899 ,  0.9567275643348694 ,
   0.15546657145023346, -0.40675967931747437, -0.1961074024438858 ,
   0.5271826386451721 ,  0.7666815519332886 ,  0.3350040912628174 ,
  -0.29782402515411377,  1.2676937580108643 , -0.34911391139030457,
   0.2698201537132263 ,  0.4749886095523834 ,  0.6777269244194031 ,
   0.5307188630104065 ,  3.1400222778320312 , -0.32410556077957153,
   0.32978466153144836, -1.410732626914978  ,  0.21777252852916718,
  -0.15742352604866028, -0.09023551642894745,  0.3723246455192566 ,
   1.8951221704483032 ,  0.8415512442588806 , -0.0117405392229557 ,
  -0.2382986694574356 ,  1.5700701475143433 ,  0.7620261311531067 ,
  -0.30799904465675354, -1.5254424810409546 ],
 [-0.5039206743240356 , -0.00445644231513143, -0.19645170867443085,
   0.06912638247013092, -0.34452083706855774, -0.06319409608840942,
  -0.0913994163274765 , -0.1426333487033844 ,  0.2671317160129547 ,
   0.07485989481210709, -0.4276965856552124 ,  0.4197319447994232 ,
   1.243879795074463  ,  0.24528859555721283, -0.0509520098567009 ,
   0.03655088320374489, -0.8949064612388611 ,  0.32466748356819153,
  -0.7298173904418945 ,  0.16989360749721527,  0.04998760670423508,
   0.5414457321166992 ,  0.4242171347141266 ,  0.28963711857795715,
  -0.5848110318183899 ,  0.5580343008041382 ,  0.8000523447990417 ,
   0.987221896648407  ,  0.12004096806049347, -1.4173860549926758 ,
  -0.0332246832549572 , -0.3350260555744171 ,  0.31684640049934387,
   0.7128432393074036 ,  0.5202483534812927 ,  1.058281421661377  ,
  -0.3611098825931549 ,  0.07388374209403992, -0.15478071570396423,
  -0.6243525147438049 , -0.23990221321582794,  0.46516507863998413,
  -0.7870320677757263 ,  0.442394882440567  ,  0.10962053388357162,
   0.23463080823421478,  0.2959745228290558 , -1.607703685760498  ,
  -0.2516690790653229 ,  1.3817601203918457 , -0.8794365525245667 ,
   0.01897499524056911,  0.3502509295940399 ,  0.3856370449066162 ,
   0.22676591575145721,  0.10940641164779663,  0.384012907743454  ,
  -0.19148778915405273, -0.18689975142478943,  0.6940088272094727 ,
  -0.19105595350265503, -2.850506067276001  , -0.8050780892372131 ,
  -0.6651813983917236 ,  0.5061686038970947 ,  0.37841999530792236,
  -0.8381693363189697 ,  0.05472784489393234, -0.20437407493591309,
   0.33602023124694824, -1.520758032798767  ,  0.7989411354064941 ,
   0.1411047726869583 ,  0.26084744930267334,  0.6530510187149048 ,
   0.2552059292793274 ,  0.1637730896472931 , -0.3257487118244171 ,
  -1.9371248483657837 , -0.8622545599937439 ,  0.37934181094169617,
  -0.562069296836853  , -0.18317045271396637, -1.313538670539856  ,
  -0.14690734446048737, -0.8726670742034912 , -1.348197340965271  ,
   0.12689074873924255, -0.3442908525466919 , -0.18996798992156982,
  -0.6524533629417419 , -0.37180444598197937,  0.46513277292251587,
   0.5772596001625061 , -0.05264674127101898, -0.40609291195869446,
   0.4059771001338959 , -0.5061740279197693 ,  0.3104177415370941 ,
   0.02312777377665043,  0.04813453555107117, -0.6319994330406189 ,
   0.1597749888896942 , -0.5408770442008972 ,  0.5317531824111938 ,
  -0.23408445715904236, -0.0134712178260088 ,  0.21299730241298676,
  -0.144221693277359  ,  0.5764269232749939 ,  0.8541688919067383 ,
   0.5043051838874817 , -2.7967545986175537 ,  0.9552306532859802 ,
  -0.08296887576580048,  0.5115836262702942 , -1.9213308095932007 ,
   0.5343577265739441 ,  0.05301772430539131,  1.1499063968658447 ,
   0.7167375683784485 , -1.5117337703704834 , -0.00142043991945684,
   0.26951050758361816, -0.4619178771972656 , -0.17927631735801697,
  -0.15119872987270355, -0.931128203868866  ],
 [-0.13266097009181976, -2.1231284141540527 ,  0.2566332519054413 ,
   0.24267588555812836,  0.22453628480434418, -0.272384375333786  ,
  -0.6347312331199646 ,  0.279451847076416  ,  0.40881267189979553,
   0.17271701991558075,  0.3957284390926361 , -0.10243207961320877,
   0.05383865535259247,  0.5279006958007812 , -0.5798173546791077 ,
  -0.19350728392601013,  0.03513791784644127,  0.4347732663154602 ,
  -1.1893295049667358 , -0.06203213706612587, -1.4624416828155518 ,
  -0.25039219856262207, -2.731814384460449  , -0.9572956562042236 ,
  -0.21703538298606873, -0.5073275566101074 , -0.6859143972396851 ,
  -0.5528337359428406 , -0.5408803224563599 ,  0.11944106221199036,
   0.02059799060225487,  0.33236074447631836, -1.7492424249649048 ,
  -0.19290103018283844,  0.22377850115299225, -0.7105181217193604 ,
  -0.13743214309215546,  0.24305282533168793, -0.5533953309059143 ,
  -0.2668509781360626 , -3.596104383468628  , -2.332005739212036  ,
  -0.24904970824718475, -0.39145225286483765,  0.5908851027488708 ,
  -1.6607682704925537 , -2.115825653076172  , -0.03366021439433098,
   0.09340110421180725,  0.18962650001049042, -0.11215773969888687,
  -0.15835870802402496, -1.2489383220672607 ,  0.12522093951702118,
   0.412046879529953  , -0.9157269597053528 ,  0.16403642296791077,
  -0.47879767417907715, -1.2203381061553955 , -0.47965577244758606,
  -0.4559588134288788 , -1.1347540616989136 , -2.5385913848876953 ,
  -0.2066141963005066 , -0.5545670390129089 , -0.5732667446136475 ,
  -0.5305993556976318 , -0.07954277098178864, -0.06352701783180237,
  -1.3078217506408691 , -0.41012874245643616, -0.19325749576091766,
  -0.32222074270248413, -0.16714948415756226, -0.367291659116745  ,
  -0.597297728061676  ,  0.09805967658758163, -1.808665156364441  ,
   0.41865450143814087,  0.19878102838993073, -2.106618881225586  ,
  -1.0439289808273315 , -0.5087456703186035 , -2.2781178951263428 ,
  -0.13802875578403473, -0.7093847990036011 ,  0.07422416657209396,
  -0.4074433445930481 ,  0.38965338468551636,  0.00335334241390228,
  -0.5989953875541687 , -0.7916345000267029 , -0.7620548605918884 ,
  -1.4782885313034058 , -0.6305681467056274 ,  0.07489647716283798,
  -0.8738195896148682 , -1.2615993022918701 , -0.179262176156044  ,
   0.01787733472883701, -2.5696396827697754 , -0.1345825493335724 ,
   0.182535320520401  , -0.8616925477981567 , -0.17317594587802887,
   0.11191433668136597, -0.5692330002784729 ,  0.20938704907894135,
   0.10922238230705261, -0.2515684962272644 , -0.03195171803236008,
  -0.5931617617607117 , -0.23174601793289185, -0.8419012427330017 ,
   0.17756783962249756, -0.634371280670166  , -0.18413656949996948,
  -1.694230556488037  , -3.049906015396118  , -0.19262844324111938,
  -0.20187994837760925, -0.3969801962375641 , -0.5967995524406433 ,
  -0.5783751010894775 , -0.4075542688369751 ,  0.16697202622890472,
   0.2150103747844696 , -1.2565218210220337 ],
 [ 0.21383973956108093, -0.26412034034729004, -1.035386085510254  ,
  -0.41223210096359253,  0.3282557427883148 ,  0.5914490818977356 ,
  -0.5444133877754211 ,  0.4272168278694153 ,  0.13994836807250977,
   0.8490638136863708 ,  0.7135263681411743 ,  0.2313075214624405 ,
  -0.23943674564361572, -0.9321687817573547 , -0.1461280733346939 ,
   1.1056615114212036 , -0.8497551679611206 ,  0.15771643817424774,
   0.5663689374923706 ,  0.5986383557319641 , -0.0483756810426712 ,
   0.08323673158884048, -0.668028712272644  ,  0.24814444780349731,
   0.24401918053627014, -1.6723803281784058 , -0.5079947710037231 ,
  -0.3126574456691742 , -0.5068270564079285 , -0.06843561679124832,
  -0.14308643341064453, -0.6478440761566162 ,  0.6130332350730896 ,
  -0.8600431084632874 ,  0.15382619202136993, -1.4173988103866577 ,
  -0.7385589480400085 ,  0.01220226194709539,  0.6061648726463318 ,
  -0.49769142270088196, -0.8712775111198425 , -1.1604288816452026 ,
   0.6737445592880249 ,  0.01855463348329067,  0.08785146474838257,
  -1.2111852169036865 ,  0.05270697176456451,  0.9988791346549988 ,
  -0.5604142546653748 , -0.0632072240114212 ,  0.5485348701477051 ,
  -0.2003222405910492 , -1.476497769355774  ,  0.01430073566734791,
  -0.1075422465801239 , -0.30863144993782043,  0.34721267223358154,
   0.00273619289509952, -0.40756434202194214, -0.9107549786567688 ,
   0.5791437029838562 , -0.11914509534835815, -0.6201888918876648 ,
  -0.6330579519271851 ,  0.3338567614555359 ,  0.5292931795120239 ,
   0.03086370788514614, -0.7174715399742126 , -0.548625111579895  ,
  -0.12774808704853058, -0.9055156707763672 ,  0.00117282045539469,
   0.9673383831977844 , -0.15668316185474396, -0.2531627416610718 ,
  -0.6570872068405151 , -1.7099910974502563 , -0.42899855971336365,
  -1.0543856620788574 , -0.17047595977783203,  0.1809612512588501 ,
   0.06210694462060928, -0.8888947367668152 ,  0.7012935280799866 ,
   0.5189405083656311 ,  0.48404791951179504, -0.14471761882305145,
  -0.36688345670700073, -1.4609918594360352 , -0.48572972416877747,
   0.3061893582344055 ,  0.6708580851554871 , -2.4820544719696045 ,
  -0.27947789430618286,  0.18355129659175873, -0.07676302641630173,
   0.16850414872169495, -1.2713820934295654 , -0.07920720428228378,
  -0.23854127526283264, -1.2644612789154053 , -0.47265827655792236,
  -0.19147716462612152,  0.3000243604183197 ,  0.41202402114868164,
   0.12405471503734589,  0.41880568861961365, -1.3904484510421753 ,
  -2.145232915878296  ,  0.517117440700531  , -0.22596341371536255,
   0.6457441449165344 ,  0.25002795457839966, -0.00774033786728978,
  -0.05587347969412804,  0.7073947787284851 ,  0.6484866738319397 ,
  -0.30859413743019104, -0.7277738451957703 ,  0.28432807326316833,
  -0.3457222878932953 , -0.7402498722076416 , -1.1218258142471313 ,
  -0.077288419008255  ,  0.20674647390842438, -0.5539841055870056 ,
  -0.4641059339046478 , -0.07212792336940765],
 [ 0.15671920776367188, -0.8456168174743652 ,  0.08325430005788803,
  -0.8115074038505554 , -0.5966979265213013 ,  0.27653610706329346,
  -2.2926065921783447 , -0.07824455201625824,  0.4502791464328766 ,
   0.27460378408432007, -0.24528422951698303, -0.0540703609585762 ,
  -0.06589916348457336,  0.6803531646728516 , -0.04013451188802719,
   0.103080615401268  , -0.5204025506973267 ,  0.47313326597213745,
  -1.3383021354675293 , -0.3745726943016052 , -0.24959653615951538,
  -0.2918601930141449 , -1.3166989088058472 , -0.8569868803024292 ,
  -0.6387527585029602 , -0.9919198155403137 , -0.31966492533683777,
  -0.3279731869697571 , -0.6732516884803772 , -0.7719398140907288 ,
   0.01453723572194576, -0.11643995344638824, -0.7499117255210876 ,
   0.5520927309989929 , -0.14876440167427063,  0.24460430443286896,
   0.4852100610733032 ,  0.04623141884803772, -0.5280623435974121 ,
  -0.11004303395748138, -0.7129114866256714 , -0.503218412399292  ,
   0.6531055569648743 ,  0.41526004672050476,  0.4071807861328125 ,
  -0.8895796537399292 ,  1.0112882852554321 , -1.1347726583480835 ,
   0.00715907197445631, -0.22093555331230164, -0.1503440886735916 ,
   0.03247081115841866, -0.5677732825279236 ,  0.00417745253071189,
  -1.1081843376159668 ,  0.46460235118865967,  1.0198813676834106 ,
  -0.05465441942214966,  0.4133055806159973 , -0.2083340287208557 ,
  -0.9598718285560608 , -0.8232389688491821 , -2.077444076538086  ,
  -0.00399767793715   , -0.3886210322380066 ,  0.07485456019639969,
   0.31963327527046204, -0.01487595681101084, -0.9824678897857666 ,
  -0.26967349648475647, -0.3888794183731079 ,  0.12756121158599854,
   0.03449476510286331,  0.21447019279003143, -0.625105082988739  ,
   1.6638785600662231 ,  0.7681360840797424 , -0.728294312953949  ,
  -0.4307151734828949 ,  0.0885351151227951 ,  0.21273310482501984,
  -0.34087827801704407, -0.47192710638046265,  0.02826204337179661,
  -0.05552023276686668, -1.242850422859192  , -0.04305409640073776,
   0.16943322122097015,  0.07755409926176071,  0.08880310505628586,
  -0.25042155385017395, -0.38856324553489685, -1.8979233503341675 ,
  -0.25172799825668335, -0.6971471309661865 , -0.08509831875562668,
  -0.04186194017529488,  0.2513280510902405 , -0.9757137298583984 ,
  -0.04970060661435127,  1.0532925128936768 , -0.9283644556999207 ,
   0.06044919043779373,  0.23595890402793884,  0.07467745989561081,
  -1.5335232019424438 ,  0.40130314230918884, -0.6032147407531738 ,
  -0.3795969784259796 , -0.6542787551879883 , -0.142009437084198  ,
  -0.3863292932510376 ,  0.2106751799583435 , -0.3976577818393707 ,
   0.00586507935076952, -1.9554778337478638 ,  0.13752010464668274,
  -1.478247046470642  , -0.7862576246261597 ,  0.21249976754188538,
  -0.00201276922598481, -0.6871095895767212 , -1.0776163339614868 ,
   0.26822635531425476, -0.5712054967880249 ,  0.12219749391078949,
   0.21673434972763062,  0.3282180726528168 ],
 [ 0.10093468427658081,  0.506748616695404  ,  0.01860928535461426,
   0.6000801920890808 ,  0.5931903123855591 ,  0.2672956883907318 ,
  -0.41633620858192444, -0.3819645345211029 , -1.4616613388061523 ,
  -0.6256970167160034 ,  0.05412624776363373, -0.04161858186125755,
   0.9643720984458923 , -0.1671881079673767 , -1.4976816177368164 ,
   0.12111848592758179,  0.972641110420227  ,  0.2536875903606415 ,
   0.5519765019416809 , -0.4668278098106384 ,  0.01959092542529106,
   0.24279528856277466,  0.07615699619054794, -0.4594567120075226 ,
   0.12040384113788605, -0.4659958481788635 , -0.2096097469329834 ,
  -0.7774510383605957 ,  0.3513435125350952 , -0.9136807918548584 ,
  -0.01201979070901871, -0.3818974196910858 ,  0.26677125692367554,
   0.6256819367408752 , -0.15135899186134338, -0.13097622990608215,
   0.1479022204875946 , -0.1499929428100586 ,  1.1481959819793701 ,
   0.05845203995704651,  0.13731658458709717,  0.5796236991882324 ,
  -0.7096773386001587 ,  0.26673251390457153, -0.27591240406036377,
  -0.8206606507301331 ,  0.14108018577098846, -0.848628044128418  ,
   0.02193473465740681,  1.6486942768096924 , -0.3408306837081909 ,
  -0.18060210347175598,  0.11020908504724503,  0.26800963282585144,
   0.3886193037033081 , -0.5276082158088684 , -0.4457707107067108 ,
   0.8928046822547913 ,  0.5222352147102356 , -0.01001396775245667,
   0.8174325823783875 ,  0.7211297750473022 ,  0.10398213565349579,
  -1.1226060390472412 ,  0.7869998216629028 ,  0.24626432359218597,
   0.7704471945762634 ,  0.1115545928478241 , -0.04319700598716736,
  -0.23321837186813354,  0.9352405667304993 ,  0.6945216059684753 ,
   0.4691481292247772 ,  0.30224233865737915,  2.022768259048462  ,
   0.3008679151535034 ,  0.01771972142159939,  0.40942078828811646,
   0.06541454792022705, -0.11882920563220978, -0.00201623234897852,
  -0.7575160264968872 , -0.00382412411272526,  0.3521132171154022 ,
   0.04018700122833252,  1.1324641704559326 , -0.10098548978567123,
   0.3630063235759735 , -0.02449597045779228, -0.13485434651374817,
  -0.14683079719543457, -0.11854738742113113, -0.8204395174980164 ,
   0.25841856002807617,  0.21884077787399292, -1.2478822469711304 ,
   1.5798366069793701 , -0.05434162169694901,  0.286773681640625  ,
   0.03015225939452648,  0.532601535320282  , -0.521783709526062  ,
   0.06835799664258957,  0.3927595615386963 ,  0.6066065430641174 ,
   0.38065019249916077, -0.657662570476532  , -0.14553214609622955,
   0.1448623538017273 , -0.00204524956643581,  0.4167860746383667 ,
   0.6364306211471558 , -1.8780204057693481 ,  0.14781755208969116,
  -0.25003543496131897,  0.7055553197860718 ,  0.5731313824653625 ,
   1.0720101594924927 ,  0.49564889073371887,  0.4776095747947693 ,
   0.15882070362567902,  0.8461406230926514 ,  0.509655237197876  ,
   0.7091829776763916 ,  0.3454480469226837 ,  0.1587866246700287 ,
  -0.4140198826789856 , -0.08211833983659744],
 [-0.5322659015655518 , -0.188356414437294  ,  0.2145276963710785 ,
  -1.158443808555603  ,  0.3058694005012512 , -0.14819775521755219,
  -0.4474497139453888 , -0.09676574170589447, -0.15184956789016724,
   0.18336492776870728,  0.4590444564819336 ,  0.14393268525600433,
   0.2320769727230072 , -0.11487862467765808,  0.17083217203617096,
   0.01976464129984379, -1.2448749542236328 ,  0.02770695649087429,
   0.43388018012046814, -0.02120648510754108,  0.00727419834583998,
   0.35164129734039307, -0.469868004322052  , -0.7391369938850403 ,
  -0.8830184936523438 ,  0.09138547629117966, -0.6845507621765137 ,
  -0.45882514119148254, -0.06816041469573975, -0.10845304280519485,
  -0.20386438071727753, -0.7569276094436646 ,  0.14622919261455536,
   0.02050205506384373, -0.04390817880630493, -2.6424670219421387 ,
  -0.9373990297317505 ,  0.16602902114391327,  0.419673353433609  ,
   0.15017715096473694,  0.4445793926715851 ,  0.12739142775535583,
  -0.4122774302959442 ,  0.17177142202854156,  0.02351497299969196,
  -0.7437461018562317 ,  0.11711791902780533,  0.25156524777412415,
  -0.898769736289978  , -0.26575422286987305,  0.7558282017707825 ,
   0.443604052066803  , -0.34798377752304077, -0.23404860496520996,
   0.303326278924942  ,  1.177786111831665  ,  0.00217049103230238,
   0.5662863254547119 , -0.20519597828388214,  0.11612273752689362,
  -0.4302060306072235 , -0.17496967315673828, -0.5334179997444153 ,
  -0.3494222164154053 , -0.4298132359981537 , -1.1088125705718994 ,
   0.1442359983921051 ,  0.2255708873271942 ,  0.0323827937245369 ,
  -0.8001160025596619 ,  0.1962709277868271 , -0.8092193007469177 ,
  -0.6408511996269226 ,  0.7491344809532166 , -0.21175651252269745,
   0.3654477894306183 ,  0.35962975025177   , -0.5667522549629211 ,
  -0.0201880969107151 ,  0.5509178638458252 ,  0.13942156732082367,
   0.32861846685409546, -0.4277999699115753 , -0.497873991727829  ,
  -0.28192201256752014,  0.09787827730178833, -1.0603227615356445 ,
   0.2500690221786499 ,  0.32158082723617554,  0.0802459791302681 ,
  -0.3129187524318695 , -0.05353756994009018, -0.1540042757987976 ,
   0.07295817136764526,  0.784471869468689  ,  0.04329559579491615,
   0.19439509510993958, -0.47356700897216797,  0.10661578178405762,
  -0.17733262479305267, -0.46278634667396545, -0.2699893116950989 ,
   0.07963017374277115, -0.41967615485191345, -0.14130377769470215,
  -0.5870137214660645 ,  0.6798996329307556 , -0.01988052763044834,
  -0.19323867559432983, -0.12852534651756287,  0.5254796743392944 ,
   0.2074209302663803 , -0.42608124017715454,  0.28938397765159607,
  -0.01427543722093105, -0.3004860281944275 , -0.37468060851097107,
   0.34666725993156433,  0.38000860810279846, -0.1557200402021408 ,
  -0.20752710103988647,  0.22484271228313446, -0.2310812771320343 ,
  -0.7718887329101562 , -0.03969701379537582,  0.19238243997097015,
  -0.05415621772408485,  0.5637549161911011 ],
 [-0.03863885626196861, -1.433889389038086  ,  0.07131388038396835,
   0.6943621635437012 , -0.7810904383659363 , -0.14830371737480164,
  -0.16478295624256134, -0.13001562654972076, -1.694499135017395  ,
   0.02786329016089439, -0.4404500722885132 ,  0.03747505322098732,
  -0.2422960102558136 , -1.4392715692520142 ,  0.04432965815067291,
  -0.15469643473625183,  0.23499153554439545,  0.07990752905607224,
  -1.1425118446350098 ,  0.12094762921333313,  0.437206506729126  ,
  -0.27929219603538513,  0.19636672735214233, -0.5870547294616699 ,
   0.9219561815261841 ,  0.00870796944946051, -0.0946776494383812 ,
  -0.5130963921546936 ,  0.00863809697329998,  0.03343012556433678,
   0.07366141676902771,  0.18327777087688446, -0.1777065545320511 ,
   0.23656432330608368, -0.08957023918628693, -0.3738762438297272 ,
  -0.02511024288833141, -0.10444000363349915, -0.4104025065898895 ,
   0.28919872641563416,  0.08616584539413452, -0.1969662755727768 ,
  -0.7719078063964844 , -0.20030467212200165,  0.5569425821304321 ,
   0.4661683440208435 , -0.9264699816703796 ,  0.2861795425415039 ,
  -0.05848076194524765,  0.141220360994339  , -0.09738138318061829,
  -0.25048762559890747, -0.3179318904876709 ,  0.01127874851226807,
  -0.3476414084434509 , -0.43306559324264526, -1.140692114830017  ,
   0.3928726315498352 ,  0.4423174262046814 , -1.0199412107467651 ,
   0.4779352843761444 , -1.8581972122192383 ,  0.08890978991985321,
  -0.19500912725925446,  0.4761340022087097 , -0.45715904235839844,
   0.2476244419813156 ,  0.1520148366689682 ,  0.495945543050766  ,
  -1.5269429683685303 , -0.2850884199142456 , -0.1313052624464035 ,
  -0.05930143967270851, -0.8766549825668335 , -1.0907773971557617 ,
   0.06654755026102066, -0.9556492567062378 ,  0.08982566744089127,
  -1.1593130826950073 ,  0.10223624110221863, -0.45877307653427124,
   0.02044293098151684,  0.45051825046539307, -0.27526602149009705,
   0.02821288257837296,  0.05611623078584671,  0.20347172021865845,
  -0.2610169053077698 , -0.01842373423278332,  0.0996832549571991 ,
  -1.033820390701294  , -0.0617564357817173 ,  0.4581754207611084 ,
   0.5288745164871216 , -0.0151375774294138 , -0.0897342711687088 ,
   0.1464894711971283 , -0.23604144155979156, -0.16900165379047394,
   0.04111680388450623,  0.08522938936948776,  0.27480068802833557,
  -0.01860738359391689, -0.9976274967193604 ,  0.0434446781873703 ,
  -0.44933995604515076, -0.14222559332847595, -0.5798859000205994 ,
  -1.3753700256347656 , -0.30930015444755554, -0.18847480416297913,
   0.23057031631469727,  0.5318963527679443 , -0.20813731849193573,
   0.05593214929103851, -2.9012951850891113 ,  0.2417469620704651 ,
   0.47253158688545227, -0.15824773907661438, -0.06533028185367584,
   0.04432123899459839, -0.4210355281829834 ,  0.04285173118114471,
  -0.2260950654745102 , -0.6196923851966858 , -0.02628901600837708,
   0.23904500901699066, -1.4903826713562012 ],
 [-0.23842136561870575,  0.03788629546761513,  0.2758299708366394 ,
   0.1820112019777298 ,  0.12571413815021515, -2.6885464191436768 ,
   1.5992543697357178 ,  0.23555932939052582,  0.40997645258903503,
  -1.166705846786499  ,  1.6366617679595947 , -0.0869981199502945 ,
  -0.6349467039108276 ,  0.10477761179208755, -0.6965358853340149 ,
   0.11178707331418991, -0.9980895519256592 , -0.18603715300559998,
  -1.8366496562957764 ,  1.0528024435043335 ,  0.5898222327232361 ,
  -0.6837525367736816 ,  0.03899205103516579, -0.2824035584926605 ,
   0.4398535192012787 , -0.8970208168029785 ,  0.32801568508148193,
  -2.3259005546569824 , -0.4476356506347656 , -0.2298351377248764 ,
   0.44297412037849426,  0.27708253264427185, -0.06082794815301895,
  -1.1523829698562622 ,  0.02814164012670517, -0.0388217493891716 ,
  -0.810736358165741  , -0.3188261091709137 ,  0.10413194447755814,
   0.08536087721586227, -0.9182886481285095 , -0.2699105441570282 ,
   0.9912632703781128 , -0.05209290608763695,  1.3551040887832642 ,
   0.18286211788654327,  0.5744886994361877 ,  0.8499013185501099 ,
  -0.48511308431625366,  0.5152619481086731 ,  0.00404575793072581,
   0.08063291758298874,  0.6644601821899414 ,  0.01550326775759459,
   0.4489220678806305 , -0.04180849343538284, -0.9681311845779419 ,
   2.1876461505889893 ,  0.2522999346256256 ,  0.6660010814666748 ,
  -0.5925236344337463 ,  0.1394062042236328 ,  0.23359018564224243,
  -0.43608611822128296,  0.6489717960357666 , -0.987358033657074  ,
   0.00986334402114153, -0.16381803154945374,  0.36251020431518555,
  -0.30805227160453796, -0.0951085090637207 , -0.32060706615448   ,
   0.5786334276199341 , -0.13481204211711884,  0.7385704517364502 ,
   0.19767607748508453, -0.03688018769025803,  0.20878562331199646,
   0.8774856925010681 , -2.5312631130218506 ,  0.8861969709396362 ,
  -0.5917861461639404 , -0.04159538820385933, -1.2473024129867554 ,
   1.9602611064910889 , -0.406712144613266  , -3.129621744155884  ,
   0.5462157130241394 ,  0.3303273320198059 , -0.7381840944290161 ,
   0.6848710179328918 ,  0.0774209201335907 , -0.6919646263122559 ,
   0.2118905633687973 , -0.27644461393356323,  0.7136682868003845 ,
  -1.648016333580017  ,  0.4708961546421051 , -0.2314140796661377 ,
  -0.04888078197836876,  0.01456283964216709,  0.14840760827064514,
  -0.40015751123428345, -0.4906555116176605 ,  0.3293479084968567 ,
  -1.5971100330352783 , -0.6105304956436157 , -0.2055331915616989 ,
   0.05224526301026344, -0.32119736075401306, -1.73183012008667   ,
  -0.7272666692733765 , -0.8817839026451111 ,  0.5106624960899353 ,
  -0.2245866358280182 ,  0.3979039192199707 , -0.00337528879754245,
   0.03242303803563118, -0.10644714534282684,  1.6967593431472778 ,
  -0.05981879308819771, -1.8134933710098267 ,  0.26184725761413574,
  -0.10461072623729706, -1.5812838077545166 ,  0.11356815695762634,
   0.5573638081550598 ,  1.2041707038879395 ],
 [-0.6211293339729309 , -0.07383445650339127, -0.1380540430545807 ,
  -0.09206897765398026, -0.9413164854049683 ,  0.2490713745355606 ,
  -0.3287259638309479 , -0.185629203915596  , -0.2715906500816345 ,
   0.18280866742134094,  0.13314910233020782, -0.1954997330904007 ,
   0.32896488904953003, -0.333210825920105  ,  0.21625152230262756,
  -0.05230598151683807,  0.49510666728019714, -0.261785089969635  ,
   0.05223260819911957, -0.6515027284622192 ,  0.11452265828847885,
   0.02730721049010754,  0.06133230775594711, -0.23220805823802948,
  -0.19307167828083038,  0.05428707599639893,  0.250135213136673  ,
  -0.5833377838134766 , -0.06336100399494171, -0.17556023597717285,
   0.03152112290263176,  0.9053493142127991 ,  0.27844658493995667,
   0.23562651872634888, -0.10554063320159912,  0.36115652322769165,
  -0.740588366985321  , -0.03325919806957245, -0.12207380682229996,
   0.3637399971485138 , -0.01857147552073002, -0.3111592233181    ,
  -0.13482312858104706, -0.02934480831027031, -0.02395637333393097,
   0.26764944195747375,  0.05323701724410057,  0.8096771836280823 ,
   0.31967854499816895, -0.7105922102928162 , -0.2723711431026459 ,
  -0.7747609615325928 , -0.30119845271110535,  0.14921154081821442,
  -0.40422743558883667, -0.26909154653549194, -0.8794541358947754 ,
  -0.2397388368844986 , -0.05878672003746033, -0.4895099699497223 ,
  -0.00299532059580088,  0.24358642101287842,  0.12119774520397186,
   0.32631686329841614,  0.03454523161053658, -0.26204371452331543,
   0.02965683862566948, -0.1749453842639923 ,  0.21094897389411926,
  -0.05010756105184555, -0.3388371467590332 ,  0.00709346635267138,
  -0.01375514175742865,  0.474073201417923  ,  1.0595663785934448 ,
   0.00423676520586014,  0.3659392297267914 ,  0.0116356760263443 ,
  -0.04881076142191887,  0.6892935633659363 , -0.2855515778064728 ,
  -0.13402877748012543, -0.10147074609994888, -0.5035917162895203 ,
  -0.0599043071269989 ,  0.3009107708930969 , -0.15491856634616852,
  -0.5054429173469543 , -0.0590958297252655 , -0.6877814531326294 ,
  -0.3713335394859314 , -0.1045295000076294 , -0.6288306713104248 ,
  -0.32137373089790344, -0.00608057575300336,  0.06699322164058685,
   0.3404166102409363 , -0.881180465221405  , -0.13287994265556335,
  -0.14069010317325592,  0.15528278052806854,  0.07239106297492981,
   0.09988965839147568,  0.18300560116767883, -0.6959667205810547 ,
   0.01009899843484163,  0.3260786831378937 ,  0.15486864745616913,
  -0.44213995337486267,  0.23638217151165009,  0.45682135224342346,
  -0.0960787683725357 ,  0.03554366156458855,  0.2270500361919403 ,
   0.04642385989427567,  0.14642082154750824, -0.3762286901473999 ,
  -0.2240947186946869 , -0.18295612931251526,  0.8691250681877136 ,
   0.16050384938716888,  0.06942954659461975, -0.04435239732265472,
  -0.1712476760149002 ,  0.16905944049358368,  0.07137943804264069,
   0.06511219590902328,  0.30875852704048157],
 [ 0.09636776149272919, -0.13524742424488068,  0.0166014302521944 ,
   0.760453462600708  , -0.31711459159851074, -0.28903689980506897,
   0.09323503077030182,  0.14193789660930634,  0.05110900849103928,
   0.03412604704499245, -0.2403600960969925 , -0.10027610510587692,
  -0.15245851874351501, -0.07580363005399704,  0.2919231951236725 ,
  -0.03042600490152836,  0.04687919467687607,  0.39233097434043884,
   0.18283306062221527,  0.13836701214313507,  0.1585872918367386 ,
  -0.7202474474906921 ,  0.03346745669841766, -0.1325022578239441 ,
   0.3158029615879059 , -0.08274935185909271, -0.3559775650501251 ,
   0.2866344451904297 ,  0.1842249184846878 , -0.11106383800506592,
  -0.02217190526425838, -0.00027449382469058, -0.0027116674464196 ,
   0.310336172580719  , -0.25370118021965027, -0.7282888889312744 ,
  -0.00951025076210499,  0.04157821089029312,  0.3986720144748688 ,
   0.12235654890537262,  0.164741650223732  ,  0.06702689826488495,
   0.34488043189048767,  0.05066197365522385,  0.07467435300350189,
  -0.22896701097488403, -0.19838453829288483,  0.07262760400772095,
   0.545568585395813  , -0.24589654803276062,  0.1018512025475502 ,
   0.18176236748695374,  0.04645870625972748, -0.1749698668718338 ,
  -0.3223615288734436 , -0.00489487685263157,  0.28950920701026917,
   0.3157713711261749 , -0.06010199338197708, -0.00533040938898921,
   0.16721026599407196,  0.04100514575839043,  0.05962889268994331,
  -0.25710123777389526,  0.15511862933635712,  0.07215163111686707,
  -0.02879295125603676,  0.22855204343795776, -0.6013739109039307 ,
  -0.3880039155483246 ,  0.3851703703403473 ,  0.08157312870025635,
   0.16121156513690948,  0.20275354385375977, -0.39039698243141174,
  -0.16553466022014618,  0.05811941251158714,  0.07123986631631851,
  -0.44515562057495117, -0.7056215405464172 ,  0.0619417279958725 ,
  -0.1104263961315155 ,  0.13310866057872772,  0.06794418394565582,
   0.02891002967953682,  0.13947784900665283, -0.23688611388206482,
   0.2202589213848114 ,  0.09642872214317322, -0.3548544943332672 ,
  -0.10982261598110199, -0.18649977445602417, -0.03944246098399162,
  -0.03365718945860863,  0.4169375002384186 , -0.03297418728470802,
  -0.1368379443883896 , -0.2689409852027893 , -0.05014699324965477,
  -0.11290615051984787, -0.10550904273986816, -0.42207321524620056,
  -0.03843726590275764, -0.21432161331176758, -0.00188761635217816,
  -0.00371585343964398,  0.00965495128184557, -0.11607861518859863,
   0.2934928834438324 ,  0.03817041963338852,  0.3091106712818146 ,
   0.2329319566488266 ,  0.38919341564178467, -0.15392979979515076,
  -0.18291915953159332, -0.12297140061855316,  0.17935946583747864,
   0.490538626909256  , -0.00182708946522325, -0.08939462900161743,
  -0.2740265727043152 , -0.16103647649288177,  0.08202656358480453,
  -0.3836175799369812 ,  0.04458697140216827,  0.26713255047798157,
   0.20801718533039093, -0.12946492433547974],
 [ 0.04940301924943924, -0.6107497215270996 , -0.39335551857948303,
   0.19237788021564484,  0.5472031235694885 ,  0.7485628128051758 ,
   0.4152601361274719 , -0.05227634310722351,  0.9671762585639954 ,
  -0.4990677237510681 , -1.8133153915405273 , -0.44426316022872925,
  -0.11739350110292435, -0.3285394310951233 ,  0.5838862657546997 ,
   0.06895176321268082, -1.792548418045044  ,  0.20727616548538208,
  -0.3099140524864197 ,  0.6025582551956177 , -0.46931788325309753,
   0.21688604354858398,  0.5375041365623474 , -0.6500336527824402 ,
   1.1471827030181885 ,  0.395913302898407  ,  0.7444770932197571 ,
  -0.00946579128503799, -0.35004791617393494, -0.18904167413711548,
   0.3099553883075714 ,  0.6612120270729065 , -0.27030229568481445,
   0.641160249710083  ,  0.468867689371109  ,  0.8980766534805298 ,
   2.1480562686920166 , -0.2012428492307663 ,  0.26451802253723145,
  -0.1847757250070572 ,  0.09541217982769012,  0.2117481380701065 ,
  -0.7407710552215576 ,  2.626601219177246  , -0.4592021405696869 ,
   0.21524794399738312, -0.9551860690116882 , -0.8698276877403259 ,
  -2.2778451442718506 ,  1.0090771913528442 , -0.25228458642959595,
  -0.06929038465023041,  2.1070375442504883 ,  0.321333110332489  ,
   0.6857258081436157 , -0.37503552436828613, -0.6008826494216919 ,
   0.6490901112556458 , -0.39835771918296814,  0.1117701530456543 ,
   0.6119752526283264 , -0.8761609792709351 ,  0.38613584637641907,
   1.8175427913665771 ,  0.32228514552116394, -0.23897036910057068,
  -0.6848239898681641 , -0.30668091773986816,  0.07264214754104614,
   0.317106157541275  , -1.140497088432312  , -0.06272118538618088,
  -0.3357260525226593 , -1.2263649702072144 ,  1.4414185285568237 ,
   1.1902050971984863 ,  0.8879923224449158 , -0.10607459396123886,
  -0.5446524620056152 , -0.6164029240608215 , -0.8244601488113403 ,
  -0.6410294771194458 , -0.42725634574890137,  0.20125818252563477,
   0.6537171602249146 , -0.3378323018550873 ,  0.4432786703109741 ,
   0.05387831851840019,  0.8539939522743225 ,  0.3410051167011261 ,
  -1.2970869541168213 , -0.9201706647872925 ,  1.3174999952316284 ,
   0.0399864986538887 , -0.8702743053436279 ,  0.3623918890953064 ,
  -0.5566185116767883 ,  0.4759494960308075 , -0.7611953020095825 ,
   0.5673204064369202 , -0.7883692383766174 , -0.07932480424642563,
   0.1948779821395874 , -0.12081310153007507,  0.7569003701210022 ,
   0.6118326783180237 , -0.09303463995456696,  0.21026454865932465,
  -0.2229674905538559 ,  0.37778374552726746, -0.18786919116973877,
   1.8741118907928467 ,  0.17530401051044464,  0.45804402232170105,
   0.06884664297103882, -0.8137161731719971 , -0.985194742679596  ,
   1.3171038627624512 , -0.2956543564796448 ,  1.5423517227172852 ,
   1.3281512260437012 ,  1.9932808876037598 , -0.8985626697540283 ,
   0.6406020522117615 , -0.8777440190315247 ,  0.6139212846755981 ,
  -0.16066543757915497, -1.9878671169281006 ],
 [ 0.25485777854919434,  0.00020313457935117,  0.00839317589998245,
   0.7819730043411255 , -0.6685148477554321 ,  0.33577874302864075,
  -0.01457156985998154, -0.03094723261892796, -1.4080921411514282 ,
   0.1406051218509674 ,  0.23403416574001312, -0.2573496997356415 ,
  -0.995415449142456  , -0.1463511884212494 , -0.5267623662948608 ,
   0.03022104501724243,  0.40414026379585266, -0.47675788402557373,
   0.25796619057655334, -0.1953895390033722 ,  0.01440559048205614,
   0.161056250333786  ,  0.25318434834480286, -0.08615170419216156,
   0.1907326728105545 ,  0.15815922617912292, -0.2969924509525299 ,
  -0.5807870626449585 , -0.296569287776947  , -1.0883402824401855 ,
   0.17439542710781097, -0.12850144505500793,  0.15439404547214508,
   0.374072402715683  ,  0.14709678292274475, -0.21934789419174194,
   1.1304069757461548 , -0.02331100404262543,  0.5487448573112488 ,
  -0.3021095097064972 , -0.07702093571424484, -0.0743752270936966 ,
   0.25458136200904846,  0.19489027559757233,  0.13864365220069885,
  -0.16292595863342285,  0.23975005745887756, -0.11620859801769257,
   0.3710392713546753 , -1.2104474306106567 ,  1.9231606721878052 ,
   0.12576018273830414,  0.02274102903902531,  0.12253731489181519,
   0.29085150361061096, -0.6844995617866516 , -0.09591282159090042,
   0.9047565460205078 ,  0.257659912109375  , -0.38913723826408386,
   0.04909542575478554, -0.29098448157310486, -0.41709914803504944,
  -0.01282408460974693, -0.5386421084403992 ,  0.24716788530349731,
  -0.6472246646881104 ,  0.23355241119861603, -2.4685099124908447 ,
  -0.22787433862686157,  0.03976129367947578,  0.12061630934476852,
  -0.41810449957847595,  0.2453150600194931 , -0.3957313001155853 ,
  -0.07880514860153198,  0.3989688456058502 , -1.0423470735549927 ,
   1.6792947053909302 ,  0.2300155907869339 ,  0.21156267821788788,
   0.36505362391471863, -0.06453105807304382,  0.05174374580383301,
   0.10619646310806274,  0.6411605477333069 ,  0.46690723299980164,
   0.1669590026140213 ,  0.04205414280295372,  0.4315972924232483 ,
  -0.4294593334197998 ,  0.00676680030301213, -0.7414429783821106 ,
   0.2647109925746918 ,  0.828514575958252  ,  0.14099706709384918,
  -0.5612648129463196 , -0.7172333598136902 , -0.7598983645439148 ,
   0.19063663482666016, -0.2690616250038147 ,  0.4915269613265991 ,
   0.11013276875019073, -0.24237146973609924,  0.22437269985675812,
   1.203273057937622  ,  0.38651707768440247, -0.18164131045341492,
  -0.8058969378471375 ,  0.09011107683181763,  0.31230923533439636,
   0.11274527758359909, -0.4873882234096527 , -0.37033048272132874,
   0.46123427152633667, -0.22343334555625916, -0.6172007322311401 ,
   0.20983760058879852, -0.3556457459926605 , -0.4384775161743164 ,
  -0.28408369421958923,  0.9872457385063171 ,  0.21170298755168915,
  -0.8717576265335083 , -0.02284510992467403, -0.42574581503868103,
   0.22564014792442322, -0.23725217580795288],
 [ 0.27550235390663147,  0.992824912071228  , -0.296979159116745  ,
   0.40185171365737915,  0.02464133687317371,  0.03353509306907654,
   1.7228034734725952 ,  0.0097531508654356 ,  0.25999125838279724,
  -0.07840587198734283,  0.6249032020568848 ,  0.14756980538368225,
  -0.07457822561264038, -0.6872400641441345 ,  0.169062539935112  ,
   0.3699559271335602 ,  0.31552886962890625, -0.5418251156806946 ,
   0.24564319849014282,  0.6450361013412476 , -0.7560208439826965 ,
   0.2692592442035675 , -0.40504613518714905,  0.97230464220047   ,
   0.690462589263916  ,  0.267223060131073  , -0.7681662440299988 ,
   0.255314439535141  , -0.12403693795204163, -0.7881842851638794 ,
  -0.15411481261253357,  0.7325656414031982 , -0.40323546528816223,
   0.46072182059288025,  0.4423082768917084 , -1.2126421928405762 ,
  -0.1064748540520668 ,  0.02285260707139969, -0.7282091379165649 ,
  -0.4875138998031616 , -0.32913529872894287, -3.2783291339874268 ,
  -1.5915493965148926 ,  0.7800055146217346 , -0.15790939331054688,
  -0.6436260342597961 ,  0.5042864680290222 , -0.5663397312164307 ,
  -0.00993936788290739,  0.453419029712677  ,  0.07486958801746368,
  -0.11213084310293198, -0.8402031064033508 ,  0.07942485064268112,
   1.222564458847046  , -0.31751513481140137, -0.5750417709350586 ,
   0.2086724042892456 ,  0.21505071222782135, -0.893801212310791  ,
   0.04713472351431847, -0.3797488212585449 , -0.5930442810058594 ,
   0.5066245198249817 , -0.7707318067550659 , -0.5413398742675781 ,
  -0.8429905772209167 , -0.30362164974212646, -2.5157132148742676 ,
   0.40974825620651245, -0.6354463696479797 ,  0.0008282289491035 ,
   0.06754180788993835, -0.6507849097251892 ,  0.15631650388240814,
   0.5938419699668884 ,  0.5117588043212891 , -2.9263861179351807 ,
  -0.9017910957336426 , -0.19352507591247559, -0.40094825625419617,
   0.2978570759296417 ,  0.3030390739440918 , -2.7762064933776855 ,
   0.1941598802804947 , -0.5783120393753052 ,  0.10680563002824783,
  -0.3417819142341614 ,  0.3476923704147339 ,  0.18934562802314758,
   0.06355178356170654, -0.42498719692230225, -1.8910071849822998 ,
   0.1586218923330307 , -0.5613172650337219 , -0.6764730215072632 ,
  -0.5510575175285339 , -1.113199234008789  , -0.31243377923965454,
  -0.14654162526130676, -0.01319484878331423,  0.6499881744384766 ,
  -0.01357147004455328,  0.6433065533638    , -0.11592183262109756,
   0.15268395841121674, -0.2934286296367645 , -2.3042967319488525 ,
  -2.619175672531128  ,  0.7247769832611084 ,  0.2035694569349289 ,
   0.20158995687961578, -1.090691328048706  , -0.6702725887298584 ,
   0.20495571196079254, -0.5185980200767517 ,  0.05993206426501274,
   0.759211540222168  , -0.16771788895130157,  0.3638995289802551 ,
  -2.139218330383301  , -1.0172698497772217 , -0.2040017694234848 ,
  -0.06306885182857513, -0.14218151569366455,  0.08275435119867325,
  -0.48835986852645874, -0.06727458536624908],
 [-0.1932084560394287 ,  0.39968398213386536,  0.07784830778837204,
   0.7965539693832397 ,  0.39534348249435425,  1.1758049726486206 ,
   0.6736809015274048 ,  0.02847953885793686, -0.7192421555519104 ,
  -0.5477786064147949 ,  0.6964846253395081 , -0.2137206792831421 ,
   0.06734864413738251, -0.1929195523262024 , -0.31503698229789734,
  -0.10819296538829803,  1.4084703922271729 ,  0.08069897443056107,
  -0.9130522012710571 , -0.12320300191640854,  0.4155569076538086 ,
  -0.5163342952728271 , -0.15663665533065796,  0.00369646237231791,
   0.19239583611488342, -0.20047864317893982, -1.4637831449508667 ,
   0.5863867998123169 ,  0.16848042607307434,  0.16144753992557526,
   0.15583913028240204,  1.210219144821167  ,  0.3593807816505432 ,
   0.10990720242261887,  0.04468096047639847, -0.407379686832428  ,
   0.3774387240409851 , -0.09502647817134857,  0.5159953832626343 ,
   0.03029796667397022,  1.3453565835952759 , -0.02950182184576988,
   1.06735360622406   , -0.08497461676597595,  0.7839867472648621 ,
   0.9173997640609741 , -1.0560054779052734 , -0.2520498037338257 ,
   0.4027790129184723 ,  0.6164591908454895 , -1.1805375814437866 ,
  -0.02393308468163013, -0.47981032729148865,  0.15290358662605286,
  -1.4558143615722656 , -0.49963629245758057, -0.7728031277656555 ,
  -0.30199575424194336, -0.03209547698497772,  0.8981888890266418 ,
  -0.79413241147995   , -0.23504240810871124, -0.03547104820609093,
  -1.1654359102249146 ,  0.5498766303062439 ,  0.7088274955749512 ,
   0.7228586077690125 , -0.2919202744960785 ,  2.1115636825561523 ,
  -0.00191592599730939, -0.6607286930084229 ,  0.19433827698230743,
  -0.05019816756248474,  0.5909610986709595 ,  1.6363080739974976 ,
  -0.08816856145858765, -0.9496760964393616 , -0.6818935871124268 ,
  -0.5637994408607483 , -0.25333136320114136, -0.194166362285614  ,
   0.5633953213691711 ,  0.11842681467533112, -0.23617640137672424,
  -0.03994925692677498, -0.27328142523765564, -0.09465619921684265,
   0.06005265191197395,  0.04823656380176544,  0.0972980186343193 ,
   0.4925142824649811 ,  0.7153846025466919 ,  0.271441251039505  ,
   0.08776737004518509,  0.14999131858348846, -0.18052834272384644,
  -1.7839995622634888 , -0.369343638420105  , -0.23806443810462952,
   0.01902914233505726, -0.4020921587944031 ,  0.9025119543075562 ,
   0.02292744442820549,  0.03012537956237793,  0.10526606440544128,
   0.18876604735851288,  0.17662253975868225,  0.04321013018488884,
   0.17479048669338226,  0.06668105721473694, -0.4611504077911377 ,
  -1.0021212100982666 ,  2.1510355472564697 , -0.2948324978351593 ,
   0.07293105125427246, -1.198477029800415  ,  0.17858943343162537,
   0.6056471467018127 ,  0.22203564643859863,  1.1503819227218628 ,
   0.25840896368026733,  0.04849370941519737,  0.11200691759586334,
  -0.25219887495040894, -0.3426337242126465 ,  0.09908588230609894,
   0.5418061017990112 ,  0.45935341715812683],
 [-0.13708844780921936, -0.6798543334007263 , -0.2623175084590912 ,
  -0.3182888627052307 ,  0.250363290309906  ,  0.23263274133205414,
  -0.7173736095428467 ,  0.8033668398857117 , -0.3815634250640869 ,
   0.19085104763507843,  0.13247866928577423, -0.04816751554608345,
   0.13841737806797028, -1.2413393259048462 ,  0.11371327191591263,
   0.13254624605178833,  0.21340411901474   , -0.3564750850200653 ,
   0.05073240771889687,  0.02736714296042919, -0.08638957887887955,
   0.00465821335092187,  0.16116608679294586,  0.10942400246858597,
   0.30252939462661743, -1.401168942451477  ,  0.17198997735977173,
   0.5845906138420105 ,  0.08919927477836609,  0.742424488067627  ,
   0.38303840160369873, -1.2972770929336548 ,  0.06660500913858414,
   0.09484758973121643, -0.3482542932033539 , -0.4658576548099518 ,
  -0.00635439064353704, -0.278511643409729  ,  0.1327226310968399 ,
   0.03505047410726547, -0.26330113410949707, -0.192718043923378  ,
  -0.42919471859931946,  0.07532352954149246,  0.8163148760795593 ,
  -0.36653614044189453,  0.18529574573040009,  0.16880954802036285,
  -0.15338465571403503, -0.8292137980461121 ,  0.4205051064491272 ,
   1.2561511993408203 , -0.27366432547569275, -0.5261911749839783 ,
   0.20205143094062805,  0.1657027304172516 ,  0.2664407789707184 ,
   0.80303555727005   , -0.23993517458438873, -0.11625434458255768,
   0.3262980282306671 ,  0.8921498656272888 , -0.11550384759902954,
  -0.7357057929039001 ,  0.7057252526283264 ,  0.7005341053009033 ,
   0.16246968507766724,  0.2533770799636841 , -0.3330531418323517 ,
  -0.15853771567344666,  0.49227985739707947, -0.0758465901017189 ,
   0.2324855625629425 ,  0.04390953108668327,  0.22540035843849182,
  -0.0978308767080307 , -1.2960467338562012 , -0.17655877768993378,
   0.446256548166275  ,  0.21456298232078552, -0.41492509841918945,
  -0.1979343444108963 ,  0.15407142043113708,  0.13409094512462616,
  -0.04343955963850021, -0.7619493007659912 ,  0.1474635899066925 ,
   0.6934221982955933 ,  0.600860595703125  , -0.147536039352417  ,
  -0.08882175385951996, -0.05903567001223564, -0.9448167085647583 ,
   0.25302547216415405,  0.3853493630886078 ,  0.3897853493690491 ,
  -0.5814366936683655 , -1.4698020219802856 , -0.6261121034622192 ,
   0.63539057970047   , -0.59603351354599   ,  0.20654486119747162,
   0.6392474174499512 ,  0.4346877634525299 ,  0.6017060279846191 ,
   0.5993339419364929 ,  0.47216007113456726, -0.5493766069412231 ,
  -1.7530055046081543 ,  0.30568405985832214,  1.0761886835098267 ,
   0.3843766152858734 ,  0.314818412065506  , -0.3185129165649414 ,
   0.04179632291197777, -0.543480634689331  , -0.23356777429580688,
   0.14129212498664856, -0.6086408495903015 ,  0.46062347292900085,
  -0.8091747164726257 ,  0.21733446419239044, -0.08007276058197021,
   0.36026138067245483,  0.03078861720860004,  1.4856274127960205 ,
  -0.23584814369678497,  0.41098397970199585],
 [-0.3672099709510803 ,  0.10100001841783524,  0.00030657832394354,
   0.17553427815437317,  0.1404118537902832 ,  0.20083047449588776,
  -0.03843889385461807,  0.4094347059726715 ,  0.2967003583908081 ,
   0.01027947850525379, -0.6332015991210938 , -0.6168603301048279 ,
   0.15333972871303558, -0.29845452308654785, -0.27163007855415344,
   0.40294432640075684, -1.6604561805725098 , -0.05612896010279655,
  -0.8297154903411865 ,  0.05467478930950165,  0.1897028535604477 ,
   0.08838015794754028,  0.06062290444970131,  0.5605988502502441 ,
  -0.6253505349159241 ,  0.5011549592018127 ,  0.4148317575454712 ,
   0.3531757891178131 , -0.12360815703868866, -0.11982956528663635,
  -1.1792460680007935 , -0.3204220235347748 ,  0.4398829936981201 ,
  -0.19873292744159698,  0.09004154801368713, -0.2669987082481384 ,
   0.1832660436630249 ,  0.45401015877723694,  0.3479955196380615 ,
   0.3847882151603699 ,  0.19249926507472992, -0.10264360904693604,
   0.3652553856372833 ,  1.253048062324524  ,  0.11499746888875961,
   0.37583300471305847,  0.140506312251091  , -0.16213607788085938,
  -1.3586848974227905 , -0.02293053455650806,  0.38386762142181396,
  -0.19586068391799927,  0.4184364080429077 , -0.05713231861591339,
   0.4837196171283722 , -0.692420244216919  ,  0.05689997225999832,
  -0.23352891206741333, -1.0360748767852783 , -0.21138466894626617,
   0.10617854446172714, -0.04801999405026436, -0.2605477571487427 ,
   0.20254459977149963, -0.6941880583763123 ,  0.09695424139499664,
  -0.7820333242416382 , -0.09053900092840195, -0.10146742314100266,
  -0.1292123794555664 , -1.7551156282424927 , -0.30792540311813354,
   0.2283240705728531 ,  0.16331429779529572,  0.02915421314537525,
   0.8464890122413635 , -0.35792306065559387, -0.773415207862854  ,
   0.6487001776695251 , -1.731139063835144  , -0.3935890197753906 ,
  -0.09501457959413528, -0.122751384973526  , -0.32597705721855164,
  -0.03509663417935371, -0.21227644383907318,  0.17665033042430878,
  -1.350224494934082  ,  0.08727332949638367, -1.42322838306427   ,
   0.3074210584163666 , -0.47315284609794617, -0.17859894037246704,
  -0.31764256954193115, -0.6834022402763367 ,  0.5639280676841736 ,
  -1.1363377571105957 , -0.45206695795059204,  0.06745775789022446,
  -0.7139811515808105 , -1.229264736175537  , -0.41130977869033813,
  -0.03646619245409966, -0.9985065460205078 , -0.4635842442512512 ,
  -0.6122156977653503 , -0.47472840547561646, -0.32088738679885864,
   0.12640753388404846,  0.6990652084350586 , -0.5237135887145996 ,
   0.1625596582889557 , -0.9191524386405945 ,  0.235471710562706  ,
   0.18181303143501282, -0.07226947695016861,  1.1929152011871338 ,
  -0.6726534962654114 , -2.624431848526001  , -0.03154833242297173,
   0.632190465927124  , -0.39521923661231995, -1.1889219284057617 ,
  -0.11217585951089859, -0.0152262644842267 ,  0.8065494298934937 ,
  -0.2782660126686096 ,  0.6314919590950012 ],
 [-0.02756832167506218,  0.5119991302490234 ,  0.2596346139907837 ,
   0.06845151633024216, -1.03761887550354   , -0.07350108027458191,
   0.24613843858242035, -0.06739549338817596,  0.68937748670578   ,
  -0.28956419229507446, -0.5143709182739258 ,  0.03964519873261452,
   0.5472885370254517 ,  0.18137596547603607, -0.03788832575082779,
  -0.7809976935386658 ,  0.1898004114627838 , -0.1674555242061615 ,
  -1.8618162870407104 , -0.5345358848571777 ,  0.5239092707633972 ,
  -0.25769829750061035,  0.8057525157928467 ,  1.1160523891448975 ,
   0.11822938174009323,  0.2519257068634033 ,  0.02780384384095669,
   0.5519445538520813 , -1.4134454727172852 , -1.4085543155670166 ,
   0.12438835203647614, -0.14027203619480133,  0.5628134608268738 ,
  -0.8077709078788757 , -0.14456677436828613, -2.6300885677337646 ,
  -0.38778355717658997, -0.0581575408577919 ,  0.4346655011177063 ,
   0.3335631191730499 , -1.3609956502914429 , -0.308621346950531  ,
  -0.358285129070282  ,  0.34664174914360046,  0.9711750149726868 ,
   1.3424748182296753 ,  0.10500091314315796,  0.505238950252533  ,
   0.31622597575187683, -0.31638774275779724,  0.2207964062690735 ,
  -0.32705947756767273, -0.08563414216041565, -0.06127067282795906,
   0.0709063932299614 ,  0.0386083647608757 , -0.1568981558084488 ,
  -0.18548350036144257, -0.01903149858117104,  0.37419942021369934,
  -0.07847660779953003, -0.9372865557670593 ,  0.7495359182357788 ,
  -0.08034028112888336, -0.14215746521949768,  0.8437574505805969 ,
  -1.4569414854049683 , -1.189516544342041  , -0.5340421199798584 ,
  -0.6049445867538452 , -0.4138028621673584 , -0.4273858070373535 ,
   0.1496751308441162 , -2.8732187747955322 , -0.2986869812011719 ,
  -1.5294996500015259 ,  0.6596286296844482 ,  1.1477460861206055 ,
  -0.07338599115610123, -0.9375348687171936 , -0.2900857627391815 ,
  -1.168906569480896  ,  0.3071500062942505 , -1.275246024131775  ,
  -0.01701340451836586, -0.1335587501525879 ,  0.102705217897892  ,
   0.5773565769195557 , -0.4946082532405853 ,  0.11039529740810394,
  -1.0710691213607788 , -0.28765755891799927,  0.7312946915626526 ,
   0.13332998752593994,  1.0303938388824463 , -1.1649982929229736 ,
   0.37127918004989624,  0.5709501504898071 , -0.25475430488586426,
   0.19285540282726288,  1.0308054685592651 ,  0.05673397704958916,
   0.14182019233703613, -0.11447735875844955,  0.22561229765415192,
   0.23499347269535065,  0.0214159432798624 , -0.9657638072967529 ,
  -1.5709456205368042 , -1.1898388862609863 ,  0.5299960970878601 ,
  -0.2060733288526535 , -0.3515818417072296 ,  0.6758647561073303 ,
   0.2857474088668823 , -2.1576266288757324 ,  0.5967977046966553 ,
   1.1849831342697144 ,  0.34898343682289124, -0.7395198345184326 ,
  -1.7573840618133545 ,  0.6194511651992798 , -0.49188104271888733,
   0.6063060760498047 , -0.5247364640235901 ,  0.05522441118955612,
  -0.6204603314399719 , -0.9575750827789307 ],
 [-0.12687818706035614, -0.15573568642139435, -0.3312070369720459 ,
   0.3704712688922882 , -0.7470460534095764 , -0.05659194663167   ,
  -0.07485737651586533,  0.14790426194667816, -0.7206680774688721 ,
  -1.3623926639556885 , -0.43186014890670776, -0.10937908291816711,
   1.392425298690796  , -0.29038599133491516, -1.2029293775558472 ,
   0.6375170946121216 , -0.560005784034729  , -0.3826095461845398 ,
  -2.161553144454956  ,  0.09192538261413574, -0.49712541699409485,
   0.5378097295761108 , -0.6664798259735107 , -0.5344830751419067 ,
   0.28508174419403076, -0.14620186388492584, -0.0844007283449173 ,
  -0.15338455140590668, -0.7706694602966309 , -0.5591066479682922 ,
   1.97208833694458   ,  0.8297001719474792 ,  0.1959846317768097 ,
   0.20397888123989105,  0.33301296830177307, -0.7000780701637268 ,
  -1.389069676399231  , -0.7670557498931885 ,  0.4614216089248657 ,
  -0.7564507722854614 , -0.4105640649795532 , -0.43976783752441406,
  -0.06690271198749542, -0.2986994981765747 ,  0.10750512778759003,
   0.2806943655014038 , -0.32515260577201843, -1.0992980003356934 ,
  -0.33743539452552795,  1.4445443153381348 , -0.03940161690115929,
   0.04882889986038208,  0.06332491338253021,  2.2440762519836426 ,
  -0.7049064636230469 ,  0.4123844802379608 , -0.8179003000259399 ,
   2.0942347049713135 ,  0.5675296187400818 ,  0.5785079598426819 ,
   0.7289311289787292 , -1.2872775793075562 , -0.00774142611771822,
  -2.481898307800293  , -1.6479177474975586 , -0.5978430509567261 ,
  -0.26965582370758057, -0.12922410666942596,  1.0906599760055542 ,
   0.11475790292024612, -0.5756517052650452 , -0.49680715799331665,
  -0.7799228429794312 , -0.23110146820545197,  1.90511953830719   ,
  -1.1815251111984253 ,  0.18460656702518463, -0.17116782069206238,
  -0.9011022448539734 , -0.30506110191345215,  0.3746888041496277 ,
  -1.0310810804367065 , -0.03707636147737503, -0.5000969171524048 ,
   2.0541810989379883 , -0.3042266368865967 , -0.5782026648521423 ,
  -0.3068808615207672 , -0.7257785797119141 ,  0.5136793851852417 ,
  -0.05060435459017754, -0.15385979413986206, -0.9958130121231079 ,
   0.08910501003265381, -0.22653305530548096, -0.37921711802482605,
  -0.02386187575757504,  0.303272545337677  , -0.9000409245491028 ,
   1.776734471321106  , -1.024619221687317  , -0.09004479646682739,
  -0.3659743666648865 , -0.6249273419380188 , -2.086427927017212  ,
   0.03611557558178902, -1.6807559728622437 ,  0.4109634757041931 ,
  -0.3525867164134979 ,  0.41163936257362366, -0.42423564195632935,
   0.2932971119880676 ,  0.4998250901699066 , -0.15565605461597443,
  -0.497092604637146  , -0.4920215606689453 , -0.22126245498657227,
  -0.36367106437683105, -0.5300649404525757 ,  1.7464007139205933 ,
  -1.993632435798645  , -0.5178970694541931 ,  0.0718795657157898 ,
  -0.5061957240104675 , -2.449991226196289  , -1.3568795919418335 ,
  -0.35698652267456055, -0.5929011702537537 ],
 [ 0.11575029045343399, -0.2853337526321411 , -0.05098222941160202,
  -0.7193339467048645 , -0.14100433886051178, -0.5020217895507812 ,
   0.3042812943458557 ,  0.13016550242900848,  0.9926788806915283 ,
   0.1404949575662613 ,  0.4662027657032013 ,  0.24995823204517365,
   0.1824008971452713 , -0.5282911658287048 , -0.5810045003890991 ,
   0.06079269200563431, -0.31206321716308594,  0.40570974349975586,
  -0.20773638784885406,  0.31832829117774963, -0.10085448622703552,
   0.39799320697784424, -0.5361229181289673 ,  1.302219271659851  ,
  -1.8856383562088013 ,  2.008174419403076  ,  0.3084616959095001 ,
  -1.0311027765274048 ,  0.6444653868675232 , -0.13740970194339752,
   0.02007149904966354, -0.15285271406173706, -1.2888615131378174 ,
  -0.3043343424797058 ,  0.150640606880188  , -0.8638580441474915 ,
  -0.5496401190757751 ,  0.09106162190437317, -0.29026293754577637,
   0.10724557191133499, -0.25675278902053833, -0.1426665186882019 ,
   0.41120725870132446,  0.9363916516304016 , -1.2706979513168335 ,
   0.37973588705062866,  0.49886828660964966, -0.7048161625862122 ,
   0.42372721433639526,  0.7984771728515625 , -1.207555890083313  ,
   0.07899294793605804, -0.9856041669845581 ,  0.11407243460416794,
   0.7940940260887146 ,  1.0276579856872559 ,  0.47596776485443115,
  -0.2072593867778778 , -0.6590829491615295 ,  1.0245617628097534 ,
  -1.1248793601989746 , -1.223905086517334  ,  0.40775197744369507,
   0.7010195255279541 , -0.07592393457889557,  0.16664524376392365,
  -0.7560322284698486 , -0.8960410952568054 , -0.02774350717663765,
   1.321613073348999  , -0.8253493905067444 ,  0.8504524827003479 ,
  -0.29683801531791687, -0.48397669196128845, -0.2803475558757782 ,
  -0.21856625378131866, -0.13132569193840027, -0.695151150226593  ,
   0.1203172504901886 , -0.3440369665622711 ,  0.834300696849823  ,
  -0.9052785038948059 ,  0.45197638869285583, -0.2879066467285156 ,
  -0.16688162088394165, -1.6247681379318237 ,  0.4859069585800171 ,
  -0.05268875882029533,  0.01149446982890368,  0.0439748726785183 ,
   0.7874007821083069 ,  0.7955186367034912 ,  0.03541802242398262,
  -0.4581258296966553 ,  1.4889556169509888 , -0.3036632239818573 ,
   1.878271222114563  , -1.816663384437561  ,  0.0219514649361372 ,
   0.05940043926239014, -0.6014147400856018 , -2.520031690597534  ,
   0.08317971229553223,  0.13203886151313782,  0.32506263256073   ,
  -0.39280882477760315, -0.4023109972476959 , -0.46221157908439636,
   0.702946126461029  ,  0.21590906381607056, -0.2444005310535431 ,
   0.13339538872241974,  0.13285446166992188, -0.9863516092300415 ,
   0.00716255139559507,  0.7022015452384949 ,  1.0071146488189697 ,
   0.6951165199279785 , -0.4922071397304535 ,  0.1885569840669632 ,
   0.03960734978318214, -3.332045555114746  , -2.357736825942993  ,
   1.1547967195510864 , -0.1993095725774765 ,  0.03624015673995018,
   0.17690575122833252, -1.0085020065307617 ],
 [-0.1496029645204544 ,  0.0192671250551939 , -0.06098521500825882,
   0.5144511461257935 ,  0.3094131052494049 ,  0.19713322818279266,
   0.529886782169342  ,  0.05699975043535233,  0.5185199975967407 ,
  -0.15726050734519958, -0.04733226075768471, -0.12839867174625397,
   0.3627951741218567 , -0.07382640987634659,  0.18905684351921082,
   0.02464212663471699, -0.23741090297698975,  0.04865342378616333,
   0.09985805302858353,  0.1548316478729248 , -0.0697680190205574 ,
  -0.2625969648361206 , -0.10908398777246475, -0.08916955441236496,
   0.5948388576507568 ,  0.3659423589706421 , -0.18945516645908356,
   0.31492453813552856, -0.5836841464042664 , -0.497819185256958  ,
   0.05564914643764496,  0.21045216917991638,  0.00211768690496683,
  -0.2440544217824936 , -0.07051950693130493, -1.1691499948501587 ,
   0.02563119493424892, -0.09947124123573303, -0.03024226613342762,
  -0.02509441412985325,  0.01647400483489037, -0.02740766108036041,
   0.14911021292209625,  0.10861487686634064, -0.02525422163307667,
   0.16493943333625793, -0.01680254563689232,  0.13908372819423676,
  -0.189988374710083  ,  0.6765480041503906 , -0.08982383459806442,
   0.32002875208854675,  0.09038539975881577,  0.06105267256498337,
   0.05812577530741692, -0.18786197900772095,  0.09921672195196152,
  -0.5534899830818176 ,  0.10315491259098053, -0.4184493124485016 ,
  -0.30573320388793945, -0.04484880715608597,  0.06052877753973007,
   0.0005239121383056 , -0.01743284240365028, -0.22294194996356964,
   0.0732743889093399 , -0.16778749227523804, -0.4021153151988983 ,
   0.18850144743919373, -0.03463925048708916, -0.2065429538488388 ,
  -0.08777549862861633, -0.10119394958019257, -0.20440161228179932,
   0.10043292492628098, -0.04977984726428986, -0.01031526923179626,
   0.5208204388618469 , -0.4300992786884308 ,  0.382016658782959  ,
   0.05272142216563225, -0.11721876263618469,  0.08787752687931061,
  -0.03252262249588966, -0.11044930666685104,  0.18501389026641846,
   0.01054384931921959,  0.1317373365163803 ,  0.00878164265304804,
  -0.11241231113672256, -0.3496398627758026 ,  0.14822669327259064,
   0.07660559564828873,  0.3232116103172302 , -0.31965529918670654,
  -0.35613951086997986, -0.34480562806129456,  0.171751469373703  ,
  -0.02775807492434978,  0.0764584094285965 , -0.2495625764131546 ,
   0.006445721257478  , -0.2766324579715729 ,  0.16921937465667725,
   0.05758160725235939,  0.058194600045681  , -0.28530028462409973,
   0.09252069145441055, -0.07204185426235199,  0.15895530581474304,
  -0.09655096381902695, -0.06051594018936157, -0.04574834182858467,
   0.00556133734062314, -0.0600932389497757 , -0.19333097338676453,
  -0.06377747654914856, -0.03763921931385994, -0.07896605134010315,
  -0.09125139564275742,  0.05786103010177612, -0.02510076202452183,
   0.0174073837697506 ,  0.18103305995464325, -0.11572962254285812,
  -0.6205023527145386 ,  0.16742274165153503],
 [-0.4577409327030182 ,  0.21338877081871033,  0.00850073248147964,
  -0.3895348310470581 , -1.4618972539901733 , -1.305112361907959  ,
   0.3441868722438812 ,  0.02208649553358555,  0.07810679078102112,
  -0.09171158075332642, -0.5805501937866211 , -0.07951632887125015,
   0.27230018377304077, -0.00431091943755746, -0.03262913599610329,
  -0.0511140264570713 , -1.2456854581832886 , -0.02869312278926373,
  -0.28124845027923584,  0.00854187924414873,  0.3570123612880707 ,
   0.04995434358716011, -0.12308460474014282, -0.00948514044284821,
  -0.5916126370429993 , -0.7036016583442688 ,  0.3264335095882416 ,
   0.2900759279727936 , -0.24780936539173126,  0.01664503663778305,
   0.00613568536937237, -0.7437106370925903 ,  0.281537264585495  ,
  -0.04437712579965591, -0.1102210059762001 , -0.5190194845199585 ,
  -0.45218759775161743,  0.04128601402044296, -0.37889865040779114,
  -0.17061686515808105, -1.1327052116394043 , -0.24943338334560394,
  -0.12247266620397568, -0.17915254831314087, -2.0827105045318604 ,
  -0.2202131152153015 , -0.5908512473106384 ,  0.4585341811180115 ,
  -0.21746458113193512,  0.07512492686510086, -1.0286825895309448 ,
  -0.18088188767433167, -0.06875165551900864,  0.055926114320755  ,
   0.11475528031587601, -0.0920373946428299 ,  0.3837102949619293 ,
  -0.15842947363853455,  0.05992897599935532, -0.1558462679386139 ,
   0.2035973072052002 , -0.34220948815345764, -0.43872132897377014,
  -0.07508033514022827, -0.41100752353668213,  0.5971761345863342 ,
  -0.546953022480011  , -0.7162064909934998 , -0.3521967828273773 ,
   0.2791787385940552 ,  0.0901612788438797 ,  0.4020419120788574 ,
   0.07269740849733353, -0.02367235533893108,  0.6212454438209534 ,
  -0.21787860989570618,  0.2878960371017456 ,  0.06811898946762085,
  -0.34139522910118103,  0.13987410068511963, -0.4515174627304077 ,
  -2.893869161605835  ,  0.24253976345062256, -0.5742103457450867 ,
   0.0426260344684124 , -0.00309210200794041,  0.02460280619561672,
  -0.2044527232646942 ,  0.06577742844820023,  0.15585553646087646,
  -0.09137728810310364,  0.24043525755405426, -2.345728874206543  ,
   0.32010823488235474,  0.12547427415847778,  0.08020059019327164,
  -1.1711359024047852 , -0.06194280087947845,  0.2729201316833496 ,
  -0.16874592006206512,  0.21801292896270752, -0.5678501129150391 ,
  -0.10933772474527359, -0.04587845504283905, -0.5505917072296143 ,
   0.30800971388816833, -1.2398202419281006 ,  0.5528228282928467 ,
  -0.07358425855636597, -0.4516783356666565 ,  0.04532835632562637,
  -0.16825172305107117, -0.8891357779502869 , -0.283971905708313  ,
  -0.02977298758924007, -0.641375720500946  ,  0.07042770087718964,
   0.3950379490852356 , -0.5147193670272827 , -0.3497309684753418 ,
  -0.4589460492134094 ,  0.21363089978694916, -0.07158399373292923,
  -0.38203006982803345, -0.8240959048271179 ,  0.18449948728084564,
   0.877960741519928  ,  0.22271360456943512],
 [-0.4631345570087433 , -0.33437269926071167, -0.01471655815839767,
  -1.0921368598937988 , -0.9554033279418945 , -3.0565383434295654 ,
  -1.124383568763733  , -0.09318351745605469, -0.38507992029190063,
  -0.01262756995856762, -0.08061433583498001, -0.31125128269195557,
   0.23312298953533173, -0.4064798653125763 , -0.17711025476455688,
  -0.02633219584822655,  0.07583393156528473,  0.3379395306110382 ,
   0.21115724742412567, -0.30420905351638794, -0.26674792170524597,
   0.03040989115834236, -0.45083528757095337, -1.4737902879714966 ,
  -0.8090060353279114 , -0.9929966926574707 , -0.22682803869247437,
  -2.4784538745880127 ,  0.04281248897314072, -0.19695636630058289,
  -0.15434783697128296, -0.6318860054016113 , -0.3705439567565918 ,
  -0.44008028507232666, -0.49476319551467896, -1.2891180515289307 ,
   0.4154266119003296 ,  0.1371370255947113 , -0.20956206321716309,
   0.1093456894159317 , -0.14834602177143097,  0.3183882534503937 ,
  -0.20081651210784912, -0.24664165079593658, -1.6196215152740479 ,
   0.09556441009044647, -0.8026502132415771 ,  0.04424217343330383,
  -1.0972124338150024 ,  0.27464598417282104, -0.0493919812142849 ,
   0.1600135862827301 , -0.02383271604776382,  0.06469061970710754,
  -0.4674522876739502 , -0.5307645201683044 ,  0.26969555020332336,
  -0.13138510286808014,  0.1749061793088913 ,  0.3887564241886139 ,
   0.4626156687736511 , -0.5434324741363525 , -0.5232056379318237 ,
  -0.4492225646972656 ,  0.3290178179740906 ,  0.35037320852279663,
  -0.10277722775936127,  0.27788761258125305,  0.21363574266433716,
  -0.6871033906936646 ,  0.46360185742378235,  0.8554520606994629 ,
   0.01334072463214397, -0.01451010257005692,  0.3495127558708191 ,
   0.3675326704978943 , -0.07880763709545135,  0.24954555928707123,
   0.7598238587379456 , -0.6435827612876892 , -0.3343323767185211 ,
   0.22098638117313385, -0.04369006305932999,  0.2719363570213318 ,
   0.01903204992413521,  0.33965563774108887, -0.4936477839946747 ,
  -0.35760194063186646, -0.3687765598297119 , -0.2650020718574524 ,
  -0.7272107005119324 , -1.5487457513809204 ,  0.08577088266611099,
   0.15163008868694305, -0.22151456773281097, -0.4180355370044708 ,
   0.2733384668827057 ,  1.1341571807861328 , -0.00246418314054608,
  -0.10781680047512054, -0.06782200187444687,  0.5109651684761047 ,
   0.10008150339126587, -0.10117210447788239, -0.3320024907588959 ,
  -0.24029850959777832,  0.594423770904541  , -0.8487746715545654 ,
  -0.5521202683448792 , -0.15584790706634521, -0.05562529712915421,
  -0.1181379109621048 ,  0.06205445528030396, -0.2730153501033783 ,
  -0.3861391842365265 , -0.7903586030006409 ,  0.28292396664619446,
   0.16516150534152985,  0.1599569320678711 ,  0.05149229615926743,
  -0.0141285415738821 ,  0.1960945725440979 , -0.3811786472797394 ,
  -0.43213847279548645, -0.20665477216243744,  0.04859109595417976,
  -0.4591967761516571 , -0.9482461214065552 ],
 [-0.20497523248195648, -0.39128732681274414,  0.3642028868198395 ,
   0.375515878200531  ,  0.4216025173664093 ,  0.17467375099658966,
  -0.4284088611602783 , -0.05306578427553177, -0.35451775789260864,
  -0.5430247187614441 ,  0.24324169754981995, -0.6698154211044312 ,
  -0.31755906343460083, -0.02308505214750767, -1.567978024482727  ,
  -0.0838589072227478 , -0.34013015031814575,  0.7478561997413635 ,
   1.047241449356079  ,  1.5078847408294678 , -0.49845272302627563,
   0.9153279066085815 ,  0.6516849994659424 ,  0.5195397734642029 ,
  -0.29165107011795044,  0.08415830880403519,  0.5069319009780884 ,
   0.395626425743103  , -0.4465973377227783 ,  0.5850189924240112 ,
  -0.01866435818374157,  0.8021047711372375 ,  0.30914831161499023,
   0.9914767146110535 , -0.4897706210613251 ,  0.1892264038324356 ,
   1.0538861751556396 , -0.25219646096229553,  0.7374734878540039 ,
  -1.0076746940612793 , -0.02827482298016548,  0.02920462191104889,
   0.2338658720254898 ,  1.0846294164657593 , -0.5598412156105042 ,
   0.44645795226097107,  0.33716925978660583, -0.1805468052625656 ,
  -1.258697509765625  ,  1.445656418800354  , -0.7262728214263916 ,
   0.14966300129890442, -0.20971769094467163,  1.4135280847549438 ,
  -0.01674435846507549,  0.3355713188648224 , -0.1597490906715393 ,
  -0.05673219263553619,  0.07326063513755798,  0.7250692844390869 ,
  -0.7301953434944153 ,  0.6239215135574341 ,  0.13409826159477234,
  -1.5015995502471924 , -0.6093191504478455 ,  0.2005937695503235 ,
   0.23528489470481873,  0.36755692958831787, -0.0311382319778204 ,
  -0.2764185965061188 ,  0.08710254728794098,  0.14085040986537933,
  -0.5879382491111755 , -0.0356256328523159 ,  0.16077426075935364,
  -0.2813568413257599 ,  0.7545809149742126 , -0.20994959771633148,
   0.4602750241756439 ,  0.08468808233737946, -0.40532851219177246,
   0.01568492688238621,  0.21338661015033722,  1.1601145267486572 ,
   0.01660997420549393, -0.3144403398036957 , -0.04837391525506973,
   0.31791141629219055,  0.5815441012382507 ,  0.8233100771903992 ,
  -0.5125936269760132 ,  0.01579436287283897,  2.5584447383880615 ,
   0.8300456404685974 ,  0.16199630498886108,  1.0896289348602295 ,
  -0.9135966897010803 ,  0.13525907695293427,  1.110678791999817  ,
   0.5669779181480408 , -1.8148789405822754 , -1.021077036857605  ,
   0.4409940838813782 ,  1.3928083181381226 ,  0.301053524017334  ,
  -1.0108299255371094 , -0.31856822967529297,  0.17656520009040833,
   0.02303970977663994,  1.305701494216919  ,  0.45191439986228943,
  -0.5509927272796631 , -0.15295468270778656,  0.6806362867355347 ,
  -0.1397518515586853 ,  0.09800808131694794,  0.8494945764541626 ,
   0.4652716815471649 , -0.9415714740753174 ,  0.9368640184402466 ,
  -1.3860511779785156 ,  0.3706423044204712 , -0.2640188932418823 ,
  -0.668189525604248  ,  1.9903124570846558 , -0.9318625926971436 ,
  -0.7096682190895081 ,  0.04492484778165817],
 [-0.09264841675758362, -0.4525517225265503 ,  0.04001513123512268,
   0.12434840202331543,  0.29172655940055847, -0.20325787365436554,
   0.0549904853105545 ,  0.25288307666778564,  0.3830893337726593 ,
  -0.04481819272041321,  0.31509310007095337, -0.02298776619136333,
  -0.1674547791481018 ,  0.00406187167391181, -0.2116805762052536 ,
  -0.06553250551223755,  0.8042618036270142 ,  0.5971776247024536 ,
  -0.08980444073677063, -0.2926086485385895 , -0.04608893021941185,
  -0.2107420712709427 , -0.18350128829479218, -0.6437176465988159 ,
   0.45536214113235474, -0.35928744077682495,  0.33504197001457214,
  -0.28523969650268555,  0.3841351866722107 , -0.5136620402336121 ,
   0.08806616812944412,  0.5765791535377502 , -0.2973393201828003 ,
   0.2965979278087616 , -0.31357112526893616,  0.1435205638408661 ,
   0.2880840301513672 ,  0.10588181018829346,  0.04430191963911057,
  -0.17716118693351746, -0.32092583179473877,  0.05603932216763496,
  -0.00686358287930489,  0.16020840406417847,  0.27165812253952026,
   0.18626633286476135, -0.4193758964538574 , -0.2703978717327118 ,
  -0.8439889550209045 , -0.5591723918914795 ,  0.1683320701122284 ,
  -0.24822628498077393, -0.3850063681602478 ,  0.08837851136922836,
  -0.5561766624450684 ,  0.02879448048770428,  0.0831630527973175 ,
  -0.01649495959281921,  0.01784718222916126, -0.2954184114933014 ,
   0.09570850431919098,  0.5759466886520386 , -0.12362851947546005,
  -0.4050511121749878 ,  0.20899534225463867,  0.31219401955604553,
   0.21711549162864685,  0.101127989590168  ,  0.43735674023628235,
  -0.31088563799858093,  0.21664899587631226,  0.21057987213134766,
  -0.42571890354156494,  0.09185697883367538,  0.5146234035491943 ,
   0.7452439665794373 ,  0.15870748460292816,  0.23340140283107758,
   0.9871218800544739 ,  0.41668224334716797, -0.16835112869739532,
   0.4678341746330261 , -0.11333561688661575,  0.2693799138069153 ,
   0.01514582801610231,  0.41620489954948425,  0.364632248878479  ,
   0.46626830101013184, -0.3442908823490143 ,  0.7135630249977112 ,
   0.32612431049346924, -0.2899841368198395 , -0.24920052289962769,
  -0.17716658115386963, -0.16315628588199615,  0.2065671980381012 ,
   0.30820432305336   ,  1.1415395736694336 ,  0.5134944915771484 ,
   0.04815636575222015, -0.18665540218353271, -0.46295100450515747,
   0.18474891781806946, -0.06948348879814148, -0.14140979945659637,
  -0.18533721566200256,  0.8367282748222351 , -0.19091536104679108,
   0.11334553360939026,  0.21567629277706146, -0.5874801278114319 ,
  -0.03291129693388939,  0.7421301007270813 , -0.18989090621471405,
  -0.3532242476940155 ,  0.24116267263889313,  0.6443943977355957 ,
   0.16679759323596954, -0.3282884359359741 ,  0.1065187156200409 ,
   0.17052097618579865,  0.04436685144901276,  0.2374781370162964 ,
  -0.20118993520736694, -0.03139996901154518,  0.35986146330833435,
   0.533446192741394  , -0.10987883806228638],
 [-0.20598582923412323, -0.18521399796009064, -0.06580333411693573,
  -0.8741225600242615 , -0.34101566672325134, -0.05503669753670692,
  -0.42502954602241516,  0.01746716164052486, -0.3603963553905487 ,
   0.23874302208423615, -0.23532021045684814, -0.3474557399749756 ,
  -0.4115224778652191 ,  0.0625351145863533 , -0.5567174553871155 ,
   0.04734427854418755, -0.3154126703739166 , -0.12210994958877563,
   0.00310414494015276, -0.03333801776170731, -0.285919189453125  ,
   0.3546261191368103 , -0.7272396087646484 , -0.06434888392686844,
  -0.783327043056488  , -0.765822172164917  , -0.8385741114616394 ,
  -0.46194586157798767,  0.3098965287208557 ,  0.2576189637184143 ,
  -0.2913037836551666 , -1.6764355897903442 , -1.1988842487335205 ,
   0.5500802993774414 ,  0.450737863779068  ,  0.09792841970920563,
   0.19290156662464142,  0.23119983077049255, -0.48325222730636597,
  -0.2635369598865509 , -0.8678766489028931 ,  0.13897283375263214,
  -0.42670178413391113, -0.2641165554523468 , -1.2187438011169434 ,
  -0.11584962904453278, -0.2232760190963745 , -0.04921146482229233,
  -0.08800908178091049, -0.565777063369751  ,  0.11654140800237656,
  -0.0512460470199585 ,  0.10778084397315979, -0.4368823170661926 ,
  -0.6546748876571655 ,  0.3648413419723511 ,  0.10093057155609131,
  -0.35197028517723083, -0.40481337904930115,  0.5268217921257019 ,
   0.53138667345047   ,  0.20619796216487885, -1.1868836879730225 ,
   0.2425035983324051 ,  0.01574594713747501, -0.5133723616600037 ,
  -0.8657129406929016 , -0.2801833748817444 , -0.5786175727844238 ,
  -0.8461440801620483 , -0.13196039199829102, -0.12388186156749725,
   0.21474558115005493, -0.3543086647987366 , -0.4746885299682617 ,
  -0.139878049492836  ,  0.494026780128479  , -1.4594529867172241 ,
  -2.8171191215515137 , -0.46917596459388733,  0.37189072370529175,
  -0.5694944262504578 , -0.5833903551101685 , -1.9193588495254517 ,
   0.12140584737062454,  0.7227472066879272 ,  0.2213980257511139 ,
   0.20704689621925354,  0.15473057329654694, -0.2317531406879425 ,
  -1.0245989561080933 , -0.49438294768333435,  0.3136293292045593 ,
  -1.4294461011886597 ,  0.543113112449646  , -0.6616983413696289 ,
   0.09241107851266861, -1.310631275177002  , -0.8844783902168274 ,
  -0.19532768428325653,  0.04358160868287086, -0.725534975528717  ,
   0.0791235938668251 , -0.24644257128238678,  0.2131452113389969 ,
   0.03732915967702866,  0.29440927505493164, -0.37950360774993896,
  -0.36731186509132385, -0.06771358102560043, -0.2570071518421173 ,
  -0.11795719712972641, -0.19362114369869232, -0.14919547736644745,
   0.27363118529319763, -0.2994369864463806 , -0.12835007905960083,
   0.3157784044742584 ,  0.08088374137878418, -0.857102632522583  ,
   0.08708459138870239,  1.5530840158462524 ,  0.00996332429349422,
  -0.1893632560968399 , -0.28275418281555176, -0.3287149667739868 ,
  -0.31763744354248047,  0.7210893034934998 ],
 [-0.16727976500988007, -0.12989462912082672, -0.02801908738911152,
  -0.7415867447853088 , -0.05317126959562302, -0.09300819039344788,
   0.03918785229325294, -0.09733060747385025,  0.05710197612643242,
  -0.03717779368162155,  0.27223825454711914,  0.22230055928230286,
  -0.4728288948535919 , -0.02070342376828194,  0.02740650810301304,
   0.09462682902812958, -1.0508614778518677 ,  0.5453701615333557 ,
   0.2661987543106079 , -0.5893522500991821 ,  0.6520466804504395 ,
   0.1602071225643158 ,  0.16573117673397064, -0.575084924697876  ,
   0.30743134021759033,  0.2890538275241852 ,  0.209318608045578  ,
   0.37976592779159546,  0.4971367418766022 , -0.07899485528469086,
  -0.13938163220882416, -0.1674458235502243 , -0.20730865001678467,
   0.8830364346504211 , -0.11355945467948914,  0.500823438167572  ,
   0.3662402331829071 , -0.18677639961242676, -0.1481434553861618 ,
  -0.2672399878501892 , -0.9039204716682434 , -0.6056985855102539 ,
  -0.18752074241638184,  0.5719179511070251 ,  0.05394275486469269,
  -0.18187692761421204, -0.5605833530426025 ,  0.4165913760662079 ,
  -0.5969255566596985 ,  0.3379746377468109 , -1.0630542039871216 ,
  -0.3422088623046875 , -0.15441085398197174, -0.18334783613681793,
  -1.4358861446380615 , -0.413522332906723  ,  0.5678761601448059 ,
  -0.4228355884552002 , -0.17614392936229706,  0.01460839062929153,
  -0.03769171983003616,  0.6328526735305786 , -0.07436077296733856,
   0.47829774022102356, -0.4370589554309845 , -0.38365018367767334,
  -0.3792179822921753 ,  0.16041822731494904, -0.0919414833188057 ,
   0.31669142842292786, -0.23860666155815125, -0.19457130134105682,
   0.13201268017292023,  0.9713320732116699 , -0.02090505324304104,
   0.02187148295342922,  0.07650600373744965,  0.6057918071746826 ,
   0.10347989946603775,  0.37728169560432434, -0.46970680356025696,
   0.4316830337047577 , -0.6804805994033813 , -0.1149035170674324 ,
   0.09404043108224869, -0.10315350443124771,  0.30260586738586426,
   0.3996429145336151 , -0.4249393045902252 , -0.627059280872345  ,
  -0.30491822957992554, -0.00321796466596425,  0.1139337345957756 ,
  -0.5196700096130371 ,  0.03677799925208092,  0.41836485266685486,
  -0.0673464760184288 , -0.12105995416641235,  0.14438804984092712,
   0.10816314071416855, -0.16024239361286163,  0.5744546055793762 ,
   0.00105419801548123, -0.01820730976760387, -0.6481595635414124 ,
   0.1279105544090271 ,  0.1268538534641266 , -0.8124631643295288 ,
  -0.4476056694984436 , -0.01012299861758947, -0.07027612626552582,
   0.7600074410438538 , -0.6180202960968018 ,  0.10925980657339096,
  -0.2327231466770172 ,  0.3273462951183319 , -0.46628227829933167,
   0.5970987677574158 ,  0.03990969061851501,  0.3722430467605591 ,
  -0.1795816570520401 ,  0.23970435559749603,  0.04039223492145538,
  -0.6396963596343994 ,  0.0975625142455101 ,  0.5305123925209045 ,
   0.8528962135314941 ,  0.5004643797874451 ],
 [ 0.15779249370098114,  0.14161325991153717,  0.7265075445175171 ,
  -0.08302051573991776, -1.2473863363265991 , -0.8679766654968262 ,
   0.5707863569259644 ,  0.629019558429718  , -1.7654075622558594 ,
  -0.38361427187919617, -2.9713120460510254 ,  0.3713204562664032 ,
   0.00602460559457541, -0.8175764679908752 ,  0.5890527963638306 ,
  -1.0838918685913086 ,  0.16871871054172516, -0.08025500178337097,
  -1.7485334873199463 , -1.788170576095581  , -1.045832633972168  ,
  -0.0761917382478714 , -2.7646071910858154 , -2.2988312244415283 ,
  -0.9240209460258484 ,  0.4674091041088104 , -0.9805068969726562 ,
   0.139410138130188  , -1.7081685066223145 , -0.13190530240535736,
   0.3489024341106415 ,  0.02849646843969822, -1.2480581998825073 ,
   0.8493759632110596 ,  0.2181745022535324 , -0.9350512623786926 ,
  -0.5940927267074585 ,  0.02784614264965057,  0.18739621341228485,
   0.3776669204235077 ,  1.0452277660369873 , -0.6190459132194519 ,
  -0.7508283257484436 , -3.6741690635681152 , -0.8283439874649048 ,
  -1.2724860906600952 ,  0.8576233983039856 ,  0.24506886303424835,
   0.47760605812072754, -0.9576383233070374 ,  0.20658203959465027,
   0.36201369762420654, -1.7228260040283203 , -0.14318902790546417,
  -1.9053117036819458 ,  1.1638383865356445 , -2.277874231338501  ,
   0.00184084195643663,  0.02923185564577579,  0.5214404463768005 ,
   0.4993050694465637 ,  0.6690428256988525 , -0.8905327916145325 ,
  -0.15176774561405182,  0.16416029632091522, -0.38660919666290283,
  -1.299938440322876  , -1.445619821548462  , -0.3983379602432251 ,
   0.03520722314715385, -1.875211238861084  , -0.15658344328403473,
   0.5287321209907532 , -3.872182607650757  , -0.1235811784863472 ,
  -0.9616559147834778 , -0.1452397257089615 , -0.7187800407409668 ,
  -0.800294041633606  , -0.2822616696357727 , -1.348341703414917  ,
  -1.26340913772583   ,  0.24323149025440216, -1.461191177368164  ,
  -0.9242838621139526 , -0.4557883143424988 , -0.24382999539375305,
   1.3590679168701172 , -0.24163603782653809,  0.8851946592330933 ,
   0.3306827247142792 ,  0.07571687549352646,  0.3638553023338318 ,
   1.8138493299484253 ,  0.40225765109062195, -1.9698134660720825 ,
   0.03854360431432724,  0.15399621427059174,  0.3225748836994171 ,
   0.4267861843109131 , -0.3290848135948181 , -0.9000402688980103 ,
   0.275856614112854  ,  0.05262770131230354,  1.376596212387085  ,
  -0.7007826566696167 , -0.15399549901485443, -0.4568306505680084 ,
  -0.5544934272766113 , -1.89337158203125   , -0.44414758682250977,
  -1.1706277132034302 , -1.4251054525375366 ,  0.15551136434078217,
   0.8202810287475586 ,  0.2872920334339142 , -0.19632944464683533,
  -0.9351993799209595 , -0.2453949898481369 ,  0.07181338220834732,
  -1.6530406475067139 , -3.681152582168579  , -0.5594913363456726 ,
   0.3775689899921417 , -0.23014326393604279, -1.1955024003982544 ,
   0.1628754884004593 , -0.8469094038009644 ],
 [-0.45734211802482605,  0.1255468726158142 ,  0.4039075970649719 ,
   0.07106960564851761,  0.02015611529350281, -0.6240038275718689 ,
   0.17088094353675842, -0.07806658744812012, -0.3965928256511688 ,
   0.02560808695852757, -0.9820085167884827 ,  0.19665394723415375,
  -0.12380733340978622, -1.087981939315796  , -0.20678403973579407,
  -0.00714909005910158, -0.02240386232733727,  0.5368247628211975 ,
  -1.3677072525024414 , -0.500527024269104  , -0.18274423480033875,
   0.23151099681854248, -0.6295189261436462 , -0.9256381988525391 ,
  -1.15651273727417   , -1.3382823467254639 ,  0.14260102808475494,
  -2.507542133331299  , -0.11090883612632751,  0.43341830372810364,
  -0.01114804204553366, -1.0028291940689087 , -1.2200769186019897 ,
  -2.8681955337524414 , -0.4649426341056824 ,  0.1858922690153122 ,
  -0.44787201285362244,  0.1544152945280075 , -0.11228553205728531,
  -0.06464147567749023, -0.38012585043907166,  0.19550397992134094,
   0.10833417624235153, -1.3148733377456665 ,  0.09705168008804321,
   0.3928159773349762 , -1.7595083713531494 ,  0.06081214547157288,
  -0.610618531703949  , -0.35895705223083496,  0.08224949240684509,
   0.00457374472171068,  0.12385743111371994,  0.15142150223255157,
  -1.5959911346435547 , -0.8811580538749695 ,  0.3199208080768585 ,
  -0.14097145199775696, -0.02906447649002075, -0.20944945514202118,
  -0.22996379435062408,  0.40411967039108276, -1.3959052562713623 ,
   0.04751714318990707, -0.6178628206253052 , -0.3377760648727417 ,
  -0.05099653452634811,  0.02442017011344433,  0.01985293254256248,
  -0.48036208748817444,  0.4452042281627655 ,  0.2178686261177063 ,
   0.19479914009571075,  0.573597252368927  ,  0.44672971963882446,
  -0.5537733435630798 ,  0.10689999163150787,  0.7468391060829163 ,
   0.00872939452528954, -0.29616230726242065, -0.39689522981643677,
  -0.11553195863962173,  0.29148706793785095,  0.4414837956428528 ,
  -0.21318942308425903,  0.16836462914943695, -0.8585680723190308 ,
   0.02899729833006859,  0.2415570169687271 ,  0.2067287117242813 ,
  -2.4458506107330322 , -1.9896858930587769 ,  0.46685755252838135,
   0.21246232092380524,  0.4067365825176239 ,  0.5510110259056091 ,
  -0.2026074081659317 , -0.801699161529541  ,  0.5841230154037476 ,
   0.03621942177414894,  0.11761664599180222, -0.09650500863790512,
  -0.43857377767562866, -0.9393795728683472 , -0.00811177119612694,
   0.6179199814796448 ,  0.15048307180404663,  0.03037121333181858,
   0.6076867580413818 , -0.6859138011932373 ,  0.10123125463724136,
   0.4407540261745453 ,  0.38618844747543335,  0.4823172986507416 ,
  -0.16311492025852203, -1.1321884393692017 ,  0.23970471322536469,
   0.18827120959758759, -0.33382171392440796, -0.5022604465484619 ,
   0.5091131329536438 ,  0.04350590705871582, -0.2006116360425949 ,
   0.0201895646750927 , -0.3378441631793976 ,  0.3719004988670349 ,
  -0.8989002108573914 , -1.2556873559951782 ],
 [ 1.8530992269515991 , -0.6942669749259949 ,  0.24811972677707672,
   0.7102429866790771 ,  0.22065076231956482, -0.2678062319755554 ,
  -0.6392841339111328 ,  0.06703196465969086,  0.8375734686851501 ,
  -0.22943107783794403,  0.5997724533081055 ,  0.2563731074333191 ,
   0.31385818123817444, -1.403909683227539  ,  0.6537117958068848 ,
  -0.12002981454133987,  0.3952616751194    , -0.851600170135498  ,
  -0.7617448568344116 , -0.09440094232559204, -0.4205727279186249 ,
  -0.18996044993400574,  0.20530156791210175,  0.29343923926353455,
  -0.7301109433174133 , -0.02622681856155396,  1.3295038938522339 ,
   0.22301948070526123, -1.4230867624282837 ,  0.3081791400909424 ,
   0.15810397267341614,  0.9142470955848694 ,  0.6057651042938232 ,
  -0.01711458340287209,  0.47881022095680237, -0.3244085907936096 ,
   0.5428320169448853 , -0.12176297605037689,  0.5433590412139893 ,
  -0.1696912795305252 , -0.7247874140739441 , -1.0145180225372314 ,
   0.08222279697656631, -0.1877758502960205 , -1.3008979558944702 ,
   0.9099323153495789 , -0.05352408811450005, -0.31468117237091064,
   0.5530943870544434 ,  0.02519545704126358,  0.3664949834346771 ,
   0.37478888034820557, -0.8811492919921875 ,  0.2153235375881195 ,
  -0.04870739951729774,  0.28721946477890015,  0.36385321617126465,
  -0.29258084297180176, -0.00975488498806953, -0.2156347930431366 ,
   0.7532326579093933 ,  0.4143288731575012 , -0.41685935854911804,
  -0.42903801798820496,  0.4827703535556793 ,  1.9103338718414307 ,
   0.3861488699913025 ,  0.04992695152759552,  0.3840472400188446 ,
   0.3175477385520935 ,  0.01393193937838078, -0.43753209710121155,
  -0.26996856927871704, -1.2429016828536987 ,  0.36883020401000977,
   0.47602805495262146, -1.292044997215271  ,  1.1861751079559326 ,
  -0.23852068185806274,  0.8986144065856934 , -0.933800995349884  ,
   0.38245633244514465,  1.507035732269287  ,  0.37233126163482666,
  -0.12936362624168396,  1.878350853919983  ,  1.189948320388794  ,
   0.33703774213790894, -0.5017494559288025 ,  1.238399624824524  ,
  -0.1808319091796875 ,  0.8017067313194275 ,  0.02401099540293217,
   0.00141426594927907,  0.7552313208580017 ,  0.06238490343093872,
  -0.5645556449890137 , -0.04455859586596489,  0.25154078006744385,
   0.4047245979309082 ,  0.47847220301628113,  0.545621395111084  ,
   0.08095249533653259, -1.4481322765350342 ,  1.0434452295303345 ,
   1.5182512998580933 ,  0.3753910958766937 , -0.4485342800617218 ,
   1.218644380569458  ,  0.3327873945236206 ,  0.10215408354997635,
   0.4297510087490082 ,  0.9280096888542175 ,  0.781930148601532  ,
   0.9893790483474731 , -1.219377040863037  ,  1.416251301765442  ,
  -0.5495575666427612 ,  0.01072836853563786,  0.611145555973053  ,
  -0.3439500331878662 ,  0.40661293268203735, -0.3848957419395447 ,
   0.4170975685119629 ,  0.805221676826477  ,  0.05433579906821251,
   0.9541972875595093 ,  0.2977471351623535 ],
 [-0.8201079368591309 ,  0.2164825052022934 , -0.2527894377708435 ,
   0.03278224915266037,  0.9806061387062073 , -0.7757272124290466 ,
   1.1203852891921997 ,  0.8104727864265442 , -0.13240428268909454,
   0.23011107742786407,  0.42666682600975037, -0.5567828416824341 ,
  -0.5667215585708618 ,  0.2939777970314026 , -0.2374824583530426 ,
   0.21656523644924164,  1.3575495481491089 ,  0.19756124913692474,
   1.1076730489730835 , -0.06247957423329353, -0.4837661385536194 ,
   0.1452329307794571 ,  0.43753623962402344,  0.24460312724113464,
  -0.42858633399009705, -0.40885165333747864,  0.20938082039356232,
  -1.2932316064834595 ,  0.7423401474952698 , -0.215301513671875  ,
  -0.7749298810958862 , -1.4085577726364136 , -0.11965302377939224,
  -0.913090169429779  ,  0.3159043490886688 ,  2.1438052654266357 ,
   1.6698576211929321 ,  0.3033250570297241 ,  2.092005729675293  ,
  -1.4584306478500366 ,  0.7719162702560425 , -0.7463536858558655 ,
   0.5733691453933716 , -0.4476573169231415 ,  2.2424302101135254 ,
   0.06435966491699219,  0.03531302884221077,  1.4939988851547241 ,
   0.4329203963279724 , -0.9825653433799744 , -0.8297485709190369 ,
   0.45197299122810364,  1.3811601400375366 ,  0.27394187450408936,
   2.11556339263916   ,  0.2640380859375    ,  0.08027128130197525,
   0.00467593362554908, -1.6315513849258423 , -0.00572313740849495,
  -0.331092894077301  , -0.06655789911746979,  0.06696963310241699,
   0.49490073323249817, -0.09490760415792465,  0.7010652422904968 ,
  -0.24878087639808655,  0.3774356544017792 , -0.08171916753053665,
   0.8241937756538391 ,  0.9378297328948975 ,  0.821742832660675  ,
  -0.6079494953155518 ,  0.5455530881881714 ,  1.1000518798828125 ,
   1.5490736961364746 ,  1.2609517574310303 ,  1.9528757333755493 ,
   1.44785737991333   ,  0.06128311529755592, -0.03083537146449089,
   0.6139017939567566 , -1.895724892616272  ,  1.656893014907837  ,
   0.0264815166592598 ,  0.33326709270477295,  0.6248733401298523 ,
   0.2379613071680069 , -0.3443486988544464 , -0.4322657287120819 ,
   2.7756898403167725 , -1.2165846824645996 ,  1.2069125175476074 ,
  -0.6520880460739136 ,  0.7595627903938293 , -0.13292144238948822,
  -0.344297856092453  ,  1.8996293544769287 ,  0.05851783230900764,
  -0.23124635219573975,  0.2861720323562622 , -0.389331191778183  ,
   0.243797168135643  ,  1.8522918224334717 , -0.650496780872345  ,
   0.4600532650947571 ,  0.3585749864578247 , -0.26674482226371765,
  -0.6530815958976746 ,  0.04353427886962891, -0.0880948156118393 ,
   0.7457071542739868 ,  1.8836511373519897 , -0.1223738044500351 ,
  -0.01731869578361511, -0.07550224661827087,  0.3843848407268524 ,
   0.7910784482955933 ,  0.8974658846855164 , -0.33020392060279846,
   0.4705851972103119 , -0.2287089079618454 , -0.0664968267083168 ,
   0.2736874520778656 ,  0.208634614944458  ,  0.181999072432518  ,
  -0.090009905397892  , -0.5539284348487854 ],
 [ 0.18441428244113922,  2.5565755367279053 , -0.09697207808494568,
  -0.21776871383190155,  0.14574521780014038, -1.409590721130371  ,
  -0.8216608166694641 , -0.06357619166374207,  1.7878271341323853 ,
   0.11154937744140625,  0.16789939999580383, -0.10902753472328186,
   0.05972952023148537,  0.5799785852432251 ,  0.20119380950927734,
  -0.12163472175598145, -0.7460232377052307 ,  0.15946750342845917,
  -1.2152272462844849 , -0.05996312573552132, -0.16727322340011597,
   0.2345089316368103 , -1.0362621545791626 ,  0.9591419696807861 ,
   0.29244861006736755, -3.218268394470215  , -0.5610983967781067 ,
   0.11173094809055328, -0.2939750850200653 ,  0.31976670026779175,
   0.01441793609410524, -0.07564087212085724, -0.5270358920097351 ,
  -0.47215738892555237,  0.0960736945271492 , -0.3229511082172394 ,
  -0.5682998299598694 ,  0.03611490875482559,  0.5000545382499695 ,
  -0.06065744161605835, -1.3081550598144531 ,  0.7983402013778687 ,
   0.8092501759529114 , -1.3415203094482422 , -0.575242280960083  ,
  -0.26258185505867004, -0.04464133828878403, -0.00424552382901311,
  -0.11990482360124588,  0.10586022585630417, -0.7594833374023438 ,
   0.10552754253149033, -0.19900988042354584, -0.15549783408641815,
  -0.19627277553081512, -0.24364179372787476, -0.8433681726455688 ,
  -0.1864626407623291 ,  0.00347414799034595,  0.7757238745689392 ,
   0.04621621966362   ,  0.4471569061279297 ,  0.0974188968539238 ,
   0.5831565856933594 ,  0.05951736494898796, -1.638413667678833  ,
   0.7286062240600586 ,  0.02551618963479996, -0.12751559913158417,
   0.8489775061607361 , -0.5466048121452332 ,  0.10531528294086456,
   0.20175863802433014, -0.1340002864599228 ,  0.6638292670249939 ,
   0.3142290413379669 , -1.5784704685211182 ,  0.436330109834671  ,
  -1.4065344333648682 ,  0.3858286142349243 ,  0.00259316875599325,
   0.0083245811983943 ,  0.08815667033195496,  1.066941738128662  ,
   0.00672413734719157,  0.3310036063194275 , -1.0140821933746338 ,
   0.6264250874519348 , -0.0627247542142868 , -0.07668380439281464,
   0.3706951141357422 ,  0.2088569700717926 ,  0.5476820468902588 ,
  -0.1134551465511322 ,  0.14287379384040833,  0.1600043773651123 ,
  -2.004662275314331  , -0.4412606954574585 , -0.4988630712032318 ,
  -0.18972623348236084, -0.06139081344008446,  1.0562070608139038 ,
   0.19375117123126984,  0.1541847139596939 ,  0.15200945734977722,
  -0.6650986671447754 , -0.02921048924326897,  0.6680876016616821 ,
  -0.08680856972932816, -0.03828536719083786,  0.33484145998954773,
  -0.6602224707603455 ,  0.11597442626953125, -3.0549209117889404 ,
  -0.115317203104496  ,  0.722928524017334  ,  0.07344545423984528,
  -0.10353085398674011, -0.15360824763774872, -0.45722803473472595,
  -1.138454794883728  ,  0.11846126616001129,  0.12015145272016525,
   0.29036465287208557, -0.7848674654960632 , -0.05447811633348465,
  -0.20249412953853607,  0.03869730979204178],
 [ 1.795468807220459  , -0.3374061584472656 , -0.3971869647502899 ,
  -0.07734303921461105,  0.5550817251205444 ,  0.7362291812896729 ,
  -0.34883901476860046,  1.7256567478179932 , -0.2976822853088379 ,
  -0.4519248902797699 ,  0.2903076708316803 ,  0.27287834882736206,
   0.7671717405319214 , -0.30558905005455017,  0.39147838950157166,
   0.585908055305481  , -0.290322482585907  , -1.6004339456558228 ,
   0.29006174206733704, -0.4391079843044281 ,  0.05311502516269684,
   0.6034982204437256 ,  0.4792284369468689 ,  0.24010887742042542,
   0.370545357465744  ,  0.8862478733062744 ,  0.2564281225204468 ,
   1.3406983613967896 , -0.01996556669473648,  0.3061053454875946 ,
   0.992914080619812  ,  0.3083844482898712 ,  0.1917864829301834 ,
   0.6657677292823792 ,  0.5568684935569763 , -0.39228788018226624,
   0.29858261346817017, -0.36247503757476807,  0.27387377619743347,
  -0.7222371101379395 , -0.3462105095386505 ,  0.4695155620574951 ,
  -0.0617947056889534 ,  1.0039316415786743 ,  0.20512454211711884,
  -0.15938906371593475,  0.30289849638938904,  0.12863054871559143,
  -0.615915060043335  ,  1.5109111070632935 ,  0.5512772798538208 ,
  -0.18731524050235748,  0.17674419283866882,  0.05800075083971024,
   0.03003097325563431, -0.1290663182735443 ,  0.10881610214710236,
   1.0568246841430664 ,  0.7991366982460022 ,  0.5552043318748474 ,
   0.36917877197265625,  0.8984692692756653 , -0.2806745767593384 ,
   0.8662895560264587 , -0.37874096632003784,  0.49272510409355164,
   0.07196378707885742,  0.5168360471725464 , -0.8604554533958435 ,
   0.3078806698322296 , -1.4960076808929443 ,  0.3075973689556122 ,
   0.588676393032074  , -0.569542646408081  ,  1.578696608543396  ,
   0.49739351868629456,  0.093500055372715  , -0.7312763929367065 ,
  -0.02696481719613075, -0.8796579837799072 ,  0.9750409722328186 ,
   0.27299895882606506, -0.2548537850379944 , -0.18463203310966492,
   0.06539946794509888,  0.8368306756019592 , -0.25330132246017456,
   0.02737827226519585, -1.425402283668518  ,  2.334829092025757  ,
  -0.480264812707901  , -0.22536183893680573, -0.3511851131916046 ,
   0.5377346277236938 , -0.06680949777364731,  0.3706454336643219 ,
  -0.01855112239718437, -1.1596336364746094 ,  0.07003343105316162,
   2.0976240634918213 , -0.9344924092292786 , -0.04167607054114342,
   0.05279658362269402, -0.14374755322933197,  0.5064091086387634 ,
   0.074190154671669  ,  0.9292303919792175 ,  0.315263956785202  ,
  -0.1519298106431961 ,  0.4094899892807007 ,  0.25729551911354065,
   0.5055856704711914 , -0.39661672711372375,  0.24967697262763977,
  -0.1787233054637909 , -0.08155489712953568,  1.4857195615768433 ,
   0.17355327308177948, -1.4399346113204956 ,  1.2524194717407227 ,
   0.18396177887916565,  0.18290257453918457, -0.48297345638275146,
   1.30081307888031   ,  0.35801973938941956, -0.3514537513256073 ,
   0.8774716258049011 , -0.3413955569267273 ],
 [ 0.20847927033901215,  1.864732027053833  ,  0.4954668879508972 ,
  -2.4102487564086914 ,  0.8754606246948242 ,  1.6483463048934937 ,
   0.00854945089668036,  0.11700951308012009,  0.29541856050491333,
   0.4096077084541321 ,  1.0962684154510498 ,  0.01384961232542992,
  -0.1747657209634781 , -0.14324118196964264,  0.54049152135849   ,
   0.01838186383247375, -2.046238899230957  ,  0.2455758899450302 ,
   3.4676408767700195 , -1.4250818490982056 ,  1.4956542253494263 ,
   1.112648844718933  ,  0.9364150166511536 ,  1.2460601329803467 ,
  -0.852040708065033  ,  2.3214848041534424 , -0.08253329247236252,
   0.7209464907646179 , -0.07306132465600967,  1.474461555480957  ,
  -0.10187143087387085,  0.20354391634464264,  0.986153244972229  ,
   0.9018146395683289 , -0.17691387236118317, -0.7569490671157837 ,
   1.0991145372390747 ,  0.42698293924331665,  1.9297964572906494 ,
  -0.5465766191482544 ,  1.8627465963363647 ,  1.7140140533447266 ,
   0.8600378632545471 ,  1.1034992933273315 , -1.8838318586349487 ,
  -0.6810932755470276 ,  1.6450374126434326 ,  0.6077561378479004 ,
   0.7585462331771851 , -0.02894958667457104,  1.2191916704177856 ,
   0.12519171833992004,  1.6449791193008423 ,  0.16949504613876343,
  -0.5978642106056213 ,  0.15329264104366302, -1.5313383340835571 ,
  -0.3292884826660156 , -0.5001029372215271 ,  0.24247364699840546,
  -0.6163052320480347 , -0.24137423932552338,  2.796271324157715  ,
   0.7969993352890015 ,  0.6299611330032349 ,  0.8953969478607178 ,
   1.0617873668670654 ,  0.9481360912322998 , -1.2307283878326416 ,
   0.03029180131852627,  0.8759506344795227 , -0.13697631657123566,
  -0.35220587253570557, -0.465900719165802  ,  0.5232706069946289 ,
   1.4388703107833862 ,  1.197665810585022  ,  0.2658226788043976 ,
  -1.5517373085021973 , -0.00761605985462666, -0.3036869466304779 ,
   1.2490628957748413 , -0.4366062581539154 , -1.7615107297897339 ,
  -0.07206744700670242, -0.28282439708709717, -1.0830351114273071 ,
   0.4846841096878052 , -0.4202525019645691 ,  0.5079472064971924 ,
  -1.5468783378601074 ,  0.2488066703081131 ,  1.8343640565872192 ,
   0.08490975201129913, -0.01697657443583012,  1.3682464361190796 ,
   1.1965254545211792 , -0.8929973244667053 ,  0.9852693676948547 ,
   0.4081933796405792 , -0.2734883725643158 , -0.05989600718021393,
   0.21282337605953217,  2.121640205383301  ,  1.1201564073562622 ,
  -0.508980393409729  ,  0.40769973397254944,  1.3337736129760742 ,
  -0.34228330850601196, -0.8118728399276733 ,  0.3398195207118988 ,
   0.6080716252326965 ,  2.0792417526245117 , -0.5306890606880188 ,
  -0.2870439887046814 ,  0.12752468883991241,  0.64863520860672   ,
  -0.5990583300590515 , -0.484412282705307  ,  0.1261722892522812 ,
  -1.3590924739837646 , -0.08932129293680191,  0.16912558674812317,
  -1.0090337991714478 ,  0.71988445520401   ,  0.6584140658378601 ,
   0.6597292423248291 ,  1.6597124338150024 ],
 [-0.05932768806815147,  0.394281268119812  ,  0.04971157759428024,
  -0.5417875051498413 ,  0.14061595499515533, -0.3817724287509918 ,
   0.42843177914619446, -0.9552265405654907 ,  0.3754658102989197 ,
  -0.01035870052874088, -0.22802972793579102, -0.1427757441997528 ,
   0.41520780324935913, -0.3868870139122009 ,  0.01487809605896473,
  -0.11679423600435257, -2.895047903060913  , -0.02281805127859116,
  -0.04667872190475464, -0.11668568849563599,  0.3877216875553131 ,
  -0.5493123531341553 ,  0.05458669736981392, -1.1039860248565674 ,
  -1.0587342977523804 ,  0.15082846581935883,  0.45583945512771606,
  -0.24832598865032196,  0.17574654519557953, -0.11658662557601929,
   0.607812225818634  , -0.00544326798990369, -0.5115320086479187 ,
  -0.775150716304779  , -0.4822155237197876 ,  0.07280585169792175,
  -1.0389628410339355 , -0.40737828612327576,  0.3074069023132324 ,
  -0.02636720240116119,  0.42721670866012573,  0.7748256921768188 ,
   0.20880809426307678, -1.3899803161621094 , -0.37488338351249695,
   0.37936317920684814,  0.40932026505470276,  0.35629937052726746,
   0.0627441480755806 , -0.3301488161087036 ,  0.23715123534202576,
  -0.2781554162502289 , -0.9247777462005615 ,  0.168150395154953  ,
   0.04863802716135979,  0.10311419516801834, -0.44812265038490295,
   0.03795212507247925, -1.6414215564727783 ,  0.14193083345890045,
   0.4187852442264557 ,  0.10080883651971817, -0.15036523342132568,
   0.09265538305044174, -1.0841392278671265 ,  0.6409745216369629 ,
   0.4643087685108185 , -0.6380978226661682 , -1.2974274158477783 ,
  -0.2561008632183075 ,  0.13739657402038574, -0.40016359090805054,
  -0.24832522869110107,  0.5231084227561951 ,  0.1575295776128769 ,
  -1.1936184167861938 , -0.46208691596984863, -0.9933137893676758 ,
   0.33741632103919983, -0.5685134530067444 ,  0.5330911874771118 ,
  -1.7592461109161377 , -0.1336948424577713 , -1.208822250366211  ,
  -0.01456916239112616,  0.5351732969284058 , -0.4427606761455536 ,
  -0.04050406813621521, -0.7659424543380737 ,  0.12172232568264008,
  -0.7887634634971619 , -0.31926828622817993, -1.7157224416732788 ,
   0.36301255226135254,  0.07986050099134445,  0.00765157584100962,
  -0.3127175271511078 , -2.0477211475372314 , -0.3317517638206482 ,
   0.03292785212397575, -0.5796706080436707 , -2.184870481491089  ,
  -0.15970773994922638, -2.0653672218322754 ,  0.6723051071166992 ,
  -0.6547609567642212 ,  0.31556516885757446,  0.2841472029685974 ,
  -0.7682797312736511 , -0.2403292953968048 , -0.42645353078842163,
  -1.1523491144180298 , -0.7767897844314575 ,  0.42356443405151367,
  -0.5477485656738281 , -0.6664917469024658 ,  0.3783162832260132 ,
  -0.2964707911014557 , -0.2922549545764923 , -0.5715512037277222 ,
  -1.1582720279693604 , -0.05839457362890244,  0.22037091851234436,
  -0.8047646284103394 , -2.0422892570495605 , -0.6305648684501648 ,
   0.5513295531272888 ,  0.03554993495345116],
 [ 0.01602694950997829,  0.23304346203804016, -0.00400990573689342,
  -0.143661230802536  , -0.19105759263038635,  0.12405746430158615,
  -0.37967661023139954,  0.44082191586494446,  0.04409842938184738,
   0.01519906520843506, -0.43471577763557434, -0.05612009763717651,
  -0.42974159121513367,  0.6109991073608398 ,  0.21178367733955383,
  -0.14639315009117126,  0.4248410761356354 ,  0.49001646041870117,
   0.5449711084365845 ,  0.11100205779075623, -0.2934308648109436 ,
  -0.09014245867729187,  0.5783036351203918 , -0.3532399833202362 ,
   0.40674784779548645,  0.4198199510574341 , -0.00823325105011463,
   0.3492206931114197 ,  0.06160527840256691,  0.2992386221885681 ,
  -0.15641005337238312,  0.2255188375711441 ,  0.2552453279495239 ,
   0.7739558219909668 , -0.42278796434402466,  2.7123425006866455 ,
   0.9593901038169861 , -0.06024858728051186,  1.0925716161727905 ,
  -0.09606774896383286, -0.24012358486652374, -0.11711457371711731,
   0.2937979996204376 ,  2.3017187118530273 , -0.29178547859191895,
   0.20691396296024323, -0.39935484528541565,  0.2792171835899353 ,
   0.0426238551735878 ,  0.23343628644943237, -0.595893919467926  ,
  -0.0385114885866642 ,  0.5642107725143433 ,  0.41031503677368164,
  -0.32005834579467773, -0.35328418016433716, -0.0605849102139473 ,
  -0.20284393429756165,  0.5023167133331299 ,  0.05146393552422523,
  -0.29993903636932373,  0.716537356376648  ,  0.10388632863759995,
  -0.41532012820243835,  0.22792665660381317, -0.3206259310245514 ,
  -0.01873108930885792,  0.11196393519639969, -0.6983277201652527 ,
   0.08135072141885757, -0.1674896776676178 ,  0.32250556349754333,
  -0.35932105779647827, -0.341697096824646  ,  0.43377768993377686,
   1.1942862272262573 , -0.27888908982276917,  0.29458314180374146,
   0.3758542835712433 , -0.14294523000717163, -0.07779671251773834,
  -0.2527528703212738 ,  0.4567456543445587 ,  0.10991200059652328,
   0.22371256351470947,  0.04740559309720993,  0.5503158569335938 ,
  -0.06146160140633583, -0.23662814497947693,  0.09491551667451859,
   0.02048017270863056,  0.2996998131275177 ,  0.8451748490333557 ,
  -0.3356136083602905 ,  0.4355131685733795 ,  0.1916775107383728 ,
  -1.5929678678512573 ,  0.7923102378845215 ,  0.4134705662727356 ,
  -0.03491990268230438, -0.5939972400665283 ,  0.23360028862953186,
  -0.0690099224448204 ,  0.33990103006362915,  0.25687071681022644,
   0.19757506251335144, -0.3893510401248932 , -0.00783419329673052,
   0.2700263261795044 , -0.5029587149620056 ,  0.36052030324935913,
   0.6976506114006042 ,  0.5487332344055176 ,  0.1744024157524109 ,
  -0.42000076174736023,  1.0309277772903442 ,  0.23357442021369934,
   0.35269707441329956,  0.00084544409764931,  0.7350391149520874 ,
   0.05384315922856331,  0.5531758069992065 , -0.09619317203760147,
   0.6567854881286621 ,  1.1570302248001099 , -0.794992983341217  ,
  -0.5340214967727661 , -0.5112676620483398 ],
 [-0.31961286067962646,  0.5346730351448059 , -0.07864904403686523,
   0.1546698808670044 ,  0.2363397479057312 ,  0.1564008593559265 ,
  -0.542579710483551  ,  0.3058871328830719 ,  0.34203362464904785,
   0.11817216873168945, -1.0006333589553833 , -0.0825924277305603 ,
  -0.00458377320319414, -0.09209237992763519, -0.09272373467683792,
   0.16250841319561005,  0.16449056565761566, -0.05745762959122658,
  -0.3030988872051239 ,  0.2636256515979767 , -0.4242718517780304 ,
   0.2104703187942505 , -1.8412914276123047 ,  0.3313302993774414 ,
  -2.5051040649414062 ,  0.20749379694461823, -0.4073563516139984 ,
  -0.47595542669296265,  0.44998276233673096, -1.103759765625     ,
  -0.08407263457775116,  0.6848120093345642 , -0.18127965927124023,
   0.5173108577728271 , -0.29230645298957825,  0.7771374583244324 ,
   0.20839928090572357,  0.07259400933980942, -1.4103206396102905 ,
  -0.10119445621967316, -0.13908834755420685, -1.0290333032608032 ,
   0.2238585203886032 ,  0.3281514346599579 , -0.8275479078292847 ,
  -2.011383533477783  , -0.3013473451137543 ,  0.43363064527511597,
   0.24546276032924652,  0.3465620279312134 ,  0.22672776877880096,
   0.00602815160527825, -0.6208532452583313 , -0.25252479314804077,
   1.1757150888442993 , -0.14609475433826447, -1.1429343223571777 ,
  -0.4278228282928467 , -0.19258758425712585, -0.7399669885635376 ,
  -1.3075799942016602 , -0.3698160946369171 , -1.167398452758789  ,
   0.6665772199630737 ,  0.33696284890174866,  0.0261022113263607 ,
  -0.1066998541355133 ,  0.06824944168329239, -0.6229556798934937 ,
   0.6332967281341553 ,  0.17791172862052917,  0.17029555141925812,
  -0.5355410575866699 , -0.22179722785949707,  0.01663171872496605,
   0.3081730306148529 , -0.8851766586303711 ,  0.17143018543720245,
  -1.3150944709777832 , -0.2621263861656189 ,  0.429374098777771  ,
   0.5610341429710388 , -0.38003110885620117, -0.28081566095352173,
   0.07487896829843521, -0.5066862106323242 ,  0.421577125787735  ,
   0.13865816593170166,  0.3387879729270935 , -0.13808941841125488,
   1.0340360403060913 ,  0.3140658140182495 ,  0.546846330165863  ,
  -0.8687459230422974 , -1.4587849378585815 ,  0.05590557679533958,
   0.7449661493301392 , -0.467463880777359  ,  0.158305823802948  ,
  -0.07094570994377136,  0.6862050294876099 , -0.5866836309432983 ,
   0.02703671716153622, -0.18078555166721344,  0.2710546851158142 ,
   0.24942372739315033,  0.2008189857006073 ,  0.4556390345096588 ,
   0.29122889041900635,  0.504814088344574  ,  0.6075674295425415 ,
  -0.16147160530090332, -0.7302664518356323 ,  0.4992963671684265 ,
  -0.20442862808704376,  0.06546460837125778, -0.6753216981887817 ,
   0.08982812613248825, -0.1629592776298523 , -0.32424598932266235,
  -0.20920448005199432, -0.34732508659362793,  0.869584858417511  ,
  -0.07951230555772781, -0.09245588630437851,  0.06368918716907501,
  -0.08972242474555969, -0.26552125811576843],
 [-0.10267337411642075,  0.43935418128967285, -0.02668458968400955,
   0.07537325471639633,  0.258881539106369  , -0.8039207458496094 ,
  -2.07597017288208   ,  0.5155101418495178 ,  1.758171796798706  ,
  -0.20191706717014313, -0.08495408296585083, -0.2392050325870514 ,
   1.0315669775009155 ,  0.05415964499115944, -0.27742621302604675,
  -0.01277070306241512, -1.2876986265182495 ,  0.40712258219718933,
   0.6879563927650452 ,  0.5739597082138062 ,  0.33122676610946655,
   0.3051430881023407 , -0.14578014612197876,  3.415870428085327  ,
   0.8656307458877563 ,  0.4989818334579468 ,  0.6756119728088379 ,
   0.8900662660598755 , -0.24595768749713898,  0.34696152806282043,
   0.26105543971061707, -1.3719900846481323 , -0.5133802890777588 ,
  -0.7769278287887573 ,  0.02417942509055138, -0.986292839050293  ,
  -0.8468955755233765 ,  0.10239702463150024,  0.01130038686096668,
   0.33911049365997314, -0.9918771386146545 , -0.2223534733057022 ,
   0.712699294090271  ,  0.2971547245979309 ,  0.24261870980262756,
  -0.09709221124649048, -0.7421662211418152 ,  0.40165939927101135,
  -0.3947893977165222 , -0.16553737223148346, -0.3446263074874878 ,
   0.00100043555721641,  0.03751751035451889, -0.0569755844771862 ,
   1.5305721759796143 ,  0.4028334617614746 , -0.22536641359329224,
   0.36940500140190125,  0.17089836299419403, -0.20565007627010345,
   0.10918454825878143,  0.849524736404419  ,  0.44475096464157104,
   0.8805579543113708 , -0.33106309175491333, -0.9066888689994812 ,
   0.4904288351535797 ,  0.02422657981514931,  0.9920505285263062 ,
   0.9493982195854187 ,  0.19974461197853088,  1.0696569681167603 ,
  -0.10924331843852997,  0.8737512826919556 , -0.9574892520904541 ,
   0.18269230425357819,  0.28208205103874207, -0.3855036497116089 ,
   1.8762472867965698 , -1.2369662523269653 , -0.6120196580886841 ,
   0.5464600324630737 , -0.08108483999967575, -0.06125059723854065,
   0.0860702320933342 , -0.1395636349916458 ,  1.7445390224456787 ,
  -0.4638049006462097 , -0.00446792272850871,  0.1802029311656952 ,
   0.41584476828575134, -0.6840071082115173 , -0.7101301550865173 ,
   0.38507455587387085, -0.39073866605758667,  0.2950282394886017 ,
  -1.928551435470581  ,  0.36817285418510437,  0.5531924366950989 ,
  -0.03731417283415794,  0.307368665933609  ,  0.04953012615442276,
  -0.12118881940841675,  0.24252457916736603,  0.466133177280426  ,
   0.01400444190949202, -1.1548453569412231 ,  0.7446285486221313 ,
   0.4876975417137146 ,  0.13692699372768402,  0.4731850028038025 ,
  -0.01577556878328323,  0.08352328836917877,  0.07435444742441177,
   0.5161799788475037 ,  0.4205382168292999 ,  0.17615912854671478,
  -0.15097932517528534,  0.45055243372917175,  0.15048420429229736,
  -0.8515943884849548 ,  0.23669493198394775, -0.2273881733417511 ,
  -0.09010142832994461,  0.09803193807601929,  0.3920700252056122 ,
  -1.7700639963150024 , -0.7382572293281555 ],
 [ 0.2436562031507492 ,  0.01799861341714859, -0.08948986977338791,
   0.3266984224319458 ,  0.6400370597839355 ,  0.28911417722702026,
   0.17377816140651703, -0.8457777500152588 ,  0.04868524149060249,
  -0.23753094673156738, -0.03866901248693466,  0.6006677150726318 ,
  -0.1837313175201416 , -0.39398765563964844, -0.01191086694598198,
   0.43743762373924255, -0.44856157898902893, -0.8848069906234741 ,
   0.7146152257919312 ,  1.383824348449707  , -0.24228328466415405,
   0.21935610473155975,  0.4049593508243561 ,  1.5816519260406494 ,
   0.4190608263015747 , -0.2113388329744339 ,  0.12500005960464478,
   0.00431274762377143, -0.333861380815506  ,  0.26292601227760315,
  -0.29748913645744324, -0.9106389880180359 ,  0.40955933928489685,
  -0.2193438857793808 , -0.09398060292005539, -0.16265161335468292,
   0.1182033121585846 , -0.24425183236598969, -0.05616569519042969,
  -1.2520421743392944 , -0.00473439926281571,  0.4364303946495056 ,
   0.7011922001838684 ,  1.0182790756225586 , -1.3124237060546875 ,
  -0.18225297331809998,  0.08805233240127563,  1.3559123277664185 ,
  -0.47922462224960327, -0.8112795948982239 ,  0.3685842454433441 ,
   0.05950237065553665,  0.31413671374320984,  0.20932619273662567,
   1.192183017730713  , -1.0781382322311401 ,  0.4281294345855713 ,
   0.1700771600008011 , -2.500344753265381  , -1.7618364095687866 ,
  -0.08007815480232239, -0.0943567156791687 ,  0.05783971771597862,
   0.7696854472160339 , -0.719748318195343  ,  0.2553422451019287 ,
   0.4830377697944641 ,  0.402478963136673  , -0.00405238242819905,
   0.01369887217879295,  0.23999850451946259,  0.40092357993125916,
   1.013585090637207  , -0.4375511407852173 ,  0.42897385358810425,
   0.5859792828559875 , -0.5707210302352905 , -0.5526646375656128 ,
   0.5164718627929688 ,  0.5262420773506165 , -1.1336570978164673 ,
   1.511461615562439  , -0.5862581729888916 ,  0.15175597369670868,
  -0.7227463126182556 ,  0.0886201336979866 , -0.26637399196624756,
  -0.7077361941337585 , -0.2197175920009613 ,  0.3613058030605316 ,
   0.9430087804794312 ,  0.08894401043653488, -0.2648310959339142 ,
   0.03142267465591431,  0.10552623122930527, -0.20734962821006775,
   0.3960375189781189 ,  0.5168209671974182 , -1.0059778690338135 ,
   0.6080357432365417 , -1.2917526960372925 , -0.3687606155872345 ,
  -0.1428949534893036 ,  0.6303051710128784 , -0.6667938828468323 ,
   0.09093958139419556,  0.10630153119564056,  0.3958347737789154 ,
  -0.13896748423576355,  0.6407272815704346 , -0.321546345949173  ,
   1.532606601715088  , -0.7427919507026672 ,  0.1488967388868332 ,
  -0.8085795044898987 ,  0.6996334791183472 ,  0.22832772135734558,
   0.5113512277603149 , -1.7980705499649048 , -0.6242492198944092 ,
  -0.2602510452270508 , -0.26764577627182007,  0.01452990807592869,
   0.05651114881038666,  0.9810559749603271 ,  0.3381114602088928 ,
   0.4997364580631256 , -0.20321905612945557],
 [-0.07541042566299438,  0.08413942903280258,  0.2046162188053131 ,
  -0.1937410533428192 , -0.3241409957408905 , -0.8800723552703857 ,
  -0.09727892279624939,  0.08092726767063141,  0.5832443833351135 ,
  -1.2428632974624634 , -0.28900057077407837,  0.23728667199611664,
   0.7732421159744263 ,  0.3003804683685303 , -0.6618670225143433 ,
   0.2777405381202698 , -0.21360419690608978, -0.5439021587371826 ,
  -0.29120978713035583, -0.11798129230737686,  0.3495084047317505 ,
   0.9335452318191528 , -0.6700930595397949 ,  0.9621579647064209 ,
   0.02136381343007088, -0.6260732412338257 , -0.2635146379470825 ,
  -1.0102213621139526 ,  0.08063001930713654, -0.25247466564178467,
   1.5344138145446777 ,  0.11018659174442291, -0.06925401836633682,
   0.6786588430404663 ,  0.30415529012680054, -1.14439058303833   ,
  -0.7313133478164673 ,  0.22598208487033844,  0.712742805480957  ,
   0.24324247241020203, -0.361398309469223  ,  0.19560322165489197,
   0.4913610517978668 ,  0.5196096897125244 ,  0.22047238051891327,
  -0.0835580974817276 ,  1.4062230587005615 , -0.4438784122467041 ,
  -0.3158845007419586 ,  1.0298678874969482 , -0.6697143912315369 ,
  -0.1850687563419342 ,  0.08268997073173523,  1.2782329320907593 ,
   0.15209688246250153,  0.4821953773498535 , -0.9774042367935181 ,
   1.6872854232788086 , -0.0335124135017395 ,  0.6424359679222107 ,
  -0.3067196011543274 , -0.5999546051025391 ,  0.8739532828330994 ,
   0.4609473645687103 ,  0.567062497138977  , -0.76641845703125   ,
  -1.0433388948440552 ,  0.17931486666202545,  0.6667744517326355 ,
  -0.9625847935676575 , -0.3367581069469452 , -0.2613331973552704 ,
   0.0995355173945427 , -0.35292279720306396,  1.303577184677124  ,
  -0.29425880312919617,  0.01473242975771427,  0.23000706732273102,
   0.9906230568885803 , -0.9546350836753845 ,  0.17780686914920807,
  -0.2333132028579712 ,  0.37716856598854065, -0.21368610858917236,
   1.4600814580917358 , -0.0090613691136241 , -0.5950886011123657 ,
   0.14751003682613373,  0.11895769834518433,  1.3315544128417969 ,
   0.9807530045509338 , -0.2821067273616791 ,  0.5183154940605164 ,
  -0.45672744512557983, -0.2902091145515442 , -0.3213745057582855 ,
   0.5149875283241272 , -0.01542762015014887, -0.18008261919021606,
   1.4500383138656616 , -0.45802661776542664,  1.338726282119751  ,
  -0.29919376969337463, -1.5590808391571045 , -0.7682822346687317 ,
  -0.03520272672176361,  0.01641906052827835,  0.19185331463813782,
   0.29456084966659546,  0.5689101815223694 , -0.22707298398017883,
   0.3349730372428894 , -0.18805724382400513, -0.05936683714389801,
  -0.16064682602882385,  0.22628645598888397, -0.5126057267189026 ,
  -0.4810388684272766 , -1.167589545249939  ,  1.1535271406173706 ,
  -0.5298296809196472 , -1.1753780841827393 , -0.8827556371688843 ,
  -0.3469853401184082 ,  1.414768099784851  , -0.5362433195114136 ,
  -0.5574212074279785 , -0.456302285194397  ],
 [ 0.7103126645088196 ,  0.02547482401132584, -0.11382771283388138,
   0.12063655257225037,  0.812734842300415  , -0.23487944900989532,
   0.2415217161178589 ,  0.15420883893966675, -0.1138000562787056 ,
  -0.02680279314517975,  0.2375694364309311 , -0.503460168838501  ,
  -0.04006535559892654,  0.7400853633880615 ,  0.45485541224479675,
   0.03778479248285294,  0.00600947067141533,  0.00661032740026712,
  -0.2516895532608032 ,  0.265531063079834  ,  0.4023267328739166 ,
  -0.5784156322479248 ,  0.19137011468410492,  0.8699404001235962 ,
   0.9212864637374878 ,  0.4043150544166565 ,  0.30152490735054016,
   0.5399952530860901 , -1.110011100769043  ,  0.14464159309864044,
  -0.01613757200539112,  0.49946099519729614,  0.47170016169548035,
  -0.339933842420578  ,  0.20598284900188446, -1.3562415838241577 ,
   0.7477430701255798 , -0.11851797997951508, -0.4353135824203491 ,
  -0.10959015786647797,  0.04684699326753616,  0.23502573370933533,
  -0.5962318778038025 , -0.3073160946369171 , -1.1644078493118286 ,
   0.28751203417778015,  1.0040814876556396 , -0.1475754827260971 ,
  -0.22550585865974426,  0.17942258715629578,  0.23704245686531067,
   0.28729748725891113,  0.2738294303417206 ,  0.1339089274406433 ,
   0.26477232575416565,  0.3414338529109955 , -0.12239789217710495,
  -0.1285538226366043 ,  0.4673127233982086 ,  0.05073367059230804,
   0.5379299521446228 ,  0.08474654704332352, -0.3571927547454834 ,
  -0.21399478614330292,  0.005507733207196  , -0.00348277762532234,
  -0.17249636352062225,  0.837695837020874  ,  0.7731448411941528 ,
   0.14176324009895325, -0.1658901870250702 , -0.02910977602005005,
   0.23510554432868958, -0.04646207764744759,  0.00528016453608871,
  -0.2578643560409546 , -0.13855504989624023,  0.16361267864704132,
   0.45248889923095703, -0.5757392644882202 , -0.03149034455418587,
  -0.4791433811187744 ,  0.3202722668647766 , -0.2294343262910843 ,
   0.05518588423728943, -0.15290291607379913, -0.6061626076698303 ,
  -0.16686776280403137, -0.01792015880346298, -0.0975421890616417 ,
  -0.2451949566602707 ,  0.30103015899658203, -0.6452360153198242 ,
  -0.05427680164575577, -0.00424121832475066,  0.2657517194747925 ,
  -1.8621883392333984 , -0.2514248788356781 ,  0.385647177696228  ,
   0.1741231083869934 ,  0.5646649599075317 , -0.3313313126564026 ,
  -0.02005641907453537,  0.4792484939098358 , -0.26246002316474915,
   0.5180778503417969 , -0.3580833077430725 , -1.0064668655395508 ,
   0.06978373974561691,  0.23393958806991577,  0.568030834197998  ,
   0.5059837102890015 ,  0.42706963419914246, -0.2482239007949829 ,
   0.04010619223117828, -0.27211806178092957, -0.20360875129699707,
   0.49915581941604614, -0.3486862778663635 ,  0.3834631145000458 ,
   0.12289544939994812,  0.28269875049591064,  0.11558086425065994,
   0.5024884343147278 , -0.1267404705286026 , -0.01470568124204874,
  -0.42619189620018005, -0.04076812043786049],
 [ 0.12257612496614456, -0.03180763870477676, -0.03456658497452736,
   0.0603218674659729 ,  0.0965299978852272 , -0.07059873640537262,
  -0.04008772969245911,  0.17336617410182953, -0.3436164855957031 ,
   0.12860479950904846, -0.09748382121324539,  0.05836281180381775,
  -0.46910861134529114, -0.02794064953923225, -0.20764559507369995,
  -0.04953018203377724, -0.21995289623737335,  0.08302728831768036,
   0.00292260502465069,  0.1274058222770691 , -0.0037936472799629 ,
  -0.1300976425409317 ,  0.01936165429651737, -0.16550949215888977,
  -0.10969632118940353, -0.06882989406585693, -0.20835070312023163,
  -0.19057118892669678,  0.20764051377773285, -0.01909737847745419,
   0.18898321688175201, -0.07904934883117676,  0.13111716508865356,
  -0.4022485017776489 , -0.19411468505859375, -0.08808261156082153,
   0.09444887191057205,  0.00676919519901276,  0.10399799048900604,
  -0.08693645149469376, -0.05739367753267288, -0.10309971123933792,
   0.1495797336101532 , -0.04602272808551788, -0.12811914086341858,
  -0.03223790228366852,  0.05612363666296005,  0.09379366040229797,
   0.27266234159469604,  0.8105431199073792 ,  0.36457276344299316,
  -0.15061186254024506,  0.10714779794216156,  0.14226435124874115,
   0.05299649015069008,  0.1869017779827118 , -0.15307578444480896,
  -0.16934731602668762, -0.0087080430239439 ,  0.2251482605934143 ,
   0.22342026233673096, -0.2723483741283417 ,  0.04450173303484917,
   0.00292717851698399, -0.04835348576307297,  0.07002007961273193,
   0.06086025759577751, -0.03673642128705978,  0.46417486667633057,
   0.07415776699781418,  0.02253350429236889, -0.08212077617645264,
  -0.1457844078540802 , -0.12810270488262177, -0.44923609495162964,
   0.07649093121290207, -0.11409388482570648, -0.01398754864931107,
   0.5403919816017151 , -0.17772191762924194,  0.3661705255508423 ,
   0.06623442471027374, -0.12209700047969818,  0.00941361486911774,
   0.00425473041832447,  0.15313346683979034, -0.31943613290786743,
  -0.01593426428735256, -0.03332437947392464,  0.06401968002319336,
   0.05750338733196259, -0.03233551234006882,  0.26672863960266113,
  -0.03255156800150871, -0.3631662428379059 , -0.13044114410877228,
   0.10902124643325806,  0.03471346944570541,  0.0244322195649147 ,
   0.18791283667087555,  0.00304470374248922, -0.6547154188156128 ,
  -0.00557103659957647, -0.03855372965335846,  0.03000154532492161,
  -0.16701874136924744, -0.3192210793495178 ,  0.11931097507476807,
  -0.05612112209200859, -0.09396998584270477, -0.959036111831665  ,
  -0.15679527819156647,  0.15360762178897858, -0.04000002145767212,
  -0.3913416266441345 , -0.02099782228469849, -0.2616426944732666 ,
   0.02077246643602848, -0.01175399217754602,  0.17303866147994995,
  -0.00393343670293689,  0.07465812563896179, -0.0982009544968605 ,
  -0.11934085190296173, -0.00248181959614158,  0.0475580245256424 ,
   0.28660014271736145, -0.1867484450340271 ],
 [-0.10686958581209183,  0.915277361869812  ,  0.2691260576248169 ,
   0.45657122135162354, -0.6635974645614624 ,  0.01664357632398605,
  -0.2672840356826782 , -0.07989544421434402, -0.14263279736042023,
  -0.16461758315563202,  0.34393420815467834, -0.1259273886680603 ,
   0.863562285900116  ,  0.04252847284078598, -0.48927104473114014,
  -0.06758161634206772,  0.9673892259597778 , -0.02313321083784103,
   0.3233543336391449 , -0.42333006858825684, -0.3013867139816284 ,
  -0.11259131878614426,  0.9018992781639099 ,  1.2518322467803955 ,
   1.9399802684783936 , -0.05639434978365898,  0.361762672662735  ,
  -0.7956143617630005 , -0.03657536581158638, -0.03820175305008888,
   0.3235026001930237 ,  0.3172869384288788 ,  0.115524522960186  ,
   0.7475549578666687 , -0.14743493497371674,  0.5719257593154907 ,
  -0.2753618359565735 , -0.05810442566871643,  0.21971377730369568,
   1.3304963111877441 ,  0.29901063442230225, -0.6211785078048706 ,
   0.04051517322659492,  0.3786170184612274 ,  0.10811859369277954,
  -0.33737310767173767,  0.561927855014801  , -0.5994866490364075 ,
  -0.3577573299407959 ,  1.8707538843154907 , -0.31267404556274414,
   0.11980700492858887,  0.00597219122573733,  0.07662767171859741,
   0.7417515516281128 , -0.3987222909927368 ,  0.4002333879470825 ,
  -0.33457013964653015,  1.5772318840026855 ,  0.8912971615791321 ,
   0.8244139552116394 , -1.8192272186279297 ,  2.4719784259796143 ,
  -0.1986219733953476 ,  0.22265654802322388,  1.1483488082885742 ,
   1.115254282951355  ,  0.4374691843986511 ,  0.9014608860015869 ,
   0.398749977350235  , -0.6884629726409912 ,  0.48100337386131287,
  -0.2595086395740509 ,  0.32735177874565125,  1.0478928089141846 ,
   0.19882619380950928,  0.5202258229255676 , -0.03331374749541283,
   0.18567243218421936, -0.9320865869522095 , -0.0953773558139801 ,
   0.09703785181045532,  1.1876044273376465 , -0.43875572085380554,
  -0.10456611216068268,  0.15513621270656586,  0.9165355563163757 ,
   0.00769357243552804,  0.3100745975971222 ,  0.16758964955806732,
  -0.45049986243247986,  0.39901402592658997,  0.5087239742279053 ,
   0.3534529507160187 , -0.7081606984138489 ,  0.09245474636554718,
   0.7957077622413635 ,  1.2992827892303467 ,  0.3142154812812805 ,
   0.24939879775047302,  0.2404526174068451 , -0.5831252336502075 ,
  -0.1265651136636734 ,  0.34440454840660095,  0.09112217277288437,
   0.00473600998520851, -0.7133845686912537 , -0.10888314247131348,
  -0.32721418142318726,  0.624182939529419  , -0.4759376645088196 ,
   0.5271070003509521 , -0.19897639751434326, -0.620703399181366  ,
   0.25991058349609375,  0.7788840532302856 ,  0.7643985748291016 ,
   0.30368635058403015, -0.9037848114967346 ,  0.2653976380825043 ,
   0.07839173823595047, -0.2098027616739273 ,  0.2559148669242859 ,
   0.6540369987487793 , -0.14245010912418365, -0.385758638381958  ,
  -0.6766449809074402 , -0.11739057302474976],
 [-0.00579076213762164, -0.05985219404101372,  0.01490700151771307,
  -0.10631723701953888,  0.07969609647989273, -0.03351717442274094,
  -0.40585413575172424, -0.05320084095001221, -0.17280252277851105,
   0.16139870882034302, -0.11982029676437378,  0.10200225561857224,
  -0.6229692697525024 , -0.0282843392342329 , -0.37491342425346375,
   0.14456389844417572, -0.0364086925983429 ,  0.2587471604347229 ,
   0.03172916173934937, -0.03385698795318604,  0.0934590995311737 ,
  -0.11492307484149933,  0.15558242797851562, -0.01132538449019194,
  -0.5201935768127441 ,  0.0029341084882617 ,  0.05620637908577919,
  -0.3514804542064667 , -0.00098320108372718,  0.3313179910182953 ,
   0.06487234681844711, -0.5625817775726318 , -0.06335577368736267,
  -0.01130407210439444,  0.18846328556537628,  0.5042247176170349 ,
   0.04620436951518059,  0.05052090436220169,  0.03741505369544029,
   0.2723175883293152 , -0.0027763512916863 , -0.00969706755131483,
   0.270491361618042  ,  0.03818462789058685,  0.02216603793203831,
   0.0167753342539072 , -0.10168090462684631, -0.04365465417504311,
   0.14846892654895782, -0.9370166063308716 ,  0.10030467808246613,
  -0.16594155132770538,  0.00779865682125092,  0.10556675493717194,
   0.1643560379743576 ,  0.00675411010161042, -0.11967093497514725,
   0.2079538255929947 , -0.13470248878002167, -0.29991844296455383,
  -0.24755772948265076,  0.08625738322734833, -0.02144953608512878,
   0.18026202917099   , -0.20453350245952606,  0.02442988753318787,
   0.03353194892406464, -0.02481476962566376,  0.7457971572875977 ,
   0.12491651624441147,  0.1287103146314621 , -0.5616309642791748 ,
  -0.10958165675401688,  0.05588584020733833,  0.14131897687911987,
   0.1114325225353241 ,  0.01216881722211838,  0.01491100527346134,
  -0.33492085337638855,  0.10793975740671158, -0.07836894690990448,
   0.3715328276157379 , -0.06189525127410889,  0.02041373401880264,
   0.03068234398961067,  0.12351520359516144, -0.06818100810050964,
   0.00866266526281834,  0.00809190329164267, -0.26238444447517395,
   0.00141316850204021,  0.03700610622763634,  0.3137916624546051 ,
  -0.03967723995447159, -0.04846757650375366, -0.2335038185119629 ,
  -0.22557485103607178, -0.22187429666519165,  0.5488560199737549 ,
   0.08649253845214844,  0.03692390397191048,  0.5841324925422668 ,
  -0.04135232791304588,  0.33110836148262024,  0.48156753182411194,
  -0.08605847507715225,  0.0503842793405056 , -0.03802429139614105,
  -0.18411287665367126,  0.02373158372938633, -0.543877899646759  ,
   0.00619162991642952, -0.20692014694213867,  0.0487721748650074 ,
   0.06599164754152298,  0.03152257576584816, -0.3985063135623932 ,
   0.1046336218714714 , -0.10320063680410385,  0.5072904825210571 ,
  -0.00014081005065236, -0.28144633769989014, -0.01582811214029789,
   0.4956250488758087 ,  0.02204296737909317,  0.1762632131576538 ,
  -0.837383508682251  ,  0.09570382535457611],
 [ 0.07123353332281113, -1.201530933380127  , -0.04187386855483055,
   0.4924199879169464 , -0.4046531915664673 ,  0.0251559279859066 ,
  -0.27033889293670654,  0.0114024868234992 , -0.8345600366592407 ,
   0.1296595335006714 ,  0.21842846274375916,  0.04609164595603943,
  -0.19839079678058624, -0.5992227792739868 ,  0.1293623000383377 ,
   0.02750417590141296, -0.8789442777633667 , -0.3438718914985657 ,
   0.5623736381530762 , -2.2078514099121094 , -0.00717338128015399,
  -0.2523624300956726 , -0.04016172885894775, -0.37675535678863525,
  -0.22218024730682373, -0.4151322543621063 ,  0.10900533199310303,
   0.43018004298210144, -0.5550267100334167 ,  0.00538672972470522,
  -0.01649283431470394,  0.7263579368591309 ,  0.7324700355529785 ,
  -0.941727340221405  , -0.29440587759017944,  0.4653618335723877 ,
   0.14147622883319855, -0.14206485450267792,  0.9807267189025879 ,
  -0.3147629499435425 ,  0.3209340274333954 , -0.6765508651733398 ,
   0.11306682974100113, -0.04829449206590652,  0.413209468126297  ,
   0.5108750462532043 ,  0.6320643424987793 ,  0.8076481223106384 ,
  -0.4325586259365082 ,  0.7184844017028809 , -0.2881052792072296 ,
  -0.22887185215950012, -1.1482778787612915 , -0.273048460483551  ,
  -0.14588963985443115, -0.29089707136154175, -0.01374397706240416,
   0.3773314952850342 ,  0.4151558578014374 , -0.00258156796917319,
   0.14152631163597107,  0.2714378833770752 ,  0.09993059933185577,
   0.01094909012317657,  0.12757469713687897, -1.149490237236023  ,
  -0.09275955706834793,  0.2772965431213379 ,  0.9815022349357605 ,
   0.3617686629295349 , -0.9088773727416992 ,  0.1655261218547821 ,
   0.3029839098453522 ,  0.6423121690750122 ,  0.01145464181900024,
   0.7546755075454712 , -0.7177332639694214 ,  0.6001814603805542 ,
  -0.5198895931243896 ,  0.02660574577748775, -0.00246439082548022,
   0.7694128751754761 , -0.7874297499656677 , -0.06215563043951988,
   0.21922677755355835,  0.1400086134672165 , -0.10489676892757416,
   0.3598358929157257 , -0.08259357511997223,  0.24024297297000885,
  -0.5849563479423523 ,  0.1618691086769104 ,  0.6346684694290161 ,
  -0.33767542243003845,  0.6040933132171631 ,  0.44853028655052185,
   0.728550910949707  , -0.3673251271247864 ,  0.04904649034142494,
  -0.07235800474882126,  0.1745656132698059 , -0.8747124671936035 ,
   0.00594512652605772, -0.14661437273025513,  0.13919860124588013,
  -0.24497075378894806,  0.2980313003063202 ,  0.02065059542655945,
   0.20444078743457794, -0.04695188254117966,  0.35804665088653564,
   0.4681940972805023 ,  0.972449004650116  ,  1.1175259351730347 ,
  -0.08592578768730164, -0.00363652035593987, -0.5952492356300354 ,
   0.6791414022445679 , -0.17524993419647217,  0.5995952486991882 ,
   0.22388103604316711, -0.6051123142242432 ,  0.00352474115788937,
  -0.2092839479446411 , -0.5892605185508728 , -0.11074353009462357,
  -0.16003920137882233,  0.967792809009552  ],
 [ 0.5499293804168701 ,  0.8185324668884277 , -0.3218056261539459 ,
  -1.3400804996490479 , -0.543721079826355  ,  0.6872041821479797 ,
  -3.449972629547119  ,  0.15671057999134064, -0.9453311562538147 ,
  -0.33250337839126587, -0.10068143159151077,  0.2070438712835312 ,
   0.07831036299467087,  0.9253778457641602 , -0.9507707953453064 ,
   0.18864434957504272,  0.49974170327186584, -0.39232638478279114,
  -0.6122098565101624 ,  0.9491382241249084 , -0.4503873288631439 ,
   0.38579076528549194, -0.9450714588165283 ,  0.938751757144928  ,
   0.29654791951179504, -2.130234479904175  , -0.20575383305549622,
   0.14963123202323914, -2.5275914669036865 ,  0.01703343726694584,
  -0.07141686975955963, -0.4479394257068634 , -1.0938304662704468 ,
  -1.0282268524169922 ,  0.1986675262451172 , -1.0057239532470703 ,
  -1.1034653186798096 , -0.01392323803156614,  0.4275580942630768 ,
  -0.25095030665397644, -0.201918825507164  ,  1.491426706314087  ,
   0.28781577944755554,  0.34270891547203064, -0.37151598930358887,
   0.08349739015102386, -1.957931637763977  , -0.7576590180397034 ,
   0.530570387840271  , -0.8012250065803528 , -0.01696441508829594,
   0.3096460700035095 ,  0.27331629395484924,  0.03225241228938103,
   0.0954047441482544 ,  0.3453555405139923 ,  1.444286823272705  ,
  -0.2001792937517166 , -0.2056317776441574 , -1.7593883275985718 ,
  -0.18220490217208862,  1.0716310739517212 , -0.9475135803222656 ,
  -1.1451629400253296 , -0.05359515175223351,  0.1569695621728897 ,
  -0.9002406001091003 , -1.0161235332489014 ,  0.44305816292762756,
   1.4260704517364502 , -0.49847733974456787, -0.26892170310020447,
  -0.27546000480651855, -1.1275520324707031 , -0.16188539564609528,
  -0.3670898377895355 , -0.193216010928154  , -0.5434403419494629 ,
  -0.8743075728416443 ,  0.1493125855922699 ,  0.22390638291835785,
  -1.0097323656082153 , -0.29951491951942444,  0.03035250678658485,
   0.29826274514198303, -1.3664356470108032 , -0.12410178780555725,
   0.8836654424667358 ,  0.16990196704864502, -0.06409580260515213,
   1.3327010869979858 , -0.223640576004982  ,  0.16058146953582764,
  -0.0609453059732914 ,  0.01651264168322086, -0.4179409146308899 ,
  -0.5528485774993896 , -1.6184108257293701 , -0.723956286907196  ,
   0.07369108498096466, -0.10380125045776367,  0.4859471321105957 ,
   0.13479451835155487, -0.4292571544647217 ,  0.2750707268714905 ,
   0.9024710655212402 , -0.2622016370296478 , -0.08438408374786377,
  -0.3877224028110504 ,  0.1593066304922104 , -0.4215881824493408 ,
   0.7915394902229309 , -0.4183565378189087 , -1.595724105834961  ,
  -0.4164631962776184 ,  0.3486916720867157 ,  0.36207544803619385,
  -0.7581493854522705 , -0.06337689608335495,  0.30571800470352173,
  -0.3713805675506592 , -0.10904031246900558, -0.08453329652547836,
   0.04157594963908195, -0.12939521670341492, -0.2382909059524536 ,
  -0.263971209526062  ,  0.587997317314148  ],
 [-0.35809358954429626, -0.6539981365203857 ,  0.10182372480630875,
  -0.8224422335624695 , -0.01547367591410875, -0.5007947087287903 ,
   0.2882010042667389 ,  0.33040890097618103,  0.9321571588516235 ,
   0.34808579087257385,  0.38128846883773804,  0.08923562616109848,
  -0.22997108101844788, -0.6884421110153198 ,  0.12236889451742172,
   0.02629830874502659, -1.3996950387954712 ,  0.24909862875938416,
  -0.14426590502262115,  0.08713210374116898, -0.3673538267612457 ,
   0.16867846250534058, -0.4939618706703186 , -0.16259698569774628,
  -2.401355266571045  , -2.392202138900757  , -0.6145772337913513 ,
  -0.1102321594953537 , -0.30271413922309875, -0.2806475758552551 ,
  -0.16000787913799286,  0.6674701571464539 , -2.088468074798584  ,
   0.01964947208762169,  0.10350143909454346, -0.30426889657974243,
  -0.14559306204319   ,  0.14219503104686737,  0.7329645156860352 ,
  -0.2346918284893036 , -0.9429952502250671 ,  0.09270086884498596,
   0.24474315345287323,  0.537902295589447  ,  0.06822628527879715,
   0.20289647579193115, -2.1908936500549316 , -0.20033888518810272,
   0.03942858800292015, -1.2967774868011475 , -1.0971038341522217 ,
  -0.01943853124976158,  0.09969230741262436, -0.00679733324795961,
   0.2616631090641022 , -0.3961775004863739 ,  0.7741560339927673 ,
  -0.46505340933799744, -0.07389383018016815, -0.16412541270256042,
  -0.5994904637336731 , -1.1996408700942993 , -1.3160916566848755 ,
  -0.4879297912120819 , -0.6713420748710632 ,  0.13220439851284027,
  -0.00722739147022367,  0.17998608946800232, -1.5632874965667725 ,
   0.04371022060513496, -0.7110608220100403 , -0.06961774826049805,
  -0.1656394749879837 , -0.02367759682238102, -0.04227275028824806,
   0.09069618582725525, -0.4316338896751404 , -0.6132889986038208 ,
  -2.1448073387145996 ,  0.19457174837589264,  0.5921263694763184 ,
  -2.145857810974121  , -0.2153356969356537 , -0.532626748085022  ,
  -0.07715879380702972, -0.2878251373767853 ,  0.551637589931488  ,
  -0.17128312587738037, -0.15057410299777985,  0.05531270802021027,
  -0.4691287577152252 ,  0.10368095338344574,  0.1464543491601944 ,
   0.536907970905304  , -0.08368796110153198, -0.03297630697488785,
   0.02853719517588615, -0.8434661030769348 ,  0.13042527437210083,
   0.11712460219860077, -0.44115519523620605, -1.8460564613342285 ,
  -0.16555777192115784,  0.1965516358613968 , -0.06045874580740929,
  -0.836283802986145  ,  0.03862186521291733, -0.7989833354949951 ,
   0.494587779045105  ,  0.02190354466438293, -1.0043787956237793 ,
   0.7955191135406494 ,  0.20407482981681824, -3.683910608291626  ,
   0.1842871755361557 ,  0.6318134069442749 , -0.3064148426055908 ,
   0.814632773399353  , -0.21422187983989716,  0.34450453519821167,
  -0.04016713798046112,  0.0539669394493103 , -0.8995137214660645 ,
   0.2956387996673584 ,  0.08865166455507278, -0.2897683382034302 ,
   0.35578009486198425, -1.5857104063034058 ],
 [ 0.02474955655634403, -0.0481877438724041 ,  0.00191535509657115,
  -0.3629228174686432 ,  0.4303268790245056 ,  0.09897878766059875,
  -0.00777167128399014,  0.09095550328493118, -0.05693269520998001,
   0.0460955910384655 ,  0.08825550228357315,  0.0454242117702961 ,
  -0.23538543283939362,  0.04227099940180779,  0.23851406574249268,
   0.06944269686937332, -0.04453622177243233,  0.06097492575645447,
  -0.0232922974973917 , -0.20039522647857666,  0.01878772675991058,
  -0.09859790652990341,  0.04521569237112999,  0.02295712567865849,
   0.32321032881736755,  0.2988191246986389 , -0.47527530789375305,
   0.5121572613716125 , -0.1318657398223877 , -0.12793508172035217,
   0.02461308427155018,  0.42539381980895996,  0.06985681504011154,
  -0.03227362036705017,  0.05123425647616386, -0.11689450591802597,
  -0.01568038016557693,  0.02169045619666576, -0.00050417374586686,
  -0.11484955251216888,  0.04078368097543716,  0.00489725125953555,
   0.00472983997315168, -0.09109073132276535,  0.08207610249519348,
   0.15054041147232056,  0.04718436673283577, -0.05109016597270966,
   0.4255128800868988 ,  0.3233147859573364 , -0.03211311623454094,
   0.14994674921035767, -0.0577881932258606 , -0.22587239742279053,
   0.09943081438541412, -0.5262482762336731 ,  0.07855008542537689,
   0.01853220537304878, -0.06343396753072739, -0.7687792778015137 ,
  -0.0838223546743393 ,  0.15078112483024597, -0.02339635044336319,
  -0.17266498506069183, -0.11550409346818924, -0.04656445980072021,
  -0.03676024451851845,  0.00080276210792363,  0.05154451355338097,
  -0.04890525713562965,  0.04348038136959076, -0.12696656584739685,
   0.11858242750167847,  0.00559906428679824,  0.02407043986022472,
   0.02835443615913391,  0.02222603186964989,  0.00865452084690332,
  -0.30499470233917236,  0.13326627016067505, -0.15998807549476624,
  -0.00128445262089372, -0.09320897608995438, -0.00888530071824789,
   0.005840755533427  , -0.04565314948558807, -0.10660889744758606,
  -0.06262100487947464, -0.39825090765953064, -0.03212465345859528,
   0.01512485463172197, -0.14892582595348358, -0.26183146238327026,
   0.1100231483578682 ,  0.04987367242574692,  0.13551391661167145,
   0.06547914445400238,  0.224683940410614  ,  0.17724758386611938,
   0.01826993376016617,  0.03308260813355446, -0.052780881524086  ,
  -0.1757141351699829 ,  0.03609624132514   , -0.4290015399456024 ,
  -0.03283915668725967,  0.37527284026145935,  0.08605674654245377,
   0.13738863170146942, -0.00875225476920605,  0.174690842628479  ,
  -0.21874119341373444,  0.15109401941299438, -0.06666973233222961,
  -0.12025091797113419,  0.06677968800067902, -0.13044503331184387,
  -0.00346420332789421, -0.03107307851314545, -0.09483039379119873,
  -0.08586906641721725, -0.124308742582798  ,  0.04298894107341766,
  -0.2201867401599884 , -0.10031882673501968,  0.2052966058254242 ,
  -0.06549708545207977,  0.18171943724155426],
 [-0.0657404437661171 , -0.6147107481956482 ,  0.04947451129555702,
  -0.5485137104988098 ,  0.05799292773008347, -0.9295589327812195 ,
   0.11030850559473038, -0.0362812764942646 ,  0.3407288193702698 ,
   0.08050964027643204, -0.4565286338329315 , -0.00136632495559752,
   0.04957696422934532, -0.16421553492546082,  0.11337273567914963,
  -0.03677161037921906, -0.6767174601554871 ,  0.3091709315776825 ,
  -0.5335401296615601 ,  0.05299590528011322, -0.08016511797904968,
   0.2313353717327118 , -0.21112394332885742,  0.21310697495937347,
  -0.00048493925714865,  0.00955096445977688,  0.2890577018260956 ,
  -1.261560082435608  , -0.19022774696350098,  0.47375136613845825,
  -0.02247528731822968, -1.3486542701721191 , -1.0759469270706177 ,
   0.6281242966651917 , -0.10943932831287384,  1.041955828666687  ,
   0.11649142950773239,  0.06638981401920319, -0.02619393169879913,
  -0.11043693870306015, -0.583855390548706  , -0.1656670868396759 ,
   0.03595714643597603, -0.9786089658737183 , -0.01056641060858965,
  -0.17320698499679565,  0.3749845623970032 , -0.4083908796310425 ,
  -0.29546117782592773,  0.02237912453711033, -0.3217866122722626 ,
   0.09692201763391495,  0.08484943956136703, -0.04158249124884605,
   0.12083206325769424,  0.01760681346058846, -1.2692610025405884 ,
  -0.3230937719345093 ,  0.2596896290779114 , -0.207171231508255  ,
   0.03629624843597412, -0.6236732602119446 , -0.17733792960643768,
   0.14869201183319092, -0.7753521800041199 , -2.220094919204712  ,
  -0.57550448179245   , -0.4978201687335968 ,  0.25085198879241943,
  -0.3606361448764801 ,  0.24688486754894257,  0.37034469842910767,
  -0.00168059719726443, -0.12105020135641098,  0.03954978287220001,
  -0.00292741484008729, -0.00921100191771984, -0.672161340713501  ,
   0.8911759853363037 , -0.10088733583688736, -0.11486431956291199,
   0.691581130027771  , -0.3044702708721161 , -1.5763779878616333 ,
  -0.8347800970077515 , -0.09468738734722137, -0.4341402053833008 ,
   0.06410717964172363,  0.00286777666769922,  0.04421372339129448,
   0.19639843702316284,  0.00795248243957758,  0.1940217763185501 ,
  -0.10994171351194382, -0.15796269476413727, -0.18136028945446014,
  -1.4025301933288574 , -0.15446288883686066,  0.26593491435050964,
  -0.02243116311728954,  0.6810913681983948 ,  0.43080848455429077,
   0.16011157631874084, -0.7054135203361511 ,  0.1668127477169037 ,
   0.15483435988426208,  0.3551070988178253 , -0.5265817046165466 ,
  -0.28434091806411743,  0.14372539520263672, -0.15114638209342957,
  -0.9021983742713928 , -0.5289389491081238 , -0.0988752469420433 ,
  -0.0095254322513938 , -0.757141649723053  , -0.09003863483667374,
   0.8372092247009277 ,  0.3528154194355011 , -0.20045776665210724,
  -0.33785393834114075,  0.05393103510141373, -0.8689188361167908 ,
  -0.23212584853172302, -0.3673270046710968 ,  0.02335382997989655,
  -0.1197328194975853 ,  0.2967936396598816 ]])
    b2 = jnp.array([-0.13318894803524017, -0.28615349531173706,  1.084829330444336  ,
 -0.6526830196380615 , -1.8417757749557495 , -0.9079123735427856 ,
 -0.618980348110199  , -2.2172248363494873 , -0.2706361711025238 ,
  0.26685771346092224, -1.1583447456359863 , -0.02487083524465561,
 -1.4563438892364502 , -0.40745073556900024, -0.31161558628082275,
 -0.9252951741218567 , -0.8801271915435791 , -0.12362851947546005,
 -0.5584339499473572 , -1.343237280845642  ,  0.42270752787590027,
 -1.5363364219665527 , -0.46996474266052246, -0.43437516689300537,
 -0.518986165523529  , -0.3161785900592804 , -0.4215046763420105 ,
 -0.42185571789741516, -0.2302391678094864 , -0.7040847539901733 ,
 -1.5942142009735107 , -1.1734371185302734 , -0.5179494023323059 ,
 -0.5553373098373413 , -0.2203117460012436 , -0.8280015587806702 ,
 -0.6984202265739441 , -0.01021495647728443, -1.3862308263778687 ,
  0.05723822116851807, -0.3402314782142639 ,  0.6049615740776062 ,
 -0.9454906582832336 , -1.6981004476547241 , -0.3209867775440216 ,
 -0.21120750904083252, -0.47462931275367737, -0.49209529161453247,
 -0.8449237942695618 , -1.7530393600463867 , -0.5700230002403259 ,
 -0.4214676320552826 , -0.16994446516036987, -1.1376771926879883 ,
 -0.7006009221076965 , -0.5525407195091248 , -0.42825689911842346,
 -2.032142400741577  , -0.14846791326999664, -0.9547633528709412 ,
 -0.939829409122467  , -0.8987848162651062 , -0.3358987867832184 ,
 -0.28915587067604065, -0.4182979166507721 , -0.5099419355392456 ,
 -0.16326238214969635, -0.30749449133872986, -0.777104377746582  ,
 -0.4877782166004181 , -0.29010796546936035, -0.01409272756427526,
 -2.533269166946411  , -0.8022250533103943 , -2.1174254417419434 ,
 -1.4046876430511475 , -0.20039735734462738, -0.7999417185783386 ,
 -1.24363112449646   , -0.38851508498191833, -0.8436776995658875 ,
 -0.8425124287605286 ,  0.2290547639131546 , -0.357087105512619  ,
 -1.2610942125320435 , -0.13150526583194733, -0.3697516620159149 ,
 -0.23906783759593964,  0.04303459450602531, -2.350137710571289  ,
 -0.5019490122795105 ,  0.05774541571736336, -0.785728394985199  ,
  0.44300854206085205, -0.3778268098831177 , -1.2138314247131348 ,
 -0.4518958628177643 , -0.9119753241539001 , -0.3252335786819458 ,
 -2.7589614391326904 ,  0.9775905013084412 , -0.6428359150886536 ,
 -0.04069336503744125, -0.4723084568977356 , -0.28719988465309143,
 -0.14492246508598328, -0.2651626169681549 ,  0.16276893019676208,
 -0.7824637293815613 , -1.6572275161743164 , -0.46176329255104065,
 -1.5089648962020874 , -0.49584126472473145, -0.14561349153518677,
  0.1439160406589508 , -0.9964129328727722 , -0.3748416602611542 ,
 -1.3059147596359253 ,  0.24487461149692535, -2.74047589302063   ,
 -0.6809260249137878 , -0.8797426819801331 , -0.1416134089231491 ,
 -0.3299342095851898 , -0.9015891551971436 , -0.9874178767204285 ,
 -0.42806875705718994, -1.2427414655685425 ])
    W3 = jnp.array([[ 0.42707234621047974,  1.3158140182495117 ,  0.16978466510772705,
  -0.37226736545562744, -0.24530284106731415,  1.3244636058807373 ,
   0.33562180399894714,  0.7808724641799927 , -0.08979624509811401,
   1.0520877838134766 ,  0.5336780548095703 , -0.07175956666469574,
   0.25661250948905945,  0.3230772316455841 , -0.5664600133895874 ,
  -0.08794672042131424,  0.21020527184009552, -0.9979880452156067 ,
   0.3602037727832794 , -0.07537277787923813,  0.7206051349639893 ,
   1.1999380588531494 , -0.2920741140842438 , -0.399972528219223  ,
   0.6944285035133362 , -0.8765077590942383 , -0.11016944795846939,
   1.6604903936386108 , -1.2634958028793335 ,  0.1575452834367752 ,
   1.0934906005859375 ,  0.3197273910045624 , -0.5799945592880249 ,
  -0.08966024965047836, -0.6157005429267883 ,  0.13010141253471375,
   2.0170276165008545 , -0.34952473640441895,  2.6628243923187256 ,
   0.5632680654525757 ,  0.8058643937110901 ,  0.01841873116791248,
  -0.6396350264549255 , -0.6961356997489929 , -1.54944908618927   ,
  -0.5091833472251892 ,  0.6765789985656738 , -0.7932758927345276 ,
   0.7776469588279724 ,  0.845458447933197  ,  0.38471901416778564,
  -0.04343079403042793,  0.24627096951007843,  0.4538663625717163 ,
   0.9267796874046326 ,  0.08579438179731369,  0.6146002411842346 ,
   0.0658722072839737 ,  0.02481606975197792, -1.3826658725738525 ,
   0.14734499156475067, -0.20952898263931274, -0.14413850009441376,
  -0.3401429057121277 ],
 [ 0.0516938604414463 ,  0.3826500177383423 ,  0.7918732762336731 ,
   0.14812922477722168, -0.21907667815685272,  0.7842683792114258 ,
   0.34233659505844116,  0.4635714292526245 , -0.3742314577102661 ,
   0.4866873025894165 , -0.1729629784822464 , -0.16008920967578888,
   0.002280488377437  , -0.20863312482833862, -1.6095603704452515 ,
  -0.1129901185631752 , -0.02221835218369961, -1.6406511068344116 ,
  -0.6124411225318909 , -0.3914833664894104 ,  0.10885047167539597,
   0.4510651230812073 ,  0.0781502053141594 , -0.41855388879776   ,
   0.20411691069602966, -0.02245388552546501, -0.5163755416870117 ,
   0.00678683118894696,  0.2993467152118683 , -0.47859668731689453,
   0.5635572075843811 , -0.1478530615568161 ,  0.13951881229877472,
  -0.8555849194526672 , -0.1634015142917633 ,  0.10576391965150833,
   0.6999514698982239 , -0.5178569555282593 , -0.6555070877075195 ,
  -0.9202871918678284 , -1.5687180757522583 , -0.8379417061805725 ,
   1.563796043395996  , -0.26798611879348755,  0.3393857479095459 ,
   0.4774218499660492 ,  0.3706270456314087 , -0.08555935323238373,
  -1.4748135805130005 ,  0.5887775421142578 , -0.01343390718102455,
   0.32148879766464233, -0.0068113817833364 ,  0.6782538890838623 ,
   0.2781195342540741 ,  0.7829116582870483 , -1.004547119140625  ,
   0.721099853515625  , -0.5041099786758423 ,  0.33913126587867737,
   0.8926241993904114 , -0.5504767298698425 , -0.74213045835495   ,
   0.16958318650722504],
 [-1.3826617002487183 , -0.4834475517272949 , -0.2566145956516266 ,
  -0.5048238635063171 ,  0.07024069875478745, -0.7491154670715332 ,
   0.24002590775489807, -0.41323694586753845, -0.44875603914260864,
  -1.9067803621292114 , -0.605792224407196  , -0.2770635485649109 ,
  -0.4593735933303833 , -0.18411210179328918, -0.21965482831001282,
  -0.08621882647275925, -0.461778461933136  , -0.255420446395874  ,
  -0.35728558897972107, -0.11954512447118759,  0.04199186712503433,
  -0.417362242937088  , -0.5975404381752014 ,  0.12156673520803452,
   0.07539805769920349,  0.13111163675785065, -0.1751597374677658 ,
  -1.0807865858078003 , -1.9704182147979736 , -0.3174316883087158 ,
  -0.8237972855567932 ,  0.03838449716567993, -0.5112139582633972 ,
  -0.8030237555503845 , -0.13042475283145905, -0.18608029186725616,
  -0.2935364842414856 , -0.2980123460292816 , -0.4645206928253174 ,
  -0.11121942847967148,  0.1689719706773758 ,  0.359271377325058  ,
  -0.11802662163972855, -0.00279691792093217,  0.0548422746360302 ,
  -1.5503482818603516 ,  0.1055084690451622 , -0.8195127844810486 ,
  -0.6828834414482117 , -0.00169000506866723, -0.9211329221725464 ,
  -0.00324861286208034, -0.3021320700645447 , -0.21480236947536469,
  -1.186006784439087  , -1.0236990451812744 , -0.3236372470855713 ,
   0.3534865379333496 ,  0.01510641351342201, -0.5825267434120178 ,
  -0.20934735238552094,  0.24987763166427612,  0.07114806771278381,
   0.2439068853855133 ],
 [ 0.9552440047264099 , -0.8976749181747437 , -0.022051602602005  ,
   0.4101095497608185 ,  0.330567330121994  , -0.27793461084365845,
  -0.3024626672267914 ,  0.5826188325881958 , -0.08791240304708481,
   1.4339672327041626 ,  0.4911382496356964 , -0.04867415875196457,
  -0.03482595086097717,  0.37362349033355713,  0.2965462803840637 ,
   0.21957853436470032,  0.3846518397331238 ,  0.14540141820907593,
   0.16253477334976196,  0.05866295099258423,  1.420649528503418  ,
  -0.7637187838554382 , -0.41453489661216736, -0.24669910967350006,
  -0.5940107107162476 , -0.04174590110778809, -0.1409148871898651 ,
   0.6693310737609863 , -0.12993597984313965, -0.13132989406585693,
   0.5723053812980652 , -0.14241673052310944,  0.5878564119338989 ,
   0.20326025784015656,  0.07545407861471176, -0.04776076227426529,
   0.6785509586334229 , -0.05625214800238609,  0.33517640829086304,
  -0.33840709924697876,  0.0841468796133995 ,  0.00005852197864442,
  -0.23130738735198975,  0.357945054769516  ,  0.5050272345542908 ,
  -1.5109894275665283 , -0.2822094261646271 , -1.0206853151321411 ,
   0.5241442322731018 , -0.5664504170417786 ,  1.1371620893478394 ,
   0.1107107475399971 , -0.43887001276016235, -0.05762934684753418,
  -0.01162102911621332, -0.3681294620037079 , -0.2948657274246216 ,
  -0.04215576872229576, -0.1430707722902298 ,  0.08474241942167282,
  -1.3844507932662964 , -0.41912806034088135, -0.23402650654315948,
  -0.44039446115493774],
 [-0.02763044647872448, -0.00091791618615389,  0.2456357628107071 ,
  -1.2665307521820068 ,  0.03453817963600159,  0.11394435167312622,
   1.3077291250228882 ,  0.28961074352264404, -0.0201600193977356 ,
   0.2027221918106079 , -0.6099473834037781 , -1.2443757057189941 ,
  -1.177466869354248  ,  0.4414166808128357 , -0.09606682509183884,
   0.05813333764672279,  0.5550578236579895 , -0.521190881729126  ,
  -2.511380434036255  , -0.11593654006719589,  0.16529585421085358,
  -0.1721302717924118 ,  0.21227234601974487, -0.38431644439697266,
   0.8507821559906006 , -0.00782989803701639,  0.16293370723724365,
  -0.07400039583444595,  0.606158971786499  ,  0.14038684964179993,
   0.32547473907470703,  0.04037416726350784, -0.6494630575180054 ,
  -2.289168357849121  , -1.5943148136138916 , -0.6092962622642517 ,
  -2.0119993686676025 , -0.5640865564346313 , -0.256661593914032  ,
  -3.5502500534057617 ,  0.5504606366157532 , -0.0627797320485115 ,
   0.17872241139411926,  0.09226761758327484,  0.03042916022241116,
   0.6056593060493469 , -0.5929285883903503 , -0.00230814889073372,
   0.30719155073165894,  0.3174647390842438 , -0.35501229763031006,
   0.01349153835326433, -1.2151073217391968 , -0.7924789190292358 ,
  -1.4056462049484253 ,  0.2644103467464447 , -0.3247024714946747 ,
  -0.16129128634929657,  0.16498076915740967, -0.20459552109241486,
  -0.4736957252025604 , -0.1908554881811142 , -0.7156304717063904 ,
   0.45671793818473816],
 [-0.1383294314146042 , -0.7769845128059387 , -0.1253240406513214 ,
  -1.9392991065979004 , -0.39328107237815857, -0.6869182586669922 ,
   0.18418797850608826, -0.470310240983963  , -0.04315009713172913,
  -0.8550460338592529 ,  0.528076171875     ,  0.2942696809768677 ,
   1.039818525314331  ,  0.4551563560962677 ,  0.7831813097000122 ,
   0.13723914325237274,  0.4951667785644531 ,  0.13980253040790558,
   0.9354941248893738 ,  0.04516710340976715,  0.4492689371109009 ,
  -1.978636622428894  , -0.27297255396842957,  0.18096928298473358,
   0.46196654438972473, -0.22385276854038239, -0.11532830446958542,
   0.26625779271125793,  0.6554652452468872 , -0.3043309450149536 ,
  -0.22713811695575714, -0.7132090926170349 ,  0.13274207711219788,
  -0.30830633640289307, -1.6142468452453613 , -0.8986025452613831 ,
   0.7224611043930054 , -1.1415138244628906 ,  1.8048166036605835 ,
   0.5751243829727173 ,  0.41131091117858887, -0.13750720024108887,
  -0.6668689250946045 ,  0.0274478904902935 , -0.08852691948413849,
   0.03575050458312035, -1.292883276939392  ,  0.9657122492790222 ,
   1.195366621017456  ,  0.05025354400277138, -0.29963937401771545,
  -0.1726541817188263 ,  0.6621040105819702 ,  0.20094405114650726,
  -0.364737331867218  ,  1.0709627866744995 ,  0.4937207102775574 ,
  -0.6075515151023865 ,  1.5362979173660278 ,  0.9095636010169983 ,
  -0.4010738432407379 , -0.00799816567450762,  0.06695855408906937,
  -0.5133187174797058 ],
 [-0.8201977014541626 , -0.5272479057312012 ,  0.12919306755065918,
   0.3144274950027466 ,  0.1764848232269287 ,  0.02438506484031677,
  -0.6681184768676758 ,  0.04519164189696312,  0.18379199504852295,
  -0.37515127658843994, -0.6897687911987305 , -0.11319320648908615,
   0.1854117065668106 ,  1.0981123447418213 ,  0.17660196125507355,
  -0.6461565494537354 , -0.00471107475459576,  1.0033073425292969 ,
   1.4894815683364868 ,  0.03643696010112762, -0.2126632183790207 ,
   0.09542664885520935, -0.7001151442527771 , -0.6221720576286316 ,
   0.5147857666015625 ,  0.2378820925951004 , -0.04462632164359093,
   0.6976940631866455 ,  0.37589162588119507,  0.6747041940689087 ,
   0.2789672315120697 , -0.11672655493021011, -0.5385924577713013 ,
   0.7970767021179199 , -0.6161926984786987 , -0.28170719742774963,
  -0.8117119669914246 ,  0.9964715838432312 , -0.3694002032279968 ,
  -0.19689396023750305,  0.3920094966888428 ,  0.01906332932412624,
  -0.38437145948410034, -0.8749625086784363 ,  0.09051614254713058,
   0.5535284280776978 , -0.36089780926704407, -1.863822340965271  ,
   0.6335043907165527 ,  0.4411359429359436 , -1.2283159494400024 ,
  -0.2969907224178314 ,  0.4849979877471924 , -0.3967719078063965 ,
  -1.6294012069702148 , -0.5738749504089355 ,  0.4086170792579651 ,
   0.28955474495887756,  0.3236263394355774 , -0.46454331278800964,
  -0.40009042620658875,  0.35399699211120605,  0.5081523656845093 ,
   0.4476873278617859 ],
 [-0.33357465267181396, -0.28811395168304443, -0.05564836785197258,
  -0.5603868365287781 ,  0.3474920392036438 ,  0.34073951840400696,
  -0.04688156396150589, -0.292570024728775  , -0.1554563194513321 ,
   0.02984649874269962, -0.14928974211215973,  0.08091673254966736,
   0.299031138420105  ,  0.22082890570163727,  0.10765615105628967,
   0.07229675352573395,  0.03212138265371323, -0.6999498009681702 ,
  -0.15417331457138062, -0.07226679474115372, -0.125226691365242  ,
   0.16040804982185364,  0.04641420021653175,  0.10715118050575256,
  -0.39479783177375793, -0.0331437885761261 , -0.03657737001776695,
  -0.2901346981525421 , -0.5805837512016296 ,  0.2090928554534912 ,
  -0.26870864629745483, -0.14822453260421753, -0.8819698095321655 ,
  -0.06992022693157196, -0.08697596937417984, -0.4607824385166168 ,
   0.03388295322656631, -0.8290340900421143 , -0.37409353256225586,
  -1.242460012435913  ,  0.17408935725688934,  0.04679477959871292,
  -0.2951449155807495 , -0.1618689000606537 ,  0.3247303068637848 ,
   0.6661699414253235 ,  0.06105413287878036, -0.00300091435201466,
   0.9296797513961792 , -1.5399380922317505 , -0.5484580397605896 ,
   0.06831566989421844, -0.02182662487030029, -1.224868655204773  ,
  -1.4439483880996704 , -1.137838363647461  ,  0.01002908311784267,
  -0.06733130663633347,  0.1538330614566803 , -0.22397714853286743,
  -0.573046088218689  ,  0.18365278840065002, -0.07308723032474518,
  -0.4066265821456909 ],
 [ 0.9549562335014343 ,  0.1675848364830017 ,  0.31703710556030273,
   1.3277188539505005 , -0.00878393463790417,  0.6950680613517761 ,
  -0.5831920504570007 ,  0.45286670327186584,  0.2090093195438385 ,
   1.4883922338485718 ,  0.18321636319160461, -0.45738697052001953,
   0.8089631795883179 ,  1.0006824731826782 , -0.22994504868984222,
   0.18510755896568298,  1.5688811540603638 , -2.69085693359375   ,
   0.12011341750621796,  0.06844265013933182, -0.3148726224899292 ,
   0.31279799342155457, -1.5700708627700806 , -0.22416231036186218,
  -0.04272887855768204, -1.018856167793274  , -0.14207781851291656,
   1.7429851293563843 , -0.14870613813400269,  0.9249921441078186 ,
  -0.01569669879972935,  0.37644991278648376,  0.9741100668907166 ,
  -0.2798424959182739 , -2.482006072998047  ,  0.39067408442497253,
  -0.43498551845550537, -0.32050320506095886, -0.09174829721450806,
   0.7134035229682922 ,  2.5360376834869385 , -0.04838234558701515,
  -0.2313491553068161 ,  0.5466126203536987 ,  0.5093797445297241 ,
   0.6602337956428528 , -0.06042768061161041,  0.8385248780250549 ,
  -0.6633597612380981 , -0.17066994309425354, -1.5179094076156616 ,
  -1.7379891872406006 , -1.613045573234558  , -0.5842558741569519 ,
   0.5458609461784363 , -0.371532678604126  , -0.3633878231048584 ,
  -0.1564924418926239 ,  0.14489857852458954,  0.7431344985961914 ,
  -0.3503929078578949 , -0.5602005124092102 , -0.4062424898147583 ,
   1.4249237775802612 ],
 [ 0.08979751169681549,  0.2372124046087265 ,  0.11907612532377243,
  -0.9080750346183777 , -0.08425538241863251,  0.05867643281817436,
  -0.2799552381038666 ,  0.00650986190885305,  0.0299011655151844 ,
   0.08881384134292603, -0.05548006668686867, -0.6925825476646423 ,
  -1.2794157266616821 , -0.3490138649940491 , -0.06609847396612167,
   0.6209722757339478 , -0.5639550089836121 , -0.11350229382514954,
  -1.0301178693771362 , -0.07884302735328674,  0.06872235238552094,
  -0.42581385374069214, -0.9162619113922119 , -0.3956655263900757 ,
   0.29856905341148376,  0.03200259804725647, -0.01449046190828085,
  -0.4533628225326538 , -0.3267399072647095 , -0.6286667585372925 ,
  -0.8650569915771484 ,  0.39353707432746887,  0.03090475127100945,
  -0.31112030148506165, -0.256996750831604  ,  0.11707179248332977,
  -0.1807476431131363 , -0.2560519874095917 , -0.3690301775932312 ,
  -0.10884401947259903,  0.13740281760692596, -0.05447882786393166,
  -0.1835809201002121 , -0.07167541980743408, -0.5476264953613281 ,
  -0.5037282705307007 , -0.01821559108793736, -0.7302946448326111 ,
  -0.9011461138725281 , -0.11875956505537033, -1.4247570037841797 ,
  -0.10787942260503769, -0.5366371870040894 , -0.06104855239391327,
   0.09772051870822906, -0.0198657214641571 , -0.074310302734375  ,
   0.03446782752871513,  0.03872910141944885,  0.17302292585372925,
  -0.07216417789459229,  0.2695756256580353 ,  0.02653327211737633,
  -0.10570920258760452],
 [ 0.22131502628326416, -0.7368330955505371 ,  0.2117786556482315 ,
   0.16318197548389435, -0.06410326063632965,  0.12410596013069153,
   0.9583477973937988 ,  0.2577005922794342 , -0.14435970783233643,
   0.11392626911401749,  0.6942487955093384 ,  1.8184248208999634 ,
  -2.3557064533233643 ,  0.48834437131881714, -1.48825204372406   ,
   0.8389508724212646 , -0.3907233476638794 , -0.07399606704711914,
   0.2959495782852173 ,  0.04437234625220299, -0.07815414667129517,
   0.42049872875213623, -0.6459863781929016 ,  0.6842886209487915 ,
  -0.43758729100227356,  0.7461100220680237 ,  0.14199049770832062,
  -0.07651805877685547, -0.20739856362342834, -0.06332982331514359,
  -0.04398404434323311, -1.0094482898712158 ,  0.10135409981012344,
  -1.09980309009552   , -0.25255751609802246, -0.07229340821504593,
   0.17481352388858795,  0.21435953676700592,  0.02404100075364113,
  -0.6559436917304993 ,  0.3477976620197296 , -0.1188012957572937 ,
   0.06667521595954895, -0.2873978316783905 ,  1.5806996822357178 ,
  -0.38225868344306946,  0.11239704489707947, -0.3731842637062073 ,
  -1.4544808864593506 , -0.2889164090156555 ,  0.1583530455827713 ,
   0.2033272236585617 , -0.1203785091638565 , -0.8613795042037964 ,
  -0.07380856573581696,  0.650654137134552  ,  0.5777931213378906 ,
   0.08074752241373062,  0.2585175335407257 ,  0.2795279920101166 ,
   0.22800536453723907,  0.8994249701499939 ,  0.9255561828613281 ,
  -0.20665062963962555],
 [ 0.01964635215699673, -0.10505162179470062,  0.16453826427459717,
   0.39639291167259216,  0.21418197453022003, -0.7224332094192505 ,
  -0.25098109245300293,  0.6442276835441589 , -0.26605525612831116,
   2.967618227005005  ,  0.6181628704071045 ,  0.25649455189704895,
   0.6154523491859436 ,  0.09563599526882172, -0.07673909515142441,
   0.15296314656734467,  0.5987379550933838 , -0.5703911185264587 ,
   0.14698608219623566,  0.1607426106929779 , -1.410164713859558  ,
   0.9867324829101562 ,  0.6745977997779846 , -0.0684589371085167 ,
  -0.42985963821411133,  0.0386766716837883 , -0.16003957390785217,
   1.3033816814422607 ,  1.3276631832122803 ,  0.16735492646694183,
   0.3022564649581909 , -0.09352404624223709, -0.9791883230209351 ,
  -0.6787899732589722 ,  0.21010422706604004, -0.6362708210945129 ,
   0.5513677597045898 , -0.00406049285084009, -0.18027298152446747,
  -0.1269008070230484 , -0.288711816072464  ,  0.0796409547328949 ,
   0.04208121448755264,  0.16924166679382324,  0.26661911606788635,
  -0.9961016178131104 ,  0.05674203857779503, -0.7025057673454285 ,
   0.3278460204601288 , -0.7497153282165527 , -0.4937724769115448 ,
  -0.03993722051382065,  0.0042928010225296 ,  0.19218076765537262,
   1.5883092880249023 ,  1.4006307125091553 ,  0.43174830079078674,
   0.01584027893841267, -0.3223542273044586 , -0.8743342161178589 ,
   0.1430549919605255 , -1.4887539148330688 , -0.08098699152469635,
  -0.6626445055007935 ],
 [ 0.2911941707134247 , -0.01460314448922873,  0.07141506671905518,
  -0.35970985889434814,  0.26007112860679626,  0.12091247737407684,
  -0.2622867226600647 , -0.15229903161525726,  0.08088168501853943,
   0.34447675943374634, -0.2558642327785492 ,  0.04166373610496521,
  -1.3583215475082397 ,  0.4200685918331146 ,  0.09743614494800568,
   0.3798206150531769 , -0.06073285266757011,  0.49523860216140747,
  -0.4154610335826874 ,  0.05594643950462341,  0.11580852419137955,
  -0.8754064440727234 ,  0.42917636036872864,  0.28392311930656433,
  -0.24343328177928925,  0.24472448229789734,  0.30222558975219727,
   0.03659863770008087,  0.33712682127952576,  0.21158482134342194,
   0.37362566590309143, -0.1236351877450943 ,  0.6315000653266907 ,
   0.20738323032855988, -1.0560201406478882 ,  0.1983519047498703 ,
   0.2690371870994568 ,  0.15539704263210297, -0.8606829047203064 ,
  -0.5204675793647766 ,  0.16780412197113037, -0.5174614787101746 ,
   0.22701194882392883, -0.39095839858055115,  0.448800653219223  ,
   0.5039350390434265 ,  1.1819299459457397 , -0.04043978452682495,
  -0.02705572545528412, -0.3558667004108429 ,  0.27288898825645447,
   0.348604679107666  ,  0.3804478645324707 , -0.4516697824001312 ,
   0.18441905081272125, -1.5761996507644653 , -0.0695042833685875 ,
  -0.04444394633173943,  0.20437346398830414,  0.05509265139698982,
  -0.03950116038322449,  0.10407502204179764, -0.18966616690158844,
  -0.01192242652177811],
 [ 0.18650248646736145,  0.05433755367994308,  0.6742858290672302 ,
  -0.03746648505330086,  0.4973798990249634 ,  0.2583678960800171 ,
   0.23338179290294647,  0.09528513252735138,  0.1641560047864914 ,
   0.09449528157711029,  0.7712965607643127 ,  0.25467434525489807,
   0.13437750935554504,  0.40495815873146057, -1.297520399093628  ,
   0.4085676372051239 ,  0.04269793629646301, -1.1875486373901367 ,
  -0.12310615181922913,  0.08156099915504456,  0.4701724350452423 ,
   1.3087247610092163 , -1.2525780200958252 , -0.8085910081863403 ,
  -0.5604904890060425 , -0.5129668116569519 , -0.9197690486907959 ,
  -0.7736737728118896 , -0.31626206636428833, -0.7987257838249207 ,
   0.04650269821286201, -0.12093072384595871, -0.2869970500469208 ,
   0.7635217905044556 ,  0.19585448503494263, -0.26973316073417664,
   0.34634923934936523, -0.2677020728588104 ,  0.5151034593582153 ,
   0.48574501276016235,  0.04822328686714172, -0.5241801142692566 ,
   0.14323078095912933, -0.6162075400352478 ,  0.4081079959869385 ,
   0.24858926236629486,  0.5182514190673828 , -0.21337024867534637,
  -0.11863250285387039,  0.4772813320159912 ,  0.13219054043293   ,
   0.14598174393177032,  0.23812958598136902,  0.16909818351268768,
   1.080137014389038  ,  0.7440742254257202 , -0.7589207887649536 ,
   0.35241395235061646, -0.9010071754455566 , -1.8561674356460571 ,
   0.532867431640625  , -0.3655936121940613 , -0.9328182935714722 ,
   0.7241498827934265 ],
 [ 0.8978239893913269 ,  0.5934350490570068 ,  0.5621721744537354 ,
   0.11529314517974854,  0.40565329790115356,  0.7610434889793396 ,
   0.0112681370228529 ,  1.3284239768981934 , -0.2927888035774231 ,
   1.060360312461853  , -0.04974411055445671,  1.0078099966049194 ,
  -0.03079568222165108, -0.05202426761388779,  1.1307226419448853 ,
  -0.3211236596107483 ,  0.2098453789949417 , -1.1430877447128296 ,
   0.5930020213127136 ,  0.14622937142848969,  0.3507520258426666 ,
   0.3615271747112274 ,  0.3788442313671112 , -0.2475503832101822 ,
  -0.47830674052238464, -0.2541349232196808 , -0.2766786217689514 ,
   1.0650367736816406 , -0.5629420876502991 , -0.22867345809936523,
   1.1117630004882812 ,  0.7613403797149658 ,  1.3719074726104736 ,
   0.6490932106971741 , -1.3184865713119507 ,  1.0236364603042603 ,
  -0.8871152400970459 ,  0.41117435693740845,  0.28144747018814087,
   0.3030165731906891 ,  0.08410908281803131, -0.39411312341690063,
   0.7316725850105286 , -0.7901281714439392 , -0.5949000716209412 ,
  -0.17532141506671906, -2.0027832984924316 ,  0.02639223448932171,
   1.077721357345581  , -0.4893496632575989 ,  0.7701483368873596 ,
   0.318246990442276  , -0.843352198600769  ,  0.47711503505706787,
   0.0589134506881237 , -0.6768496632575989 ,  0.8976317048072815 ,
  -0.16663585603237152,  0.5145191550254822 , -1.1556406021118164 ,
   0.8836460709571838 ,  0.08852174133062363,  0.05196977034211159,
  -0.05416976660490036],
 [ 0.966951847076416  , -0.01380723714828491,  0.33854860067367554,
  -0.01129056699573994,  0.4837721586227417 , -0.1145436093211174 ,
  -0.05756685137748718,  0.1295737326145172 ,  0.1842753142118454 ,
   0.08096694946289062,  0.08164295554161072, -1.427781343460083  ,
  -0.4997861385345459 , -0.7332482933998108 , -0.3227069079875946 ,
  -0.09151916950941086,  0.04965806007385254, -0.11330907791852951,
  -0.22861838340759277,  0.38989129662513733, -0.7492971420288086 ,
  -0.14789438247680664,  0.22442419826984406, -0.31119877099990845,
  -0.4253653287887573 , -0.5780383348464966 , -0.2573204040527344 ,
   0.2971721589565277 ,  0.31839320063591003, -0.0227544903755188 ,
   0.5239906311035156 , -0.4795674979686737 , -0.5831936597824097 ,
   0.04155890643596649, -0.5529459714889526 , -0.32365331053733826,
  -0.3191775679588318 , -0.16173464059829712,  0.4660986363887787 ,
  -1.1938703060150146 , -0.5276688933372498 , -0.37019777297973633,
   0.23029431700706482, -0.3220655024051666 , -0.6396335959434509 ,
   0.1397123634815216 , -0.9318822622299194 ,  0.0582527257502079 ,
  -0.2251717448234558 , -0.08198053389787674, -0.12320487201213837,
  -0.583138108253479  , -0.14527447521686554, -0.4200851321220398 ,
   0.3246100842952728 ,  0.3122260272502899 , -0.3233650028705597 ,
  -0.40358060598373413,  0.2250930517911911 ,  0.07119601964950562,
  -0.05136048048734665, -0.4209272265434265 , -0.5142295956611633 ,
  -0.8964108228683472 ],
 [ 0.9779720306396484 , -0.25577136874198914, -0.07692957669496536,
  -2.1436243057250977 , -0.32712212204933167,  0.8657180666923523 ,
   0.43305322527885437, -0.3258928656578064 ,  0.2103906124830246 ,
   1.7411917448043823 , -0.605501651763916  , -0.7397631406784058 ,
  -1.67446768283844   , -1.0100077390670776 ,  0.30019113421440125,
   0.38518211245536804,  1.3347684144973755 , -0.8655030131340027 ,
  -0.3769401013851166 ,  0.05715404078364372, -0.7714165449142456 ,
  -0.09632144123315811, -1.1180516481399536 ,  0.2968160808086395 ,
  -0.09856262803077698,  0.4579120874404907 , -0.47376585006713867,
   0.4258982539176941 , -0.1830902248620987 , -0.7586058378219604 ,
   0.9766659140586853 , -0.6388899087905884 ,  0.6951485276222229 ,
  -0.5113273859024048 , -0.02531112544238567, -0.8461331129074097 ,
   1.7594491243362427 , -1.4461522102355957 ,  0.05715528503060341,
   0.8694764375686646 , -2.3159449100494385 ,  0.01845875568687916,
   0.3881130814552307 ,  0.668555498123169  ,  0.7832801342010498 ,
  -0.4751686751842499 , -0.9854073524475098 , -0.1264687180519104 ,
  -1.2888166904449463 , -0.980500340461731  , -0.09702364355325699,
   0.11215871572494507,  0.21078911423683167, -0.4171961843967438 ,
   1.5016412734985352 ,  0.8333282470703125 ,  0.5004149675369263 ,
   0.14018592238426208, -1.1811752319335938 ,  0.24132663011550903,
  -0.02803959883749485, -2.1267096996307373 ,  0.6197586059570312 ,
   0.4697108566761017 ],
 [-0.07129386812448502,  0.06797745078802109, -0.14212031662464142,
  -0.01117481291294098,  0.08098047971725464,  0.08602498471736908,
  -0.5398725867271423 , -0.15388983488082886,  0.13520553708076477,
  -0.5239565968513489 , -0.38925591111183167, -0.06998259574174881,
  -0.595997154712677  , -0.12352600693702698,  0.2534303665161133 ,
  -0.02937318198382854,  0.06864428520202637, -0.4659738540649414 ,
   0.05599220469594002, -0.04679112136363983, -0.13464629650115967,
  -0.05431205406785011, -0.36522176861763   ,  0.28962498903274536,
  -0.06711024791002274,  0.14968352019786835,  0.19226686656475067,
  -0.02929368242621422,  0.11144424974918365,  0.04470508545637131,
  -0.35245394706726074, -0.9052175879478455 , -0.04049154371023178,
  -0.03605657443404198, -1.0135130882263184 , -0.01137980073690414,
   0.04085827991366386, -1.238908290863037  ,  0.15334218740463257,
  -0.2516172230243683 ,  0.04619958996772766, -0.06263241916894913,
  -0.17823337018489838, -0.05407421663403511,  0.06228327751159668,
  -0.15225140750408173, -0.5776366591453552 ,  0.1425500512123108 ,
  -0.6682931780815125 , -0.1961979866027832 ,  0.02279843017458916,
   0.00649964762851596,  0.0713738426566124 , -1.2405791282653809 ,
  -0.5489571690559387 , -0.23005311191082   ,  0.00962390657514334,
  -0.08627829700708389,  0.2731357216835022 ,  0.09627437591552734,
  -0.43370598554611206, -0.18458086252212524, -0.24855682253837585,
   0.23005567491054535],
 [-0.36481738090515137,  0.11834073811769485,  0.1486557126045227 ,
  -0.8119195103645325 ,  0.17067381739616394,  0.6567133069038391 ,
  -0.04774453490972519, -0.6980887651443481 ,  0.05076036229729652,
   0.4421246647834778 , -0.02076053433120251,  0.16814565658569336,
  -0.40357261896133423,  0.03520147874951363, -0.30556637048721313,
  -0.01026680320501328, -2.1358447074890137 , -0.529235303401947  ,
   1.2171729803085327 , -1.507752537727356  ,  0.01910552196204662,
   0.5575767159461975 , -0.5951172113418579 ,  0.27615344524383545,
  -0.934169352054596  , -0.45018091797828674,  0.07311937212944031,
   0.2300635129213333 , -0.41779252886772156, -0.06338484585285187,
  -0.7036186456680298 ,  0.46610045433044434,  0.04504895582795143,
  -1.5545545816421509 ,  0.39428719878196716, -1.6601982116699219 ,
  -1.4384044408798218 , -0.7835440039634705 ,  0.6915403008460999 ,
   0.7870878577232361 ,  0.8486480116844177 , -0.13282662630081177,
  -0.4679783284664154 ,  0.6449063420295715 , -2.3227014541625977 ,
   0.11818176507949829,  0.531926691532135  ,  0.0783650204539299 ,
  -0.44911888241767883,  0.33191990852355957, -0.09149724990129471,
   0.3181585371494293 , -0.703895092010498  ,  0.0651528537273407 ,
   0.4071360230445862 , -0.19383053481578827,  0.04940654709935188,
   0.11404556035995483,  0.0715041384100914 , -0.09841518104076385,
   0.7698667645454407 , -0.10949412733316422, -0.10344751924276352,
   0.00787969306111336],
 [ 0.00541082629933953,  0.18482287228107452, -0.08095043152570724,
   0.15017615258693695, -0.3745536506175995 , -0.12823599576950073,
   0.13926257193088531,  0.4197647273540497 , -0.23520055413246155,
   0.6998355984687805 , -0.42214706540107727,  1.2128515243530273 ,
   0.46921277046203613,  0.2952393889427185 , -0.441150963306427  ,
  -0.16439712047576904, -1.259272575378418  ,  0.19176295399665833,
  -0.24265475571155548, -0.06254848092794418,  0.00820649415254593,
  -0.39434781670570374, -0.1999114453792572 ,  0.11437905579805374,
   0.42961907386779785,  0.0724632516503334 , -0.04764808341860771,
   0.2854028046131134 , -0.09833347052335739,  0.1372154951095581 ,
   0.1653600037097931 ,  0.18707740306854248, -1.0645523071289062 ,
  -0.12321542203426361,  0.33728352189064026,  0.1650485247373581 ,
   0.5595249533653259 ,  0.00737758399918675,  0.4382855296134949 ,
  -0.4297579824924469 , -0.22301694750785828, -0.9100924730300903 ,
  -0.04167112335562706, -0.6231600046157837 , -0.07726224511861801,
  -0.2799466550350189 , -0.5750753283500671 , -0.00288964039646089,
   0.6051751375198364 ,  0.135272815823555  , -0.4248442053794861 ,
  -0.7250502109527588 , -0.505477786064148  , -0.9286800026893616 ,
   0.12429722398519516, -0.37483954429626465,  0.01751468516886234,
  -0.4389963746070862 ,  0.5188678503036499 , -0.4058150053024292 ,
  -0.02650138363242149,  0.3808092474937439 ,  0.29499348998069763,
  -0.9690614342689514 ],
 [-1.0167509317398071 ,  0.04903620108962059, -0.1782274693250656 ,
  -2.043890953063965  , -0.23703043162822723,  0.14665450155735016,
  -0.35057878494262695, -0.48061856627464294,  0.07349957525730133,
  -0.2835563123226166 , -0.13675938546657562, -0.18781910836696625,
  -0.05113660171627998, -0.7167670726776123 ,  0.26541271805763245,
   0.06095382943749428,  0.41476234793663025, -0.5537022352218628 ,
   0.07966116070747375,  0.03781545907258987, -0.05766494199633598,
  -0.2742196321487427 ,  0.33932387828826904, -0.17003828287124634,
  -0.05943262577056885, -0.43963566422462463,  0.07663626223802567,
  -0.5856502056121826 , -0.7343130111694336 ,  0.22564725577831268,
  -0.3442172408103943 , -0.08851948380470276, -2.9640283584594727 ,
  -0.29928058385849   ,  0.1650710105895996 , -0.3690069317817688 ,
  -0.4287218749523163 , -0.2211208939552307 , -1.995557188987732  ,
  -1.074721097946167  , -0.5298276543617249 ,  0.22186455130577087,
  -0.4229266345500946 , -0.2714330852031708 , -1.0204148292541504 ,
   0.11431533843278885,  0.06625872105360031, -0.3822648227214813 ,
   0.47264039516448975, -0.5739631652832031 , -0.3430121839046478 ,
  -0.5705605745315552 , -0.6393705010414124 , -0.4150192141532898 ,
  -0.7776839137077332 , -0.21823354065418243, -0.26445481181144714,
  -0.3751581907272339 , -1.2092448472976685 , -0.19561418890953064,
  -0.416970819234848  , -0.14754214882850647, -0.9779807329177856 ,
   0.4285367727279663 ],
 [ 0.25648823380470276, -0.6232929229736328 , -0.13798776268959045,
   0.16027973592281342, -0.01929119788110256,  0.49317485094070435,
   0.11189006268978119, -0.28420066833496094,  0.01016775239259005,
   0.11394879966974258, -0.501076877117157  , -0.2007785141468048 ,
   0.5650653839111328 ,  0.18848179280757904,  0.384807825088501  ,
   0.07643268257379532,  0.6479743123054504 , -0.4066239893436432 ,
  -0.5022727847099304 ,  0.04663224145770073,  0.0280940867960453 ,
   0.1834697425365448 ,  0.13914304971694946,  0.13631129264831543,
  -0.3989367187023163 , -0.712348461151123  , -0.1410973221063614 ,
   0.3384978473186493 , -0.42152005434036255, -0.7691705226898193 ,
   0.2719589173793793 ,  0.575257420539856  ,  0.07987039536237717,
   0.3004070520401001 , -0.5332192182540894 ,  0.06926288455724716,
  -0.03861149773001671, -0.7529977560043335 ,  0.25416961312294006,
   0.5220760107040405 , -0.15265029668807983,  0.04617080837488174,
  -0.10126317292451859,  0.15233519673347473,  0.1652148813009262 ,
   0.05258644372224808,  0.6094935536384583 , -0.4076008200645447 ,
  -0.21722395718097687,  0.19941531121730804, -0.09902425110340118,
  -0.07698288559913635,  0.21108561754226685,  0.4472852945327759 ,
   0.5158496499061584 , -0.24836905300617218, -0.31995704770088196,
  -0.2226881980895996 ,  0.05274363234639168, -0.13032370805740356,
  -0.06088832393288612,  0.17753057181835175,  0.01838391646742821,
  -0.39424505829811096],
 [-0.18772511184215546, -0.7083033323287964 ,  0.07527921348810196,
   0.5017381310462952 ,  0.7810294032096863 , -2.201373815536499  ,
   0.8605847358703613 , -0.32490119338035583, -0.39464911818504333,
  -1.5732063055038452 , -0.1627168506383896 ,  0.5993531942367554 ,
   0.3893028497695923 ,  0.3994426727294922 ,  0.2585511803627014 ,
   0.19296476244926453,  0.9355000853538513 ,  0.8760714530944824 ,
   1.0164027214050293 ,  0.6342534422874451 , -0.29171887040138245,
  -0.724920392036438  , -0.3066892623901367 ,  0.22152653336524963,
  -0.5416309833526611 ,  0.1933918297290802 , -0.423545241355896  ,
  -1.0280370712280273 , -0.23704850673675537, -0.40726831555366516,
  -1.0314642190933228 ,  1.173333764076233  ,  0.5894557237625122 ,
  -0.5639297366142273 , -0.24374622106552124, -0.5944408178329468 ,
  -0.7186155319213867 ,  0.2403222769498825 ,  0.4123004972934723 ,
  -1.7446527481079102 ,  0.21965597569942474,  0.18937967717647552,
   0.18755200505256653, -0.5975134968757629 , -0.35830914974212646,
  -0.22627955675125122,  0.833175778388977  ,  0.04178832843899727,
   0.32706087827682495, -2.5839226245880127 , -1.6229783296585083 ,
   0.09656145423650742,  0.00008960915147327, -1.2828948497772217 ,
  -1.4576705694198608 , -0.37066298723220825, -0.03320669010281563,
  -0.6244258284568787 , -0.1596783995628357 , -1.0155290365219116 ,
  -0.5994707345962524 ,  0.207396000623703  ,  0.07290229201316833,
  -0.729955792427063  ],
 [ 0.7115654349327087 ,  0.08528817445039749, -0.16290226578712463,
   0.34705817699432373,  0.2093563675880432 , -0.0604199692606926 ,
  -2.245077133178711  ,  0.284790575504303  ,  0.07302284985780716,
  -0.16829968988895416,  0.3694944977760315 ,  0.4870729446411133 ,
   0.5261784791946411 , -0.02746606059372425,  0.5593024492263794 ,
   1.4349853992462158 , -0.8343460559844971 ,  0.40512508153915405,
  -1.0253455638885498 ,  0.2410886287689209 ,  0.10628602653741837,
  -2.2084600925445557 , -2.948415756225586  ,  0.4765884578227997 ,
   0.4144837260246277 ,  0.5527988076210022 , -0.9164337515830994 ,
   0.4397732615470886 ,  0.34415552020072937, -0.8269374966621399 ,
   0.27455511689186096,  0.02777767181396484, -2.2810637950897217 ,
  -0.02888600900769234, -0.373501181602478  , -1.0255911350250244 ,
  -0.43200254440307617, -0.662855327129364  ,  0.5116491913795471 ,
  -1.798964500427246  ,  0.14472228288650513, -0.49059319496154785,
   0.00526929693296552,  0.35157960653305054,  0.786791205406189  ,
   0.756003201007843  ,  0.5990387797355652 , -1.2065937519073486 ,
  -0.2635977864265442 , -0.9758764505386353 ,  0.4162387549877167 ,
   0.49963730573654175,  0.4095340371131897 ,  1.345731496810913  ,
   0.21075685322284698,  0.39032623171806335,  0.3928655982017517 ,
   0.9389283657073975 , -0.06817959994077682, -0.4258483648300171 ,
  -1.169454574584961  ,  0.38752174377441406, -0.08704400807619095,
  -0.6597573757171631 ],
 [ 0.2373059093952179 ,  0.14655964076519012, -0.40760546922683716,
  -0.13452184200286865, -0.00156619097106159,  0.3301399350166321 ,
  -0.1493350863456726 ,  0.7994785904884338 ,  0.09735150635242462,
  -0.1728784441947937 ,  0.3638235032558441 ,  0.02249371446669102,
   0.08337510377168655,  0.09778919070959091,  0.13580870628356934,
  -0.22350619733333588,  0.10659278929233551, -0.16245122253894806,
   0.4555096924304962 ,  0.01132184080779552, -0.02673510648310184,
   0.15243487060070038, -0.3851679861545563 ,  0.14551079273223877,
   0.2229897826910019 ,  0.16833706200122833, -0.05893748253583908,
  -0.6578089594841003 ,  0.9343845248222351 ,  0.03566814586520195,
  -0.19859197735786438, -0.10672111064195633, -0.5119370818138123 ,
   0.517838180065155  ,  0.42157936096191406, -0.04463700205087662,
  -1.2111711502075195 , -0.8241873383522034 ,  1.033861517906189  ,
  -0.06301651895046234,  0.14753393828868866,  0.3028365671634674 ,
   0.20123396813869476, -0.03717879951000214, -1.2533799409866333 ,
  -0.9471566081047058 ,  0.03638507798314095, -1.0564850568771362 ,
   0.5894120931625366 , -0.00639994675293565, -0.2201049029827118 ,
  -1.0105717182159424 ,  0.2479579746723175 , -0.03046279773116112,
  -0.3380754888057709 ,  0.06805966794490814,  0.08917897194623947,
  -0.00898007769137621, -0.11445658653974533, -0.5019176006317139 ,
  -0.4359334409236908 ,  0.6386582851409912 ,  0.2751099765300751 ,
   0.08734682947397232],
 [-0.9807454347610474 ,  0.10132703185081482, -0.42947623133659363,
   0.023394875228405  , -0.5160219073295593 ,  0.47878870368003845,
   1.0057218074798584 ,  0.33159932494163513,  0.18940971791744232,
   0.6692098379135132 , -0.51736980676651   ,  1.0920883417129517 ,
  -0.7067639231681824 ,  0.0233220998197794 , -0.23504334688186646,
  -0.28267979621887207, -0.9648391604423523 , -0.7827109098434448 ,
  -1.6029261350631714 , -0.21203625202178955, -0.5208244919776917 ,
  -0.9658756852149963 , -1.3208798170089722 ,  0.2781490683555603 ,
  -0.3537731468677521 ,  0.37128350138664246,  0.7139708399772644 ,
   0.15209871530532837, -0.13288940489292145,  0.13372106850147247,
  -0.30000418424606323, -0.20150302350521088,  0.8829368948936462 ,
  -0.15454699099063873,  0.4655168354511261 , -0.7119084000587463 ,
   0.9987210035324097 ,  0.4363440275192261 , -4.072428226470947  ,
  -2.591392993927002  , -0.530741810798645  ,  0.5061262845993042 ,
  -0.01320422906428576, -1.1814053058624268 ,  0.7979349493980408 ,
   0.9217529892921448 , -1.170374870300293  , -1.0997633934020996 ,
   0.5270429849624634 , -0.7205629348754883 , -0.2119714319705963 ,
  -0.2168424129486084 ,  0.6131323575973511 ,  1.153552532196045  ,
  -1.7875820398330688 , -1.3378838300704956 ,  0.8054257035255432 ,
  -0.15171685814857483,  0.2038302719593048 , -0.30519574880599976,
   0.35684868693351746, -0.2511988580226898 ,  0.27729448676109314,
  -0.3400897681713104 ],
 [ 4.0592570304870605 , -0.6504161953926086 ,  0.503469705581665  ,
  -1.2148191928863525 , -0.6969251036643982 ,  0.6294747591018677 ,
   1.566004991531372  , -0.6513212323188782 , -0.1728852540254593 ,
  -0.6345146894454956 , -0.10526182502508163,  0.214713916182518  ,
  -2.3209187984466553 ,  1.2985022068023682 ,  0.7404775023460388 ,
   0.44828757643699646, -0.20420752465724945,  0.926960289478302  ,
   1.6034177541732788 ,  0.8204914331436157 ,  1.19625985622406   ,
  -0.8328595161437988 , -0.24383436143398285,  0.5018553733825684 ,
  -0.5089465379714966 , -0.5909838676452637 ,  0.08183585852384567,
  -0.7123990058898926 ,  0.6282309889793396 ,  1.2576425075531006 ,
   0.9042709469795227 ,  0.4724177122116089 , -0.439998984336853  ,
  -0.07622931897640228, -0.01381311751902103,  1.2607494592666626 ,
   0.12391998618841171,  0.142401784658432  ,  0.45810478925704956,
   1.928636908531189  , -1.1549949645996094 , -1.1536043882369995 ,
   0.12770389020442963,  1.4878450632095337 , -0.8043442368507385 ,
   0.5078762173652649 ,  0.45329222083091736,  0.4212658405303955 ,
   0.9869762659072876 ,  0.2956407964229584 ,  0.11622188985347748,
   0.5070569515228271 ,  1.2592109441757202 , -0.03344177827239037,
   1.4264460802078247 ,  1.1767735481262207 , -0.05016490444540977,
   0.12942682206630707,  0.2695387601852417 , -0.34661373496055603,
   0.3063141405582428 , -0.44879645109176636, -0.5347601175308228 ,
   0.7441492676734924 ],
 [ 0.03815685585141182, -0.534132182598114  ,  0.1722046136856079 ,
  -0.44613125920295715,  0.3026457726955414 , -1.0634770393371582 ,
   0.2821967601776123 ,  0.2302526831626892 ,  0.3084915578365326 ,
   0.29509782791137695, -0.10845211893320084, -0.6139254570007324 ,
   0.2316441833972931 ,  0.4456114172935486 , -1.1128863096237183 ,
  -0.6953229308128357 , -0.5890387296676636 ,  0.2063446193933487 ,
   0.8385841846466064 , -0.43018972873687744, -0.34580865502357483,
   0.5443800091743469 , -0.9496839642524719 , -0.08679747581481934,
   0.6357209086418152 , -0.06967861205339432,  0.26383525133132935,
   0.3729003071784973 ,  0.6212102770805359 , -0.06397875398397446,
  -0.01010861061513424, -0.3824106454849243 , -0.3471296727657318 ,
   0.07617194205522537, -0.15158042311668396, -0.26403847336769104,
  -0.9785712361335754 ,  0.4636635184288025 , -0.738712728023529  ,
   0.10371647775173187, -0.47758591175079346, -0.22453036904335022,
   0.12572582066059113, -0.00537893548607826, -1.2768285274505615 ,
  -0.296917587518692  , -2.6191627979278564 ,  0.03345941752195358,
  -0.2244817167520523 , -0.16691452264785767,  0.7436361908912659 ,
   0.3708410859107971 , -0.8562515377998352 , -0.29109981656074524,
  -0.7153546214103699 , -0.6466425657272339 ,  0.5852561593055725 ,
   0.1329672932624817 , -0.02407946810126305,  0.1713406890630722 ,
  -1.4495906829833984 ,  0.09946734458208084,  0.01621501706540585,
   0.5965016484260559 ],
 [ 1.470307469367981  ,  1.4783838987350464 ,  0.5737729668617249 ,
   1.1621376276016235 , -0.23113246262073517,  0.0508529432117939 ,
  -0.4716093838214874 ,  0.06357331573963165, -0.38902682065963745,
   0.8455461263656616 ,  1.4627997875213623 ,  0.00165962241590023,
  -0.04753359779715538,  0.510120153427124  ,  1.6808098554611206 ,
   1.4549241065979004 ,  1.1079007387161255 , -0.0349365659058094 ,
   1.8972141742706299 , -0.1431960016489029 , -0.15111857652664185,
   0.4293413460254669 ,  2.7827494144439697 ,  0.05694399401545525,
  -0.16360478103160858,  1.5848796367645264 , -0.01574677415192127,
   0.8713658452033997 , -0.02199415490031242, -0.0165703110396862 ,
   1.5092782974243164 ,  0.01232132781296968, -0.4023021161556244 ,
   2.5454347133636475 ,  0.71175616979599   ,  0.48817119002342224,
   2.4614357948303223 , -0.6619733572006226 ,  0.33871230483055115,
   1.21433687210083   ,  1.7549619674682617 , -0.6930292248725891 ,
   1.6127760410308838 ,  0.2817818224430084 ,  0.45034897327423096,
   0.7606256604194641 ,  0.22455692291259766,  1.2451082468032837 ,
   1.726363182067871  ,  0.3809620440006256 ,  0.7484614253044128 ,
  -0.19536955654621124,  1.5207419395446777 ,  0.44846922159194946,
   2.1334283351898193 ,  1.1155904531478882 ,  0.42434534430503845,
   0.5347961187362671 ,  0.09133081138134003, -0.7319517135620117 ,
   0.3129359483718872 , -0.49666792154312134,  1.0873816013336182 ,
  -1.1836811304092407 ],
 [ 0.01834332942962646,  0.07645131647586823, -0.28648489713668823,
   0.5846110582351685 ,  0.40324777364730835, -2.250988721847534  ,
  -0.10311910510063171,  0.04062160104513168, -0.24304617941379547,
  -0.02563572116196156,  0.5412771105766296 , -0.22923032939434052,
   0.5583544969558716 ,  0.28822505474090576, -0.3043516278266907 ,
   0.06517116725444794, -0.30851632356643677,  0.20945358276367188,
  -0.35600942373275757,  0.39393123984336853, -0.24427343904972076,
  -1.1730411052703857 ,  0.5553776025772095 ,  0.0941062942147255 ,
  -0.02485277876257896,  0.25254347920417786,  0.9886016249656677 ,
   0.8939101696014404 ,  0.5332863330841064 , -0.0224397499114275 ,
  -0.44608139991760254, -1.255324125289917  ,  0.09463971108198166,
   0.26299411058425903,  0.21827368438243866, -0.03058446943759918,
   0.5493305325508118 , -0.7729825377464294 ,  0.632611095905304  ,
  -0.16248640418052673,  0.6842305064201355 ,  0.38685256242752075,
  -0.15525349974632263, -0.6259817481040955 , -1.2141293287277222 ,
  -0.9399082660675049 , -1.7138729095458984 ,  0.24618661403656006,
  -0.04881152138113976, -0.38880908489227295,  0.16496622562408447,
  -0.6482640504837036 , -2.8603625297546387 ,  0.20065774023532867,
   0.915311872959137  , -1.8293704986572266 ,  0.17725180089473724,
  -0.30924850702285767, -0.2441469430923462 ,  0.466648668050766  ,
   0.01563361845910549,  0.6328957676887512 ,  0.47390928864479065,
  -0.3809567391872406 ],
 [-0.20068222284317017,  0.2928255498409271 ,  0.24468781054019928,
   0.01271237246692181,  0.3418966829776764 , -0.7449349164962769 ,
   0.2861851751804352 , -1.7267425060272217 , -0.2742149233818054 ,
  -0.731675922870636  ,  0.3119765520095825 ,  0.2897622883319855 ,
  -0.49600133299827576,  0.08643385022878647, -0.11900490522384644,
  -0.09634920954704285,  0.05469349771738052,  0.1970328986644745 ,
   0.5426750183105469 ,  0.08234958350658417, -0.25378182530403137,
   0.05623220652341843, -0.4511272609233856 , -0.21200719475746155,
  -0.07157011330127716, -0.2833839952945709 ,  0.2737705409526825 ,
  -0.35825157165527344, -0.20073798298835754,  0.2218567281961441 ,
   0.4607609510421753 , -0.3181869089603424 , -0.28281134366989136,
   0.1460365504026413 , -0.3572038412094116 , -0.06630679219961166,
   0.12395554035902023,  0.28889259696006775, -0.41293564438819885,
  -0.05870675295591354, -0.3665306270122528 ,  0.01297671720385551,
   0.03394557908177376,  0.02912116423249245,  0.33911657333374023,
  -0.33649224042892456, -0.31739622354507446,  0.6773955821990967 ,
  -0.4443393647670746 ,  0.35474979877471924,  0.12550120055675507,
  -0.37918370962142944, -0.1373215615749359 , -0.5598387718200684 ,
  -2.368725538253784  , -1.3910740613937378 , -0.464545875787735  ,
   0.19899789988994598,  0.07500538975000381,  0.15310673415660858,
  -0.0453997440636158 , -0.4381950795650482 , -0.397236704826355  ,
  -0.5150110125541687 ],
 [-0.0920623317360878 , -0.10484921932220459,  0.27114224433898926,
   0.4957607090473175 ,  0.16875413060188293, -1.259762167930603  ,
   0.35370340943336487, -0.27749326825141907, -0.03568709269165993,
  -1.5097280740737915 ,  0.7470653057098389 , -0.06451959908008575,
   0.15820130705833435,  0.37370043992996216, -0.6717737317085266 ,
   0.17424795031547546, -0.3837127089500427 ,  0.05709080398082733,
  -0.7909586429595947 ,  0.06094958633184433,  1.471099615097046  ,
  -0.16697949171066284, -0.17800267040729523,  0.4791663885116577 ,
   0.21359547972679138, -1.2505205869674683 , -0.15792429447174072,
   0.19669069349765778,  0.03822058811783791, -1.1162227392196655 ,
  -1.014673113822937  ,  0.8720771670341492 ,  0.6291742920875549 ,
   0.2886616289615631 , -1.2628051042556763 , -1.124550461769104  ,
   1.3991118669509888 , -0.04975380003452301, -0.21092981100082397,
   0.2358189821243286 ,  0.1049504354596138 , -0.3408146798610687 ,
  -0.12666110694408417, -0.2097788006067276 ,  0.4220792055130005 ,
  -0.3263241946697235 , -0.5389944314956665 ,  1.2181317806243896 ,
   0.7801456451416016 ,  0.00679043075069785, -0.25939446687698364,
  -0.12216275185346603, -2.1180944442749023 , -0.19392849504947662,
   0.834529459476471  ,  0.31610479950904846, -0.5758943557739258 ,
  -0.04019895195960999,  0.09681841731071472,  0.03039185702800751,
  -0.36671367287635803, -1.6148531436920166 ,  0.5764626860618591 ,
  -0.9183738827705383 ],
 [-0.6885706782341003 ,  0.34886255860328674, -0.5255413055419922 ,
  -0.4853041172027588 , -0.2500903308391571 ,  0.40340226888656616,
  -0.3833656907081604 , -0.7155835628509521 ,  0.26383957266807556,
  -1.2093467712402344 ,  0.4575137794017792 ,  0.4635547995567322 ,
   0.6829000115394592 , -0.06310774385929108,  1.0035086870193481 ,
  -0.4914563298225403 , -0.7899399995803833 , -0.6463474631309509 ,
  -0.506633996963501  , -0.384628564119339  ,  0.40410861372947693,
  -1.186741590499878  , -0.6334151029586792 ,  0.34112390875816345,
   0.3409074544906616 , -0.9054778814315796 , -0.12378279864788055,
  -0.2681097686290741 ,  0.5924849510192871 , -0.3137362003326416 ,
  -0.0214676633477211 ,  0.4808465540409088 ,  0.02867742255330086,
   0.07972193509340286, -1.4755784273147583 , -0.8545124530792236 ,
  -1.6316032409667969 ,  0.17753364145755768,  0.06798848509788513,
  -1.5547312498092651 , -0.786476731300354  ,  0.5238873362541199 ,
  -0.28582486510276794, -0.20031856000423431,  0.5107019543647766 ,
  -0.04844583570957184, -0.09276849776506424, -0.5494642853736877 ,
   0.18579958379268646,  0.6877871155738831 , -0.9440876245498657 ,
  -0.8513013124465942 , -0.46463698148727417, -0.03586991131305695,
  -0.5979306697845459 ,  0.04301473870873451, -0.20432734489440918,
  -0.3991236388683319 ,  0.05562279000878334, -0.8383051753044128 ,
  -0.01817528530955315,  0.7428168654441833 , -0.8776604533195496 ,
   0.34217318892478943],
 [-0.146081805229187  ,  0.38930267095565796,  0.34162575006484985,
  -0.37975168228149414,  0.05476244539022446,  0.2971917986869812 ,
  -0.0806654691696167 , -0.30218058824539185,  0.06344543397426605,
   0.13244743645191193, -0.16168248653411865, -0.2544495165348053 ,
  -0.3004506826400757 , -0.2022407352924347 ,  0.25458547472953796,
   0.12363988906145096, -0.7428892850875854 ,  0.21102523803710938,
   0.31781455874443054,  0.15035340189933777,  0.2986656725406647 ,
   0.3631848394870758 ,  0.22145989537239075, -0.25119122862815857,
   0.14853249490261078,  0.03317911550402641,  0.02327975071966648,
   0.4385809004306793 , -0.13349629938602448,  0.2325170636177063 ,
   0.1947024166584015 , -0.44397687911987305,  0.6563039422035217 ,
   0.12547075748443604,  0.15176276862621307, -1.0595505237579346 ,
  -0.16175825893878937,  0.4072817265987396 ,  0.7534384727478027 ,
  -0.2232794612646103 , -0.17563079297542572,  0.15727758407592773,
   0.23455959558486938, -0.45921170711517334,  0.3762510418891907 ,
  -0.42498859763145447,  0.6803149580955505 , -0.5717355012893677 ,
   0.4730587601661682 ,  0.3118762969970703 ,  0.6029130220413208 ,
  -0.29169073700904846, -1.1434499025344849 , -0.3801914155483246 ,
   0.6272169351577759 , -1.5703004598617554 ,  0.21928121149539948,
   0.43956518173217773, -0.0282437689602375 ,  0.12396080791950226,
  -0.15375511348247528,  1.1908025741577148 ,  0.7330131530761719 ,
  -0.1511656939983368 ],
 [-0.545135498046875  , -0.10399802029132843, -0.02961749956011772,
   0.03210549801588058,  0.03922712057828903, -0.43693020939826965,
  -1.2675448656082153 , -0.3026343882083893 , -0.00744140380993485,
  -0.5458014011383057 , -0.39599061012268066, -0.01456372067332268,
  -0.9073591232299805 , -0.3142225742340088 , -0.13773776590824127,
  -0.192940354347229  , -1.0188626050949097 , -0.29641449451446533,
   0.10435150563716888,  0.1989368349313736 , -0.10307367891073227,
  -0.7413026094436646 , -0.35576164722442627, -0.34436070919036865,
   0.23011071979999542,  0.1159282997250557 , -0.09170015901327133,
  -0.18083423376083374,  0.20360729098320007, -0.10361596941947937,
  -0.3139518201351166 , -0.2614574730396271 , -0.1832507997751236 ,
  -0.45618370175361633, -0.4344647228717804 ,  0.14071696996688843,
  -0.591643214225769  , -0.10040391981601715, -0.01608065515756607,
  -0.48000574111938477, -0.06134144589304924,  0.02845031768083572,
  -0.1039028912782669 , -0.05538026988506317, -0.09122177213430405,
   0.156196728348732  , -0.15427909791469574, -0.03888765722513199,
  -0.5554140210151672 , -0.2894705832004547 ,  0.0743207186460495 ,
  -0.01194804720580578, -0.18111076951026917, -0.6366026401519775 ,
  -1.2129864692687988 , -0.2184247523546219 , -0.5092190504074097 ,
   0.13652224838733673,  0.20176799595355988, -0.1020769402384758 ,
   0.01365397777408361, -0.02733589895069599,  0.09347622096538544,
   0.5254945755004883 ],
 [-0.29028987884521484, -0.3034951388835907 , -0.20530956983566284,
  -1.0341733694076538 ,  0.13157576322555542, -0.25546810030937195,
  -0.903991162776947  ,  0.06904706358909607,  0.22490331530570984,
  -0.4099246561527252 , -0.220481738448143  , -0.6806756854057312 ,
  -0.49207669496536255, -1.0446701049804688 , -0.3296669125556946 ,
   0.30341389775276184, -0.2745837867259979 , -0.9190970659255981 ,
  -0.4155403971672058 , -0.42817452549934387,  0.05924195796251297,
  -0.6520658135414124 , -0.5585529804229736 , -1.0292518138885498 ,
  -0.39517900347709656, -0.17562860250473022,  0.16943201422691345,
  -0.7579448223114014 , -0.512129545211792  , -0.6527279615402222 ,
  -0.2335834950208664 , -0.16035720705986023, -0.783820629119873  ,
  -1.225540280342102  ,  0.02750203385949135, -0.9131487011909485 ,
  -0.1172323226928711 ,  0.12361876666545868, -0.28746840357780457,
  -0.02451622113585472, -0.5048820376396179 ,  0.27106431126594543,
  -0.4398203492164612 , -0.18017350137233734, -0.3311178684234619 ,
   0.5463255047798157 , -2.30289363861084   ,  0.5631356239318848 ,
   0.3045852482318878 , -0.9038662910461426 , -0.3787427246570587 ,
  -0.07737808674573898, -0.32617974281311035, -0.6117057204246521 ,
  -0.5297272801399231 ,  0.3813263773918152 , -1.3922027349472046 ,
  -0.20978015661239624,  0.2136491984128952 ,  0.06089275702834129,
   0.00381910265423357, -0.12209612131118774,  0.13854850828647614,
   0.4301155209541321 ],
 [-0.21192198991775513, -0.4539726674556732 , -0.3920242190361023 ,
   0.1564631462097168 , -0.09041333943605423,  0.06592833995819092,
   0.3941822052001953 , -0.7465100884437561 , -0.00830931589007378,
  -1.1725480556488037 ,  0.1659661829471588 ,  0.20227673649787903,
   0.6350946426391602 , -1.904396653175354  , -0.20377375185489655,
  -0.08424483239650726, -0.5522061586380005 , -0.3575691282749176 ,
  -0.10391733795404434, -0.00808012951165438,  0.4638940691947937 ,
  -0.6071284413337708 , -0.34893476963043213,  0.3372155725955963 ,
  -0.3820171356201172 , -0.23264279961585999,  0.5031911134719849 ,
  -0.8168640732765198 ,  0.02498335763812065, -0.5030738711357117 ,
   0.6515680551528931 ,  0.169394388794899  ,  0.3207991421222687 ,
  -0.8202969431877136 ,  0.02033704333007336, -1.0009859800338745 ,
   0.3479063808917999 ,  0.22885574400424957,  0.2967863082885742 ,
   0.5501163005828857 ,  0.1807863563299179 ,  0.00104083132464439,
   0.16622686386108398, -0.21350301802158356, -0.28069940209388733,
   0.13144119083881378, -1.0063362121582031 , -2.0141611099243164 ,
   0.8388746976852417 ,  0.28324317932128906,  0.7371249794960022 ,
   0.7610054016113281 ,  0.00675302697345614,  0.588283360004425  ,
  -0.2044350504875183 , -0.590975821018219  , -1.6912776231765747 ,
  -0.7104014754295349 ,  0.2824435532093048 , -0.03163699060678482,
  -0.12315231561660767, -1.6900473833084106 , -0.8433607220649719 ,
  -0.0434039980173111 ],
 [ 0.29440802335739136, -0.4956923723220825 , -0.2215825617313385 ,
  -0.6294287443161011 , -0.0892564207315445 , -0.2997165024280548 ,
  -0.0511072613298893 , -1.0840295553207397 ,  0.47446125745773315,
  -0.24007214605808258, -0.10288245230913162, -0.537479043006897  ,
   0.1499914824962616 , -0.09430268406867981, -0.245260089635849  ,
  -0.13132399320602417,  0.48548567295074463, -0.069185771048069  ,
  -0.13559789955615997, -0.500095784664154  , -0.11768218874931335,
   0.00261693634092808,  0.0593424066901207 , -1.116456389427185  ,
  -0.09135645627975464, -0.283754825592041  ,  0.0702500194311142 ,
   0.37747111916542053, -0.35586369037628174, -0.2204674631357193 ,
  -0.09315016120672226, -0.6870117783546448 , -0.19367370009422302,
  -0.2697688341140747 , -1.2238320112228394 , -0.4692005217075348 ,
   0.5150025486946106 ,  0.17961937189102173, -0.226630300283432  ,
   0.09810376167297363, -0.1465916782617569 , -0.09483818709850311,
   0.47735661268234253, -0.1421148180961609 , -1.068678855895996  ,
  -0.04921006411314011, -0.4641900658607483 , -1.4773333072662354 ,
   0.0031452514231205 , -0.01750750653445721, -0.113960400223732  ,
  -0.16130505502223969, -0.14790423214435577, -0.11874457448720932,
   0.327914297580719  , -0.30408987402915955,  0.04059144109487534,
  -0.3676866292953491 , -0.46630504727363586,  0.2345868945121765 ,
  -0.35702452063560486,  0.05076318979263306,  0.3013826012611389 ,
   0.31724411249160767],
 [-0.4685073494911194 ,  0.4821353852748871 ,  0.04270191118121147,
  -0.12395736575126648, -0.5269449353218079 ,  0.8817545175552368 ,
  -0.7190179824829102 ,  0.3975256383419037 ,  0.0586547814309597 ,
  -0.3114587664604187 ,  0.24653320014476776, -0.7234851717948914 ,
  -0.6525081992149353 ,  0.16722337901592255, -0.25308725237846375,
  -0.18610887229442596,  0.31606146693229675, -0.4573225677013397 ,
   0.2210676521062851 , -0.1865282952785492 , -0.8966118693351746 ,
   0.21727880835533142, -0.4251372218132019 , -1.085638403892517  ,
  -0.6712136268615723 , -1.0950592756271362 , -0.5542523860931396 ,
   0.37122640013694763, -1.29987633228302   , -0.6225254535675049 ,
   0.34233778715133667, -0.0881224200129509 ,  0.5116693377494812 ,
   0.8003827929496765 ,  0.5751602053642273 ,  0.4007532298564911 ,
   0.8104040622711182 , -0.40442895889282227,  0.28770431876182556,
   1.0261603593826294 , -1.3142164945602417 , -0.12196303904056549,
  -0.09988044202327728,  0.08061377704143524,  0.17591659724712372,
  -0.7176213264465332 , -0.09143934398889542, -0.27435988187789917,
  -2.133639097213745  , -0.16574813425540924,  0.4866199493408203 ,
   0.15556223690509796,  0.10594220459461212, -1.6852601766586304 ,
   0.07771049439907074, -0.6193535327911377 ,  0.30755722522735596,
  -0.18832573294639587,  0.1176588162779808 , -0.7515808939933777 ,
   0.8814690709114075 ,  0.4249970018863678 , -0.9583552479743958 ,
   0.2406778633594513 ],
 [ 2.781177520751953  ,  0.3050532341003418 ,  0.24057859182357788,
   0.11921241879463196,  0.33340927958488464, -1.0944068431854248 ,
  -0.11582885682582855,  0.17157511413097382, -0.5640394687652588 ,
  -1.5518780946731567 , -0.2431049793958664 ,  0.40311625599861145,
  -0.21674558520317078, -0.7566776275634766 ,  0.2409122884273529 ,
   0.7643455862998962 , -0.17256303131580353, -1.531886339187622  ,
   0.5799804925918579 , -0.6775545477867126 ,  0.17004454135894775,
  -0.2927362322807312 ,  0.5496267676353455 , -0.1232115626335144 ,
   0.06270261853933334, -0.09152545034885406,  0.00367551832459867,
  -0.8538939952850342 ,  1.4237688779830933 , -0.1964469999074936 ,
  -0.45426782965660095,  0.2103693038225174 ,  0.6886463761329651 ,
   0.25894588232040405, -2.270988702774048  , -0.3074567914009094 ,
  -0.5056995153427124 , -0.2780393362045288 ,  0.2502689063549042 ,
  -0.03519063070416451, -0.14000463485717773,  0.5244925618171692 ,
   0.06752527505159378, -0.25265711545944214,  0.25852617621421814,
  -0.15715506672859192, -0.7241170406341553 , -0.3161168694496155 ,
   0.5736097693443298 ,  0.22410359978675842,  0.20898646116256714,
  -0.8749092221260071 , -1.6638554334640503 ,  0.09222716093063354,
   0.34825000166893005, -0.05645926669239998, -0.3912077844142914 ,
   0.4306859076023102 ,  0.12900279462337494, -0.416870653629303  ,
   0.19173118472099304, -0.3289696276187897 ,  0.6961691975593567 ,
   0.8314333558082581 ],
 [ 0.7823880910873413 , -1.1553922891616821 ,  0.3616504967212677 ,
   0.5825009346008301 ,  0.01808074489235878, -2.2287585735321045 ,
  -0.02405108511447906, -0.02740142680704594, -0.09849908202886581,
   0.5061622858047485 ,  0.17964619398117065, -0.41214051842689514,
  -0.4043027460575104 ,  0.3708966374397278 , -0.9313469529151917 ,
   0.4297773241996765 ,  0.5707064270973206 ,  0.9373719692230225 ,
  -2.366189956665039  , -0.05783406272530556, -0.8763824105262756 ,
  -0.28094184398651123, -0.4124988913536072 , -0.078019879758358  ,
   1.693829894065857  ,  0.314468115568161  ,  0.7643828392028809 ,
   0.8789985179901123 ,  0.6508974432945251 , -0.1836589127779007 ,
   0.8302695155143738 , -0.22676612436771393,  0.523154079914093  ,
   0.6041320562362671 ,  0.4469098448753357 ,  1.146849513053894  ,
  -0.37903153896331787, -0.22225438058376312, -1.5741982460021973 ,
  -1.0363967418670654 ,  2.542983055114746  , -0.10954130440950394,
  -0.9594473838806152 ,  0.02907545678317547,  1.1749393939971924 ,
  -1.737933874130249  ,  1.287564754486084  , -1.000415563583374  ,
  -0.3083845376968384 , -4.523988246917725  ,  1.6058508157730103 ,
   0.20436546206474304,  1.0075476169586182 ,  1.601501226425171  ,
   0.5989821553230286 ,  0.8920886516571045 , -0.1421336978673935 ,
   0.21551333367824554, -0.3543800115585327 ,  0.563147783279419  ,
   0.49868184328079224,  0.5208208560943604 ,  1.44476318359375   ,
  -0.6177101731300354 ],
 [-0.40262314677238464, -0.46255043148994446,  0.3198828399181366 ,
  -1.2070555686950684 , -0.0744422972202301 ,  0.43091025948524475,
  -1.7204550504684448 , -0.41792914271354675,  0.02698863670229912,
   0.36126241087913513, -0.32084864377975464, -1.7721099853515625 ,
  -1.724502444267273  ,  0.5693358778953552 , -1.394672155380249  ,
  -1.4871751070022583 , -2.45993709564209   , -1.4122525453567505 ,
  -0.14854054152965546, -0.2887837588787079 , -0.07354112714529037,
   0.07622096687555313, -1.0305159091949463 , -0.35418739914894104,
   0.840452253818512  , -0.16875362396240234,  0.17153474688529968,
  -0.24690468609333038, -0.39562928676605225,  0.1453009843826294 ,
  -0.28327926993370056, -3.2383031845092773 , -1.0503872632980347 ,
  -0.819917619228363  , -1.974738597869873  ,  0.1253017783164978 ,
  -0.01626011542975903, -0.9397351145744324 ,  1.103538990020752  ,
  -0.8058978319168091 , -0.45802465081214905, -0.1675562858581543 ,
  -1.0116344690322876 ,  0.47855818271636963, -1.0380027294158936 ,
   0.37472134828567505,  0.2751932442188263 , -1.3625988960266113 ,
  -1.3865361213684082 ,  0.09293857216835022,  0.16376180946826935,
  -0.27229273319244385, -0.4216160774230957 , -0.24173183739185333,
  -0.16946107149124146, -2.168771743774414  , -0.5727956295013428 ,
   0.259148508310318  , -0.1189582422375679 ,  0.18857406079769135,
   0.20512691140174866,  0.3525945842266083 ,  0.4944731593132019 ,
   0.5198140740394592 ],
 [ 0.27878791093826294, -0.3138788044452667 ,  0.2934390604496002 ,
  -1.050942063331604  ,  0.00962155498564243, -1.1021469831466675 ,
   0.12328583002090454,  0.3011779487133026 , -0.16283190250396729,
   0.16934911906719208,  0.1940893530845642 , -0.8826671242713928 ,
  -1.7341593503952026 ,  0.3530677258968353 ,  0.16017259657382965,
   0.6986580491065979 , -0.18997375667095184, -1.0080887079238892 ,
   0.7140359282493591 ,  0.3516978919506073 ,  0.2460888922214508 ,
  -0.49538370966911316,  0.1451716125011444 , -0.00921106804162264,
   0.2593516707420349 ,  0.3216302990913391 , -1.3806805610656738 ,
  -0.22268818318843842,  0.2552269697189331 , -0.03138207644224167,
   0.10270422697067261, -1.4602558612823486 , -1.5238099098205566 ,
  -2.5070855617523193 ,  0.1625337451696396 ,  0.29420584440231323,
   0.6676404476165771 , -1.2574737071990967 , -0.48732680082321167,
  -1.1825019121170044 ,  0.22276826202869415, -0.00882078800350428,
   0.5225223302841187 ,  0.30022886395454407,  0.7912914752960205 ,
  -0.07030470669269562, -0.7975848317146301 , -0.06680378317832947,
   0.03015722334384918, -0.418283075094223  , -0.6760023832321167 ,
   0.3679792284965515 ,  0.23372924327850342, -0.5156452655792236 ,
  -0.16061203181743622,  0.25284186005592346,  0.03719528764486313,
  -0.20504139363765717,  0.7972081303596497 , -0.6761908531188965 ,
   0.15182408690452576,  0.4116051197052002 ,  0.2544713318347931 ,
  -0.6404392719268799 ],
 [ 0.09642861783504486,  0.22840166091918945, -0.2919921576976776 ,
  -0.25324106216430664, -0.54946368932724   ,  0.2802199423313141 ,
  -0.09090016782283783, -0.2876778841018677 , -0.02576730772852898,
  -0.3119297921657562 ,  0.1361258625984192 ,  1.7075653076171875 ,
  -0.9585530161857605 , -1.0711292028427124 ,  0.12511500716209412,
   0.0734269917011261 ,  0.3578226566314697 , -2.6191201210021973 ,
  -0.7641968727111816 , -0.6045028567314148 ,  0.23816163837909698,
  -2.8196494579315186 , -1.97035813331604   ,  0.583185076713562  ,
  -0.01936010085046291,  0.44676345586776733,  0.43145912885665894,
   0.24953745305538177, -0.42044129967689514, -1.7119550704956055 ,
   0.19058442115783691,  0.2989942729473114 , -0.2171221524477005 ,
   0.24341002106666565, -0.05738608539104462,  0.6269134879112244 ,
  -1.4191112518310547 , -2.0215585231781006 ,  0.6855823397636414 ,
   0.852226734161377  , -0.8709388971328735 ,  0.14061503112316132,
  -0.39944666624069214, -0.1974426656961441 , -0.6042600274085999 ,
   0.20837415754795074,  0.5048890709877014 , -0.6272316575050354 ,
  -1.3045566082000732 , -2.885683536529541  ,  1.1085864305496216 ,
  -0.8141247034072876 ,  0.6381726861000061 , -0.7414359450340271 ,
   0.7324456572532654 ,  0.19209344685077667,  0.2861093580722809 ,
   0.2563028335571289 , -0.15242353081703186,  0.08783125877380371,
   0.03563139215111732, -1.5025345087051392 , -0.49201586842536926,
   0.423402339220047  ],
 [-0.15255267918109894, -0.00703044608235359, -0.11568208783864975,
   0.44552236795425415, -0.18624518811702728, -0.3287404179573059 ,
   0.383224755525589  , -0.04999339953064919,  0.10193674266338348,
  -0.7226004004478455 , -1.060200810432434  , -0.43160688877105713,
   0.14693307876586914,  0.10603703558444977, -0.8388230800628662 ,
   0.03432594612240791, -0.9334554076194763 ,  0.16794641315937042,
  -0.33100876212120056, -0.1918693482875824 ,  0.03435537964105606,
  -1.47164785861969   , -0.7838969826698303 , -0.00401713233441114,
   0.16480444371700287,  0.10377246886491776,  0.4019438624382019 ,
  -0.07436352968215942, -0.2625747621059418 ,  0.0536101870238781 ,
  -0.20173050463199615,  0.19305597245693207, -1.7047120332717896 ,
   0.12801216542720795,  1.721700668334961  , -0.04138936847448349,
   0.5269138813018799 , -0.5970767140388489 , -0.8964548707008362 ,
  -1.807598352432251  , -0.41342175006866455,  0.1856611669063568 ,
   0.24848882853984833, -0.11927410215139389, -0.07221163809299469,
  -0.0630088672041893 , -1.5390466451644897 ,  0.10593955963850021,
  -0.48897987604141235, -1.2233844995498657 ,  0.5646792054176331 ,
   0.09267421811819077, -0.5057063698768616 , -1.6425458192825317 ,
  -0.33472418785095215,  0.14320561289787292, -0.22405587136745453,
  -0.14333243668079376,  0.4200078845024109 ,  0.04462366178631783,
  -0.00056212034542114, -0.03806999325752258, -0.66189044713974   ,
   0.20107027888298035],
 [-0.4139021039009094 , -0.3566058874130249 , -0.2079816609621048 ,
  -0.2830176055431366 ,  0.8918924927711487 , -0.3457440435886383 ,
  -0.5615229606628418 ,  0.18420740962028503, -0.17664796113967896,
   0.6524078249931335 , -1.4555119276046753 ,  0.5357633829116821 ,
  -0.32934868335723877, -0.6729066967964172 ,  0.6477597951889038 ,
   0.0608224980533123 ,  0.16401104629039764,  0.2908659875392914 ,
  -0.221918985247612  ,  0.1587522029876709 , -1.016371250152588  ,
  -0.9889113903045654 , -0.32668256759643555,  0.3890390992164612 ,
  -0.1881798654794693 , -1.1792529821395874 ,  0.13300171494483948,
  -0.21118712425231934, -0.18765833973884583, -0.48761773109436035,
  -0.00440701516345143, -1.0554466247558594 , -1.5437520742416382 ,
  -0.9185186624526978 , -0.6201081871986389 , -0.6374635100364685 ,
  -1.4829325675964355 ,  0.2856139838695526 ,  0.25638777017593384,
  -0.25534459948539734, -0.8566878437995911 ,  0.0757119357585907 ,
  -0.6850540637969971 , -0.6836621165275574 , -0.591528594493866  ,
  -0.9703138470649719 , -0.44129911065101624,  0.8089206218719482 ,
  -0.21504947543144226, -0.9078056216239929 , -0.4174399673938751 ,
  -0.557581901550293  , -0.503516674041748  , -0.0494091585278511 ,
  -0.782228410243988  , -0.49640288949012756, -0.4898136556148529 ,
   0.26761317253112793,  0.4286329746246338 , -0.3108806908130646 ,
   0.810344934463501  , -1.8090474605560303 , -0.8288258910179138 ,
  -0.5213822722434998 ],
 [ 1.0887750387191772 , -0.3629724085330963 ,  0.6951030492782593 ,
   2.1445064544677734 , -0.00576340314000845, -1.0027780532836914 ,
   1.6446571350097656 , -0.09985876828432083,  0.03992778807878494,
  -0.8163486123085022 ,  0.7148069739341736 ,  1.1260374784469604 ,
  -0.42005184292793274, -0.7615569829940796 , -0.3498225510120392 ,
   1.2217985391616821 ,  1.5895700454711914 ,  1.3794821500778198 ,
   0.6170082688331604 ,  0.3917323648929596 ,  0.451255202293396  ,
  -0.02902585826814175,  0.3237014412879944 ,  0.3948412835597992 ,
   0.6554554104804993 ,  0.45627325773239136,  0.37341076135635376,
   1.4032073020935059 , -0.8005642890930176 ,  0.18982157111167908,
   0.7752856612205505 ,  0.12807106971740723,  0.01696627028286457,
  -0.6353158354759216 , -1.9010272026062012 ,  0.5320630669593811 ,
   1.1172780990600586 ,  0.35776013135910034, -0.22602897882461548,
  -1.5017845630645752 , -1.6548901796340942 , -0.04864387586712837,
   1.2988835573196411 ,  0.2981676459312439 , -0.36891427636146545,
  -1.4181289672851562 ,  0.08807165175676346,  0.9700201749801636 ,
  -0.22980737686157227, -0.9236619472503662 ,  0.5733397006988525 ,
   0.955081582069397  , -0.49899527430534363, -0.24184785783290863,
   1.2618857622146606 ,  0.5471828579902649 ,  0.1824549287557602 ,
   0.6315551400184631 , -0.05457451567053795,  0.4143327474594116 ,
   1.1691458225250244 ,  0.23787571489810944,  0.7198898792266846 ,
  -0.6308466196060181 ],
 [ 0.3103336989879608 , -0.05088784173130989,  0.4123390018939972 ,
  -0.01680961064994335,  0.03186961263418198,  0.40396860241889954,
   0.5656399130821228 ,  0.2638338804244995 ,  0.35998091101646423,
  -0.46033981442451477, -0.5549696683883667 ,  0.10974837094545364,
  -1.629471778869629  ,  0.38270103931427   , -0.6737070083618164 ,
   0.9846897125244141 ,  0.12512952089309692,  0.5801485180854797 ,
  -0.17141087353229523, -0.712289571762085  , -0.6667126417160034 ,
   0.00248583778738976, -0.5225961208343506 ,  0.4050648510456085 ,
   0.00423436099663377,  0.30190300941467285,  0.28706276416778564,
   0.35675594210624695, -0.24119074642658234, -0.2684255838394165 ,
  -0.0371384397149086 ,  0.08308663964271545, -0.32524535059928894,
   0.8704124689102173 ,  0.16895228624343872, -0.6345595717430115 ,
   0.18641768395900726, -0.7135487794876099 ,  0.10293255746364594,
  -0.4907642900943756 , -0.44234368205070496,  0.03798519447445869,
  -0.2503947615623474 , -0.12060975283384323, -0.3893938958644867 ,
   0.4694058895111084 ,  0.17362353205680847, -1.4160152673721313 ,
  -0.2216682881116867 , -0.20259298384189606, -1.033379077911377  ,
  -0.06676560640335083,  0.4234561622142792 , -1.4140307903289795 ,
  -0.7509045004844666 , -0.7945987582206726 ,  0.1631297618150711 ,
  -0.10142318159341812,  0.46323591470718384,  0.19788311421871185,
  -0.6375614404678345 ,  0.3884773552417755 ,  0.6412885189056396 ,
   0.04966982081532478],
 [ 0.3089759349822998 ,  0.28640496730804443,  0.11472441256046295,
   0.8842822313308716 ,  0.06691881269216537,  1.3264414072036743 ,
   0.3649444580078125 ,  1.6667912006378174 ,  0.07072041928768158,
   1.7910332679748535 , -0.6000605821609497 ,  0.1827782541513443 ,
  -0.4302980303764343 ,  0.3194323182106018 , -0.17812085151672363,
   0.01939031668007374,  1.664389967918396  ,  0.6302948594093323 ,
  -0.33046698570251465,  0.03907980397343636, -0.25608912110328674,
   0.3251608908176422 , -0.30587831139564514,  0.00397943938151002,
   0.19437943398952484, -0.49692922830581665, -0.0772605687379837 ,
   1.2039828300476074 , -0.5355823040008545 , -0.145033597946167  ,
   1.2883132696151733 , -0.3060417175292969 ,  1.0921207666397095 ,
  -0.3071884214878082 , -0.41661006212234497,  0.964012622833252  ,
   0.8262413144111633 ,  0.42354750633239746,  0.908885657787323  ,
   0.36317649483680725,  0.6211410164833069 , -0.11405279487371445,
  -0.6972464323043823 , -0.32407525181770325,  0.625821590423584  ,
  -0.03797875344753265, -0.00453205360099673, -0.2178889662027359 ,
  -0.03149536997079849, -0.3375558853149414 ,  1.564212679862976  ,
   0.02391104586422443, -0.13183265924453735,  0.12935850024223328,
   2.297849655151367  ,  0.06874784082174301, -0.14153070747852325,
   0.03389707580208778,  0.12308700382709503, -0.48905760049819946,
  -0.7956745028495789 , -1.4935157299041748 ,  0.0176058504730463 ,
   0.04085963219404221],
 [ 0.19110752642154694,  0.2663211226463318 , -0.07852643728256226,
  -0.07702698558568954,  0.12856769561767578,  0.46014806628227234,
  -0.1305866688489914 ,  0.4861883521080017 , -0.05193896591663361,
  -1.3322573900222778 ,  0.6143881678581238 ,  0.03521471098065376,
   0.9609758853912354 , -0.144897922873497  ,  0.02286947891116142,
  -0.03120252303779125, -0.7323216795921326 , -2.392019748687744  ,
  -0.01905471459031105,  0.10015852004289627, -0.4668509066104889 ,
  -0.17976965010166168,  0.23301668465137482, -0.06777314096689224,
  -0.34831756353378296, -0.1649237424135208 , -0.22540044784545898,
  -0.1357211172580719 ,  0.07478619366884232, -2.3496224880218506 ,
   0.02992919459939003, -0.6857597231864929 , -2.5086240768432617 ,
  -0.6088294386863708 , -0.47850432991981506, -1.2247252464294434 ,
   0.2941837012767792 , -0.09545651078224182,  0.2630865275859833 ,
  -0.6445692181587219 , -0.03561704605817795, -0.00709794880822301,
  -0.23933684825897217, -0.12401236593723297,  0.01247777417302132,
  -0.45885246992111206,  0.5459801554679871 , -0.42824727296829224,
  -0.5178275108337402 ,  0.09990132600069046, -0.9165870547294617 ,
  -0.3040218949317932 ,  0.14873702824115753,  0.19621899724006653,
   0.7448837161064148 , -0.28154611587524414,  0.2037804126739502 ,
   0.04973592609167099,  0.0822891891002655 , -0.01457231398671865,
  -0.16881409287452698, -2.1208853721618652 ,  0.0951593816280365 ,
  -0.2447470873594284 ],
 [-0.04097355902194977, -0.633133053779602  , -0.7525269389152527 ,
  -0.10285691916942596, -0.02162928692996502,  0.29315775632858276,
   0.2984623908996582 ,  0.8119403123855591 , -0.09565228223800659,
   0.2389029860496521 ,  0.5615348815917969 ,  0.5866236686706543 ,
  -1.737971544265747  ,  0.06920946389436722, -0.03915632888674736,
  -0.08552814275026321, -0.6464450359344482 , -0.6376160383224487 ,
  -0.01894661970436573,  0.22404783964157104,  0.02456174790859222,
  -1.4418230056762695 ,  1.0299934148788452 ,  0.4589473605155945 ,
  -0.09541948139667511, -0.5689955353736877 , -0.09835892170667648,
   1.3825476169586182 ,  1.2138594388961792 ,  0.1622994989156723 ,
   0.612360417842865  , -0.37344035506248474,  0.5871685147285461 ,
   1.0723545551300049 , -0.5881192088127136 , -0.9405238032341003 ,
  -1.1263335943222046 , -0.07169722020626068, -0.02552877739071846,
  -0.6045829057693481 ,  0.6643744111061096 ,  0.09507396817207336,
   1.4905914068222046 ,  0.16023433208465576, -0.7807936668395996 ,
   0.08229615539312363, -0.37099796533584595, -1.0611109733581543 ,
   0.2795445919036865 , -0.31158462166786194,  0.03233131393790245,
   0.473271906375885  ,  0.4155307114124298 ,  0.6494714021682739 ,
   1.2780553102493286 ,  0.3238126337528229 , -0.8683363795280457 ,
  -0.03866122663021088, -0.39447522163391113, -1.1989920139312744 ,
  -0.150918111205101  , -0.8808618783950806 ,  0.8387675285339355 ,
  -0.23775629699230194],
 [ 2.7829978466033936 ,  0.20767544209957123,  1.1052969694137573 ,
   0.05378398671746254,  0.5050560832023621 ,  0.34982243180274963,
  -0.4207313358783722 ,  0.8080489635467529 , -0.829323947429657  ,
   3.187839984893799  , -0.19389434158802032,  1.2895705699920654 ,
  -0.37857380509376526, -0.3726295828819275 ,  0.45891544222831726,
   0.2707862854003906 ,  0.9362984895706177 ,  0.6070129871368408 ,
   0.6706902980804443 , -0.3760579824447632 , -0.2934749126434326 ,
   0.4827936291694641 , -0.7413132190704346 , -0.6084920167922974 ,
  -1.11317777633667   , -0.5492395758628845 , -0.4811265170574188 ,
   3.1731293201446533 ,  1.1744874715805054 , -0.01768489554524422,
  -0.2694566249847412 , -0.37292367219924927,  0.6933837532997131 ,
  -2.1616876125335693 , -1.2767378091812134 ,  0.05673547089099884,
  -0.23218263685703278, -0.00873450934886932,  3.012465476989746  ,
  -0.5250612497329712 ,  0.12422224134206772, -1.1112619638442993 ,
   2.426231861114502  , -0.18520745635032654, -0.6116294264793396 ,
   0.49331143498420715, -1.3476784229278564 ,  0.9601474404335022 ,
   0.24171675741672516, -1.1507601737976074 , -0.08118084818124771,
  -0.41475197672843933,  0.275018572807312  , -0.763027548789978  ,
   0.18013346195220947,  0.8132191896438599 , -0.15526878833770752,
   0.6629467010498047 ,  1.152391791343689  , -1.4831877946853638 ,
   0.2572678327560425 , -0.5679073929786682 ,  0.24961784482002258,
  -1.4086525440216064 ],
 [-0.03087419830262661, -0.45833301544189453, -0.6052423119544983 ,
   0.8000544309616089 , -0.5313340425491333 ,  0.33097171783447266,
   0.12149990350008011,  2.6265153884887695 ,  0.1772676706314087 ,
  -1.3579246997833252 , -0.19207875430583954,  0.6342974305152893 ,
  -0.06388897448778152,  0.2638823986053467 ,  0.6802964806556702 ,
  -0.9324365258216858 , -0.5078151226043701 , -0.5828269124031067 ,
  -3.3935389518737793 , -0.7330236434936523 , -0.2730632424354553 ,
  -0.7590827941894531 ,  0.2588562071323395 ,  0.8156114816665649 ,
  -1.5681450366973877 , -1.2839446067810059 , -0.02288686670362949,
   0.4107300043106079 , -0.6386734247207642 ,  0.3319205641746521 ,
  -0.5797747373580933 ,  0.8920410871505737 ,  0.3787306845188141 ,
  -1.8253012895584106 , -0.5721961259841919 ,  0.20855142176151276,
   0.4213595390319824 ,  0.21781814098358154, -0.5703439712524414 ,
  -2.6332995891571045 ,  0.47211945056915283, -0.03174976631999016,
  -0.28132352232933044,  0.18863211572170258,  1.1214109659194946 ,
  -1.355520248413086  , -0.45502734184265137,  1.0854072570800781 ,
  -0.32896825671195984, -0.09982699900865555, -0.38651496171951294,
  -0.3487609624862671 ,  0.08292873948812485,  0.09542713314294815,
  -0.42075663805007935, -0.1241590678691864 ,  0.4946907162666321 ,
   0.07059445977210999, -0.34283655881881714, -0.0510200671851635 ,
  -0.34534212946891785,  0.918555736541748  ,  1.4818426370620728 ,
   0.45621463656425476],
 [ 0.21946848928928375, -1.0318059921264648 , -0.14970436692237854,
  -0.22452953457832336,  0.5861570239067078 , -0.5847113728523254 ,
   0.04859468340873718, -0.24566325545310974, -0.4312101900577545 ,
   0.23595115542411804, -1.0310964584350586 ,  0.26318633556365967,
  -0.45410341024398804, -0.1252201348543167 , -0.10425202548503876,
   0.01580852828919888,  0.1684955656528473 ,  0.07961802184581757,
  -0.02326587773859501,  0.09807339310646057, -0.1787489950656891 ,
  -0.42374688386917114,  0.42228999733924866, -0.03312597423791885,
  -0.1363796591758728 , -0.29329317808151245,  0.08224859088659286,
   0.3224596083164215 ,  0.44153445959091187, -0.1149521991610527 ,
   0.3127453029155731 , -0.7192842364311218 ,  0.28539004921913147,
  -0.16018971800804138,  0.6543287634849548 , -0.27710017561912537,
  -0.6026027798652649 , -0.47717729210853577, -0.2117280811071396 ,
  -0.28068283200263977, -0.1467035859823227 , -0.03572853282094002,
   0.03000914677977562, -0.3439692556858063 , -0.04335499927401543,
   0.3351109027862549 , -0.5946840047836304 ,  0.674049437046051  ,
  -1.5967656373977661 , -0.2335590273141861 ,  0.10189531743526459,
  -0.1457834541797638 , -0.2713162899017334 , -0.01754592731595039,
  -0.2078324258327484 , -0.32786375284194946, -0.1392824500799179 ,
   0.19576908648014069, -0.2196391224861145 , -0.26605531573295593,
  -0.24975265562534332,  0.3836359679698944 ,  0.6226434111595154 ,
   0.03861270099878311],
 [ 0.1597745567560196 , -0.22136420011520386,  0.14573539793491364,
  -0.3408971130847931 ,  0.26642826199531555, -0.08399082720279694,
  -1.4434902667999268 ,  0.17812557518482208, -0.08576183766126633,
  -0.8400475382804871 , -0.3274383246898651 ,  0.62007075548172   ,
  -1.7611490488052368 , -0.3756810128688812 , -1.5034714937210083 ,
  -1.2256991863250732 ,  1.2240341901779175 ,  0.4129979908466339 ,
  -0.8243813514709473 , -0.4879734516143799 ,  0.6350735425949097 ,
  -3.786304235458374  , -1.4781079292297363 ,  0.65658038854599   ,
   0.447559654712677  , -0.5491949915885925 , -0.13644134998321533,
   0.2582240402698517 , -0.00094008888117969, -0.1048896536231041 ,
   0.13638773560523987, -0.9343611598014832 ,  0.3067348003387451 ,
  -0.3771473169326782 , -1.8827095031738281 , -0.39715775847435   ,
   0.46038681268692017, -2.2464311122894287 , -0.2827828824520111 ,
  -1.336559772491455  , -0.8818119764328003 , -0.19090160727500916,
   0.06664342433214188, -0.48874348402023315, -1.2246572971343994 ,
  -0.3128551244735718 ,  0.3846111297607422 ,  0.4601493775844574 ,
  -2.6502010822296143 , -1.4157190322875977 , -1.2428464889526367 ,
  -0.24679265916347504, -0.13091839849948883,  0.8035843372344971 ,
   0.17703551054000854, -3.666877269744873  ,  0.5761368274688721 ,
   0.03683564811944962, -0.06859948486089706,  0.09403295069932938,
  -0.9153431057929993 , -2.7039072513580322 , -0.3349578082561493 ,
   0.10739804059267044],
 [ 2.1682300567626953 ,  0.3044697344303131 ,  0.5589684247970581 ,
  -0.5223549008369446 , -0.5595357418060303 ,  1.1433467864990234 ,
  -0.44502440094947815,  0.8628536462783813 , -0.774297833442688  ,
   2.6963655948638916 ,  0.42488428950309753,  0.7957152128219604 ,
   0.081917904317379  , -0.12958243489265442,  0.07170064747333527,
   0.5962303280830383 ,  0.01523090619593859,  0.5042729377746582 ,
  -0.1007266640663147 ,  0.8955944776535034 ,  0.17999207973480225,
   0.711181104183197  ,  0.18479536473751068,  0.48055875301361084,
  -0.31741106510162354, -0.04688580706715584,  0.4088858366012573 ,
   1.9710108041763306 , -0.8840178847312927 ,  0.7688413262367249 ,
   0.6054909229278564 ,  0.5171654224395752 ,  0.8152663111686707 ,
  -1.447739839553833  ,  0.874416172504425  ,  1.03713858127594   ,
   1.8416504859924316 , -0.4763607084751129 ,  1.4867630004882812 ,
   2.1650922298431396 ,  0.5739699006080627 , -0.7670002579689026 ,
   1.601406455039978  ,  0.7535644173622131 ,  1.2675827741622925 ,
   0.38138115406036377,  1.4461749792099    , -1.2308275699615479 ,
  -0.6126517057418823 , -0.776931643486023  ,  0.05555106699466705,
   0.01602886244654655, -0.07300559431314468,  0.6102234721183777 ,
   1.536812663078308  ,  1.737241506576538  , -0.5946959853172302 ,
   0.6401199698448181 ,  0.05624515563249588,  0.2260802984237671 ,
   0.8769518733024597 , -1.1019319295883179 , -0.5303612351417542 ,
  -1.002941608428955  ],
 [ 0.23099248111248016,  0.04522297531366348, -0.19514785706996918,
  -1.0212019681930542 , -0.13688182830810547,  0.204701766371727  ,
   0.2299293577671051 ,  0.33301207423210144,  0.20764195919036865,
   0.301272988319397  , -0.8219977617263794 ,  0.23485925793647766,
   0.02361619099974632, -0.2834133505821228 ,  0.5799679756164551 ,
   0.9813331365585327 ,  0.5925516486167908 , -0.128781259059906  ,
  -1.4051100015640259 , -0.22277188301086426,  0.09242010116577148,
   0.0955011174082756 , -0.6514831781387329 , -0.6019822955131531 ,
   0.286661833524704  , -0.8291260004043579 , -0.36113646626472473,
  -0.6498100757598877 ,  0.21227005124092102, -0.49017271399497986,
   0.7533887028694153 , -0.44901376962661743, -1.0720415115356445 ,
  -1.7665164470672607 , -0.6841678023338318 ,  0.16721144318580627,
  -2.1323606967926025 , -0.8984572291374207 , -0.4174315929412842 ,
   0.4272245168685913 , -0.8162308931350708 ,  0.139184832572937  ,
  -0.7763519883155823 ,  0.00562418485060334,  0.5366135835647583 ,
  -0.27915534377098083,  0.21051537990570068,  0.49796080589294434,
  -0.631067156791687  , -0.8908833861351013 , -0.3673679530620575 ,
  -0.5694467425346375 , -0.4774009585380554 ,  0.7949856519699097 ,
  -0.28667768836021423, -0.3275883197784424 ,  0.7713905572891235 ,
  -0.3064861595630646 ,  0.04085521399974823,  0.03098759986460209,
   0.15040786564350128,  0.2807013988494873 , -0.24046097695827484,
   0.279535174369812  ],
 [ 0.04766317084431648,  0.215835303068161  ,  0.16244180500507355,
  -1.0936155319213867 ,  0.5195954442024231 ,  0.2765946388244629 ,
  -1.543243169784546  ,  0.05226433277130127, -0.23498114943504333,
   0.2889409363269806 ,  0.7112999558448792 ,  0.26201990246772766,
  -0.4405891001224518 , -0.3462167978286743 ,  0.281421959400177  ,
   0.15998421609401703, -0.6827236413955688 ,  0.5607661008834839 ,
  -0.5386003255844116 ,  0.2105797380208969 , -0.6561201214790344 ,
  -0.15719538927078247, -0.2138228416442871 , -0.24037830531597137,
  -0.11376425623893738, -0.13285192847251892,  0.11160276085138321,
  -0.49197909235954285, -0.49751925468444824, -0.03234574571251869,
  -0.1735544502735138 ,  0.2946568429470062 , -0.4803540110588074 ,
   0.01060363557189703, -0.5532203316688538 , -0.00736050587147474,
  -0.33764779567718506, -0.6255097389221191 , -1.1925054788589478 ,
  -0.5532750487327576 , -0.1326877325773239 , -0.1395615041255951 ,
   0.05156712234020233, -0.21963125467300415, -0.01452791597694159,
  -0.3964366018772125 ,  0.4278000593185425 , -0.02455504424870014,
  -0.22450976073741913,  0.01904799789190292, -1.1672303676605225 ,
   0.22718378901481628, -0.06090459600090981,  0.21685275435447693,
  -1.2578740119934082 , -0.7975180149078369 ,  0.796820342540741  ,
   0.2716823220252991 ,  0.11256232857704163, -0.18835870921611786,
   0.18904906511306763, -0.09663497656583786, -0.31895533204078674,
  -0.23426176607608795],
 [ 0.8972116112709045 ,  0.0783381387591362 ,  0.43657079339027405,
  -1.2948969602584839 , -0.1797798126935959 , -0.00682054786011577,
  -0.8957622647285461 ,  0.3507654368877411 ,  0.00491466885432601,
  -1.5368658304214478 ,  0.13447320461273193, -0.24307163059711456,
  -0.3166566789150238 , -0.2692895233631134 ,  0.1099780723452568 ,
   0.00028617755742744,  0.3779163956642151 , -0.590787947177887  ,
   0.2943761646747589 , -0.2154245674610138 ,  0.1912820041179657 ,
   0.5342795252799988 ,  0.17233696579933167, -0.08718453347682953,
  -0.5702390074729919 , -0.46430179476737976,  0.18094588816165924,
  -0.26043516397476196, -0.01087638270109892,  0.19007523357868195,
   0.05864182114601135,  0.3772522211074829 , -0.01090633310377598,
  -0.43246138095855713, -1.2987563610076904 , -0.17599810659885406,
   0.9851495623588562 , -0.35957249999046326,  1.0505515336990356 ,
  -0.5283899903297424 ,  0.5395058393478394 ,  0.48609471321105957,
  -0.47540098428726196,  0.6881356239318848 ,  0.4541979134082794 ,
  -3.0285544395446777 , -0.1772279292345047 ,  0.37099623680114746,
   0.8338706493377686 ,  0.48219189047813416,  0.22355353832244873,
  -0.19569644331932068, -0.7363865971565247 ,  0.15285944938659668,
   1.0900707244873047 , -0.417189359664917  , -1.0833078622817993 ,
   0.01479354687035084,  0.01972216740250587, -0.10965943336486816,
   0.2732219696044922 , -0.15916167199611664,  0.08480013906955719,
   0.5517739057540894 ],
 [ 1.219496488571167  ,  0.09687293320894241, -0.21053573489189148,
  -0.23661202192306519,  0.1768946796655655 ,  1.258920431137085  ,
  -0.8503676652908325 ,  1.0917284488677979 , -0.1768423616886139 ,
   0.5900017619132996 ,  0.7998606562614441 ,  0.19979794323444366,
   2.0682120323181152 , -0.23649756610393524, -0.19619101285934448,
  -0.0173569917678833 ,  1.072580337524414  ,  0.3505484163761139 ,
   0.5238627791404724 ,  0.17473259568214417,  0.5203684568405151 ,
   0.4803808629512787 , -2.0572662353515625 , -0.07926522940397263,
  -0.9042485952377319 ,  0.09214421361684799, -0.5482761859893799 ,
  -1.011600136756897  , -1.2490710020065308 , -0.896273136138916  ,
   0.5625092387199402 , -0.20009814202785492, -1.0393630266189575 ,
  -2.18727445602417   ,  0.2756064534187317 , -0.13432972133159637,
   1.219194769859314  , -0.10704567283391953,  0.02796582691371441,
   1.4677621126174927 ,  1.0430375337600708 , -0.6757073998451233 ,
  -0.3787325620651245 , -1.4298720359802246 , -0.7980685830116272 ,
   0.1106136217713356 ,  0.20638002455234528, -1.4452612400054932 ,
  -0.2644202709197998 , -0.1546895056962967 , -0.4315127432346344 ,
  -0.14838553965091705,  0.09956922382116318, -1.548740267753601  ,
   0.6838295459747314 , -2.435929536819458  ,  0.21477673947811127,
   0.46144425868988037, -0.09702379256486893,  0.3100239336490631 ,
   0.4169653654098511 ,  0.45063161849975586,  1.2624101638793945 ,
  -0.2528347373008728 ],
 [ 0.14676296710968018, -0.44066375494003296,  0.6151591539382935 ,
   0.36857470870018005,  0.3574996590614319 ,  0.2851361632347107 ,
   0.28903675079345703,  0.3898772597312927 , -1.0449849367141724 ,
   0.37866440415382385,  0.15729933977127075, -0.04811133071780205,
  -0.00335335871204734, -2.0941078662872314 , -0.01363870315253735,
   0.3006877899169922 ,  0.5246029496192932 ,  0.3128907382488251 ,
   0.26491907238960266,  0.5536222457885742 ,  0.3598853349685669 ,
  -1.3941212892532349 , -1.197419285774231  ,  0.07520715892314911,
  -0.9623863101005554 , -0.575768232345581  , -0.07619047164916992,
   0.07757564634084702,  0.8540206551551819 , -0.5770028829574585 ,
  -0.13453726470470428,  0.05054067447781563,  0.6933692097663879 ,
  -1.6323922872543335 , -0.15284600853919983,  0.10655288398265839,
   0.16538900136947632,  0.14776112139225006, -0.06941334903240204,
  -0.5490314364433289 ,  0.453297883272171  , -0.247477188706398  ,
  -0.0948653519153595 , -0.24980729818344116, -0.31952613592147827,
   1.221948266029358  , -0.4914020895957947 ,  0.635668933391571  ,
   0.02925513684749603,  0.01087888982146978,  0.23186098039150238,
  -0.5439465641975403 , -0.0679386630654335 ,  0.01720459572970867,
   0.6543618440628052 , -0.4344846308231354 , -0.5340933203697205 ,
  -0.5275153517723083 ,  0.47936657071113586, -0.16737012565135956,
   0.38205888867378235, -0.20861220359802246,  0.1925523579120636 ,
  -1.2464599609375    ],
 [-0.23092590272426605,  0.1939588189125061 , -0.11406932771205902,
   1.0086525678634644 ,  0.2961777150630951 ,  0.3737042248249054 ,
   0.8082664012908936 ,  0.02217493765056133,  0.09686904400587082,
   0.6930878162384033 ,  0.21788553893566132,  0.48566529154777527,
  -1.9039247035980225 , -0.9011325836181641 , -0.472949355840683  ,
   0.5816656947135925 ,  1.025701880455017  , -1.451037883758545  ,
  -0.5511967539787292 , -0.09050951898097992,  0.23538900911808014,
  -0.767089307308197  , -0.09914186596870422,  0.5656755566596985 ,
   0.7208877205848694 , -0.7319192290306091 , -0.7003673315048218 ,
   0.33509814739227295,  0.11153154820203781, -0.20702999830245972,
   0.42769137024879456,  0.681125819683075  ,  0.46138253808021545,
   0.393706351518631  , -0.533491313457489  ,  0.2317802459001541 ,
  -0.05080869421362877,  1.206034541130066  ,  1.028281331062317  ,
  -0.9690194129943848 , -0.22889332473278046,  0.0589027926325798 ,
  -0.20796437561511993, -0.1785363107919693 , -0.6166702508926392 ,
   0.8976539373397827 ,  0.7196559906005859 ,  0.9543195366859436 ,
  -1.9815716743469238 , -0.2515859603881836 , -0.11762569844722748,
   0.3304080069065094 ,  0.10799837112426758,  0.7617499232292175 ,
   0.5332794785499573 ,  1.054216742515564  , -0.6630154252052307 ,
   0.19756264984607697, -0.6299686431884766 , -0.09709807485342026,
   1.2510701417922974 ,  0.821222186088562  ,  0.29561299085617065,
   0.147157222032547  ],
 [-1.850899338722229  ,  0.7619916796684265 , -0.15233120322227478,
  -1.6298691034317017 ,  0.5021862387657166 ,  1.425594687461853  ,
   0.34413617849349976,  0.00011850034206873,  0.06518075615167618,
   0.5552172064781189 , -0.14475186169147491,  0.41390591859817505,
  -0.25285688042640686,  0.3959391415119171 , -0.6064026951789856 ,
  -0.25793102383613586, -0.5778828263282776 ,  0.6081355214118958 ,
   0.35284021496772766,  0.571572482585907  , -0.8124894499778748 ,
   1.027829885482788  , -0.16607509553432465,  0.28975802659988403,
  -1.3806489706039429 ,  0.2542230486869812 ,  0.6069433689117432 ,
  -0.47113120555877686,  0.4262390732765198 , -0.33045634627342224,
  -0.14765429496765137,  0.5414340496063232 , -0.23972265422344208,
   0.5297396183013916 , -0.21922776103019714, -1.9503591060638428 ,
   0.18184255063533783,  0.05990448221564293, -3.5497279167175293 ,
   1.034893274307251  ,  0.21293547749519348,  0.4152420163154602 ,
  -1.0495997667312622 , -0.3129844665527344 ,  0.35192689299583435,
  -0.24060767889022827, -1.2616604566574097 , -0.8500649333000183 ,
   0.6171780228614807 , -1.5357868671417236 , -0.7192677855491638 ,
  -0.25740116834640503,  0.8059431314468384 , -0.30746495723724365,
  -0.6348519325256348 , -0.11132923513650894,  0.17582103610038757,
  -0.03622382879257202, -0.20493647456169128, -0.2984602153301239 ,
  -0.22993658483028412, -0.08825487643480301,  0.3786134719848633 ,
   0.8446432948112488 ],
 [ 0.04514903575181961, -0.10274287313222885, -0.5452362298965454 ,
  -0.2049243003129959 , -0.1168607696890831 , -0.5107750296592712 ,
   0.02079076133668423, -1.1552326679229736 ,  0.10883273929357529,
   0.5837289094924927 , -0.7894514799118042 , -0.48758426308631897,
   0.19531220197677612, -1.7047946453094482 , -0.8422963619232178 ,
  -1.1218384504318237 , -0.764235258102417  , -0.9050850868225098 ,
   0.07322902232408524, -0.0854148268699646 ,  0.18595510721206665,
  -0.8017553091049194 , -0.4276620149612427 , -0.8029270768165588 ,
   0.2506219446659088 ,  0.5327722430229187 , -0.05754430219531059,
  -0.9770674109458923 ,  0.13373732566833496, -0.8559269309043884 ,
   1.3819785118103027 , -0.31958696246147156, -0.7669997811317444 ,
  -0.4908864200115204 ,  0.3052920997142792 , -2.117663860321045  ,
   0.9584285616874695 ,  0.9204729199409485 , -3.3222358226776123 ,
   0.4379867911338806 ,  0.1350269317626953 ,  0.14572450518608093,
   0.01632731221616268, -0.090599425137043  ,  0.6473307609558105 ,
  -2.753047227859497  , -0.04418433457612991, -0.2131773680448532 ,
  -0.36167311668395996,  0.44110310077667236, -1.245144009590149  ,
  -0.05455369502305984, -1.2549598217010498 , -1.3697715997695923 ,
  -0.5141902565956116 , -1.3051871061325073 ,  0.06996455788612366,
  -0.01431478094309568,  0.11949816346168518, -0.25819748640060425,
  -0.25877124071121216,  0.4153517484664917 ,  0.3512979745864868 ,
   0.0774291381239891 ],
 [-0.12967738509178162, -0.2600470781326294 , -0.19355341792106628,
  -2.6432199478149414 ,  0.3225434422492981 , -0.6512402296066284 ,
  -0.04000473767518997, -0.3994278013706207 , -0.221802219748497  ,
   1.3010485172271729 , -0.9560525417327881 , -1.4011632204055786 ,
  -0.10577093809843063, -0.3337092697620392 , -0.5721830725669861 ,
  -1.7444097995758057 , -0.32053759694099426, -0.8361244797706604 ,
   0.3164648711681366 , -0.3067277669906616 ,  0.19396258890628815,
  -0.6494963765144348 ,  0.4137493968009949 , -0.10322081297636032,
  -0.34186258912086487,  0.32985836267471313,  0.19858945906162262,
   0.7057337164878845 ,  0.9155744910240173 ,  0.4519670009613037 ,
  -0.3464376628398895 , -3.9949700832366943 ,  0.1057693287730217 ,
  -0.0535811223089695 , -0.44566190242767334, -0.823025107383728  ,
   0.49180299043655396,  0.15972623229026794, -0.1796308010816574 ,
   0.7298051714897156 , -0.4905421733856201 ,  0.2815125286579132 ,
  -0.527108371257782  ,  0.1157442256808281 ,  0.04869550094008446,
  -0.9152787327766418 , -1.2196842432022095 ,  1.1146634817123413 ,
   0.6069611310958862 ,  0.12034423649311066, -0.6663187742233276 ,
   0.09235559403896332, -0.4721618592739105 , -0.00529542053118348,
   0.2920503318309784 , -0.7288491129875183 , -0.8692843914031982 ,
   0.19459301233291626,  0.11380793899297714,  0.5906965136528015 ,
   0.07427957653999329,  1.0281070470809937 ,  0.26775193214416504,
   0.26016661524772644],
 [-0.1971627175807953 , -0.25655654072761536,  0.23418721556663513,
  -0.5881516933441162 ,  0.2014017105102539 , -0.49258801341056824,
  -0.7899031043052673 , -0.5062986612319946 , -0.12899300456047058,
  -0.8096883296966553 , -3.7220451831817627 , -0.18248271942138672,
  -1.4638229608535767 , -0.4482603669166565 ,  0.47684359550476074,
   0.18360337615013123, -0.4800032377243042 ,  0.6450673937797546 ,
  -0.5360309481620789 ,  0.3083648979663849 , -0.6836256384849548 ,
   0.17430749535560608, -2.5210680961608887 , -0.12945427000522614,
   0.17244736850261688,  0.3175675570964813 ,  0.11926646530628204,
  -0.19100074470043182,  0.4986163377761841 , -0.5078816413879395 ,
  -0.3084091246128082 ,  0.2487686574459076 , -0.01277539599686861,
  -3.6623454093933105 , -0.8618050813674927 , -0.4134593605995178 ,
  -0.8002777099609375 , -0.15544100105762482, -0.46947282552719116,
   0.6607698202133179 , -0.03289353847503662, -0.0760139599442482 ,
   0.317948579788208  , -0.2970294654369354 ,  0.24316291511058807,
   1.186650037765503  ,  0.35887688398361206, -0.5472650527954102 ,
  -0.22545991837978363, -1.6370924711227417 , -1.2009189128875732 ,
  -0.6439732313156128 , -0.021537771448493  , -0.2616823613643646 ,
   0.16770875453948975,  0.40989282727241516, -0.17300572991371155,
  -0.02795888297259808, -0.11453868448734283,  0.07122202962636948,
  -0.3864234685897827 , -0.3874344825744629 , -0.23650908470153809,
  -0.02786406315863132],
 [ 0.3659495413303375 , -0.8451803922653198 ,  0.33533522486686707,
  -0.6561660170555115 , -0.9459128975868225 , -1.4846218824386597 ,
  -1.0241246223449707 ,  0.6851457953453064 ,  0.11067239195108414,
  -1.6155130863189697 ,  0.201968252658844  , -1.3410183191299438 ,
  -3.4836630821228027 , -1.1420352458953857 , -0.1837998628616333 ,
   0.3886112570762634 , -0.17152759432792664, -0.49134865403175354,
   0.12885037064552307, -0.6149414777755737 ,  0.37668246030807495,
  -1.2331124544143677 , -0.5855608582496643 , -1.0315073728561401 ,
   0.3998083472251892 , -0.4925278425216675 , -0.7110735774040222 ,
   0.9619907736778259 , -0.6851580739021301 , -0.8349388837814331 ,
   0.3216208815574646 , -0.8227485418319702 ,  1.1566458940505981 ,
   0.1685340702533722 , -0.4082700312137604 , -0.2009045034646988 ,
  -1.183923602104187  ,  0.22584718465805054,  0.18824468553066254,
   0.3069765865802765 ,  0.14046597480773926, -0.6170152425765991 ,
  -0.15552926063537598,  0.02721583843231201, -0.10927430540323257,
  -0.5180400013923645 , -0.9540445804595947 , -1.1923755407333374 ,
  -1.8308149576187134 ,  0.3469996154308319 ,  0.9255794286727905 ,
   0.31205204129219055, -0.5482603907585144 ,  0.40499937534332275,
   0.3788718283176422 ,  0.48784759640693665, -0.43107014894485474,
   0.3585503101348877 ,  0.18385829031467438, -0.19552594423294067,
   0.3509533107280731 , -0.22909855842590332,  0.08635455369949341,
  -0.7376599907875061 ],
 [ 0.7772475481033325 , -0.7125399708747864 , -0.00115230947267264,
  -0.09426335990428925,  0.0150492349639535 , -0.11126329004764557,
  -0.442059725522995  , -0.5597124695777893 ,  0.13672836124897003,
   0.5511643290519714 , -0.36711207032203674,  0.5454452633857727 ,
  -0.3322334885597229 , -0.05085886269807816,  0.16282610595226288,
  -0.3821622431278229 , -0.623309850692749  , -1.3723808526992798 ,
  -1.484493613243103  , -0.01057894062250853,  0.1295793354511261 ,
  -1.3705215454101562 , -1.5484603643417358 , -0.675300121307373  ,
   0.4314783215522766 ,  0.08217032253742218, -1.325472116470337  ,
   0.09184424579143524, -0.6091569662094116 ,  0.3804401457309723 ,
   1.3377795219421387 ,  0.02427811175584793,  0.5809143781661987 ,
  -2.570077657699585  , -0.23422488570213318,  0.4463246166706085 ,
   0.2747359275817871 , -1.3630393743515015 ,  0.7032722234725952 ,
   0.7221188545227051 , -0.18121376633644104,  0.5317168235778809 ,
   0.7236783504486084 , -1.0854216814041138 ,  1.3471959829330444 ,
  -2.861147403717041  , -0.17807631194591522,  1.9058282375335693 ,
  -0.315624475479126  , -0.4367765784263611 , -1.5123422145843506 ,
  -1.783739447593689  , -1.1597838401794434 , -0.4462900757789612 ,
   0.638925313949585  ,  1.028701663017273  , -1.8656611442565918 ,
  -0.35074862837791443,  0.6181119084358215 , -0.3107185661792755 ,
   1.2030439376831055 , -0.17347773909568787,  0.16212786734104156,
   0.32860180735588074],
 [ 0.17504779994487762,  0.30950307846069336, -0.33640241622924805,
   0.03585626557469368, -0.02363185398280621,  0.01527861319482327,
  -0.4668196737766266 ,  0.583925187587738  , -0.04955627769231796,
   0.22694218158721924, -0.366456538438797  , -0.4496614634990692 ,
   0.10655033588409424,  0.15708036720752716,  0.00356643972918391,
   0.13283179700374603, -0.8659577369689941 ,  0.9144266843795776 ,
  -0.03678666800260544,  0.08469852060079575,  0.29639706015586853,
   0.28723153471946716,  0.09040249139070511, -0.7593455910682678 ,
  -0.5232561230659485 , -0.34776636958122253,  0.07339204847812653,
  -0.653023898601532  ,  0.41117629408836365, -0.06913397461175919,
  -0.1104254275560379 , -0.6783714890480042 ,  0.20378927886486053,
  -0.24735097587108612,  0.9785204529762268 , -0.01434758026152849,
   0.49641093611717224,  0.1853838860988617 , -0.19032661616802216,
   0.2624445855617523 , -1.343632698059082  ,  0.2754119336605072 ,
   0.09423257410526276, -0.4148145914077759 , -1.5937700271606445 ,
  -0.4734625518321991 ,  0.42776671051979065, -0.6013364791870117 ,
  -0.8573549389839172 , -0.4226585328578949 , -0.07948357611894608,
  -0.9531707763671875 , -0.8344763517379761 , -0.01234414055943489,
   0.715948224067688  , -0.973932147026062  ,  0.5068946480751038 ,
  -0.01542115584015846,  0.08028095960617065, -0.9228739738464355 ,
   0.2553173899650574 ,  0.37613916397094727,  0.8462586402893066 ,
   0.0762452483177185 ],
 [ 0.4271790087223053 , -0.8308091759681702 ,  0.4359917938709259 ,
   1.3345263004302979 , -0.22693809866905212, -0.8880434632301331 ,
   0.3312948942184448 ,  0.04414005950093269,  0.09199325740337372,
  -1.1592482328414917 , -1.7832725048065186 , -0.2696036696434021 ,
   0.5467236638069153 ,  0.34577491879463196, -1.3679865598678589 ,
  -0.3391845226287842 ,  1.044293761253357  ,  0.4518526494503021 ,
   1.172079086303711  ,  1.1141473054885864 , -1.2727259397506714 ,
  -0.2069469541311264 ,  0.7160786390304565 ,  0.06550470739603043,
   0.7606908679008484 ,  0.34376755356788635, -0.01983083970844746,
   1.5041141510009766 ,  1.0539674758911133 ,  0.02817349880933762,
   0.6480779051780701 ,  0.6266098022460938 , -0.30193987488746643,
  -0.3238268494606018 , -0.33405601978302   , -0.7400023341178894 ,
  -1.7197840213775635 ,  0.28022247552871704, -0.6185440421104431 ,
  -0.1528511494398117 ,  1.482427954673767  , -0.06492380052804947,
  -0.21487152576446533,  0.21583010256290436,  0.1822357326745987 ,
  -0.7218232750892639 ,  0.02728475444018841,  1.1107136011123657 ,
  -0.2533116936683655 , -1.9629572629928589 ,  1.147230625152588  ,
   0.16101723909378052,  0.745681643486023  , -0.01534127257764339,
   1.2777479887008667 , -2.8933544158935547 , -0.03650156036019325,
   0.02777059935033321,  0.8920383453369141 ,  0.7189444899559021 ,
  -1.4278382062911987 ,  1.4231538772583008 ,  1.1555622816085815 ,
   0.5264956951141357 ],
 [ 0.6399292349815369 , -0.2608294188976288 , -0.2771366536617279 ,
  -2.6279678344726562 ,  0.08063039183616638, -0.30009695887565613,
  -0.7728082537651062 ,  0.8244295120239258 , -0.11384288221597672,
  -0.6720226407051086 , -1.3830996751785278 ,  0.47179269790649414,
  -1.0968172550201416 , -1.9385126829147339 ,  0.15408748388290405,
  -0.4668149948120117 , -0.1980489194393158 , -1.964172124862671  ,
  -1.418411374092102  , -0.2791450321674347 , -0.4476904571056366 ,
   0.03729343414306641,  0.8036038279533386 ,  0.44227150082588196,
  -1.7069060802459717 , -2.283391237258911  , -0.21693716943264008,
   0.5991808772087097 ,  0.3751439154148102 ,  0.18813765048980713,
  -0.4090420603752136 , -0.12432174384593964,  0.01365430001169443,
  -1.6742602586746216 ,  0.19364848732948303, -0.0429919958114624 ,
   0.1518249213695526 ,  0.11685334146022797, -0.6293097138404846 ,
  -0.12234267592430115, -0.38698527216911316, -0.403476357460022  ,
   0.6123864650726318 ,  0.23480454087257385,  0.6906588077545166 ,
  -1.7779570817947388 , -1.4294512271881104 , -0.14672628045082092,
   0.00996615085750818,  0.12351928651332855, -0.6813549399375916 ,
   0.01796894334256649, -0.546210527420044  , -1.0411320924758911 ,
   0.095926932990551  ,  1.3360793590545654 ,  0.15074415504932404,
   0.09207380563020706, -0.21954339742660522, -0.19109950959682465,
   0.1459953337907791 , -3.499229907989502  ,  0.2523341774940491 ,
  -0.2066609263420105 ],
 [ 0.6794878244400024 ,  0.9160927534103394 , -0.0352381095290184 ,
   0.5008271336555481 ,  0.4515205919742584 ,  0.11150044947862625,
   0.12244755774736404,  1.6252241134643555 , -0.12531358003616333,
   0.03325406089425087,  1.4316719770431519 ,  0.27928590774536133,
   0.6454974412918091 , -0.5439644455909729 , -0.6359687447547913 ,
   0.8661314249038696 ,  2.194969654083252  , -0.3382246196269989 ,
  -0.48833122849464417, -0.16482053697109222,  0.36043137311935425,
  -0.03344879299402237,  1.2708122730255127 , -0.3387393653392792 ,
  -0.6740492582321167 , -0.07685507088899612, -0.2983294129371643 ,
   1.2319815158843994 , -1.3939629793167114 ,  1.1245747804641724 ,
   1.9957839250564575 ,  0.5140841603279114 ,  0.1568841189146042 ,
   0.9416601657867432 ,  0.13499006628990173,  0.8812452554702759 ,
  -0.21614012122154236,  0.34809044003486633,  2.006273031234741  ,
   1.5485790967941284 ,  0.5058444738388062 , -0.3644747734069824 ,
   0.12604956328868866,  0.18766076862812042,  0.8658157587051392 ,
  -0.6394777297973633 ,  1.0249814987182617 ,  0.20596051216125488,
   0.02058643288910389, -0.28056779503822327,  0.33209121227264404,
   0.4056709408760071 ,  1.138335108757019  ,  0.22877249121665955,
   1.0279710292816162 , -0.04798382893204689,  0.8069608211517334 ,
   0.253898948431015  , -0.6020837426185608 , -0.5792072415351868 ,
  -0.3123161792755127 , -0.1767626702785492 ,  0.16992951929569244,
   0.6607475876808167 ],
 [-0.5058205723762512 , -0.7966898083686829 ,  0.18979418277740479,
   0.5789650082588196 ,  0.19435863196849823, -1.002569556236267  ,
  -2.0866386890411377 ,  0.640833854675293  , -0.08842918276786804,
   1.5823456048965454 , -0.00526752602308989,  0.13353540003299713,
   0.5743489265441895 , -1.1459250450134277 ,  0.03438979014754295,
   0.2568984925746918 , -0.27294060587882996, -1.6220687627792358 ,
   0.28507572412490845, -0.1259729564189911 ,  0.5544818639755249 ,
  -0.8587132692337036 ,  0.8132126331329346 ,  0.2549822926521301 ,
  -5.073445796966553  , -1.3551452159881592 ,  0.15908391773700714,
   3.4844563007354736 ,  0.48159101605415344,  0.08468402177095413,
   0.00413242261856794, -0.0744103193283081 ,  1.8132643699645996 ,
  -2.244091749191284  , -0.2899980843067169 , -3.035097360610962  ,
   0.3085363805294037 , -1.4607055187225342 ,  1.906296730041504  ,
   1.790163278579712  , -0.13705122470855713, -0.21710427105426788,
   1.5952569246292114 ,  0.12265834957361221,  1.2439770698547363 ,
   1.9548381567001343 , -0.8899829387664795 ,  0.8318614363670349 ,
  -0.1718444526195526 , -1.8682875633239746 ,  0.9039962887763977 ,
  -0.10151489078998566,  0.4740813076496124 , -0.8359997272491455 ,
   0.8598150610923767 , -0.11690955609083176,  0.46825194358825684,
  -0.5306552052497864 ,  0.375667005777359  ,  0.6323011517524719 ,
  -0.21135295927524567,  1.1080098152160645 , -0.35436782240867615,
  -1.4602142572402954 ],
 [-0.04454635456204414, -0.43833228945732117, -0.00975841470062733,
  -0.4716171324253082 , -0.00777434790506959, -0.6080015897750854 ,
  -0.979563295841217  ,  0.4046213626861572 , -0.01821083389222622,
  -0.12499861419200897, -0.1685306280851364 , -0.8818159103393555 ,
  -0.5898218750953674 , -0.6565248966217041 , -0.06739415228366852,
   0.44261443614959717, -0.288075715303421  , -0.19862154126167297,
   0.42048177123069763, -0.00521412352100015, -0.1524651050567627 ,
  -0.11523851752281189, -0.6490834355354309 , -0.19267357885837555,
   0.4093107581138611 , -0.09641410410404205,  0.6458792686462402 ,
  -0.07311029732227325, -0.08197139203548431,  0.4962005317211151 ,
  -0.19826166331768036, -0.5984688401222229 , -1.4380167722702026 ,
  -0.21102741360664368,  0.3104402422904968 , -1.050480842590332  ,
   0.1696358323097229 ,  0.2130669355392456 , -0.1481911838054657 ,
   0.4654962420463562 ,  0.46034732460975647,  0.13456466794013977,
  -0.6743948459625244 ,  0.1398809403181076 ,  0.33331307768821716,
   0.42853844165802   , -0.3763536214828491 ,  0.2130618542432785 ,
   0.28597673773765564, -0.6558079719543457 , -0.17951007187366486,
   0.10916417092084885, -0.05376450344920158,  1.5125792026519775 ,
  -0.2167510688304901 , -0.11018866300582886, -0.13811790943145752,
  -0.22633422911167145, -0.46199095249176025,  0.2920355498790741 ,
  -0.4954613745212555 ,  0.27611321210861206,  0.5096117854118347 ,
   0.40878358483314514],
 [-0.6097579002380371 ,  0.2566952109336853 , -0.13345684111118317,
  -0.4224526286125183 ,  0.4250538647174835 , -0.6194536685943604 ,
   0.8464525938034058 , -0.21960221230983734,  0.20957978069782257,
   0.09417670220136642, -0.5782071352005005 , -0.07007391005754471,
  -0.01437452808022499, -0.4730322062969208 ,  1.0682413578033447 ,
   0.01290759071707726, -0.849029004573822  ,  0.24642619490623474,
  -0.6361956000328064 ,  0.1398187130689621 , -1.0579677820205688 ,
  -0.01289851870387793, -1.099626064300537  ,  0.1650930792093277 ,
  -2.375957489013672  , -0.3758607804775238 ,  0.02364741079509258,
   0.953814685344696  , -1.1989928483963013 , -0.2691223621368408 ,
   0.00136431213468313, -0.13070642948150635, -1.0089529752731323 ,
  -0.0400017723441124 , -0.9510316252708435 , -0.648496150970459  ,
  -2.1669270992279053 ,  0.06659748405218124, -1.6081053018569946 ,
  -0.27830976247787476,  0.7612926363945007 , -0.08651723712682724,
  -0.3721173405647278 , -0.05707656219601631, -0.6304328441619873 ,
   0.02669994346797466,  0.05217336490750313, -0.14481884241104126,
   2.2412304878234863 ,  0.2741129696369171 , -0.7809104323387146 ,
  -1.7412141561508179 , -1.091988444328308  ,  0.27015987038612366,
  -0.7391509413719177 ,  1.1987382173538208 , -0.3032696843147278 ,
   0.6303169131278992 ,  0.11424991488456726, -0.2440720647573471 ,
   0.1776609867811203 , -0.8284353017807007 , -0.06790541857481003,
  -1.224865198135376  ],
 [ 0.06033048778772354,  0.3179484009742737 ,  0.12560424208641052,
  -0.5650527477264404 , -0.13369733095169067, -0.6086205244064331 ,
  -2.516623020172119  , -0.45368438959121704,  0.11180275678634644,
  -3.3207623958587646 , -1.2520155906677246 , -1.017558217048645  ,
  -0.6566111445426941 ,  0.34308484196662903, -0.01902199536561966,
  -0.38697031140327454,  0.570758044719696  ,  0.00752294342964888,
  -0.15431083738803864,  0.10704749822616577,  0.20954300463199615,
  -0.03764278814196587, -0.518940806388855  ,  0.01770809106528759,
  -0.17005981504917145, -0.0619230642914772 ,  0.04211913794279099,
   0.27707651257514954, -3.3463563919067383 , -0.4612564444541931 ,
   0.27534642815589905,  0.20355702936649323, -0.16795411705970764,
  -0.7923266291618347 , -0.6064532995223999 ,  0.17282137274742126,
   0.6094462871551514 , -1.713019847869873  ,  0.03876639902591705,
   0.12406376749277115,  0.05737999454140663,  0.03122293017804623,
  -0.1974104642868042 , -0.1377803087234497 , -0.08332200348377228,
  -1.3761895895004272 , -1.1567997932434082 ,  0.06437607109546661,
  -0.17811593413352966, -0.198212668299675  ,  0.14688508212566376,
  -0.20612095296382904,  0.22898542881011963,  0.36580735445022583,
   0.07528157532215118, -0.6464027762413025 ,  0.10829328000545502,
  -0.08442038297653198, -0.19622290134429932,  0.13627193868160248,
   0.12972179055213928, -0.27618148922920227,  0.14241641759872437,
   0.5264515280723572 ],
 [ 0.10261501371860504,  0.14287984371185303,  0.18943899869918823,
  -0.5065404772758484 ,  0.13227932155132294,  0.7277411818504333 ,
  -0.39237329363822937, -1.199636697769165  ,  0.10460011661052704,
   0.14932796359062195, -0.18028953671455383, -0.11785061657428741,
  -0.3794018626213074 , -0.6150773763656616 , -1.8492541313171387 ,
   0.44470539689064026,  0.5667910575866699 , -0.45330727100372314,
   0.06632574647665024,  0.5945131182670593 ,  0.6734249591827393 ,
   0.06195339933037758, -1.7409822940826416 , -0.33201342821121216,
  -0.04393301904201508,  0.2589229643344879 ,  0.13537448644638062,
  -1.8943530321121216 ,  0.15189535915851593,  0.8505337238311768 ,
   0.7876309156417847 , -0.01319828722625971,  0.7266526222229004 ,
  -2.407379388809204  , -1.1209245920181274 , -2.309056282043457  ,
  -0.14487431943416595, -0.2593923807144165 ,  0.01348666846752167,
  -0.0231280867010355 ,  0.142556294798851  , -0.42810961604118347,
  -1.7313562631607056 ,  0.2359529733657837 , -0.5038846135139465 ,
  -0.5792409777641296 ,  0.22313182055950165, -1.3781201839447021 ,
  -0.7811972498893738 , -0.10800310969352722, -2.8443331718444824 ,
  -0.1719692200422287 , -0.4908295273780823 ,  0.0856531411409378 ,
  -0.8879141807556152 , -0.8302880525588989 ,  0.46147051453590393,
   0.11660858988761902,  0.02520343661308289, -1.165176272392273  ,
  -0.11935894191265106,  0.0993868038058281 , -0.4950270354747772 ,
   0.3839380741119385 ],
 [-0.16403833031654358,  0.0621849037706852 , -0.10509874671697617,
  -1.0996133089065552 , -0.8520668745040894 ,  0.351202130317688  ,
   0.08909022063016891, -0.7996699810028076 ,  0.182868093252182  ,
  -0.6352853178977966 ,  0.714349091053009  ,  0.9777148962020874 ,
  -2.3762214183807373 ,  0.3052462339401245 , -1.624293327331543  ,
   0.13933764398097992,  0.4080658555030823 ,  0.02262947708368301,
  -3.055077314376831  , -1.3575913906097412 , -1.2587510347366333 ,
   0.271065890789032  , -1.1636723279953003 , -2.0005836486816406 ,
  -0.7907817363739014 , -0.41337960958480835,  0.6875051259994507 ,
  -0.6235655546188354 , -1.052707552909851  , -1.0088765621185303 ,
  -0.7122829556465149 , -0.49444782733917236, -0.0451187863945961 ,
  -0.2272353172302246 , -0.09530679136514664,  0.13814865052700043,
  -0.23038703203201294,  0.15618759393692017,  1.1024209260940552 ,
   0.2815374732017517 ,  0.24823017418384552, -0.13890986144542694,
  -0.28569847345352173,  0.35591670870780945,  0.11614672839641571,
  -0.3788934350013733 ,  0.01047232933342457,  0.25411662459373474,
  -3.58532452583313   , -0.6402757167816162 ,  0.5165061950683594 ,
  -0.9632599353790283 , -0.40749531984329224, -0.9701287746429443 ,
  -0.22246845066547394,  0.1861966848373413 ,  0.13676297664642334,
   0.05446446314454079,  0.7068414092063904 ,  0.00588479684665799,
  -0.4488275647163391 , -0.5929940342903137 ,  0.21856923401355743,
   0.6531391739845276 ],
 [ 0.3481486141681671 , -0.16781842708587646, -0.04227572679519653,
  -0.6312587261199951 , -0.19128093123435974,  0.5531406402587891 ,
  -1.7656012773513794 , -0.07022447139024734,  0.3760603070259094 ,
  -1.3451259136199951 , -0.8919914364814758 , -0.6057090759277344 ,
  -0.9691338539123535 , -0.9206752777099609 , -0.4671866297721863 ,
   0.06103753671050072, -0.397306352853775  , -0.4594568610191345 ,
   1.140702247619629  ,  0.02562778070569038, -0.9188195466995239 ,
  -0.04023922234773636, -0.54921555519104   ,  0.13936111330986023,
   0.01237406115978956, -0.8747702836990356 ,  0.4658964276313782 ,
   0.06843867897987366, -1.4364538192749023 ,  0.6574788093566895 ,
  -0.3378579914569855 , -0.6317638754844666 ,  0.6960939764976501 ,
  -1.026046872138977  , -0.9775983691215515 ,  0.08535020798444748,
  -0.05182977765798569,  0.1634560525417328 , -0.04556934535503387,
  -0.19540686905384064,  0.4862825572490692 ,  0.4559563398361206 ,
  -0.1311582326889038 ,  0.1559005230665207 ,  0.8092703223228455 ,
  -1.0061975717544556 , -0.16765442490577698, -0.18416185677051544,
  -1.0126614570617676 ,  0.7030791640281677 ,  1.255228042602539  ,
   0.3857557475566864 ,  0.11248364299535751, -0.4516565203666687 ,
   0.26777011156082153, -0.4394560158252716 ,  0.986492395401001  ,
  -0.448407381772995  , -0.03934881836175919,  0.04256768524646759,
   0.27292361855506897,  0.3424869477748871 ,  0.4238533675670624 ,
   0.577512800693512  ],
 [ 1.3889203071594238 ,  0.6224120259284973 ,  0.17979399859905243,
   0.3722907304763794 ,  0.21638868749141693,  0.2810900807380676 ,
  -0.24959273636341095, -0.29955610632896423, -0.17526793479919434,
   0.8263867497444153 , -0.36262282729148865,  0.11524933576583862,
   0.22035525739192963, -0.5783261656761169 ,  0.12857677042484283,
  -0.08395927399396896,  1.1693272590637207 , -0.8323485851287842 ,
  -0.09413523226976395,  0.16130545735359192,  0.37120598554611206,
   0.7715646624565125 , -1.382163166999817  ,  0.15727028250694275,
  -0.6059391498565674 ,  0.3112978935241699 , -0.6909592151641846 ,
   2.3797216415405273 ,  0.4768528938293457 , -0.4015672504901886 ,
   0.9682978391647339 , -0.03442084044218063,  0.7453989386558533 ,
  -0.5311114192008972 ,  0.04552534967660904,  0.9249821305274963 ,
   1.4907699823379517 ,  0.7146417498588562 ,  1.658004879951477  ,
   0.5068358778953552 , -0.4755442440509796 , -0.14704465866088867,
   1.8199964761734009 , -0.37278637290000916, -0.6855904459953308 ,
  -0.29340457916259766, -1.626990556716919  ,  1.3331671953201294 ,
  -0.2371523529291153 , -1.0819528102874756 , -0.22181537747383118,
   0.24246123433113098,  0.4074692726135254 , -0.28911879658699036,
   3.0646767616271973 ,  1.067123532295227  ,  0.17266274988651276,
   0.1460600048303604 ,  0.01886747777462006, -1.139181137084961  ,
  -0.00773931806907058, -0.26227185130119324, -0.11818145960569382,
   0.06065019220113754],
 [-0.04719744622707367, -1.1582300662994385 , -0.00573134981095791,
   0.27439311146736145, -0.47145482897758484, -0.413027822971344  ,
   0.06231221184134483,  0.8957247138023376 , -0.23841391503810883,
  -0.3668235242366791 ,  0.28159719705581665,  0.290914386510849  ,
  -1.7679413557052612 , -1.0075829029083252 ,  0.00504247983917594,
   0.1926838606595993 , -0.3057805597782135 , -0.5641481876373291 ,
  -0.4408331513404846 ,  0.31374260783195496, -0.7182909250259399 ,
  -1.4800498485565186 , -0.5700703263282776 ,  0.9092439413070679 ,
   0.2891540825366974 , -0.8216878175735474 ,  0.8609731197357178 ,
   0.07192838191986084,  0.8493456840515137 ,  0.7695804238319397 ,
  -0.11604902148246765, -0.24562060832977295, -0.42940711975097656,
  -1.4960088729858398 ,  0.05937375500798225,  0.1740499585866928 ,
  -0.5034211277961731 , -0.08835013955831528, -0.5961924195289612 ,
  -0.00535899819806218, -0.25760790705680847,  0.48922866582870483,
   0.2789117097854614 ,  0.6068552732467651 ,  0.8686568737030029 ,
   0.07773993909358978,  0.783042311668396  ,  0.687731921672821  ,
  -0.49546077847480774, -1.0242469310760498 , -0.6008914113044739 ,
   0.2785942852497101 , -0.20765025913715363,  0.2191244214773178 ,
   0.2445923537015915 ,  0.3768363296985626 ,  0.5564823150634766 ,
   0.1044926568865776 ,  1.057348370552063  ,  0.12596887350082397,
   0.01746871508657932, -0.10333823412656784, -0.06115277484059334,
  -0.3202815651893616 ],
 [ 0.02355766668915749, -0.7057017683982849 , -0.22417683899402618,
   0.3493281304836273 ,  0.5979335308074951 ,  0.26537325978279114,
   0.05563309416174889, -0.3502860367298126 ,  0.2865896224975586 ,
  -0.5738500952720642 , -0.39939579367637634, -1.7064707279205322 ,
  -0.6023302674293518 ,  0.3147750496864319 ,  0.3713069558143616 ,
  -0.34157341718673706, -0.8993135094642639 , -0.08588892221450806,
   0.8226354122161865 ,  0.15130256116390228, -0.7761858105659485 ,
  -2.5481350421905518 , -0.02920402586460114, -0.5006338953971863 ,
  -0.12292736768722534,  0.10984406620264053, -0.43971264362335205,
  -0.02936470322310925, -0.4974880516529083 , -1.0300854444503784 ,
  -1.0226975679397583 , -0.9361142516136169 ,  0.5211664438247681 ,
   0.35920968651771545,  0.6249710321426392 , -0.38503530621528625,
   0.1989656239748001 , -0.08911234140396118,  0.22903211414813995,
  -0.22370348870754242, -0.37220650911331177,  0.3425965905189514 ,
  -0.03949884325265884,  0.5302063226699829 ,  0.6353657245635986 ,
  -0.0041153053753078 , -1.9818572998046875 ,  1.0314491987228394 ,
  -0.18895578384399414,  0.13938167691230774,  0.1465466320514679 ,
  -0.07216484844684601,  0.17936891317367554,  0.23105639219284058,
   0.23669716715812683,  0.06090499460697174, -1.140652060508728  ,
   0.3317123353481293 ,  0.6456575393676758 ,  0.5064748525619507 ,
  -1.3407566547393799 , -0.06221407279372215, -0.07700072973966599,
   0.10014183819293976],
 [-0.5154170989990234 ,  0.7913806438446045 , -0.4493582844734192 ,
  -0.6140702366828918 ,  0.7468339800834656 , -0.20158620178699493,
   0.01528466865420341, -0.8274377584457397 , -0.4257288873195648 ,
   1.351493239402771  , -1.3323469161987305 ,  0.3188796043395996 ,
  -0.27155640721321106,  0.3321416676044464 ,  0.480802446603775  ,
   0.04803553968667984,  0.7981991171836853 , -0.7257723808288574 ,
  -0.1637239158153534 , -0.05074496939778328, -0.37463080883026123,
  -0.2372414469718933 , -0.21062050759792328,  0.20333430171012878,
  -0.13238203525543213, -0.7423973679542542 , -0.18946541845798492,
  -0.2731546461582184 , -0.32993394136428833, -0.19879034161567688,
   0.06974304467439651,  0.4019618034362793 , -1.1226474046707153 ,
  -0.5034648180007935 ,  0.02688837051391602,  0.49613404273986816,
   0.02439425513148308, -0.04193518310785294, -0.44584354758262634,
   1.013715147972107  , -0.4812556505203247 ,  1.1082007884979248 ,
   0.01633632555603981, -1.2148882150650024 ,  0.20886947214603424,
  -0.5289985537528992 , -0.35439643263816833, -1.2305169105529785 ,
   0.03985326737165451,  0.46848902106285095,  0.2054894119501114 ,
  -1.1289175748825073 ,  0.14296211302280426, -0.8789153099060059 ,
   0.9051245450973511 , -0.9893232583999634 ,  0.32526254653930664,
   0.2757438123226166 ,  0.00805780943483114, -0.754123866558075  ,
  -0.4127660393714905 , -1.3204295635223389 , -0.6167322993278503 ,
   0.08497246354818344],
 [ 0.09689627587795258, -0.1740228831768036 ,  0.2335730344057083 ,
  -0.02312812767922878, -0.36181360483169556,  0.3276578187942505 ,
  -2.7785959243774414 ,  0.3103542923927307 ,  0.16235101222991943,
  -0.1556129902601242 , -2.891677141189575  , -0.5919978022575378 ,
   0.14555685222148895,  0.4793204367160797 , -0.7849314212799072 ,
  -1.686241626739502  , -1.3908222913742065 , -0.8376486897468567 ,
  -0.36388835310935974, -0.07720710337162018, -1.3870052099227905 ,
  -0.24536018073558807, -2.533860206604004  , -0.12381098419427872,
  -0.05951788648962975,  0.14073435962200165, -0.14531469345092773,
   0.1682136207818985 , -0.12434674799442291, -2.0369882583618164 ,
   0.9574061632156372 , -0.11337754875421524,  0.6980353593826294 ,
  -1.8869742155075073 , -1.8954174518585205 ,  0.4110460877418518 ,
  -1.9153215885162354 , -0.6700131893157959 , -1.865774393081665  ,
  -0.39157629013061523,  0.6898582577705383 , -0.11113210022449493,
   0.1362456977367401 ,  0.17512153089046478, -0.19125597178936005,
  -0.6841604113578796 ,  0.16817113757133484, -0.02204330451786518,
  -0.9577671885490417 , -0.9291756749153137 , -0.5208778977394104 ,
   0.42742446064949036,  0.32565516233444214, -1.0424937009811401 ,
  -0.24156798422336578, -0.73870450258255   , -0.6243755221366882 ,
   0.2849295139312744 , -1.0999399423599243 , -0.28741031885147095,
  -0.07064203917980194,  0.17037807404994965,  1.0240052938461304 ,
   0.04801413044333458],
 [ 0.28119364380836487,  0.12222906202077866,  0.3224826455116272 ,
   0.3129001557826996 ,  0.5447745323181152 , -0.05345340445637703,
  -1.1216095685958862 ,  0.9129195809364319 , -0.30931761860847473,
  -1.332618236541748  , -0.5718824863433838 , -1.0505889654159546 ,
  -0.8433486819267273 ,  0.00074095069430768, -0.2535730004310608 ,
  -0.20882859826087952, -1.606760859489441  , -0.7465646266937256 ,
  -2.3028364181518555 , -0.1594298928976059 , -0.04881161823868752,
  -0.11478406190872192, -0.43474581837654114,  0.2936845123767853 ,
  -0.08439619094133377,  0.07155729085206985, -0.2736770510673523 ,
  -0.00233562500216067, -0.46938276290893555, -0.7834092378616333 ,
   0.3993619978427887 ,  0.39930856227874756,  0.4336860477924347 ,
  -0.05697396397590637, -0.8547220230102539 ,  0.22369162738323212,
   0.00288723595440388, -0.05386920273303986, -0.4410329759120941 ,
  -0.5215097665786743 , -0.96653151512146   , -0.38244006037712097,
   0.2300180345773697 , -0.18589355051517487,  0.35962629318237305,
  -0.33106884360313416, -0.31261178851127625, -0.3314119875431061 ,
  -0.29196596145629883, -0.3814729154109955 ,  0.16782118380069733,
  -0.4796004891395569 ,  0.05106579884886742,  0.19976688921451569,
   0.3951072096824646 , -0.298126220703125  , -0.8607268929481506 ,
   0.4762938320636749 ,  0.7960038185119629 , -0.2626635730266571 ,
   0.2209283709526062 ,  0.10887943208217621, -0.9172437191009521 ,
   0.07613053917884827],
 [ 0.01255633868277073,  0.04032186046242714, -0.2491314858198166 ,
  -0.7419150471687317 ,  0.9281376600265503 , -0.2927555441856384 ,
  -0.7083467245101929 ,  0.2450900822877884 , -0.3850858807563782 ,
   0.43490099906921387, -0.9704322218894958 ,  0.10330867767333984,
  -0.2201913446187973 , -1.4552521705627441 ,  0.3229333460330963 ,
   0.0768074169754982 , -0.40871715545654297, -0.13030892610549927,
   0.16793066263198853,  0.49419933557510376, -0.22113342583179474,
   0.5385610461235046 , -1.2143430709838867 ,  0.3758328855037689 ,
  -0.6665422916412354 , -0.36962324380874634, -0.49878838658332825,
   1.128358244895935  , -2.1234211921691895 , -0.05208517238497734,
  -0.1210188940167427 ,  0.04504650458693504, -4.017436981201172  ,
  -0.4460243880748749 , -0.13953904807567596, -0.16086366772651672,
  -0.8761033415794373 ,  0.35263100266456604,  0.6133010387420654 ,
  -0.5504170060157776 , -0.04583117738366127,  0.28715085983276367,
   0.7854763269424438 , -0.896589994430542  , -0.21619722247123718,
  -1.5747487545013428 , -0.08117017894983292, -0.14151330292224884,
  -0.8328667879104614 , -0.4425864517688751 ,  1.401063084602356  ,
  -2.061116933822632  , -0.1464672088623047 , -0.04532518237829208,
  -0.47570285201072693, -2.0210554599761963 ,  0.36475473642349243,
  -0.20322845876216888,  0.5227704048156738 , -0.6129363775253296 ,
   1.0095666646957397 , -0.46683552861213684, -0.3238071799278259 ,
  -0.8423170447349548 ],
 [ 0.9893857836723328 ,  0.7650448679924011 , -0.9825548529624939 ,
   0.00061926350463182,  1.2461577653884888 , -0.6923843622207642 ,
   0.9784142971038818 ,  1.3005367517471313 , -0.3041764497756958 ,
  -1.042069673538208  ,  1.1490228176116943 ,  1.5906552076339722 ,
   1.4406099319458008 ,  1.1549705266952515 , -0.34057286381721497,
   1.9689396619796753 ,  0.7372153997421265 ,  0.9747543931007385 ,
   0.5451166033744812 ,  0.5669801831245422 , -0.3044869005680084 ,
  -0.9826028943061829 ,  2.606851100921631  , -1.1523401737213135 ,
  -1.8021013736724854 , -0.03764789924025536, -0.25396785140037537,
  -0.39486926794052124, -0.3227006793022156 ,  1.137940764427185  ,
   3.156045913696289  ,  1.3842347860336304 , -0.05238286778330803,
  -0.00540811289101839, -1.0624268054962158 ,  1.6992813348770142 ,
   1.630741000175476  , -0.6990851163864136 ,  0.7822716236114502 ,
   2.022019863128662  ,  1.0477352142333984 , -0.5972468852996826 ,
   0.9742261171340942 ,  0.6811696887016296 ,  2.322611093521118  ,
  -0.09148713946342468,  1.177381157875061  ,  3.657606363296509  ,
   0.6160019040107727 ,  1.414021372795105  ,  1.0271306037902832 ,
   0.8718364238739014 ,  0.0820808857679367 ,  0.26881474256515503,
   0.8387171030044556 , -0.914177656173706  ,  0.3949572443962097 ,
  -0.24633298814296722,  1.1362007856369019 , -0.19594798982143402,
   0.04307982698082924, -0.4275427758693695 , -0.8864006400108337 ,
   1.209396481513977  ],
 [ 0.7705021500587463 , -0.30332639813423157, -0.5136097073554993 ,
  -0.12385527789592743, -0.07049912214279175,  0.9442946910858154 ,
  -0.42177242040634155,  0.05466863885521889, -0.10635212063789368,
  -0.4521983861923218 , -1.2380635738372803 ,  1.4476476907730103 ,
   1.313697338104248  , -0.5640456080436707 ,  0.8514884114265442 ,
   0.8960890769958496 ,  0.9064497351646423 ,  0.9895383715629578 ,
  -0.7195128798484802 ,  0.21148818731307983,  0.2434099018573761 ,
  -2.5396571159362793 ,  0.7187477946281433 , -1.2891346216201782 ,
  -2.1066372394561768 ,  0.10334468632936478,  0.5959601998329163 ,
   1.8362507820129395 ,  0.30061131715774536,  0.15927274525165558,
   2.4059863090515137 ,  0.4673699140548706 ,  0.3887985944747925 ,
   0.16416846215724945,  0.29391324520111084,  0.4323391318321228 ,
  -1.4113456010818481 , -0.7131372094154358 , -0.7031410932540894 ,
   1.0412482023239136 ,  0.6701956391334534 ,  0.17658580839633942,
   2.0004124641418457 , -0.23885592818260193,  1.7165831327438354 ,
  -1.0089850425720215 ,  0.17109796404838562, -2.262570858001709  ,
   0.18589480221271515, -0.821700930595398  , -2.5661213397979736 ,
  -0.25882142782211304,  1.4001846313476562 , -0.7706134915351868 ,
  -0.5076607465744019 ,  0.4174661636352539 ,  0.42296162247657776,
  -0.13141106069087982,  0.18225574493408203, -0.6085196733474731 ,
   0.935883641242981  , -2.2895069122314453 ,  0.15525636076927185,
  -0.4251856505870819 ],
 [ 0.24318496882915497,  0.17314977943897247, -0.03504138439893723,
  -0.10159150511026382,  0.17801140248775482,  0.19802828133106232,
  -0.09567113220691681,  0.25221484899520874,  0.00210661208257079,
  -0.3848394453525543 , -0.38347914814949036, -0.30565953254699707,
  -0.337417334318161  ,  0.05218731984496117,  0.5952480435371399 ,
  -0.28252753615379333, -0.6471759676933289 , -0.5771754384040833 ,
  -0.15703660249710083,  0.0029013364110142 , -0.3616000711917877 ,
  -0.5104014277458191 ,  0.1080758348107338 , -0.2878236174583435 ,
   0.07515452802181244,  0.31953462958335876, -0.1862601488828659 ,
  -0.01598038896918297, -0.40496402978897095,  0.2194758504629135 ,
   0.7349984645843506 ,  0.4210074841976166 ,  0.7047709822654724 ,
  -0.9336028099060059 , -0.14275476336479187,  0.1493028849363327 ,
  -1.7286261320114136 ,  0.0501294806599617 ,  0.3932381868362427 ,
   0.19811858236789703,  0.16590578854084015,  0.01183527894318104,
  -0.1669609397649765 ,  0.234334334731102  , -0.08411198109388351,
  -1.3401871919631958 , -0.3724145293235779 ,  0.5693270564079285 ,
   0.29517826437950134,  0.08585361391305923, -0.8615671992301941 ,
  -0.37989822030067444, -0.44707489013671875,  0.17923052608966827,
   0.15954510867595673,  0.43229541182518005, -0.8920789361000061 ,
   0.09742312878370285,  0.2022927701473236 ,  0.11582665145397186,
   0.0362221747636795 , -0.7413234114646912 ,  0.5623074173927307 ,
   0.30316606163978577],
 [-0.6448280215263367 , -0.3203091621398926 , -0.08362869918346405,
  -1.1455256938934326 ,  0.10440872609615326,  0.31301891803741455,
  -1.3638228178024292 , -2.3310017585754395 ,  1.124968409538269  ,
  -1.6043468713760376 , -0.9711050987243652 , -0.6433256268501282 ,
  -0.31406426429748535, -0.546227753162384  , -1.1891824007034302 ,
   0.559390664100647  , -1.6334173679351807 , -1.1934000253677368 ,
  -0.5121678113937378 ,  0.18794889748096466,  0.0223399493843317 ,
  -1.3532227277755737 , -0.6734330654144287 , -0.9178639054298401 ,
   0.17490312457084656, -0.9851478934288025 , -0.7647866606712341 ,
  -1.949562668800354  , -0.33369967341423035, -0.07430391758680344,
  -1.2378987073898315 , -0.15393684804439545, -1.3095589876174927 ,
  -1.6972591876983643 , -1.7080613374710083 , -1.678287386894226  ,
  -1.1428805589675903 ,  0.22341789305210114, -0.7692015171051025 ,
  -1.3668683767318726 , -0.9518232941627502 , -0.4789676070213318 ,
  -0.11506865918636322, -0.2266080230474472 , -2.5105140209198    ,
  -1.1387972831726074 , -1.8372660875320435 , -0.8668736815452576 ,
  -0.04852287471294403,  0.02627817913889885,  1.3178160190582275 ,
  -0.42420104146003723, -0.1372275948524475 , -2.233647346496582  ,
   0.6724879741668701 , -3.433091878890991  , -0.26268962025642395,
  -0.17542342841625214, -0.1806170493364334 ,  0.1200665831565857 ,
  -0.33261582255363464,  0.2213904708623886 , -0.7558590173721313 ,
  -0.8498769402503967 ],
 [ 0.7731319665908813 , -1.0545016527175903 ,  0.2506398856639862 ,
  -0.06510467827320099,  0.6365010142326355 , -1.0623936653137207 ,
  -0.23266518115997314,  0.2128002941608429 , -0.12575143575668335,
  -0.20476685464382172, -0.3677838146686554 , -0.13239823281764984,
   0.5297876596450806 ,  0.16138306260108948,  0.7078391909599304 ,
  -1.0531604290008545 , -0.49956750869750977,  1.5461530685424805 ,
   0.08548873662948608, -0.523483157157898  , -1.3393524885177612 ,
  -1.0904279947280884 , -0.12012031674385071,  0.1720937043428421 ,
   0.35554108023643494,  0.296068400144577  , -0.2130652219057083 ,
   0.36489221453666687, -0.00540897669270635,  0.10198339074850082,
   0.7983531951904297 , -1.3835395574569702 ,  0.7758529782295227 ,
   0.6769618988037109 ,  0.25785839557647705,  0.8263083100318909 ,
   0.49873265624046326, -1.6234354972839355 , -0.17345231771469116,
   0.3215484023094177 , -0.11797384917736053, -0.08914472907781601,
   0.8931938409805298 , -0.7211316823959351 , -1.1008747816085815 ,
  -0.9901594519615173 , -1.6701165437698364 , -0.08502333611249924,
   0.13212236762046814,  0.3020763397216797 ,  1.4207686185836792 ,
   0.3394014239311218 , -0.999622642993927  ,  0.3486560881137848 ,
   0.5970967411994934 , -0.408515602350235  ,  0.2175055891275406 ,
  -0.16409584879875183, -0.5709840059280396 ,  0.522169291973114  ,
   0.2808009684085846 , -0.0729641392827034 ,  1.0393105745315552 ,
  -0.5982404947280884 ],
 [ 2.2800936698913574 , -0.22431965172290802, -0.6190200448036194 ,
   0.45484036207199097,  0.1037786677479744 ,  0.4634859263896942 ,
   0.7517105937004089 ,  1.175943374633789  , -0.15798619389533997,
   0.24349822103977203, -0.1266881674528122 ,  0.6258758902549744 ,
   0.8440252542495728 ,  0.06919796764850616, -0.5812990665435791 ,
  -0.8274102807044983 ,  0.32576727867126465, -1.3486841917037964 ,
  -0.14166338741779327,  0.31440258026123047,  0.3400499224662781 ,
   0.7491530179977417 ,  0.6763729453086853 , -0.4710523784160614 ,
   0.6577818989753723 ,  0.09874917566776276,  0.3963446617126465 ,
  -0.9204778075218201 ,  0.20107658207416534, -0.2779190242290497 ,
   0.2612963914871216 ,  0.03741981834173203,  1.4473737478256226 ,
  -0.3413855731487274 ,  0.2592075765132904 ,  1.6797358989715576 ,
   0.7291679978370667 , -0.5811728835105896 ,  1.442989468574524  ,
   0.7501232028007507 ,  0.4835689961910248 ,  0.3107554316520691 ,
   0.43208110332489014, -0.4945932626724243 ,  0.5667538642883301 ,
  -0.04398722946643829,  0.45682504773139954, -0.6959276795387268 ,
   0.45448172092437744,  1.0942636728286743 ,  0.7312185764312744 ,
   0.17129701375961304,  0.44132864475250244, -0.45411789417266846,
   0.9792429208755493 , -0.7396741509437561 ,  0.3006596267223358 ,
   0.3014875054359436 ,  0.8604174256324768 ,  0.10801002383232117,
  -1.6953843832015991 ,  0.23958301544189453,  0.6114456653594971 ,
   0.2729034721851349 ],
 [ 0.3847236931324005 ,  0.2710397243499756 , -0.218950554728508  ,
  -0.2936665713787079 ,  0.02631960809230804,  0.4650850296020508 ,
  -1.6356582641601562 ,  0.3721671402454376 , -0.148187518119812  ,
   0.7160027027130127 , -3.545684576034546  , -0.20346786081790924,
  -1.4984174966812134 , -1.9390738010406494 , -0.28938642144203186,
   0.02973700501024723,  0.7627396583557129 , -0.43469274044036865,
  -0.1666315495967865 ,  0.06205595657229424, -0.6049352288246155 ,
  -2.0260190963745117 , -0.5429152250289917 ,  0.00112697714939713,
  -0.751904308795929  , -0.513185441493988  , -0.14731517434120178,
   0.9774659872055054 , -1.1362345218658447 , -0.6279472708702087 ,
   1.1321247816085815 , -0.7251105904579163 ,  0.4957987666130066 ,
   0.2882302403450012 ,  1.5646536350250244 ,  0.26489150524139404,
   0.7248008251190186 , -2.528987169265747  , -1.4143747091293335 ,
   0.02511889673769474, -1.2959262132644653 , -0.1923883557319641 ,
  -0.6371659636497498 , -0.6804527640342712 ,  0.6003968715667725 ,
  -0.35821571946144104, -0.17540760338306427, -0.0486229881644249 ,
  -0.14778001606464386, -0.2752094268798828 , -1.909896731376648  ,
   0.07871776074171066, -0.8063304424285889 ,  0.15638408064842224,
  -0.1806044727563858 ,  0.27214691042900085,  1.4090654850006104 ,
  -0.02772934176027775,  0.0141747435554862 , -0.23933574557304382,
   0.43883591890335083, -0.7202473282814026 , -0.5080456733703613 ,
  -0.04161937534809113],
 [-0.6565064191818237 , -0.13319608569145203, -0.27554064989089966,
   0.13171306252479553,  0.5498462319374084 , -0.5460481643676758 ,
   0.176442489027977  , -2.4465177059173584 , -0.18202659487724304,
  -0.20976565778255463, -0.3045261800289154 , -0.06659875810146332,
   0.2864817976951599 , -0.15519936382770538, -0.34831076860427856,
  -0.14614036679267883, -0.04548141360282898,  0.13960301876068115,
  -0.2897002100944519 ,  0.23786678910255432, -0.5730946660041809 ,
  -0.48392805457115173,  0.19517117738723755,  0.15704847872257233,
   0.2839548885822296 ,  0.2169758528470993 , -0.4227384328842163 ,
  -0.924007773399353  ,  0.02331710234284401,  0.2003856599330902 ,
  -0.211576446890831  ,  0.24236178398132324,  0.47726550698280334,
   0.29384320974349976,  0.06590839475393295,  0.12969054281711578,
   0.2374533712863922 ,  0.10889670252799988, -0.07531149685382843,
   0.07520626485347748,  0.06458582729101181,  0.3916321396827698 ,
  -0.1327137053012848 , -0.6223591566085815 , -0.2210073173046112 ,
   0.4694042205810547 , -0.18181058764457703, -0.352376788854599  ,
   0.2584786117076874 , -0.4477942883968353 , -0.332630455493927  ,
  -0.02140344493091106, -0.3507736921310425 , -0.0215434804558754 ,
  -0.8988351821899414 ,  0.4419757127761841 , -0.05461244657635689,
   0.18622852861881256,  0.02729865908622742,  0.21482111513614655,
   0.02933866530656815,  0.7119860649108887 ,  0.02312242239713669,
   0.17230375111103058],
 [ 1.8858566284179688 , -0.4024635851383209 ,  0.3508901000022888 ,
  -0.12818101048469543,  0.5903149843215942 ,  0.5023086667060852 ,
  -0.21776258945465088,  0.9424142241477966 , -0.2687552869319916 ,
  -0.19219297170639038,  1.4043805599212646 , -0.08584897965192795,
  -0.0815160870552063 ,  0.29411840438842773,  0.11756939440965652,
   0.2743852138519287 , -0.38621875643730164, -1.1675405502319336 ,
   0.4879361093044281 , -0.10469387471675873, -1.9589356184005737 ,
   0.01429861783981323, -0.3529817461967468 ,  0.11209619790315628,
  -0.07986726611852646, -0.7665458917617798 , -0.9200795888900757 ,
   1.4507254362106323 ,  1.2549065351486206 , -0.9389438033103943 ,
   0.6127524971961975 , -0.19549547135829926,  0.263448029756546  ,
   0.0696503147482872 ,  0.2850542366504669 ,  0.08651647716760635,
   0.9109451174736023 , -0.3147736191749573 , -0.27330490946769714,
   0.38161394000053406, -0.05699874088168144, -0.5703742504119873 ,
   0.19124186038970947, -0.12843072414398193, -0.2823053300380707 ,
  -2.8943827152252197 ,  0.45770353078842163, -0.2912362813949585 ,
  -0.6076566576957703 , -0.8089460730552673 , -0.6109238862991333 ,
  -0.48787227272987366, -0.5740473866462708 , -0.10442293435335159,
   0.9877552390098572 ,  1.114804744720459  , -0.14531701803207397,
   0.24036085605621338,  0.6389509439468384 , -0.24749669432640076,
   0.2451554834842682 , -0.5272839665412903 , -0.2172207087278366 ,
  -0.3139244019985199 ],
 [ 1.0302172899246216 ,  0.24001125991344452,  0.2264285832643509 ,
  -0.21269753575325012,  0.01712863519787788, -1.3504058122634888 ,
   0.6291449666023254 ,  0.8072885870933533 ,  0.20751062035560608,
  -0.3195302486419678 , -0.3732241690158844 ,  0.37287214398384094,
   0.35763612389564514, -0.3104933500289917 , -1.5547312498092651 ,
   0.31108567118644714,  0.16716915369033813, -0.03503961861133575,
  -0.7806068062782288 , -0.03254517540335655, -0.18821021914482117,
  -0.28285667300224304, -0.5758788585662842 ,  0.23554006218910217,
   0.15669719874858856, -0.00796730816364288, -0.7089053392410278 ,
   0.38917720317840576,  0.28613603115081787, -0.5544650554656982 ,
  -0.586843729019165  , -0.8728300333023071 ,  0.6535433530807495 ,
   0.02495555952191353,  0.2919948101043701 ,  0.40715616941452026,
   0.9291249513626099 ,  0.17630645632743835, -0.2355576753616333 ,
  -0.7616747617721558 , -0.2980133891105652 , -0.10202379524707794,
  -0.00313438568264246, -0.35580435395240784, -0.5620861649513245 ,
  -3.5761804580688477 ,  0.18212515115737915, -0.16704078018665314,
  -0.6422536969184875 , -0.6427512764930725 ,  0.10690608620643616,
  -0.13572072982788086, -0.14482027292251587, -2.2283577919006348 ,
   0.7845269441604614 ,  0.10816888511180878,  0.44850558042526245,
  -0.20797117054462433,  0.23100052773952484,  0.5135473608970642 ,
  -0.2339596301317215 ,  0.7807360887527466 ,  0.4430842399597168 ,
  -0.21627995371818542],
 [ 0.5370783805847168 , -1.6824352741241455 , -0.08978402614593506,
   0.7142030000686646 , -0.03217406198382378, -0.26435115933418274,
   0.3235096335411072 , -0.01910199224948883,  0.0471135713160038 ,
  -1.013407826423645  , -0.3132511377334595 ,  0.7661972045898438 ,
   0.12279404699802399, -0.4300108850002289 , -0.20196376740932465,
  -0.16268214583396912, -0.1503806859254837 , -0.5594624280929565 ,
   0.5986611843109131 ,  0.12944461405277252, -1.4092787504196167 ,
  -2.2557992935180664 ,  0.5236232280731201 , -0.25608330965042114,
   0.4522572159767151 , -0.2627640664577484 ,  0.180049866437912  ,
   0.4782586395740509 , -1.5555543899536133 , -0.6168093681335449 ,
  -0.8009165525436401 ,  0.6629118919372559 , -0.6972010731697083 ,
  -0.6491705775260925 , -0.20464174449443817, -1.717358946800232  ,
   0.6768534183502197 , -0.12296520918607712,  0.5276054739952087 ,
  -0.625973641872406  , -1.1971603631973267 ,  0.07635924220085144,
   0.14139942824840546, -0.1738900989294052 , -0.9753420352935791 ,
  -1.6565302610397339 , -0.28516948223114014, -0.6510195136070251 ,
   0.24781548976898193,  0.2019243687391281 ,  0.2305312603712082 ,
  -0.7983635067939758 ,  0.06351344287395477, -0.36978158354759216,
  -0.4984423518180847 , -1.734409213066101  , -0.6102741360664368 ,
   0.09058208763599396, -0.1826392114162445 ,  0.17197835445404053,
  -0.48090776801109314, -0.5032418370246887 ,  0.3534591495990753 ,
  -0.2573257088661194 ],
 [ 0.6740372180938721 , -0.3470941185951233 , -0.05133040249347687,
   0.06884053349494934, -0.3406603932380676 ,  0.15181338787078857,
  -0.7851325273513794 , -0.35992223024368286,  0.4006774425506592 ,
   0.9893608093261719 , -0.2911956012248993 , -0.3495033383369446 ,
  -0.9538501501083374 , -0.8279047608375549 , -0.4403635859489441 ,
  -0.06702692806720734, -0.09078820049762726, -1.3824931383132935 ,
   0.38241100311279297,  0.024369727820158  , -0.39108729362487793,
  -1.2185182571411133 , -1.0041104555130005 , -0.2267468124628067 ,
  -0.29698121547698975, -0.3592465817928314 , -0.6123307347297668 ,
   0.23233231902122498,  0.05341026932001114,  0.21222525835037231,
  -0.08150243014097214, -0.27630940079689026,  1.7334598302841187 ,
   0.28762105107307434,  1.591494083404541  ,  0.4377221465110779 ,
  -0.7841780185699463 , -0.53115314245224   ,  0.3577667474746704 ,
  -0.3787100911140442 ,  0.09912897646427155,  0.11890055984258652,
   0.7412323355674744 ,  0.15520156919956207,  0.7532349824905396 ,
  -0.00639868760481477,  0.23055852949619293,  0.9384234547615051 ,
   0.10208943486213684, -0.00944182928651571, -0.13102690875530243,
   0.1816885620355606 ,  0.16332630813121796,  1.1671956777572632 ,
  -0.12749694287776947,  0.04060088843107224,  1.1494348049163818 ,
  -0.10619691759347916,  0.17243683338165283, -0.3054497539997101 ,
   0.09487151354551315,  0.43201929330825806,  0.688861072063446  ,
   0.5032214522361755 ],
 [ 1.442498803138733  ,  0.25529271364212036, -0.1332368552684784 ,
  -0.02439867332577705, -0.5173619985580444 , -1.209046483039856  ,
   1.467801809310913  ,  0.44957855343818665, -0.2793866991996765 ,
   0.16458334028720856,  0.600264310836792  , -0.7992221117019653 ,
  -0.06036745384335518, -0.8885346055030823 ,  0.8145191073417664 ,
  -0.07068680971860886,  0.4672643840312958 ,  1.2205082178115845 ,
  -0.63853520154953   ,  0.05824213847517967,  1.6265805959701538 ,
  -2.596827983856201  , -0.36181357502937317, -0.00710714235901833,
   0.10544886440038681,  1.1742326021194458 ,  0.1266542375087738 ,
  -0.39645621180534363,  0.12059804052114487,  0.8203599452972412 ,
   0.15562762320041656, -0.19333507120609283,  2.2831857204437256 ,
  -0.6711394190788269 ,  0.8855609893798828 ,  0.645450234413147  ,
   2.2482211589813232 ,  0.39287710189819336,  0.7786413431167603 ,
  -0.505077064037323  ,  1.0299363136291504 , -0.46553486585617065,
  -0.05598645284771919, -0.9246762990951538 ,  0.00740867061540484,
  -2.370739698410034  , -0.21363577246665955,  1.6694482564926147 ,
   0.8407057523727417 , -0.1235927939414978 , -0.1545717567205429 ,
   1.0643503665924072 ,  0.11858461797237396, -0.4041379690170288 ,
   1.0558013916015625 ,  0.20958060026168823,  0.00646350393071771,
   0.2635095715522766 ,  0.48682743310928345, -0.06588830798864365,
  -0.28246423602104187,  0.702898383140564  , -1.0366212129592896 ,
  -0.42607712745666504],
 [-2.8825523853302    , -0.19094175100326538,  0.06367340683937073,
  -0.163160502910614  ,  0.34428369998931885, -0.7436003088951111 ,
  -0.9773529171943665 ,  1.1366218328475952 , -0.09354297816753387,
  -0.7504035830497742 ,  0.5414880514144897 ,  0.12708452343940735,
  -0.16525466740131378,  0.03417306020855904, -0.1944521814584732 ,
   0.12791845202445984, -0.9419369101524353 , -0.02860205620527267,
   0.2868722081184387 ,  0.270976722240448  , -0.04096432402729988,
   0.476754367351532  , -0.0630103275179863 ,  0.23336850106716156,
  -0.3133656978607178 , -0.02503575757145882, -0.6366258859634399 ,
   0.06603103876113892,  0.2728022038936615 , -0.4054318070411682 ,
  -0.6241253018379211 ,  0.6973836421966553 ,  0.437603235244751  ,
   0.05472041666507721,  0.05438601598143578,  0.1909192055463791 ,
  -1.4570338726043701 ,  0.17794831097126007,  0.13013949990272522,
   0.3309883773326874 ,  0.1082441657781601 , -0.6047978401184082 ,
   0.01499837357550859,  0.06439785659313202, -0.31226488947868347,
   0.17580628395080566, -0.3394664525985718 , -0.1154521256685257 ,
  -0.5845103859901428 , -0.6859461665153503 , -0.4370575547218323 ,
   0.16872631013393402,  0.2380850613117218 ,  0.41006726026535034,
   1.2579007148742676 ,  0.3308976888656616 , -0.9553440809249878 ,
  -0.0789165198802948 ,  0.5552412867546082 ,  0.01231019850820303,
  -0.37542325258255005, -0.06582470238208771, -0.1631651371717453 ,
  -0.8041396141052246 ],
 [-0.16490311920642853,  0.00078860786743462, -0.0767536610364914 ,
  -0.16493383049964905, -0.15776632726192474,  0.19294625520706177,
  -0.24535173177719116, -0.4377550184726715 ,  0.18658213317394257,
  -0.14843127131462097, -0.02416929230093956, -0.15786480903625488,
  -0.5016524791717529 ,  0.16096509993076324, -0.9832790493965149 ,
  -0.7307379245758057 , -0.8782190084457397 , -0.17583760619163513,
  -0.18413564562797546, -0.23566150665283203, -0.03056031838059425,
  -0.15242408215999603, -1.275138258934021  , -0.2481766939163208 ,
   0.15758825838565826, -0.05598937347531319,  0.29512134194374084,
  -0.08220367133617401,  0.30636540055274963, -0.5933467745780945 ,
  -0.4713727533817291 , -0.6002998352050781 ,  0.15313464403152466,
   0.12382752448320389, -1.6668879985809326 , -0.11925511062145233,
  -0.48139095306396484, -0.18650232255458832,  0.1725953221321106 ,
  -0.16702373325824738, -0.8805697560310364 ,  0.1751781851053238 ,
  -0.17208312451839447, -0.17080563306808472, -0.11275576800107956,
  -0.03879767656326294,  0.13135750591754913, -1.5549060106277466 ,
  -1.915982961654663  , -0.2708694338798523 ,  0.4380892217159271 ,
  -0.0323980338871479 , -0.3276459276676178 , -0.5500907301902771 ,
  -0.4906531572341919 , -1.097906470298767  , -0.21892257034778595,
  -0.08485911786556244,  0.1109456866979599 ,  0.1556571125984192 ,
  -0.5891944169998169 , -0.15685120224952698,  0.03028502687811852,
   0.3447049856185913 ],
 [ 1.584956169128418  , -0.2570423483848572 , -0.19139941036701202,
   0.8776245713233948 , -0.3192664384841919 , -0.8484392762184143 ,
   1.3535805940628052 ,  1.0733765363693237 ,  0.2526368200778961 ,
  -0.31937870383262634, -1.0202964544296265 , -0.1622055321931839 ,
  -0.06413575261831284, -0.6259058713912964 , -0.03701173886656761,
  -0.5763760805130005 ,  0.4960174560546875 ,  1.1966685056686401 ,
  -1.2190423011779785 , -0.2721357047557831 , -0.6744059920310974 ,
  -1.5467413663864136 , -0.4029218256473541 ,  0.096660315990448  ,
  -0.41316935420036316,  0.11065017431974411,  0.00607037218287587,
   2.914714813232422  ,  0.10017679631710052, -1.0718655586242676 ,
   1.0544400215148926 , -0.4798886179924011 ,  2.372823715209961  ,
  -0.527894914150238  ,  0.02256504260003567,  0.5025666356086731 ,
  -0.622272253036499  , -0.04980552196502686, -0.07968287169933319,
   0.45719537138938904,  1.416293740272522  ,  0.4344910979270935 ,
   1.2808609008789062 , -0.3085825443267822 , -0.7305008172988892 ,
  -0.8164471983909607 ,  0.01615038141608238,  2.4064619541168213 ,
  -0.8787596225738525 , -0.7077184915542603 ,  0.6653898358345032 ,
  -0.7935464978218079 , -0.06413387507200241, -0.9125622510910034 ,
   1.3776280879974365 ,  0.5023110508918762 , -0.4480465352535248 ,
  -0.32617849111557007,  0.06059804558753967, -0.21707072854042053,
   0.7453439235687256 , -0.2126855105161667 , -0.3394329249858856 ,
   0.7132474184036255 ],
 [-0.5975468158721924 ,  0.39656171202659607,  0.01490967348217964,
   0.2611112594604492 ,  0.11331617832183838, -1.154456615447998  ,
  -0.38792696595191956, -0.20933178067207336, -0.2879525125026703 ,
   0.03900426626205444, -0.18080219626426697,  0.13848459720611572,
   0.35300973057746887, -0.05033111199736595,  0.29287880659103394,
  -0.03606797754764557, -0.17684747278690338, -0.6957410573959351 ,
  -0.01458675041794777, -0.15585483610630035, -0.4506966471672058 ,
  -0.3320469260215759 ,  0.6893312335014343 , -0.17791397869586945,
  -0.5366256833076477 , -0.6615105867385864 , -0.11203030496835709,
  -0.19099244475364685, -0.04997729510068893,  0.21098990738391876,
  -0.23436227440834045,  0.01932207867503166, -0.6707532405853271 ,
  -0.8887559771537781 ,  0.02586876042187214, -0.34787222743034363,
  -1.593102216720581  ,  0.16692499816417694, -2.0645761489868164 ,
  -0.1629481017589569 , -0.4326145052909851 ,  0.1061328575015068 ,
  -0.921409547328949  , -0.04332533106207848, -0.8816578388214111 ,
  -1.9516242742538452 , -0.3509974479675293 , -0.42938488721847534,
   0.35424572229385376, -0.04586716368794441, -0.3587379455566406 ,
   0.00687438994646072,  0.30976638197898865,  0.35224097967147827,
  -0.6429986953735352 ,  0.2325928658246994 ,  0.23455725610256195,
   0.127128005027771  ,  0.03439841791987419, -0.3049980103969574 ,
   0.27748140692710876, -1.2059311866760254 ,  0.03813888505101204,
  -0.09473436325788498],
 [ 1.0208241939544678 , -2.134413003921509  ,  0.08556823432445526,
  -0.32986265420913696, -0.6365699172019958 ,  0.5752544403076172 ,
   0.18962900340557098,  1.1014214754104614 , -0.24948427081108093,
   0.06483400613069534,  0.3249701261520386 ,  0.950748860836029  ,
  -0.07354681193828583,  0.09305671602487564, -2.114696502685547  ,
  -0.23371486365795135, -0.24328382313251495, -0.5432085394859314 ,
  -0.7047936916351318 , -0.11778862029314041,  1.1042245626449585 ,
   0.32732707262039185, -0.9659863114356995 , -0.15405550599098206,
  -0.41183316707611084,  0.36820581555366516, -0.41150447726249695,
   0.26298171281814575,  0.5059928894042969 ,  0.8564273118972778 ,
   0.4293881058692932 , -0.15744677186012268,  1.0629686117172241 ,
   0.34275850653648376, -0.426267147064209  , -1.341713786125183  ,
  -0.43264713883399963,  1.1012049913406372 , -2.040870428085327  ,
  -1.6829782724380493 , -0.7751925587654114 ,  0.29119858145713806,
   0.18721383810043335,  0.12608136236667633, -0.1358286589384079 ,
   1.099310040473938  , -0.7365776896476746 ,  1.2727842330932617 ,
  -2.780231237411499  ,  0.33911001682281494,  0.26080459356307983,
   0.237155482172966  , -0.971135139465332  , -0.3647337257862091 ,
   0.09140222519636154, -1.4161128997802734 ,  1.1031372547149658 ,
  -0.09165865927934647, -0.2189881056547165 ,  0.26778003573417664,
   0.7087180614471436 , -0.22441522777080536, -0.5232678055763245 ,
  -0.16624923050403595],
 [ 0.294817179441452  , -0.46984779834747314,  0.09743763506412506,
  -0.874444842338562  , -0.00524722691625357, -0.06784199178218842,
   0.26442888379096985, -0.24007044732570648,  0.01125451922416687,
  -1.0692925453186035 ,  0.06611216068267822,  0.00504950666800141,
   0.0626809373497963 , -0.5286616683006287 , -0.336220920085907  ,
  -0.4107430875301361 ,  0.2393663078546524 , -0.15556761622428894,
   0.2107367217540741 , -0.1973980814218521 ,  0.2742806077003479 ,
  -0.2499788999557495 , -1.2095441818237305 , -0.46040719747543335,
   0.14185753464698792, -0.4177823066711426 , -0.09534992277622223,
  -0.00939714256674051,  0.1776571422815323 , -0.47664743661880493,
  -0.3527785539627075 ,  0.03032398782670498,  0.27891889214515686,
  -0.20050181448459625, -1.1623178720474243 ,  0.09670940786600113,
   0.5603493452072144 , -0.22178857028484344,  0.07150173932313919,
  -0.07451381534337997,  0.3083014190196991 , -0.04110947623848915,
   0.025341821834445  , -0.18557099997997284, -0.14699043333530426,
   0.16096583008766174,  0.15634344518184662,  0.38994765281677246,
   0.4556484818458557 , -0.7269687056541443 , -0.429731547832489  ,
   0.12669768929481506,  0.30857014656066895, -1.707749366760254  ,
   0.3484010398387909 ,  0.2434549182653427 ,  0.1703399270772934 ,
  -0.06744808703660965, -0.11465776711702347,  0.23460660874843597,
   0.6613669395446777 , -0.01953076384961605, -0.06648025661706924,
  -0.1657591462135315 ],
 [ 1.4737746715545654 ,  0.1559423804283142 ,  0.5252200365066528 ,
  -2.2381110191345215 ,  0.2005685567855835 ,  1.935142993927002  ,
   0.8283261656761169 , -0.5095075368881226 ,  0.06819970160722733,
  -0.25295400619506836, -0.25112590193748474,  0.8012629151344299 ,
   1.1009283065795898 ,  0.35708168148994446, -0.30718567967414856,
   0.7829785943031311 , -0.1286269724369049 ,  0.7034790515899658 ,
  -0.7789271473884583 , -0.24559614062309265, -0.4957573413848877 ,
   0.19194497168064117, -0.7493654489517212 , -0.07789763808250427,
  -0.12990009784698486,  0.23689448833465576, -0.26026588678359985,
   0.21023717522621155, -0.7434295415878296 , -2.3618898391723633 ,
   0.9630292654037476 , -3.0186307430267334 ,  1.921862244606018  ,
  -0.755381166934967  , -0.32679420709609985, -0.5012631416320801 ,
   0.2343134731054306 ,  0.20902635157108307,  1.9570063352584839 ,
  -0.6883777379989624 ,  0.2696620523929596 , -0.6400269269943237 ,
  -1.080064296722412  ,  0.25379833579063416, -1.5805459022521973 ,
   0.5807119011878967 , -0.44362473487854004,  0.00209407415241003,
  -1.2958163022994995 ,  0.872277021408081  , -2.8870391845703125 ,
  -0.03253810480237007, -0.9274380803108215 , -1.264763593673706  ,
   1.2852792739868164 , -0.40780773758888245,  0.02566708624362946,
   0.05736269429326057,  0.615386426448822  , -1.2967631816864014 ,
   0.05530859157443047, -0.39453601837158203,  0.06126946955919266,
  -0.04576676711440086],
 [ 0.4018109142780304 , -0.10883230715990067, -0.04181282222270966,
   0.17032867670059204,  0.19128593802452087,  0.33147865533828735,
   0.13222765922546387,  0.21444763243198395, -0.3737642168998718 ,
   1.0824658870697021 ,  0.5853314995765686 , -0.5756622552871704 ,
   0.6880092620849609 , -0.07088707387447357,  0.233900249004364  ,
   0.064405158162117  ,  0.49365171790122986, -0.16083495318889618,
  -0.24501466751098633,  0.500643253326416  ,  0.32706621289253235,
   1.087223768234253  , -0.26023590564727783,  0.11666030436754227,
  -1.6020393371582031 ,  0.11866445094347   , -0.3404727578163147 ,
   0.9904653429985046 , -0.23599877953529358, -1.0336023569107056 ,
   1.7713526487350464 ,  0.3175981342792511 , -0.00669837510213256,
  -0.318367600440979  , -2.489254951477051  ,  0.8717373013496399 ,
  -0.7575501799583435 , -0.0122699523344636 ,  0.26657265424728394,
   1.0157139301300049 ,  0.5514222979545593 , -0.3244503438472748 ,
   0.49223101139068604, -0.2719821333885193 ,  1.1707425117492676 ,
   0.03203727677464485,  0.6291664838790894 , -0.6606655120849609 ,
  -0.02145795151591301,  0.8353181481361389 , -0.06816864013671875,
  -0.20112808048725128, -0.6910499334335327 , -0.17313078045845032,
   1.8907171487808228 , -0.21196375787258148, -0.3825260400772095 ,
   0.3239361047744751 , -0.33072730898857117, -0.8587192893028259 ,
   0.05051953718066216,  0.2384684830904007 , -0.00409912830218673,
  -0.7055338621139526 ],
 [-0.6673198938369751 , -0.6717602610588074 ,  0.982412576675415  ,
  -0.4793328642845154 , -0.7733451128005981 , -0.33231890201568604,
  -0.5845600962638855 ,  0.451009601354599  ,  0.01105483807623386,
  -0.12022196501493454,  0.85817950963974   , -0.4046190083026886 ,
  -0.3833184242248535 , -2.3846435546875    , -0.5530111789703369 ,
  -0.635861873626709  , -0.5482252836227417 ,  0.4466871917247772 ,
   0.47133827209472656, -0.4898128807544708 ,  0.07505743950605392,
   0.7585088610649109 ,  0.13101054728031158, -1.357229232788086  ,
   1.0501912832260132 , -0.15708525478839874, -0.09656045585870743,
  -0.5823308229446411 ,  0.820214569568634  ,  0.05276443809270859,
   0.6361207365989685 , -0.8321114182472229 ,  0.18368613719940186,
   1.0279266834259033 , -0.09760354459285736,  0.7762640714645386 ,
  -1.0464361906051636 ,  0.7965219616889954 , -0.2753628194332123 ,
  -0.5205329656600952 , -0.37797465920448303, -0.5349858403205872 ,
   0.5465285778045654 , -0.3034643232822418 , -0.18629074096679688,
   0.5317734479904175 , -0.8567624688148499 , -0.07809527963399887,
  -1.410070538520813  , -0.37722015380859375, -1.3228600025177002 ,
   0.06496696174144745,  0.00953942351043224, -0.3888869881629944 ,
   0.28877881169319153, -0.6808614134788513 ,  0.14866076409816742,
   0.676673173904419  ,  0.3190498948097229 , -0.1673700511455536 ,
   0.19061428308486938, -0.15345239639282227, -0.04334054514765739,
   0.342500239610672  ],
 [ 0.4212959408760071 , -0.5114826560020447 ,  0.03603143617510796,
  -0.00738494005054235, -0.06652393937110901, -0.2239709049463272 ,
  -0.7938625812530518 , -0.391410231590271  ,  0.27411627769470215,
  -0.7159269452095032 , -2.690441131591797  , -0.26411372423171997,
   1.0311135053634644 , -1.2760125398635864 , -0.5232806205749512 ,
   0.07572762668132782,  0.05193272978067398,  0.47383829951286316,
   0.6535775065422058 , -0.8260158896446228 , -0.6356877088546753 ,
  -2.838278293609619  ,  0.6389244198799133 ,  0.09779851138591766,
  -0.10498311370611191,  0.22876206040382385, -0.3304784595966339 ,
  -0.3595111072063446 , -0.16220630705356598, -1.5983762741088867 ,
   0.7341726422309875 , -0.6472387313842773 ,  0.0114434165880084 ,
   1.3986653089523315 , -0.9273682236671448 , -2.1729469299316406 ,
  -1.1799876689910889 , -0.544129490852356  , -0.12533161044120789,
  -0.13162128627300262,  0.7972316145896912 ,  0.11707966774702072,
  -0.7522426247596741 ,  0.01005023531615734,  0.7550573945045471 ,
  -1.9844967126846313 , -0.938060462474823  ,  0.6878446936607361 ,
   0.6235601902008057 , -0.6197659373283386 ,  0.04968472197651863,
   0.13516663014888763, -2.7360711097717285 , -1.4314308166503906 ,
  -0.21109162271022797, -1.5844422578811646 , -1.3533183336257935 ,
  -0.22425708174705505, -0.9646106958389282 , -0.0901113823056221 ,
  -0.9269328713417053 , -0.6868332028388977 , -0.3114110827445984 ,
   0.30337071418762207],
 [ 0.7406845688819885 , -0.22387029230594635, -0.27688512206077576,
  -0.10342389345169067, -0.5554310083389282 , -0.29808858036994934,
  -0.11597559601068497,  0.21343614161014557, -0.3175821900367737 ,
  -0.1752467155456543 , -0.1805039942264557 , -0.7035736441612244 ,
   0.18554042279720306,  0.01543180737644434, -0.4594350755214691 ,
  -0.26569586992263794, -0.6072037220001221 , -0.6769778728485107 ,
  -0.18230374157428741, -0.36354583501815796,  0.37568074464797974,
  -0.02214916981756687, -0.5354169607162476 , -0.13543066382408142,
   0.23715835809707642, -0.496908038854599  , -0.18757131695747375,
   0.44862836599349976, -0.3147643208503723 ,  0.06925349682569504,
   0.22495806217193604, -0.7483575940132141 , -0.1515769064426422 ,
  -0.14856480062007904,  0.19541634619235992,  0.04881915077567101,
  -0.2362368255853653 , -0.8655146360397339 ,  0.19773170351982117,
   0.3798757791519165 ,  0.6925060153007507 , -0.12814311683177948,
   0.39162635803222656, -0.268526554107666  ,  0.00063590961508453,
  -0.15902653336524963, -0.2733519673347473 , -0.08133885264396667,
   0.0309642031788826 , -0.22548967599868774, -0.9412405490875244 ,
   0.18229444324970245, -0.43935123085975647, -1.1713958978652954 ,
  -0.6191070079803467 , -0.3471395969390869 , -0.11952506750822067,
   0.4175426959991455 ,  0.13584202527999878, -0.7609566450119019 ,
  -0.3580484688282013 , -1.6323752403259277 ,  0.6609157919883728 ,
  -1.5270450115203857 ],
 [ 1.0490994453430176 ,  0.07226988673210144,  0.2777504026889801 ,
   0.10417772084474564,  0.2523348927497864 ,  1.7438170909881592 ,
   0.48225292563438416,  0.7492998838424683 ,  0.03100274316966534,
   1.112412452697754  ,  0.6328977942466736 , -0.21644577383995056,
  -0.7981530427932739 ,  0.6650348901748657 , -0.0566711500287056 ,
   0.03310061246156693,  2.0544681549072266 ,  0.39015164971351624,
   0.01193596795201302,  0.31998705863952637,  0.10132217407226562,
   0.35120880603790283, -0.6867809295654297 , -0.3231595456600189 ,
  -0.2473830282688141 ,  0.0172086376696825 ,  0.4639971852302551 ,
   1.3902711868286133 , -0.6773390173912048 , -0.6851471066474915 ,
   0.39616069197654724, -0.04078678041696548, -0.22616465389728546,
   0.3387852609157562 , -0.7415799498558044 , -0.2971932590007782 ,
   2.0618159770965576 ,  0.6350754499435425 ,  1.2394558191299438 ,
  -0.01907286420464516, -0.39789777994155884,  0.25762584805488586,
   0.3284884989261627 , -1.1859418153762817 , -0.7227246761322021 ,
  -2.3626549243927    ,  0.509002685546875  ,  0.6213353872299194 ,
  -0.1075504794716835 ,  0.7417420744895935 ,  1.979732632637024  ,
  -0.21073931455612183, -0.5556584596633911 ,  0.6652873158454895 ,
   0.17061495780944824,  1.3696187734603882 ,  0.2702604830265045 ,
  -0.085417240858078  , -0.05548891052603722, -0.43306708335876465,
   0.36308762431144714,  0.7152872085571289 , -0.45848098397254944,
   0.1529679298400879 ],
 [ 0.15677198767662048, -0.41625234484672546, -0.1839546114206314 ,
   0.04195538908243179, -0.44812288880348206, -1.3963490724563599 ,
   0.593834400177002  ,  0.02665521204471588, -0.05222285911440849,
  -0.8252953886985779 , -0.6063142418861389 ,  0.6157016158103943 ,
  -0.14034585654735565, -1.170885443687439  , -0.5554348826408386 ,
  -0.45314157009124756,  0.46011993288993835, -1.6745586395263672 ,
  -0.4078325629234314 ,  0.3298706114292145 , -0.19742214679718018,
  -0.3185490071773529 , -0.44514989852905273, -0.19512563943862915,
  -0.4088486433029175 , -0.4767613410949707 , -0.5712795853614807 ,
  -0.20753340423107147,  0.459175705909729  ,  0.18983778357505798,
   0.42447519302368164,  0.7965575456619263 , -1.339408278465271  ,
  -0.28908392786979675,  1.3890167474746704 , -0.05859446153044701,
   0.7810966968536377 , -0.6508806347846985 , -0.40251439809799194,
  -0.04664834588766098,  0.41079798340797424,  0.3448250889778137 ,
  -0.42205682396888733, -0.0598648302257061 , -1.1763955354690552 ,
   0.1173962876200676 , -1.5040472745895386 , -0.32255882024765015,
  -0.08426734060049057, -0.5445273518562317 , -0.8595100045204163 ,
  -0.04730656743049622,  0.6378061771392822 ,  0.8646774291992188 ,
   0.6158125400543213 ,  0.8289506435394287 ,  0.30526983737945557,
   0.37732994556427   ,  0.37569743394851685, -0.15260794758796692,
  -0.45351293683052063,  0.6615410447120667 , -0.14034834504127502,
  -1.0336620807647705 ],
 [ 0.3940385580062866 ,  0.1441870629787445 , -0.2471887171268463 ,
   0.37360796332359314, -0.03418188169598579,  0.5585796236991882 ,
  -0.378673791885376  ,  0.7565131783485413 ,  0.14308691024780273,
  -1.2651923894882202 , -0.4453449547290802 ,  0.43997907638549805,
   1.0221202373504639 ,  0.2997437119483948 , -0.26841315627098083,
  -0.3008630871772766 ,  0.6100821495056152 , -0.03446308150887489,
  -2.2246687412261963 , -0.4133952558040619 ,  0.24164554476737976,
  -0.7936997413635254 , -1.5880697965621948 , -0.36219215393066406,
   0.49052175879478455,  0.3037363886833191 , -0.01082373037934303,
   0.5614463090896606 , -0.06635338813066483, -0.02805935963988304,
  -0.44361406564712524, -0.33710938692092896, -1.6712651252746582 ,
   0.5311151146888733 ,  1.2158129215240479 , -1.0036994218826294 ,
  -0.1021309643983841 , -0.6306275725364685 , -0.03503062203526497,
   0.729995846748352  , -1.9709800481796265 ,  0.20114849507808685,
   0.25449830293655396,  0.12071763724088669, -0.4762655198574066 ,
   0.2939003109931946 ,  0.3744484782218933 , -0.3652958273887634 ,
  -0.37620100378990173, -1.742328405380249  , -0.4064854681491852 ,
  -0.5457586646080017 , -0.0801360234618187 , -1.0385910272598267 ,
   0.6710172891616821 ,  0.2666400969028473 , -0.7780312299728394 ,
   0.2502223551273346 , -0.26920193433761597,  0.17306293547153473,
  -0.2790602445602417 , -0.06415659934282303, -0.21493175625801086,
   0.7726688981056213 ],
 [-0.32222649455070496,  0.19334657490253448, -0.10966649651527405,
  -0.1397402137517929 ,  0.20645920932292938,  0.6650540828704834 ,
  -1.444156289100647  , -0.46015262603759766,  0.05280713737010956,
  -1.8287420272827148 ,  0.561806321144104  , -0.5784904360771179 ,
  -3.0223960876464844 ,  0.10234677046537399, -0.06945639103651047,
  -0.07855979353189468, -0.45115286111831665, -0.09308578073978424,
  -0.43463775515556335, -0.01331863086670637, -0.00572432158514857,
   0.6685227155685425 , -0.2191801816225052 ,  0.37373852729797363,
  -1.2245875597000122 ,  0.00314900814555585, -0.14312146604061127,
  -1.2406013011932373 , -1.9876899719238281 , -0.0006444156751968 ,
  -0.7561930418014526 ,  0.23202399909496307, -0.17886152863502502,
  -2.4571640491485596 ,  0.04017835855484009,  0.2817920446395874 ,
   0.5152648687362671 ,  0.4916638433933258 ,  0.4684946537017822 ,
   0.33469027280807495, -0.00810012873262167, -0.03829628229141235,
  -0.34906134009361267,  0.18284496665000916, -0.8807567358016968 ,
  -0.5408328175544739 ,  0.10736015439033508,  0.12116411328315735,
   0.32338008284568787,  0.41782259941101074,  0.29392874240875244,
  -1.3077247142791748 , -0.5186798572540283 ,  0.9064784049987793 ,
   0.37547484040260315,  0.1732335090637207 , -0.37100571393966675,
  -0.1824316680431366 ,  0.17764540016651154, -0.12058946490287781,
  -0.40082305669784546,  0.45478853583335876, -0.4663121998310089 ,
   0.32596659660339355],
 [ 0.25506412982940674,  0.3922903537750244 , -0.26670488715171814,
  -0.0238113459199667 , -0.0889539048075676 ,  1.393304705619812  ,
   0.9718424677848816 ,  1.419016718864441  ,  0.3004917800426483 ,
   0.7351572513580322 ,  0.17444433271884918,  0.153094083070755  ,
  -1.1755505800247192 ,  0.9822710156440735 ,  0.07542584091424942,
   0.8840572834014893 ,  0.15854820609092712, -0.3269670009613037 ,
  -0.5251057744026184 , -0.35892024636268616,  0.10023856908082962,
  -0.17819371819496155, -0.09364878386259079,  0.5616781711578369 ,
   0.41179153323173523,  0.05942919850349426, -0.12222825735807419,
   1.3440732955932617 ,  0.2943488657474518 ,  0.21520456671714783,
   1.2657967805862427 , -0.12949243187904358,  0.38575977087020874,
   1.3966203927993774 , -0.43066954612731934,  0.5526790618896484 ,
  -0.07235304266214371,  0.20984411239624023,  1.711618185043335  ,
   0.7833173871040344 ,  0.29859861731529236,  0.14965657889842987,
   0.03516840562224388, -0.4142671227455139 , -0.00111417809966952,
  -0.01158125046640635,  0.6377665400505066 ,  0.9196634888648987 ,
   0.49157434701919556, -0.36702579259872437,  1.2296888828277588 ,
   0.11146217584609985, -0.4318299889564514 ,  0.35110393166542053,
   0.43123582005500793,  0.04970100894570351,  0.6235262751579285 ,
  -0.2540954053401947 ,  0.4951351284980774 , -0.05197352170944214,
   0.02927514538168907, -0.32622724771499634, -0.1992354840040207 ,
   0.33711060881614685],
 [ 0.00321158207952976, -0.07375247776508331, -0.0319078341126442 ,
   0.42846572399139404, -0.5236401557922363 , -1.4369816780090332 ,
  -1.6493374109268188 ,  0.7130959630012512 ,  0.18349049985408783,
  -3.3480865955352783 ,  0.04112139344215393,  0.6482682824134827 ,
   0.28275734186172485, -1.465744972229004  ,  0.48450493812561035,
   0.2638932168483734 , -0.37407511472702026, -0.520626962184906  ,
   0.920982837677002  ,  0.06734307110309601,  0.4081786274909973 ,
  -2.0739099979400635 , -1.2500762939453125 , -0.08634817600250244,
   0.01520185265690088, -1.9608426094055176 ,  0.4080883860588074 ,
   1.1009440422058105 , -1.2970434427261353 ,  0.43524491786956787,
   0.2325129508972168 , -0.17096851766109467,  1.3685978651046753 ,
   0.4992917478084564 ,  0.2489611655473709 , -0.2227047085762024 ,
  -1.5589325428009033 ,  0.6538752913475037 ,  0.5869461894035339 ,
   1.3415509462356567 ,  0.0040718475356698 ,  0.23818713426589966,
   0.00511611253023148,  0.2225738912820816 , -0.02183784544467926,
  -1.0911542177200317 ,  0.31107133626937866,  1.324142575263977  ,
   0.5839231610298157 ,  0.36938005685806274,  0.8023791909217834 ,
   0.10209807753562927, -0.0578274317085743 ,  0.1903885304927826 ,
  -0.4813591539859772 ,  0.6065121293067932 , -0.34308966994285583,
   0.2955182194709778 ,  0.37448662519454956,  0.05778801441192627,
  -0.124844029545784  ,  0.13112859427928925, -0.6898917555809021 ,
  -0.2937679588794708 ],
 [-1.094817042350769  ,  0.1787930279970169 ,  0.07375765591859818,
   0.08117645978927612,  0.900735080242157  ,  1.5714191198349    ,
  -0.10700935125350952,  2.0163021087646484 , -0.24518713355064392,
  -2.257317304611206  ,  0.5129594206809998 ,  0.17437541484832764,
  -0.5350146293640137 ,  0.2719217538833618 ,  0.2901681065559387 ,
   0.5473072528839111 , -0.49534130096435547,  0.03464329242706299,
  -0.31363725662231445,  0.06492903083562851, -0.26351699233055115,
   0.3449346721172333 , -0.8462242484092712 ,  0.36598873138427734,
   0.19970516860485077, -0.07877510040998459, -0.7070703506469727 ,
  -0.3159199357032776 ,  0.41431334614753723, -0.1379033476114273 ,
   0.4349472224712372 , -0.8068322539329529 ,  0.5937293171882629 ,
  -0.06051740795373917, -0.2779543399810791 ,  0.1205035150051117 ,
   0.583075225353241  , -0.09257551282644272, -0.1496521532535553 ,
  -0.08582299947738647, -0.03583689779043198, -0.6433591842651367 ,
  -0.48184728622436523,  0.8360868692398071 ,  0.00347731774672866,
   0.04505036398768425,  0.19484582543373108,  0.2359572798013687 ,
   0.16601097583770752,  1.3527326583862305 , -2.146273374557495  ,
   0.6348021030426025 ,  0.7201927304267883 ,  0.4219335615634918 ,
  -1.0215626955032349 , -1.7305834293365479 ,  0.09410788863897324,
   0.0531698502600193 ,  0.1835382729768753 , -0.01276026200503111,
   0.2603883743286133 , -0.02767728641629219, -0.3638358414173126 ,
  -0.23629926145076752],
 [ 0.201787069439888  , -0.5064122676849365 , -0.10926355421543121,
   0.5988351106643677 , -0.7007730007171631 , -0.02404863759875298,
   0.25183966755867004,  0.02773375436663628,  0.05630648136138916,
  -0.2029474675655365 , -0.11617512255907059, -1.9743918180465698 ,
   0.4008767604827881 ,  0.2780441343784332 , -2.037954092025757  ,
  -0.11100810021162033, -0.2899631857872009 ,  0.8562217354774475 ,
  -0.8514719605445862 , -0.3292323648929596 , -0.06515472382307053,
  -0.08668947964906693,  0.9847044348716736 , -1.0525513887405396 ,
   0.23896151781082153,  0.37568220496177673, -0.05914853140711784,
   0.5996280908584595 ,  0.7627105712890625 , -0.6103107929229736 ,
   0.4416612386703491 , -0.04142966121435165,  0.37593895196914673,
   0.415303498506546  ,  0.04294154793024063, -0.539921224117279  ,
  -0.07238341867923737, -0.6234583854675293 , -0.08829271048307419,
   0.6211078763008118 ,  0.6133535504341125 ,  0.08020909130573273,
   0.07835454493761063,  0.16404752433300018,  0.6686625480651855 ,
  -0.09155372530221939,  0.2995273470878601 , -0.9775275588035583 ,
  -0.34799429774284363, -0.45101797580718994,  0.31248152256011963,
  -0.29884299635887146,  0.7932313680648804 ,  0.3063993752002716 ,
   0.4171847701072693 , -0.10098573565483093, -1.9431262016296387 ,
   0.04764712229371071,  0.5026278495788574 , -0.1073908880352974 ,
  -2.542473077774048  ,  0.9604416489601135 , -0.2954939603805542 ,
  -0.16890637576580048],
 [-0.15449391305446625, -0.11058535426855087,  0.01563780009746552,
   0.5739429593086243 , -0.09295781701803207,  0.1289770007133484 ,
  -1.3869078159332275 , -0.548893392086029  ,  0.14312134683132172,
  -0.6644428372383118 , -0.40041399002075195, -0.25493600964546204,
  -3.720625877380371  ,  0.05190040543675423,  0.7300410270690918 ,
  -2.8356261253356934 ,  0.725226879119873  , -0.00245286920107901,
  -0.9983705878257751 , -0.14351890981197357, -0.02623044140636921,
  -0.7711219191551208 , -2.5398313999176025 , -0.20195581018924713,
  -0.4173470735549927 , -0.1370658129453659 , -0.0477902889251709 ,
  -0.1102045327425003 , -0.16164422035217285,  0.06303279101848602,
   0.04764112085103989, -0.12806349992752075, -0.08569178730249405,
  -1.3462886810302734 ,  1.6277836561203003 ,  0.05003580823540688,
  -2.082895517349243  , -1.3163610696792603 , -0.4079379141330719 ,
  -0.32327333092689514, -0.0790860503911972 , -0.09440767765045166,
  -0.25337010622024536,  0.06482091546058655, -1.2380917072296143 ,
  -0.25344160199165344, -0.14737802743911743,  0.06745900213718414,
   1.1450481414794922 , -0.5567110180854797 , -0.9802002310752869 ,
  -0.7919420003890991 , -0.3385542035102844 , -0.6168290972709656 ,
  -0.05108897760510445, -0.04888291656970978, -0.2090865969657898 ,
  -0.14369291067123413, -0.11491851508617401, -0.14116109907627106,
   0.3219011723995209 , -1.121903419494629  ,  0.03996909782290459,
   0.3438628017902374 ],
 [-0.4480009377002716 , -0.2425604611635208 , -0.29909083247184753,
   0.05891658738255501,  0.13488495349884033, -0.4351414442062378 ,
  -1.1876499652862549 , -1.9571731090545654 , -0.03464915603399277,
   0.827184796333313  ,  0.2600531578063965 ,  0.22665640711784363,
  -0.5694157481193542 , -1.1308791637420654 , -1.4922351837158203 ,
   0.10692495107650757,  1.0411242246627808 , -1.0947314500808716 ,
  -0.8185696005821228 ,  0.00511306757107377, -0.21471481025218964,
  -0.7445673942565918 , -0.29059556126594543, -0.36010614037513733,
  -0.01756050623953342, -2.2627549171447754 , -2.2815334796905518 ,
  -1.6512348651885986 , -1.045350193977356  , -0.00924655515700579,
  -0.3075985610485077 , -1.056013822555542  , -1.4745339155197144 ,
  -0.00154611060861498, -0.9582290649414062 , -1.758327603340149  ,
  -0.4916090965270996 ,  0.37683936953544617, -0.6922599077224731 ,
   0.756377100944519  ,  0.16009795665740967, -0.29173779487609863,
  -0.02660273760557175, -0.17756769061088562, -1.0211018323898315 ,
  -0.6324779987335205 , -1.636996865272522  ,  0.2225465476512909 ,
   0.03234110772609711,  0.16913945972919464, -0.36628133058547974,
  -0.01248340960592031, -0.49380171298980713, -1.1352084875106812 ,
  -2.0387957096099854 , -0.6164708733558655 ,  0.9024253487586975 ,
  -0.2929500639438629 , -0.15774965286254883, -0.4595310091972351 ,
  -0.19279566407203674, -0.7678123712539673 ,  0.01795831322669983,
  -0.6956009268760681 ],
 [ 0.5322951674461365 , -0.04650098457932472, -0.7058445811271667 ,
  -0.00084504095138982,  0.6357043385505676 ,  0.3665880858898163 ,
  -0.32250648736953735, -0.9427164196968079 , -0.14452502131462097,
   0.22012591361999512,  0.05518373474478722,  0.1336156129837036 ,
   0.25558367371559143,  0.4477880001068115 , -1.9178667068481445 ,
   1.643029808998108  , -0.16707731783390045,  1.9871089458465576 ,
   0.8575168251991272 , -0.28194460272789   ,  0.29071924090385437,
  -0.17031334340572357, -1.9578871726989746 , -0.06664004921913147,
  -0.8073681592941284 , -0.2554433345794678 ,  0.20628440380096436,
  -0.3693586587905884 ,  1.1332060098648071 , -2.45318865776062   ,
   0.09922780841588974, -0.04850136116147041,  2.161979913711548  ,
   0.450522243976593  , -0.3465255796909332 ,  0.10362552106380463,
   0.5940504670143127 , -1.2617233991622925 ,  0.05537991970777512,
   0.5083115100860596 , -0.07972802221775055,  0.11100345849990845,
   1.1255738735198975 ,  0.21288998425006866,  1.4868582487106323 ,
  -1.9190176725387573 ,  0.42221325635910034,  0.6163637638092041 ,
  -1.9163306951522827 , -0.00037513225106522,  1.203693151473999  ,
   0.49937760829925537, -1.6059504747390747 ,  0.3536725640296936 ,
   0.41383764147758484, -0.12325259298086166, -0.4071975350379944 ,
   0.0191090926527977 , -0.07268865406513214,  0.7941312193870544 ,
  -0.35506755113601685, -1.0160000324249268 , -1.1086006164550781 ,
   0.1727912724018097 ],
 [ 0.1600864678621292 , -0.1815912425518036 ,  0.3040582537651062 ,
   0.4901864230632782 ,  0.2153005301952362 , -2.439835548400879  ,
   0.00395953329280019,  0.07318820804357529, -0.172976553440094  ,
   0.5541862845420837 , -0.42517271637916565, -0.10246674716472626,
   0.6843174695968628 , -2.5582168102264404 , -0.8066822290420532 ,
   0.7293782830238342 , -0.03277364745736122,  0.04788511246442795,
  -0.5600452423095703 , -0.2469518631696701 , -0.22748705744743347,
  -1.006960153579712  , -0.0260915718972683 , -0.14522111415863037,
  -0.5339133739471436 , -0.05689246207475662, -2.304084300994873  ,
   0.49094679951667786, -0.46312662959098816, -2.9667916297912598 ,
   0.03007679805159569, -0.6792802810668945 , -0.9332292675971985 ,
  -0.12104267627000809,  0.1861492544412613 , -0.194248765707016  ,
   0.5870066285133362 ,  0.19896391034126282,  0.3707157075405121 ,
  -0.2854209542274475 ,  0.45368146896362305,  0.03869824856519699,
   0.255921334028244  , -0.5513461232185364 , -3.251274585723877  ,
  -3.2441329956054688 , -0.3893294930458069 ,  0.6663291454315186 ,
   0.2842618525028229 ,  0.9386781454086304 , -1.0935226678848267 ,
   0.17721468210220337, -2.2089381217956543 , -1.7680343389511108 ,
  -0.0245837140828371 , -1.5492373704910278 ,  0.6258963942527771 ,
  -0.2916255593299866 ,  0.01335950382053852,  0.07870128750801086,
   0.47190991044044495,  0.2957175672054291 , -0.08169778436422348,
  -0.9680805206298828 ],
 [ 0.5217441916465759 , -0.6422788500785828 ,  0.45372408628463745,
   0.6109561920166016 , -0.9292487502098083 , -1.1247552633285522 ,
   0.02627131156623363,  0.8113452792167664 ,  0.41457465291023254,
  -0.7644748687744141 ,  0.2369643747806549 , -0.9973659515380859 ,
  -0.4590665102005005 ,  0.02126545831561089, -1.1313825845718384 ,
   0.8365993499755859 , -0.4594587981700897 , -0.01599984057247639,
  -1.3747491836547852 , -0.3212716579437256 , -0.2231936752796173 ,
  -0.3262644112110138 , -0.6720741987228394 , -2.391021251678467  ,
  -0.5480072498321533 , -0.45329195261001587, -1.4197241067886353 ,
   0.9644419550895691 ,  0.15627768635749817, -0.067319855093956  ,
   0.5303815603256226 , -0.08240029215812683,  0.1447700560092926 ,
  -0.06444836407899857,  0.6782515645027161 , -0.3460007607936859 ,
   0.07900647819042206, -0.04119286686182022,  0.07786914706230164,
  -0.156584233045578  ,  0.8441490530967712 , -0.33832597732543945,
   0.39922264218330383,  0.31959590315818787, -0.32532915472984314,
  -1.115126132965088  , -1.389430046081543  , -1.2155545949935913 ,
  -1.8933414220809937 , -2.347377061843872  ,  0.10797236114740372,
  -0.24271222949028015,  0.2517946660518646 , -0.3803945779800415 ,
   0.8614020347595215 , -2.0479302406311035 , -0.30464059114456177,
   0.2286447286605835 ,  0.08156126737594604, -0.08718541264533997,
  -3.284956693649292  , -0.06926332414150238,  0.15900593996047974,
  -0.40167999267578125],
 [ 1.290099024772644  , -0.6723331212997437 , -0.23532311618328094,
   0.2584401071071625 , -0.19566978514194489, -0.5569507479667664 ,
   0.7690237164497375 ,  2.0044054985046387 ,  0.12559856474399567,
   0.888964831829071  ,  1.04497230052948   ,  0.17171135544776917,
   0.8406013250350952 ,  0.02511583641171455,  0.5265337824821472 ,
   0.32593151926994324,  0.9967843890190125 ,  0.3871336877346039 ,
  -0.9129596948623657 ,  0.0232924222946167 , -0.26413074135780334,
   1.2807538509368896 , -0.05372702702879906,  0.4821065366268158 ,
  -0.4419023394584656 , -0.17105111479759216,  0.12990926206111908,
   1.5445550680160522 , -0.06065438315272331,  0.6618165373802185 ,
   1.4828908443450928 ,  0.980147123336792  , -0.09767183661460876,
   0.8977477550506592 , -0.38937681913375854,  0.21132256090641022,
   1.2850725650787354 ,  0.02493511326611042,  0.4744974374771118 ,
   0.3893517553806305 ,  0.1691308170557022 , -0.19364744424819946,
   0.5543128252029419 ,  0.4164937734603882 ,  1.0161747932434082 ,
   0.327548086643219  ,  0.4423463046550751 ,  1.1890698671340942 ,
   1.3680733442306519 ,  0.6148660778999329 ,  0.4354263246059418 ,
  -0.15955118834972382,  0.42439234256744385,  0.2588643729686737 ,
   1.4584708213806152 ,  0.286467581987381  , -0.07014823704957962,
  -0.0004591801261995 ,  0.4382765591144562 ,  0.11732685565948486,
   0.10285355150699615, -0.31680309772491455, -0.3821205198764801 ,
   0.5100446343421936 ],
 [-0.17027974128723145, -2.1659913063049316 , -0.11259983479976654,
  -0.28623268008232117, -0.8284869194030762 ,  0.01574817299842834,
   0.23800157010555267, -1.0926642417907715 ,  0.42569029331207275,
   0.6853064298629761 , -0.21782900393009186,  1.0220741033554077 ,
   0.5952194333076477 , -0.4838007092475891 ,  0.7720346450805664 ,
  -0.06279667466878891,  0.1392688900232315 , -0.10814454406499863,
  -0.07131095975637436,  0.5096732378005981 , -0.13141463696956635,
   0.03956324979662895,  0.39028432965278625,  0.38633018732070923,
   0.07967574149370193,  0.09620225429534912, -1.4383209943771362 ,
   0.585355281829834  ,  0.5016392469406128 , -0.00831273384392262,
   0.6747303605079651 , -0.24640299379825592, -0.33223748207092285,
   0.48007386922836304, -2.2619569301605225 , -0.03027849830687046,
  -0.531819760799408  ,  0.01356838550418615, -0.37423548102378845,
   0.6386227011680603 ,  0.19911208748817444,  0.20471712946891785,
   0.19992157816886902, -0.1667700856924057 , -0.06822627037763596,
   0.03332996740937233,  0.8992966413497925 ,  0.3651113212108612 ,
   0.5165852904319763 ,  0.2780156135559082 ,  0.31120437383651733,
  -0.6373208165168762 , -1.238353967666626  ,  0.2809927463531494 ,
   0.2890538275241852 , -0.05490781739354134, -0.5247693657875061 ,
  -1.3092564344406128 , -1.3725090026855469 ,  0.06536366790533066,
  -0.2260783463716507 , -0.2098093181848526 ,  0.9566947221755981 ,
  -0.13155603408813477],
 [-0.25137820839881897,  0.00298639456741512,  0.6505048274993896 ,
  -2.110661745071411  ,  0.226884663105011  ,  0.12810081243515015,
  -1.5574678182601929 ,  0.3295884430408478 , -0.0219039935618639 ,
  -0.16155920922756195,  0.2742190957069397 ,  0.490253746509552  ,
  -0.09750525653362274,  0.2963772118091583 , -0.08414813876152039,
  -0.18040500581264496,  0.3984347879886627 , -1.2979581356048584 ,
  -0.16390186548233032,  0.1310422420501709 ,  0.07117626070976257,
   0.38940441608428955, -0.442161500453949  ,  0.2024681568145752 ,
  -0.20151017606258392, -0.04603265970945358,  0.45910584926605225,
   0.10510149598121643, -0.4753875136375427 , -0.14706872403621674,
  -0.05428092181682587, -0.8713656663894653 , -0.5592459440231323 ,
  -1.0422428846359253 ,  0.48797929286956787,  0.03599982708692551,
   0.02167403697967529, -0.32760855555534363, -0.6839162707328796 ,
  -0.22363285720348358,  0.0082573089748621 ,  0.19644425809383392,
   0.08564990013837814, -0.14298374950885773,  0.2206568568944931 ,
  -0.5067356824874878 ,  0.262762188911438  ,  0.30535584688186646,
   0.24812017381191254, -0.4336307942867279 , -0.15968017280101776,
  -0.00687698973342776, -0.04778610914945602, -2.007842540740967  ,
   0.00818416848778725, -1.2398614883422852 , -0.1325460970401764 ,
   0.04585711658000946,  0.0901096761226654 , -0.19784556329250336,
   0.9242377877235413 , -0.5097314715385437 ,  0.4737958610057831 ,
   0.2594522535800934 ],
 [ 0.19867561757564545,  0.3810945153236389 , -0.15127171576023102,
   0.7845076322555542 , -0.18108153343200684,  1.272567868232727  ,
   0.1219249740242958 ,  0.7486384510993958 ,  0.12076383829116821,
  -0.06872531771659851,  0.5467226505279541 ,  0.16875389218330383,
  -0.8271700143814087 , -0.25663337111473083, -0.5107983350753784 ,
  -0.12113408744335175,  0.34710827469825745,  0.00411341013386846,
  -0.03331079334020615,  0.05536792054772377,  0.25811854004859924,
  -0.23141667246818542, -0.00804080534726381, -0.06155974417924881,
   0.3175797760486603 ,  0.21730904281139374, -0.7536788582801819 ,
   1.1639119386672974 ,  0.678817629814148  ,  0.2535417377948761 ,
   0.41368114948272705,  0.24862679839134216, -0.7791048288345337 ,
   0.15745478868484497,  0.06463455408811569, -0.2254837602376938 ,
   0.7658421397209167 , -0.16521891951560974,  0.2557890713214874 ,
   0.02213688567280769,  0.10262218117713928,  0.21181674301624298,
  -0.599276065826416  , -0.01872782036662102, -0.03350221365690231,
   0.35278794169425964, -0.8998799324035645 , -0.15207259356975555,
  -0.9164232015609741 ,  0.28623875975608826,  0.6006633043289185 ,
  -0.11547216027975082, -0.15913812816143036, -0.38249850273132324,
  -0.32334545254707336, -0.45292240381240845,  0.26098620891571045,
   0.08019751310348511,  0.16460095345973969, -0.01275751739740372,
   0.8744368553161621 ,  0.4387368857860565 ,  0.22991003096103668,
   0.23725625872612   ],
 [-0.13665874302387238, -0.00998923741281033, -0.04359927028417587,
   0.1227252185344696 , -0.21028585731983185,  0.13365620374679565,
   0.34439927339553833,  0.1354854255914688 ,  0.02927817218005657,
   0.19228550791740417,  0.48264870047569275,  0.08281403034925461,
  -0.9673749208450317 , -0.6179930567741394 , -0.9411534070968628 ,
  -0.5716259479522705 , -0.6168522238731384 , -0.12997913360595703,
  -0.01850428618490696, -0.1793312132358551 ,  0.05622609704732895,
   0.31943365931510925, -1.2765558958053589 ,  0.7426578998565674 ,
  -0.10999588668346405, -0.30245527625083923,  0.41103595495224   ,
  -1.0399818420410156 ,  0.9345963597297668 , -0.46911174058914185,
  -2.564422845840454  , -1.1322590112686157 ,  0.5891805291175842 ,
  -0.9076716303825378 ,  0.06644219905138016, -1.09183931350708   ,
  -1.0217607021331787 , -0.5815446972846985 , -0.1208314448595047 ,
   0.33781471848487854, -0.40365147590637207,  0.06248030066490173,
  -0.7934496998786926 ,  0.03147163242101669, -0.05200057476758957,
   0.22561754286289215,  0.45462751388549805, -0.907366931438446  ,
   0.10258648544549942, -1.6758211851119995 , -1.9082995653152466 ,
   0.34344854950904846, -1.8040883541107178 , -0.0466836653649807 ,
  -1.4294832944869995 , -0.3449684977531433 , -0.13559377193450928,
   0.03289487212896347,  0.7587729096412659 , -0.6713809967041016 ,
  -1.1322437524795532 , -0.19499807059764862, -0.04579142108559608,
   0.10126075893640518]])
    b3 = jnp.array([-1.3646259307861328 , -1.0182559490203857 ,  0.5433087348937988 ,
 -0.07182713598012924, -1.0769743919372559 ,  0.13911835849285126,
  0.359899640083313  , -2.0407662391662598 ,  2.238767385482788  ,
 -0.7616875767707825 , -0.41115015745162964, -1.4455772638320923 ,
 -0.28176257014274597, -0.9127123355865479 , -0.46196290850639343,
 -3.6735050678253174 , -0.2297012209892273 , -0.14982515573501587,
 -0.8265009522438049 ,  0.4945486783981323 , -0.7622408866882324 ,
  0.30164799094200134,  0.3888223171234131 , -0.704339861869812  ,
  1.7759066820144653 ,  0.00184660230297595,  1.109479308128357  ,
 -0.8330708742141724 ,  1.07419753074646   ,  0.05026405677199364,
 -1.5334594249725342 , -0.8489382266998291 , -0.8463457822799683 ,
 -0.25862789154052734, -0.45150262117385864, -0.7366195917129517 ,
 -0.2980155944824219 ,  0.11745546758174896, -1.0382521152496338 ,
 -0.6717022061347961 , -0.28444498777389526, -0.43080204725265503,
  0.11065007001161575,  0.26921576261520386, -1.0977904796600342 ,
  0.8690788745880127 , -0.7689796090126038 , -0.9410704374313354 ,
 -0.409496933221817  ,  0.15755879878997803, -0.15086284279823303,
 -0.14481478929519653, -0.19873349368572235, -0.26036524772644043,
 -0.9037812948226929 ,  0.3881511390209198 , -1.100509524345398  ,
 -1.5966600179672241 , -2.303565740585327  ,  1.503185510635376  ,
 -1.3044191598892212 ,  0.27501577138900757,  1.1609731912612915 ,
  0.36670318245887756])
    W4 = jnp.array([[-0.33353015780448914,  0.11366940289735794,  0.07523101568222046,
  -0.8066102266311646 , -1.143678903579712  , -0.6907758116722107 ,
  -0.04711341112852097, -0.496753066778183  , -0.46158096194267273,
  -0.38720932602882385, -0.4710786044597626 , -0.8774663805961609 ,
  -0.3617565333843231 , -0.10338591784238815, -0.3719862699508667 ,
  -0.9098479747772217 ,  0.05073005333542824, -0.8666329979896545 ,
  -0.7675027847290039 ,  0.15655508637428284, -0.9115954041481018 ,
  -0.04326218739151955, -0.05823076888918877, -0.04230191186070442,
  -0.15372316539287567, -0.3523859977722168 ,  0.35158771276474   ,
  -0.6608150601387024 , -0.08463400602340698, -0.9965850710868835 ,
  -0.15208181738853455,  0.15910381078720093, -0.9432966113090515 ,
  -0.2731691598892212 , -0.44494152069091797, -0.5653576850891113 ,
  -0.9154657125473022 , -0.3684655427932739 , -0.11688004434108734,
  -0.5320471525192261 ,  0.02402286976575851, -3.3519060611724854 ,
  -0.864272952079773  , -0.9187014102935791 , -0.0676160454750061 ,
  -0.22593696415424347, -0.726568877696991  ,  0.2752971053123474 ,
  -1.0412755012512207 , -0.4897419512271881 , -0.5507066249847412 ,
  -1.8126505613327026 , -0.3888770639896393 , -0.13369126617908478,
  -1.5614677667617798 , -0.5696452856063843 , -0.5492663383483887 ,
  -0.6659572720527649 ,  0.23638290166854858, -0.18609720468521118,
   0.4894106984138489 ,  0.2918553650379181 , -0.2757364511489868 ,
  -0.4056881070137024 ],
 [-1.3676424026489258 , -1.5265939235687256 ,  1.5448253154754639 ,
  -0.42135128378868103, -0.5607861280441284 ,  0.7793572545051575 ,
   0.21641309559345245,  0.04995277151465416,  3.885545253753662  ,
  -0.5568350553512573 , -0.35226258635520935,  0.9144472479820251 ,
   0.5500615239143372 ,  0.10112037509679794, -1.3915032148361206 ,
  -0.4213351607322693 , -0.8097678422927856 ,  0.07086089253425598,
  -2.3186028003692627 , -0.02341989427804947, -0.04228339716792107,
   0.5706795454025269 , -0.7337129712104797 , -2.1085121631622314 ,
   1.2169631719589233 ,  0.19538551568984985,  1.3600544929504395 ,
  -1.4726570844650269 , -0.11614113301038742, -2.469520330429077  ,
  -0.33635279536247253,  0.00131086516194046,  0.9107453227043152 ,
   0.7325078248977661 ,  0.5583115220069885 , -0.49578115344047546,
  -0.18535637855529785,  0.3383997976779938 , -1.1436195373535156 ,
   0.2809479832649231 , -4.251450061798096  ,  1.249627709388733  ,
   1.2433489561080933 ,  1.9631165266036987 , -0.5170018672943115 ,
  -2.088533401489258  ,  1.987821340560913  ,  1.9765280485153198 ,
  -0.8932936191558838 ,  0.7661281824111938 ,  2.309910297393799  ,
   0.6036428809165955 , -0.2514885663986206 ,  1.7389079332351685 ,
   0.46725961565971375,  1.2725507020950317 , -0.00916077103465796,
  -0.817547082901001  ,  1.5518263578414917 ,  2.121454954147339  ,
   0.6635688543319702 ,  1.1480133533477783 ,  1.3559290170669556 ,
   1.416467308998108  ],
 [-0.8395838141441345 , -0.15476638078689575, -0.3553657829761505 ,
  -0.34938013553619385, -0.01617606543004513, -0.0832294449210167 ,
  -0.16336554288864136,  0.0658302754163742 ,  0.6289511322975159 ,
   0.3557237386703491 ,  0.1132882609963417 , -0.35378846526145935,
   0.04448996111750603,  0.01835185475647449,  0.46028417348861694,
  -0.33121728897094727, -0.5917572975158691 , -0.5577930212020874 ,
   0.2948371171951294 , -0.2157791256904602 , -0.17903746664524078,
   0.20229572057724   ,  0.09549039602279663, -0.00057616567937657,
  -0.17349949479103088, -0.3866721987724304 ,  0.28216102719306946,
  -1.0811687707901    , -0.1919834315776825 , -0.0064250179566443 ,
  -0.18073542416095734, -0.5972209572792053 , -0.512574315071106  ,
  -0.03845180943608284, -0.04891011118888855,  0.1904696524143219 ,
  -0.8924388289451599 ,  0.06134252622723579,  0.13049675524234772,
  -0.18838392198085785,  0.3003910183906555 , -0.3547804653644562 ,
   0.5357978343963623 ,  0.4399402141571045 , -0.11439624428749084,
  -0.5880452990531921 , -0.11799921095371246,  0.36441558599472046,
  -0.1948411613702774 ,  0.14162689447402954, -0.36867377161979675,
  -0.45051828026771545, -0.5965559482574463 , -0.4649702310562134 ,
   0.16231514513492584, -0.166100412607193  , -0.00660470919683576,
   0.2920076251029968 ,  0.16233289241790771,  0.3548625707626343 ,
   0.43897873163223267,  0.41591018438339233, -0.1946018487215042 ,
   0.2161012440919876 ],
 [ 0.1312558948993683 , -0.06144127994775772, -0.8181037902832031 ,
  -0.0610208623111248 ,  0.5059040188789368 ,  0.603354811668396  ,
   0.2890123128890991 , -0.3109937906265259 , -1.2255151271820068 ,
   0.18751096725463867, -0.7530158162117004 ,  0.2905575931072235 ,
   0.27300962805747986,  0.05768387764692307, -0.45255568623542786,
  -0.2889862656593323 ,  0.6841924786567688 , -2.0712413787841797 ,
  -0.8921828269958496 , -0.8089644908905029 ,  0.04566723480820656,
   0.00907773058861494,  0.00748302554711699,  0.1373334676027298 ,
  -0.10509604215621948, -0.5598524212837219 , -0.46490898728370667,
  -0.2686431407928467 ,  0.09374329447746277, -0.3794865012168884 ,
  -0.11566317081451416, -0.398278146982193  ,  0.06214548647403717,
   0.3566989600658417 , -0.02880697697401047, -0.6408664584159851 ,
  -0.5360711812973022 ,  0.13657067716121674, -0.0015394581714645 ,
   0.04406082257628441, -0.0366830974817276 ,  1.1707395315170288 ,
  -0.3398008644580841 , -2.513946771621704  ,  0.3131667971611023 ,
   0.9615517854690552 , -0.5206974744796753 , -4.275073051452637  ,
   0.8359662294387817 , -0.27009451389312744, -1.8915448188781738 ,
   0.07288241386413574,  0.4496651291847229 , -0.9874110221862793 ,
  -1.0714691877365112 ,  0.15888766944408417, -0.16710038483142853,
  -1.918366551399231  , -0.09690722823143005, -0.5236771702766418 ,
   0.45037493109703064,  0.16167956590652466, -0.183699369430542  ,
  -0.5467823147773743 ],
 [-0.21185943484306335, -0.1727578490972519 , -0.11650827527046204,
   0.02849389612674713, -0.2072792947292328 ,  0.03476818278431892,
  -0.06436169892549515, -0.3729430139064789 , -0.19948869943618774,
  -0.1518542617559433 ,  0.09470632672309875,  0.00927243288606405,
  -0.19248077273368835, -0.05823749303817749, -0.04526935517787933,
  -0.02136010676622391,  0.19034326076507568, -0.04408003389835358,
  -0.11931556463241577, -0.24686114490032196, -0.20452049374580383,
   0.16892044246196747, -0.05849280208349228,  0.305087685585022  ,
  -0.12887492775917053,  0.07429205626249313,  0.3832159638404846 ,
   0.36009693145751953,  0.22325356304645538, -0.32069167494773865,
   0.6062124371528625 ,  0.05878950655460358, -0.07770240306854248,
  -0.07782386243343353,  0.1630840301513672 , -0.40049487352371216,
  -0.16722463071346283,  0.2380383163690567 , -0.22593550384044647,
  -0.07943262159824371,  0.15401911735534668, -0.04072483256459236,
   0.315102219581604  , -0.23271441459655762, -0.03613623231649399,
  -0.08674928545951843, -0.09568586200475693,  0.16318394243717194,
   0.07968712598085403, -0.09639278054237366,  0.20568203926086426,
  -0.16710799932479858, -0.42628276348114014, -0.04598991945385933,
   0.04028573259711266, -0.2857741117477417 , -0.11803964525461197,
  -0.26904162764549255, -0.19648033380508423,  0.3979743719100952 ,
   0.3141002655029297 , -0.09758175164461136, -0.11380475014448166,
  -0.30727294087409973],
 [ 0.28689685463905334, -0.3070339262485504 , -1.0545867681503296 ,
   0.04072744026780128,  0.3024866580963135 ,  0.08603721857070923,
   0.30313605070114136, -0.10189961642026901, -1.8407981395721436 ,
   0.15094268321990967,  0.12204230576753616,  0.3659006953239441 ,
   0.10662926733493805, -0.9652320742607117 , -0.4492214620113373 ,
   0.15622036159038544, -0.8486050367355347 , -0.26640784740448   ,
   0.08719242364168167,  0.01931561529636383, -0.42415642738342285,
   0.10036005079746246, -0.10738538950681686, -0.5009190440177917 ,
  -0.3495606482028961 ,  0.17187447845935822, -0.7749775052070618 ,
  -0.11034439504146576,  0.12244080007076263,  0.03718441724777222,
  -0.01279982458800077, -0.7384372353553772 ,  0.12642191350460052,
  -0.5721926093101501 , -0.19643534719944   , -0.714219331741333  ,
  -0.5818886756896973 ,  0.05592141672968864, -0.7289963364601135 ,
  -0.5343504548072815 , -0.6689070463180542 ,  0.20890970528125763,
  -0.43342962861061096, -0.7793871760368347 ,  0.5666711926460266 ,
  -1.2301435470581055 ,  0.39888161420822144, -1.4221493005752563 ,
  -0.6493847966194153 , -0.02824590355157852, -1.1543433666229248 ,
  -0.15073177218437195, -1.1127991676330566 , -0.7742554545402527 ,
  -1.0179611444473267 , -1.392601490020752  ,  0.02635886147618294,
  -0.3335082530975342 , -0.19364005327224731, -0.6222148537635803 ,
  -0.4351215958595276 , -0.15369638800621033, -0.1623745709657669 ,
  -0.7029314041137695 ],
 [ 0.4493294656276703 ,  0.18620766699314117, -0.8425122499465942 ,
  -0.06302202492952347, -0.372293084859848  ,  0.23884904384613037,
   0.12409376353025436, -0.48905232548713684, -1.0792917013168335 ,
   0.28484657406806946, -0.2887188792228699 ,  0.2520279884338379 ,
  -0.04495895281434059, -0.11989308148622513, -0.18308758735656738,
   0.19038966298103333, -2.1979010105133057 ,  0.8976293802261353 ,
  -0.09329965710639954, -0.10710416734218597, -0.6384885311126709 ,
  -0.9203628897666931 ,  0.15185284614562988, -0.389173686504364  ,
  -0.07532081753015518,  0.5723182559013367 , -0.22684471309185028,
  -0.7435990571975708 , -0.14356881380081177,  0.00186383095569909,
   0.3223242163658142 , -0.03473674133419991,  0.22284652292728424,
   0.12394745647907257, -0.02654473297297955, -0.3854830265045166 ,
  -0.7610201239585876 ,  0.4303831458091736 ,  0.27847808599472046,
   0.11884543299674988, -0.23681673407554626, -0.7779105305671692 ,
   0.13528503477573395,  0.33480244874954224,  0.4292883276939392 ,
   0.6331586241722107 ,  0.10040987282991409,  0.01564994268119335,
   0.00270875217393041, -0.5336118936538696 ,  0.43519341945648193,
  -0.01823604293167591,  0.30005183815956116, -1.8436027765274048 ,
   0.6498239040374756 ,  0.34026893973350525, -0.13588908314704895,
   0.2713751196861267 , -0.18580317497253418, -0.06115105748176575,
  -0.9898505210876465 , -0.01869499869644642, -0.38881638646125793,
  -0.8658129572868347 ],
 [-2.1950786113739014 , -0.808600664138794  ,  0.08373341709375381,
   0.1655293107032776 ,  0.7162187099456787 ,  0.04131146892905235,
   0.1086084321141243 , -0.2737416625022888 , -0.29772084951400757,
  -0.15313729643821716,  0.25716379284858704,  1.3563764095306396 ,
   0.6187523007392883 ,  0.05507570505142212,  0.32107239961624146,
   0.8351839780807495 ,  0.3308400809764862 , -1.8241227865219116 ,
  -1.438057780265808  , -0.7048689126968384 , -1.4781514406204224 ,
   0.6374710202217102 ,  0.06285732984542847,  0.45602837204933167,
  -2.9896841049194336 , -1.8919506072998047 ,  0.5380093455314636 ,
   0.5836583971977234 ,  0.12468563765287399, -0.3754279315471649 ,
   0.23347806930541992,  1.0677063465118408 , -0.31432509422302246,
  -0.41313686966896057, -0.8617275357246399 , -2.01336669921875   ,
   0.547895073890686  , -0.2467961609363556 , -1.692542552947998  ,
  -1.376718521118164  , -0.00038837533793412,  0.5281631946563721 ,
   0.8150338530540466 ,  0.32551810145378113, -0.2613104581832886 ,
  -0.37167710065841675,  0.23341765999794006,  0.03812552988529205,
  -2.1551880836486816 ,  0.21988654136657715,  0.22670622169971466,
   1.1126608848571777 ,  0.4644811749458313 , -1.1481643915176392 ,
  -0.0890963226556778 , -0.6121190786361694 ,  0.490339457988739  ,
  -0.2003387063741684 ,  1.0773502588272095 , -0.389896959066391  ,
   0.593925952911377  , -0.30332261323928833,  1.3223689794540405 ,
  -1.7557148933410645 ],
 [-0.1938624083995819 , -0.1457524448633194 , -1.5348807573318481 ,
  -0.09227214753627777,  0.3392677903175354 ,  0.3054044842720032 ,
  -0.8038831949234009 ,  0.40676116943359375,  0.10906421393156052,
   0.6524662971496582 , -0.6115441918373108 , -0.8710371255874634 ,
  -0.698390543460846  , -0.41565945744514465, -0.20180188119411469,
  -0.1519722044467926 , -0.7553132772445679 , -0.15848346054553986,
  -0.479082316160202  , -0.2719307243824005 , -0.09119680523872375,
  -0.7698797583580017 , -0.2435612678527832 ,  0.03722681477665901,
  -0.6539931297302246 , -0.37354570627212524,  0.21328988671302795,
  -0.3404736816883087 ,  0.00281438999809325, -0.00307133980095387,
  -0.06448686122894287, -0.0425194650888443 , -0.3018243610858917 ,
   0.2257406860589981 , -0.4304331839084625 ,  0.02543257176876068,
  -0.19957739114761353, -0.2189764678478241 , -0.5522860288619995 ,
  -0.7778595089912415 , -1.748872995376587  , -0.3506704270839691 ,
  -0.15627023577690125, -0.141018807888031  , -0.13176055252552032,
  -1.9880667924880981 , -1.269200325012207  , -1.6242001056671143 ,
  -0.5110008120536804 , -0.08859008550643921,  0.2241920530796051 ,
   0.4258229732513428 , -0.2456246018409729 ,  0.33658793568611145,
  -0.6529048085212708 , -0.6003577709197998 ,  0.3688541054725647 ,
   0.08268722146749496, -0.04198771715164185, -0.9651690125465393 ,
  -0.38776731491088867, -0.3480817675590515 , -0.14949101209640503,
  -0.705872118473053  ],
 [ 0.20741026103496552, -0.8522128462791443 ,  0.5920252799987793 ,
  -0.49669724702835083, -0.40455278754234314,  0.11171863228082657,
  -2.8183035850524902 , -0.23746463656425476,  0.9249041676521301 ,
   0.00078944640699774,  0.12786996364593506,  0.4645631015300751 ,
   0.00116621074266732, -0.48169103264808655, -0.17044062912464142,
   0.43063899874687195,  0.4202849566936493 ,  1.0558714866638184 ,
  -0.8098724484443665 , -0.7673503160476685 , -1.3480751514434814 ,
  -0.14996610581874847, -0.37255075573921204,  0.7403271794319153 ,
  -0.12869910895824432, -0.3597889542579651 ,  1.0478204488754272 ,
  -0.30801329016685486, -0.8853157162666321 , -1.949087381362915  ,
   0.4083743393421173 , -0.33514079451560974,  0.28214025497436523,
  -0.25072523951530457,  0.488089919090271  , -0.7815966606140137 ,
  -0.33266153931617737,  0.20340804755687714,  1.4523457288742065 ,
   0.15921159088611603, -0.08460390567779541,  0.09173049032688141,
   0.600266695022583  , -1.2792584896087646 , -1.557766318321228  ,
   1.3392263650894165 , -1.130022644996643  ,  0.47382816672325134,
  -0.6323217749595642 ,  0.2215767800807953 , -0.7656675577163696 ,
  -0.6083546876907349 ,  0.7883385419845581 ,  0.646551787853241  ,
   0.0526411198079586 , -0.7936161160469055 ,  0.18582475185394287,
   1.425561547279358  ,  0.37953585386276245, -0.510962724685669  ,
  -0.2625584006309509 ,  0.26661157608032227,  1.3427481651306152 ,
   1.3769311904907227 ],
 [-1.3571776151657104 ,  0.5094533562660217 ,  0.3620639741420746 ,
  -0.6796013116836548 ,  0.7524194121360779 , -0.9989083409309387 ,
   0.4811389744281769 ,  0.06715869158506393, -0.44359588623046875,
  -0.6211062669754028 ,  0.5557355880737305 , -0.837736964225769  ,
  -0.18389566242694855, -0.17858779430389404,  0.18002627789974213,
  -0.3795924782752991 ,  0.387529194355011  ,  0.05041772127151489,
  -0.03723543882369995, -1.5034198760986328 , -1.066711664199829  ,
   1.0697509050369263 , -0.15113627910614014, -0.25150978565216064,
  -0.9561410546302795 ,  0.2861611843109131 ,  0.5697645545005798 ,
  -2.2096285820007324 ,  0.17599403858184814, -0.6709024906158447 ,
  -0.628674328327179  ,  1.88059401512146   , -0.4296368956565857 ,
   0.5365611910820007 ,  0.48932474851608276, -0.7925119996070862 ,
   1.115602731704712  ,  1.0175577402114868 ,  0.17349621653556824,
  -0.8512405157089233 ,  0.8566632866859436 ,  0.47783276438713074,
   0.9370200634002686 ,  0.08379288017749786,  0.00664844177663326,
   1.5399532318115234 ,  0.9734658598899841 ,  1.9768239259719849 ,
  -1.369315266609192  , -0.9368816018104553 ,  0.6654763221740723 ,
   0.4043276906013489 , -0.6188409328460693 ,  0.7855138182640076 ,
  -0.5688194036483765 , -0.02174454927444458,  0.03653841465711594,
   0.06434425711631775,  0.0005808713613078 ,  1.2766071557998657 ,
   0.15090224146842957, -0.4244079291820526 ,  1.1772974729537964 ,
  -1.281662106513977  ],
 [ 0.2359081655740738 , -0.23692014813423157,  0.9219355583190918 ,
  -0.07551869750022888, -0.20872172713279724, -0.19772900640964508,
  -0.0111328512430191 ,  0.7601567506790161 , -1.4615432024002075 ,
  -0.2832282781600952 , -2.7376492023468018 , -0.11800136417150497,
   0.3405509293079376 , -0.0299429576843977 , -0.11210303008556366,
  -0.08601595461368561, -0.1617150604724884 , -0.30481234192848206,
   0.06569578498601913,  0.16031695902347565, -0.9941018223762512 ,
   0.3118833005428314 , -0.09189902991056442,  0.04994209855794907,
   0.11323808133602142,  0.512864351272583  ,  0.05197763815522194,
  -0.256325900554657  ,  0.04213877394795418,  0.9795791506767273 ,
   0.1788293719291687 ,  0.8437306880950928 , -0.07100881636142731,
   0.42503753304481506, -0.12582074105739594, -0.9488610029220581 ,
   0.8434751033782959 , -0.27313703298568726, -0.8619030714035034 ,
  -0.02056545205414295,  0.00875063613057137, -0.7313132882118225 ,
   0.24040310084819794, -0.6316973567008972 , -0.4657226502895355 ,
  -0.1834205985069275 , -0.5021592974662781 , -0.08777066320180893,
  -0.2713477313518524 , -0.8433288335800171 , -0.14200513064861298,
  -1.1634365320205688 , -0.2106894850730896 ,  0.6727758646011353 ,
  -0.33969053626060486, -0.22920435667037964,  0.2443113476037979 ,
   0.7582096457481384 , -0.03372430428862572,  0.09136929363012314,
   0.20749245584011078, -0.3473087251186371 ,  0.35712960362434387,
   0.21705219149589539],
 [-2.509809732437134  , -0.317985862493515  , -0.04921456053853035,
  -0.19407187402248383,  0.5432204008102417 , -0.6940796375274658 ,
   0.21075484156608582, -0.14062172174453735,  0.04171721264719963,
   0.00549200642853975, -0.5138108134269714 , -0.07826581597328186,
  -0.8923209309577942 ,  0.34795835614204407, -1.891523838043213  ,
  -1.3058310747146606 ,  0.273750901222229  , -2.200847625732422  ,
   0.04606848955154419,  0.5633758306503296 ,  0.4406585395336151 ,
  -0.5503399968147278 ,  0.33683136105537415, -0.9671269655227661 ,
  -0.4283290207386017 ,  0.14711619913578033, -0.3922745883464813 ,
  -0.4519234299659729 , -0.6686968207359314 , -0.7502089142799377 ,
  -0.22158470749855042, -1.856849193572998  ,  0.14374329149723053,
   0.08466216176748276,  0.6521906852722168 ,  0.04287391901016235,
   0.10309547930955887, -0.24487018585205078, -0.28410810232162476,
  -0.8921994566917419 , -0.34115785360336304,  1.0189601182937622 ,
   0.4303530156612396 ,  0.18595921993255615,  0.7555208802223206 ,
  -0.25899970531463623, -1.0426926612854004 ,  0.8548985719680786 ,
  -2.317030906677246  ,  0.9975631237030029 , -0.6930971145629883 ,
  -0.14460134506225586, -1.180946707725525  ,  0.12158592790365219,
   0.6277033090591431 ,  0.02720135264098644, -0.02222517877817154,
  -0.7398154139518738 , -0.6812707185745239 , -0.5608774423599243 ,
  -0.5957228541374207 , -0.3040357530117035 ,  0.25329434871673584,
   1.0250022411346436 ],
 [ 0.7984511852264404 , -0.4312180280685425 , -0.6724041700363159 ,
   0.06786367297172546,  0.4174211919307709 ,  0.6763656139373779 ,
   0.05613777041435242,  1.2402760982513428 , -0.20642264187335968,
  -0.41152167320251465,  0.17022660374641418,  0.11663538217544556,
  -0.17535054683685303, -0.08343935012817383, -0.7941479086875916 ,
  -0.0138294855132699 ,  0.3582237958908081 ,  0.6791470050811768 ,
  -0.5973401069641113 , -2.129054307937622  ,  0.7182000279426575 ,
  -0.7657778263092041 ,  0.5227864980697632 , -0.13239389657974243,
  -2.145312547683716  , -1.0972038507461548 ,  1.7935911417007446 ,
  -0.4014410674571991 , -0.776046633720398  , -0.9882645606994629 ,
   0.6169828772544861 ,  1.1728551387786865 , -0.43127763271331787,
  -1.2230808734893799 ,  0.856020450592041  ,  0.41314375400543213,
  -2.212583303451538  , -0.8840205669403076 ,  0.1583765298128128 ,
   0.48993366956710815, -0.09909931570291519,  1.7270607948303223 ,
  -0.07919654250144958,  0.1142655536532402 , -2.751861095428467  ,
  -2.125645160675049  ,  2.3057351112365723 ,  0.9216163158416748 ,
  -0.12748844921588898,  0.0525178499519825 ,  1.819211483001709  ,
   2.7745168209075928 , -1.2517905235290527 ,  0.6249556541442871 ,
   1.2197591066360474 , -1.2161478996276855 , -0.6517750024795532 ,
   0.5103004574775696 , -0.6850996613502502 , -0.10151054710149765,
  -0.6522697806358337 ,  0.5248931646347046 ,  1.2625648975372314 ,
  -0.6112782955169678 ],
 [-1.1250901222229004 , -0.6815210580825806 , -0.713240385055542  ,
  -0.08737855404615402, -0.536500871181488  , -0.05196717754006386,
   0.32920041680336   , -0.06272883713245392,  0.9264395833015442 ,
  -0.1267785280942917 , -0.07887930423021317, -0.2991260588169098 ,
  -0.7358474731445312 , -0.04538271948695183, -1.9884759187698364 ,
  -0.18568284809589386, -0.487863689661026  , -0.17180116474628448,
  -1.3245320320129395 , -0.5912627577781677 , -1.7340302467346191 ,
   0.544489860534668  , -0.03819004818797112,  0.6556006073951721 ,
   0.23393157124519348, -0.35283905267715454,  0.324126273393631  ,
   0.38430389761924744,  0.07696229219436646,  0.28027263283729553,
   0.04855256155133247,  0.1758907437324524 ,  0.10745251178741455,
   1.0987812280654907 , -0.9380714893341064 ,  0.10461463034152985,
  -0.2938806712627411 ,  0.0768207535147667 , -0.661539614200592  ,
  -0.12161402404308319,  0.11978957802057266,  0.2450341135263443 ,
   0.4177509844303131 ,  0.6730248332023621 , -0.17505548894405365,
   0.9397064447402954 ,  0.00559283420443535, -1.5040724277496338 ,
  -0.7985901832580566 ,  0.3521060049533844 ,  0.4218851327896118 ,
   0.5882602334022522 , -0.8777478933334351 ,  0.7949473857879639 ,
  -0.6868060827255249 , -0.18935053050518036,  0.07688172906637192,
  -0.39133158326148987,  0.6234835982322693 ,  0.6923146843910217 ,
   0.5013225674629211 , -0.13160008192062378,  1.0453975200653076 ,
   0.07121745496988297],
 [-0.3397350311279297 ,  0.3911484479904175 , -1.139512062072754  ,
  -0.02040374651551247, -0.14109618961811066,  0.5083449482917786 ,
   0.2839878797531128 ,  0.839857816696167  , -0.3471865952014923 ,
  -0.6880345940589905 , -0.3070610761642456 , -1.7592277526855469 ,
   0.9239003658294678 ,  0.19991269707679749,  0.19244667887687683,
  -0.529596745967865  , -2.4393789768218994 ,  0.2584553360939026 ,
   0.15451347827911377,  0.3454280495643616 , -1.0032927989959717 ,
  -1.6206609010696411 ,  0.12548288702964783, -0.27762067317962646,
   0.3565290570259094 ,  0.7035658359527588 , -0.6836966872215271 ,
  -1.5512455701828003 ,  0.09208286553621292,  0.575248658657074  ,
  -0.1716574728488922 ,  1.1250734329223633 , -0.11709263920783997,
   0.9735745787620544 ,  0.23695078492164612,  1.0130289793014526 ,
  -0.5001200437545776 , -0.2882530689239502 , -0.269005686044693  ,
   0.2860376834869385 , -0.02838069759309292,  0.20439033210277557,
  -1.0652042627334595 , -1.1577101945877075 , -0.4224667251110077 ,
  -1.0282195806503296 , -4.498273849487305  , -0.11012744158506393,
  -1.0038801431655884 ,  0.5519024133682251 , -0.30437496304512024,
   0.14608356356620789, -0.5885512232780457 , -0.48267772793769836,
  -0.5576456189155579 , -1.0091636180877686 ,  0.12514010071754456,
   0.28483062982559204,  1.3650683164596558 , -0.37593793869018555,
  -0.8163473010063171 ,  0.5873275995254517 , -0.5731664896011353 ,
   0.41209691762924194],
 [-0.5051576495170593 , -0.33123746514320374,  0.10589416325092316,
   0.01700546219944954,  0.15450292825698853,  0.15030235052108765,
   0.26696398854255676,  0.5468868613243103 , -1.775439977645874  ,
  -0.06321248412132263, -0.01702198386192322,  0.10042762011289597,
  -0.7064433693885803 ,  0.11896561086177826, -0.20050907135009766,
   0.14703553915023804, -0.4509525001049042 ,  0.41709020733833313,
   0.27899330854415894,  0.02343451976776123,  0.00376150221563876,
   0.10711250454187393, -0.8217329978942871 , -0.18217237293720245,
  -0.05970184504985809,  0.2709541320800781 ,  0.3329223096370697 ,
   0.3453916311264038 , -0.0104155819863081 ,  0.4557527005672455 ,
   0.00445226859301329,  0.665283739566803  ,  0.2668841779232025 ,
   0.949489414691925  , -0.0447411835193634 , -0.01469817291945219,
  -0.3328389823436737 ,  0.32831284403800964,  0.3178933262825012 ,
  -0.9489880204200745 ,  0.4382008910179138 , -0.346206933259964  ,
  -0.633196234703064  , -0.8193413019180298 , -2.3813469409942627 ,
  -0.2131556272506714 ,  0.00936356745660305, -1.7463914155960083 ,
  -0.14730508625507355, -0.19505682587623596, -0.2388594001531601 ,
  -0.29015809297561646,  0.2020454704761505 ,  0.525534987449646  ,
  -1.5871013402938843 , -1.1758140325546265 ,  0.0813959613442421 ,
   0.40108025074005127,  0.8934759497642517 ,  0.12420941144227982,
   0.4695563018321991 , -0.5671178698539734 , -0.16697350144386292,
   0.7857256531715393 ],
 [ 0.20009025931358337,  0.3776988983154297 ,  0.31920987367630005,
   0.06314796209335327,  0.37406665086746216, -0.07179521024227142,
  -0.4953117072582245 , -0.28155186772346497, -0.9279426336288452 ,
   0.02520143426954746, -1.4101946353912354 ,  0.7276914715766907 ,
  -0.462117999792099  ,  0.18773764371871948, -0.3645181655883789 ,
   0.04536796733736992, -0.5094013810157776 , -0.8670492172241211 ,
  -0.04114118590950966, -0.00997260492295027, -0.9892567992210388 ,
   0.05979548394680023, -0.11965171992778778, -1.0060774087905884 ,
  -0.6936665177345276 , -0.35552549362182617, -0.36735275387763977,
  -0.05880172550678253,  0.331095814704895  , -0.6261845827102661 ,
  -0.2504214942455292 ,  0.7785856127738953 ,  0.10422786325216293,
  -0.6165136098861694 ,  0.4645758867263794 , -0.4686101973056793 ,
  -1.670169472694397  ,  0.22560766339302063, -0.02359367907047272,
  -0.09153569489717484, -0.21378852427005768, -0.44243037700653076,
  -0.11925293505191803, -0.325610488653183  ,  0.4275486469268799 ,
  -0.22151337563991547, -1.184144377708435  ,  0.00346491672098637,
   0.5580094456672668 ,  0.06202741712331772, -0.13820359110832214,
   0.12244004011154175,  0.20522578060626984,  0.07101978361606598,
   0.6922318935394287 , -0.5007014274597168 ,  0.16012026369571686,
  -0.43057939410209656, -0.2385987490415573 , -0.44161978363990784,
  -1.0596288442611694 , -0.19470922648906708,  0.7054163813591003 ,
   1.108677625656128  ],
 [-0.1164470985531807 , -0.6096699237823486 , -0.04629050567746162,
   0.47407808899879456,  1.6476590633392334 ,  0.23248831927776337,
  -0.08575881272554398,  0.45758944749832153, -0.2089150995016098 ,
  -0.25027143955230713, -0.7507164478302002 ,  0.3201625347137451 ,
  -0.6313478350639343 ,  0.0087808333337307 , -0.1528523564338684 ,
  -0.608677864074707  ,  0.5869966745376587 , -0.14383025467395782,
   0.20210087299346924, -0.490952730178833  ,  0.40284252166748047,
  -0.07909488677978516, -0.11920063942670822,  0.16724038124084473,
   0.01277732755988836, -0.2510082721710205 , -0.04522941261529922,
   0.8659372329711914 ,  0.2266370803117752 , -0.8961818218231201 ,
   0.01403425075113773,  0.9419934153556824 ,  0.19782720506191254,
  -1.5514718294143677 ,  0.05919225141406059,  0.05574658513069153,
  -1.063051462173462  ,  0.24890370666980743,  0.7573167085647583 ,
   0.13351885974407196, -0.11403325200080872,  0.5773284435272217 ,
   0.3494213819503784 ,  0.0091557502746582 , -0.6472948789596558 ,
   0.07028137892484665,  0.17581181228160858,  0.4864680767059326 ,
   0.4339942932128906 ,  0.19865332543849945, -0.23631075024604797,
  -0.00925772823393345, -0.325608491897583  ,  0.2118545025587082 ,
  -0.8167504668235779 , -0.04542751610279083,  0.274915486574173  ,
  -0.6847267150878906 , -0.14607089757919312, -0.5058364868164062 ,
   0.37468284368515015,  0.41554152965545654,  0.2788872718811035 ,
   0.10457182675600052],
 [ 0.13060502707958221,  0.00559638347476721,  0.07714345306158066,
  -0.26564282178878784, -0.5630469918251038 ,  0.2599015235900879 ,
  -0.08966594189405441,  0.01651454530656338, -0.5103227496147156 ,
  -0.19646809995174408,  0.12685082852840424, -0.1934482455253601 ,
  -0.10990090668201447, -0.07045541703701019,  0.02382059954106808,
  -0.17101554572582245, -0.02873203903436661, -0.14112012088298798,
   0.20701195299625397,  0.0287795327603817 ,  0.3681582808494568 ,
  -0.1818583756685257 ,  0.0221309382468462 ,  0.21884702146053314,
  -0.0523054338991642 , -0.8481678366661072 ,  0.06304649263620377,
  -0.40953949093818665, -0.2141517698764801 ,  0.20962141454219818,
   0.13172706961631775, -0.4956137239933014 , -0.18714044988155365,
  -0.4107365310192108 , -0.31164899468421936,  0.2666341960430145 ,
   0.38430121541023254, -0.4628567099571228 ,  0.637583315372467  ,
  -0.1159922331571579 , -0.07943686097860336, -0.2700679302215576 ,
   0.09915776550769806, -0.544083833694458  , -0.580990731716156  ,
   0.34170016646385193,  0.13963736593723297,  0.2101196050643921 ,
   0.07313361018896103, -0.7937648296356201 , -0.300454705953598  ,
  -0.791948139667511  ,  0.33182743191719055, -0.3286397457122803 ,
  -0.03342109173536301, -0.01480680983513594, -0.19650402665138245,
  -0.7689306139945984 , -0.00794501323252916,  0.3237093389034271 ,
   0.02760485373437405, -0.07795479893684387, -0.2578520178794861 ,
  -0.3062320649623871 ],
 [-0.1426214873790741 ,  0.47090020775794983,  0.39602789282798767,
   0.0470820777118206 ,  0.13766604661941528, -0.6344102025032043 ,
   1.088912844657898  , -0.03809291124343872, -0.277863472700119  ,
  -1.4163366556167603 ,  1.1017600297927856 ,  1.9477826356887817 ,
  -1.0308098793029785 , -1.2448991537094116 , -1.5079388618469238 ,
  -0.6726000905036926 , -0.12284626811742783,  0.3101341724395752 ,
   1.6309404373168945 ,  2.008157253265381  ,  0.25483274459838867,
   2.037174701690674  ,  1.0669955015182495 ,  1.6352945566177368 ,
   0.09142178297042847,  0.6787208914756775 ,  0.31872454285621643,
  -2.205458164215088  ,  0.8348409533500671 , -0.8971137404441833 ,
   0.1764504462480545 ,  2.046564817428589  ,  0.952080249786377  ,
   1.2842302322387695 ,  0.5182481408119202 ,  1.6045526266098022 ,
  -0.9570581912994385 , -2.167872190475464  , -0.04837090522050858,
  -0.2733888626098633 ,  1.543865442276001  ,  1.7920238971710205 ,
   1.9659976959228516 ,  0.368849515914917  ,  0.00031335843959823,
   0.8939148783683777 ,  1.7451649904251099 , -0.7170341610908508 ,
  -1.4013224840164185 ,  1.07520592212677   ,  0.6602697372436523 ,
  -0.5404913425445557 , -0.6159988641738892 ,  1.0909500122070312 ,
  -0.25104719400405884, -0.6175275444984436 , -0.7367951273918152 ,
  -0.37365373969078064,  2.129593849182129  ,  3.3836917877197266 ,
   0.6390048265457153 , -1.0943400859832764 ,  0.48441460728645325,
   2.415980577468872  ],
 [ 0.3635784685611725 , -0.03775424510240555, -0.3808891773223877 ,
   0.01391835045069456, -0.1122836023569107 , -0.11948554217815399,
  -0.19221478700637817, -0.3014318346977234 , -3.6066551208496094 ,
   0.06307491660118103,  0.32263854146003723,  0.25811323523521423,
   0.00462251808494329,  0.7553163170814514 ,  0.7119632959365845 ,
  -0.16143706440925598,  0.3535638749599457 ,  0.48308876156806946,
  -0.01365880202502012,  0.6116787791252136 ,  0.14011986553668976,
  -0.36362138390541077,  0.24473847448825836, -0.25190749764442444,
  -0.3567950129508972 , -0.21001236140727997, -0.6370506882667542 ,
   0.1680702418088913 , -0.06509244441986084, -0.32834625244140625,
   0.0941695049405098 , -1.5016884803771973 ,  0.14302247762680054,
  -1.2042579650878906 , -0.46770721673965454, -0.5847122669219971 ,
   0.7495575547218323 , -1.2030415534973145 , -0.33078503608703613,
  -0.03893529251217842, -0.8637963533401489 , -0.566372811794281  ,
  -0.1709292232990265 , -1.798676609992981  ,  0.5180931687355042 ,
  -1.8508552312850952 , -2.174185276031494  , -0.8150132894515991 ,
   0.34656330943107605, -0.32984793186187744, -0.8776566386222839 ,
   0.04531084001064301,  0.21075871586799622, -1.0914791822433472 ,
   0.30181002616882324, -0.8054832220077515 ,  0.21825793385505676,
   0.7099158763885498 , -0.7328619360923767 , -0.34873658418655396,
  -0.42411795258522034, -0.08732089400291443,  0.3475746512413025 ,
  -0.09059901535511017],
 [ 0.3297852873802185 ,  0.3849129378795624 , -0.34456464648246765,
  -0.5064830183982849 , -0.08223360031843185, -0.3798961937427521 ,
   0.6545794010162354 , -0.6589580178260803 , -2.163295030593872  ,
  -0.11888853460550308,  0.6504226922988892 ,  0.8261750936508179 ,
  -0.08543524891138077, -0.47661489248275757, -0.3603595793247223 ,
  -1.048093318939209  ,  0.6606580018997192 ,  0.2844173014163971 ,
   0.43186110258102417,  0.712158203125     ,  0.40361225605010986,
  -0.07544445991516113,  0.03435656428337097, -0.21922191977500916,
  -2.32218599319458   , -0.39449724555015564,  0.7018371224403381 ,
   0.2095060646533966 ,  0.07419563829898834, -1.2061691284179688 ,
   0.10574361681938171, -1.2834117412567139 , -0.38251155614852905,
   0.185675248503685  ,  0.35648152232170105,  0.28518906235694885,
   1.3046265840530396 , -1.3395687341690063 ,  0.9639758467674255 ,
   0.4951057434082031 , -1.420487403869629  , -0.8048605918884277 ,
  -0.05046109855175018, -0.32438233494758606,  0.02966074086725712,
  -1.2298667430877686 , -0.7593344449996948 , -0.39962238073349   ,
  -0.6642557382583618 ,  0.09162828326225281,  0.4222630262374878 ,
   0.35641172528266907,  0.5253413915634155 ,  0.6511790752410889 ,
   0.8844560384750366 ,  0.5222667455673218 ,  0.831587553024292  ,
   0.5317975878715515 , -0.5752126574516296 ,  0.2860458493232727 ,
  -0.13548438251018524,  0.31776174902915955,  0.5464214086532593 ,
   0.19826087355613708],
 [ 0.09293144196271896,  0.05220740288496017, -0.433401882648468  ,
   0.35610872507095337,  0.29958945512771606,  0.35250821709632874,
  -0.2046949416399002 ,  0.6187398433685303 ,  0.1413213610649109 ,
   0.09946248680353165, -0.4579385817050934 , -1.4562060832977295 ,
  -0.03578590229153633, -0.25377440452575684, -0.0383509024977684 ,
  -0.00836508441716433,  0.01814510487020016,  0.22188739478588104,
  -0.2967168092727661 , -0.4555507302284241 ,  0.25284820795059204,
  -1.37296724319458   , -0.09281700104475021,  0.22951120138168335,
  -0.41482770442962646, -0.39778584241867065, -0.0206359401345253 ,
  -0.27572503685951233,  0.22463802993297577, -0.953296422958374  ,
   0.22638945281505585,  0.33292436599731445, -0.15513037145137787,
   0.24748089909553528,  0.7168954014778137 ,  0.38197144865989685,
  -1.0037082433700562 , -0.4646604657173157 , -0.26404494047164917,
  -0.10389075428247452,  0.22910985350608826, -0.15534095466136932,
   0.38499438762664795, -0.42622020840644836, -0.10096143931150436,
   0.14245036244392395,  0.01139182690531015, -0.09811847656965256,
  -0.23557604849338531, -0.9335441589355469 , -0.07039640098810196,
   1.0039008855819702 , -0.31216126680374146,  0.3499301075935364 ,
  -0.15409187972545624, -0.01688984222710133, -0.5486624240875244 ,
  -0.4376316964626312 , -0.49678516387939453,  0.505079984664917  ,
   0.35414770245552063,  0.13195303082466125,  0.36714544892311096,
  -0.30089202523231506],
 [-0.1365707665681839 ,  0.02248358726501465, -1.3120005130767822 ,
  -0.12544313073158264, -0.03855382278561592, -0.501073956489563  ,
  -0.5084428787231445 , -0.08943025767803192, -0.08500441908836365,
   0.16941902041435242, -1.9824937582015991 , -0.9111226797103882 ,
   0.08386842161417007, -0.1928621381521225 , -0.6671649813652039 ,
  -0.17388607561588287, -0.2284291833639145 , -0.5620332360267639 ,
  -0.2494387924671173 , -0.05948694422841072, -0.961576521396637  ,
  -0.5389651656150818 , -0.09152236580848694, -0.29363998770713806,
  -0.01058007683604956,  0.07572877407073975, -0.07053349167108536,
  -0.22106443345546722, -0.13142971694469452, -0.4045053720474243 ,
  -0.09721830487251282, -1.7144452333450317 ,  0.09383496642112732,
  -0.39281368255615234,  0.27149760723114014, -0.13802272081375122,
  -1.1173357963562012 ,  0.06567041575908661, -0.05674347653985023,
  -0.1419374644756317 , -1.5963107347488403 ,  0.09444563835859299,
  -0.12253624945878983, -0.13757477700710297, -0.2784281373023987 ,
  -0.8810667991638184 , -2.710198402404785  , -0.32491251826286316,
  -1.3597676753997803 , -0.01221225783228874, -0.6594824194908142 ,
  -0.1449504792690277 , -0.03271446749567986, -0.3508237600326538 ,
  -0.8262557983398438 , -0.5085166692733765 , -0.0849287137389183 ,
  -0.6979329586029053 , -0.781786322593689  , -0.8695995807647705 ,
  -0.40403956174850464, -0.05016331002116203,  0.17207759618759155,
   0.353942334651947  ],
 [-1.3087022304534912 , -0.20544879138469696,  0.5046364665031433 ,
   0.00467455061152577, -0.16127151250839233, -1.018524408340454  ,
  -0.32434940338134766,  0.7637364864349365 ,  0.09000013023614883,
  -0.6859164237976074 , -1.7744101285934448 ,  0.6159985065460205 ,
   0.13234703242778778, -1.19566810131073   , -1.1879390478134155 ,
   0.21142850816249847,  2.0538487434387207 ,  0.14928054809570312,
  -3.064640522003174  , -0.11177009344100952,  1.2990474700927734 ,
   1.4925172328948975 , -0.8265635967254639 ,  1.3682557344436646 ,
   0.01539332512766123,  0.07224991917610168,  0.7773665189743042 ,
   0.40803584456443787, -0.35193759202957153,  0.2705695629119873 ,
  -0.21969014406204224,  1.630517840385437  ,  0.6552842259407043 ,
   0.6110966801643372 ,  0.32708120346069336,  0.448813259601593  ,
   1.0507898330688477 , -0.498090535402298  , -1.5340803861618042 ,
   1.028769850730896  , -1.1251766681671143 ,  0.3339804708957672 ,
   0.47876784205436707, -0.08956821262836456, -0.9392167925834656 ,
  -1.4687038660049438 ,  1.1865390539169312 , -1.1591373682022095 ,
  -2.132014513015747  ,  0.2944604158401489 ,  1.8888945579528809 ,
   0.4594033360481262 , -1.5172308683395386 ,  0.6665694117546082 ,
  -5.449181079864502  , -2.2744007110595703 ,  0.3159049153327942 ,
   1.7489794492721558 ,  0.9135564565658569 ,  0.7305818796157837 ,
   1.9865014553070068 ,  0.1968831866979599 ,  1.09598970413208   ,
   0.9728906750679016 ],
 [-0.22720281779766083, -0.17475223541259766,  0.3026117980480194 ,
  -0.14855854213237762,  0.09204604476690292, -0.26340198516845703,
   0.07530003786087036,  0.08412773907184601, -3.3463597297668457 ,
   0.20404097437858582,  0.15383599698543549, -1.778490662574768  ,
  -0.07190912961959839, -0.20233502984046936, -0.699522852897644  ,
  -0.03121247701346874,  0.04061730951070786, -0.8763463497161865 ,
   0.6139236092567444 , -0.26145413517951965, -0.2546950578689575 ,
   0.09543264657258987,  0.20673416554927826, -0.5622171759605408 ,
   0.12954702973365784, -0.15484924614429474, -0.01025600638240576,
  -0.18949942290782928,  0.06190470233559608, -0.07224369794130325,
   0.01334614306688309, -2.4844443798065186 ,  0.14762908220291138,
  -0.811859130859375  , -1.3124949932098389 , -0.2584991753101349 ,
  -0.0963394045829773 ,  0.04794881120324135, -0.30599525570869446,
  -0.18974021077156067,  0.6299667358398438 , -1.1746273040771484 ,
  -0.6054652333259583 , -0.39382728934288025, -0.23017537593841553,
   0.27781111001968384, -0.583124041557312  , -0.2680826187133789 ,
  -0.422630250453949  ,  0.03663719072937965, -0.24497878551483154,
  -0.3007429838180542 ,  0.00920033175498247,  0.05710862949490547,
  -0.5705448985099792 , -0.3963961601257324 , -0.11712903529405594,
  -0.8112984299659729 ,  0.29103681445121765, -0.8251125812530518 ,
  -0.20622913539409637, -0.10222871601581573, -1.1603602170944214 ,
  -0.24454453587532043],
 [-1.6187835931777954 ,  0.33318495750427246, -2.617697238922119  ,
   0.1176602765917778 , -0.5179714560508728 ,  0.49092036485671997,
  -0.21948125958442688,  0.17875497043132782, -0.49107375741004944,
  -0.13538502156734467,  0.45245712995529175, -1.87162184715271   ,
  -0.9461528062820435 , -0.45056772232055664, -0.5306101441383362 ,
  -0.1081366091966629 , -0.12502950429916382, -0.2782200276851654 ,
  -0.623502790927887  , -0.16805696487426758,  0.5659271478652954 ,
  -0.09021670371294022, -0.2706211507320404 ,  0.19121544063091278,
  -0.732014536857605  ,  0.2860420048236847 ,  0.46142348647117615,
   0.3746066987514496 , -0.239175945520401  ,  0.00022786141198594,
  -0.09021665900945663, -0.3658658266067505 , -0.48217546939849854,
  -0.04224877059459686,  0.11051564663648605,  0.5382091403007507 ,
  -0.04369552060961723, -0.6279124617576599 ,  0.5576520562171936 ,
   0.35263529419898987, -0.01650173589587212, -1.207944631576538  ,
   0.3647304177284241 ,  0.5208340883255005 ,  0.09449765086174011,
   0.561240553855896  , -0.11368781328201294, -0.22206956148147583,
   0.2125256359577179 , -2.09189510345459   ,  0.6160039901733398 ,
   0.3917495310306549 , -0.03786352276802063,  0.22806669771671295,
   0.24891828000545502,  0.08483254909515381,  0.0421086922287941 ,
  -1.7317054271697998 , -0.09027302265167236,  0.07493026554584503,
  -0.15773703157901764,  0.01652160286903381, -0.14944805204868317,
  -0.05754176154732704],
 [-0.7670657634735107 , -0.03635071218013763, -1.1565423011779785 ,
  -0.15853871405124664, -0.19205524027347565, -0.5308588743209839 ,
  -0.4163520634174347 , -0.37443530559539795, -0.22719933092594147,
   0.1761104315519333 ,  0.5228587985038757 , -1.4424303770065308 ,
  -0.38953372836112976,  0.01253176014870405, -0.38777390122413635,
  -0.6054288744926453 ,  0.06310354173183441, -0.3090502917766571 ,
   0.14766179025173187, -0.09105496108531952, -0.00732848513871431,
  -0.32464638352394104,  0.3469579219818115 , -0.23238636553287506,
  -0.9158776998519897 , -0.17717581987380981,  0.1713714748620987 ,
   0.6325967311859131 , -0.22710908949375153, -0.6563481092453003 ,
  -0.01272954978048801, -0.756024181842804  , -0.38339269161224365,
  -1.6836638450622559 , -0.4700629711151123 ,  0.32793760299682617,
  -1.4025222063064575 ,  0.07033853977918625,  0.13531845808029175,
   0.28273725509643555, -0.01576234959065914,  0.03914336487650871,
  -0.4446558952331543 , -0.03689316287636757, -0.00146129843778908,
   0.10587334632873535, -1.1703155040740967 ,  0.03717175126075745,
  -0.8485881686210632 ,  0.12260894477367401,  0.3155483901500702 ,
   0.20161163806915283, -0.3622400462627411 , -1.7722270488739014 ,
  -0.49269890785217285, -1.2143912315368652 , -0.19633710384368896,
   0.5083747506141663 ,  0.14849159121513367, -0.9349305629730225 ,
  -0.30271297693252563, -0.12016895413398743, -0.3054042160511017 ,
   0.33403146266937256],
 [ 0.4421294331550598 , -0.6749699115753174 ,  0.10195370763540268,
   0.3831283450126648 , -0.29953187704086304, -0.5668546557426453 ,
  -0.47993868589401245,  1.005189061164856  ,  0.7088198065757751 ,
  -0.07691411674022675,  0.14837270975112915, -0.9362175464630127 ,
  -0.5595210194587708 , -1.0625016689300537 ,  0.29119619727134705,
   1.0426476001739502 , -0.2333715409040451 , -0.38645699620246887,
  -2.0632059574127197 ,  0.4051901400089264 ,  0.6650665998458862 ,
   1.91502845287323   , -0.6888985633850098 ,  1.0239453315734863 ,
   0.9528484344482422 ,  0.1075742244720459 ,  0.7802941799163818 ,
  -1.3685011863708496 , -0.6055769324302673 ,  1.2831004858016968 ,
  -0.53877854347229   ,  0.7457475066184998 ,  0.6203931570053101 ,
   0.9170665144920349 , -1.1864099502563477 ,  0.2599427402019501 ,
  -1.3806495666503906 ,  0.9773688316345215 , -2.6333205699920654 ,
  -0.11926094442605972,  0.6325444579124451 ,  0.0745011642575264 ,
   0.19082976877689362,  0.5875570774078369 , -0.277087539434433  ,
   0.06430190801620483, -2.477322578430176  ,  0.9504378437995911 ,
   0.7302904725074768 ,  0.49244609475135803,  0.2556019425392151 ,
  -1.5235618352890015 ,  0.02167192846536636,  2.0414836406707764 ,
  -0.03610728681087494, -0.08008924871683121,  0.03161771968007088,
   0.1118183359503746 ,  1.0149208307266235 ,  0.5720528364181519 ,
   1.9354952573776245 ,  0.5079418420791626 ,  1.8279224634170532 ,
   1.754132866859436  ],
 [-0.1881645917892456 , -0.20406244695186615, -0.4094448685646057 ,
  -0.46530961990356445, -0.34443801641464233, -0.618889570236206  ,
   0.08217474818229675,  0.3943043351173401 ,  0.49512889981269836,
  -0.02383096516132355,  0.08015865832567215,  0.1392495036125183 ,
   0.22423914074897766,  0.0801749899983406 , -0.07478372752666473,
  -0.03479066491127014, -0.09436152130365372, -0.3045247793197632 ,
  -0.7809871435165405 , -0.42423737049102783,  0.13685542345046997,
   0.621582567691803  ,  0.02465054951608181,  0.5336669683456421 ,
  -0.49025794863700867,  0.0711212009191513 ,  0.4976101517677307 ,
   0.08900382369756699, -0.4104900062084198 , -0.6995548605918884 ,
  -0.04515161737799644,  0.9129209518432617 ,  0.11562500149011612,
   0.02683675102889538, -1.0261943340301514 ,  0.06419764459133148,
   0.49238476157188416, -0.8843525648117065 , -0.1561535745859146 ,
   0.11440662294626236,  0.18289239704608917, -4.597512722015381  ,
   0.613038182258606  ,  0.2863066494464874 ,  0.27886009216308594,
   0.38939517736434937, -0.26962795853614807, -0.6585791110992432 ,
  -0.5058031678199768 , -0.5266358852386475 ,  0.5887278318405151 ,
  -0.2954483926296234 , -0.702627420425415  ,  0.16056163609027863,
   0.4851406514644623 ,  0.4373454749584198 ,  0.17641809582710266,
  -0.30310356616973877,  0.6303569078445435 ,  0.4686393737792969 ,
   0.6135244965553284 , -0.10466215014457703, -0.07759501039981842,
   0.32375380396842957],
 [-0.1632499247789383 , -0.03212736919522285,  0.1304803490638733 ,
  -0.48796895146369934, -0.01561215054243803,  0.7694656252861023 ,
   0.14924563467502594, -0.4116172790527344 , -0.44882190227508545,
  -0.12940242886543274, -0.5902892351150513 , -0.42313694953918457,
   0.2576755881309509 , -0.02360491454601288, -0.6162157654762268 ,
  -1.4689077138900757 ,  0.529792070388794  , -0.7992054224014282 ,
   0.10278007388114929,  0.7367291450500488 , -0.8016929626464844 ,
   1.2537223100662231 ,  0.0879243016242981 ,  0.8914517760276794 ,
   0.00329121970571578,  0.19397781789302826, -0.13174252212047577,
  -0.22239895164966583,  0.0270268302410841 , -0.8051022291183472 ,
   0.03348395973443985,  0.02422666363418102,  0.36200425028800964,
   0.2989359498023987 ,  0.3959503769874573 ,  0.25266680121421814,
  -1.265192985534668  ,  0.456073135137558  , -0.8426313996315002 ,
   0.0914936289191246 ,  0.24850882589817047,  0.3114745020866394 ,
   0.4719533920288086 ,  0.48870471119880676, -0.7233952879905701 ,
  -0.13335411250591278,  0.8251921534538269 ,  0.7215867638587952 ,
  -0.7613245844841003 ,  0.08714582026004791, -0.6901677846908569 ,
   0.00159208744298667, -0.34450650215148926,  0.6565077900886536 ,
  -1.2926863431930542 , -0.5366967916488647 , -1.0628721714019775 ,
  -3.196495771408081  ,  1.6984710693359375 ,  0.49348101019859314,
   0.8057371973991394 ,  0.07199019193649292,  0.44208744168281555,
   1.6548594236373901 ],
 [ 0.33590462803840637,  0.08709194511175156, -0.6283218860626221 ,
  -0.03229549527168274, -0.16574160754680634, -1.1015105247497559 ,
  -0.3835669159889221 , -0.191634401679039  ,  0.04658441245555878,
   0.1770087033510208 ,  0.11122264713048935, -0.788807213306427  ,
  -0.21794362366199493, -0.7754644751548767 , -0.13861937820911407,
  -0.8109281063079834 ,  0.20106874406337738, -0.3887423574924469 ,
  -1.4892746210098267 ,  0.6869708895683289 , -0.7302545309066772 ,
  -0.3395884037017822 ,  0.13782787322998047, -0.508293628692627  ,
  -0.00423265574499965, -0.44376280903816223, -0.27471664547920227,
  -0.05328738316893578, -0.03181247785687447,  0.4851773679256439 ,
   0.06667597591876984,  0.09082519263029099,  0.09715686738491058,
  -0.9634218215942383 ,  0.1933283656835556 , -0.51700758934021   ,
   0.23584166169166565,  0.40931135416030884,  0.02486339770257473,
  -0.28211653232574463, -0.03167156875133514, -0.48568490147590637,
   0.3767140805721283 , -1.1038950681686401 , -0.31581196188926697,
  -1.8554556369781494 ,  0.2932666838169098 , -1.669202446937561  ,
  -1.7463713884353638 , -0.3869962692260742 , -0.9331029057502747 ,
  -0.7890218496322632 , -0.3366309106349945 ,  0.1046898290514946 ,
  -0.05414837598800659, -1.1666394472122192 , -0.27563464641571045,
   0.14159245789051056, -0.7170552611351013 ,  0.3345694839954376 ,
  -0.6117130517959595 ,  0.1046191155910492 , -0.2091917097568512 ,
  -0.8681392669677734 ],
 [-0.972101092338562  ,  0.13308610022068024,  0.6223839521408081 ,
   0.01359173469245434,  0.40074071288108826,  0.01995453797280788,
  -0.6155511140823364 ,  0.2500397264957428 ,  0.3184904158115387 ,
  -0.3408176004886627 , -0.5539898872375488 ,  0.5015614032745361 ,
   0.5372931361198425 ,  0.01633157767355442,  0.656406044960022  ,
   0.23669512569904327,  0.14007586240768433, -1.2705632448196411 ,
   0.479604572057724  , -0.5186741352081299 ,  0.21772357821464539,
   0.45517465472221375,  0.21592186391353607,  0.51726233959198   ,
   0.5984007716178894 ,  0.7525689601898193 , -0.10420894622802734,
   0.4817117750644684 , -0.8154650926589966 ,  0.00690743466839194,
  -0.10341477394104004,  0.855317234992981  , -0.1273202747106552 ,
   1.4951870441436768 ,  0.490467369556427  ,  0.14215470850467682,
  -1.1478469371795654 , -0.02368340268731117, -0.2287513017654419 ,
   0.4940498471260071 ,  0.2955801784992218 , -0.2787187695503235 ,
  -0.5806426405906677 , -0.49988359212875366,  1.0701996088027954 ,
   1.0992259979248047 ,  0.662449300289154  , -0.7282886505126953 ,
  -1.920220136642456  , -0.20521576702594757, -0.3151018023490906 ,
  -0.01539377681910992, -0.08587990701198578, -0.538987934589386  ,
   0.21941408514976501,  0.31843212246894836,  0.8170549273490906 ,
   1.0733481645584106 , -0.09483326971530914,  0.01254136115312576,
   0.4440952241420746 ,  0.2079332172870636 , -0.11876486986875534,
   0.5019862055778503 ],
 [-1.3134288787841797 , -0.00987064559012651, -0.14367835223674774,
  -0.290371298789978  ,  0.3551839292049408 ,  0.3001995384693146 ,
  -0.01071365177631378, -0.00989076402038336, -0.48001840710639954,
   0.14387407898902893,  0.22667527198791504, -0.9069556593894958 ,
  -0.5032172203063965 , -1.0820531845092773 , -3.0248754024505615 ,
   0.6403908729553223 ,  0.5712116956710815 , -0.5828886032104492 ,
  -1.676106572151184  ,  0.14024925231933594,  0.40089643001556396,
  -0.5037803649902344 , -0.3984098434448242 , -1.0985138416290283 ,
   0.22379878163337708,  0.08312658220529556, -0.12909474968910217,
   0.5049704313278198 ,  0.06187667697668076,  1.0768553018569946 ,
   0.17042438685894012,  0.2647905945777893 ,  0.3427986204624176 ,
   0.6497583389282227 ,  0.42879611253738403,  0.16899697482585907,
  -0.3212413489818573 , -0.39807969331741333,  0.25405433773994446,
  -0.7004515528678894 , -0.1435082107782364 ,  0.6014649868011475 ,
   0.07552210986614227,  0.40124350786209106, -1.2428836822509766 ,
   0.2879962623119354 , -1.1367313861846924 ,  0.6221741437911987 ,
   0.49259525537490845, -0.46360790729522705, -0.07633121311664581,
  -0.04503185302019119,  0.40606123208999634, -3.1628501415252686 ,
   0.02648710273206234, -0.6820202469825745 , -0.06657500565052032,
  -2.536590099334717  , -0.24262358248233795, -0.345660924911499  ,
  -1.2097409963607788 ,  0.3766835331916809 ,  0.18235206604003906,
   0.01375225745141506],
 [ 0.6703559160232544 , -1.9639474153518677 , -0.36501583456993103,
   0.1360737830400467 ,  0.9144795536994934 , -1.195233702659607  ,
   0.08230791240930557,  0.37970787286758423,  2.0192854404449463 ,
  -0.8734833598136902 , -0.00882686581462622,  0.8243486881256104 ,
  -0.2892111539840698 ,  0.0826672613620758 , -0.01737666688859463,
  -1.1902626752853394 ,  0.9078662991523743 , -0.7285191416740417 ,
  -0.9993886351585388 , -0.0533486120402813 , -0.6386407613754272 ,
   1.5524110794067383 , -2.200436592102051  ,  0.9014182090759277 ,
   0.33877360820770264,  0.5671698451042175 ,  0.8975374698638916 ,
  -0.17637129127979279,  0.164887934923172  , -1.911993384361267  ,
   0.2700376510620117 ,  1.3449070453643799 ,  0.5737331509590149 ,
   0.3089354634284973 ,  1.1474558115005493 ,  0.9498081207275391 ,
   0.2637328803539276 ,  0.32036423683166504, -0.0222929809242487 ,
   0.679648220539093  ,  1.9351102113723755 , -0.26453328132629395,
   0.5888585448265076 ,  0.8113709092140198 ,  1.3284472227096558 ,
  -0.02187165059149265, -0.74848872423172   , -0.3135102689266205 ,
  -0.2911321520805359 ,  0.01897324062883854,  0.6625237464904785 ,
   1.9098631143569946 ,  0.9651534557342529 ,  1.638089895248413  ,
  -2.0241880416870117 ,  0.22464053332805634,  1.1108486652374268 ,
   0.5076020956039429 ,  1.8498573303222656 ,  1.99915611743927   ,
  -0.17546428740024567, -0.15555739402770996,  1.429030179977417  ,
   0.23718157410621643],
 [-0.8536322116851807 ,  0.081193707883358  , -0.16592183709144592,
  -0.01122220791876316,  0.11839018017053604,  0.31071701645851135,
   0.0420321449637413 ,  0.32653599977493286, -1.2895056009292603 ,
   0.08668301999568939, -0.7906782627105713 ,  0.11296536028385162,
  -1.2024060487747192 ,  0.07499703019857407, -0.7857893705368042 ,
  -0.32100799679756165,  0.3986305296421051 , -0.36098217964172363,
   0.04900489002466202, -0.2577294707298279 , -0.7595047354698181 ,
  -0.00851002521812916,  0.3026595115661621 ,  0.00971110537648201,
  -0.2556520104408264 ,  0.13211718201637268,  0.20689168572425842,
   0.2849426865577698 , -0.04980006814002991,  0.6972883939743042 ,
  -0.1268967241048813 , -0.46318018436431885,  0.14294375479221344,
  -1.5118976831436157 ,  0.5281343460083008 , -1.5601774454116821 ,
  -0.8886704444885254 ,  0.11157657206058502,  0.16596364974975586,
  -0.5139694809913635 , -0.3878842890262604 , -0.28757044672966003,
  -0.40875500440597534, -0.8850319981575012 , -0.6859318614006042 ,
   0.7836675643920898 , -0.7298721075057983 , -0.37112686038017273,
   0.12069187313318253, -0.09575418382883072, -0.48641082644462585,
   0.13434068858623505,  0.515400767326355  , -0.9431649446487427 ,
  -0.27228015661239624,  0.02511385455727577,  0.12353141605854034,
  -0.16541916131973267, -0.988436222076416  , -0.6032868027687073 ,
  -0.21528558433055878,  0.09278436750173569, -0.74349445104599   ,
  -1.4737590551376343 ],
 [ 0.3731946647167206 ,  0.31808680295944214,  0.6496891379356384 ,
  -0.1169150322675705 , -0.35595589876174927,  0.4528432786464691 ,
  -1.3570674657821655 ,  0.06371483206748962,  0.01734292693436146,
  -0.03732651844620705,  0.02841456979513168, -0.99290531873703   ,
   0.1320953518152237 , -0.23394353687763214,  0.5852835178375244 ,
  -0.8412747979164124 , -0.18018563091754913,  0.14772984385490417,
   0.06979236751794815,  0.7529135346412659 ,  0.6132846474647522 ,
   0.1857466846704483 ,  0.01398375071585178, -0.0348486602306366 ,
  -1.4994264841079712 , -0.21400007605552673, -0.21733658015727997,
  -0.9997352361679077 , -0.6662983298301697 , -1.1684250831604004 ,
  -0.3310949206352234 ,  0.39073246717453003,  0.03996904939413071,
  -0.5356313586235046 ,  0.25714975595474243,  0.23194797337055206,
   0.9650381207466125 , -0.3202803134918213 ,  0.4981120526790619 ,
   0.14093442261219025,  0.03191597387194633, -0.02094803564250469,
  -0.33666548132896423, -0.07473313808441162,  0.03119109943509102,
   0.49192166328430176,  0.5044804811477661 ,  0.40891310572624207,
   0.5050849914550781 , -0.2400265634059906 ,  0.315897136926651  ,
   0.20288625359535217,  0.47374218702316284,  0.5872210264205933 ,
   0.36683887243270874,  0.45619693398475647, -0.11207863688468933,
  -0.8147047758102417 ,  0.7783375978469849 ,  0.00315456627868116,
   0.26463618874549866,  0.20985285937786102, -0.03020120039582253,
  -0.7794872522354126 ],
 [ 0.4767239987850189 , -0.2012946456670761 , -0.5457763671875    ,
   0.00728748179972172, -0.5014926195144653 , -0.8048878908157349 ,
  -0.34555870294570923, -0.45413661003112793, -0.33907169103622437,
   0.3434869647026062 , -1.0697509050369263 , -0.2688573896884918 ,
  -0.2899447977542877 ,  0.31675985455513   , -0.9365099668502808 ,
  -0.5338965654373169 ,  0.8063875436782837 ,  0.12506121397018433,
  -1.0524508953094482 , -0.28288137912750244,  0.12535935640335083,
  -0.34363287687301636,  0.02061162330210209, -0.312093585729599  ,
  -0.4321315586566925 , -1.2746212482452393 ,  0.6217024922370911 ,
   0.12248545140028   ,  0.20714476704597473, -0.7329071760177612 ,
   0.06437632441520691, -2.454031467437744  , -0.03605853021144867,
  -2.4688491821289062 , -0.4157275855541229 , -0.2711254060268402 ,
   0.3620550036430359 , -0.28945615887641907,  0.5869243741035461 ,
   0.1462717205286026 , -0.09881479293107986, -0.7379676699638367 ,
  -0.35853099822998047, -0.5708494782447815 , -0.46319863200187683,
  -1.213402271270752  ,  0.33878976106643677, -0.3180907964706421 ,
  -2.0972814559936523 , -0.3259175717830658 , -0.1327279806137085 ,
  -0.6644667387008667 , -1.890568494796753  , -0.5352121591567993 ,
   0.2935951352119446 , -1.5346779823303223 ,  0.08745666593313217,
  -0.7014297842979431 , -0.1989714354276657 , -0.0041016056202352 ,
  -0.11453458666801453, -0.2279507964849472 , -0.43168431520462036,
  -1.0479717254638672 ],
 [ 0.7238060832023621 ,  0.3083142638206482 ,  1.1992542743682861 ,
   0.21623599529266357, -1.6657567024230957 , -1.598401665687561  ,
  -0.43871593475341797,  0.2601301372051239 , -0.06785494834184647,
   0.11741276830434799, -0.5070216059684753 ,  0.23700425028800964,
  -1.2475937604904175 , -0.6289314031600952 , -1.391858458518982  ,
  -1.2394719123840332 ,  0.5446682572364807 , -0.5305533409118652 ,
   0.27800682187080383, -0.4169197976589203 , -0.2131975144147873 ,
   0.01711880788207054,  0.08158548921346664, -0.16721557080745697,
   0.1455264389514923 , -2.5392348766326904 , -0.9437446594238281 ,
  -0.16321200132369995,  0.23687176406383514, -0.2652343809604645 ,
  -0.45989730954170227, -0.7765651941299438 ,  0.23571759462356567,
  -0.611822783946991  ,  0.8464687466621399 ,  0.1960245966911316 ,
  -0.6064108610153198 , -0.6194332242012024 , -0.5450482368469238 ,
   0.4432755410671234 , -1.5052229166030884 ,  1.2465660572052002 ,
  -0.8136296272277832 ,  0.8382037878036499 ,  0.7414838671684265 ,
  -5.108951091766357  ,  0.18981321156024933, -3.22444486618042   ,
  -0.19900435209274292, -0.5746928453445435 , -0.13260911405086517,
   0.3624846041202545 ,  0.11799101531505585, -0.2554733455181122 ,
   0.27006980776786804, -0.4801081418991089 , -0.429834246635437  ,
  -1.0919626951217651 , -1.6485077142715454 , -1.0030632019042969 ,
   0.19333374500274658,  0.07062143087387085, -0.11552838236093521,
   0.12390579283237457],
 [-1.9844636917114258 ,  0.14315055310726166, -0.39914119243621826,
  -0.19234147667884827, -0.23453037440776825,  0.13832393288612366,
  -2.0485663414001465 , -0.4952928125858307 ,  0.18827161192893982,
   0.21556036174297333, -1.8859236240386963 , -0.30833303928375244,
   0.35248711705207825, -1.3027676343917847 , -1.9355627298355103 ,
  -0.2864035665988922 ,  0.04396604374051094,  0.3378128409385681 ,
   0.32863059639930725,  0.3569799065589905 ,  0.22505895793437958,
  -0.3885241746902466 , -0.3512314260005951 , -0.10173333436250687,
   0.5673316717147827 ,  0.03798718750476837,  0.14154928922653198,
  -0.7669193744659424 , -0.14841420948505402, -0.1689094454050064 ,
  -0.0488537922501564 , -1.0855904817581177 ,  0.02462673000991344,
  -0.18846018612384796,  0.07377682626247406, -0.6399075388908386 ,
   0.2619045078754425 , -0.25707364082336426, -0.31140559911727905,
  -0.6906446814537048 , -0.26772889494895935, -0.03604034706950188,
  -1.0540510416030884 , -0.1643844097852707 , -0.5379536747932434 ,
  -1.2233726978302002 ,  0.13724318146705627, -0.7645106315612793 ,
  -0.5841073989868164 ,  0.06027363613247871, -0.02826453000307083,
  -0.10418776422739029, -0.8101571202278137 ,  1.2325197458267212 ,
  -0.8772007822990417 , -0.342337042093277  ,  0.0794157162308693 ,
  -1.849957823753357  ,  0.6027392745018005 , -0.555990993976593  ,
   0.29166385531425476, -0.8186071515083313 , -1.7813925743103027 ,
   0.03007291257381439],
 [ 0.05826987698674202, -0.03611262142658234, -0.5386663675308228 ,
  -0.3245064914226532 , -0.8282179236412048 ,  0.02786994725465775,
  -0.04185618087649345,  0.00134591187816113,  0.30965492129325867,
  -0.3236113488674164 ,  0.15121860802173615, -0.02175247110426426,
  -0.17024701833724976, -0.01628586649894714, -0.08215029537677765,
  -0.0619061291217804 , -0.5469932556152344 , -0.02421218156814575,
  -2.1064579486846924 , -0.3985457420349121 , -0.16511280834674835,
  -0.4840798079967499 ,  0.0977076068520546 , -0.21481530368328094,
   0.05635774508118629, -0.4654308259487152 , -0.25321725010871887,
  -0.11008662730455399,  0.03957392275333405, -0.41629016399383545,
   0.2845917046070099 ,  0.04122627153992653,  0.2021971195936203 ,
   0.28983280062675476, -0.5107815265655518 ,  0.25890323519706726,
  -0.277248352766037  ,  0.17316441237926483,  0.05213876813650131,
   0.05784148722887039,  0.19039581716060638,  0.21109142899513245,
  -0.3963402509689331 , -0.30882343649864197, -0.06422922015190125,
  -0.30104881525039673,  0.25564831495285034, -0.08767583221197128,
  -0.4598723351955414 , -0.35391050577163696, -0.5853233337402344 ,
  -0.18644097447395325, -0.1172599121928215 ,  0.1587674766778946 ,
  -0.15955819189548492,  0.1646731048822403 , -0.1577935367822647 ,
  -1.4239912033081055 ,  0.24859368801116943, -0.10774137079715729,
  -0.27967992424964905, -0.24270321428775787, -0.5397682785987854 ,
  -0.08663732558488846],
 [-0.6311883926391602 , -0.05593270063400269,  0.1999836266040802 ,
  -0.11110826581716537, -0.3934676945209503 , -0.17265008389949799,
  -0.1596023291349411 , -0.01352060865610838,  0.06928782910108566,
   0.14718325436115265,  0.5084403157234192 , -0.6184499859809875 ,
  -0.3981057405471802 , -0.3145603537559509 , -0.6716489791870117 ,
   0.06138486787676811,  0.02194200269877911, -0.47212210297584534,
   0.37352919578552246,  0.4136013090610504 ,  0.22417052090168   ,
  -0.12633267045021057, -0.3155391216278076 , -0.7078929543495178 ,
  -0.21957086026668549,  0.00309571041725576,  0.450724720954895  ,
  -0.7463274002075195 , -0.42448699474334717, -0.03742304071784019,
  -0.02268881909549236, -0.9243918061256409 , -0.11929940432310104,
   0.14585267007350922, -0.5722612738609314 , -0.2530537545681    ,
  -0.03281687945127487,  0.2379123866558075 , -0.16707225143909454,
  -0.3547992408275604 , -0.1958317905664444 ,  0.4132150113582611 ,
  -0.7281457185745239 ,  0.3072955012321472 , -0.25763511657714844,
  -1.0798863172531128 ,  0.17985160648822784, -0.00050670997006819,
   0.2699468731880188 , -0.07278180122375488, -0.04343826696276665,
   0.30115509033203125, -0.6257820129394531 ,  0.02766930684447289,
  -0.27805230021476746, -0.01357807032763958,  0.08236464112997055,
  -0.16397811472415924, -0.5074239373207092 , -0.4012449085712433 ,
  -0.4265807867050171 , -0.6059324741363525 , -0.276136189699173  ,
  -0.9507408738136292 ],
 [ 0.6141369938850403 , -0.03733363747596741,  1.7353700399398804 ,
  -0.26088353991508484, -0.6125116944313049 , -0.7332189083099365 ,
   0.103873111307621  ,  0.09949902445077896,  0.8012323975563049 ,
  -0.1537008285522461 , -0.0892186164855957 ,  0.615266740322113  ,
  -0.7056885957717896 , -0.8048356175422668 ,  0.1990237534046173 ,
   0.01441614702343941,  0.9164277911186218 , -0.2815970778465271 ,
  -0.4671674072742462 ,  0.5403156280517578 ,  0.6077056527137756 ,
   0.9475157260894775 ,  0.14818432927131653,  0.1669810563325882 ,
   0.501609206199646  ,  0.24833649396896362,  0.17698414623737335,
  -0.7519375681877136 ,  0.2208915501832962 , -1.3243969678878784 ,
   0.31427499651908875,  1.7297606468200684 ,  0.05521906539797783,
   0.16519148647785187,  1.144486427307129  ,  0.01363642048090696,
   0.12339799851179123, -0.13496874272823334,  0.6693469882011414 ,
  -0.32119232416152954,  0.12288757413625717, -0.32125356793403625,
   0.7219371199607849 , -0.6216057538986206 ,  0.3735799491405487 ,
   0.11775433272123337,  1.4403105974197388 ,  0.7484445571899414 ,
  -0.05688661709427834, -0.23778818547725677,  0.08853618800640106,
  -2.7486250400543213 , -0.5652479529380798 ,  0.5749073624610901 ,
   0.28964561223983765, -0.24401848018169403, -0.18094995617866516,
  -0.8947927951812744 ,  0.18648652732372284,  0.3431282043457031 ,
   0.6305423974990845 ,  0.310901403427124  , -1.054707407951355  ,
   1.1349895000457764 ],
 [ 1.82546865940094   , -0.9749038815498352 , -1.6753627061843872 ,
   0.5330774188041687 , -1.6989543437957764 ,  0.8503434062004089 ,
   0.09187531471252441, -0.7254294157028198 , -1.6130479574203491 ,
  -0.37477558851242065, -0.5450140237808228 , -0.16211166977882385,
   0.7617799639701843 ,  0.3558112382888794 ,  0.8759308457374573 ,
   0.6449962258338928 ,  0.770005464553833  ,  0.5111880898475647 ,
   0.2697359025478363 , -1.1980490684509277 , -0.6323133707046509 ,
   0.6773993372917175 , -0.00880565214902163,  0.18724189698696136,
   0.5910183191299438 ,  0.9157307147979736 ,  0.8650161623954773 ,
  -0.08497262001037598,  0.28418025374412537,  1.300171136856079  ,
  -0.2969476878643036 ,  0.7594452500343323 ,  0.21083244681358337,
  -0.26822590827941895,  0.9711146354675293 ,  0.2944917380809784 ,
   0.064396433532238  , -1.3069727420806885 , -0.5272508263587952 ,
  -0.6383886337280273 ,  0.5166672468185425 , -0.7872746586799622 ,
   0.08725068718194962, -0.11224480718374252,  0.4037712812423706 ,
   1.0685008764266968 , -0.5593172907829285 , -0.19545772671699524,
  -0.4346303939819336 , -0.2239258885383606 , -0.5818636417388916 ,
  -2.778801441192627  ,  0.06215307861566544, -0.24020005762577057,
   0.02146231941878796, -0.00303824478760362,  0.9759595990180969 ,
   1.542335867881775  , -0.5291455388069153 ,  0.6754724979400635 ,
   0.7264584898948669 ,  0.02176878415048122,  0.6922404170036316 ,
   0.09514699876308441],
 [ 0.3267882168292999 , -0.06503095477819443,  0.3482656180858612 ,
  -0.0108393682166934 ,  0.21006062626838684,  0.10420980304479599,
  -0.8969071507453918 ,  0.08680112659931183, -1.0611562728881836 ,
   0.26991239190101624, -0.5735976099967957 , -0.04188189655542374,
  -0.28479716181755066, -0.2161567509174347 , -0.1752697080373764 ,
  -0.19119255244731903, -0.8850646018981934 , -0.18075621128082275,
   0.1608264446258545 , -0.43065956234931946, -2.4151225090026855 ,
  -0.614852786064148  , -0.14644679427146912,  0.2537781596183777 ,
   0.25335654616355896, -0.18564753234386444, -0.09804405272006989,
  -0.6110571622848511 ,  0.00691348966211081, -0.6268945932388306 ,
  -0.20924842357635498, -1.1835246086120605 ,  0.3626694977283478 ,
   0.01568509638309479, -0.3616364300251007 , -0.4101715683937073 ,
   0.0570257231593132 , -0.22339880466461182, -0.03162426128983498,
   0.07605203986167908, -0.20084431767463684,  0.00749840540811419,
  -0.04607514664530754, -1.1095792055130005 ,  0.08014172315597534,
   0.34135982394218445, -0.7520434856414795 , -1.6489931344985962 ,
  -0.2726518511772156 , -0.17062707245349884, -0.47674643993377686,
   0.04594402387738228, -0.61219722032547   , -1.4104232788085938 ,
  -0.12468566000461578, -0.3073507249355316 , -0.01894880644977093,
  -0.8563677072525024 , -1.135832667350769  , -0.7653282880783081 ,
  -0.3343356251716614 , -0.29718437790870667, -1.4871855974197388 ,
  -0.06069543957710266],
 [-0.578862726688385  , -0.5594887137413025 , -0.11733199656009674,
  -0.7293803691864014 , -1.2333390712738037 ,  1.0058941841125488 ,
   1.261951208114624  , -0.8872069120407104 , -2.7010369300842285 ,
  -0.14230945706367493,  2.018272876739502  ,  0.6519722938537598 ,
  -0.9355422258377075 , -1.3372136354446411 , -0.9536375999450684 ,
  -0.7038232088088989 ,  2.1512014865875244 ,  1.1831238269805908 ,
  -0.47307267785072327, -2.300624370574951  , -1.9390225410461426 ,
  -1.315086841583252  , -0.9886696338653564 ,  0.9798738360404968 ,
   1.081017017364502  ,  1.0184282064437866 ,  1.236150860786438  ,
  -0.700860321521759  ,  0.5499051213264465 ,  1.0398610830307007 ,
   0.7231347560882568 , -2.759103775024414  , -1.2189456224441528 ,
   0.7222438454627991 ,  1.8139985799789429 ,  1.69119131565094   ,
   0.13087472319602966, -0.2105720043182373 ,  0.06912644952535629,
  -3.4780232906341553 ,  0.8397577404975891 ,  1.4962682723999023 ,
   0.40008196234703064, -0.1936039924621582 , -0.9500410556793213 ,
   2.0477497577667236 , -0.04077014699578285, -1.108513593673706  ,
  -0.7808402180671692 ,  0.23637039959430695,  0.6370900869369507 ,
  -0.15132881700992584, -1.9073138236999512 ,  0.16284097731113434,
  -1.3552383184432983 , -2.3552017211914062 , -0.13030114769935608,
   1.1660743951797485 ,  1.0294920206069946 ,  0.8814464807510376 ,
   0.5649874210357666 ,  0.8347834944725037 , -1.404604434967041  ,
   1.8272203207015991 ],
 [-0.22970525920391083,  0.00777450855821371, -1.361759066581726  ,
   0.06932477653026581, -0.17945003509521484, -1.279410719871521  ,
   0.09647233039140701,  0.24735327064990997, -1.7021619081497192 ,
   0.03165397047996521, -1.2886954545974731 , -1.5162696838378906 ,
  -0.0440576933324337 , -0.3395417332649231 ,  0.00587482610717416,
  -1.1090422868728638 ,  0.14608220756053925, -0.8316551446914673 ,
  -0.4359924793243408 ,  0.1630721092224121 ,  1.0203022956848145 ,
   0.9439466595649719 , -0.12664645910263062,  0.6089498400688171 ,
  -0.00227536587044597, -1.3700984716415405 , -0.32958194613456726,
  -0.3107641637325287 , -0.7955523729324341 , -0.05583211034536362,
  -0.21530777215957642,  0.8510985970497131 ,  0.03365588933229446,
  -1.0282636880874634 , -0.9273563623428345 , -1.6761478185653687 ,
  -1.3004150390625    , -1.0713093280792236 , -0.21650639176368713,
  -0.14791303873062134,  0.20843753218650818,  0.02974735014140606,
  -0.8556053042411804 , -3.4199259281158447 ,  0.09360135346651077,
  -2.8518059253692627 , -0.9506452083587646 , -0.5299490094184875 ,
  -0.02484574727714062, -0.44794848561286926,  0.05629930272698402,
   0.4133942127227783 ,  0.1692972183227539 ,  0.04328802600502968,
  -0.06312736868858337,  0.25917768478393555,  0.1335269808769226 ,
  -0.1554650515317917 ,  0.6310837864875793 ,  0.7697899341583252 ,
   0.3833054006099701 , -0.20418839156627655,  0.7123140692710876 ,
   0.14519910514354706],
 [-0.1374463587999344 , -1.0502569675445557 ,  0.44703155755996704,
  -0.4746154546737671 , -0.9131660461425781 ,  0.0115467943251133 ,
   0.4787234365940094 , -0.27410393953323364, -0.17502281069755554,
  -0.3828176259994507 ,  0.8398575186729431 , -1.046565294265747  ,
  -0.34011831879615784,  0.11663471907377243,  0.4673489034175873 ,
  -0.8689640760421753 ,  0.37565797567367554,  0.5579099059104919 ,
  -0.36420169472694397, -1.786843180656433  , -0.097081758081913  ,
  -0.19101779162883759,  0.15048827230930328, -0.07554987072944641,
  -1.702511191368103  , -0.712769091129303  ,  0.5546317100524902 ,
  -0.9333783388137817 , -0.13203363120555878,  0.45754027366638184,
   0.0579601526260376 ,  0.7327483892440796 , -0.48747915029525757,
   0.8253852725028992 , -0.3262278139591217 ,  0.4119771420955658 ,
   0.9286646842956543 , -0.26555898785591125, -1.4600924253463745 ,
  -1.0858675241470337 , -0.15876731276512146,  0.716741681098938  ,
  -0.30889856815338135,  0.5830210447311401 , -1.7857551574707031 ,
  -0.47222286462783813, -0.5625109672546387 ,  0.12756434082984924,
   0.5868356823921204 , -0.22816793620586395,  0.7058586478233337 ,
  -0.7113264203071594 ,  0.24509216845035553,  0.9756353497505188 ,
  -1.0792466402053833 , -0.11255419999361038, -0.02743737399578094,
  -0.4852738380432129 ,  0.6016018986701965 ,  0.381511926651001  ,
  -0.43084782361984253, -0.15085820853710175,  0.8317813873291016 ,
   1.0671017169952393 ],
 [ 0.06590403616428375,  0.31807127594947815,  1.3527415990829468 ,
  -0.2810937464237213 , -0.8171108961105347 , -0.23155559599399567,
  -0.7007331252098083 ,  0.19249847531318665,  0.6044274568557739 ,
  -0.23070940375328064,  0.45755600929260254,  0.9333512783050537 ,
   0.22106420993804932,  0.3949744701385498 ,  0.7259881496429443 ,
  -0.39066654443740845,  1.4732356071472168 , -0.44165778160095215,
   0.9463446140289307 ,  0.4417247772216797 ,  0.4215580224990845 ,
   0.3898477256298065 , -0.5564513802528381 , -2.7128121852874756 ,
   0.6633626222610474 , -0.05356577411293983,  0.6188668012619019 ,
  -2.2014026641845703 , -0.17056241631507874, -0.01809665188193321,
  -0.25664225220680237,  0.9804941415786743 , -0.30488428473472595,
   1.1380857229232788 , -0.9136015772819519 ,  0.7025303244590759 ,
   0.4526546597480774 , -1.1901071071624756 ,  1.0202574729919434 ,
   0.32989582419395447, -3.5238492488861084 ,  0.19385243952274323,
  -0.6550715565681458 ,  0.083615243434906  , -1.2426921129226685 ,
   0.9495977163314819 ,  0.646078884601593  , -0.6972986459732056 ,
  -1.833397626876831  ,  0.44986844062805176,  1.1772196292877197 ,
  -0.4443700611591339 ,  0.9541498422622681 ,  0.4374229609966278 ,
  -0.00107517931610346,  0.8082101941108704 ,  0.2231728881597519 ,
   0.2293136566877365 ,  1.2554264068603516 , -0.4485645890235901 ,
   1.3127403259277344 , -0.22654412686824799,  1.2409446239471436 ,
   0.5990203022956848 ],
 [-1.1651774644851685 ,  0.5317065119743347 , -0.2856401205062866 ,
  -0.06542100757360458, -0.9131505489349365 , -1.539538025856018  ,
  -0.23760120570659637,  0.1937742531299591 , -1.389234185218811  ,
   0.3066452145576477 , -1.164276361465454  ,  0.15396136045455933,
  -0.4675605893135071 ,  0.104450523853302  , -0.8581229448318481 ,
  -0.5443044304847717 , -1.3636420965194702 , -1.1016298532485962 ,
  -1.2531753778457642 , -0.2567570209503174 , -0.5220434665679932 ,
  -0.1972292810678482 ,  0.48496994376182556,  0.19465650618076324,
   0.19591067731380463, -0.17665432393550873,  0.08638177812099457,
  -0.3530275225639343 ,  0.17733512818813324, -1.910102128982544  ,
  -0.5217462778091431 , -0.37636518478393555,  0.16732612252235413,
  -1.074579119682312  , -0.05962461978197098,  0.3327905833721161 ,
   0.18274615705013275,  0.06620423495769501,  0.16304798424243927,
  -0.562370240688324  , -0.840617835521698  , -0.11326514184474945,
   0.06767988204956055,  0.5424331426620483 , -1.7534031867980957 ,
  -1.6300934553146362 , -1.4369179010391235 , -0.22856774926185608,
   0.2005275934934616 , -0.06779725104570389, -1.3017367124557495 ,
  -1.5278443098068237 ,  0.2544984221458435 , -1.3354369401931763 ,
  -0.6260526776313782 ,  0.1231968030333519 , -0.14964474737644196,
  -0.24483536183834076, -0.3825262486934662 ,  0.24829047918319702,
  -0.05905948206782341, -0.13832393288612366, -0.7566015720367432 ,
  -0.3888070285320282 ],
 [-0.8804500699043274 , -0.6085182428359985 , -0.3296077847480774 ,
   0.2684476971626282 ,  0.22456999123096466,  1.2210687398910522 ,
   1.153916597366333  ,  1.033415675163269  , -1.0110208988189697 ,
  -0.5926737189292908 ,  1.2184312343597412 ,  0.35918545722961426,
  -0.04537276551127434, -1.4631619453430176 , -0.1350727528333664 ,
  -0.518514096736908  ,  0.2475174367427826 , -0.415976881980896  ,
  -1.1357403993606567 ,  0.27503371238708496,  1.7784091234207153 ,
   2.5082008838653564 ,  0.5430903434753418 ,  1.9320229291915894 ,
   0.6188791990280151 ,  1.8360167741775513 ,  0.3574445843696594 ,
   0.6631153225898743 ,  0.795492947101593  , -0.07413213700056076,
   0.6106693148612976 ,  2.0954980850219727 ,  1.1426947116851807 ,
   1.7850725650787354 , -0.6212642192840576 , -0.00472271116450429,
   1.8790359497070312 ,  1.047519564628601  ,  1.196943759918213  ,
   1.0456228256225586 ,  1.6121255159378052 , -0.5065624713897705 ,
   2.1840262413024902 , -0.20857742428779602, -1.1492562294006348 ,
   1.8733806610107422 ,  3.1572742462158203 ,  0.9773015975952148 ,
   0.9358856678009033 , -0.12383559346199036,  0.5222017765045166 ,
   2.066371440887451  , -0.4237360954284668 ,  1.356090784072876  ,
  -0.815611720085144  ,  1.046796441078186  , -0.25007614493370056,
  -0.5389801263809204 ,  1.5218040943145752 ,  1.8806724548339844 ,
   1.2268067598342896 , -0.7187864780426025 , -0.35362210869789124,
   0.6076536774635315 ],
 [-1.6798763275146484 , -0.8577763438224792 ,  0.4306788146495819 ,
  -0.7027722001075745 , -1.336808681488037  , -0.8241672515869141 ,
  -0.5613202452659607 , -0.26909175515174866,  1.2723033428192139 ,
  -0.19228143990039825,  0.5968568325042725 , -0.6241357922554016 ,
   0.3001735806465149 , -0.2410588264465332 , -0.47911158204078674,
   0.05421930924057961,  0.5361441969871521 , -1.0094877481460571 ,
   0.05380310863256454, -0.36957573890686035, -0.8431912064552307 ,
   0.7377421259880066 ,  0.6831875443458557 ,  0.0652298778295517 ,
  -0.05994847416877747,  0.4210011959075928 ,  0.7544795870780945 ,
  -0.6899694800376892 , -0.26974302530288696,  1.1590752601623535 ,
  -0.5565304160118103 ,  0.796613335609436  ,  0.2624819576740265 ,
   0.9273512959480286 , -0.2245047390460968 , -0.22120660543441772,
   0.8437775373458862 ,  0.6703492999076843 ,  0.09739816933870316,
  -1.6224050521850586 ,  0.19764812290668488, -0.15158867835998535,
   0.5523448586463928 ,  0.14039406180381775, -0.28477829694747925,
   1.2303082942962646 ,  0.05868471413850784, -0.9715529084205627 ,
  -0.7647483944892883 , -0.16253574192523956,  0.9159882068634033 ,
   0.02932091988623142, -1.201675534248352  ,  0.14611822366714478,
  -0.7904790639877319 , -0.6673310399055481 ,  0.1956786811351776 ,
   1.0308849811553955 ,  1.3250519037246704 ,  1.385451078414917  ,
  -0.22985497117042542,  0.18111932277679443,  0.9128106832504272 ,
   0.45484426617622375],
 [-0.9133650660514832 ,  0.2141227424144745 , -0.03321012109518051,
  -0.14744873344898224,  0.47107598185539246, -0.2877010703086853 ,
   0.0947161465883255 ,  0.02540493011474609, -0.09278392791748047,
   0.06953027099370956, -1.3751633167266846 , -1.4925103187561035 ,
  -0.46449726819992065,  0.4947890639305115 , -0.05749419704079628,
   0.04525551572442055, -0.11731467396020889, -2.2756567001342773 ,
  -0.13359804451465607,  0.07551197707653046,  0.09719128906726837,
  -0.28544753789901733, -0.7434213161468506 ,  0.03326455131173134,
  -0.12266185879707336, -0.37189194560050964, -0.3883501887321472 ,
   0.06335102021694183,  0.06589766591787338, -1.2000257968902588 ,
   0.18171639740467072, -0.11116858571767807,  0.2281261533498764 ,
  -0.01445610076189041, -0.7314144372940063 , -0.6387754678726196 ,
  -0.8186773657798767 , -0.00120270240586251,  0.22300505638122559,
  -0.5163277983665466 ,  0.236435204744339  ,  0.3351822793483734 ,
  -0.7267246246337891 , -1.4227126836776733 ,  0.4289977252483368 ,
   0.6080820560455322 , -0.6991784572601318 ,  0.7893297672271729 ,
   0.19116713106632233,  0.0865834578871727 , -2.8183815479278564 ,
   0.22903603315353394, -0.713504433631897  ,  0.8208177089691162 ,
  -1.804792881011963  , -0.10070251673460007,  0.05326113477349281,
   0.06757987290620804,  0.2872439920902252 ,  0.5536052584648132 ,
  -0.00901673175394535, -0.05873475596308708, -0.693839430809021  ,
  -0.595541775226593  ],
 [ 0.27227187156677246, -0.349275141954422  ,  0.4711708128452301 ,
  -0.8712002635002136 , -1.5786757469177246 , -0.705912172794342  ,
  -0.3753756582736969 , -0.06008617207407951,  0.200114443898201  ,
  -0.1385185718536377 , -1.330466389656067  ,  0.5587145090103149 ,
   0.23476049304008484, -0.2985711693763733 ,  0.13865061104297638,
  -1.9058910608291626 , -1.7342973947525024 ,  0.480940580368042  ,
  -0.0419270247220993 , -0.10895691066980362, -0.36452725529670715,
   0.30502739548683167, -0.03708037361502647, -0.28833961486816406,
  -0.8931261301040649 , -2.690805196762085  ,  0.249113067984581  ,
  -1.0167057514190674 , -0.40335938334465027,  0.08899054676294327,
   0.34875017404556274, -0.1884021908044815 , -0.36498555541038513,
  -1.6585478782653809 ,  0.2020600140094757 , -0.2835676372051239 ,
  -0.7788651585578918 , -0.5473824739456177 , -0.09028106182813644,
  -0.1561814248561859 , -0.27695325016975403, -0.8263530731201172 ,
  -1.43306565284729   , -0.16663439571857452, -0.03139260038733482,
  -4.588142395019531  , -0.6793733835220337 , -0.12730033695697784,
  -0.6173642873764038 ,  0.5432772040367126 ,  0.3298673629760742 ,
  -0.28674983978271484, -0.27970167994499207, -0.26361098885536194,
   0.5305961966514587 ,  0.40314939618110657,  0.20376747846603394,
  -0.01316946279257536, -0.33150514960289   , -0.47736772894859314,
   0.10576741397380829, -0.1371443271636963 ,  0.398485392332077  ,
  -0.295071542263031  ],
 [-0.574619710445404  ,  0.43981125950813293,  0.22688710689544678,
  -0.6888463497161865 ,  0.230211079120636  , -0.2478359490633011 ,
   0.45670244097709656, -1.002332091331482  , -0.36975425481796265,
   0.31341758370399475,  0.4414235055446625 ,  1.0194182395935059 ,
   0.5884844064712524 , -2.0983986854553223 ,  0.1853180080652237 ,
  -0.6270303130149841 , -1.6374682188034058 ,  0.7468578815460205 ,
  -0.7447275519371033 ,  0.5734758973121643 , -0.4016662538051605 ,
  -0.7491853833198547 , -0.07309015840291977,  0.25185900926589966,
   0.5317636132240295 , -1.369639277458191  , -0.07300619035959244,
  -0.6796523928642273 ,  0.2060980498790741 , -0.7082204222679138 ,
   0.3701988458633423 , -1.4562228918075562 , -0.04838020727038383,
   0.3047419786453247 , -0.32080626487731934, -0.6215173602104187 ,
  -1.3291162252426147 , -1.1818081140518188 , -1.1041080951690674 ,
  -0.23813390731811523, -0.10827428102493286,  0.7043755054473877 ,
  -1.5445103645324707 , -0.33282917737960815,  0.5277264714241028 ,
  -0.3810177743434906 ,  0.0337911918759346 , -2.5869288444519043 ,
  -1.168902039527893  ,  0.5306441187858582 , -0.44965749979019165,
   0.09907061606645584,  0.06348764151334763,  0.8338762521743774 ,
   0.8055458664894104 ,  0.46797671914100647, -0.24025283753871918,
  -0.23841062188148499,  0.6276758909225464 ,  0.03527282550930977,
  -0.5977437496185303 ,  0.17353111505508423, -0.8096578121185303 ,
   0.3682296574115753 ],
 [ 0.2884446084499359 , -0.4146425426006317 , -1.0115114450454712 ,
   0.20554576814174652, -1.676559329032898  , -0.9965671300888062 ,
  -0.1824524998664856 , -0.6265859603881836 , -0.6855442523956299 ,
  -0.40402519702911377, -1.8830342292785645 ,  1.9304752349853516 ,
   0.03927493840456009,  0.8512956500053406 , -0.7293877601623535 ,
  -0.826021134853363  ,  1.0082215070724487 , -0.11005448549985886,
   0.2092026323080063 ,  0.3368980288505554 , -0.4089772701263428 ,
   0.01401029340922832,  0.18735121190547943,  0.9597995281219482 ,
   0.41522398591041565, -0.22298486530780792,  0.7989306449890137 ,
   0.5598981976509094 ,  1.0283689498901367 , -1.0171047449111938 ,
  -0.25374332070350647,  0.6032282114028931 , -0.00339698418974876,
  -0.8030584454536438 ,  0.20550784468650818, -2.1656103134155273 ,
   0.3040369153022766 , -0.06784281879663467, -0.44824108481407166,
   0.19840861856937408,  2.0395877361297607 , -0.2828035056591034 ,
  -0.29832005500793457, -0.4812726676464081 ,  1.1397813558578491 ,
   0.00879210140556097, -1.4498287439346313 , -0.46189823746681213,
  -1.2528648376464844 ,  0.8240859508514404 , -1.4309964179992676 ,
  -0.9834285974502563 ,  0.4831302762031555 ,  0.9385177493095398 ,
   0.21896961331367493, -1.8132919073104858 ,  0.6534257531166077 ,
   0.6029887199401855 , -0.11749474704265594,  0.27182963490486145,
  -0.7850283980369568 , -0.16621947288513184,  0.7684609889984131 ,
   0.6253204941749573 ],
 [ 0.07481210678815842,  0.11296344548463821, -0.2886592149734497 ,
  -0.1373058557510376 , -0.769722580909729  , -0.3955657184123993 ,
   0.03825336694717407, -0.08397071808576584, -0.4767100512981415 ,
  -0.3716370463371277 , -0.4547822177410126 ,  0.06162210181355476,
  -0.506881833076477  , -0.04260816425085068, -0.12009954452514648,
   0.07469140738248825, -1.8410214185714722 ,  0.0361168384552002 ,
  -0.0138836307451129 ,  0.10803057998418808, -0.3972876965999603 ,
  -0.04137797281146049, -0.0202505923807621 ,  0.10113917291164398,
  -0.0027011742349714 , -0.10282992571592331, -0.06974855810403824,
  -0.34448906779289246, -0.02138466946780682,  0.10336872190237045,
   0.26439324021339417, -0.8224424719810486 , -0.12471701204776764,
  -0.03175054490566254, -0.3620915710926056 ,  0.2927550673484802 ,
  -0.6108716726303101 , -0.5329837799072266 , -0.41058945655822754,
  -0.02277711592614651, -0.08694443106651306, -0.33142536878585815,
  -0.26937055587768555,  0.43710434436798096,  0.0655359998345375 ,
   0.27659872174263   , -0.5823155045509338 , -0.6624007225036621 ,
  -0.8094407916069031 ,  0.282617449760437  ,  0.1805545836687088 ,
   0.18633291125297546, -0.3418143093585968 , -0.07246015220880508,
  -1.8071529865264893 , -0.06482827663421631, -0.20804689824581146,
   0.00887030642479658, -0.46607157588005066, -0.06966929882764816,
  -0.11684221029281616, -0.06939296424388885, -0.11887741833925247,
  -0.08559315651655197],
 [ 0.2945363521575928 , -0.01849471405148506, -0.5009603500366211 ,
  -0.21713696420192719,  0.5508089661598206 , -1.12962007522583   ,
  -0.07329413294792175,  0.01362968981266022,  0.19854487478733063,
  -0.17671771347522736, -0.2519873380661011 , -0.07321380823850632,
  -0.4442918002605438 , -0.05221334099769592, -0.0836271271109581 ,
  -0.27695760130882263, -1.3996167182922363 , -0.02659460343420506,
   0.19466367363929749,  0.13186824321746826, -0.6893619894981384 ,
   0.02290505357086658, -0.0817505344748497 , -0.08272853493690491,
  -0.03428320959210396, -0.38351887464523315,  0.10984783619642258,
  -0.7918784618377686 , -0.34466177225112915,  0.21839502453804016,
   0.09184327721595764, -0.8407541513442993 , -0.08479458838701248,
   0.0250919982790947 , -0.47713151574134827, -0.16698989272117615,
  -0.40591973066329956, -0.07166997343301773, -0.11373934894800186,
  -0.04576171189546585,  0.01943913847208023,  0.0042994967661798 ,
  -0.41100630164146423, -0.27943965792655945, -0.7949754595756531 ,
  -1.2735236883163452 , -0.3535146713256836 , -0.939099133014679  ,
  -0.04278269782662392, -0.00227354629896581,  0.2531548738479614 ,
  -0.1705794334411621 ,  0.5627714991569519 ,  0.37119781970977783,
  -0.04988334700465202, -0.18833468854427338,  0.07032100111246109,
   0.41974490880966187, -0.8778784275054932 ,  0.11735154688358307,
   0.4189699590206146 , -0.2630229592323303 ,  0.21808907389640808,
  -0.47149476408958435],
 [-0.6876051425933838 , -0.19715838134288788,  0.43348395824432373,
  -0.08765602856874466, -0.2575659155845642 , -0.6852301955223083 ,
  -0.6457667350769043 , -0.453346312046051  , -0.23958013951778412,
   0.43991777300834656, -0.5993256568908691 , -0.5854195952415466 ,
  -0.13279081881046295, -0.3557640314102173 , -1.1575225591659546 ,
  -0.1144617348909378 , -1.058088779449463  ,  0.07474309206008911,
  -0.07314548641443253,  0.12976312637329102, -0.19295161962509155,
   0.04465917870402336, -0.07955644279718399, -0.6813124418258667 ,
   0.09574498236179352,  0.2439802885055542 ,  0.22248134016990662,
  -1.297053337097168  , -0.12665338814258575, -2.319009304046631  ,
   0.00999148655682802,  0.3994295001029968 , -0.13985343277454376,
   0.12273097038269043, -0.5418914556503296 , -0.00328605063259602,
   0.19672434031963348, -0.49209362268447876,  0.0081415930762887 ,
   0.3169947862625122 , -5.566184997558594  , -0.3362395167350769 ,
  -0.5016026496887207 , -0.01480313763022423,  0.04669086262583733,
  -1.9036743640899658 , -0.2596418857574463 , -2.152883768081665  ,
  -0.9237980246543884 , -0.07525290548801422,  0.1340072900056839 ,
  -0.2879497706890106 , -0.6382197141647339 , -0.6216807961463928 ,
  -0.49932485818862915, -0.07813295722007751, -0.1548858880996704 ,
  -0.1959279179573059 , -0.22608742117881775, -0.9951444268226624 ,
  -0.2682267129421234 , -0.10009711235761642,  0.07400470227003098,
  -1.122296690940857  ],
 [-0.16357684135437012, -0.14476926624774933,  0.4294137954711914 ,
   0.2924242615699768 ,  0.04972381889820099,  0.23334956169128418,
  -0.9578760862350464 ,  0.22768421471118927, -0.41886603832244873,
  -0.09686723351478577, -0.3590118885040283 ,  0.6506350636482239 ,
   0.2718261778354645 ,  0.29555743932724   , -0.7744850516319275 ,
  -0.14732004702091217,  0.36204424500465393,  0.25303542613983154,
  -0.44506964087486267,  0.00876549445092678,  0.3648880422115326 ,
  -0.8967251777648926 ,  0.00721830874681473, -0.2493675798177719 ,
  -0.131048783659935  , -0.84369295835495   ,  0.20756104588508606,
  -0.676923930644989  ,  0.34057265520095825, -0.34671124815940857,
   0.06761226803064346,  0.3949745297431946 , -0.13992270827293396,
   0.1921108216047287 , -1.6677708625793457 ,  0.16608548164367676,
   0.4573971927165985 ,  0.03025360219180584, -0.6824972629547119 ,
  -0.1025337278842926 ,  0.18454492092132568,  0.20617137849330902,
  -0.27462488412857056,  0.36502447724342346,  0.05756978318095207,
  -0.8257691264152527 , -0.9209529161453247 , -0.12898856401443481,
  -0.2208489328622818 ,  0.29234227538108826,  0.18072114884853363,
   0.2931611239910126 ,  0.5513507127761841 , -0.24235469102859497,
  -0.8867356181144714 , -0.15408053994178772,  0.21107828617095947,
   1.0186375379562378 , -0.5860660076141357 ,  0.17762206494808197,
   0.3090798556804657 , -0.15875190496444702, -1.6649152040481567 ,
   0.7547711730003357 ],
 [-0.7979413270950317 ,  0.02388743311166763, -0.6418938040733337 ,
  -0.08070017397403717,  0.14195793867111206, -0.6635485887527466 ,
  -1.323870062828064  , -0.01196788717061281, -0.718291163444519  ,
   0.23751558363437653, -0.941840410232544  , -0.6021055579185486 ,
  -0.02827080525457859, -0.2454046756029129 , -0.12771055102348328,
  -0.0092475526034832 , -0.7424478530883789 , -0.02676431462168694,
   0.46209919452667236, -0.897666871547699  , -0.03403089568018913,
  -0.7181137800216675 , -0.0221557579934597 , -0.01605883426964283,
  -0.3530033230781555 ,  0.1863214671611786 , -0.2297399640083313 ,
   0.16315750777721405, -0.03414236754179001,  0.04643399640917778,
  -0.05599908158183098, -0.25237610936164856,  0.27228254079818726,
  -0.4981299936771393 , -0.8978662490844727 ,  0.02314342744648457,
   0.32248353958129883, -0.04669718816876411, -0.00875972304493189,
  -0.48142966628074646,  0.2596779763698578 , -0.9273858666419983 ,
   0.2973323464393616 , -0.8812515139579773 , -0.04128909111022949,
  -0.04726234823465347, -0.1231950893998146 ,  0.22810940444469452,
  -0.1346561163663864 ,  0.01517161540687084,  0.1989113837480545 ,
  -0.13768401741981506, -0.8865420818328857 , -1.0367687940597534 ,
  -0.44396471977233887, -1.8783570528030396 , -0.15630196034908295,
  -0.4957202970981598 , -0.9974915981292725 , -0.47259122133255005,
  -0.6109679341316223 ,  0.03150150179862976, -0.24025416374206543,
  -0.9090259075164795 ],
 [ 0.1840725690126419 ,  0.03851484507322311, -0.9070482850074768 ,
   0.08496294170618057,  0.19022785127162933, -0.336805135011673  ,
  -0.7039923667907715 , -0.07097002118825912, -0.2548258900642395 ,
   0.073000967502594  , -0.44021689891815186, -0.6034064888954163 ,
  -0.15112534165382385, -0.1427776962518692 ,  0.01263049803674221,
  -0.1793416142463684 , -1.0592783689498901 ,  0.23571451008319855,
  -0.32995522022247314, -1.0077688694000244 , -0.352986216545105  ,
  -0.8228132724761963 , -0.11179029941558838, -0.36342379450798035,
  -0.23861050605773926, -0.14309173822402954, -0.00696795480325818,
  -0.4050728976726532 , -0.16391298174858093, -1.1640905141830444 ,
  -0.02107181586325169, -2.2154414653778076 ,  0.10883738845586777,
  -0.5171213150024414 , -0.1496340036392212 , -0.22424814105033875,
  -0.6576814651489258 , -0.20511190593242645,  0.16473254561424255,
  -0.31565338373184204, -0.3012031316757202 , -0.7220527529716492 ,
  -0.13294145464897156, -0.24544599652290344, -0.04250654578208923,
  -1.8880882263183594 , -0.9741575121879578 , -0.4159708321094513 ,
  -0.5165297985076904 , -0.17843841016292572, -0.6032496094703674 ,
   0.02970699593424797, -0.1241869255900383 , -0.38499829173088074,
  -0.34367385506629944, -0.11129816621541977,  0.01651753298938274,
  -0.07199954241514206, -0.9627442359924316 , -0.4585331976413727 ,
  -0.5601890683174133 , -0.01215842273086309, -1.4081840515136719 ,
  -0.6376153230667114 ],
 [-0.4788505733013153 , -0.17803871631622314, -0.915669322013855  ,
   0.02247819863259792, -0.23805831372737885, -0.7294148206710815 ,
  -0.4160955548286438 , -0.19736897945404053, -2.047445297241211  ,
   0.07006583362817764, -2.187096118927002  , -1.929702639579773  ,
  -0.03167600929737091, -0.19973547756671906, -0.3629071116447449 ,
  -0.05477627739310265, -0.549368143081665  , -0.14268602430820465,
  -0.7224855422973633 , -0.2526783347129822 , -0.02049535699188709,
  -0.6135528683662415 , -0.2425483763217926 , -1.3214235305786133 ,
  -0.3773689568042755 , -0.17968563735485077, -0.5591030120849609 ,
  -0.2930220365524292 , -0.1910775899887085 , -0.7843955755233765 ,
  -0.18786217272281647, -1.722779631614685  ,  0.263683944940567  ,
  -0.1171097382903099 , -1.112921118736267  , -0.07167861610651016,
  -0.29185304045677185, -0.44621968269348145, -0.10341231524944305,
  -0.1935218870639801 , -0.4790685772895813 , -0.6156295537948608 ,
  -0.6458418965339661 , -0.2234751582145691 , -0.5773841738700867 ,
   0.1848423182964325 , -2.436290979385376  , -0.33253058791160583,
   0.02677875943481922, -0.17649298906326294, -0.7009435892105103 ,
  -0.5647470355033875 ,  0.05730139091610909, -2.3048198223114014 ,
  -0.8303564190864563 , -0.7498956322669983 , -0.13592283427715302,
  -0.388041228055954  , -1.244563341140747  , -0.40155234932899475,
  -0.29832106828689575,  0.06043578684329987, -0.7931737303733826 ,
  -0.00472111161798239]])
    b4 = jnp.array([-0.9106773138046265 ,  1.6127196550369263 , -0.6155756711959839 ,
  0.01898445188999176, -0.29660412669181824, -0.15508043766021729,
  0.8234370946884155 ,  1.0927132368087769 , -0.29009729623794556,
  2.0481486320495605 , -0.44662684202194214, -1.1220306158065796 ,
 -0.37712332606315613,  1.4313642978668213 ,  0.5285846590995789 ,
  0.0961809903383255 , -0.7760072946548462 , -0.07194223999977112,
 -1.6298614740371704 ,  0.8132202625274658 ,  0.4581066966056824 ,
 -1.2626465559005737 ,  1.8627092838287354 , -3.0946691036224365 ,
  1.3430242538452148 , -0.46277308464050293, -1.2491178512573242 ,
 -0.14385277032852173, -0.759763777256012  , -0.5380963087081909 ,
 -0.07595174014568329, -1.0042030811309814 ,  1.0317178964614868 ,
 -1.4110662937164307 , -1.0899525880813599 ,  0.3035833239555359 ,
 -0.12545551359653473, -0.15282711386680603,  0.6086091995239258 ,
  1.106329321861267  , -3.3236796855926514 ,  0.1393347978591919 ,
 -0.45147910714149475,  0.5140073299407959 ,  0.8433811068534851 ,
 -0.9881926774978638 ,  0.16564379632472992, -0.01002198923379183,
 -0.5679389834403992 , -0.05405009165406227, -1.9790408611297607 ,
  0.02413010038435459, -0.96571946144104   , -2.5824592113494873 ,
  0.6853266954421997 ,  0.5964006185531616 ,  1.5006600618362427 ,
 -1.1096969842910767 , -0.894329309463501  ,  0.3491307497024536 ,
 -0.7123404741287231 ,  2.129709482192993  , -1.1268391609191895 ,
 -0.29607900977134705])
    W5 = jnp.array([[ 0.20046959817409515, -0.5559655427932739 , -0.15074577927589417,
  -1.2334617376327515 , -0.6892552971839905 , -0.8234667181968689 ,
  -0.24874632060527802, -0.6101937294006348 ,  1.2434906959533691 ,
  -0.16273243725299835,  0.9204427003860474 , -0.19141651690006256,
   0.3771098554134369 ,  0.884403645992279  , -0.44348976016044617,
  -0.09314163774251938,  0.05072909593582153, -0.6302798986434937 ,
  -0.37964579463005066, -0.2762536108493805 , -0.3044714629650116 ,
  -0.0700775757431984 ,  0.05255711078643799, -0.16360759735107422,
   0.10026802867650986,  0.28589269518852234, -0.19030846655368805,
  -1.2840968370437622 , -0.9422627091407776 ,  0.29775822162628174,
  -0.3840632736682892 ,  0.3258858323097229 ],
 [-0.09392613917589188, -0.38866183161735535, -0.00046233314787969,
  -0.3000192940235138 ,  0.15369591116905212, -0.06055270135402679,
  -0.47366034984588623, -0.05611079931259155, -0.6265514492988586 ,
   0.04326780140399933, -0.5139585137367249 , -0.2428572177886963 ,
  -0.8669282793998718 ,  0.20787127315998077, -1.7110984325408936 ,
  -0.08681157976388931,  0.2394293248653412 , -0.5316738486289978 ,
  -0.5138255953788757 , -1.0788925886154175 , -0.10660979896783829,
   0.07734709233045578, -0.05692062899470329, -0.01632468402385712,
   0.01847116090357304, -0.22567814588546753, -0.07128335535526276,
  -0.7658576369285583 , -0.3921559751033783 , -0.06527315825223923,
  -0.00828575156629086, -0.06187749654054642],
 [ 0.03544926270842552,  0.06350734829902649, -0.04819293692708015,
   0.9629352688789368 , -0.07593779265880585, -0.1610998660326004 ,
   0.5188602209091187 , -0.02253288961946964, -0.07511979341506958,
  -0.6262351274490356 ,  0.4548761248588562 , -0.44732964038848877,
   0.45295652747154236, -0.30698853731155396, -0.6216142177581787 ,
  -0.25834211707115173, -0.0538443922996521 ,  0.6679975390434265 ,
  -0.6304723024368286 , -3.283973217010498  ,  0.7418901324272156 ,
  -0.729642391204834  ,  0.41833776235580444, -0.2333035171031952 ,
  -1.1194545030593872 , -0.521933913230896  ,  0.7759583592414856 ,
  -0.7564834952354431 ,  0.6180201172828674 , -0.2716722786426544 ,
  -0.10822168737649918,  0.45236602425575256],
 [-0.09776599705219269,  0.788371205329895  , -0.6609992980957031 ,
  -0.5541001558303833 ,  2.2058045864105225 ,  0.725226104259491  ,
   0.3572574853897095 ,  0.43997490406036377, -0.9437262415885925 ,
  -0.6077849864959717 ,  0.5915807485580444 , -0.24086213111877441,
   0.12697994709014893,  0.7452443838119507 , -0.2639789879322052 ,
  -0.00393014587461948, -0.5867523550987244 ,  1.2734953165054321 ,
   0.7258468866348267 , -0.5485793352127075 ,  1.461195945739746  ,
  -0.10864932090044022,  0.12894099950790405,  0.21658319234848022,
   0.3897295594215393 , -0.6164771318435669 ,  0.28781530261039734,
  -0.3271174728870392 ,  0.4630594849586487 , -0.4147437810897827 ,
  -0.2851305603981018 ,  0.26117151975631714],
 [-0.03247597813606262, -0.14888109266757965, -0.03395549952983856,
  -0.3646008372306824 , -1.1276313066482544 ,  0.04110309109091759,
   0.06759337335824966, -0.03006563894450665, -1.0669703483581543 ,
  -0.29778867959976196,  0.10853814333677292,  0.2514873743057251 ,
  -0.7025294899940491 , -0.455258846282959  ,  0.30483266711235046,
   0.04162021353840828,  0.00100307574030012, -1.4999407529830933 ,
  -0.12361524999141693,  0.2097819447517395 , -0.484003484249115  ,
   0.07893712818622589,  0.06584389507770538,  0.01060966867953539,
   0.06329376250505447,  0.05174743756651878,  0.01744595728814602,
   0.01417917013168335, -0.02416283823549747, -0.41029253602027893,
   0.0283595435321331 , -0.15492238104343414],
 [ 0.4009258449077606 ,  0.01714813522994518,  0.13727667927742004,
   0.11648483574390411, -2.1804511547088623 , -0.326570987701416  ,
   0.08352168649435043, -0.02673732303082943,  0.566288411617279  ,
   0.08991185575723648, -0.7668139934539795 , -0.5016991496086121 ,
  -0.04055279865860939, -0.697037935256958  , -0.06651657074689865,
  -0.07521027326583862, -0.0909108966588974 , -1.8486754894256592 ,
   0.22400511801242828,  0.07592208683490753, -0.4930870831012726 ,
  -0.2779272198677063 ,  0.0949917733669281 ,  0.04739363491535187,
  -0.15759965777397156,  0.17526954412460327,  0.571475625038147  ,
  -0.839970588684082  ,  0.28170090913772583, -0.5183061957359314 ,
  -0.456091046333313  ,  0.38396841287612915],
 [ 0.6549997925758362 , -1.3877524137496948 ,  0.2895931899547577 ,
   1.152284860610962  , -1.1220276355743408 , -0.6077773571014404 ,
  -0.4063355326652527 , -0.01108399964869022, -0.5042307376861572 ,
  -0.4125840365886688 ,  1.472597599029541  , -0.353889137506485  ,
  -0.7727878093719482 ,  0.6803873777389526 , -1.431864857673645  ,
  -0.4456925094127655 , -0.654967725276947  ,  0.08114487677812576,
   0.8815653324127197 , -0.44179588556289673, -0.00075483485125005,
   0.21350927650928497,  0.46673235297203064, -0.4897863566875458 ,
   0.33859941363334656, -0.3601807951927185 ,  0.8789299726486206 ,
  -1.1471439599990845 , -1.7520431280136108 ,  0.13496863842010498,
  -0.9052870273590088 ,  0.5556365251541138 ],
 [-0.01043848041445017, -0.40505078434944153,  0.2321358323097229 ,
  -0.02895063348114491,  0.5790618658065796 , -0.18850146234035492,
   0.02144348248839378, -0.03475801274180412, -0.2656402289867401 ,
  -0.00293501722626388, -0.14575858414173126, -0.409694105386734  ,
  -0.444900244474411  ,  0.1066039502620697 ,  0.4078691303730011 ,
   0.09128643572330475,  0.1386869102716446 ,  0.6527472138404846 ,
   0.03937710076570511,  0.25153833627700806, -0.8523509502410889 ,
  -0.02882304415106773, -0.00977815315127373,  0.01308451686054468,
  -0.06887675821781158, -0.16983316838741302,  0.08361677825450897,
   0.01472144015133381,  0.10125420242547989, -0.14273415505886078,
   0.0444856770336628 , -0.539620578289032  ],
 [-1.4259856939315796 ,  0.48607972264289856, -0.23115596175193787,
  -1.994199275970459  , -0.04818495735526085, -0.38149064779281616,
  -0.25427156686782837, -0.29661211371421814, -0.31838440895080566,
  -1.7418453693389893 , -0.9759052395820618 ,  0.5106120109558105 ,
   0.3242446780204773 , -2.1503872871398926 ,  0.35279858112335205,
   0.16102713346481323, -0.12521515786647797, -0.12315789610147476,
   0.5950755476951599 , -0.15143533051013947,  1.4243894815444946 ,
  -0.6148695945739746 , -0.13557516038417816, -0.7957207560539246 ,
  -0.1186574175953865 , -0.42112231254577637, -0.3530092239379883 ,
  -0.26311784982681274,  0.1677844524383545 , -0.33167386054992676,
  -0.8722038269042969 , -0.10729224234819412],
 [ 0.01320191007107496, -0.10864150524139404,  0.5971881747245789 ,
  -0.38781148195266724, -0.08854150772094727, -0.00070706265978515,
  -0.2748717963695526 , -0.08115207403898239, -0.551705002784729  ,
   0.12367032468318939, -0.00581658771261573, -0.28486841917037964,
  -0.44150346517562866, -0.04581663385033607, -0.808885395526886  ,
  -0.11098765581846237,  0.13550111651420593, -0.1565680354833603 ,
  -0.36058279871940613, -1.5548653602600098 , -0.3956238925457001 ,
  -0.11222225427627563, -0.00386225315742195, -0.02717898227274418,
  -0.06120825931429863, -0.1154198944568634 , -0.06535539031028748,
  -0.5404983758926392 , -0.7888001203536987 , -0.24840454757213593,
  -0.05523603409528732, -0.04614052176475525],
 [ 0.8765506148338318 ,  0.7455887198448181 , -0.14803281426429749,
   0.598709225654602  ,  0.35707205533981323,  0.11692530661821365,
   0.06861208379268646, -0.04859493300318718,  0.2209918349981308 ,
  -0.45965516567230225, -0.564565122127533  , -0.46829137206077576,
  -0.273707777261734  , -0.42545682191848755, -0.05843584984540939,
   0.30859875679016113, -0.3020908534526825 , -0.5397383570671082 ,
  -0.08167453110218048,  0.1257522851228714 ,  0.7773203253746033 ,
  -0.480743408203125  , -0.17976738512516022,  0.06395946443080902,
  -0.6337763667106628 ,  0.01323290541768074, -0.03651908412575722,
  -1.2475944757461548 , -0.1529625803232193 ,  0.8547404408454895 ,
  -1.3158056735992432 , -0.8617404699325562 ],
 [-0.0800311416387558 ,  0.4078909754753113 , -0.35982784628868103,
  -1.866824984550476  , -0.00964805483818054, -0.4771380126476288 ,
  -0.7725232243537903 , -0.2973023056983948 , -0.7129204273223877 ,
  -1.0646486282348633 , -0.25324636697769165,  0.24542643129825592,
  -0.23541513085365295,  0.43225985765457153, -0.28911539912223816,
  -0.2781219482421875 , -0.35041311383247375,  0.922328531742096  ,
   0.5337018370628357 , -0.912247359752655  ,  0.41459137201309204,
   0.1692907214164734 , -0.7095032930374146 , -0.14577975869178772,
  -0.4178418219089508 ,  0.01158302649855614, -1.317686676979065  ,
   0.1276032030582428 , -1.0267473459243774 , -1.0596004724502563 ,
  -0.19353115558624268, -0.6120201945304871 ],
 [-0.8262324333190918 ,  0.9769671559333801 , -0.6861276626586914 ,
  -0.23527956008911133, -0.22230838239192963,  0.37974026799201965,
  -1.1215506792068481 , -0.642916202545166  ,  1.0241659879684448 ,
  -0.38903117179870605,  0.9655621647834778 , -0.13222149014472961,
   0.8114858865737915 , -0.7882376313209534 ,  0.7703128457069397 ,
   0.06648576259613037, -0.5898183584213257 ,  1.594014286994934  ,
   0.4439723789691925 , -1.208164095878601  ,  0.4031517803668976 ,
   0.2890685200691223 ,  0.22374814748764038, -0.03766504302620888,
  -0.11360812187194824,  0.1342490166425705 ,  1.471872091293335  ,
  -0.07750695943832397,  1.6401708126068115 ,  0.5999491810798645 ,
  -0.8371458649635315 , -0.9496400952339172 ],
 [-0.5053472518920898 , -0.5051543712615967 ,  0.09122586995363235,
  -0.22073082625865936,  1.1417253017425537 , -0.42629581689834595,
  -0.2832197844982147 , -0.25461432337760925, -1.302446722984314  ,
   0.21228592097759247,  0.00972471572458744, -0.08829616010189056,
  -0.6403002142906189 , -0.1466071903705597 , -1.6457756757736206 ,
  -0.23470351099967957,  0.40697208046913147,  0.7362321019172668 ,
   0.24275431036949158, -1.8063279390335083 , -0.43329188227653503,
  -0.08269422501325607, -0.07622086256742477, -0.15080320835113525,
  -0.2728547155857086 ,  0.24180848896503448, -0.923198938369751  ,
  -0.8879539966583252 , -2.0423622131347656 , -0.23090611398220062,
   0.05250503122806549,  0.0708429291844368 ],
 [-0.1289282888174057 , -1.2898545265197754 , -0.3737117648124695 ,
  -0.967029869556427  ,  0.8208404183387756 , -0.04050486162304878,
  -0.301703542470932  ,  0.00811895355582237, -0.37551558017730713,
  -0.5514845252037048 ,  0.17029258608818054,  0.05182870104908943,
  -0.9358402490615845 , -0.28774493932724   , -0.5496819615364075 ,
  -0.09954886883497238, -0.08185885101556778,  0.6383948922157288 ,
  -0.762401819229126  , -0.5024209022521973 , -0.2441687434911728 ,
  -0.25042349100112915,  0.17509713768959045, -0.01541383285075426,
   0.033014927059412  ,  0.18438372015953064,  0.03544830903410912,
  -0.49639955163002014, -0.6950154900550842 ,  0.5202431678771973 ,
   0.1955416202545166 , -0.2573167085647583 ],
 [ 0.11143740266561508,  1.173412561416626  , -0.58480304479599   ,
  -0.6493647694587708 ,  0.9077866673469543 ,  0.6619641184806824 ,
   0.0316249318420887 ,  0.17290069162845612, -0.6815834045410156 ,
  -0.9508306980133057 ,  0.4350537955760956 , -0.04976208508014679,
   0.29794439673423767,  0.6717850565910339 ,  0.8962287306785583 ,
  -0.0143881207332015 ,  0.32532817125320435,  1.0322556495666504 ,
  -0.5059600472450256 , -0.7922123670578003 ,  1.1427570581436157 ,
  -0.4630158245563507 ,  0.01030753180384636, -0.09684097021818161,
  -0.00541925569996238, -0.25305020809173584, -0.2887778580188751 ,
   2.032238721847534  , -0.3035386800765991 ,  0.80469810962677   ,
   0.6196776032447815 ,  0.1684054732322693 ],
 [-0.10177496075630188,  0.27711227536201477, -0.26995885372161865,
  -1.0184189081192017 ,  0.8381818532943726 ,  0.5070790648460388 ,
  -0.12864501774311066, -0.2054569125175476 , -0.7543574571609497 ,
   0.21733473241329193, -0.01471450459212065, -0.08012232184410095,
   0.6742057800292969 , -1.6115282773971558 ,  0.0903102457523346 ,
   0.07559390366077423,  0.6466225981712341 , -0.8470792174339294 ,
   0.997401237487793  , -1.2228724956512451 ,  0.6709240078926086 ,
   0.31383341550827026, -0.03241676092147827, -0.37256062030792236,
  -0.5125702619552612 , -0.8939291834831238 , -0.6639739871025085 ,
   1.1112394332885742 , -2.2818169593811035 , -0.32215672731399536,
   0.3072085976600647 , -0.24903206527233124],
 [ 0.14420315623283386,  0.20188981294631958, -0.20104242861270905,
   0.22009055316448212,  0.5269618630409241 ,  0.6709763407707214 ,
   0.06422566622495651, -0.03073239140212536,  0.28709182143211365,
  -0.3585350513458252 , -0.21526242792606354, -0.2961896061897278 ,
   0.3152983486652374 , -0.00206128833815455, -0.04853704199194908,
   0.04895276203751564,  0.24598747491836548,  0.34898433089256287,
   0.23049461841583252, -0.7922744154930115 ,  0.6934671401977539 ,
  -0.09097721427679062, -0.2224930375814438 , -0.06994691491127014,
  -0.04179845377802849, -0.23469239473342896, -1.0809751749038696 ,
  -1.0931179523468018 , -0.03053915686905384,  0.07319612801074982,
  -0.4179219901561737 ,  0.3191700577735901 ],
 [-0.3258821666240692 ,  0.7069851160049438 ,  0.3138415515422821 ,
   0.12895357608795166, -1.5394660234451294 , -0.237147256731987  ,
  -0.47051694989204407, -0.17728936672210693,  0.2433769851922989 ,
   0.06114354357123375, -0.3521171808242798 , -0.7096513509750366 ,
  -0.16910266876220703,  0.8366323709487915 , -0.5420476794242859 ,
   0.17798689007759094, -0.42192351818084717, -1.944927453994751  ,
  -0.48477262258529663,  0.29814612865448   , -1.5313303470611572 ,
  -0.21012240648269653, -0.1301736831665039 ,  0.08763370662927628,
  -0.19089151918888092,  0.0273345373570919 , -0.17501096427440643,
   0.4688996374607086 ,  0.557472825050354  ,  0.7227827906608582 ,
   0.24185530841350555, -1.5992192029953003 ],
 [-0.18870572745800018,  0.8089825510978699 , -0.0840243548154831 ,
   0.1357477307319641 ,  0.04836580529808998, -0.7472438216209412 ,
  -0.04869384691119194, -0.00670557282865047, -1.0628527402877808 ,
  -0.2937591075897217 , -0.10032052546739578, -0.11782515794038773,
  -0.02305101230740547, -0.8260045051574707 , -0.13697223365306854,
  -0.24422842264175415,  0.38301554322242737, -0.8601045608520508 ,
   0.5213916897773743 , -0.30558645725250244, -0.05001846328377724,
  -0.13048794865608215,  0.11435592919588089,  0.03962540999054909,
  -0.4520765244960785 , -0.09052547812461853, -0.8845906853675842 ,
  -0.5716918706893921 ,  0.46258819103240967, -0.4829012453556061 ,
   0.09507440775632858, -1.1226783990859985 ],
 [ 0.32019615173339844,  0.4552972614765167 ,  0.4112759828567505 ,
  -0.36155790090560913,  0.10695846378803253,  0.482308566570282  ,
  -0.13712432980537415,  0.13316696882247925, -0.7050766348838806 ,
   0.4318651854991913 , -1.2054939270019531 , -0.16599616408348083,
  -0.5797334909439087 , -0.355366975069046  ,  0.30112510919570923,
  -0.5471007823944092 , -0.2043299823999405 ,  0.8932890295982361 ,
  -2.2524008750915527 ,  0.13532955944538116, -0.11445662379264832,
  -0.29361456632614136, -0.5915810465812683 ,  0.00180983322206885,
  -0.2132670134305954 ,  0.04548566043376923,  0.47679728269577026,
   0.4436240494251251 , -1.130226969718933  , -0.4404429495334625 ,
  -0.33846086263656616, -0.7604354619979858 ],
 [-1.9508421421051025 , -0.9147107601165771 ,  0.0078587681055069 ,
  -1.0350942611694336 , -0.18337112665176392,  0.3278352618217468 ,
   0.10829252749681473, -0.15832512080669403,  0.339426726102829  ,
  -1.0223599672317505 ,  1.881224513053894  , -0.2748214602470398 ,
   0.01961503550410271, -0.21064290404319763, -0.17881713807582855,
  -0.7760581374168396 , -0.8233473300933838 , -0.42744603753089905,
  -0.01651543751358986,  0.21126927435398102, -2.2837233543395996 ,
  -1.0959372520446777 , -1.3718020915985107 , -0.15065616369247437,
  -0.939767062664032  , -0.27310505509376526, -0.4643179476261139 ,
   0.2087348848581314 , -0.4368783235549927 , -0.9545827507972717 ,
  -1.2078503370285034 , -1.1837780475616455 ],
 [ 0.00083514914149418, -0.04513458162546158,  0.09589657187461853,
   0.3966095447540283 , -0.4294690489768982 ,  0.4023449420928955 ,
  -0.02357864193618298,  0.01744949817657471,  0.07654830068349838,
   0.03563074395060539, -0.379338800907135  , -0.19933506846427917,
  -0.08715619891881943, -0.5756482481956482 , -0.8076767325401306 ,
  -0.3964218497276306 ,  0.3635021150112152 , -0.5136000514030457 ,
  -0.283413827419281  , -1.020939588546753  ,  0.07366978377103806,
   0.04488734528422356, -0.00935071613639593,  0.03538484871387482,
  -0.3160035014152527 , -0.11492117494344711, -0.04848530516028404,
  -2.504075527191162  ,  0.12410879880189896, -0.2908884882926941 ,
  -0.29310813546180725, -0.07701703906059265],
 [-2.017245054244995  , -3.1653692722320557 ,  0.2452540397644043 ,
  -1.5160645246505737 , -0.7171490788459778 , -1.997674822807312  ,
  -0.28721296787261963, -0.42045095562934875, -0.8174000382423401 ,
  -1.3885351419448853 , -1.9526232481002808 , -0.141107439994812  ,
  -0.2735600769519806 , -1.4313665628433228 , -0.2806698977947235 ,
  -0.1266557276248932 , -0.802132785320282  ,  0.22403500974178314,
  -0.7289188504219055 , -0.04513256251811981, -1.1573342084884644 ,
  -1.4415127038955688 ,  0.0748782530426979 , -0.13091127574443817,
  -0.2263382077217102 , -0.9182993173599243 , -0.16215725243091583,
  -0.24797439575195312, -0.37219780683517456, -1.2343693971633911 ,
   0.20679523050785065,  0.00461897859349847],
 [-0.21152831614017487,  0.23490457236766815,  0.11892787367105484,
  -0.810718834400177  ,  0.33458083868026733, -0.8615867495536804 ,
  -0.19477687776088715,  0.01815305836498737, -1.1763551235198975 ,
  -0.09863171726465225, -0.6282254457473755 ,  0.40763986110687256,
  -0.8114888668060303 , -0.02426529489457607, -1.832296371459961  ,
  -0.1820869743824005 , -0.17552243173122406,  0.3491273522377014 ,
  -0.2553888261318207 , -0.6295288801193237 , -0.09815569967031479,
  -0.23357711732387543, -0.20198841392993927,  0.10475192219018936,
  -0.08649817854166031, -0.01393827050924301, -0.615836501121521  ,
  -2.4693102836608887 ,  0.00358641962520778,  0.3117528557777405 ,
  -0.36317938566207886, -0.11975173652172089],
 [-0.5628507733345032 ,  0.15725383162498474, -0.03049303218722343,
   0.5944538116455078 ,  1.0961403846740723 , -1.121677041053772  ,
  -0.20644089579582214,  0.19416286051273346,  0.01164252404123545,
   0.28313764929771423, -1.3519176244735718 ,  0.3927999436855316 ,
   0.40436697006225586,  0.02218456193804741, -2.056063413619995  ,
  -0.8411658406257629 , -0.21407237648963928,  0.10981754213571548,
  -0.5953443646430969 , -0.13990585505962372,  0.8429347276687622 ,
  -0.7793000340461731 ,  0.16902612149715424,  0.10514094680547714,
   0.2798169255256653 , -0.2539713978767395 , -0.2629181742668152 ,
  -2.0490787029266357 ,  0.7267370223999023 ,  0.28267204761505127,
   0.48906606435775757, -0.12415730953216553],
 [-0.23354889452457428, -0.7314664125442505 , -0.06356281787157059,
  -1.3614299297332764 ,  0.42973899841308594, -0.493075430393219  ,
   0.02511196956038475, -0.05150596424937248, -0.11304668337106705,
  -0.581176221370697  , -0.18761201202869415, -0.5162352919578552 ,
   0.2964416742324829 , -1.200322151184082  ,  0.102634496986866  ,
  -0.2815379500389099 , -0.2701781094074249 ,  0.11326221376657486,
  -0.07385344803333282, -0.03948218002915382, -1.1347036361694336 ,
  -0.46065035462379456, -0.03611036762595177, -0.00339333480224013,
  -0.09692686796188354, -0.07699444890022278, -0.2374124675989151 ,
  -0.12944020330905914,  0.08035732060670853, -0.24409691989421844,
  -0.6404574513435364 , -2.341789484024048  ],
 [ 0.14585216343402863,  0.10803260654211044, -0.2854379117488861 ,
  -1.0112859010696411 , -0.7217223644256592 , -0.10310424864292145,
  -0.19856135547161102, -0.326386958360672  , -0.47370070219039917,
   0.13379235565662384,  0.3812604546546936 , -0.09874409437179565,
   0.5496923923492432 ,  0.6788585782051086 , -0.15716689825057983,
   0.01379586104303598,  0.3460274636745453 , -1.2512824535369873 ,
  -0.5729635953903198 , -0.0376875251531601 ,  0.09726378321647644,
  -0.06422244012355804,  0.19192732870578766,  0.01260306406766176,
  -0.37730905413627625, -0.08919155597686768, -0.08577985316514969,
   0.13625413179397583, -1.130244493484497  ,  0.04198210313916206,
  -1.2925962209701538 , -0.18720823526382446],
 [-0.6441646218299866 ,  0.639995813369751  , -0.24931970238685608,
  -0.5106794834136963 ,  0.21265369653701782,  0.38174256682395935,
   0.1966521441936493 , -0.11304286867380142, -0.7189093828201294 ,
  -0.5795102119445801 , -1.3908605575561523 ,  0.6448732614517212 ,
   0.10434376448392868, -0.1329323649406433 , -0.3703870475292206 ,
   0.00081345159560442, -0.5068556666374207 ,  1.0457446575164795 ,
   0.086735799908638  , -0.05793869495391846, -1.0064949989318848 ,
  -1.037561297416687  , -0.5940038561820984 , -0.26522737741470337,
   0.38055890798568726,  0.3296521306037903 , -0.1216672882437706 ,
  -0.02486204169690609, -0.4823227822780609 , -0.7029386162757874 ,
  -1.0771452188491821 , -0.36000800132751465],
 [ 0.5146501660346985 ,  0.4663074314594269 ,  0.00928797945380211,
  -0.14313460886478424, -1.7677595615386963 , -1.0556296110153198 ,
   0.00389030878432095, -0.05765280872583389, -0.8246667981147766 ,
   0.1915905475616455 , -0.24440450966358185,  0.35064697265625   ,
   0.4174596071243286 , -2.678922653198242  , -0.11068791151046753,
  -0.45772773027420044,  0.32309266924858093, -1.8863563537597656 ,
   0.3348143398761749 , -0.43790915608406067, -0.02396252378821373,
   0.06934190541505814, -0.587195873260498  , -0.05882584676146507,
  -0.39309632778167725,  0.00847619026899338, -0.26008322834968567,
  -0.37363189458847046,  0.1302633136510849 ,  0.0769277811050415 ,
  -0.287526398897171  , -2.107628583908081  ],
 [-0.17592264711856842, -0.28098803758621216, -0.11334379017353058,
  -0.25005555152893066, -0.39311039447784424, -0.4368995428085327 ,
   0.33088353276252747, -0.13500617444515228, -0.03882316499948502,
   0.13577517867088318, -0.18714044988155365, -0.14100195467472076,
  -0.03437734395265579, -0.7007973194122314 , -0.12010794132947922,
  -0.08699305355548859, -0.2017054557800293 , -0.15753906965255737,
  -0.03491935878992081, -0.05056799203157425, -0.23625028133392334,
  -0.34394049644470215, -0.02889042347669601, -0.08218491822481155,
  -0.14788733422756195, -0.328003853559494  ,  0.22472847998142242,
   0.08917562663555145, -0.08249974250793457, -0.4290662407875061 ,
  -0.6479170918464661 , -0.860862672328949  ],
 [-0.9819568991661072 , -1.533365249633789  , -0.19440129399299622,
   0.8468438982963562 , -0.10391506552696228, -0.7009137868881226 ,
  -0.00373419770039618, -0.6212604641914368 , -1.512886881828308  ,
  -1.3604860305786133 , -1.2120202779769897 ,  0.09788462519645691,
   0.07797534018754959, -1.7750746011734009 ,  0.05333889648318291,
  -0.5753107070922852 , -1.231690526008606  , -0.7145708203315735 ,
  -1.8901987075805664 , -0.3071010410785675 , -0.4784006178379059 ,
  -0.7019491791725159 ,  0.03537175804376602, -0.6398873329162598 ,
  -0.3190457224845886 , -0.37102222442626953, -0.05365951359272003,
  -0.42755040526390076, -0.08525392413139343,  0.00703016808256507,
  -0.4588135778903961 , -1.586910367012024  ],
 [-0.19824375212192535, -0.0364123173058033 ,  0.03764443472027779,
  -0.10152376443147659, -2.457449197769165  ,  0.06805691868066788,
  -0.23352426290512085, -0.08765503764152527, -0.0696469247341156 ,
  -0.06505226343870163, -0.11208941042423248, -0.21593241393566132,
  -0.48056551814079285, -0.17910584807395935, -0.7146757245063782 ,
  -0.20856033265590668,  0.1654595285654068 , -2.907186269760132  ,
  -0.24463559687137604, -1.1952344179153442 , -0.08792564272880554,
  -0.0540534183382988 ,  0.00310641434043646, -0.01745443977415562,
  -0.15958955883979797, -0.19995127618312836, -0.39509451389312744,
  -1.1891995668411255 , -0.5821740031242371 , -0.2906155586242676 ,
  -0.21612828969955444, -0.0765695869922638 ],
 [ 0.3701987862586975 , -0.29717952013015747,  0.08798930048942566,
  -0.8678694367408752 ,  0.7382809519767761 , -0.30730706453323364,
  -0.1760469675064087 ,  0.06186380982398987, -0.42127129435539246,
   0.43147405982017517, -0.4004674553871155 , -0.03578489646315575,
  -0.23315615952014923, -0.23732709884643555,  0.15793071687221527,
  -0.02397718839347363, -0.4634338617324829 ,  0.6913402080535889 ,
   0.01416889578104019,  0.12608548998832703,  0.11715803295373917,
  -0.01530506182461977,  0.01084477920085192,  0.01328636240214109,
  -0.04094419255852699,  0.24533358216285706, -0.17533037066459656,
  -0.00188483367674053, -0.7215133905410767 ,  0.10152236372232437,
  -0.16334010660648346, -0.45561861991882324],
 [ 0.23728707432746887, -1.1150282621383667 , -0.35194748640060425,
  -0.5245648622512817 ,  0.3719521462917328 , -0.18860876560211182,
   0.09715873003005981, -0.7529491782188416 ,  0.20681628584861755,
   0.07505553215742111,  0.15993621945381165, -0.7878246903419495 ,
   0.5469675660133362 , -1.1919665336608887 , -0.5104637742042542 ,
  -0.10436255484819412,  0.11341270059347153, -0.4071379601955414 ,
   0.7432916164398193 , -0.06152687966823578, -0.14160096645355225,
  -0.1578027307987213 , -0.8899543285369873 ,  0.02561420388519764,
  -0.20926102995872498, -0.19482068717479706, -0.21629200875759125,
  -0.5049744844436646 , -0.8760384321212769 , -0.15063247084617615,
  -2.3415815830230713 , -2.2920119762420654 ],
 [-0.0034432855900377 , -0.5163053870201111 , -0.24568916857242584,
  -0.3530437648296356 ,  0.34729164838790894, -0.38389748334884644,
  -0.21472430229187012, -0.1193147748708725 ,  0.03225214779376984,
  -0.4576815068721771 ,  0.09512149542570114, -0.03925140202045441,
   0.08017756044864655, -0.03060208819806576,  0.38195446133613586,
  -0.09550248831510544, -0.00350730633363128,  0.21180804073810577,
  -0.01650003716349602, -0.6060699224472046 , -0.7091297507286072 ,
  -0.1558770090341568 , -0.12535686790943146,  0.04130067676305771,
   0.11597340553998947,  0.01078120339661837, -0.2586749792098999 ,
  -0.26955652236938477, -0.5977733731269836 , -0.01525336038321257,
   0.11542095988988876, -0.6436821222305298 ],
 [-0.2970735430717468 , -0.5198578238487244 , -0.02548931911587715,
   0.34116652607917786, -0.4206703007221222 , -0.5890243053436279 ,
  -0.16184686124324799, -0.4748738408088684 ,  0.4393604099750519 ,
  -0.3287009000778198 , -0.05581182613968849, -0.3387333154678345 ,
   0.44247937202453613, -0.01645903661847115,  0.0891042947769165 ,
  -0.5130191445350647 , -0.22086092829704285, -0.19683684408664703,
  -0.13473114371299744,  0.7016438841819763 ,  0.8741158843040466 ,
  -0.4979105591773987 , -0.22022247314453125, -0.5688262581825256 ,
  -0.5184341073036194 , -0.4107961356639862 , -0.307550847530365  ,
   0.00233960198238492, -0.4891458749771118 , -0.02970734611153603,
  -0.2299935221672058 , -0.00276030879467726],
 [-0.0724370926618576 , -0.01281428709626198, -0.3581152558326721 ,
  -0.7558544278144836 ,  1.1394603252410889 ,  0.2736046314239502 ,
   0.05241944640874863,  0.04345105215907097,  0.14613938331604004,
  -0.7175952196121216 ,  0.3231004774570465 , -0.2381095290184021 ,
   0.22645030915737152, -0.20929965376853943, -0.28707370162010193,
  -0.2811037003993988 ,  0.10393962264060974,  1.139265775680542  ,
   0.5645887851715088 ,  0.1343374252319336 , -1.2330061197280884 ,
   0.06999333202838898, -0.38399550318717957,  0.06609562039375305,
   0.20140251517295837, -0.30889642238616943, -0.18420852720737457,
  -0.150213822722435  , -1.0613662004470825 , -0.0627138763666153 ,
  -0.6706194877624512 ,  0.2540988624095917 ],
 [ 0.03529936820268631, -0.3433248996734619 ,  0.03323100507259369,
   0.3974941372871399 ,  0.07626531273126602, -0.07653381675481796,
  -0.33454203605651855,  0.02831593714654446,  0.15629994869232178,
   0.2035401314496994 ,  0.583976686000824  , -0.18832361698150635,
  -0.24387069046497345, -0.5571914911270142 , -1.0111570358276367 ,
  -0.11105030030012131,  0.21867550909519196, -1.1062382459640503 ,
   0.05018915608525276, -0.464701771736145  ,  0.02847555838525295,
  -0.17915479838848114, -0.08313167840242386,  0.01257722452282906,
  -0.24531802535057068, -0.11798819154500961,  0.0129376444965601 ,
   0.23685897886753082, -0.3462070822715759 , -0.2704797387123108 ,
  -0.225099578499794  , -0.15197843313217163],
 [ 0.5943059921264648 ,  0.31964394450187683,  0.01005275454372168,
  -1.0696585178375244 ,  1.2628589868545532 ,  0.1353360116481781 ,
   0.09750650823116302,  0.14548027515411377, -1.7881617546081543 ,
   0.5674806833267212 , -0.46313342452049255,  0.3923720717430115 ,
  -0.8568546772003174 ,  0.35176390409469604, -2.6648991107940674 ,
   0.2311200499534607 , -0.00672584120184183,  1.3044060468673706 ,
  -0.0885910838842392 , -0.9538291096687317 , -0.19170047342777252,
   0.10534896701574326, -0.6384148597717285 , -0.04634805023670197,
  -0.15142527222633362, -0.1021672710776329 , -1.1426204442977905 ,
  -2.0561134815216064 , -0.4047921895980835 ,  0.4291030466556549 ,
   0.8258655667304993 ,  0.475943386554718  ],
 [-1.3513582944869995 , -0.6123432517051697 ,  0.07226826995611191,
  -2.1734297275543213 , -0.29865869879722595, -0.8873555660247803 ,
  -0.6874421834945679 , -0.9271148443222046 , -0.9377018809318542 ,
  -1.0523003339767456 ,  1.0108038187026978 , -0.22686132788658142,
  -0.472305566072464  , -3.2141671180725098 , -0.9258807897567749 ,
  -0.6871410608291626 , -0.27238184213638306, -0.30835235118865967,
  -0.137599915266037  ,  0.04780524224042892, -0.5492548942565918 ,
   0.3097383975982666 ,  0.6388585567474365 , -0.9081699848175049 ,
  -0.8482325077056885 , -0.22153343260288239, -0.6215327382087708 ,
  -0.16631858050823212, -1.346383810043335  ,  1.1360267400741577 ,
  -3.28482723236084   , -1.963658332824707  ],
 [-0.6576601266860962 , -0.8963465690612793 ,  0.13904954493045807,
  -0.11953403055667877, -0.45979851484298706, -0.8475193381309509 ,
  -0.48712924122810364, -0.23165270686149597, -0.02772461250424385,
  -0.14559470117092133, -0.19575825333595276, -0.08160220086574554,
  -0.6081887483596802 , -0.17293629050254822,  0.3510836958885193 ,
  -0.47453805804252625, -0.19127044081687927, -0.13178107142448425,
  -0.8165833353996277 ,  0.5052484273910522 ,  0.4393027126789093 ,
  -0.5759115815162659 , -0.34058889746665955, -0.00969504192471504,
  -0.48893237113952637,  0.23742985725402832,  0.23612719774246216,
  -1.5921955108642578 , -0.16385172307491302,  0.10482925921678543,
   0.02818598784506321, -0.6841174960136414 ],
 [-0.7357063293457031 ,  0.5911259055137634 ,  0.01388886850327253,
   0.1396588534116745 , -0.208054780960083  , -1.591467261314392  ,
   0.10586296021938324, -0.1025862991809845 ,  0.39549997448921204,
  -0.4952089786529541 , -0.6066093444824219 ,  0.04848971217870712,
   0.09645792841911316, -0.2300877571105957 , -0.08725439012050629,
  -0.12727254629135132, -0.3601936399936676 , -0.2676827907562256 ,
  -0.01127375662326813, -0.06509530544281006,  0.11081018298864365,
   0.2867875099182129 , -0.5047484040260315 , -0.28412964940071106,
   0.01191383693367243,  0.04212947562336922, -0.03125885128974915,
  -0.1474093347787857 , -0.35631754994392395,  0.9201076626777649 ,
  -0.6323352456092834 ,  0.1933276653289795 ],
 [ 0.15839028358459473, -0.4443204998970032 , -0.1477603167295456 ,
   0.18655291199684143,  0.7486083507537842 , -0.02306529134511948,
   0.4628843665122986 , -0.20959210395812988, -0.5309320092201233 ,
  -0.4027203917503357 , -1.154904842376709  ,  0.07256627082824707,
  -0.10972455143928528, -0.43252384662628174,  0.21728424727916718,
  -0.518142819404602  ,  0.31942880153656006,  0.01306759007275105,
  -1.1318868398666382 , -0.3024432957172394 ,  0.22062940895557404,
  -0.00276701897382736,  0.18091906607151031,  0.0063946028240025 ,
   0.12389540672302246,  0.10066857933998108,  0.37083473801612854,
   0.2928774058818817 , -0.4369701147079468 ,  0.1302376538515091 ,
  -0.7516567707061768 ,  0.5270422697067261 ],
 [ 0.07637957483530045, -0.44878730177879333,  0.04323263466358185,
  -0.2545869052410126 , -0.8343888521194458 , -0.4760175049304962 ,
   0.02028896100819111, -0.04742377251386642, -0.695975124835968  ,
   0.19802778959274292, -0.3509051501750946 , -0.1467294842004776 ,
  -0.33611446619033813, -0.4902269244194031 ,  0.06546958535909653,
   0.04854597896337509,  0.08870051056146622, -0.993495762348175  ,
   0.33488842844963074,  0.00442333798855543, -0.38658952713012695,
  -0.00099202955607325, -0.08503355830907822,  0.00235762889496982,
   0.00519088096916676, -0.04484128579497337, -0.12423256039619446,
   0.05457587540149689,  0.12899920344352722,  0.11855313926935196,
   0.05322489887475967, -0.3324698209762573 ],
 [-0.7598746418952942 , -0.2870917320251465 ,  0.02525991573929787,
   0.3335098624229431 , -0.5792418122291565 , -1.8110110759735107 ,
  -0.15771642327308655, -0.38574060797691345,  0.38846516609191895,
  -0.7650966644287109 , -0.25769171118736267, -0.8822968006134033 ,
  -0.5414047241210938 ,  0.39193692803382874,  0.4355618357658386 ,
  -0.18222981691360474, -0.5021231174468994 , -0.3970488905906677 ,
  -0.43985632061958313, -0.09768642485141754, -0.48306000232696533,
  -0.18407166004180908, -0.4828868806362152 , -0.8798661828041077 ,
  -0.1684868484735489 , -0.17438940703868866,  0.5337188243865967 ,
  -0.18740248680114746,  0.31788170337677   , -0.15347380936145782,
   0.0673811137676239 , -0.27080145478248596],
 [ 0.5381799936294556 ,  0.20933972299098969, -0.06180458888411522,
  -1.4886431694030762 ,  1.199837327003479  , -0.7666352987289429 ,
   0.12005585432052612, -0.01475129369646311, -1.9436707496643066 ,
   1.0236490964889526 ,  0.5329834222793579 ,  0.8831916451454163 ,
   0.23922164738178253, -0.10849474370479584,  0.7358161807060242 ,
   0.11598841845989227,  0.7595616579055786 ,  1.0846073627471924 ,
  -0.41161590814590454,  0.11544058471918106, -0.02846282348036766,
  -0.00601874943822622,  0.04466260969638824, -0.0730106309056282 ,
   0.00140843156259507, -1.226516842842102  , -1.190710186958313  ,
   0.748700737953186  ,  0.6662890911102295 ,  0.37058714032173157,
   0.41626793146133423,  0.3588401973247528 ],
 [-0.9474169611930847 , -0.5919231176376343 , -0.29854318499565125,
   0.43357202410697937,  0.3015434443950653 , -0.06175236403942108,
   0.19742293655872345, -0.31596609950065613,  0.17703591287136078,
  -0.5628354549407959 , -0.8326120376586914 ,  0.1177225410938263 ,
   0.24477845430374146, -1.752151370048523  , -0.4035446047782898 ,
  -0.00318729854188859,  0.00459804479032755,  0.360822468996048  ,
  -0.3619106113910675 , -0.45482948422431946,  0.399542897939682  ,
  -0.07396921515464783, -0.7539510130882263 ,  0.08498283475637436,
  -0.3045214116573334 , -0.0078436816111207 ,  0.30472737550735474,
   0.4365342855453491 , -0.44747692346572876, -0.01071817893534899,
  -0.8397650122642517 , -2.2717955112457275 ],
 [-0.5631745457649231 , -1.1888301372528076 , -0.6556458473205566 ,
   0.2667357921600342 ,  1.4489836692810059 , -0.2700652778148651 ,
   0.5560817718505859 , -0.2641153037548065 ,  1.1875782012939453 ,
  -0.8645689487457275 ,  0.5100314617156982 ,  0.4968424141407013 ,
   0.7337682843208313 ,  0.7974762320518494 ,  0.09036976844072342,
  -0.4923222064971924 , -0.00438076164573431,  2.011361837387085  ,
   0.18504400551319122, -1.472128987312317  ,  0.9623266458511353 ,
  -0.15174484252929688,  0.05332044884562492, -0.6288954019546509 ,
  -0.33269861340522766, -0.26556622982025146,  0.5225963592529297 ,
  -0.00916915573179722,  1.5269254446029663 ,  0.48939257860183716,
   0.386360764503479  , -1.023144245147705  ],
 [ 0.45541003346443176,  0.01299749873578548, -0.2538878321647644 ,
  -0.19065238535404205, -0.21178676187992096,  0.7282431721687317 ,
   0.12313807755708694,  0.03063746355473995,  0.7345971465110779 ,
  -1.0788887739181519 ,  0.02125293761491776, -0.34909138083457947,
  -0.21008265018463135,  0.19479195773601532,  0.27748188376426697,
   0.00213133497163653, -1.1510018110275269 ,  0.00625884253531694,
   0.34411510825157166,  0.1329120546579361 ,  0.2784114181995392 ,
  -0.3416920602321625 ,  0.17677409946918488, -0.5423359274864197 ,
   0.14528770744800568, -0.48837894201278687, -0.2300003319978714 ,
  -0.18378765881061554,  0.5891032218933105 ,  0.31411924958229065,
   0.00821980834007263,  0.11768248677253723],
 [-0.6290936470031738 , -0.6990250945091248 ,  0.01493114419281483,
  -0.26227590441703796,  0.35237494111061096, -1.061768889427185  ,
  -0.15586210787296295, -0.09223613142967224,  0.7253584265708923 ,
  -0.437908411026001  , -1.63591730594635   ,  0.3300609290599823 ,
  -0.1417333483695984 , -3.2350785732269287 , -0.19843542575836182,
  -0.27901700139045715, -0.3631203770637512 ,  0.11453692615032196,
  -0.4733293652534485 , -0.07136668264865875, -1.0359500646591187 ,
   0.5802057981491089 , -1.3577901124954224 , -0.099312424659729  ,
  -0.602066695690155  , -0.2215946465730667 , -0.1024353951215744 ,
   0.2586402893066406 ,  0.08909090608358383, -0.6011858582496643 ,
  -0.2893919050693512 , -0.7885823845863342 ],
 [ 0.37989768385887146,  0.3577999472618103 ,  0.05748562142252922,
  -0.45029181241989136, -0.30925998091697693, -0.09449993818998337,
   0.04147388786077499, -0.1751849502325058 ,  0.6590831279754639 ,
   0.19746267795562744, -1.4413071870803833 , -0.16826131939888   ,
   0.4540589451789856 ,  0.03370846062898636, -0.00665903277695179,
  -0.06654367595911026,  0.4495382308959961 ,  0.3922102153301239 ,
  -1.0002847909927368 , -0.6152399778366089 , -0.18401703238487244,
   0.00259825377725065, -0.1289404034614563 ,  0.02091778442263603,
  -0.03827252611517906, -0.17956416308879852, -0.42176657915115356,
  -0.5038476586341858 , -1.2095353603363037 ,  0.7137869000434875 ,
  -0.5288551449775696 , -0.09556512534618378],
 [ 0.8544023633003235 ,  0.20715580880641937, -1.1934894323349    ,
  -1.248007893562317  ,  1.1005779504776    ,  1.0369230508804321 ,
   0.30487483739852905,  0.05108997970819473, -0.8576030135154724 ,
   0.15780633687973022,  0.3004350960254669 ,  0.19144989550113678,
   0.48004069924354553,  0.1723531037569046 ,  0.6217974424362183 ,
   0.11815901845693588,  0.4433184862136841 ,  1.4458297491073608 ,
   0.09905087947845459,  1.6905313730239868 , -0.09596990048885345,
  -0.09961412847042084,  0.19944103062152863, -0.03483782336115837,
   0.25645846128463745, -0.43186184763908386,  0.3474195599555969 ,
  -3.985861301422119  ,  0.278502494096756  ,  0.09761633723974228,
   0.24318407475948334,  0.5581030249595642 ],
 [-0.4585561454296112 , -0.9677499532699585 , -0.5683925747871399 ,
  -1.072974443435669  , -0.3751947283744812 , -0.04273759946227074,
   0.02461115084588528, -0.00916772056370974, -0.04004498198628426,
  -1.9652711153030396 , -0.00304658780805767, -0.7972750663757324 ,
  -0.04933401942253113, -0.19789518415927887, -0.2717318534851074 ,
  -0.10866636782884598, -0.15146328508853912,  0.2741261124610901 ,
   0.32360559701919556, -0.4795163869857788 ,  0.04545850679278374,
   0.20534613728523254, -0.8982376456260681 , -0.1288103610277176 ,
  -0.18063409626483917, -0.3889298737049103 ,  0.5088982582092285 ,
  -0.08418173342943192,  0.3245362341403961 , -0.5341160297393799 ,
  -0.7526211142539978 ,  0.41965606808662415],
 [ 0.6379613876342773 ,  0.06597498804330826,  0.11713998019695282,
   0.729846715927124  , -0.8355238437652588 ,  0.03312569484114647,
  -0.27611058950424194, -0.08271539211273193,  0.4020794630050659 ,
   0.6707677841186523 ,  0.6129530668258667 ,  0.01975155435502529,
  -0.46054378151893616, -1.4577562808990479 , -0.1585170477628708 ,
  -0.05037663131952286,  0.24352964758872986, -0.4488688111305237 ,
  -0.2577403485774994 ,  0.05110416188836098, -1.1212347745895386 ,
   0.11936398595571518, -0.22658175230026245,  0.03515034914016724,
  -0.01408576313406229,  0.10130614042282104,  0.49438920617103577,
  -2.0791804790496826 , -0.48335835337638855, -0.09732693433761597,
   0.06861069798469543, -1.5010701417922974 ],
 [ 0.6522149443626404 ,  0.6603809595108032 , -0.104596346616745  ,
  -0.5666607618331909 , -0.9116154313087463 , -0.915606677532196  ,
   0.2760983109474182 ,  0.2663395404815674 , -1.4150358438491821 ,
   0.8236171007156372 ,  0.20348985493183136, -0.12312890589237213,
   0.8322836756706238 , -0.8231362700462341 ,  0.23394201695919037,
   0.25107675790786743,  0.7374396920204163 , -0.09804053604602814,
  -1.9840822219848633 , -1.705074429512024  ,  0.6700058579444885 ,
  -0.5025085806846619 ,  0.34935060143470764, -0.1650254875421524 ,
  -0.3546186685562134 , -0.12858042120933533,  0.09097827225923538,
   1.2150615453720093 , -0.45675522089004517,  0.6721910238265991 ,
   0.4674275815486908 , -0.135380819439888  ],
 [-0.12716440856456757,  0.36872509121894836,  0.0950097143650055 ,
  -0.7441834211349487 ,  0.6160241961479187 , -0.1976047307252884 ,
   0.03194843977689743, -0.0409579761326313 ,  0.13604885339736938,
  -0.384658545255661  , -0.23342426121234894,  0.41934138536453247,
  -0.22405602037906647, -0.06784455478191376, -0.3865021765232086 ,
  -0.20879851281642914,  0.4920956492424011 ,  0.35359764099121094,
  -0.26661786437034607, -0.34948357939720154, -0.823933482170105  ,
  -0.00132575130555779,  0.0047259284183383 , -0.04194214940071106,
  -0.19169184565544128,  0.01795072667300701, -0.3798728883266449 ,
  -0.33307355642318726,  0.01310882810503244, -0.27469027042388916,
  -0.17368344962596893, -0.04295573011040688],
 [ 0.18961109220981598, -1.1555860042572021 , -0.16659662127494812,
  -1.5827968120574951 , -1.0121841430664062 , -0.38124534487724304,
   0.08617197722196579, -0.0990639477968216 ,  0.3417104184627533 ,
  -0.1910502314567566 , -1.1262264251708984 , -0.24410350620746613,
  -0.16534045338630676, -1.2797938585281372 ,  0.23087364435195923,
   0.01717016845941544, -0.6071800589561462 , -0.9967734217643738 ,
  -0.25094708800315857, -0.6450546979904175 ,  0.31963711977005005,
  -0.3764682710170746 , -1.3557051420211792 , -0.07786982506513596,
   0.06205590441823006, -0.9751986861228943 ,  0.20079664885997772,
   0.4978538453578949 , -0.01522908546030521, -0.2970341145992279 ,
  -0.5298734903335571 , -0.9705998301506042 ],
 [ 0.6944364309310913 , -0.18807272613048553, -0.17977523803710938,
  -0.3369908928871155 ,  0.07819201052188873,  0.05714013800024986,
  -0.05702903121709824, -0.10479545593261719, -0.5699579119682312 ,
   0.19655364751815796,  0.8149197697639465 , -0.31359201669692993,
   0.21738044917583466, -0.8778138160705566 ,  0.3774684965610504 ,
  -0.15548083186149597,  0.4276953935623169 ,  0.3793219029903412 ,
  -0.636108934879303  , -0.04840448498725891,  0.30902814865112305,
  -0.07978610694408417, -0.28314998745918274, -0.16914238035678864,
   0.04784339666366577, -0.3131740689277649 , -0.13900020718574524,
  -0.6832456588745117 ,  0.5551578402519226 ,  0.09357019513845444,
  -0.3183850347995758 ,  0.1523110568523407 ],
 [ 0.35785892605781555, -0.01885700412094593, -0.04963000491261482,
   0.27239862084388733, -0.41918665170669556,  0.46284377574920654,
   0.5283048152923584 ,  0.00415359297767282,  0.16577333211898804,
  -0.10243377089500427, -0.9922869801521301 ,  0.08641406148672104,
   0.40132594108581543,  0.11465801298618317,  0.4001779556274414 ,
  -0.04626258835196495,  0.1831718534231186 , -0.8618781566619873 ,
  -0.1619570255279541 ,  0.0294343251734972 ,  0.44602668285369873,
   0.15308552980422974, -1.131011724472046  , -0.0413038581609726 ,
   0.07717030495405197,  0.11108136177062988,  0.11058668792247772,
  -0.08143527060747147, -0.1200689747929573 ,  0.4323272109031677 ,
  -0.7873618602752686 ,  0.3114486336708069 ],
 [-0.21306593716144562,  0.507864773273468  , -0.0937328040599823 ,
   0.42767634987831116,  0.19860497117042542,  0.3123582601547241 ,
   0.3059566915035248 , -0.13465087115764618, -0.5163570642471313 ,
  -0.7064782381057739 , -1.3684964179992676 , -0.1566585749387741 ,
   0.24556799232959747, -2.2948524951934814 ,  0.14886750280857086,
   0.05461558699607849, -0.195171520113945  ,  0.17126183211803436,
   0.06069719791412354, -0.10784834623336792, -0.7663618922233582 ,
   0.04921292886137962, -0.9811428785324097 , -0.1397038996219635 ,
  -0.10028574615716934, -0.12989331781864166, -0.03607373684644699,
  -0.00948310177773237, -0.2630540728569031 ,  0.23937922716140747,
  -0.6566455960273743 , -1.3267914056777954 ],
 [-0.07278164476156235,  0.02192972041666508,  0.16637249290943146,
   0.2063513994216919 ,  0.00525292940437794, -0.41606834530830383,
  -0.25185084342956543, -0.1790248304605484 , -0.00789531879127026,
  -0.06338509917259216,  0.06649088859558105,  0.08908393234014511,
  -0.6643863916397095 , -0.10603955388069153, -0.7398091554641724 ,
  -0.02058787643909454,  0.24554096162319183, -0.03371089324355125,
  -0.4233497083187103 , -0.6264923810958862 , -0.3661678731441498 ,
  -0.1590711772441864 , -0.07414023578166962, -0.08235646039247513,
  -0.5685299634933472 ,  0.0449766181409359 ,  0.3256058096885681 ,
  -0.26939696073532104, -0.29282763600349426, -0.8693422079086304 ,
  -0.07603737711906433, -0.3349439799785614 ],
 [-0.9782552123069763 , -0.05789621919393539, -0.172989159822464  ,
  -0.282355934381485  , -0.05371135473251343,  0.05921952426433563,
   0.17863251268863678, -0.04035494849085808,  0.7832731604576111 ,
  -0.1968509554862976 ,  0.7816984057426453 , -0.21987958252429962,
  -0.19043324887752533, -0.47045737504959106, -0.2713794410228729 ,
  -0.325784295797348  , -0.06040773168206215, -1.059302568435669  ,
   0.03648845478892326,  0.03318046778440475, -0.1629362553358078 ,
   0.03761370107531548,  0.04665621370077133, -0.40220916271209717,
  -0.5924451351165771 ,  0.04874270409345627,  0.5154762864112854 ,
  -0.2597403824329376 , -0.7127649784088135 , -0.1888045072555542 ,
   0.06491253525018692,  0.37826773524284363],
 [ 0.09781540930271149, -0.5034408569335938 ,  0.384300172328949  ,
  -0.1971060335636139 , -0.1225571557879448 , -0.16391228139400482,
  -0.17560043931007385,  0.01496512722223997,  0.02079037018120289,
   0.17822809517383575, -0.44541868567466736, -0.08816217631101608,
  -0.1428440660238266 ,  0.2520087659358978 , -0.5461077690124512 ,
  -0.06520606577396393, -0.04621420800685883, -0.710685670375824  ,
   0.11542048305273056, -0.315588116645813  , -0.8751490116119385 ,
  -0.13571758568286896,  0.05795130506157875, -0.1914387196302414 ,
  -0.5532307624816895 , -0.15436863899230957, -0.01006502192467451,
  -0.23291721940040588, -2.36195969581604   , -0.3581776022911072 ,
   0.02160648815333843, -0.4601282775402069 ]])
    b5 = jnp.array([ 1.09262216091156   , -0.11125797033309937,  1.8163763284683228 ,
 -0.7091550827026367 , -1.057586431503296  , -0.33788374066352844,
  0.312472939491272  ,  0.10671119391918182, -0.8758371472358704 ,
 -1.0077273845672607 ,  0.6206343770027161 , -0.03406568616628647,
  0.3338933289051056 ,  0.5773726105690002 ,  1.5110219717025757 ,
  0.3100455701351166 ,  1.6645007133483887 , -1.1330175399780273 ,
  0.09150683134794235,  1.0530674457550049 ,  0.4788970947265625 ,
 -0.2303817868232727 , -0.06985171139240265, -0.5010262131690979 ,
  0.11067313700914383,  0.15218058228492737,  0.02768061682581902,
  1.441838264465332  , -0.25783872604370117,  0.8516780734062195 ,
  0.7663211226463318 ,  0.5821524858474731 ])
    W6 = jnp.array([[ 0.03296064957976341],
 [-0.07733018696308136],
 [ 0.15182553231716156],
 [-0.0818905234336853 ],
 [-0.09796320647001266],
 [-0.07937665283679962],
 [-0.1075182780623436 ],
 [-0.6831719279289246 ],
 [-0.09217619895935059],
 [-0.07515443861484528],
 [ 0.03122068382799625],
 [-0.2228831946849823 ],
 [-0.07702066749334335],
 [ 0.03819212689995766],
 [-0.09136472642421722],
 [-0.5080581307411194 ],
 [ 0.09341970831155777],
 [-0.06348160654306412],
 [-0.1101192981004715 ],
 [ 0.2286478728055954 ],
 [ 0.07409658282995224],
 [-0.49763214588165283],
 [ 0.01793276704847813],
 [-0.7409486174583435 ],
 [-0.6237672567367554 ],
 [-0.48677974939346313],
 [-0.04666214808821678],
 [-0.12367575615644455],
 [-0.14620859920978546],
 [ 0.0370597243309021 ],
 [ 0.04145015776157379],
 [ 0.03889959305524826]])
    b6 = jnp.array([-3.215707778930664])
    x = jnp.array([beta_st, log_pc] + list(log_ps_nsat))
    x = (x - mean_X) / scale_X
    h1 = jnn.gelu(jnp.dot(x, W1) + b1, approximate=False)
    h2 = jnn.gelu(jnp.dot(h1, W2) + b2, approximate=False)
    h3 = jnn.gelu(jnp.dot(h2, W3) + b3, approximate=False)
    h4 = jnn.gelu(jnp.dot(h3, W4) + b4, approximate=False)
    h5 = jnn.gelu(jnp.dot(h4, W5) + b5, approximate=False)
    out_log = jnp.dot(h5, W6) + b6
    return jnp.exp(out_log[0])
