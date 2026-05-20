r"""Gravitational wave event likelihood implementations"""

import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.flows.flow import Flow
from jesterTOV.logging_config import get_logger
import jesterTOV.utils as utils

logger = get_logger("jester")


class GWLikelihoodResampled(LikelihoodBase):
    """
    Gravitational wave likelihood for a single GW event using normalizing flow posteriors

    This likelihood evaluates the GW posterior by:
    1. Sampling masses (m1, m2) from the trained normalizing flow
    2. Interpolating tidal deformabilities (Λ1, Λ2) from the EOS
    3. Evaluating the NF log probability on (m1, m2, Λ1, Λ2)

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    flow : Flow
        Normalizing flow model for this GW event
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    flow: Flow

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Sample all N_masses_evaluation samples from NF in one go
        all_nf_samples: Float[Array, "n_samples 2"] = self.flow.sample(
            key, (self.N_masses_evaluation,)
        )

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single NF sample

            Note: jax.lax.map with batch_size still applies the function to individual
            elements, not batches. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Single sample with [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # batch_size helps with compilation memory, not runtime batching
        all_logprobs = jax.lax.map(
            process_sample, all_nf_samples, batch_size=self.N_masses_batch_size
        )

        # Average over all samples for this event
        log_likelihood = jnp.mean(all_logprobs)

        return log_likelihood


class GWLikelihood(LikelihoodBase):
    """
    Gravitational wave likelihood using pre-sampled masses for deterministic evaluation

    This likelihood improves upon GWLikelihoodResampled by pre-sampling mass pairs once at
    initialization, eliminating the need for the _random_key parameter and providing
    deterministic likelihood evaluations critical for sampler convergence.

    Key improvements over GWLikelihoodResampled:
    1. Deterministic: Same EOS parameters → same likelihood value
    2. No _random_key hack: Uses fixed seed at initialization
    3. Scalable: Can use N=10,000+ samples efficiently on GPU
    4. Fair comparison: All EOS evaluated at identical mass points
    5. Better convergence: Smooth likelihood surface for MCMC/SMC

    The likelihood works by:

    1. Pre-sampling (m1, m2) pairs from the trained flow at initialization
    2. For each EOS evaluation: interpolate Λ1, Λ2 from the candidate EOS at
       the fixed mass points, evaluate flow log_prob on (m1, m2, Λ1_EOS, Λ2_EOS),
       apply penalties for masses exceeding Mtov, and average over all
       pre-sampled mass pairs

    Parameters
    ----------
    event_name : str
        Name of the GW event (e.g., "GW170817")
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float, optional
        Penalty value for samples where masses exceed Mtov (default: 0.0, i.e. no penalty)
    N_masses_evaluation : int, optional
        Number of mass samples to pre-sample (default: 2000)
        Large values recommended - GPU parallelization makes this cheap!
    N_masses_batch_size : int, optional
        Batch size for jax.lax.map processing (default: 1000)
    seed : int, optional
        Random seed for mass pre-sampling (default: 42)
        Fixed seed ensures reproducibility across runs

    Attributes
    ----------
    event_name : str
        Name of the GW event
    model_dir : str
        Path to directory containing the trained normalizing flow model
    penalty_value : float
        Penalty value for samples where masses exceed Mtov
    N_masses_evaluation : int
        Number of pre-sampled mass pairs
    N_masses_batch_size : int
        Batch size for processing
    seed : int
        Random seed used for pre-sampling
    flow : Flow
        Normalizing flow model for this GW event
    fixed_mass_samples : Float[Array, "n_samples 2"]
        Pre-sampled (m1, m2) pairs from the flow, shape [N, 2]

    Notes
    -----
    This class does NOT require _random_key in the parameter dictionary,
    unlike GWLikelihoodResampled. The seed is only used once at initialization.

    GPU parallelization via jax.lax.map means N=10,000 samples costs nearly
    the same as N=20, so use large N for near-integration accuracy.

    Examples
    --------
    Configure in YAML::

        likelihoods:
          - type: "gw"
            enabled: true
            parameters:
              events:
                - name: "GW170817"
              N_masses_evaluation: 2000  # Default value
              N_masses_batch_size: 1000
              seed: 42
    """

    event_name: str
    model_dir: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int
    flow: Flow
    fixed_mass_samples: Float[Array, "n_samples 2"]

    def __init__(
        self,
        event_name: str,
        model_dir: str,
        penalty_value: float = 0.0,
        N_masses_evaluation: int = 2000,
        N_masses_batch_size: int = 1000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.event_name = event_name
        self.model_dir = model_dir
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        # Load Flow model for this event
        logger.info(f"Loading NF model for {event_name} from {model_dir}")
        self.flow = Flow.from_directory(model_dir)
        logger.info(f"Loaded NF model for {event_name}")

        # Pre-sample masses ONCE at initialization
        logger.info(
            f"Pre-sampling {N_masses_evaluation} mass pairs with seed={seed} for {event_name}"
        )
        key = jax.random.key(seed)
        samples = self.flow.sample(key, (N_masses_evaluation,))
        # Extract only (m1, m2), discard Lambda values from flow
        self.fixed_mass_samples = samples[:, :2]  # Shape: [N, 2]
        logger.info(
            f"Pre-sampled mass range: m1=[{jnp.min(self.fixed_mass_samples[:, 0]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 0]):.3f}] Msun, "
            f"m2=[{jnp.min(self.fixed_mass_samples[:, 1]):.3f}, "
            f"{jnp.max(self.fixed_mass_samples[:, 1]):.3f}] Msun"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

            Note: Does NOT require '_random_key' (unlike GWLikelihood)

        Returns
        -------
        Float
            Log likelihood value for this GW event
        """
        # Extract EOS parameters (no _random_key needed!)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        def process_sample(sample: Float[Array, " 2"]) -> Float:
            """
            Process a single pre-sampled mass pair

            Note: jax.lax.map with batch_size applies function to individual
            elements. The batch_size parameter is for compilation optimization.

            Parameters
            ----------
            sample : Float[Array, " 2"]
                Pre-sampled mass pair [m1, m2]

            Returns
            -------
            Float
                Log probability including penalties for this sample
            """
            m1 = sample[0]
            m2 = sample[1]

            # Interpolate lambdas from candidate EOS
            lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
            lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)

            # Evaluate log_prob on single sample
            ml_sample = jnp.array([m1, m2, lambda_1, lambda_2])
            logpdf = self.flow.log_prob(ml_sample)

            # Penalties for masses exceeding Mtov
            penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
            penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)

            # Return log prob + penalties for this sample
            return logpdf + penalty_m1 + penalty_m2

        # Use jax.lax.map with batching for memory-efficient processing
        # Process all pre-sampled mass pairs
        all_logprobs = jax.lax.map(
            process_sample, self.fixed_mass_samples, batch_size=self.N_masses_batch_size
        )

        # Take logsumexp over all pre-sampled mass pairs
        log_likelihood = logsumexp(all_logprobs) - jnp.log(self.N_masses_evaluation)

        return log_likelihood


class MockLambdaLikelihood(LikelihoodBase):
    """
    Mock 4D Likelihood evaluating deterministic skewed correlated posteriors for binary pairs.
    
    Reads a paired CSV (m1, m2, l1, l2), treats the diagonal covariance
    matrices efficiently by separating them into 2K independent 1D integrals,
    and sums the log-likelihoods correctly.
    """
    def __init__(
        self,
        csv_file: str,
        penalty_value: float = -1e10,
        N_masses_evaluation: int = 200,
        integration_sigma_cut: float = 8.0,
    ) -> None:
        super().__init__()
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.integration_sigma_cut = integration_sigma_cut
        self.y_key = "Lambdas_EOS"

        data = np.genfromtxt(csv_file, delimiter=",", names=True)
        self.K_pairs = len(data)
        self.K = self.K_pairs * 2  # Each binary pair contributes 2 separable component likelihoods

        # Parse component 1
        centers_1 = jnp.stack([data["Mass1_Center_Noise"], data["Lambda1_Center_Noise"]], axis=-1)
        std_m1 = jnp.array(data["Std_Mass1"])
        std_l1 = jnp.array(data["Std_Lambda1"])

        # Parse component 2
        centers_2 = jnp.stack([data["Mass2_Center_Noise"], data["Lambda2_Center_Noise"]], axis=-1)
        std_m2 = jnp.array(data["Std_Mass2"])
        std_l2 = jnp.array(data["Std_Lambda2"])

        # Stack into 2K independent effective observations
        self.centers = jnp.concatenate([centers_1, centers_2], axis=0)
        std_m = jnp.concatenate([std_m1, std_m2], axis=0)
        std_l = jnp.concatenate([std_l1, std_l2], axis=0)

        # Assume 0 cross-covariance and skew within the sub-components based on pair CSV structure
        cov_val = jnp.zeros(self.K)
        self.skews = jnp.zeros((self.K, 2))

        covs = np.zeros((self.K, 2, 2))
        covs[:, 0, 0] = std_m**2 + 1e-12
        covs[:, 1, 1] = std_l**2 + 1e-12
        self.covs = jnp.array(covs)

        self.inv_covs = jnp.linalg.inv(self.covs)
        self.log_det_covs = jnp.linalg.slogdet(self.covs)[1]
        self.omegas = jnp.sqrt(jnp.diagonal(self.covs, axis1=1, axis2=2))
        self.alpha_primes = self.skews / self.omegas

    def evaluate(self, params: dict) -> float:
        masses_EOS = params["masses_EOS"]
        y_EOS = params[self.y_key]
        
        valid_mask = jnp.isfinite(masses_EOS) & jnp.isfinite(y_EOS)
        mtov = jnp.max(jnp.where(valid_mask, masses_EOS, self.penalty_value))

        split_idx = utils.get_MR_split_index(masses_EOS, y_EOS)
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = (idx < split_idx) & valid_mask
        mask2 = (idx >= split_idx) & valid_mask

        SENTINEL = 1e30
        dummy_m = jnp.linspace(1e5, 1e5 + 10.0, masses_EOS.shape[0])

        m_eos_1 = jnp.where(mask1, masses_EOS, dummy_m)
        y_eos_1 = jnp.where(mask1, y_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, y_eos_1 = m_eos_1[sort_1], y_eos_1[sort_1]
        
        seg1_min = jnp.min(jnp.where(mask1, masses_EOS, SENTINEL))
        seg1_max = jnp.max(jnp.where(mask1, masses_EOS, self.penalty_value))

        m_eos_2 = jnp.where(mask2, masses_EOS, dummy_m)
        y_eos_2 = jnp.where(mask2, y_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, y_eos_2 = m_eos_2[sort_2], y_eos_2[sort_2]
        
        seg2_min = jnp.min(jnp.where(mask2, masses_EOS, SENTINEL))
        seg2_max = jnp.max(jnp.where(mask2, masses_EOS, self.penalty_value))

        def compute_log_integral_segment(m_eos_safe, y_eos_safe, seg_min, seg_max):
            m_eos_safe = m_eos_safe + jnp.arange(m_eos_safe.shape[0]) * 1e-12
            mass_std = self.omegas[:, 0]
            mass_center = self.centers[:, 0]
            window = self.integration_sigma_cut * mass_std

            m_start = jnp.maximum(seg_min, mass_center - window)
            m_stop = jnp.minimum(seg_max, mass_center + window)
            has_support = m_stop > m_start

            t_grid = jnp.linspace(0.0, 1.0, self.N_masses_evaluation)
            m_grid = m_start[:, None] + (m_stop - m_start)[:, None] * t_grid[None, :]
            y_grid = jnp.interp(m_grid, m_eos_safe, y_eos_safe)
            
            xy_points = jnp.stack([m_grid, y_grid], axis=-1)
            diff = xy_points - self.centers[:, None, :]
            diff_transformed = jnp.einsum("kij,knj->kni", self.inv_covs, diff)
            quad_form = jnp.sum(diff * diff_transformed, axis=-1)
            
            log_norm = -0.5 * (self.log_det_covs[:, None] + quad_form + 2 * jnp.log(2 * jnp.pi))
            skew_arg = jnp.sum(self.alpha_primes[:, None, :] * diff, axis=-1)
            log_skew = jnp.log(2.0) + norm.logcdf(skew_arg)
            
            log_prob = log_norm + log_skew
            
            in_segment = (m_grid >= seg_min) & (m_grid <= seg_max) & has_support[:, None]
            log_prob = jnp.where(in_segment, log_prob, self.penalty_value)
            penalty = jnp.where(m_grid > mtov, self.penalty_value, 0.0)
            log_prob = log_prob + penalty

            dm = (m_stop - m_start) / (self.N_masses_evaluation - 1)
            weights = jnp.ones(self.N_masses_evaluation)
            weights = weights.at[0].set(0.5)
            weights = weights.at[-1].set(0.5)
            log_weights = jnp.log(dm[:, None]) + jnp.log(weights[None, :])
            log_weights = jnp.where(has_support[:, None], log_weights, self.penalty_value)
            return logsumexp(log_prob + log_weights, axis=1)

        logL_seg1 = compute_log_integral_segment(m_eos_1, y_eos_1, seg1_min, seg1_max)
        logL_seg2 = compute_log_integral_segment(m_eos_2, y_eos_2, seg2_min, seg2_max)
        logL_individuals = jnp.logaddexp(logL_seg1, logL_seg2)
        
        total_log_likelihood = jnp.sum(logL_individuals)

        return total_log_likelihood