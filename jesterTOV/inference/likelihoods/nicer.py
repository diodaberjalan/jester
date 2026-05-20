r"""
NICER X-ray timing likelihood implementations

This module provides two implementations:
1. NICERLikelihood - Flow-based (NEW DEFAULT, more efficient)
2. NICERKDELikelihood - KDE-based (legacy, for backward compatibility)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde, norm
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger
import jesterTOV.utils as utils

from jesterTOV.inference.flows.flow import Flow

logger = get_logger("jester")


class NICERLikelihood(LikelihoodBase):
    """
    NICER likelihood using normalizing flows (NEW DEFAULT).

    This is the recommended NICER likelihood implementation that uses
    pre-trained normalizing flows on M-R posteriors for efficient and
    deterministic likelihood evaluation.

    For the legacy KDE-based version, see NICERKDELikelihood.

    The likelihood loads pre-trained flow models for one or both of the Amsterdam
    and Maryland analysis groups, and evaluates the likelihood by:
    1. Pre-sampling masses ONCE at initialization (deterministic with seed)
    2. During evaluation: interpolating radius from the EOS for pre-sampled masses
    3. Evaluating the flow log probability at (mass, radius)
    4. Averaging over all samples, then averaging over available groups

    At least one of ``amsterdam_model_dir`` or ``maryland_model_dir`` must be provided.
    If only one group is provided, the likelihood uses only that group.

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740", "J0437", "J0614")
    amsterdam_model_dir : str | None
        Path to directory containing Amsterdam flow model
        (flow_weights.eqx, metadata.json, flow_kwargs.json).
        If None, Amsterdam group is omitted.
    maryland_model_dir : str | None
        Path to directory containing Maryland flow model.
        If None, Maryland group is omitted.
    penalty_value : float, optional
        Penalty value for samples where mass exceeds Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)
    seed : int, optional
        Random seed for pre-sampling masses (default: 42)

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    seed : int
        Random seed for deterministic pre-sampling
    amsterdam_flow : Flow | None
        Normalizing flow for Amsterdam M-R posterior, or None if not provided
    maryland_flow : Flow | None
        Normalizing flow for Maryland M-R posterior, or None if not provided
    amsterdam_fixed_mass_samples : Float[Array, "n_samples"] | None
        Pre-sampled mass values from Amsterdam flow (fixed at initialization), or None
    maryland_fixed_mass_samples : Float[Array, "n_samples"] | None
        Pre-sampled mass values from Maryland flow (fixed at initialization), or None
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    seed: int

    def __init__(
        self,
        psr_name: str,
        amsterdam_model_dir: str | None = None,
        maryland_model_dir: str | None = None,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size
        self.seed = seed

        if amsterdam_model_dir is None and maryland_model_dir is None:
            raise ValueError(
                f"At least one of amsterdam_model_dir or maryland_model_dir must be "
                f"provided for {psr_name}."
            )

        key = jax.random.key(seed)
        key_amsterdam, key_maryland = jax.random.split(key)

        if amsterdam_model_dir is not None:
            self.amsterdam_flow, self.amsterdam_fixed_mass_samples = (
                self._load_flow_and_presample(
                    amsterdam_model_dir, key_amsterdam, "Amsterdam"
                )
            )
            print(
                f"Amsterdam flow loaded for {psr_name}. Pre-sampled mass range: "
                f"[{jnp.min(self.amsterdam_fixed_mass_samples):.3f}, "
                f"{jnp.max(self.amsterdam_fixed_mass_samples):.3f}] Msun"
            )
        else:
            self.amsterdam_flow = None
            self.amsterdam_fixed_mass_samples = None

        if maryland_model_dir is not None:
            self.maryland_flow, self.maryland_fixed_mass_samples = (
                self._load_flow_and_presample(
                    maryland_model_dir, key_maryland, "Maryland"
                )
            )
            print(
                f"Maryland flow loaded for {psr_name}. Pre-sampled mass range: "
                f"[{jnp.min(self.maryland_fixed_mass_samples):.3f}, "
                f"{jnp.max(self.maryland_fixed_mass_samples):.3f}] Msun"
            )
        else:
            self.maryland_flow = None
            self.maryland_fixed_mass_samples = None

        self.active_groups: list[tuple[Flow, Float[Array, "n_samples"]]] = [
            (flow, samples)
            for flow, samples in [
                (self.amsterdam_flow, self.amsterdam_fixed_mass_samples),
                (self.maryland_flow, self.maryland_fixed_mass_samples),
            ]
            if flow is not None and samples is not None
        ]

        logger.info(
            f"Loaded {len(self.active_groups)} normalizing flow(s) for {psr_name}"
        )

    def _load_flow_and_presample(
        self,
        model_dir: str,
        key: Array,
        group_name: str,
    ) -> tuple[Flow, Float[Array, "n_samples"]]:
        from jesterTOV.inference.flows.flow import Flow

        logger.info(f"Loading {group_name} flow for {self.psr_name} from {model_dir}")
        flow = Flow.from_directory(model_dir)
        mass_samples: Float[Array, "n_samples"] = flow.sample(
            key, (self.N_masses_evaluation,)
        )[:, 0]
        logger.info(
            f"Pre-sampled {group_name} mass range: "
            f"[{jnp.min(mass_samples):.3f}, {jnp.max(mass_samples):.3f}] Msun"
        )
        return flow, mass_samples

    def _get_preset_model_path(self, psr_name: str, group: str) -> str:
        """
        Get preset model path for a pulsar and analysis group.

        Parameters
        ----------
        psr_name : str
            Pulsar name (e.g., "J0030", "J0740")
        group : str
            Analysis group ("amsterdam" or "maryland")

        Returns
        -------
        str
            Path to preset model directory

        Raises
        ------
        ValueError
            If no preset exists for this pulsar/group combination
        """
        # TODO: Define preset paths once NICER flow models are trained
        # For now, this is a placeholder that will be updated in Phase 3

        # Example preset structure (to be implemented):
        # base_dir = Path(__file__).parent.parent / "flows" / "models" / "nicer_maf"
        # model_dir = base_dir / psr_name / f"{psr_name}_{group}_NICER_model"

        raise NotImplementedError(
            f"Preset model paths for {psr_name} {group} not yet implemented. "
            "Please provide explicit model_dir paths or train NICER flows first "
            "(see TODO_FLOW_TRAINING.md Phase 3)."
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters.
        Handles two segments in case of phase transition.
        Segments defined by detecting sudden jump in data gap along pc.
        Uses pre-sampled masses from initialization (deterministic evaluation).

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Phase transition splitting
        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx
        SENTINEL = 1e30

        # Segment 1
        m_eos_1 = jnp.where(mask1, masses_EOS, SENTINEL)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == SENTINEL, self.penalty_value, m_eos_1))

        # Segment 2
        m_eos_2 = jnp.where(mask2, masses_EOS, SENTINEL)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == SENTINEL, self.penalty_value, m_eos_2))

        def compute_group_logL(
            flow: Flow, mass_samples: Float[Array, "n_samples"]
        ) -> Float:
            def process_sample(
                mass: Float,
                m_eos: Float[Array, " n_points"],
                r_eos: Float[Array, " n_points"],
                seg_min: Float,
                seg_max: Float,
            ) -> Float:
                radius = jnp.interp(mass, m_eos, r_eos)
                mr_point = jnp.array([[mass, radius]])  # Shape: (1, 2)
                logpdf = flow.log_prob(mr_point)
                # Zero probability mask for extrapolated points
                in_segment = (mass >= seg_min) & (mass <= seg_max)
                logpdf = jnp.where(in_segment, logpdf, self.penalty_value)
                penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)
                return logpdf + penalty

            # Evaluate on both segments
            logprobs_1 = jax.lax.map(
                lambda m: process_sample(m, m_eos_1, r_eos_1, seg1_min, seg1_max),
                mass_samples,
                batch_size=self.N_masses_batch_size,
            )
            logprobs_2 = jax.lax.map(
                lambda m: process_sample(m, m_eos_2, r_eos_2, seg2_min, seg2_max),
                mass_samples,
                batch_size=self.N_masses_batch_size,
            )
            # Combine segments via logaddexp (disjoint domains handled naturally)
            logprobs = jnp.logaddexp(logprobs_1, logprobs_2)
            return logsumexp(logprobs) - jnp.log(logprobs.shape[0])

        group_logLs = jnp.stack(
            [compute_group_logL(flow, samples) for flow, samples in self.active_groups]
        )
        return logsumexp(group_logLs) - jnp.log(float(group_logLs.shape[0]))


class NICERKDELikelihood(LikelihoodBase):
    """
    NICER likelihood using KDE (Kernel Density Estimation) approach.

    This is the original NICER likelihood implementation that uses KDE
    on M-R posterior samples. For the flow-based version, see NICERLikelihood.

    TODO: Generalize to e.g. only one group, weights between different hotspot models,...

    This likelihood loads posterior samples from Amsterdam and Maryland groups,
    constructs KDEs, and evaluates the likelihood by:
    1. Sampling masses from the NICER posterior samples
    2. Interpolating radius from the EOS for those masses
    3. Evaluating the KDE log probability at (mass, radius)
    4. Averaging over all samples

    Parameters
    ----------
    psr_name : str
        Pulsar name (e.g., "J0030", "J0740")
    amsterdam_samples_file : str
        Path to npz file with Amsterdam group posterior samples
        Expected to contain 'mass' (Msun) and 'radius' (km) arrays
    maryland_samples_file : str
        Path to npz file with Maryland group posterior samples
        Expected to contain 'mass' (Msun) and 'radius' (km) arrays
    penalty_value : float, optional
        Penalty value for samples where mass exceeds Mtov (default: -99999.0)
    N_masses_evaluation : int, optional
        Number of mass samples per likelihood evaluation (default: 20)
    N_masses_batch_size : int, optional
        Batch size for processing mass samples (default: 10)

    Attributes
    ----------
    psr_name : str
        Pulsar name
    penalty_value : float
        Penalty value for samples where mass exceeds Mtov
    N_masses_evaluation : int
        Number of mass samples per likelihood evaluation
    N_masses_batch_size : int
        Batch size for processing mass samples
    amsterdam_masses : Float[Array, " n_amsterdam"]
        Mass samples from Amsterdam group
    maryland_masses : Float[Array, " n_maryland"]
        Mass samples from Maryland group
    amsterdam_posterior : gaussian_kde
        KDE of Amsterdam (mass, radius) posterior
    maryland_posterior : gaussian_kde
        KDE of Maryland (mass, radius) posterior
    """

    psr_name: str
    penalty_value: float
    N_masses_evaluation: int
    N_masses_batch_size: int
    amsterdam_masses: Float[Array, " n_amsterdam"]
    maryland_masses: Float[Array, " n_maryland"]
    amsterdam_posterior: gaussian_kde
    maryland_posterior: gaussian_kde

    def __init__(
        self,
        psr_name: str,
        amsterdam_samples_file: str,
        maryland_samples_file: str,
        penalty_value: float = -99999.0,
        N_masses_evaluation: int = 20,
        N_masses_batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.psr_name = psr_name
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.N_masses_batch_size = N_masses_batch_size

        # Load samples from npz files
        logger.info(
            f"Loading Amsterdam samples for {psr_name} from {amsterdam_samples_file}"
        )
        amsterdam_data = np.load(amsterdam_samples_file, allow_pickle=True)

        logger.info(
            f"Loading Maryland samples for {psr_name} from {maryland_samples_file}"
        )
        maryland_data = np.load(maryland_samples_file, allow_pickle=True)

        # Extract mass and radius samples
        # File format: mass (Msun), radius (km)
        amsterdam_mass = amsterdam_data["mass"]
        amsterdam_radius = amsterdam_data["radius"]
        maryland_mass = maryland_data["mass"]
        maryland_radius = maryland_data["radius"]

        # Store mass samples as JAX arrays for random sampling
        self.amsterdam_masses = jnp.array(amsterdam_mass)
        self.maryland_masses = jnp.array(maryland_mass)

        # Stack into [mass, radius] arrays for KDE
        # Convert to JAX arrays for JAX KDE
        amsterdam_mr = jnp.vstack([amsterdam_mass, amsterdam_radius])
        maryland_mr = jnp.vstack([maryland_mass, maryland_radius])

        # Construct KDEs using JAX implementation
        logger.info(f"Constructing JAX KDEs for {psr_name}")
        self.amsterdam_posterior = gaussian_kde(amsterdam_mr)
        self.maryland_posterior = gaussian_kde(maryland_mr)
        logger.info(f"Loaded JAX KDEs for {psr_name}")

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate log likelihood for given EOS parameters.
        Handles two segments in case of phase transition.
        Segments defined by detecting sudden jump in data gap along pc.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - '_random_key': Random seed for mass sampling (cast to int64)
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'radii_EOS': Array of neutron star radii from EOS

        Returns
        -------
        Float
            Log likelihood value for this NICER observation
        """
        # Extract parameters
        sampled_key = params["_random_key"].astype("int64")
        key = jax.random.key(sampled_key)
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        radii_EOS: Float[Array, " n_points"] = params["radii_EOS"]
        mtov: Float = jnp.max(masses_EOS)

        # Phase transition splitting
        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx
        SENTINEL = 1e30

        # Segment 1
        m_eos_1 = jnp.where(mask1, masses_EOS, SENTINEL)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == SENTINEL, self.penalty_value, m_eos_1))

        # Segment 2
        m_eos_2 = jnp.where(mask2, masses_EOS, SENTINEL)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == SENTINEL, self.penalty_value, m_eos_2))

        # Split key for Amsterdam and Maryland sampling
        key_amsterdam, key_maryland = jax.random.split(key)

        # Sample masses from the NICER posterior samples
        # Each group gets half of N_masses_evaluation samples
        n_samples_per_group: int = self.N_masses_evaluation // 2

        # Sample indices and get mass samples
        amsterdam_indices = jax.random.choice(
            key_amsterdam,
            len(self.amsterdam_masses),
            shape=(n_samples_per_group,),
            replace=True,
        )
        maryland_indices = jax.random.choice(
            key_maryland,
            len(self.maryland_masses),
            shape=(n_samples_per_group,),
            replace=True,
        )

        amsterdam_mass_samples: Float[Array, " n_amsterdam_samples"] = (
            self.amsterdam_masses[amsterdam_indices]
        )
        maryland_mass_samples: Float[Array, " n_maryland_samples"] = (
            self.maryland_masses[maryland_indices]
        )

        def compute_group_logL(
            posterior_kde: gaussian_kde, mass_samples: Float[Array, "n_samples"]
        ) -> Float:
            def process_sample(
                mass: Float,
                m_eos: Float[Array, " n_points"],
                r_eos: Float[Array, " n_points"],
                seg_min: Float,
                seg_max: Float,
            ) -> Float:
                radius = jnp.interp(mass, m_eos, r_eos)
                mr_point = jnp.array([[mass], [radius]])  # Shape: (2, 1)
                logpdf = posterior_kde.logpdf(mr_point)
                # Zero probability mask for extrapolated points
                in_segment = (mass >= seg_min) & (mass <= seg_max)
                logpdf = jnp.where(in_segment, logpdf, self.penalty_value)
                penalty = jnp.where(mass > mtov, self.penalty_value, 0.0)
                return logpdf + penalty

            # Evaluate on both segments
            logprobs_1 = jax.lax.map(
                lambda m: process_sample(m, m_eos_1, r_eos_1, seg1_min, seg1_max),
                mass_samples,
                batch_size=self.N_masses_batch_size,
            )
            logprobs_2 = jax.lax.map(
                lambda m: process_sample(m, m_eos_2, r_eos_2, seg2_min, seg2_max),
                mass_samples,
                batch_size=self.N_masses_batch_size,
            )
            # Combine segments via logaddexp
            logprobs = jnp.logaddexp(logprobs_1, logprobs_2)
            return logsumexp(logprobs) - jnp.log(logprobs.shape[0])

        logL_amsterdam = compute_group_logL(
            self.amsterdam_posterior, amsterdam_mass_samples
        )
        logL_maryland = compute_group_logL(
            self.maryland_posterior, maryland_mass_samples
        )

        return logsumexp(jnp.array([logL_amsterdam, logL_maryland])) - jnp.log(2.0)


class MockMRLikelihood(LikelihoodBase):
    def __init__(
        self,
        csv_file: str,
        penalty_value: float = -1e10,
        N_masses_evaluation: int = 200,
        center_col: tuple = ("Mass_Center_Noise", "Radius_Center_Noise"),
        std_col: tuple = ("Std_Mass", "Std_Radius"),
        skew_col: tuple = ("Skew_Mass", "Skew_Radius"),
        y_key: str = "radii_EOS",
    ) -> None:
        super().__init__()
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation
        self.y_key = y_key

        df = pd.read_csv(csv_file)
        self.K = len(df)

        self.centers = jnp.array(df[list(center_col)].values)
        std_m = jnp.array(df[std_col[0]].values)
        std_y = jnp.array(df[std_col[1]].values)  
        cov_val = jnp.array(df["Covariance"].values)
        self.skews = jnp.array(df[list(skew_col)].values)

        covs = np.zeros((self.K, 2, 2))
        covs[:, 0, 0] = std_m**2 + 1e-12
        covs[:, 1, 1] = std_y**2 + 1e-12
        covs[:, 0, 1] = cov_val
        covs[:, 1, 0] = cov_val
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

        m_eos_1 = jnp.where(mask1, masses_EOS, SENTINEL)
        y_eos_1 = jnp.where(mask1, y_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, y_eos_1 = m_eos_1[sort_1], y_eos_1[sort_1]
        
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == SENTINEL, self.penalty_value, m_eos_1))

        m_eos_2 = jnp.where(mask2, masses_EOS, SENTINEL)
        y_eos_2 = jnp.where(mask2, y_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, y_eos_2 = m_eos_2[sort_2], y_eos_2[sort_2]
        
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == SENTINEL, self.penalty_value, m_eos_2))

        m_grid = jnp.linspace(0.1, 3.5, self.N_masses_evaluation)
        dm = m_grid[1] - m_grid[0]

        def compute_log_prob_segment(m_eos_safe, y_eos_safe, seg_min, seg_max):
            m_eos_safe = m_eos_safe + jnp.arange(m_eos_safe.shape[0]) * 1e-12
            y_grid = jnp.interp(m_grid, m_eos_safe, y_eos_safe)
            
            xy_points = jnp.stack([m_grid, y_grid], axis=-1)
            diff = xy_points[None, :, :] - self.centers[:, None, :]
            diff_transformed = jnp.einsum("kij,knj->kni", self.inv_covs, diff)
            quad_form = jnp.sum(diff * diff_transformed, axis=-1)
            
            log_norm = -0.5 * (self.log_det_covs[:, None] + quad_form + 2 * jnp.log(2 * jnp.pi))
            skew_arg = jnp.sum(self.alpha_primes[:, None, :] * diff, axis=-1)
            log_skew = jnp.log(2.0) + norm.logcdf(skew_arg)
            
            log_prob = log_norm + log_skew
            
            in_segment = (m_grid >= seg_min) & (m_grid <= seg_max)
            log_prob = jnp.where(in_segment[None, :], log_prob, self.penalty_value)
            penalty = jnp.where(m_grid > mtov, self.penalty_value, 0.0)
            log_prob = log_prob + penalty[None, :]
            return log_prob

        log_prob_seg1 = compute_log_prob_segment(m_eos_1, y_eos_1, seg1_min, seg1_max)
        log_prob_seg2 = compute_log_prob_segment(m_eos_2, y_eos_2, seg2_min, seg2_max)
        
        log_prob_combined = jnp.logaddexp(log_prob_seg1, log_prob_seg2)
        logL_individuals = logsumexp(log_prob_combined, axis=1) + jnp.log(dm)
        
        total_log_likelihood = jnp.sum(logL_individuals)

        return total_log_likelihood

        
class MockMRLikelihood_old(LikelihoodBase):
    """
    Mock MR Likelihood evaluating deterministic skewed correlated posteriors.
    Integrates probability density over a mass grid and normalizes by K samples correctly in log-space.
    """

    penalty_value: float
    N_masses_evaluation: int
    K: int
    centers: Float[Array, "K 2"]
    covs: Float[Array, "K 2 2"]
    inv_covs: Float[Array, "K 2 2"]
    log_det_covs: Float[Array, "K"]
    skews: Float[Array, "K 2"]
    omegas: Float[Array, "K 2"]
    alpha_primes: Float[Array, "K 2"]

    def __init__(
        self,
        csv_file: str,
        penalty_value: float = -1e10,
        N_masses_evaluation: int = 200,
    ) -> None:
        super().__init__()
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation

        # Load data without pandas (using numpy for host-side I/O before JAX conversion)
        data = np.genfromtxt(csv_file, delimiter=",", names=True)
        self.K = len(data)

        # Parse necessary information
        self.centers = jnp.stack(
            [data["Mass_Center_Noise"], data["Radius_Center_Noise"]], axis=-1
        )
        std_m = jnp.array(data["Std_Mass"])
        std_r = jnp.array(data["Std_Radius"])
        cov_val = jnp.array(data["Covariance"])
        self.skews = jnp.stack([data["Skew_Mass"], data["Skew_Radius"]], axis=-1)

        # Build Covariance matrices (K, 2, 2)
        covs = np.zeros((self.K, 2, 2))
        covs[:, 0, 0] = std_m**2
        covs[:, 1, 1] = std_r**2
        covs[:, 0, 1] = cov_val
        covs[:, 1, 0] = cov_val
        self.covs = jnp.array(covs)

        # Precompute heavy matrix operations for the flow
        self.inv_covs = jnp.linalg.inv(self.covs)
        self.log_det_covs = jnp.linalg.slogdet(self.covs)[1]

        # Precompute skew modifiers
        self.omegas = jnp.sqrt(jnp.diagonal(self.covs, axis1=1, axis2=2))
        self.alpha_primes = self.skews / self.omegas

    def evaluate(self, params: dict) -> Float:
        masses_EOS = params["masses_EOS"]
        radii_EOS = params["radii_EOS"]
        mtov = jnp.max(masses_EOS)

        # Phase transition splitting logic
        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx

        # Segment 1 setup
        m_eos_1 = jnp.where(mask1, masses_EOS, jnp.inf)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)  # type: ignore[arg-type]
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == jnp.inf, self.penalty_value, m_eos_1))

        # Segment 2 setup
        m_eos_2 = jnp.where(mask2, masses_EOS, jnp.inf)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)  # type: ignore[arg-type]
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == jnp.inf, self.penalty_value, m_eos_2))

        # Uniform mass grid for deterministic numerical integration
        m_grid = jnp.linspace(0.1, 3.5, self.N_masses_evaluation)
        dm = m_grid[1] - m_grid[0]

        def compute_log_prob_segment(m_eos, r_eos, seg_min, seg_max):
            # Interpolate radius for the entire grid
            r_grid = jnp.interp(m_grid, m_eos, r_eos)
            mr_points = jnp.stack([m_grid, r_grid], axis=-1)  # (N, 2)

            # Vectorized multivariate evaluation over K samples and N points
            # diff shape: (K, N, 2)
            diff = mr_points[None, :, :] - self.centers[:, None, :]

            # Quadratic form via Einstein summation: (K, N)
            diff_transformed = jnp.einsum("kij,knj->kni", self.inv_covs, diff)
            quad_form = jnp.sum(diff * diff_transformed, axis=-1)

            # Normal part
            log_norm = -0.5 * (
                self.log_det_covs[:, None] + quad_form + 2 * jnp.log(2 * jnp.pi)
            )

            # Skewness part
            skew_arg = jnp.sum(self.alpha_primes[:, None, :] * diff, axis=-1)
            log_skew = jnp.log(2.0) + norm.logcdf(skew_arg)

            log_prob = log_norm + log_skew

            # Discard out-of-segment points
            in_segment = (m_grid >= seg_min) & (m_grid <= seg_max)
            log_prob = jnp.where(in_segment[None, :], log_prob, self.penalty_value)

            # Apply TOV limit penalty
            penalty = jnp.where(m_grid > mtov, self.penalty_value, 0.0)
            log_prob = log_prob + penalty[None, :]

            return log_prob

        # Evaluate both segments
        log_prob_seg1 = compute_log_prob_segment(m_eos_1, r_eos_1, seg1_min, seg1_max)
        log_prob_seg2 = compute_log_prob_segment(m_eos_2, r_eos_2, seg2_min, seg2_max)

        # Recombine segments (Log addition handles disjoint domains naturally)
        log_prob_combined = jnp.logaddexp(log_prob_seg1, log_prob_seg2)

        # Marginalize over M (Numerical integration: sum(P * dm) -> logsumexp + log(dm))
        logL_individuals = logsumexp(log_prob_combined, axis=1) + jnp.log(dm)

        # Normalize by taking the log of the arithmetic mean of the likelihoods
        total_log_likelihood = logsumexp(logL_individuals) - jnp.log(self.K)

        return total_log_likelihood
