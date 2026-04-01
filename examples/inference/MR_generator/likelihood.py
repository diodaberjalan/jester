import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jesterTOV import utils

class SkewedCorrelatedFlow:
    def __init__(self, m_center, r_center, cov_matrix, skewness):
        self.center = jnp.array([m_center, r_center])
        self.cov = jnp.array(cov_matrix)
        self.skew = jnp.array(skewness)
        self.inv_cov = jnp.linalg.inv(self.cov)
        self.log_det_cov = jnp.linalg.slogdet(self.cov)[1]

    def log_prob(self, mr_point):
        diff = mr_point - self.center 
        quad_form = jnp.sum(diff @ self.inv_cov * diff, axis=-1)
        log_norm = -0.5 * (self.log_det_cov + quad_form + 2 * jnp.log(2 * jnp.pi))
        omega = jnp.sqrt(jnp.diag(self.cov))
        alpha_prime = self.skew / omega
        skew_arg = jnp.sum(alpha_prime * diff, axis=-1)
        log_skew = jnp.log(2.0) + jax.scipy.stats.norm.logcdf(skew_arg)
        return log_norm + log_skew

class LikelihoodBase:
    pass

class MockMRLikelihood(LikelihoodBase):
    def __init__(self, csv_file: str, penalty_value: float = -1e10, N_masses_evaluation: int = 200) -> None:
        super().__init__()
        self.penalty_value = penalty_value
        self.N_masses_evaluation = N_masses_evaluation

        df = pd.read_csv(csv_file)
        self.K = len(df)
        
        self.centers = jnp.array(df[["Mass_Center_Noise", "Radius_Center_Noise"]].values)
        std_m = jnp.array(df["Std_Mass"].values)
        std_r = jnp.array(df["Std_Radius"].values)
        cov_val = jnp.array(df["Covariance"].values)
        self.skews = jnp.array(df[["Skew_Mass", "Skew_Radius"]].values)

        covs = np.zeros((self.K, 2, 2))
        covs[:, 0, 0] = std_m**2
        covs[:, 1, 1] = std_r**2
        covs[:, 0, 1] = cov_val
        covs[:, 1, 0] = cov_val
        self.covs = jnp.array(covs)

        self.inv_covs = jnp.linalg.inv(self.covs)
        self.log_det_covs = jnp.linalg.slogdet(self.covs)[1]
        self.omegas = jnp.sqrt(jnp.diagonal(self.covs, axis1=1, axis2=2))
        self.alpha_primes = self.skews / self.omegas

    def evaluate(self, params: dict) -> float:
        masses_EOS = params["masses_EOS"]
        radii_EOS = params["radii_EOS"]
        mtov = jnp.max(masses_EOS)

        split_idx = utils.get_MR_split_index(masses_EOS, radii_EOS)
        idx = jnp.arange(masses_EOS.shape[0])
        mask1 = idx < split_idx
        mask2 = idx >= split_idx

        m_eos_1 = jnp.where(mask1, masses_EOS, jnp.inf)
        r_eos_1 = jnp.where(mask1, radii_EOS, 0.0)
        sort_1 = jnp.argsort(m_eos_1)
        m_eos_1, r_eos_1 = m_eos_1[sort_1], r_eos_1[sort_1]
        seg1_min = m_eos_1[0]
        seg1_max = jnp.max(jnp.where(m_eos_1 == jnp.inf, -jnp.inf, m_eos_1))

        m_eos_2 = jnp.where(mask2, masses_EOS, jnp.inf)
        r_eos_2 = jnp.where(mask2, radii_EOS, 0.0)
        sort_2 = jnp.argsort(m_eos_2)
        m_eos_2, r_eos_2 = m_eos_2[sort_2], r_eos_2[sort_2]
        seg2_min = m_eos_2[0]
        seg2_max = jnp.max(jnp.where(m_eos_2 == jnp.inf, -jnp.inf, m_eos_2))

        m_grid = jnp.linspace(0.1, 3.5, self.N_masses_evaluation)
        dm = m_grid[1] - m_grid[0]

        def compute_log_prob_segment(m_eos, r_eos, seg_min, seg_max):
            r_grid = jnp.interp(m_grid, m_eos, r_eos)
            mr_points = jnp.stack([m_grid, r_grid], axis=-1)
            diff = mr_points[None, :, :] - self.centers[:, None, :]
            diff_transformed = jnp.einsum('kij,knj->kni', self.inv_covs, diff)
            quad_form = jnp.sum(diff * diff_transformed, axis=-1)
            log_norm = -0.5 * (self.log_det_covs[:, None] + quad_form + 2 * jnp.log(2 * jnp.pi))
            skew_arg = jnp.sum(self.alpha_primes[:, None, :] * diff, axis=-1)
            log_skew = jnp.log(2.0) + norm.logcdf(skew_arg)
            log_prob = log_norm + log_skew
            in_segment = (m_grid >= seg_min) & (m_grid <= seg_max)
            log_prob = jnp.where(in_segment[None, :], log_prob, -jnp.inf)
            penalty = jnp.where(m_grid > mtov, self.penalty_value, 0.0)
            log_prob = log_prob + penalty[None, :]
            return log_prob

        log_prob_seg1 = compute_log_prob_segment(m_eos_1, r_eos_1, seg1_min, seg1_max)
        log_prob_seg2 = compute_log_prob_segment(m_eos_2, r_eos_2, seg2_min, seg2_max)
        log_prob_combined = jnp.logaddexp(log_prob_seg1, log_prob_seg2)
        logL_individuals = logsumexp(log_prob_combined, axis=1) + jnp.log(dm)
        total_log_likelihood = logsumexp(logL_individuals) - jnp.log(self.K)

        return total_log_likelihood