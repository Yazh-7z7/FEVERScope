"""Statistical significance testing for IR comparisons.

Follows Single Responsibility Principle (SRP):
Computes paired bootstrap confidence intervals and paired Wilcoxon signed-rank tests
between retrieval runs to verify whether improvements are statistically significant (§6).
"""

import numpy as np
from scipy import stats


def paired_bootstrap_ci(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired bootstrap confidence interval of mean difference (scores_b - scores_a).

    Returns:
        mean_diff: Observed difference in means.
        ci_lower: Lower bound of (1 - alpha) CI.
        ci_upper: Upper bound of (1 - alpha) CI.
    """
    diffs = np.asarray(scores_b) - np.asarray(scores_a)
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    bootstrap_means = np.empty(n_resamples)

    for i in range(n_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(bootstrap_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2.0)))

    return mean_diff, ci_lower, ci_upper


def paired_wilcoxon_test(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
) -> tuple[float, float]:
    """Compute paired Wilcoxon signed-rank test.

    Returns:
        statistic: Wilcoxon test statistic.
        p_value: Two-sided p-value.
    """
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    stat, pval = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(pval)
