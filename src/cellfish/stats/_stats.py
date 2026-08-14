import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import median_abs_deviation


def is_outlier(adata, obs_column: str, nmads: int) -> ArrayLike:
    """Flag values more than ``nmads`` median absolute deviations from the median."""
    m = adata.obs[obs_column].values
    med = np.median(m)
    mad = median_abs_deviation(m)
    return (m < med - nmads * mad) | (med + nmads * mad < m)
