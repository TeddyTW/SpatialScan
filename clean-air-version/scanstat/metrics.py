"""Contains all functionality required to calculate the likelihood ratio F(S)
of a space-time region S as per Expectation-Based Scan Statistic paper. The
ratio is computed for all space-time regions; if significantly larger than
expected, randomisation testing is used to infer the statistical signifiance
of the event"""

import numpy as np


def likelihood_ratio_ebp(baseline_count: float, actual_count: float) -> float:
    """Simple Expression for the likelihood ratio
    Args:
        baseline_count: Sum of baseline counts in space-time region S
        actual_count: Sum of actual counts in space-time region S
    Returns:
        likelihood ratio score
    """
    if baseline_count == 0:
        return 1.0
    if actual_count > baseline_count:
        return np.power((actual_count / baseline_count), actual_count) * np.exp(
            baseline_count - actual_count
        )
    return 1.0
