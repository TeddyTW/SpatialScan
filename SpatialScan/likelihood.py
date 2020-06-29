"""Contains all functionality required to calculate the likelihood ratio F(S)
of a space-time region S as per Expectation-Based Scan Statistic paper. The
ratio is computed for all space-time regions; if significantly larger than
expected, randomisation testing is used to infer the statistical signifiance
of the event"""

import numpy as np


def likelihood_ratio(B: float, C: float) -> float:
    """Simple Expression for the likelihood ratio
    Args:
        B: Sum of baseline counts in space-time region S
        C: Sum of actual counts in space-time region S
    Returns:
        float
    """

    if C > B:
        return np.power((C / B), C) * np.exp(B - C)
    return 1.0