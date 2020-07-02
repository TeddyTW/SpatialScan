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


def likelihood_ratio_kulgen(B: float, C: float, B_tot: float, C_tot: float, eps: float):
    """Generalised likelihood expression as explained by D. Neill in more
    recent work. Allows for tuning with parameter eps. The eps = 0 case defaults
    to the original test statistic suggested by Kulldorf. Note that the original
    Kulldorf statistic and this one calculate metrics based on C, B inside AND
    outside the test region S. `likelihood_ratio()` does not; it only focuses
    on counts inside the region of interest.
    Args:
        B: Sum of baseline counts in region of interest
        C: Sum of actual counts in region of interest
        B_tot: Sum of baseline counts in global region
        C_tot: Sum of baseline counts in global region
    Returns:
        Generalised Test Statistic D_epsilon (S)
    """

    # First Calculate the Sign
    condition = C / B > (1 + eps) * (C_tot - C) / (B_tot - B)
    sign = 1 if condition else -1

    return sign * (
        C * np.log(C / ((1 + eps) * B))
        + (C_tot - C) * np.log((C_tot - C) / (B_tot - B))
        - C_tot * np.log(C_tot / (B_tot + eps * B))
    )
