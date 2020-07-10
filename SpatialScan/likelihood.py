"""Contains all functionality required to calculate the likelihood ratio F(S)
of a space-time region S as per Expectation-Based Scan Statistic paper. The
ratio is computed for all space-time regions; if significantly larger than
expected, randomisation testing is used to infer the statistical signifiance
of the event"""

import numpy as np
from scipy.special import gamma


def likelihood_ratio(B: float, C: float) -> float:
    """Simple Expression for the likelihood ratio
    Args:
        B: Sum of baseline counts in space-time region S
        C: Sum of actual counts in space-time region S
    Returns:
        float
    """
    if B == 0 and C > 0:
        return 1.0
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
    if B == 0:
        condition = True
    else:
        condition = C / B > (1 + eps) * (C_tot - C) / (B_tot - B)

    sign = 1 if condition else -1

    if B == 0:
        return np.nan
    return sign * (
        C * np.log(C / ((1 + eps) * B))
        + (C_tot - C) * np.log((C_tot - C) / (B_tot - B))
        - C_tot * np.log(C_tot / (B_tot + eps * B))
    )


def bayes_lhood_form(alpha, beta, B, C):
    if B == 0 or C == 0:
        return np.nan
    return (beta ** alpha) * gamma(alpha + C) / (((beta + B)**(alpha + C)) * gamma(alpha))

def bbayes_lhood_H1(B_in, C_in, B_tot, C_tot, num_m_samples=10):

    # Get counts outside a region
    B_out = B_tot - B_in
    C_out = C_tot - C_in

    # Assumes baselines are estimating equality of counts. (Not proprtionality)
    q_0 = 1

    # Set most priors here
    alpha_out = q_0 * B_out
    beta_in = B_in
    beta_out = B_out

    av_score = 0
    for _ in range(num_m_samples):

        # Generate m value between 1 and 3 with increment 0.2
        m = np.random.randint(5, 15) / 5

        # Set the last prior which depends on m
        alpha_in = m * q_0 * B_in

        l_score = bayes_lhood_form(alpha_in, beta_in, B_in, C_in) *\
                  bayes_lhood_form(alpha_out, beta_out, B_out, C_out)

        av_score += (l_score / num_m_samples)

    return av_score


def bbayes_lhood_H0(B_all, C_all):

    q_0 = 1

    # Setting blind priors here
    alpha_all = q_0 * B_all
    beta_all = B_all

    return bayes_lhood_form(alpha_all, beta_all, B_all, C_all)