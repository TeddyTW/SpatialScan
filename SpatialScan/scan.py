"""Module to contain the main Spatio-Temporal Scan Loop"""

import time
import pandas as pd
import numpy as np
from SpatialScan.region import (
    Region,
    event_count,
    infer_global_region,
    make_grid,
    simulate_event_count,
    aggregate_event_data,
)
from SpatialScan.likelihood import (
    likelihood_ratio,
    likelihood_ratio_kulgen,
    bbayes_lhood_H0,
    bbayes_lhood_H1,
)


def scan(
    forecast_data: pd.DataFrame,
    grid_partition: int,
    scan_type: str = "normal",
    bayes_no_outbreak_prior=0.99,
) -> pd.DataFrame:

    """Main function for looping through the sub-space-time regions (S) of
    global_region represented by data in forecast_data. We search for regions
    with the highest score according to various different metrics. The EBP one is
    given by:
                F(S) := Pr (data | H_1 (S)) / Pr (data | H_0)
    where H_0 and H_1 (S) are defined in the Expectation-Based Scan Statistic
    paper by D. Neill. The second is given in the "Detecting Significant Multidimensional
    Spatial Clusters" paper by the same author. It is a generalisation of Kulldorf's
    statistic that checks C and B counts inside AND outside the region of interest.

    Args:
        forecast_data: dataframe consisting of the detectors which lie in
                       global_region, their locations and both their
                       baseline and actual counts for the past W days.
        grid_partition: Split each spatial axis into this many partitions.
        scan_type: "normal" or 'exhaustive'. The former searches over all space-time
                   regions with t_max fixed at the current day. A search which asks
                   'which clustered events are likely to be effecting me now?'.
                   The latter searches over ALL possible t_min and t_max spanning
                   the past day. 'Which events were clustered in the past?'
        bayes_no_outbreak_prior: Prior probability of the no-outbreak hypothesis
                                 being true. Here, set uniform.
    Returns:
        Dataframe summarising each space-time region's scores for 5 metrics.
        The first is the basic summary stat described above. The next 3
        are Kulldorf's generalised statistic with epsilon = 0.0, 0.25, 0.5
        The remaining is a Blind Bayes (BBayes) posterior probability estimate.
    """

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_data)

    # Create the Grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    # Aggregate the Data to cell level
    print("Aggregating data from detector level to grid level.")
    agg_df = aggregate_event_data(forecast_data, x_ticks, y_ticks, t_ticks)

    B_tot = agg_df["baseline_agg"].sum() / 1e6
    C_tot = agg_df["count_agg"].sum() / 1e6

    # Compute Blind Bayes null likelihood
    bbayes_null_lhood = bbayes_lhood_H0(B_tot, C_tot)

    # Set Intermediate Timer
    t2 = time.perf_counter()

    # Time direction convention - reverse
    t_ticks = t_ticks[::-1]

    # Set-up scan with initial variables
    print("Beginning {} scan. Setup Time: {:.2f} seconds".format(scan_type, t2 - t1))
    num_regions = 0
    scores_dict = {}

    # Loop over all possible prisms of max size N/2 in the space
    for t in range(len(t_ticks) if scan_type == "exhaustive" else 1):  # t_max
        for s in range(t + 1, len(t_ticks)):  # t_min
            for i, _ in enumerate(x_ticks):  # x_min
                for j in range(
                    i + 1, np.min([i + (int(grid_partition / 2)), grid_partition]) + 1
                ):  # x_max
                    for k, _ in enumerate(y_ticks):  # y_min
                        for l in range(
                            k + 1,
                            np.min([k + (int(grid_partition / 2)), grid_partition]) + 1,
                        ):  # y_max

                            # At each iteration, create the space_time region
                            test_region = Region(
                                x_ticks[i],
                                x_ticks[j],
                                y_ticks[k],
                                y_ticks[l],
                                t_ticks[s],  # t_min - always changing
                                t_ticks[t],  # t_max - sometimes fixed (t_ticks[0])
                            )

                            # Count the events within the region
                            B, C = event_count(test_region, agg_df)

                            # Compute Metrics
                            ebp_l_score = likelihood_ratio(B, C)  # Normal EBP metric
                            general_l_scores = [
                                likelihood_ratio_kulgen(
                                    B, C, B_tot, C_tot, eps
                                )  # General Kulldorf
                                for eps in [0.0, 0.25, 0.50]
                            ]
                            bbayes_alt_score = bbayes_lhood_H1(
                                B, C, B_tot, C_tot
                            )  # Blind Bayes Metric

                            # Append results
                            scores_dict[num_regions] = {
                                "x_min": x_ticks[i],
                                "x_max": x_ticks[j],
                                "y_min": y_ticks[k],
                                "y_max": y_ticks[l],
                                "t_min": t_ticks[s],
                                "t_max": t_ticks[t],
                                "B_in": B,
                                "C_in": C,
                                "B_out": B_tot - B,
                                "C_out": C_tot - C,
                                "C_in/B_in": (C / B) if B != 0 else np.inf,
                                "C_out/B_out": (C_tot - C) / (B_tot - B)
                                if B_tot != B
                                else np.inf,
                                "l_score_EBP": ebp_l_score,
                                "p_value_EBP": np.nan,
                                "l_score_000": general_l_scores[0],
                                "l_score_025": general_l_scores[1],
                                "l_score_050": general_l_scores[2],
                                "l_score_bbayes": bbayes_alt_score,
                            }

                            # Count Regions
                            num_regions += 1

            # Print Progress
            print(
                "Search spatial regions with t_min = {} and t_max = {}".format(
                    t_ticks[s], t_ticks[t]
                ),
                end="\r",
            )

    region_score_df = pd.DataFrame.from_dict(scores_dict, "index")

    # Now we need to extra work to find the Bayesian scores
    region_score_df["l_score_bbayes"] *= (1 - bayes_no_outbreak_prior) / num_regions

    prob_D = (
        region_score_df["l_score_bbayes"].sum()
        + bbayes_null_lhood * bayes_no_outbreak_prior
    )

    region_score_df["posterior_bbayes"] = region_score_df["l_score_bbayes"] / prob_D

    region_score_df = region_score_df.drop("l_score_bbayes", axis=1)

    print(
        "\nNo Outbreak Posterior: {}".format(
            bbayes_null_lhood * bayes_no_outbreak_prior / prob_D
        )
    )
    print(
        "Total Outbreak Posterior: {}".format(region_score_df["posterior_bbayes"].sum())
    )

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region. Sort it so that highest `l_score_EBP`
    # score is at the top.
    region_score_df = region_score_df.sort_values("l_score_EBP", ascending=False)

    t3 = time.perf_counter()

    print("\n%d space-time regions searched in %.2f seconds" % (num_regions, t3 - t2))
    print("Total run time: {0:.2f} seconds".format(t3 - t1))

    return region_score_df


# TODO - Change this to fast scan overlap-kd tree structure
def randomisation_test(
    forecast_df: pd.DataFrame, res_df: pd.DataFrame, n_sims: int = 100
) -> tuple:

    """Functionality to perform Kulldorf-inspired randomisation testing on the
    results achieved from the main scan function `EBP()`. For each simulation,
    this method models the actual count as a Poisson Random Variable with mean
    equal to the expected counts.
    i.e. Forecasting in `timeseries.py` has found baseline counts b_i^t for each
    spatial location at each tim step of interest. The above `EBP()` method compares
    these with the actual counts c_i^t of a given space-time region S. Here,
    we compare b_i^t and Po(b_i^t) instead. Simulating `n_sims` times,
    allows us to build a a distribution of F(S) scores, and hence estimate
    a p-value for the obtained score in the main scan.

    Args:
        forecast_df: Dataframe outputted from `count_baseline()` method.
        res_df: Data frame containing results from `EBP()` method.
        n_sims: Number of simulations to perform

    Returns:
        Dataframe with a populated `p_value_basic` column.
        np.ndarray of l_score_EPB values from the resulting simulations.
    """

    # Infer grid partition from the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    # Determine whether exhaustive or normal scan from the resulting dataframe
    # What type of scan was it? Normal or Exhaustive?
    # Only way to tell is by the number of unique t_maxs in res_df
    num_t_maxs = len(res_df["t_max"].unique())

    # If more than one t_max, scan was exhaustive
    if num_t_maxs > 1:
        scan_type = "exhaustive"
    else:
        scan_type = "normal"

    print(
        "Found a grid partition = {} and a scan type = {}.".format(
            grid_partition, scan_type
        )
    )

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_df)

    print(
        "Searching over the region spanning {} hours".format(global_region.num_hours())
    )

    # Re-create the grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    # Time direction convention - reverse
    t_ticks = t_ticks[::-1]

    # Array for F(S) scores
    best_EBP_scores = []

    print("\n====================")
    print("Beginning Simulation")
    print("====================")

    # Main Loop
    for sim in range(n_sims):
        max_EBP_score = 0
        print("Performing simulation {} of {}.".format(sim + 1, n_sims), end="\r")
        for t in range(len(t_ticks) if scan_type == "exhaustive" else 1):  # t_max
            for s in range(t + 1, len(t_ticks)):  # t_min
                for i, _ in enumerate(x_ticks):  # x_min
                    for j in range(
                        i + 1,
                        np.min([i + (int(grid_partition / 2)), grid_partition]) + 1,
                    ):  # x_max
                        for k, _ in enumerate(y_ticks):  # y_min
                            for l in range(
                                k + 1,
                                np.min([k + (int(grid_partition / 2)), grid_partition])
                                + 1,
                            ):  # y_max

                                # At each iteration, create the space_time region
                                test_region = Region(
                                    x_ticks[i],
                                    x_ticks[j],
                                    y_ticks[k],
                                    y_ticks[l],
                                    t_ticks[s],  # t_min - always changing
                                    t_ticks[t],  # t_max - sometimes fixed (t_ticks[0])
                                )

                                baseline, simulated = simulate_event_count(
                                    test_region, forecast_df
                                )

                                l_score = likelihood_ratio(baseline, simulated)

                                max_EBP_score = (
                                    l_score
                                    if l_score > max_EBP_score
                                    else max_EBP_score
                                )

        best_EBP_scores.append(max_EBP_score)
    arr = np.array(best_EBP_scores)
    t2 = time.perf_counter()

    print("\nTime Elapsed: {} seconds".format(t2 - t1))

    # Estimate p-value
    res_df["p_value_EBP"] = 1 - res_df["l_score_EBP"].apply(
        lambda x: np.count_nonzero(x > arr) + 1
    ) / (n_sims + 1)
    copy = res_df

    return copy, arr


# TODO
def historical_data_test():
    return None
