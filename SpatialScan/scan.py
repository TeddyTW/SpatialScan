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
from SpatialScan.likelihood import likelihood_ratio, likelihood_ratio_kulgen


def EBP(forecast_data: pd.DataFrame, grid_partition: int) -> pd.DataFrame:

    """Main function for looping through the sub-space-time regions (S) of
    global_region represented by data in forecast_data. We search for regions
    with the highest score according to two different metrics. The basic one is
    given by:
                F(S) := Pr (data | H_1 (S)) / Pr (data | H_0)
    where H_0 and H_1 (S) are defined in the Expectation-Based Scan Statistic
    paper by D. Neill. The second is given in the "Detecting Significant Multidimensional
    Spatial Clusters" paper by the same author. It is a generalisation of Kulldorf's
    statistic that checks C and B counts inside AND outside the region of interest.

    Note that in particular, t_min is fixed here. Varying t_min functionality
    can be found by calling `EBP_exhaustive()`, with downside of extra computation
    time.

    Args:
        forecast_data: dataframe consisting of the detectors which lie in
                       global_region, their locations and both their
                       baseline and actual counts for the past W days.
        grid_partition: Split each spatial axis into this many partitions.
    Returns:
            Dataframe summarising each space-time region's scores for 6 metrics.
            The first is the basic summary stat described above. The other 5
            are Kulldorf's generalised statistic with epsilon = 0.0, 0.25, 0.5, 0.75, 1.00.
    """

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_data)

    # Create the Grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    print("Aggregating data from detector level to grid level.")
    # Aggregate the Data to cell level
    agg_df = aggregate_event_data(forecast_data, x_ticks, y_ticks, t_ticks)

    B_tot = agg_df["baseline_agg"].sum() / 1e6
    C_tot = agg_df["count_agg"].sum() / 1e6

    # Set Intermediate Timer
    t2 = time.perf_counter()

    print("Beginning Scan. Setup Time: {0:.2f} seconds".format(t2 - t1))
    num_regions = 0
    scores_dict = {}
    # Loop over all possible prisms of max size N/2 in the space
    for t in range(1, len(t_ticks)):
        for i, _ in enumerate(x_ticks):
            for j in range(
                i + 1, np.min([i + (int(grid_partition / 2)), grid_partition]) + 1
            ):
                for k, _ in enumerate(y_ticks):
                    for l in range(
                        k + 1,
                        np.min([k + (int(grid_partition / 2)), grid_partition]) + 1,
                    ):

                        # At each iteration, create the space_time region
                        test_region = Region(
                            x_ticks[i],
                            x_ticks[j],
                            y_ticks[k],
                            y_ticks[l],
                            t_ticks[0],  # t_min fixed here
                            t_ticks[t],
                        )

                        # Count the events within the region
                        B, C = event_count(test_region, agg_df)

                        # Compute Metrics
                        basic_l_score = likelihood_ratio(B, C)
                        general_l_scores = [
                            likelihood_ratio_kulgen(B, C, B_tot, C_tot, eps)
                            for eps in [0.0, 0.25, 0.50, 0.75, 1.00]
                        ]

                        # Append results
                        scores_dict[num_regions] = {
                            "x_min": x_ticks[i],
                            "x_max": x_ticks[j],
                            "y_min": y_ticks[k],
                            "y_max": y_ticks[l],
                            "t_min": t_ticks[0],
                            "t_max": t_ticks[t],
                            "B_in": B,
                            "C_in": C,
                            "B_out": B_tot - B,
                            "C_out": C_tot - C,
                            "C/B_in": (C / B) if B != 0 else np.inf,
                            "C/B_out": (C_tot - C) / (B_tot - B)
                            if B_tot != B
                            else np.inf,
                            "l_score_basic": basic_l_score,
                            #"p_value_basic": np.nan,
                            "l_score_000": general_l_scores[0],
                            #"p_value_000": np.nan,
                            "l_score_025": general_l_scores[1],
                            #"p_value_025": np.nan,
                            "l_score_050": general_l_scores[2],
                            #"p_value_050": np.nan,
                            "l_score_075": general_l_scores[3],
                            #"p_value_075": np.nan,
                            "l_score_100": general_l_scores[4],
                            #"p_value_100": np.nan,
                        }

                        # Count Regions
                        num_regions += 1

        # Print Progress
        print("{0:.2f}% complete.".format((t + 1) * 100 / len(t_ticks)), end="\r")

    region_score_df = pd.DataFrame.from_dict(scores_dict, "index")

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region.
    # Sort it so that highest F(S) score is at the top.
    region_score_df = region_score_df.sort_values("l_score_basic", ascending=False)

    t3 = time.perf_counter()

    print("\n%d space-time regions searched in %.2f seconds" % (num_regions, t3 - t2))
    print("Total run time: {0:.2f} seconds".format(t3 - t1))

    return region_score_df


def EBP_exhaustive(forecast_data: pd.DataFrame, grid_partition: int) -> pd.DataFrame:

    """Identical to `EBP()` with the difference of searching through all possible
    values of t_min. Allows us to narrow down clusters better in the temporal domain.

    Args:
        forecast_data: dataframe consisting of the detectors which lie in
                       global_region, their locations and both their
                       baseline and actual counts for the past W days.
        grid_partition: Split each spatial axis into this many partitions.
    Returns:
            Dataframe summarising each space-time region's scores for 6 metrics.
            The first is the basic summary stat described above. The other 5
            are Kulldorf's generalised statistic with epsilon = 0.0, 0.25, 0.5, 0.75, 1.00.
    """

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_data)

    # Create the Grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    print("Aggregating data from detector level to grid level.")
    # Aggregate the Data to cell level
    agg_df = aggregate_event_data(forecast_data, x_ticks, y_ticks, t_ticks)

    B_tot = agg_df["baseline_agg"].sum() / 1e6
    C_tot = agg_df["count_agg"].sum() / 1e6

    # Set Intermediate Timer
    t2 = time.perf_counter()

    print("Beginning Scan. Setup Time: {0:.2f} seconds".format(t2 - t1))
    num_regions = 0
    scores_dict = {}
    # Loop over all possible prisms of max size N/2 in the space
    for s, _ in enumerate(t_ticks):
        for t in range(s + 1, len(t_ticks)):
            for i, _ in enumerate(x_ticks):
                for j in range(
                    i + 1, np.min([i + (int(grid_partition / 2)), grid_partition]) + 1
                ):
                    for k, _ in enumerate(y_ticks):
                        for l in range(
                            k + 1,
                            np.min([k + (int(grid_partition / 2)), grid_partition]) + 1,
                        ):

                            # At each iteration, create the space_time region
                            test_region = Region(
                                x_ticks[i],
                                x_ticks[j],
                                y_ticks[k],
                                y_ticks[l],
                                t_ticks[s],  # t_min not fixed here
                                t_ticks[t],
                            )

                            # Count the events within the region
                            B, C = event_count(test_region, agg_df)

                            # Compute Metrics
                            basic_l_score = likelihood_ratio(B, C)
                            general_l_scores = [
                                likelihood_ratio_kulgen(B, C, B_tot, C_tot, eps)
                                for eps in [0.0, 0.25, 0.50, 0.75, 1.00]
                            ]

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
                                "C/B_in": (C / B) if B != 0 else np.inf,
                                "C/B_out": (C_tot - C) / (B_tot - B)
                                if B_tot != B
                                else np.inf,
                                "l_score_basic": basic_l_score,
                                #"p_value_basic": np.nan,
                                "l_score_000": general_l_scores[0],
                                #"p_value_000": np.nan,
                                "l_score_025": general_l_scores[1],
                                #"p_value_025": np.nan,
                                "l_score_050": general_l_scores[2],
                                #"p_value_050": np.nan,
                                "l_score_075": general_l_scores[3],
                                #"p_value_075": np.nan,
                                "l_score_100": general_l_scores[4],
                                #"p_value_100": np.nan,
                            }

                            # Count Regions
                            num_regions += 1

        # Print Progress
        print("{0:.2f}% complete.".format((s + 1) * 100 / len(t_ticks)), end="\r")

    region_score_df = pd.DataFrame.from_dict(scores_dict, "index")

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region.
    # Sort it so that highest F(S) score is at the top.
    region_score_df = region_score_df.sort_values("l_score_basic", ascending=False)

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
        Dataframe with a populated `p_value` column.
        np.ndarray of max F(S) values from the resulting simulations.
    """

    # Infer grid partition from the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    print("Found a grid partition = {}".format(grid_partition))

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_df)

    print(
        "Searching over the region spanning {} hours".format(global_region.num_hours())
    )

    # Re-create the grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    best_likelihood_scores = []

    print("\n====================")
    print("Beginning Simulation")
    print("====================")

    # Main Loop
    for sim in range(n_sims):
        max_score = 0
        print("Performing simulation {} of {}.".format(sim + 1, n_sims), end="\r")
        for i, _ in enumerate(x_ticks):
            for j in range(
                i + 1, np.min([i + (int(grid_partition / 2)), grid_partition]) + 1
            ):
                for k, _ in enumerate(y_ticks):
                    for l in range(
                        k + 1,
                        np.min([k + (int(grid_partition / 2)), grid_partition]) + 1,
                    ):
                        for t in range(1, len(t_ticks)):

                            # At each iteration, create the space_time region
                            test_region = Region(
                                x_ticks[i],
                                x_ticks[j],
                                y_ticks[k],
                                y_ticks[l],
                                t_ticks[0],
                                t_ticks[t],
                            )

                            baseline, simulated = simulate_event_count(
                                test_region, forecast_df
                            )

                            l_score = likelihood_ratio(baseline, simulated)

                            max_score = l_score if l_score > max_score else max_score

        best_likelihood_scores.append(max_score)
    arr = np.array(best_likelihood_scores)
    t2 = time.perf_counter()

    print("\nTime Elapsed: {} seconds".format(t2 - t1))

    # Estimate p-value
    res_df["p_value"] = 1 - res_df["likelihood_score"].apply(
        lambda x: np.count_nonzero(x > arr) + 1
    ) / (n_sims + 1)
    copy = res_df

    return copy, arr


# TODO
def historical_data_test():
    return None
