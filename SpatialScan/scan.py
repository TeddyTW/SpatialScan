"""Module to contain the main Spatio-Temporal Scan Loop"""

import time
import pandas as pd
import numpy as np
from SpatialScan.region import (
    Region,
    region_event_count,
    infer_global_region,
    make_grid,
    simulate_region_event_count,
)
from SpatialScan.likelihood import likelihood_ratio

def EBP(forecast_data: pd.DataFrame, grid_partition: int,) -> pd.DataFrame:

    """Main function for looping through the sub-space-time regions (S) of
    global_region. Searching for regions with the highest score according to the
    statistic:
                F(S) := Pr (data | H_1 (S)) / Pr (data | H_0)
    where H_0 and H_1 (S) are defined in the Expectation-Based Scan Statistic
    paper by D. Neill.

    Args:
        forecast_data: dataframe consisting of the detectors which lie in
                       global_region, their locations and both their
                       baseline and actual counts for the past W days.
        grid_partition: Split each spatial axis into this many partitions.
    Returns:
            Dataframe summarising each grid square's F(S) score.
    """

    # Set Initial Timer
    t1 = time.perf_counter()

    # Find the global region on which the observations live
    global_region = infer_global_region(forecast_data)

    # Create the Grid
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)

    num_regions = 0
    scores_dict = {}
    # Loop over all possible prisms of max size N/2 in the space
    for i, _ in enumerate(x_ticks):
        for j in range(
            i + 1, np.min([i + (int(grid_partition / 2)), grid_partition]) + 1
        ):
            for k, _ in enumerate(y_ticks):
                for l in range(
                    k + 1, np.min([k + (int(grid_partition / 2)), grid_partition]) + 1
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

                        # Note: This will search through whole data-frame each
                        # iteration. XXX - improve this first.
                        B, C = region_event_count(test_region, forecast_data)

                        l_score = likelihood_ratio(B, C)

                        scores_dict[num_regions] = {
                            "x_min": x_ticks[i],
                            "x_max": x_ticks[j],
                            "y_min": y_ticks[k],
                            "y_max": y_ticks[l],
                            "t_min": t_ticks[0],
                            "t_max": t_ticks[t],
                            "B": B,
                            "C": C,
                            "likelihood_score": l_score,
                            "p_value": np.nan,
                        }
                        # Count Regions
                        num_regions += 1

        # Print Progress
        print("{0:.2f}% complete.".format(i * 100 / len(x_ticks)), end="\r")
    print("100.00% complete.", end="\r")

    region_score_df = pd.DataFrame.from_dict(scores_dict, "index")

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region.
    # Sort it so that highest F(S) score is at the top.
    region_score_df = region_score_df.sort_values("likelihood_score", ascending=False)

    t2 = time.perf_counter()

    print("\n%d space-time regions searched in %.2f seconds" % (num_regions, t2 - t1))

    return region_score_df


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

                            baseline, simulated = simulate_region_event_count(
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
