"""Functionality for the main Spatial Scan loop over a rectangular grid"""
import time

import pandas as pd
import numpy as np

from .metrics import likelihood_ratio_ebp
from .utils import event_count


def scan(agg_df: pd.DataFrame, grid_resolution: int) -> pd.DataFrame:

    """Main function for looping through the sub-space-time regions (S) of
    global_region represented by data in forecast_data. We search for regions
    with the highest score according to various different metrics. The EBP one is
    given by:
                F(S) := Pr (data | H_1 (S)) / Pr (data | H_0)
    where H_0 and H_1 (S) are defined in the Expectation-Based Scan Statistic
    paper by D. Neill.

    Args:
        agg_df: dataframe consisting of the detectors which lie in
                       global_region, their locations and both their
                       baseline and actual counts for the past W days.
        grid_partition: Split each spatial axis into this many partitions.
    Returns:
        Dataframe summarising each space-time region's F(S) score.
    """
    # ==================
    # 1) Scan over grid
    # ==================

    # Set Initial Timer
    init_time = time.perf_counter()

    # Infer max/min time labels from the input data
    t_min = agg_df.measurement_start_utc.min()
    t_max = agg_df.measurement_end_utc.max()

    # Set up iterators
    x_ticks = range(grid_resolution + 1)
    y_ticks = range(grid_resolution + 1)
    t_ticks = pd.date_range(start=t_min, end=t_max, freq="H")

    # Each search region has spatial extent that covers less than half_max row/columns
    half_max = (
        int(grid_resolution / 2)
        if grid_resolution % 2 == 0
        else int((grid_resolution + 1) / 2)
    )

    # Time direction convention - reverse
    t_ticks = t_ticks[::-1]

    num_regions = 0
    scores_dict = {}
    # Loop over all possible spatial extents of max size grid_resolution/2 in both axes
    for t_min in t_ticks[1:]:  # t_min
        for col_min in x_ticks:  # col_min
            for col_max in range(
                col_min + 1, np.min([col_min + half_max, grid_resolution]) + 1
            ):  # col_max
                for row_min in y_ticks:  # row_min
                    for row_max in range(
                        row_min + 1, np.min([row_min + half_max, grid_resolution]) + 1
                    ):  # row_max

                        # Count up baselines and actual counts here
                        baseline_count, actual_count = event_count(
                            agg_df, col_min, col_max, row_min, row_max, t_min, t_max
                        )

                        # Compute Metric(s)
                        ebp_l_score = likelihood_ratio_ebp(
                            baseline_count, actual_count
                        )  # Normal EBP metric

                        # Append results
                        scores_dict[num_regions] = {
                            "row_min": row_min + 1,
                            "row_max": row_max,
                            "col_min": col_min + 1,
                            "col_max": col_max,
                            "measurement_start_utc": t_min,
                            "measurement_end_utc": t_max,
                            "baseline_count": baseline_count,
                            "actual_count": actual_count,
                            "l_score_ebp": ebp_l_score,
                            # "p_value_EBP": np.nan,
                        }

                        # Count Regions
                        num_regions += 1

        # Print Progress
        print(
            "Search spatial regions with t_min = {} and t_max = {}".format(
                t_min, t_max
            ),
            end="\r",
        )

    scan_time = time.perf_counter()

    print(
        "\n{} space-time regions searched in {:.2f} seconds".format(
            num_regions, scan_time - init_time
        )
    )

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for *each* search region. Sort it so that the highest `l_score_EBP`
    # score is at the top.
    all_scores = pd.DataFrame.from_dict(scores_dict, "index").sort_values(
        "l_score_ebp", ascending=False
    )

    # ====================================
    # 2) Aggregating scores to grid level
    # ====================================

    # Now, we aggregate these scores to grid level by taking an average of l_score_EBP.
    # i.e. For a given grid cell, we find all search regions that contain it, and
    # return the average score. Useful for visualisation.

    return_dict = {}
    num_regions = 0
    for t_min in t_ticks[1:]:
        num_spatial_regions = 0

        for row_num in y_ticks[1:]:
            for col_num in x_ticks[1:]:

                gridcell = all_scores[
                    (all_scores["col_min"] <= col_num)
                    & (all_scores["col_max"] >= col_num)
                    & (all_scores["row_min"] <= row_num)
                    & (all_scores["row_max"] >= row_num)
                    & (all_scores["measurement_start_utc"] == t_min)
                ]

                mean_score = gridcell["l_score_ebp"].mean()
                std = gridcell["l_score_ebp"].std()

                return_dict[num_regions] = {
                    "measurement_start_utc": t_min,
                    "measurement_end_utc": t_max,
                    "row": row_num,
                    "col": col_num,
                    "l_score_ebp_mean": mean_score,
                    "l_score_ebp_std": std,
                }

                num_spatial_regions += 1
                num_regions += 1

    agg_time = time.perf_counter()

    grid_level_scores = pd.DataFrame.from_dict(return_dict, "index")

    print(
        "\n{} Results aggregated to grid cell level in {:.2f} seconds".format(
            num_regions, agg_time - scan_time
        )
    )
    print("Total run time: {:.2f} seconds".format(agg_time - init_time))

    return all_scores, grid_level_scores
