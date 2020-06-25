"""Module to contain the main Spatio-Temporal Scan Loop"""

import pandas as pd
import numpy as np
from region import (
    Region,
    region_event_count,
    infer_global_region,
    make_grid,
)
from likelihood import likelihood_ratio
import time


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

    columns = [
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "t_min",
        "t_max",
        "B",
        "C",
        "likelihood_score",
        "p_value",
    ]
    region_score_df = pd.DataFrame(columns=columns)

    num_regions = 0
    # Loop over all possible prisms in the space
    for i, _ in enumerate(x_ticks):
        for j in range(i + 1, len(x_ticks)):
            for k, _ in enumerate(y_ticks):
                for l in range(k + 1, len(y_ticks)):
                    for t in range(1, len(t_ticks)):

                        # Count Regions
                        num_regions += 1

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

                        region_score_df = region_score_df.append(
                            {
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
                            },
                            ignore_index=True,
                        )
        # Print Progress
        print("{0:.2f}% complete.".format(i * 100 / len(x_ticks)), end="\r")
    print("100.00% complete.", end="\r")

    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region.
    # Sort it so that highest F(S) score is at the top.
    region_score_df = region_score_df.sort_values("likelihood_score", ascending=False)

    t2 = time.perf_counter()

    print("\n%d space-time regions searched in %.2f seconds" % (num_regions, t2 - t1))

    return region_score_df
