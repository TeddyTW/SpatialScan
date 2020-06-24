"""Module to contain the main Spatio-Temporal Scan Loop"""

import pandas as pd
import numpy as np
import logging
from region import (
    Region,
    region_event_count,
    infer_global_region,
    make_grid,
    simulate_region_event_count,
)
from likelihood import likelihood_ratio


def EBP(forecast_data: pd.DataFrame, grid_partition: int,
        find_signifiance: bool = False, n_sims: int = 100) -> pd.DataFrame:

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
        "likelihood_score",
        "p_value",
    ]
    region_score_df = pd.DataFrame(columns=columns)

    logging.info("Performing Loop")

    # Loop over all possible prisms in the space
    for i, _ in enumerate(x_ticks):
        for j in range(i + 1, len(x_ticks)):
            for k, _ in enumerate(y_ticks):
                for l in range(k + 1, len(y_ticks)):
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
                        
                        # Calculate the Likelihood ratio for this region
                        if B == 0 and C == 0:
                            # Space-Time Regions with no detectors are of no 
                            # interest
                            l_score = np.nan
                        else:
                            l_score = likelihood_ratio(B, C)

                        region_score_df = region_score_df.append(
                            {
                                "x_min": x_ticks[i],
                                "x_max": x_ticks[j],
                                "y_min": y_ticks[k],
                                "y_max": y_ticks[l],
                                "t_min": t_ticks[0],
                                "t_max": t_ticks[t],
                                "likelihood_score": l_score,
                                "p_value": np.nan
                            },
                            ignore_index=True,
                        )
    # At this point, we have a dataframe populated with likelihood statistic
    # scores for each search region.
    # Sort it so that highest F(S) score is at the top.
    region_score_df = region_score_df.sort_values("likelihood_score", ascending=False)

    if not find_signifiance:
        return region_score_df

    logging.info("Randomisation Testing Enabled. Performing {} simulations.".format(n_sims))

    # Perform Randomisation Testing here - very intensive.
    # Loop over all possible prisms in the space
    best_likelihood_scores = []
    for _ in range(n_sims):
        max_likelihood_score = 0
        for i, _ in enumerate(x_ticks):
            for j in range(i + 1, len(x_ticks)):
                for k, _ in enumerate(y_ticks):
                   for l in range(k + 1, len(y_ticks)):
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
                            sim_count, actual_count = simulate_region_event_count(test_region, forecast_data)
                            l_score = likelihood_ratio(sim_count, actual_count)

                            if l_score > max_likelihood_score:
                                max_likelihood_score = l_score
        best_likelihood_scores.append(max_likelihood_score)

    return region_score_df
