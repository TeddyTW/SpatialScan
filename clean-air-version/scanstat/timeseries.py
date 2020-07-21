"""Module to contain implementations of various time-series methods to be used
within the scan statistic framework. Currently only contains the Holt-Winters
exponentially smoothed method."""

import numpy as np
import pandas as pd


def holt_winters(
    proc_df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    detectors: list = None,
) -> pd.DataFrame:

    """Time series forecast using Holt-Winters method.

    Args:
        proc_df: Dataframe of 'processed' SCOOT data
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        alpha: Optimisation parameter
        beta: Optimisation parameter
        gamma: Optimisation parameter
        detectors: List of detectors to look at. Defaults to all.

    Returns:
        Dataframe forecast in same format as SCOOT input dataframe, with baseline
        counts instead of actual counts.
    """

    # Check parameter values
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= gamma <= 1

    # Get default detectors
    if detectors is None:
        detectors = proc_df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        # Notation as in Expectation-Based Scan Statistic paper
        smooth = 1
        trend = 1
        hod = np.ones(24)
        one_det = proc_df[proc_df["detector_id"] == detector]

        # Use most recent days in the past to produce forecast
        one_det = one_det.sort_values(by=["measurement_end_utc"])
        past = one_det.tail(n=24 * days_in_past)

        # HW algorithm
        for i in range(0, len(past)):
            hour = i % 24
            count = past["n_vehicles_in_interval"].iloc[i]
            smooth_new = (alpha * (count / hod[hour])) + (1 - alpha) * (smooth + trend)
            trend = beta * (smooth_new - smooth) + (1 - beta) * trend
            hod[hour] = gamma * (count / smooth_new) + (1 - gamma) * hod[hour]
            smooth = smooth_new

        baseline = []
        endtime = []
        starttime = []

        last_training_time = proc_df["measurement_end_utc"].max()
        for j in range(0, days_in_future * 24):

            start = last_training_time + np.timedelta64(j, "h")
            end = last_training_time + np.timedelta64(j + 1, "h")

            hour = j % 24
            base = (smooth + trend) * hod[hour]
            baseline.append(base)
            endtime.append(end)
            starttime.append(start)

            smooth_new = (alpha * (base / hod[hour])) + (1 - alpha) * (smooth + trend)
            trend = beta * (smooth_new - smooth) + (1 - beta) * trend
            hod[hour] = gamma * (base / smooth_new) + (1 - gamma) * hod[hour]
            smooth = smooth_new

        forecasts = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": one_det[one_det["detector_id"] == detector]["lon"].iloc[0],
                "lat": one_det[one_det["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": starttime,
                "measurement_end_utc": endtime,
                "n_vehicles_in_interval": baseline,
            }
        )
        framelist.append(forecasts)
    return pd.concat(framelist)
