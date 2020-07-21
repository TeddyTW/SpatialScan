"""Contains all utility functionality required for Spatial Scan statistics."""

import datetime
import pandas as pd


def aggregate_to_grid(
    detector_df: pd.DataFrame, forecast_df: pd.DataFrame
) -> pd.DataFrame:

    """Aggregates data from each detector in forecast_df to grid-cell level
    as prescribed by detector_df.
    Args:
        detector_df: SCOOT data with location, row, col and attached geometry (grid-cell)
        forecast_df: Forecasted SCOOT data from time series analysis
    Returns:
        agg_df: dataframe of SCOOT data aggregated to the spatial grid at hourly
                time steps. Actual counts and baseline estimates are aggregated as
                separate columns in the returned dataframe.
    """

    # Check for columns in detector_df and readings
    assert set(["detector_id", "location", "lon", "lat", "row", "col", "geom"]) == set(
        detector_df.columns
    )
    assert set(
        [
            "detector_id",
            "measurement_start_utc",
            "measurement_end_utc",
            "lon",
            "lat",
            "location",
            "count",
            "baseline",
        ]
    ) == set(forecast_df.columns)

    # Merge readings with detector grid info
    agg_df = forecast_df.merge(
        detector_df, how="left", on=["detector_id", "lon", "lat", "location"]
    )

    # These columns make no sense when aggregating to grid level, so drop
    agg_df = agg_df.drop(["detector_id", "lon", "lat", "location"], axis=1)

    # This column not particulalry needed for the scan, so no need
    # to carry it round.
    agg_df = agg_df.drop(["geom"], axis=1)

    # Sum counts and baselines at grid cell level
    agg_df = agg_df.groupby(
        ["row", "col", "measurement_start_utc", "measurement_end_utc"]
    ).sum()

    # Convert back to normal dataframe
    agg_df = agg_df.reset_index()

    return agg_df


def event_count(
    agg_df: pd.DataFrame,
    col_min: int,
    col_max: int,
    row_min: int,
    row_max: int,
    t_min: datetime,
    t_max: datetime,
) -> pd.DataFrame:

    """Aggregate the vehicle counts that fall within the region specified by
    the last 6 arguments (row/colums/time identifiers). Scaled by 1e6 for metric
    calculation.

    Args:
        agg_df: SCOOT data for actual counts and baselines aggregated to grid-cell
                level
        col_min: left boundary column number of search region (inclusive)
        col_max: right boundary column number of search region (inclusive)
        row_min: bottom boundary row_number of search region (inclusive)
        row_max: top boundary row_number of search region (inclusive)
        t_min: earliest time defining the space-time region (inclusive)
        t_max: latest_time defining the space-time region (inclusive)
    Returns:
        baseline_count: sum of detector baseline estimates in search region
        actual_count: sum of actual detector counts in search region
    Notes:
        t_max is fixed currently. The scan statistic is calculated for all space-time
        regions such that t_max is the most recent day. The search is then conducted
        over the space time regions that begin before t_max.
    """

    # Check for columns existence.
    assert set(
        [
            "row",
            "col",
            "measurement_start_utc",
            "measurement_end_utc",
            "count",
            "baseline",
        ]
    ) <= set(agg_df.columns)

    search_region_mask = (
        (agg_df["col"].between(col_min, col_max))
        & (agg_df["row"].between(row_min, row_max))
        & (agg_df["measurement_start_utc"] >= t_min)
        & (agg_df["measurement_end_utc"] <= t_max)
    )

    search_region_data = agg_df.loc[search_region_mask]

    if search_region_data.empty:
        return 0, 0
    return (
        search_region_data["baseline"].sum() / 1e6,
        search_region_data["count"].sum() / 1e6,
    )
