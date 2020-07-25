"""Test all scan statistic functions"""

from __future__ import annotations
from typing import TYPE_CHECKING
import datetime

import numpy as np
import pandas as pd
from shapely import wkb
from geoalchemy2.shape import to_shape

from odysseus.scanstat.preprocess import preprocessor
from odysseus.scanstat.forecast import forecast
from odysseus.scanstat.utils import aggregate_to_grid
from odysseus.scanstat.scan import scan

if TYPE_CHECKING:
    from odysseus.scoot import ScanScoot


def test_scan(scan_scoot: ScanScoot, scoot_writer) -> None:
    """Test whole pipeline of scan functions with settings below."""

    # Set up scan variables
    days_in_past = 28
    days_in_future = 1
    ts_method = "HW"
    # TODO - Can we get at these through scan_scoot?
    borough = "Westminster"
    grid_resolution = 8

    scoot_writer.update_remote_tables()
    readings = scan_scoot.scoot_fishnet_readings(
        borough=borough,
        start=scoot_writer.start,
        upto=scoot_writer.upto,
        output_type="df",
    )

    init_num_detectors = len(readings["detector_id"].unique())
    init_num_days = (
        readings["measurement_end_utc"].max() - readings["measurement_start_utc"].min()
    ).days

    # 1) Pre-Process data
    proc_df = preprocessor(readings)
    print(proc_df)
    preprocess_checks(proc_df, init_num_days, init_num_detectors)

    # Update the number of detectors for the rest of the test - some are thrown
    # away in the pre-process stage.
    init_num_detectors = len(proc_df["detector_id"].unique())
    t_max = proc_df["measurement_end_utc"].max()

    # 2) Produce forecast
    forecast_df = forecast(
        proc_df,
        days_in_past=days_in_past,
        days_in_future=days_in_future,
        method=ts_method,
    )
    print(forecast_df)

    forecast_checks(forecast_df, init_num_detectors, days_in_future, t_max)

    # 3) Aggregate data to grid level
    # First requires getting grid_cell information for the detectors
    # TODO - this doesn't come from `scoot_fishnet_readings` at the moment
    detector_df = scan_scoot.scoot_fishnet(borough, output_type="df")
    detector_df["geom"] = detector_df["geom"].apply(lambda x: wkb.loads(x, hex=True))
    detector_df["location"] = (
        detector_df["location"].apply(to_shape).apply(lambda x: x.wkt)
    )
    print(detector_df)

    # Now we can aggregate
    agg_df = aggregate_to_grid(detector_df, forecast_df)
    print(agg_df)

    aggregate_checks(agg_df, days_in_future, grid_resolution)

    # 4) Scan
    all_scores, grid_level_scores = scan(agg_df, grid_resolution=grid_resolution)

    scan_checks(all_scores, grid_level_scores, t_max, days_in_future, grid_resolution)
    print(all_scores)
    print(grid_level_scores)
    return


def preprocess_checks(
    proc_df: pd.DataFrame, init_num_days: int, init_num_detectors: int
) -> None:
    """Test preprocessing of data is carried out successfully in `preprocessor()`"""

    # All outputted values should not be NaN
    assert not proc_df.isnull().values.any()

    cols = [
        "detector_id",
        "lon",
        "lat",
        "location",
        "measurement_start_utc",
        "measurement_end_utc",
        "n_vehicles_in_interval",
        "rolling_threshold",
        "global_threshold",
    ]
    assert set(cols) == set(proc_df.columns)

    num_days = (
        proc_df["measurement_end_utc"].max() - proc_df["measurement_start_utc"].min()
    ).days
    assert init_num_days == num_days

    num_detectors = len(proc_df["detector_id"].unique())
    assert init_num_detectors <= num_detectors

    # Use this to check that each detector_id has a unique lon, lat and location
    assert num_detectors == len(
        proc_df.groupby(["detector_id", "lon", "lat", "location"])
    )

    assert len(proc_df) == num_days * 24 * num_detectors


def forecast_checks(
    forecast_df: pd.DataFrame,
    init_num_detectors: int,
    days_in_future: int,
    t_max: datetime,
) -> None:
    """Test that forecasts are carried out successfully in `forecast()`."""

    # All outputted values should not be NaN
    assert not forecast_df.isnull().values.any()

    cols = [
        "detector_id",
        "lon",
        "lat",
        "location",
        "measurement_start_utc",
        "measurement_end_utc",
        "count",
        "baseline",
    ]
    assert set(cols) == set(forecast_df.columns)

    num_detectors = len(forecast_df["detector_id"].unique())
    assert init_num_detectors == num_detectors

    neg_baselines = forecast_df[forecast_df["baseline"] < 0]
    assert len(neg_baselines) == 0

    assert len(forecast_df) == days_in_future * 24 * num_detectors

    assert forecast_df["measurement_end_utc"].max() == t_max
    assert forecast_df["measurement_start_utc"].min() == t_max - np.timedelta64(
        days_in_future, "D"
    )


def aggregate_checks(
    agg_df: pd.DataFrame, days_in_future: int, grid_resolution: int
) -> None:
    """Test that individual detector data is aggregated correctly to each grid
    cell using `aggregate_to_grid()`"""

    # Check that merge was successful
    assert len(agg_df) > 0

    # All outputted values should not be NaN
    assert not agg_df.isnull().values.any()

    # Correct columns
    cols = [
        "row",
        "col",
        "measurement_start_utc",
        "measurement_end_utc",
        "count",
        "baseline",
    ]
    assert set(cols) == set(agg_df.columns)

    # Check that each grid cell has correct number of readings
    min_readings = agg_df.groupby(["row", "col"])["measurement_start_utc"].count().min()
    max_readings = agg_df.groupby(["row", "col"])["measurement_start_utc"].count().max()
    assert min_readings == max_readings
    assert min_readings == 24 * days_in_future

    # Check that row and col numbers fall within range
    assert agg_df["row"].min() >= 1
    assert agg_df["col"].min() >= 1
    assert agg_df["row"].max() <= grid_resolution
    assert agg_df["col"].max() <= grid_resolution


def scan_checks(
    all_scores: pd.DataFrame,
    grid_level_scores: pd.DataFrame,
    t_max: datetime,
    days_in_future: int,
    grid_resolution: int,
) -> None:
    """ Test that output from the main `scan()` function is sensible."""

    # All outputted values should not be NaN
    assert not all_scores.isnull().values.any()
    assert not grid_level_scores.isnull().values.any()

    assert len(all_scores) > 0
    assert len(grid_level_scores) == 24 * days_in_future * grid_resolution ** 2

    all_score_cols = [
        "row_min",
        "row_max",
        "col_min",
        "col_max",
        "measurement_start_utc",
        "measurement_end_utc",
        "baseline_count",
        "actual_count",
        "l_score_ebp",
    ]
    grid_level_cols = [
        "measurement_start_utc",
        "measurement_end_utc",
        "row",
        "col",
        "l_score_ebp_mean",
        "l_score_ebp_std",
    ]

    assert set(all_score_cols) == set(all_scores.columns)
    assert set(grid_level_cols) == set(grid_level_scores.columns)

    assert all_scores["row_min"].min() >= 1
    assert all_scores["col_min"].min() >= 1
    assert all_scores["row_max"].max() <= grid_resolution
    assert all_scores["col_max"].max() <= grid_resolution
    assert grid_level_scores["row"].min() >= 1
    assert grid_level_scores["col"].min() >= 1
    assert grid_level_scores["row"].max() <= grid_resolution
    assert grid_level_scores["col"].max() <= grid_resolution

    assert (all_scores["row_max"] - all_scores["row_min"]).max() <= (
        grid_resolution / 2
    ) - 1
    assert (all_scores["col_max"] - all_scores["col_min"]).max() <= (
        grid_resolution / 2
    ) - 1

    assert len(all_scores["measurement_end_utc"].unique()) == 1
    assert all_scores.at[0, "measurement_end_utc"] == t_max
    assert all_scores["measurement_start_utc"].min() == t_max - np.timedelta64(
        days_in_future, "D"
    )

    assert all_scores["l_score_ebp"].min() >= 1
