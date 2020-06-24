"""Module to contain all region and grid-based construction code."""

from datetime import datetime
from typing import Type
import pandas as pd
import numpy as np


class Region:
    """Class to represent space-time region"""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        t_min: datetime,
        t_max: datetime,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max
        self.label = None

    def __str__(self):
        return "({}, {}) x ({}, {}) x ({}, {})".format(
            self.x_min, self.x_max, self.y_min, self.y_max, self.t_min, self.t_max
        )

    def add_label(self, label):
        self.label = label

    def num_days(self):
        return (self.t_max - self.t_min).days

    def num_hours(self):
        return (self.t_max - self.t_min).days * 24


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Check columns are in here.
    assert set(['measurement_start_utc', 'measurement_end_utc']) <= set(df.columns)

    copy_df = df
    copy_df["measurement_start_utc"] = pd.to_datetime(df["measurement_start_utc"])
    copy_df["measurement_end_utc"] = pd.to_datetime(df["measurement_end_utc"])
    return copy_df


def region_event_count(S: Type[Region], data: pd.DataFrame) -> tuple:

    """Function to calculate both the expected (B) and actual (C) count
    (vehicles) within a given space-time region S. Used in the likelihood ratio
    statistic.
    Args:
        S: Space-Time Region to count events in
        data: Usual format SCOOT dataframe
    Returns: (Tuple of floats) both types of event counts within region S.
    """

    # Check for columns existence.
    assert set(['lon', 'lat', 'measurement_end_utc', 'count', 'baseline']) <= set(data.columns)

    region_mask = (
        (data["lon"].between(S.x_min, S.x_max))
        & (data["lat"].between(S.y_min, S.y_max))
        & (data["measurement_end_utc"] > S.t_min)
        & (data["measurement_end_utc"] <= S.t_max)
    )
    S_df = data.loc[region_mask]
    if S_df.empty:
        return 0, 0
    return S_df['baseline'].sum(), S_df['count'].sum()


def simulate_region_event_count(S: Type[Region], data: pd.DataFrame) -> tuple:

    """Function to simulate the count (vehicles) within a given
    space-time region S assuming a Poisson Distribution with mean given by the
    baseline forecast. Used in randomisation testing.
    Args:
        S: Space-Time Region to count events in
        data: Usual format SCOOT dataframe
    Returns: (Tuple of floats) both types of event counts within region S.
    """

    # Check for columns existence.
    assert set(['lon', 'lat', 'measurement_end_utc', 'count', 'baseline']) <= set(data.columns)

    region_mask = (
        (data["lon"].between(S.x_min, S.x_max))
        & (data["lat"].between(S.y_min, S.y_max))
        & (data["measurement_end_utc"] > S.t_min)
        & (data["measurement_end_utc"] <= S.t_max)
    )
    S_df = data.loc[region_mask]
    if S_df.empty:
        return 0, 0
    S_df['simulated'] = np.random.poisson(S_df['baseline'])

    return S_df["simulated"].sum(), S_df["actual"].sum()

def infer_global_region(data: pd.DataFrame) -> Type[Region]:
    x_min = data["lon"].min()
    x_max = data["lon"].max()
    y_min = data["lat"].min()
    y_max = data["lat"].max()
    t_min = data["measurement_start_utc"].min()
    t_max = data["measurement_end_utc"].max()

    return Region(x_min, x_max, y_min, y_max, t_min, t_max)


def make_grid(global_region: Type[Region], N: int) -> tuple:
    """Function to create grid arrays to iterate over in the main loop. Divides
    the global region `global_region` into an N x N grid. Looping over the main
    grid is O(N^4 * W).
    Args:
        global_region: The whole domain of which the scan is performed over.
        N: Number of partitions per spatial axis.
    Returns:
        x: np.array of equally spaced values on the x axis of global_domain
        y: np.array of equally spaced values on the y axis of global_domain
        t: np.array of equally spaced values on the t axis of global_domain
    """

    x = np.linspace(global_region.x_min, global_region.x_max, N + 1)
    y = np.linspace(global_region.y_min, global_region.y_max, N + 1)

    t = pd.date_range(start=global_region.t_min, end=global_region.t_max, freq="H")

    return x, y, t
