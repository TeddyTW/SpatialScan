"""Module to contain all region and grid-based construction code."""

from datetime import datetime
from typing import Type
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


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
    assert set(["measurement_start_utc", "measurement_end_utc"]) <= set(df.columns)

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
    assert set(["lon", "lat", "measurement_end_utc", "count", "baseline"]) <= set(
        data.columns
    )

    region_mask = (
        (data["lon"].between(S.x_min, S.x_max))
        & (data["lat"].between(S.y_min, S.y_max))
        & (data["measurement_end_utc"] > S.t_min)
        & (data["measurement_end_utc"] <= S.t_max)
    )
    S_df = data.loc[region_mask]
    if S_df.empty:
        return 0, 0
    return S_df["baseline"].sum() / 1e6, S_df["count"].sum() / 1e6


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
    assert set(["lon", "lat", "measurement_end_utc", "count", "baseline"]) <= set(
        data.columns
    )

    region_mask = (
        (data["lon"].between(S.x_min, S.x_max))
        & (data["lat"].between(S.y_min, S.y_max))
        & (data["measurement_end_utc"] > S.t_min)
        & (data["measurement_end_utc"] <= S.t_max)
    )
    S_df = data.loc[region_mask]
    if S_df.empty:
        return 0, 0
    S_df["simulated"] = np.random.poisson(S_df["baseline"])

    return S_df["simulated"].sum() / 1e6, S_df["actual"].sum() / 1e6


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


def plot_region_grid(
    forecast_data: pd.DataFrame,
    time_slice: datetime,
    grid_partition: int,
    plot_type="count",
) -> None:

    """Functionality to plot the computational grid on a region of interest.
    To be mainly used as a visualisation tool."""

    global_region = infer_global_region(forecast_data)
    x_ticks, y_ticks, t_ticks = make_grid(global_region, grid_partition)
    forecast_data["cb_ratio"] = forecast_data["count"] / forecast_data["baseline"]
    forecast_data = forecast_data[forecast_data["measurement_end_utc"] == time_slice]

    sbn.scatterplot(
        data=forecast_data,
        x="lon",
        y="lat",
        size=plot_type,
        legend="brief",
        hue=plot_type,
    )

    for _, x in enumerate(x_ticks[1:-1]):
        plt.axvline(x=x, alpha=0.4, c="k")
    for _, y in enumerate(y_ticks[1:-1]):
        plt.axhline(y=y, alpha=0.4, c="k")

    plt.title("Plot Type: {}".format(plot_type))
    plt.xlim([global_region.x_min, global_region.x_max])
    plt.ylim([global_region.y_min, global_region.y_max])

    return None

def make_region_from_res(res_df: pd.DataFrame, whole_prediction_period: bool = True,
                         rank: int = 1) -> Type[Region]:
    """The output of the main spatial scan loop is a dataframe named `res_df`.
    This function enables us to create a `Region` object from that resulting
    dataframe. The default is set to `rank=1`, meaning that the function will
    default to create a space-time region corresponding to the highest scoring
    likelihood ratio from the scan.

    Args:
        res_df: Resulting dataframe from the spatial scan
        whole_prediction_period: (Boolean) res_df will contain data spanning the
                                 the whole prediction period t= 0, 1, ... W. If
                                 set to true, the resulting region will span over
                                 all of these time steps. Otherwise, it will just
                                 return the highest scoring space-time region.
        rank: Determines which space-time region to create according to their
              likelihood ratio scores as determined by the loop.
    Returns:
        Space-Time region spanning the spatial region of interest. Time period
        either spans the whole prediction period, or just the highest scoring
        slice as explained above.
    """

    x_min = res_df.iloc[rank].x_min
    x_max = res_df.iloc[rank].x_max
    y_min = res_df.iloc[rank].y_min
    y_max = res_df.iloc[rank].y_max
    if whole_prediction_period:
        t_min = res_df['t_min'].min()
        t_max = res_df['t_max'].max()
    else:
        t_min = res_df.iloc[rank].t_min
        t_max = res_df.iloc[rank].t_max
    return Region(x_min, x_max, y_min, y_max, t_min, t_max)


# Plot the time series of all detectors within a region of interest
def plot_region_time_series(region: Type[Region], forecast_df: pd.DataFrame) -> None:
    """Plots all the time series associated with a space-time region. To be used
    in conjunction with `make_region_from_res` as follows:
        1. Find Highest scoring regions from the main scan loop
        2. Convert ones of interest (high-rank) to regions using
           `make_region_from_res`
        3. Plot the individual time series within that region using this function.
    Args:
        region: Space-Time Region of interest
        forecast_df: dataframe containing all prediction data from timeseries module.
    """

    region_mask = (
        (forecast_df["lon"].between(region.x_min, region.x_max))
        & (forecast_df["lat"].between(region.y_min, region.y_max))
        & (forecast_df["measurement_end_utc"] > region.t_min)
        & (forecast_df["measurement_end_utc"] <= region.t_max)
    )
    df = forecast_df.loc[region_mask]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sbn.lineplot(data=df, x="measurement_end_utc", y="count", hue='detector_id', ax=ax)
    fig.suptitle("Actual Counts")
    return None
    