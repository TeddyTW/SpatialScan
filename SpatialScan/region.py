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


def convert_dates(
    df: pd.DataFrame,
    date_start_label="measurement_start_utc",
    date_end_label="measurement_end_utc",
) -> pd.DataFrame:

    """ Utility functionality to convert dates in a dataframe from string to
    datetime. Useful when reading in df from csv.
    Args:
        df: Any datafram with date columns
        date_start_label: Label of the column in df corresponding to the start
                          date period.
        date_end_label: Label of the column in df corresponding to the end
                          date period.
    Returns:
        Dataframe with converted date columns
    """

    # Check columns are in here.
    assert set([date_start_label, date_end_label]) <= set(df.columns)

    copy_df = df
    copy_df[date_start_label] = pd.to_datetime(df[date_start_label])
    copy_df[date_end_label] = pd.to_datetime(df[date_end_label])
    return copy_df


def aggregate_event_data(
    forecast_data: pd.DataFrame,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    t_ticks: np.ndarray,
) -> pd.DataFrame:
    """Functionality to aggregate data in forecast_data (each row represents
    an event) to a data frame consisting N^2 * W rows (each row containing the
    aggregated count in the grid cell it represents).
    Clearly needs to be called after the grid is made.
    Args:
        forecast_data: Data from `count_baseline()`
        x_ticks: x axis grid
        y_ticks: y axis grid
        t_ticks: t_axis grid
    Returns:
        Aggregated Dataframe on Grid Cell Level.
    """

    agg_dict = {}
    num_cells = 0
    for i in range(len(x_ticks) - 1):
        for j in range(len(y_ticks) - 1):
            for s in range(len(t_ticks) - 1):
                x_min = x_ticks[i]
                x_max = x_ticks[i + 1]
                y_min = y_ticks[j]
                y_max = y_ticks[j + 1]
                t_min = t_ticks[s]
                t_max = t_ticks[s + 1]

                sub_df = forecast_data[
                    (forecast_data["lon"].between(x_min, x_max))
                    & (forecast_data["lat"].between(y_min, y_max))
                    & (forecast_data["measurement_start_utc"] == t_min)
                    & (forecast_data["measurement_end_utc"] == t_max)
                ]

                b_count = sub_df["baseline"].sum()
                c_count = sub_df["count"].sum()

                agg_dict[num_cells] = {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "t_min": t_min,
                    "t_max": t_max,
                    "baseline_agg": b_count,
                    "count_agg": c_count,
                }
                num_cells += 1

    return pd.DataFrame.from_dict(agg_dict, "index")


def event_count(S: Type[Region], agg_data: pd.DataFrame) -> tuple:

    """Function to calculate both the expected (B) and actual (C) count
    (vehicles) within a given space-time region S from the grid-cell-level-aggregated.
    Used in the likelihood ratio statistic.
    Args:
        S: Space-Time Region to count events in
        agg_data: Event counts aggregated at grid level. eg. from `aggregate_event_data()`
    Returns: (Tuple of floats) both types of event counts within region S.
    """

    # Check for columns existence.
    assert set(
        [
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "t_min",
            "t_max",
            "count_agg",
            "baseline_agg",
        ]
    ) <= set(agg_data.columns)

    region_mask = (
        (agg_data["x_min"] >= S.x_min)
        & (agg_data["x_max"] <= S.x_max)
        & (agg_data["y_min"] >= S.y_min)
        & (agg_data["y_max"] <= S.y_max)
        & (agg_data["t_min"] >= S.t_min)
        & (agg_data["t_max"] <= S.t_max)
    )

    S_df = agg_data.loc[region_mask]
    if S_df.empty:
        return 0, 0
    return S_df["baseline_agg"].sum() / 1e6, S_df["count_agg"].sum() / 1e6


def simulate_event_count(S: Type[Region], forecast_data: pd.DataFrame) -> tuple:

    """Function to simulate the count (vehicles) within a given
    space-time region S assuming a Poisson Distribution with mean given by the
    baseline forecast. Used in randomisation testing.
    Args:
        S: Space-Time Region to count events in
        data: Forecast data from `count_baseline()`
    Returns: (Tuple of floats) both types of event counts within region S.
    """

    # Check for columns existence.
    assert set(["lon", "lat", "measurement_end_utc", "count", "baseline"]) <= set(
        forecast_data.columns
    )

    forecast_data["simulated"] = np.random.poisson(forecast_data["baseline"])

    region_mask = (
        (forecast_data["lon"].between(S.x_min, S.x_max))
        & (forecast_data["lat"].between(S.y_min, S.y_max))
        & (forecast_data["measurement_end_utc"] > S.t_min)
        & (forecast_data["measurement_end_utc"] <= S.t_max)
    )
    S_df = forecast_data.loc[region_mask]
    if S_df.empty:
        return 0, 0

    return S_df["baseline"].sum() / 1e6, S_df["simulated"].sum() / 1e6


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


def plot_global_region(
    forecast_data: pd.DataFrame,
    time_slice: datetime = None,
    overlay_grid: bool = True,
    grid_partition: int = 1,
    plot_type="count",
    add_legend: bool = True,
) -> None:

    """Functionality to plot the computational grid on a region of interest.
    To be mainly used as a visualisation tool for choosing the grid_partition
    value.
    Args:
        forecast_data: Resulting df from `count_baseline()`
        time_slice: Date time representing time slice to plot
        overlay_grid: Overlay computational grid or not.
        grid_parition: Number of divisions per spatial axis
        plot_type: counts, baselines or cb_ratio
    """

    # Set defaults accordingly
    time_slice = forecast_data["measurement_end_utc"].iloc[0] if time_slice is None else time_slice
    legend = "brief" if add_legend else False

    global_region = infer_global_region(forecast_data)
    x_ticks, y_ticks, _ = make_grid(global_region, grid_partition)
    forecast_data["cb_ratio"] = forecast_data["count"] / forecast_data["baseline"]
    forecast_data.loc[~np.isfinite(forecast_data["cb_ratio"]), "cb_ratio"] = np.nan
    forecast_data = forecast_data[forecast_data["measurement_end_utc"] == time_slice]

    sbn.scatterplot(
        data=forecast_data,
        x="lon",
        y="lat",
        size=plot_type,
        legend=legend,
        hue=plot_type,
    )

    if overlay_grid:
        for _, x in enumerate(x_ticks[1:-1]):
            plt.axvline(x=x, alpha=0.4, c="k")
        for _, y in enumerate(y_ticks[1:-1]):
            plt.axhline(y=y, alpha=0.4, c="k")

    plt.title("Plot Type: {}, {}".format(plot_type, time_slice))
    plt.xlim([global_region.x_min, global_region.x_max])
    plt.ylim([global_region.y_min, global_region.y_max])

    return None


def make_region_from_res(
    res_df: pd.DataFrame, whole_prediction_period: bool = True, rank: int = 0
) -> Type[Region]:
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
        t_min = res_df["t_min"].min()
        t_max = res_df["t_max"].max()
    else:
        t_min = res_df.iloc[rank].t_min
        t_max = res_df.iloc[rank].t_max
    return Region(x_min, x_max, y_min, y_max, t_min, t_max)


# Plot the time series of all detectors within a region of interest
def plot_region_time_series(
    region: Type[Region],
    forecast_df: pd.DataFrame,
    plot_type: str = "count",
    add_legend: bool = False,
) -> None:
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

    legend = "brief" if add_legend else False

    # Check for columns existence.
    assert set(["lon", "lat", "measurement_end_utc", plot_type]) <= set(
        forecast_df.columns
    )

    region_mask = (
        (forecast_df["lon"].between(region.x_min, region.x_max))
        & (forecast_df["lat"].between(region.y_min, region.y_max))
        & (forecast_df["measurement_end_utc"] > region.t_min)
        & (forecast_df["measurement_end_utc"] <= region.t_max)
    )
    df = forecast_df.loc[region_mask]

    fig, ax = plt.subplots(figsize=(15, 6))
    sbn.lineplot(
        data=df,
        x="measurement_end_utc",
        y=plot_type,
        hue="detector_id",
        ax=ax,
        legend=legend,
    )
    fig.suptitle("{}s between {} and {}".format(plot_type, region.t_min, region.t_max))
    return None


def plot_region_by_rank(
    rank: int,
    res_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    plot_type="count",
    add_legend: bool = False,
) -> None:

    """Functionality to plot the 'rank'ed region form the results dataframe
    superposed with the global grid.
    Args:
        rank: Rank of region within res_df dataframe. Best is 0.
        res_df: Resulting datafram from `EBP()`
        forecast_df: Resulting dataframe from `count_baseline()`
        plot_type: Size of dots represent actual counts, baseline or c/b ratio
    """

    legend = "brief" if add_legend else False

    # Infer grid partition from the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    # Get grid partition here
    x_min = res_df["x_min"].iloc[rank]
    x_max = res_df["x_max"].iloc[rank]
    y_min = res_df["y_min"].iloc[rank]
    y_max = res_df["y_max"].iloc[rank]
    t_min = res_df["t_min"].iloc[rank]
    t_max = res_df["t_max"].iloc[rank]

    plot_global_region(
        forecast_df,
        res_df["t_max"].iloc[rank],
        grid_partition=grid_partition,
        plot_type=plot_type,
        add_legend=legend,
    )
    plt.hlines(y_min, x_min, x_max)
    plt.hlines(y_max, x_min, x_max)
    plt.vlines(x_min, y_min, y_max)
    plt.vlines(x_max, y_min, y_max)
    plt.title("{}s between {} and {}. Rank: {}".format(plot_type, t_min, t_max, rank))

    plt.show()


def cleanse_forecast_data(forecast_df: pd.DataFrame) -> pd.DataFrame:

    """Utility function to ensure that the forecast_df from `count_baseline()`
    is in the correct format to move forward with processing. Removes NaNs, assigns
    zero to any negative baseline values, and converts dated into datetime format
    if required.
    Args:
        forecast_df: Data frame from `count_baseline()`
    Returns
        pd.DataFrame: Cleansed dataframe
    """

    init_length = len(forecast_df["count"])
    test_date = forecast_df["measurement_start_utc"].iloc[0]

    # First check that dates are in the right format
    if isinstance(test_date, datetime):
        print("Dates in datetime format. Moving to next stage.\n")
    else:
        print("Dates are not in datetime format. Attempting to convert...")
        forecast_df = convert_dates(forecast_df)
        test_date = forecast_df["measurement_start_utc"].iloc[0]
        print(
            "Dates converted successfully: {}.\n".format(
                isinstance(test_date, datetime)
            )
        )

    # Remove Count NaN's
    count_nans = forecast_df["count"].isnull().sum(axis=0)
    baseline_nans = forecast_df["baseline"].isnull().sum(axis=0)
    print(
        "{} NaN values found in 'count' column. Dropping these from the dataframe.".format(
            count_nans
        )
    )
    print(
        "{} NaN values found in 'baseline' column. Dropping these from the dataframe.\n".format(
            baseline_nans
        )
    )
    forecast_df.dropna(inplace=True)

    # Make Baseline Values Non-Negative
    negative = len(forecast_df[forecast_df["baseline"] < 0]["baseline"])
    if negative > 0:
        print(
            "{} negative baseline values found. Setting these to zero.\n".format(
                negative
            )
        )
        forecast_df["baseline"] = forecast_df["baseline"].apply(
            lambda x: np.max([0, x])
        )
    else:
        print("All baseline predictions >= 0.\n")

    final_length = len(forecast_df["count"])
    print(
        "Data cleansing complete. {} rows removed from dataframe.".format(
            init_length - final_length
        )
    )

    copy_df = forecast_df
    return copy_df
