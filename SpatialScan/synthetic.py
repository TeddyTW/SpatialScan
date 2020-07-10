import numpy as np
import pandas as pd
from numpy.random import normal
import random

from SpatialScan.scan import scan
from SpatialScan.timeseries import count_baseline, CB_plot
from SpatialScan.region import cleanse_forecast_data, plot_region_by_rank
from SpatialScan.results import database_results


def synthetic_detector(
    Y: pd.Series, noise_percentage: float = 10, dow_percentage: float = 5
) -> np.array:
    """
    Creates a synthetic count based on real counts for the means of simualting outbreak. Takes
    in the counts for a single detectors, and outputs a 24 hour periodic trace, with amplitude equal 
    to the 90th percentile of actual count data with added noise. Also includes day of the week component
    
    Args:
        Y: Series of count data for a given detector
        noise_percentage: percentage of random noise to add to the snthetic data, default 10%
        dow_percentage: strength of day of week component. 10% default means it varies trace by 10%

    Returns: np.array of synthetic counts
    """

    Y = Y.to_numpy()
    X = np.arange(0, len(Y))
    noise = normal(0, noise_percentage / 100, len(Y))
    S = (
        np.percentile(Y, 90)
        * abs((np.sin((np.pi * X / 24)) ** 2 + noise))
        * ((dow_percentage / 100) * np.sin((np.pi * X / 168)) ** 2 + 1)
    + np.percentile(Y, 3)).astype(int)
    return S


def synthetic_SCOOT(
    df: pd.DataFrame, noise_percentage: float = 10, dow_percentage: float = 10
) -> pd.DataFrame:
    """
    Creates a synthetic SCOOT dataframe based on a real input dataframe.
    Args:
        df: input SCOOT dataframe (preprocessed for best results)
        noise_percentage: percentage of random noise to add to the snthetic data, default 10%
        dow_percentage: strength of day of week component. 10% default means it varies trace by 10%

    Returns: Dataframe in SCOOT format, with synthetic data"""

    DF = df.set_index(["detector_id", "measurement_end_utc"])
    X = DF.groupby(level="detector_id")["n_vehicles_in_interval"].apply(
        lambda x: synthetic_detector(
            x, noise_percentage=noise_percentage, dow_percentage=dow_percentage
        )
    )
    DF["n_vehicles_in_interval"] = np.hstack(X.to_numpy())
    DF = DF.reset_index()
    return DF

def select_detectors(
    detector_locations: pd.DataFrame, k: int, s_center_lon: float, s_center_lat: float
) -> pd.DataFrame:

    """Selects detectors for outbreak
    Args:
        detector_locations: dataframe of detector locations
        k: number of detectors affected
        s_center_lon: lon co-ordinate of outbreak centre
        s_center_lat: lat co-ordinate of outbreak centre
    Returns:
        Dataframe of k detectors and their locations.
        """

    detector_locations["dist"] = detector_locations.apply(
        lambda x: (x.lon - s_center_lon) ** 2 + (x.lat - s_center_lat) ** 2, axis=1
    )
    detector_locations.sort_values("dist", axis=0, inplace=True)

    return detector_locations.head(k)


def add_outbreak(
    synthetic_data: pd.DataFrame,
    outbreak_detectors: np.ndarray,
    severity: float,
    outbreak_duration: int,
) -> tuple:

    """Manipulates the synthetic data by adding an outbreak to certain specified detectors

    Args:
        synthetic_data: Generated from `synthetic_SCOOT()`
        outbreak_detectors: Detectors chosen to increase/decrease in signal
        severity: Controls the gradient of activity change
        outbreak_duration: Number of most recent days in synthetic_data to manipulate
    Returns
        manipulated dataframe
        Timestamp of start of outbreak
    """

    t_max = synthetic_data["measurement_end_utc"].max()
    outbreak_start = t_max - np.timedelta64(outbreak_duration, "D")

    print("Start of outbreak: {}".format(outbreak_start))

    affected = synthetic_data[
        (synthetic_data["detector_id"].isin(outbreak_detectors))
        & (synthetic_data["measurement_start_utc"] >= outbreak_start)
    ].copy()

    affected["hours_since_outbreak"] = affected["measurement_end_utc"].apply(
        lambda x: (x - outbreak_start).days * 24 + (x - outbreak_start).seconds // 3600
    )

    weights_dict = (
        affected.groupby(["detector_id"]).n_vehicles_in_interval.sum().to_dict()
    )

    norm = sum(weights_dict.values())

    weights_dict.update((x, y / norm) for x, y in weights_dict.items())

    # print(weights_dict)
    assert np.isclose(sum(weights_dict.values()), 1)

    affected["simulated_count"] = affected.apply(
        lambda x: np.random.poisson(
            x.n_vehicles_in_interval
            * (
                1
                + severity / 100 * x.hours_since_outbreak * weights_dict[x.detector_id]
            )
        ),
        axis=1,
    )

    res_df = synthetic_data.merge(
        affected,
        how="left",
        on=[
            "detector_id",
            "measurement_start_utc",
            "measurement_end_utc",
            "n_vehicles_in_interval",
            "lon",
            "lat",
            "rolling_threshold",
            "global_threshold",
            "Num_Anom",
            "Num_Missing",
        ],
    )

    res_df["simulated_count"].fillna(res_df["n_vehicles_in_interval"], inplace=True)

    res_df["n_vehicles_in_interval"] = res_df["simulated_count"]

    res_df.drop(["hours_since_outbreak", "simulated_count"], axis=1, inplace=True)

    return res_df, outbreak_start


def simulate_outbreak(synthetic_data, severity, k_min, k_max, outbreak_duration=7):

    """Main function for simulating the outbreak. Randomly chooses epicentre, severity
    and size.

    Args:
        synthetic_data: synthetic data
        severity:
        k_min: lower bound of affected detectors
        k_max: upper_bound of affected detectors
    Returns:
        outbreak_df
        dataframe of affected detectors
        start of outbreak
    """

    detector_locations = synthetic_data.drop_duplicates(subset="detector_id")[
        ["detector_id", "lon", "lat"]
    ]
    num_rows = len(detector_locations)

    random.seed(0)

    k = random.randint(k_min, k_max)
    rand_row = detector_locations.iloc[random.randint(1, num_rows)]

    s_center_lon = rand_row.lon
    s_center_lat = rand_row.lat
    print(
        "Outbreak centred at ({}, {}) affecting {} detectors".format(
            s_center_lon, s_center_lat, k
        )
    )

    detector_df = select_detectors(detector_locations, k, s_center_lon, s_center_lat)
    detectors = detector_df["detector_id"].to_numpy()
    print(detectors)

    outbreak_df, start = add_outbreak(
        synthetic_data, detectors, severity, outbreak_duration=outbreak_duration
    )

    return outbreak_df, detector_df, start


def results_builder(
    outbreak_df: pd.DataFrame,
    outbreak_detectors: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    method: str,
    grid_partition: int,
    scan_type: str,
):

    """Builds daily results from the scan up over a number of days, determined by
    the size of outbreak_dataframe. e.g. if outbreak_dataframe has N days worth of
    data, the first `days_in_past` number of days will be dedicated to forecasting.
    This function will then return scan results for the remaining N - `days_in_past`
    days, using the settings described below. Basically - simulating what will be stored
    in the database over a period of time.

    Args:
        outbreak_dataframe: simulated from `simulate_outbreak()`
        detectors_data: dataframe of affected detectors
        start_time: outbreak_start time
        days_in_past: as in count_baseline
        days_in_future: as in count_baseline
        method: as in count_baseline
        grid_partition: as in scan()
        scan_type as in scan()

    Returns:
        Dataframe of results spanning (len(outbreak_dataframe) - days_in_past) days worth of analysis
        Dataframe of highest scoring regions per day
        """

    t_min = outbreak_df["measurement_start_utc"].min()
    t_max = outbreak_df["measurement_end_utc"].max()

    # Get outbreak characteristics
    num_outbreak_detectors = len(set(outbreak_detectors['detector_id']))
    ob_x_min = outbreak_detectors['lon'].min()
    ob_x_max = outbreak_detectors['lon'].max()
    ob_y_min = outbreak_detectors['lat'].min()
    ob_y_max = outbreak_detectors['lat'].max()

    total_num_days = (t_max - t_min).days
    print("Total number of days in dataframe: ", total_num_days)

    first_analysis_day = (
        t_min + np.timedelta64(days_in_past, "D") + np.timedelta64(days_in_future, "D")
    )

    num_forecast_days = (t_max - first_analysis_day).days + 1

    print(
        "Producing forecasts and scans for {} days in total.\n".format(
            num_forecast_days
        )
    )

    dataframe_list = []

    daily_highest_scoring_regions = {}
    today = first_analysis_day
    for i in range(num_forecast_days):

        print(
            "\nAnalysis day: {}. Looking back at last {} hours.".format(
                today, 24 * days_in_future
            )
        )

        available_today = outbreak_df[
            outbreak_df["measurement_end_utc"] <= today
        ].copy()

        forecast_df = count_baseline(
            available_today,
            days_in_past=days_in_past,
            days_in_future=days_in_future,
            method=method,
        )

        forecast_df = cleanse_forecast_data(forecast_df)

        CB_plot(forecast_df)

        res_df = scan(forecast_df, grid_partition=grid_partition, scan_type=scan_type)

        plot_region_by_rank(
            0, res_df, forecast_df, plot_type="count", add_legend=False
        )

        #  Return Highest Scoring region here
        highest_region = res_df.iloc[0][
            [
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "t_min",
                "t_max",
                "l_score_EBP",
                "l_score_000",
                "l_score_025",
                "l_score_050",
                "posterior_bbayes",
            ]
        ].to_dict()

        # Add some Spatial analysis
        x_min = highest_region['x_min']
        x_max = highest_region['x_max']
        y_min = highest_region['y_min']
        y_max = highest_region['y_max']
        
        
        num_detectors_in_highest_region = len(set(outbreak_df[(outbreak_df['lon'].between(x_min, x_max)) &\
                                                             (outbreak_df['lat'].between(y_min, y_max))].detector_id))
        
        overlap_x_min = max([x_min, ob_x_min])
        overlap_x_max = min([x_max, ob_x_max])
        overlap_y_min = max([y_min, ob_y_min])
        overlap_y_max = min([y_max, ob_y_max])
        
        num_detectors_in_highest_region_and_true = len(set(outbreak_df[(outbreak_df['lon'].between(overlap_x_min, overlap_x_max)) &\
                                                             (outbreak_df['lat'].between(overlap_y_min, overlap_y_max))].detector_id))
        
        # Calculate Spatial Precision and Recall
        precision = num_detectors_in_highest_region_and_true / num_detectors_in_highest_region
        recall = num_detectors_in_highest_region_and_true / num_outbreak_detectors
        
        highest_region["precision"] = precision
        highest_region["recall"] = recall
        highest_region["day"] = today
        
        # Append to list of dataframes
        daily_highest_scoring_regions[i] = highest_region

        # Send results to database
        database_df = database_results(res_df)

        # Updates data correctly with most reliable average likelihood scores.
        # i.e. today is wednesday, and days_in_future = 2
        # We are getting scores for monday and tuesday, and append it in a list.
        # Now, the next day, we get scores for tuesday and wednesday. We throw away the old tuesday,
        # and keep the new one.
        if len(dataframe_list) - (days_in_future - 1) >= 0:
            dataframe_list = dataframe_list[
                : len(dataframe_list) - (days_in_future - 1)
            ]

        days_dict = dict(
            iter(database_df.groupby(database_df["start_time_utc"].dt.day))
        )

        for j in range(days_in_future):
            forecast_day = (today - np.timedelta64(days_in_future - j, "D")).day
            dataframe_list.append(days_dict[forecast_day])

        today += np.timedelta64(1, "D")

    # Return list of highest scoring regions too - add to plot?
    return (
        pd.concat(dataframe_list, ignore_index=True),
        pd.DataFrame.from_dict(daily_highest_scoring_regions, "index"),
    )
