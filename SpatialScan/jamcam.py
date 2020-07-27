"""Module containing utility functionality to convert Jamcam data into usable format
with the remaining code."""

from json import JSONDecodeError
import datetime

import requests
import pandas as pd
import numpy as np


def get_raw_jamcam_data(
    jamcam_locations_df: pd.DataFrame,
    start_year: int = 2020,
    start_month: int = 6,
    start_day: int = 1,
    end_year: int = 2020,
    end_month: int = 6,
    end_day: int = 2,
    detection_class="all",
) -> pd.DataFrame:
    """Scrape API for raw data for camera given in jamcam_locations_df between
    the dats passed.
    Args:
        jamcam_locations_df: df with camera_ids, borough_names, and their locations
        start_date:
        end_date:
    Return:
        Raw counts spanning start_date to end_date for cameras in input df.
    """

    start = datetime.datetime(start_year, start_month, start_day)
    end = datetime.datetime(end_year, end_month, end_day)

    days = pd.date_range(start=start, end=end, freq="d")

    ids = jamcam_locations_df["camera_id"].apply(lambda x: x[8:])

    res_dict = {}
    num_entries = 0

    print(
        "Fetching {} data for {} cameras between {} and {}".format(
            detection_class, len(ids), start, end
        )
    )

    for i in range(len(days) - 1):

        start_y, start_m, start_d = days[i].year, days[i].month, days[i].day
        end_y, end_m, end_d = days[i + 1].year, days[i + 1].month, days[i + 1].day

        start_time_string = "{0:04d}-{1:02d}-{2:02d}T00%3A00%3A00.000Z".format(
            start_y, start_m, start_d
        )
        end_time_string = "{0:04d}-{1:02d}-{2:02d}T00%3A00%3A00.000Z".format(
            end_y, end_m, end_d
        )

        for cam_id in ids:
            try:
                jreq = requests.get(
                    "https://urbanair.turing.ac.uk/api/v1/jamcams/raw?camera_id={}".format(
                        cam_id
                    ),
                    "&detection_class={}&starttime={}&endtime={}".format(
                        detection_class, start_time_string, end_time_string
                    ),
                    auth=("admin", "x7WBcuRtrgK8255rPZcB"),
                ).json()
            except JSONDecodeError:
                print("Could not get data.")
                continue

            lon, lat, borough_name = jamcam_locations_df[
                jamcam_locations_df["camera_id"] == "JamCams_{}".format(cam_id)
            ].iloc[0][["longitude", "latitude", "borough_name"]]
            size = len(jreq)
            if not isinstance(jreq, list):
                continue
            else:
                for j in range(0, size):
                    res_dict[num_entries] = {
                        "camera_id": jreq[j]["camera_id"],
                        "lon": lon,
                        "lat": lat,
                        "counts": jreq[j]["counts"],
                        "detection_class": jreq[j]["detection_class"],
                        "measurement_start_utc": jreq[j]["measurement_start_utc"],
                        "borough_name": borough_name,
                    }
                    num_entries += 1

    # Make huge dataframe
    res_df = pd.DataFrame.from_dict(res_dict, "index")

    print("Complete.")

    return res_df


def jamcam_hourly_aggregate(raw_jamcam_df: pd.DataFrame):

    """Aggregate Raw dataframe from `get_raw_jamcam_data()` into hourly aggregates
    for each detection class. Output format is same as SCOOT. Note that counts are given
    as `n_vehicle_in_interval` even when referring to a person.

    Args:
        Raw Jamcam Dataframe
    Returns:
        Aggregated Jamcam Dataframe
    """

    # Drop duplicates
    orig_length = len(raw_jamcam_df["camera_id"])
    raw_jamcam_df.drop_duplicates(inplace=True)
    curr_length = len(raw_jamcam_df["camera_id"])

    print("Dropped {} duplicate rows from raw data.".format(orig_length - curr_length))

    # First, convert dates
    raw_jamcam_df["measurement_start_utc"] = pd.to_datetime(
        raw_jamcam_df["measurement_start_utc"]
    )

    # Set some new columns for the groupby step
    raw_jamcam_df["year"] = raw_jamcam_df["measurement_start_utc"].dt.year
    raw_jamcam_df["month"] = raw_jamcam_df["measurement_start_utc"].dt.month
    raw_jamcam_df["day"] = raw_jamcam_df["measurement_start_utc"].dt.day
    raw_jamcam_df["hour"] = raw_jamcam_df["measurement_start_utc"].dt.hour

    # Aggregate the counts
    agg_df = (
        raw_jamcam_df.groupby(
            [
                "camera_id",
                "year",
                "month",
                "day",
                "hour",
                "detection_class",
                "lon",
                "lat",
                "borough_name",
            ]
        )
        .counts.sum()
        .reset_index()
    )

    # Reconstruct the dates and drop the helper columns
    agg_df["measurement_start_utc"] = pd.to_datetime(
        dict(year=agg_df.year, month=agg_df.month, day=agg_df.day, hour=agg_df.hour)
    )
    agg_df["measurement_end_utc"] = agg_df["measurement_start_utc"].apply(
        lambda x: x + np.timedelta64(1, "h")
    )
    agg_df.drop(["year", "month", "day", "hour"], axis=1, inplace=True)

    agg_df.rename(
        {"camera_id": "detector_id", "counts": "n_vehicles_in_interval"},
        inplace=True,
        axis=1,
    )

    return agg_df


def format_jamcam_hourly_average(
    jamcam_data: pd.DataFrame, jamcam_locs: pd.DataFrame
) -> pd.DataFrame:

    """Get Jamcam data counts from hourly averages"""

    # First drop duplicates
    data = jamcam_data.copy()
    locs = jamcam_locs.copy()

    orig_length = len(data["camera_id"])
    data.drop_duplicates(inplace=True)
    curr_length = len(data["camera_id"])

    print("Dropped {} duplicate rows.".format(orig_length - curr_length))

    # Convert jamcam data into average hourly counts
    # 360 is clearly not the magic number here.
    data["n_vehicles_in_interval"] = data["avg"].apply(lambda x: int(360 * x))

    # Convert camera Ids into matching formats
    locs["short_id"] = locs["camera_id"].apply(lambda x: x[12:])
    data["camera_id"] = "{:.5f}".format(data["camera_id"])
    locs = locs.drop("camera_id", axis=1)

    # Fine the dates of which the data spans
    data["measurement_end_utc"] = pd.to_datetime(data["date_trunc"])
    data["measurement_start_utc"] = data["measurement_end_utc"] - np.timedelta64(1, "h")

    # Drop useless columns
    ss_df = data.drop(["avg", "date_trunc"], axis=1)

    # Merge with location data so that returned dataframe has lon, lat
    ss_df = ss_df.merge(locs, how="left", left_on="camera_id", right_on="short_id")

    # re-order columns
    ss_df = ss_df[
        [
            "camera_id",
            "longitude",
            "latitude",
            "measurement_start_utc",
            "measurement_end_utc",
            "detection_class",
            "n_vehicles_in_interval",
            "name",
            "borough_name",
        ]
    ]

    ss_df.sort_values("measurement_end_utc", inplace=True)

    # Rename for consitency with further work
    ss_df.rename(
        {"camera_id": "detector_id", "latitude": "lat", "longitude": "lon"},
        axis=1,
        inplace=True,
    )

    return ss_df
