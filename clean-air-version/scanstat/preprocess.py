"""Functionality to remove anomalies, and deal with missing data before scanning."""

import numpy as np
import pandas as pd

from geoalchemy2.shape import to_shape


def preprocessor(
    scoot_df: pd.DataFrame,
    percentage_missing: float = 20,
    max_anom_per_day: int = 1,
    n_sigma: float = 3,
    repeats: int = 1,
    rolling_hours: int = 24,
    global_threshold: bool = False,
) -> pd.DataFrame:

    """Takes a SCOOT dataframe, performs anomaly removal, fills missing readings, and then
    interpolates missing values if sufficiently few of them.

    Args:
        scoot_df: dataframe of SCOOT data
        percentage_missing: percentage of missing values, above which, drop detector
        n_sigma: number of standard deviations from the median to set anomaly threshold
        repeats: number of iterations for anomaly removal
        rolling_hours: number of previous hours used to calculate rolling median
        global_threshold: if True, global median used for threshold instead of rolling median

    Returns:
        Dataframe of interpolated values with detectors dropped for too many missing
        values or anomalies.
    """
    columns = [
        "detector_id",
        "lon",
        "lat",
        "location",
        "measurement_start_utc",
        "measurement_end_utc",
        "n_vehicles_in_interval",
    ]
    assert set(columns) <= set(scoot_df.columns)
    assert percentage_missing >= 0
    assert max_anom_per_day >= 0
    assert n_sigma >= 0
    assert repeats >= 0
    assert rolling_hours > 0 if not global_threshold else True

    # Convert location wkb to wkt, so can use groupby later on
    scoot_df["location"] = scoot_df["location"].apply(to_shape).apply(lambda x: x.wkt)

    # Convert dates to useful format
    scoot_df["measurement_start_utc"] = pd.to_datetime(
        scoot_df["measurement_start_utc"]
    )
    scoot_df["measurement_end_utc"] = pd.to_datetime(scoot_df["measurement_end_utc"])

    start_date = scoot_df["measurement_start_utc"].min()
    end_date = scoot_df["measurement_end_utc"].max()
    num_days = (end_date - start_date).days

    # Max allowable amount of anomalies without detector disposal
    max_anom = max_anom_per_day * num_days

    # Create array of times for which we want data
    end_times = pd.date_range(
        start=start_date + np.timedelta64(1, "h"), end=end_date, freq="H",
    )

    # Create Multi-index dataframe
    scoot_df = scoot_df.set_index(["detector_id", "measurement_end_utc"])

    if global_threshold:
        print(
            "Using the global median over {} days to remove outliers ...".format(
                num_days
            )
        )
    else:
        print(
            "Using the {}-day rolling median to remove outliers ...".format(
                rolling_hours
            )
        )
    print(
        "Using {} iterations to remove points outside of {} sigma from the median ...".format(
            repeats, n_sigma
        )
    )

    scoot_df = scoot_df.sort_values(["detector_id", "measurement_end_utc"])

    for rep in range(0, repeats):

        scoot_df["rolling_threshold"] = (
            scoot_df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .median()
            .values
            + n_sigma
            * scoot_df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .std()
            .values
        )

        scoot_df = scoot_df.join(
            scoot_df.median(level="detector_id")["n_vehicles_in_interval"]
            + n_sigma * (scoot_df.std(level="detector_id")["n_vehicles_in_interval"]),
            on=["detector_id"],
            rsuffix="anom",
        )

        scoot_df["global_threshold"] = scoot_df["n_vehicles_in_intervalanom"]
        scoot_df = scoot_df.drop(["n_vehicles_in_intervalanom"], axis=1)

        scoot_df["rolling_threshold"] = scoot_df["rolling_threshold"].fillna(
            scoot_df["global_threshold"]
        )

        if global_threshold:
            scoot_df["rolling_threshold"] = scoot_df["global_threshold"]

        scoot_df.loc[
            scoot_df["n_vehicles_in_interval"] > scoot_df["rolling_threshold"],
            ["n_vehicles_in_interval"],
        ] = float("NaN")

        print(
            "Calculating threshold(s): Iteration {} of {}...".format(rep + 1, repeats),
            end="\r",
        )

    # Calculate number of anomalies per detector
    scoot_df = scoot_df.join(
        scoot_df.isna().astype(int).sum(level="detector_id")["n_vehicles_in_interval"],
        on=["detector_id"],
        rsuffix="NaN",
    )
    scoot_df.rename({"n_vehicles_in_intervalNaN": "num_anom"}, axis=1, inplace=True)

    # Calculate original num of detectors inputted by user
    orig_set = set(scoot_df.index.get_level_values("detector_id"))
    orig_length = len(orig_set)

    # Drop detectors with too many anomalies
    print("\nDropping detectors with more than {} anomalies...".format(max_anom))
    scoot_df = scoot_df.drop(scoot_df[scoot_df["num_anom"] > max_anom].index)

    print("Filling in missing dates and times ...")
    remaining_detectors = scoot_df.index.get_level_values("detector_id").unique()
    mux = pd.MultiIndex.from_product(
        [remaining_detectors, end_times], names=("detector_id", "measurement_end_utc")
    )
    scoot_df = scoot_df.reindex(mux)

    # Find detectors with too much missing data
    detectors_to_drop = []
    for det in remaining_detectors:
        if (
            scoot_df.loc[det]["n_vehicles_in_interval"].isna().sum()
            > len(end_times) * 0.01 * percentage_missing
        ):
            detectors_to_drop.append(det)

    print(
        "Dropping detectors with sufficiently high amounts of missing data (>{}%)...".format(
            percentage_missing
        )
    )

    # If there are detectors to be dropped, remove them.
    if detectors_to_drop:
        scoot_df.drop(detectors_to_drop, level="detector_id", inplace=True)

    # Return drop information to user
    curr_set = set(scoot_df.index.get_level_values("detector_id"))
    curr_length = len(curr_set)
    print(
        "{} detectors dropped: {}\n".format(
            orig_length - curr_length, orig_set.difference(curr_set)
        )
    )

    # Linearly interpolate missing vehicle counts whilst still using multi_index
    # Fills backwards and forwards
    print(
        "Linearly interpolating between missing vehicle counts for remaining detectors ",
        "with less than {}% missing data...".format(percentage_missing),
    )
    scoot_df["n_vehicles_in_interval"] = scoot_df["n_vehicles_in_interval"].interpolate(
        method="linear", limit_direction="both", axis=0
    )

    # Create the remaining columns/fill missing row values
    scoot_df["measurement_start_utc"] = scoot_df.index.get_level_values(
        "measurement_end_utc"
    ) - np.timedelta64(1, "h")
    scoot_df = scoot_df.reset_index()
    scoot_df.sort_values(["detector_id", "measurement_end_utc"], inplace=True)

    # Fill missing lon, lat, rolling, global columns with existing values
    scoot_df = scoot_df.groupby("detector_id").apply(lambda x: x.ffill().bfill())

    # Re-Order Columns
    scoot_df = scoot_df[
        [
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
    ]

    print("Data processing complete.\n")
    return scoot_df
