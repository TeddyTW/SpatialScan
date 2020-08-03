import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

def frequency_alarm(df_d: pd.DataFrame) -> float:
    """Function that returns the false alarm probability for a given detector, for use in groupbys
    Args: 
        df_d: dataframe for single detector
    Returns:
        false alarm probability as float"""
    X=df_d.reset_index()
    X=X.drop(columns=["detector_id", "measurement_end_utc"])
    x=X.index.to_numpy()
    y=X.to_numpy().flatten()
    ls=LombScargle(x, y)
    period=np.linspace(15, 30, 300)
    pmax=ls.power(1/period).max()
    return ls.false_alarm_probability(pmax)

def data_preprocessor(
    df: pd.DataFrame,
    percentage_missing: float = 20,
    max_anom_per_day: int = 1,
    N_sigma: float = 3,
    repeats: int = 1,
    rolling_hours: int = 24,
    fap_max = float = 1e-40,
    global_threshold: bool = False,
) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, performs anomaly removal, fill_and_drop, and then
    interpolates missing values. 

    Args:
        df: Dataframe of SCOOT data
        percentage_missing: float percentage of missing values, above which drop detector
        N_sigma: Number of standard deviations to set threshold
        repeats: integer number of repeats for which to recalculate thresholds once previous anomallys
                have been removed, default 1
        rolling_hours: interger number of hours over which to calculate rolling median for threshold
        global_threshold: if True, global median used for threshold instead of rolling median

    Returns:
        Dataframe of interpolated values with detectors dropped for too many missing values.

        """
    columns = ['detector_id', 'lon', 'lat', 'measurement_start_utc',
               'measurement_end_utc', 'n_vehicles_in_interval']
    assert set(columns) <= set(df.columns)
    assert percentage_missing >= 0
    assert max_anom_per_day >= 0
    assert N_sigma >= 0
    assert repeats >= 0
    assert rolling_hours > 0 if not global_threshold else True

    # Convert dates to useful format
    df['measurement_start_utc'] = pd.to_datetime(df['measurement_start_utc'])
    df['measurement_end_utc'] = pd.to_datetime(df['measurement_end_utc'])

    start_date = df["measurement_start_utc"].min()
    end_date = df["measurement_end_utc"].max()
    num_days = (end_date - start_date).days

    # Max allowable amount of anomalies without detector disposal
    max_anom = max_anom_per_day * num_days

    # Create array of times for which we want data
    end_times = pd.date_range(start=start_date + np.timedelta64(1, 'h'), end=end_date, freq="H",)

    # Create Multi-index dataframe
    df = df.set_index(["detector_id", "measurement_end_utc"])

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
            repeats, N_sigma
        )
    )

    df = df.sort_values(["detector_id", "measurement_end_utc"])

    for r in range(0, repeats):

        df["rolling_threshold"] = (
            df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .median()
            .values
            + N_sigma
            * df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .std()
            .values
        )

        df = df.join(
            df.median(level="detector_id")["n_vehicles_in_interval"]
            + N_sigma * (df.std(level="detector_id")["n_vehicles_in_interval"]),
            on=["detector_id"],
            rsuffix="anom",
        )

        df["global_threshold"] = df["n_vehicles_in_intervalanom"]
        df = df.drop(["n_vehicles_in_intervalanom"], axis=1)

        df["rolling_threshold"] = df["rolling_threshold"].fillna(df["global_threshold"])

        if global_threshold:
            df["rolling_threshold"] = df["global_threshold"]

        df.loc[
            df["n_vehicles_in_interval"] > df["rolling_threshold"],
            ["n_vehicles_in_interval"],
        ] = float("NaN")

        print(
            "Calculating threshold(s): Iteration {} of {}...".format(r + 1, repeats),
            end="\r",
        )

    # Calculate number of anomalies per detector
    df = df.join(
        df.isna().astype(int).sum(level="detector_id")["n_vehicles_in_interval"],
        on=["detector_id"],
        rsuffix="NaN",
    )
    df.rename({'n_vehicles_in_intervalNaN': 'num_anom'}, axis=1, inplace=True)

    # Calculate original num of detectors inputted by user
    orig_set = set(df.index.get_level_values("detector_id"))
    orig_length = len(orig_set)

    # Drop detectors with too many anomalies
    print("\nDropping detectors with more than {} anomalies...".format(max_anom))
    df = df.drop(df[df["num_anom"] > max_anom].index)

    print("Filling in missing dates and times ...")
    remaining_detectors = df.index.get_level_values("detector_id").unique()
    mux = pd.MultiIndex.from_product(
        [remaining_detectors, end_times], names=("detector_id", "measurement_end_utc")
    )
    df = df.reindex(mux)

    # Find detectors with too much missing data
    detectors_to_drop = []
    for det in remaining_detectors:
        if df.loc[det]["n_vehicles_in_interval"].isna().sum() > len(end_times) * 0.01 * percentage_missing:
            detectors_to_drop.append(det)

    print(
        "Dropping detectors with sufficiently high amounts of missing data (>{}%)...".format(
            percentage_missing
        )
    )
    df.drop(detectors_to_drop, level='detector_id', inplace=True)

    # Linearly interpolate missing vehicle counts whilst still using multi_index
    # Fills backwards and forwards
    print(
        "Linearly interpolating between missing vehicle counts for remaining detectors ",
        "with less than {}% missing data...".format(percentage_missing),
    )
    df["n_vehicles_in_interval"] = df["n_vehicles_in_interval"].interpolate(
        method="linear", limit_direction="both", axis=0
    )

    df = df.join(df.groupby(level="detector_id")["n_vehicles_in_interval"].apply(lambda x: frequency_alarm(x)),
            on=["detector_id"],
            rsuffix="X")

    df.rename({'n_vehicles_in_intervalX': 'fap'}, axis=1, inplace=True)

    print("\nDropping detectors with low periodicity...")
    df = df.drop(df[df["fap"] > fap_max].index)

    # Return drop information to user
    curr_set = set(df.index.get_level_values("detector_id"))
    curr_length = len(curr_set)
    print(
        "{} detectors dropped: {}\n".format(
            orig_length - curr_length, orig_set.difference(curr_set)
        )
    )

    # print(ls)

    # Create the remaining columns/fill missing row values
    df["measurement_start_utc"] = df.index.get_level_values("measurement_end_utc") - np.timedelta64(1, "h")
    df = df.reset_index()
    df.sort_values(["detector_id", "measurement_end_utc"], inplace=True)

    # Fill missing lon, lat, rolling, global columns with existing values
    df = df.groupby('detector_id').apply(lambda x: x.ffill().bfill())



    # Re-Order Columns
    df = df[['detector_id', 'lon', 'lat', 'measurement_start_utc',
              'measurement_end_utc', 'n_vehicles_in_interval',
              'rolling_threshold', 'global_threshold', 'fap']]

    print("Data processing complete.\n")
    return df


def plot_processing(
    raw_scoot_df: pd.DataFrame, processed_scoot_df: pd.DataFrame, detector: str = None
):

    """ Helper function to compare raw and processed time-series of a detector.
    Args:
        raw_scoot_df: Dataframe directly from SCOOT query
        processed_df: Dataframe returned `data_preprocessor`
        detector: Detctor's time-series to plot. If not specified, chosen at random.

    """

    if detector is None:
        detector = raw_scoot_df["detector_id"].sample(1).iloc[0]
    print(detector)

    d_scoot = raw_scoot_df[raw_scoot_df["detector_id"] == detector][
        "n_vehicles_in_interval"
    ].to_numpy()
    t_scoot = raw_scoot_df[raw_scoot_df["detector_id"] == detector][
        "measurement_end_utc"
    ].to_numpy()
    d_proc = processed_scoot_df[processed_scoot_df["detector_id"] == detector][
        "n_vehicles_in_interval"
    ].to_numpy()
    t_proc = processed_scoot_df[processed_scoot_df["detector_id"] == detector][
        "measurement_end_utc"
    ].to_numpy()
    global_thresh = processed_scoot_df[processed_scoot_df["detector_id"] == detector][
        "global_threshold"
    ].to_numpy()
    rolling_thresh = processed_scoot_df[processed_scoot_df["detector_id"] == detector][
        "rolling_threshold"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.plot(t_scoot, d_scoot, label="Raw")
    plt.plot(t_proc, d_proc, label="Processed")
    plt.plot(
        t_proc, global_thresh, label="Global Threshold", linestyle="dashed", alpha=0.5
    )
    plt.plot(
        t_proc, rolling_thresh, label="Rolling Threshold", linestyle="dashed", alpha=0.5
    )
    plt.xlabel("Date")
    plt.ylabel("Vehicle Count")
    plt.legend()
    return

def jam_preprocessor(
    df: pd.DataFrame,
    percentage_missing: float = 20,
    max_anom: int = 30,
    N_sigma: float = 3,
    repeats: int = 1,
    rolling_hours: int = 16,
    global_threshold: bool = False) -> pd.DataFrame: 

    """Function takes a JamCam dataframe, performs anomaly removal, fill_and_drop, and then
    interpolates missing values. 

    Args:
        df: Dataframe of SCOOT data
        N_sigma: Number of standard deviations to set threshold
        percentage_missing: float percentage of missing values, above which drop detector

    Returns:
        Dataframe of interpolated values with detectors dropped for too many missing values.

        """

    #reindex

    start_date = pd.to_datetime(df["measurement_end_utc"].min())
    end_date = pd.to_datetime(df["measurement_end_utc"].max())

    num_days = (end_date - start_date).days

    N_days=(end_date-start_date).days
    T = pd.date_range(start=start_date, periods=20-start_date.hour, freq="H")
    start_of_day = T[-1] + np.timedelta64(9, "h")
    t=pd.date_range(start=start_of_day, end=start_of_day+np.timedelta64(15, "h"), freq="H")
    df["measurement_end_utc"]=df["measurement_end_utc"].astype('datetime64[ns]')
    for d in range(0, N_days):
        t = pd.date_range(start=start_of_day, end=start_of_day+np.timedelta64(15, "h"), freq="H").to_numpy()

        T=np.append(T, t)
        start_of_day= start_of_day + np.timedelta64(1, "D")

    T = np.array(T)
    T=pd.DatetimeIndex(T)
    dets=df["detector_id"].unique()
    mux = pd.MultiIndex.from_product(
            [dets, T], names=("detector_id", "measurement_end_utc")
        )
    
    df = df.set_index(["detector_id", "measurement_end_utc"])
    
    if global_threshold:
        print(
            "Using the global median over {} days to remove outliers ...".format(
                num_days
            )
        )
        print(
            "Using {} iteration(s) to remove points outside of {} sigma from the global median ...".format(
                repeats, N_sigma
            )
        )
    else:
        print("Using the {}-day rolling median to remove outliers ...".format(rolling_hours))
        print(
            "Using {} iterations to remove points outside of {} sigma from the rolling median ...".format(
                repeats, N_sigma
            )
        )

    df = df.sort_values(["detector_id", "measurement_end_utc"])

    for r in range(0, repeats):

        df["rolling_threshold"] = (
            df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .median()
            .values
            + N_sigma
            * df.groupby(level="detector_id")["n_vehicles_in_interval"]
            .rolling(window=rolling_hours)
            .std()
            .values
        )

        #         df.loc[:, "rolling_threshold"] = (
        #             df["n_vehicles_in_interval"].rolling(rolling_hours).median()
        #             + N_sigma * df["n_vehicles_in_interval"].std()
        #         )

        df = df.join(
            df.median(level="detector_id")["n_vehicles_in_interval"]
            + N_sigma * (df.std(level="detector_id")["n_vehicles_in_interval"]),
            on=["detector_id"],
            rsuffix="anom",
        )

        df["global_threshold"] = df["n_vehicles_in_intervalanom"]
        df = df.drop(["n_vehicles_in_intervalanom"], axis=1)

        df["rolling_threshold"] = df["rolling_threshold"].fillna(df["global_threshold"])
        if global_threshold:
            df["rolling_threshold"] = df["global_threshold"]

        df.loc[
            df["n_vehicles_in_interval"] > df["rolling_threshold"],
            ["n_vehicles_in_interval"],
        ] = float("NaN")
        print(
            "Calculating threshold(s): Iteration {} of {}...".format(r + 1, repeats),
            end="\r",
        )

    print("\nThreshold(s) calculated.\n")

    df = df.join(
        df.isna().astype(int).sum(level="detector_id")["n_vehicles_in_interval"],
        on=["detector_id"],
        rsuffix="NaN",
    )

    df["Num_Anom"] = df["n_vehicles_in_intervalNaN"]
    df = df.drop(["n_vehicles_in_intervalNaN"], axis=1)

    orig_set = set(df.index.get_level_values("detector_id"))
    orig_length = len(orig_set)

    print("Dropping detectors with more than {} anomalies...".format(max_anom))
    df = df.drop(df[df["Num_Anom"] > max_anom].index)


    df=df.reindex(mux)
    
    x = []
    for d in df.index.get_level_values("detector_id").unique():
        x.append([df.loc[d]["n_vehicles_in_interval"].isna().sum()] * len(df.loc[d]))

    x = np.array(x)
    x = x.flatten()

    df["Num_Missing"] = x

    # T = pd.date_range(start=start_date, end=end_date, freq="H",)

    # mux = pd.MultiIndex.from_product(
    #     [dets, T], names=("detector_id", "measurement_end_utc")
    # )
    
    print(
        "Dropping detectors with sufficiently high amounts of missing data (>{}%)...".format(
            percentage_missing
        )
    )
    df = df.drop(df[df["Num_Missing"] > ((len(T) * percentage_missing) / 100)].index)

    curr_set = set(df.index.get_level_values("detector_id"))
    curr_length = len(curr_set)
    print(
        "{} detectors dropped: {}\n".format(
            orig_length - curr_length, orig_set.difference(curr_set)
        )
    )

    # T = pd.date_range(start=start_date, end=end_date, freq="H",)

    # mux = pd.MultiIndex.from_product(
    #     [dets, T], names=("detector_id", "measurement_end_utc")
    # )

    #df=df.reindex(mux)


    #interpolate

    df["measurement_start_utc"] = df.index.get_level_values(
    "measurement_end_utc"
    ) - np.timedelta64(1, "h")
    df["n_vehicles_in_interval"] = df["n_vehicles_in_interval"].astype("float").interpolate(
        method="linear", limit_direction="both", axis=0
    )

    # T = pd.date_range(start=start_date, end=end_date, freq="H",)

    # mux = pd.MultiIndex.from_product(
    #     [dets, T], names=("detector_id", "measurement_end_utc")
    # )

    #df=df.reindex(mux)
    df = df.reset_index()

    df.sort_values(["detector_id", "measurement_end_utc"], inplace=True)

    df["lon"] = df["lon"].interpolate(method="pad", limit_direction="both", axis=0)
    df["lat"] = df["lat"].interpolate(method="pad", limit_direction="both", axis=0)
    df["measurement_start_utc"] = df["measurement_end_utc"] - np.timedelta64(1, "h")

    return df