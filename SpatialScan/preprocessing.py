import numpy as np
import pandas as pd


def drop_anom(df: pd.DataFrame, N_sigma: float = 3) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, looks for anomalous counts, and sets to NaN. 
    the median and standard deviation of counts at each hour is determined, and used to 
    define a threshold for each hour equal to median + N_sigma*standard deviation. Any
    counts above that threshold is considered to be an anomaly and set to NaN.

    Args:
        df: Dataframe of SCOOT data
        N_sigma: Number of standard deviations to set threshold, lower will result in more
                millitant removal. Default is 3

    Returns:
        Dataframe of with outliers removed.

        """
    detectors = df["detector_id"].drop_duplicates().to_numpy()
    df_list = []
    for d, detector in enumerate(detectors, 1):
        dataset = df[df["detector_id"] == detector]

        dataset["hour"] = dataset["measurement_start_utc"].dt.hour.to_numpy()

        threshold = (
            dataset.groupby("hour").median()["n_vehicles_in_interval"]
            + N_sigma * dataset.groupby("hour").std()["n_vehicles_in_interval"]
        )

        for j in range(0, len(dataset)):
            if (
                dataset.iloc[j]["n_vehicles_in_interval"]
                > threshold[dataset.iloc[j]["hour"]]
            ):
                dataset.iloc[
                    j, dataset.columns.get_loc("n_vehicles_in_interval")
                ] = float("NaN")

        df_list.append(dataset)
        print("please wait: ", d, "/", len(detectors), "detectors", end="\r")
    DF = pd.concat(df_list)
    return DF


def reindex_and_drop(df: pd.DataFrame, percentage_missing: float = 20) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, and reindexes with the full time period. Any detectors
    which more missing values that the percentage specified by percentage_missing are dropped 
    from the total dataset

    Args:
        df: Dataframe of SCOOT data (prefereably after anomalies have been removed)
        percentage_missing: float percentage of missing values, above which drop detector

    Returns:
        Dataframe which has been reindexed for missing hours, filled with NaN's.

        """

    detectors = df["detector_id"].drop_duplicates().to_numpy()
    df_list = []
    detectors_removed = []
    for d, detector in enumerate(detectors, 1):

        dataset = df[df["detector_id"] == detector]

        dataset.index = dataset["measurement_end_utc"]

        T = pd.date_range(
            start=df["measurement_end_utc"].min(),
            end=df["measurement_end_utc"].max(),
            freq="H",
        )

        dataset = dataset.reindex(T)
        num_nan = dataset["n_vehicles_in_interval"].isna().sum()
        if num_nan > (len(dataset) * percentage_missing) / 100:
            detectors_removed.append(detector)
            continue

        dataset["measurement_end_utc"] = dataset.index
        dataset["measurement_start_utc"] = dataset[
            "measurement_end_utc"
        ] - np.timedelta64(1, "h")

        df_list.append(dataset)
        print("please wait: ", d, "/", len(detectors), "detectors", end="\r")

    print("detectors dropped: ", detectors_removed)
    DF = pd.concat(df_list)

    return DF


def data_preprocessor(
    df: pd.DataFrame,
    percentage_missing: float = 20,
    N_sigma: float = 3,
    repeats: int = 1,
    rolling_hours: int = 24,
    global_threshold: bool = False,
) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, performs anomaly removal, fill_and_drop, and then
    interpolates missing values. 

    Args:
        df: Dataframe of SCOOT data
        percentage_missing: float percentage of missing values, above which drop detector
        N_sigma: Number of standard deviations to set threshold
        repeats: integer number of repeats for which to recalcualte thresholds once previous anomallys
                have been removed, default 1
        rolling_hours: interger number of hours over which to calculate rolling median for threshold
        global_threshold: if True, global median used for threshold instead of rolling median

    Returns:
        Dataframe of interpolated values with detectors dropped for too many missing values.

        """
    detectors = df["detector_id"].drop_duplicates().to_numpy()
    i = 0
    detectors_removed = []

    T = pd.date_range(
        start=df["measurement_end_utc"].min(),
        end=df["measurement_end_utc"].max(),
        freq="H",
    )

    mux = pd.MultiIndex.from_product(
        [df["detector_id"].unique(), T], names=("detector_id", "measurement_end_utc")
    )

    df = df.set_index(["detector_id", "measurement_end_utc"]).reindex(mux)

    print("reindexing complete")

    for r in range(0, repeats):

        df.loc[:, "rolling_threshold"] = (
            df["n_vehicles_in_interval"].rolling(rolling_hours).median()
            + N_sigma * df["n_vehicles_in_interval"].std()
        )
        df["global_threshold"] = np.repeat(
            (
                df.median(level="detector_id")["n_vehicles_in_interval"]
                + N_sigma * (df.std(level="detector_id")["n_vehicles_in_interval"])
            ).to_numpy(),
            len(T),
        )
        df["rolling_threshold"] = df["rolling_threshold"].fillna(df["global_threshold"])
        if global_threshold:
            df["rolling_threshold"] = df["global_threshold"]

        df.loc[
            df["n_vehicles_in_interval"] > df["rolling_threshold"],
            ["n_vehicles_in_interval"],
        ] = float("NaN")
        print("calcualting thresholds, repeats: ", r + 1, end="\r")

    print("thresholds calculated")

    x = []
    for d in df.index.get_level_values("detector_id").unique():
        x.append([df.loc[d]["n_vehicles_in_interval"].isna().sum()] * len(df.loc[d]))
    print("detectors missing values dropped")

    x = np.array(x)
    x = x.flatten()

    df["Num_Missing"] = x

    df = df.drop(df[df["Num_Missing"] > ((len(T) * percentage_missing) / 100)].index)

    print("malfunctioning detectors dropped")

    # df["detector_id"]=df.index.get_level_values('detector_id')
    # df["measurement_end_utc"]=df.index.get_level_values('measurement_end_utc')
    df["measurement_start_utc"] = df.index.get_level_values(
        "measurement_end_utc"
    ) - np.timedelta64(1, "h")
    df["n_vehicles_in_interval"] = df["n_vehicles_in_interval"].interpolate(
        method="linear", limit_direction="both", axis=0
    )
    df["lon"] = df["lon"].interpolate(method="linear", limit_direction="both", axis=0)
    df["lat"] = df["lat"].interpolate(method="linear", limit_direction="both", axis=0)
    print("interpolation complete")

    return df.reset_index()


def data_preprocessor_slow(
    df: pd.DataFrame,
    percentage_missing: float = 20,
    N_sigma: float = 3,
    repeats: int = 1,
) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, performs anomaly removal, fill_and_drop, and then
    interpolates missing values. 

    Args:
        df: Dataframe of SCOOT data
        N_sigma: Number of standard deviations to set threshold
        percentage_missing: float percentage of missing values, above which drop detector

    Returns:
        Dataframe of interpolated values with detectors dropped for too many missing values.

        """
    detectors = df["detector_id"].drop_duplicates().to_numpy()
    df_list = []
    i = 0
    detectors_removed = []

    for detector in detectors:
        dataset = df[df["detector_id"] == detector]

        dataset.loc[:, "hour"] = dataset["measurement_start_utc"].dt.hour.to_numpy()

        for k in range(0, repeats):

            threshold = (
                dataset.groupby("hour").median()["n_vehicles_in_interval"]
                + N_sigma * dataset.groupby("hour").std()["n_vehicles_in_interval"]
            )

            # global_threshold = (
            #     dataset["n_vehicles_in_interval"].median()
            #     + N_sigma * dataset["n_vehicles_in_interval"].std()
            # )

            rolling_threshold = (
                dataset["n_vehicles_in_interval"].rolling(24).median()
                + 3 * dataset["n_vehicles_in_interval"].rolling(24).std()
            )
            rolling_threshold = rolling_threshold.fillna(method="backfill")
            rolling_threshold = rolling_threshold.to_numpy()

            for j in range(0, len(dataset)):
                if (
                    dataset.iloc[j]["n_vehicles_in_interval"]
                    > threshold[dataset.iloc[j]["hour"]]
                ):
                    dataset.iloc[
                        j, dataset.columns.get_loc("n_vehicles_in_interval")
                    ] = float("NaN")

                if dataset.iloc[j]["n_vehicles_in_interval"] > rolling_threshold[j]:
                    dataset.iloc[
                        j, dataset.columns.get_loc("n_vehicles_in_interval")
                    ] = float("NaN")

        dataset.index = dataset["measurement_end_utc"]

        T = pd.date_range(
            start=df["measurement_end_utc"].min(),
            end=df["measurement_end_utc"].max(),
            freq="H",
        )

        dataset = dataset.reindex(T)
        num_nan = dataset["n_vehicles_in_interval"].isna().sum()
        if num_nan > (len(dataset) * percentage_missing) / 100:
            detectors_removed.append(detector)
            continue

        dataset["n_vehicles_in_interval"].to_numpy()

        dataset["n_vehicles_in_interval"] = dataset[
            "n_vehicles_in_interval"
        ].interpolate(method="linear", limit_direction="both", axis=0)
        dataset["detector_id"] = dataset["detector_id"].interpolate(
            method="pad", limit_direction="both", axis=0, limit=1000
        )
        dataset["lon"] = dataset["lon"].interpolate(
            method="pad", limit_direction="both", axis=0, limit=1000
        )
        dataset["lat"] = dataset["lat"].interpolate(
            method="pad", limit_direction="both", axis=0, limit=1000
        )
        dataset["measurement_end_utc"] = dataset.index
        dataset["measurement_start_utc"] = dataset[
            "measurement_end_utc"
        ] - np.timedelta64(1, "h")

        df_list.append(dataset)
        i += 1
        print("please wait: ", i, "/", len(detectors), "detectors", end="\r")
    print("detectors dropped: ", detectors_removed)
    DF = pd.concat(df_list)
    DF = DF.fillna(method="backfill")

    return DF.drop(columns=["hour"]).reset_index(drop=True)
