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
    percentile: float = 99,
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

            global_threshold = (
                dataset["n_vehicles_in_interval"].median()
                + N_sigma * dataset["n_vehicles_in_interval"].std()
            )

            for j in range(0, len(dataset)):
                if (
                    dataset.iloc[j]["n_vehicles_in_interval"]
                    > threshold[dataset.iloc[j]["hour"]]
                ):
                    dataset.iloc[
                        j, dataset.columns.get_loc("n_vehicles_in_interval")
                    ] = float("NaN")

                if dataset.iloc[j]["n_vehicles_in_interval"] > global_threshold:
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

def plot_processing(raw_scoot_df: pd.DataFrame, processed_scoot_df:pd.DataFrame, detector:str = None):

    """ Helper function to compare raw and processed time-series of a detector.
    Args:
        raw_scoot_df: Dataframe directly from SCOOT query
        processed_df: Dataframe returned `data_preprocessor`
        detector: Detctor's time-series to plot. If not specified, chosen at random.

    """

    if detector is None:
        detector = raw_scoot_df['detector_id'].sample(1).iloc[0]
    print(detector)
    
    d_scoot = raw_scoot_df[raw_scoot_df['detector_id'] == detector]['n_vehicles_in_interval'].to_numpy()
    t_scoot = raw_scoot_df[raw_scoot_df['detector_id'] == detector]['measurement_end_utc'].to_numpy()
    d_proc = processed_scoot_df[processed_scoot_df['detector_id'] == detector]['n_vehicles_in_interval'].to_numpy()
    t_proc = processed_scoot_df[processed_scoot_df['detector_id'] == detector]['measurement_end_utc'].to_numpy()
    
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.plot(t_scoot, d_scoot, label="Raw")
    plt.plot(t_proc, d_proc, label="Processed")
    plt.xlabel("Date")
    plt.ylabel("Vehicle Count")
    plt.legend()
    return