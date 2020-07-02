"""Module containing Time Series Forecast functionality"""
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.colors as colors
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.express as px


def data_preprocessor(df: pd.DataFrame, percentage_missing: float = 20, sigma: float = 3) -> pd.DataFrame:

    """Function takes a SCOOT dataframe, interpolates any missing values, and returns
    a dataframe with missing values interpolated. Any detectors missing more than 
    percentage_missing datapoints will be dropped for having to few values.

    Args:
        df: Dataframe of SCOOT data
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

        dataset["hour"] = dataset["measurement_start_utc"].dt.hour.to_numpy()

        threshold = (
            dataset.groupby("hour").median()["n_vehicles_in_interval"]
            + sigma* dataset.groupby("hour").std()["n_vehicles_in_interval"]
        )

        for j in range(0, len(dataset)):
            if (
                dataset.iloc[j]["n_vehicles_in_interval"]
                > threshold[dataset.iloc[j]["hour"]]
            ):
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
        ].interpolate(method="linear", limit_direction="forward", axis=0)
        dataset["detector_id"] = dataset["detector_id"].interpolate(
            method="pad", limit_direction="forward", axis=0
        )
        dataset["lon"] = dataset["lon"].interpolate(
            method="pad", limit_direction="forward", axis=0
        )
        dataset["lat"] = dataset["lat"].interpolate(
            method="pad", limit_direction="forward", axis=0
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

    return DF.reset_index(drop=True)


def MA(df: pd.DataFrame, detector: str, past_days: int) -> float:
    """Function calculates the average number of vehicles over the last past_days

    Args:
        df: Dataframe of SCOOT data
        detector: string representing detector ID
        past_days: integer number of days to look back and average over

    Returns:
        Average number of vehicles measured at this detector in the previous past_days days

        """

    one_D = df[df["detector_id"] == detector]
    one_D = one_D.sort_values(by=["measurement_end_utc"])
    return one_D.tail(n=24 * past_days)["n_vehicles_in_interval"].mean()


def MALD(df: pd.DataFrame, detector: str, past_days: int, hour: int) -> float:

    """Proportion of counts at this hour, looking back at all historical data

    Args:
        df: Dataframe of SCOOT data
        detector: string representing detector ID
        past_days: integer number of days to look back and average over
        hour: integer hour of the day for which to calculate MALD

    Returns:
        Proportion of vehicles expected at this hour

        """

    one_D = df[df["detector_id"] == detector].tail(n=24 * past_days)
    beta = (
        one_D[one_D["measurement_end_utc"].dt.hour == hour][
            "n_vehicles_in_interval"
        ].sum()
    ) / (one_D["n_vehicles_in_interval"].sum())
    return beta


def MALDforecast(
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    display: bool = False,
    detectors: list = None,
) -> pd.DataFrame:

    """Average forcast using MALD hourly proportions to account of hourly variation

    Args:
        df: Dataframe of SCOOT data
        detectors: List of detectors to look at
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        display: boolean which determines whether to plot forecast

    Returns:
        Dataframe forecast in same format as SCOOT input dataframe

        """
    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        meanV = MA(df, detector, days_in_past)
        pred = []
        endtime = []
        starttime = []
        for i in range(1, 24 * days_in_future + 1):
            end = df["measurement_end_utc"].to_numpy()[-1] + np.timedelta64(i, "h")
            hour = (df["measurement_end_utc"].dt.hour.to_numpy()[-1] + i) % 24
            start = df["measurement_start_utc"].to_numpy()[-1] + np.timedelta64(i, "h")
            beta = MALD(df, detector, days_in_past, hour)
            endtime.append(end)
            starttime.append(start)
            pred.append(beta * 24 * meanV)

        df2 = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": df[df["detector_id"] == detector]["lon"].iloc[0],
                "lat": df[df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": starttime,
                "measurement_end_utc": endtime,
                "n_vehicles_in_interval": pred,
            }
        )
        framelist.append(df2)
    DF = pd.concat(framelist)

    if display:
        df_plot = DF.set_index("measurement_end_utc")
        for detector in detectors:
            df_plot[df_plot["detector_id"] == detector]["n_vehicles_in_interval"].plot()

    return DF


def holt_winters(
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    display: bool = False,
    detectors: list = None,
) -> pd.DataFrame:

    """Average forecast using Holt-Winters method

    Args: 
        df: Dataframe of SCOOT data
        detectors: List of detectors to look at
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        display: boolean which determines whether to plot forecast
        alpha, beta, gamma: optimisation parameters

    Returns:
        Dataframe forecast in same format as SCOOT input dataframe

        """

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        S = 1
        T = 1
        I = np.ones(24)
        one_D = df[df["detector_id"] == detector]
        one_D = one_D.sort_values(by=["measurement_end_utc"])
        past = one_D.tail(n=24 * days_in_past)
        for i in range(0, len(past)):
            h = i % 24
            c = past["n_vehicles_in_interval"].iloc[i]
            Snew = (alpha * (c / I[h])) + (1 - alpha) * (S + T)
            T = beta * (Snew - S) + (1 - beta) * T
            I[h] = gamma * (c / Snew) + (1 - gamma) * I[h]
            S = Snew

        baseline = []
        endtime = []
        starttime = []
        for j in range(0, days_in_future * 24):
            end = df["measurement_end_utc"].to_numpy()[-1] + np.timedelta64(j, "h")
            start = df["measurement_start_utc"].to_numpy()[-1] + np.timedelta64(j, "h")
            h = j % 24
            b = (S + T) * I[h]
            baseline.append(b)
            endtime.append(end)
            starttime.append(start)

            Snew = (alpha * (b / I[h])) + (1 - alpha) * (S + T)
            T = beta * (Snew - S) + (1 - beta) * T
            I[h] = gamma * (b / Snew) + (1 - gamma) * I[h]
            S = Snew

        df2 = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": df[df["detector_id"] == detector]["lon"].iloc[0],
                "lat": df[df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": starttime,
                "measurement_end_utc": endtime,
                "n_vehicles_in_interval": baseline,
            }
        )
        framelist.append(df2)
    DF = pd.concat(framelist)

    if display:
        df_plot = DF.set_index("measurement_end_utc")
        for detector in detectors:
            df_plot[df_plot["detector_id"] == detector]["n_vehicles_in_interval"].plot()

    return DF


def count_baseline(
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    method: str = "HW",
    detectors: list = None,
    alpha: float = 0.06,
    beta: float = 0.02,
    gamma: float = 0.6,
) -> pd.DataFrame:

    """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args:
        df: Dataframe of SCOOT data
        days_in_past: Integer past days to train forecast one
        days_in_future: Days in future produce a baseline too and record count for
        method: Forecast method to use for baseline, default is "HW" for Holt-Winters, option for MLAD
        detectors: List of detectors to look at

    Returns:
        Dataframe of counts and baseline along with detector data

        """

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    prediction_start = df["measurement_end_utc"].iloc[-1] - np.timedelta64(
        days_in_future * 24, "h"
    )

    train_data = df[df["measurement_end_utc"] <= prediction_start]
    test_data = df[df["measurement_end_utc"] > prediction_start]

    if method == "HW":
        y = holt_winters(
            train_data,
            days_in_past,
            days_in_future,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            detectors=detectors,
        )
    if method == "MALD":
        y = MALDforecast(train_data, days_in_past, days_in_future, detectors=detectors)

    if method == "LSTM":
        y = LSTM_forecast(
            df,
            days_in_past=days_in_past,
            days_in_future=days_in_future,
            detectors=detectors,
        )

    sd = []

    for detector in detectors:

        sd.append(test_data[test_data["detector_id"] == detector])

    sample_test_data = pd.concat(sd)

    Y = y.merge(
        sample_test_data,
        on=[
            "lon",
            "lat",
            "measurement_end_utc",
            "detector_id",
            "measurement_start_utc",
        ],
        how="left",
    )
    Y = Y.rename(
        columns={
            "n_vehicles_in_interval_x": "baseline",
            "n_vehicles_in_interval_y": "count",
        }
    )

    return Y


def CB_plot(df: pd.DataFrame):

    """Function that plots Counts/Baseline as a 3D plot  with detector locations. Counts are 
        shown by size of point, and C/B is shown using a colourmap
        
        Args:
            Dataframe with Time, Count and Baseline columns"""

    df_format = df
    df_format["C/B"] = df_format["count"] / df_format["baseline"]
    df_format["hour_from_start"] = (
        df_format["measurement_end_utc"] - df_format["measurement_end_utc"].min()
    )
    df_format["hour_from_start"] = df_format["hour_from_start"].astype(
        dtype="timedelta64[h]"
    )
    offset = colors.DivergingNorm(vmin=0.5, vcenter=1, vmax=2)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        df_format["lon"],
        df_format["lat"],
        df_format["hour_from_start"],
        c=df_format["C/B"],
        s=df_format["count"] * 0.1,
        norm=offset,
        cmap="coolwarm",
    )
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_zlabel("Hours")
    fig.colorbar(p)
    plt.show()


# convert an array of values into a datset of X values to use for predictions (made
# up of sets of day's in the past), and Y values to prediction (sets of days in the
# future)
def create_dataset(dataset: pd.DataFrame, look_back: int = 24, look_forward: int = 24):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - look_forward + 1, look_forward):
        # a will be one set of X data to be used as inputs to the NN
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        # by will be one set of Y data, to be predicted by the NN and then
        # compared to for means of training
        b = dataset[(i + look_back) : (i + look_back + look_forward), 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


def LSTM_forecast(
    df: pd.DataFrame,
    days_in_past: int = 2,
    days_in_future: int = 1,
    detectors: list = None,
) -> pd.DataFrame:

    """Forecast using LSTM Neural Network 
    Args: 
        df: Dataframe of SCOOT data
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        detectors: List of detectors to look at


    Returns:
        Dataframe forecast in same format as SCOOT input dataframe

        """

    # extract numpy array of detector ID's
    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()
    framelist = []

    i = 0
    for detector in detectors:
        i += 1

        dataset = df[df["detector_id"] == detector]
        dataset.index = dataset["measurement_end_utc"]

        # get full temporal range, to be intepolated over for missing vlaues
        T = pd.date_range(
            start=df["measurement_end_utc"].min(),
            end=df["measurement_end_utc"].max(),
            freq="H",
        )

        # fill in missing hours with interpolation
        dataset = dataset.reindex(T)
        num_nan = dataset["n_vehicles_in_interval"].isna().sum()
        if num_nan > int(len(dataset) / 6):
            print(detector)
            continue
        dataset["n_vehicles_in_interval"] = dataset[
            "n_vehicles_in_interval"
        ].interpolate(method="linear", limit_direction="forward", axis=0)
        dataset["detector_id"] = dataset["detector_id"].interpolate(
            method="pad", limit_direction="forward", axis=0
        )
        dataset["measurement_end_utc"] = dataset.index

        # split up dataset into train and test. The test dataset is the final
        # days in future + days in the past (days in past to feed into the inputs
        # to make predictions on days in the future). the training days are used to
        # train the weights of the NN
        prediction_start = df["measurement_end_utc"].max() - np.timedelta64(
            (days_in_future + days_in_past) * 24, "h"
        )
        dataset_train = dataset[dataset["measurement_end_utc"] <= prediction_start]
        dataset_test = dataset[dataset["measurement_end_utc"] > prediction_start]

        # datasets converted to arrays then rescaled with a MinMax scaler
        train = dataset_train["n_vehicles_in_interval"].to_numpy()
        test = dataset_test["n_vehicles_in_interval"].to_numpy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train.reshape(-1, 1))
        test = scaler.fit_transform(test.reshape(-1, 1))

        # use create to reshape the data into sucessive sets of X and Y features and
        # labels that can be fed into the LSTM NN
        X_train, Y_train = create_dataset(
            train, look_back=days_in_past * 24, look_forward=24 * days_in_future
        )
        X_test, Y_test = create_dataset(
            test, look_back=days_in_past * 24, look_forward=24 * days_in_future
        )

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # initialise neural network, LSTM with day in the past inputs and
        # days in the future outputs
        model = Sequential()
        model.add(LSTM(8, input_shape=(1, 24 * days_in_past)))
        model.add(Dense(24 * days_in_future))
        model.compile(loss="mean_squared_error", optimizer="adam")

        # train LSTM with our training data!
        model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=0)

        # use our testing data to make predictions on Y_test
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        print("please wait: ", i, "/", len(detectors), end="\r")

        # reverse min_max scaler
        testPredict = scaler.inverse_transform(testPredict)

        # find the time period for our testPredictions
        prediction_start = dataset_test["measurement_end_utc"].max() - np.timedelta64(
            days_in_future * 24, "h"
        )
        t = pd.date_range(
            start=prediction_start,
            end=prediction_start + np.timedelta64(24 * days_in_future - 1, "h"),
            freq="H",
        )

        # organise data into dataframe similar to the SCOOT outputs
        df2 = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": df[df["detector_id"] == detector]["lon"].iloc[0],
                "lat": df[df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": t,
                "measurement_end_utc": t + np.timedelta64(1, "h"),
                "n_vehicles_in_interval": testPredict.flatten(),
            }
        )

        framelist.append(df2)

        # clear our Keras session
        clear_session()

    return pd.concat(framelist)
