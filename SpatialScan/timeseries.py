"""Module containing Time Series Forecast functionality"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy.optimize import minimize


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
            end = df["measurement_end_utc"].to_numpy()[-1] + np.timedelta64(j + 1, "h")
            start = df["measurement_start_utc"].to_numpy()[-1] + np.timedelta64(
                j + 1, "h"
            )
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
                "lon": one_D[one_D["detector_id"] == detector]["lon"].iloc[0],
                "lat": one_D[one_D["detector_id"] == detector]["lat"].iloc[0],
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


def HW_RSME(
    params: list, df: pd.DataFrame, days_in_past: int, detectors: list = None,
) -> float:

    """Calcualte Root-Mean Squared error on historical training data for holt winters.
    This function is minimised to optimise the hyperparameters of the Holt-Winters

    Args: 
        params: [alpha, beta, gamma] paramaters as a list
        df: Dataframe of SCOOT data
        days_in_past: Integer number of previous days to calculate for
        detectors: List of detectors to look at

    Returns:
        RMSE for HW method with histroical data for given params


    """

    alpha = params[0]
    beta = params[1]
    gamma = params[2]

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
        RSME = []
        for i in range(0, len(past) - 1):
            h = i % 24

            if i > len(past) - 24 * 8:
                RSME = np.append(
                    RSME,
                    (past["n_vehicles_in_interval"].iloc[i + 1] - ((S + T) * I[h]))
                    ** 2,
                )

            c = past["n_vehicles_in_interval"].iloc[i]
            Snew = (alpha * (c / I[h])) + (1 - alpha) * (S + T)
            T = beta * (Snew - S) + (1 - beta) * T
            I[h] = gamma * (c / Snew) + (1 - gamma) * I[h]
            S = Snew

        RSME = np.sqrt(RSME.mean())

        return RSME


def HW_opt(
    df: pd.DataFrame, days_in_past: int, days_in_future: int, detectors: list = None,
):

    """Average forecast using Holt-Winters method, where parameters have been optimised
    by minimising HW_RSME with historical data

    Args: 
        df: Dataframe of SCOOT data
        detectors: List of detectors to look at
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for

    Returns:
        Dataframe forecast in same format as SCOOT input dataframe

        """

    framelist = []

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    for d, detector in enumerate(detectors, 1):

        one_D = df[df["detector_id"] == detector]

        a = 0.1
        b = 0.1
        c = 0.1

        params = minimize(
            HW_RSME,
            [a, b, c],
            args=(one_D, days_in_past),
            method="SLSQP",
            bounds=[(0, 1), (0, 1), (0, 1)],
            options={"ftol": 3},
        )["x"]

        alpha = params[0]
        beta = params[1]
        gamma = params[2]

        framelist.append(
            holt_winters(
                one_D, days_in_past, days_in_future, alpha=alpha, beta=beta, gamma=gamma
            )
        )
        print("please wait: ", d, "/", len(detectors), end="\r")

    DF = pd.concat(framelist)

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

    df = df.drop(
        ["rolling_threshold", "global_threshold", "Num_Anom", "Num_Missing"], axis=1
    )

    t_min = df["measurement_start_utc"].min()
    t_max = df["measurement_end_utc"].max()

    print("Input dataframe contains data spanning {} to {}.".format(t_min, t_max))

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    prediction_start = df["measurement_end_utc"].iloc[-1] - np.timedelta64(
        days_in_future * 24, "h"
    )

    train_data = df[df["measurement_end_utc"] <= prediction_start]
    test_data = df[df["measurement_end_utc"] > prediction_start]

    avail_past_days = int(len(train_data["measurement_end_utc"].unique()) / 24)
    if avail_past_days < days_in_past:
        print(
            "Input dataframe only contains {} days worth of data before the prediction period.".format(
                avail_past_days
            ),
            "Setting days_in_past = {}.".format(avail_past_days),
        )

    print(
        "Using data from {} to {}, to forecast counts between {} and {} for {} detectors using {} method...".format(
            t_min, prediction_start, prediction_start, t_max, len(detectors), method
        )
    )

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

    if method == "HWO":
        y = HW_opt(train_data, days_in_past, days_in_future, detectors=detectors,)

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

    print("Forecasting complete.")

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
    offset = colors.TwoSlopeNorm(vmin=0.5, vcenter=1, vmax=2)
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

        plot_model(
            model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
        )

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
