"""Module containing Time Series Forecast functionality"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from scipy.optimize import minimize

import gpflow
from gpflow.utilities import print_summary
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from datetime import datetime



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

    """Time series forecast using Holt-Winters method.

    Args:
        df: Dataframe of 'processed' SCOOT data
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        alpha: Optimisation parameter
        beta: Optimisation parameter
        gamma: Optimisation parameter
        display: Boolean which determines whether to plot forecast.
        detectors: List of detectors to look at. Defaults to all.

    Returns:
        Dataframe forecast in same format as SCOOT input dataframe, with baseline
        counts instead of actual counts.
    """

    # Check parameter values
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= gamma <= 1

    # Get default detectors
    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        # Notation as in Expectation-Based Scan Statistic paper
        S = 1
        T = 1
        I = np.ones(24)
        one_det = df[df["detector_id"] == detector]

        # Use most recent days in the past to produce forecast
        one_det = one_det.sort_values(by=["measurement_end_utc"])
        past = one_det.tail(n=24 * days_in_past)

        # HW algorithm
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

        last_training_time = df["measurement_end_utc"].max()
        for j in range(0, days_in_future * 24):

            start = last_training_time + np.timedelta64(j, "h")
            end = last_training_time + np.timedelta64(j + 1, "h")

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
                "lon": one_det[one_det["detector_id"] == detector]["lon"].iloc[0],
                "lat": one_det[one_det["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": starttime,
                "measurement_end_utc": endtime,
                "n_vehicles_in_interval": baseline,
            }
        )
        framelist.append(df2)
    forecasts = pd.concat(framelist)

    if display:
        df_plot = forecasts.set_index("measurement_end_utc")
        for detector in detectors:
            df_plot[df_plot["detector_id"] == detector]["n_vehicles_in_interval"].plot()

    return forecasts

def HW_RSME(
    params: list, df: pd.DataFrame, days_in_past: int, days_in_future:int, detectors: list = None,
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
    count=[]
    baseline=[]
    for detector in detectors:
        S = 1
        T = 1
        I = np.ones(24)
        one_D = df[df["detector_id"] == detector]
        one_D = one_D.sort_values(by=["measurement_end_utc"])
        past = one_D
        RSME = []
        for i in range(0, len(past) - 1):
            h = i % 24

            r = i % (24*days_in_future)
            

            if(r==0):
                Sf=S
                Tf=T
                If=I
            if ((i > 24*days_in_past) and i < len(past)-24*days_in_future):
                b = (Sf + Tf) * If[h]
                baseline.append(b)
                count.append(past["n_vehicles_in_interval"].iloc[i])
                Snew = (alpha * (b / If[h])) + (1 - alpha) * (Sf + Tf)
                T = beta * (Snew - Sf) + (1 - beta) * Tf
                I[h] = gamma * (b / Snew) + (1 - gamma) * If[h]
                Sf = Snew

            c = past["n_vehicles_in_interval"].iloc[i]
            Snew = (alpha * (c / I[h])) + (1 - alpha) * (S + T)
            T = beta * (Snew - S) + (1 - beta) * T
            I[h] = gamma * (c / Snew) + (1 - gamma) * I[h]
            S = Snew
            

    RSME = np.sqrt(mean_squared_error(count, baseline))

    return RSME


def HW_SFE(
    params,
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,

    detectors: list = None,
) -> pd.DataFrame:
    
    alpha = params[0]
    beta = params[1]
    gamma = params[2]

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
    t_min = df["measurement_start_utc"].min()
    t_max = df["measurement_end_utc"].max()
    validation_start = t_min + np.timedelta64(days_in_past +1, "D")
    
    RSME=[]
    
    while(validation_start < t_max-np.timedelta64(days_in_future, "D")):
        
        #print(validation_start, t_max, end ="\r")

        if detectors is None:
            detectors = df["detector_id"].drop_duplicates().to_numpy()

        train_data = df[df["measurement_end_utc"] <= validation_start + np.timedelta64(days_in_future, "D")]
        validation_data = train_data[train_data["measurement_end_utc"] > validation_start]

        train_data = train_data[train_data["measurement_end_utc"] <= validation_start]

        y = holt_winters(
                train_data,
                days_in_past,
                days_in_future,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                detectors=detectors)
        
        validation_data= validation_data[validation_data["detector_id"].isin(detectors)]
        RSME.append(np.sqrt(mean_squared_error(validation_data["n_vehicles_in_interval"], y["n_vehicles_in_interval"])))
        validation_start = validation_start + np.timedelta64(1, "D")
    return np.array(RSME).mean()



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

        a = 0.06
        b = 0.02
        c = 0.4

        params = minimize(
            HW_SFE,
            [a, b, c],
            args=(one_D, days_in_past, days_in_future, [detector]),
            method="L-BFGS-B",
            bounds=[(0,0.15), (0, 0.15), (0.1, 0.5)],
            options={"ftol": 10},
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
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    kern=None
) -> pd.DataFrame:

    """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args:
        df: Dataframe of processed SCOOT data.
        days_in_past: Integer past days to train forecast on
        days_in_future: Days in future to produce a baseline estimate for
        method: Forecast method to use for baseline, default is "HW" for Holt-Winters.
                Options: "HW", "HWO", "MALD", "GP", "LSTM"
        detectors: List of detectors to look produce forecasts for. Default behaviour
                   produces forecasts for all detectors present in input dataframe.
        alpha: Holt-Winter parameter
        beta: Holt-winters parameter
        gamma: Holt-winters parameter
        kern: GP kernel if method="GP" used. Default available.

    Returns:
        forecast_df: Dataframe of SCOOT vehicle counts and baseline estimates
    """

    # Drop useless columns
    assert set(['rolling_threshold', 'global_threshold']) <= set(df.columns)
    df = df.drop(
        ["rolling_threshold", "global_threshold"], axis=1
    )
    assert days_in_future > 0
    assert days_in_past > 0

    t_min = df["measurement_start_utc"].min()
    t_max = df["measurement_end_utc"].max()

    print("Input dataframe contains data spanning {} to {}.".format(t_min, t_max))

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    # Organise dates of train/forecast/analysis
    prediction_start = t_max - np.timedelta64(days_in_future * 24, "h")

    train_data = df[df["measurement_end_utc"] <= prediction_start]
    actual_counts = df[df["measurement_end_utc"] > prediction_start]

    avail_past_days = (prediction_start - t_min).days

    # Print sanity checks
    if avail_past_days < days_in_past:
        print(
            "Input dataframe only contains {} days worth of data before the prediction period.".format(
                avail_past_days
            ),
            "Setting days_in_past = {}.".format(avail_past_days),
        )
        forecast_data_start = t_min
    else:
        forecast_data_start = prediction_start - np.timedelta64(days_in_past, "D")

    print("Using data from {} to {}, to build {} forecasting model.\n".format(
            forecast_data_start, prediction_start, method
        )
    )
    print("Forecasting counts between {} and {} for {} detectors.".format(
            prediction_start, t_max, len(detectors)
        )
    )

    # Select forecasting method
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

    if method == "GP":
        y = GP_forecast(
            train_data,
            days_in_past=days_in_past,
            days_in_future=days_in_future,
            detectors=detectors,
            kern = kern
        )
    print("Forecasting complete.")

    # Merge actual_count dataframe with forecast dataframe, carry out checks
    # and return.
    forecast_df = y.merge(
        actual_counts,
        on=[
            "detector_id",
            "lon",
            "lat",
            "measurement_start_utc",
            "measurement_end_utc",
        ],
        how="left",
    )
    forecast_df.rename(
        columns={
            "n_vehicles_in_interval_x": "baseline",
            "n_vehicles_in_interval_y": "count",
        },
        inplace=True,
    )

    # Add check for Nans cleanse
    count_nans = forecast_df["count"].isnull().sum(axis=0)
    baseline_nans = forecast_df["baseline"].isnull().sum(axis=0)
    assert count_nans == 0
    assert baseline_nans == 0

    # Make Baseline Values Non-Negative
    negative= len(forecast_df[forecast_df["baseline"] < 0]["baseline"])
    if negative > 0:
        print("Setting {} negative baseline values to zero.\n".format(negative))
        forecast_df["baseline"] = forecast_df["baseline"].apply(
            lambda x: np.max([0, x])
        )

    return forecast_df


def forecast_plot(df: pd.DataFrame, detector: str = None):
    """Function that plots the Count against the forecasted Baseline
        
        Args:
            df: Dataframe with Time, Count and Baseline columns
            detector: String of detector name, if none detector chosen at random"""

    detectors = df["detector_id"].drop_duplicates()
    if detector is None:
        detector = detectors.sample(n=1).to_numpy()[0]

    df_d = df[df["detector_id"] == detector]
    print(detector)
    df_d = df_d.sort_values("measurement_end_utc")
    fig= plt.figure(figsize=(15,8))
    ax = fig.add_subplot()
    if "prediction_variance" in df_d.columns:

        #df_d["measurement_end_utc"]=df_d["measurement_end_utc"].astype('O')
        ax.plot(df_d["measurement_end_utc"], df_d["baseline"], label="baseline")
        ax.plot(df_d["measurement_end_utc"], df_d["count"], label="count")
        ax.fill_between(
            df_d["measurement_end_utc"],
            df_d["baseline"] + 3*np.sqrt(df_d["prediction_variance"]),
            df_d["baseline"] - 3*np.sqrt(df_d["prediction_variance"]),
            color="C0",
            alpha=0.3, label= "3$\sigma$")
        fig.autofmt_xdate()
    else:
        ax=df_d.plot(x="measurement_end_utc", y=["baseline", "count"])
    plt.legend()

    plt.show()


def CB_plot(df: pd.DataFrame):

    """Function that plots Counts/Baseline as a 3D plot  with detector locations. Counts are 
        shown by size of point, and C/B is shown using a colourmap
        
        Args:
            Dataframe with Time, Count and Baseline columns"""

    df_format = df

    forecast_t_min = df_format["measurement_start_utc"].min()
    forecast_t_max = df_format["measurement_end_utc"].max()

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
    fig.suptitle("Forecast from {} to {}".format(forecast_t_min, forecast_t_max))
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

        # plot_model(
        #     model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
        # )

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


def GP_forecast(
    df: pd.DataFrame,
    days_in_past: int = 2,
    days_in_future: int = 1,
    detectors: list = None,
    kern = None
) -> pd.DataFrame:

    """Forecast using Gaussian Processes 
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

        dataset = df[df["detector_id"] == detector].tail(n=24 * days_in_past)

        Y = dataset["n_vehicles_in_interval"].to_numpy().reshape(-1, 1)
        Y = Y.astype(float)
        X = np.arange(1, len(Y) + 1, dtype=float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        if(kern is None):

            kern_pD = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_pW = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_SE = gpflow.kernels.SquaredExponential()
            kern_W = gpflow.kernels.White()
            #kern_M = gpflow.kernels.Matern52()

            kern_pD.period.assign(24.0)
            # kern_pD.base_kernel.variance.assign(10)
            kern_pW.period.assign(168.0)
            # kern_pW.base_kernel.variance.assign(10)

            k = kern_pD * kern_pW + kern_SE + kern_W
        else:
            k=kern

        m = gpflow.models.GPR(data=(X, y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=100)
        )

        print("please wait: ", i, "/", len(detectors), end="\r")

        ## generate test points for prediction
        xx = np.linspace(
            len(Y) + 1, len(Y) + (days_in_future * 24) + 1, (days_in_future * 24)
        ).reshape(
            (days_in_future * 24), 1
        )  # test points must be of shape (N, D)

        ## predict mean and variance of latent GP at test points
        mean, var = m.predict_f(xx)

        # reverse min_max scaler
        testPredict = scaler.inverse_transform(mean)
        testVar = scaler.inverse_transform(var)

        # find the time period for our testPredictions
        prediction_start = dataset["measurement_end_utc"].max()

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
                "prediction_variance" : testVar.flatten(),
                "99_upper": 3*np.sqrt(testVar.flatten()) + testPredict.flatten(),
                "99_lower": 3*np.sqrt(testVar.flatten()) - testPredict.flatten(),
            }
        )

        framelist.append(df2)

    return pd.concat(framelist)
