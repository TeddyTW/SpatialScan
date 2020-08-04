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
import astropy as ap

import tensorflow as tf
from scipy.optimize import minimize

import gpflow
from gpflow.utilities import print_summary
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from datetime import datetime

from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)


def holt_winters(
    proc_df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.1,
    detectors: list = None,
) -> pd.DataFrame:

    """Time series forecast using Holt-Winters method.

    Args:
        proc_df: Dataframe of 'processed' SCOOT data
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        alpha: Optimisation parameter
        beta: Optimisation parameter
        gamma: Optimisation parameter
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
        detectors = proc_df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        # Notation as in Expectation-Based Scan Statistic paper
        smooth = 1
        trend = 1
        hod = np.ones(24)
        one_det = proc_df[proc_df["detector_id"] == detector]

        # Use most recent days in the past to produce forecast
        one_det = one_det.sort_values(by=["measurement_end_utc"])
        past = one_det.tail(n=24 * days_in_past)

        # HW algorithm
        for i in range(0, len(past)):
            hour = i % 24
            count = past["n_vehicles_in_interval"].iloc[i]
            smooth_new = (alpha * (count / hod[hour])) + (1 - alpha) * (smooth + trend)
            trend = beta * (smooth_new - smooth) + (1 - beta) * trend
            hod[hour] = gamma * (count / smooth_new) + (1 - gamma) * hod[hour]
            smooth = smooth_new

        baseline = []
        endtime = []
        starttime = []

        last_training_time = proc_df["measurement_end_utc"].max()
        for j in range(0, days_in_future * 24):

            start = last_training_time + np.timedelta64(j, "h")
            end = last_training_time + np.timedelta64(j + 1, "h")

            hour = j % 24
            base = (smooth + trend) * hod[hour]
            baseline.append(base)
            endtime.append(end)
            starttime.append(start)

            smooth_new = (alpha * (base / hod[hour])) + (1 - alpha) * (smooth + trend)
            trend = beta * (smooth_new - smooth) + (1 - beta) * trend
            hod[hour] = gamma * (base / smooth_new) + (1 - gamma) * hod[hour]
            smooth = smooth_new

        forecasts = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": one_det[one_det["detector_id"] == detector]["lon"].iloc[0],
                "lat": one_det[one_det["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": starttime,
                "measurement_end_utc": endtime,
                "n_vehicles_in_interval": baseline,
                "baseline_upper": baseline,
                "baseline_lower": baseline,
            }
        )
        framelist.append(forecasts)
    return pd.concat(framelist)


def create_dataset(dataset: pd.DataFrame, look_back: int = 24, look_forward: int = 24):
    """convert an array of values into a dataset of X values to use for predictions (made
    up of sets of day's in the past), and Y values to prediction (sets of days in the
    future)

    Args:
        look_back: Number of hours in past used for forecast
        look_forward: Number of hours in future to produce a for forecast for

    Returns:
        formatted np.array for use in LSTM
    """
    data_x, data_y = [], []
    for i in range(0, len(dataset) - look_back - look_forward + 1, look_forward):
        # a will be one set of X data to be used as inputs to the NN
        set_a = dataset[i : (i + look_back), 0]
        data_x.append(set_a)
        # by will be one set of Y data, to be predicted by the NN and then
        # compared to for means of training
        set_b = dataset[(i + look_back) : (i + look_back + look_forward), 0]
        data_y.append(set_b)
    return np.array(data_x), np.array(data_y)


def lstm_forecast(
    proc_df: pd.DataFrame,
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
        detectors = proc_df["detector_id"].drop_duplicates().to_numpy()
    framelist = []


    for det_num, detector in enumerate(detectors, 1):

        dataset = proc_df[proc_df["detector_id"] == detector]
        dataset.index = dataset["measurement_end_utc"]

        # split up dataset into train and test. The test dataset is the final
        # days in future + days in the past (days in past to feed into the inputs
        # to make predictions on days in the future). the training days are used to
        # train the weights of the NN
        prediction_start = proc_df["measurement_end_utc"].max() - np.timedelta64(
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
        x_train, y_train = create_dataset(
            train, look_back=days_in_past * 24, look_forward=24 * days_in_future
        )
        x_test, y_test = create_dataset(
            test, look_back=days_in_past * 24, look_forward=24 * days_in_future
        )

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        # initialise neural network, LSTM with day in the past inputs and
        # days in the future outputs
        model = Sequential()
        model.add(LSTM(8, input_shape=(1, 24 * days_in_past)))
        model.add(Dense(24 * days_in_future))
        model.compile(loss="mean_squared_error", optimizer="adam")

        # train LSTM with our training data!
        model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)

        # plot_model(
        #     model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
        # )

        # use our testing data to make predictions on Y_test
        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)

        print("please wait: ", det_num, "/", len(detectors), end="\r")

        # reverse min_max scaler
        test_predict = scaler.inverse_transform(test_predict)

        # find the time period for our testPredictions
        prediction_start = dataset_test["measurement_end_utc"].max() - np.timedelta64(
            days_in_future * 24, "h"
        )
        forecast_period = pd.date_range(
            start=prediction_start,
            end=prediction_start + np.timedelta64(24 * days_in_future - 1, "h"),
            freq="H",
        )

        # organise data into dataframe similar to the SCOOT outputs
        forecast_df = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": proc_df[proc_df["detector_id"] == detector]["lon"].iloc[0],
                "lat": proc_df[proc_df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": forecast_period,
                "measurement_end_utc": forecast_period + np.timedelta64(1, "h"),
                "n_vehicles_in_interval": test_predict.flatten(),
                "baseline_upper": test_predict.flatten(),
                "baseline_lower": test_predict.flatten(),
            }
        )

        framelist.append(forecast_df)

        # clear our Keras session
        clear_session()
    return pd.concat(framelist)


def gp_forecast(
    proc_df: pd.DataFrame,
    days_in_past: int = 2,
    days_in_future: int = 1,
    detectors: list = None,
    kern: gpflow.kernels = None,
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
        detectors = proc_df["detector_id"].drop_duplicates().to_numpy()
    framelist = []


    for det_num, detector in enumerate(detectors, 1):

        dataset = proc_df[proc_df["detector_id"] == detector].tail(n=24 * days_in_past)

        Y = dataset["n_vehicles_in_interval"].to_numpy().reshape(-1, 1)
        Y = Y.astype(float)
        X = np.arange(1, len(Y) + 1, dtype=float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        if kern is None:

            kern_pd = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_pw = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_se = gpflow.kernels.SquaredExponential()
            kern_w = gpflow.kernels.White()
            # kern_M = gpflow.kernels.Matern52()

            kern_pd.period.assign(24.0)
            # kern_pD.base_kernel.variance.assign(10)
            kern_pw.period.assign(168.0)
            # kern_pW.base_kernel.variance.assign(10)

            k = kern_pd * kern_pw + kern_se + kern_w
        else:
            k = kern

        model = gpflow.models.GPR(data=(X, y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()

        # training_loss = model.training_loss_closure() # compile=True (default): compiles using tf.function
        # opt = tf.optimizers.Adam()

        # for step in range(500):
        #     opt.minimize(training_loss, model.trainable_variables)
        #     print(model.maximum_log_likelihood_objective(), model.log_marginal_likelihood())

        # #model=train_sensor_model(X, y, k, opt, maxiter=100)
        # #simple_training_loop(X, y, model, opt, maxiter=100, logging_freq=10)

        try:
            
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=500),
            )
        except:
            print(detector, " Covariance matrix not invertible, skipping to next detector")
            del model
            continue

        print("please wait: ", det_num, "/", len(detectors), end="\r")

        ## generate test points for prediction
        prediction_range = np.linspace(
            len(Y) + 1, len(Y) + (days_in_future * 24) + 1, (days_in_future * 24)
        ).reshape(
            (days_in_future * 24), 1
        )  # test points must be of shape (N, D)

        ## predict mean and variance of latent GP at test points
        mean, var = model.predict_f(prediction_range)

        # reverse min_max scaler
        test_predict = scaler.inverse_transform(mean)
        test_var = scaler.inverse_transform(var)

        # find the time period for our testPredictions
        prediction_start = dataset["measurement_end_utc"].max()

        forecast_period = pd.date_range(
            start=prediction_start,
            end=prediction_start + np.timedelta64(24 * days_in_future - 1, "h"),
            freq="H",
        )

        # organise data into dataframe similar to the SCOOT outputs
        forecast_df = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": proc_df[proc_df["detector_id"] == detector]["lon"].iloc[0],
                "lat": proc_df[proc_df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": forecast_period,
                "measurement_end_utc": forecast_period + np.timedelta64(1, "h"),
                "n_vehicles_in_interval": test_predict.flatten(),
                "prediction_variance": test_var.flatten(),
                "baseline_upper": 3 * np.sqrt(test_var.flatten()) + test_predict.flatten(),
                "baseline_lower": test_predict.flatten() - 3 * np.sqrt(test_var.flatten()),
            }
        )

        framelist.append(forecast_df)

        del model

    return pd.concat(framelist)




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
    for d, detector in enumerate(detectors, 1):
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
        
        print(d, "/", len(detectors), end="\r")

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

    if method == "LSTM":
        y = lstm_forecast(
            df,
            days_in_past=days_in_past,
            days_in_future=days_in_future,
            detectors=detectors,
        )

    if method == "GP":
        y = gp_forecast(
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
        forecast_df["baseline_upper"] = forecast_df["baseline_upper"].apply(
            lambda x: np.max([0, x])
        )
    forecast_df["baseline_lower"] = forecast_df["baseline_lower"].apply(
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
    #df_d["measurement_end_utc"]=df_d["measurement_end_utc"].astype('O')
    ax.plot(df_d["measurement_end_utc"], df_d["baseline"], label="baseline")
    ax.plot(df_d["measurement_end_utc"], df_d["count"], "^", label="count")

    if "prediction_variance" in df_d.columns:
        ax.fill_between(
            df_d["measurement_end_utc"],
            df_d["baseline"] + 3*np.sqrt(df_d["prediction_variance"]),
            df_d["baseline"] - 3*np.sqrt(df_d["prediction_variance"]),
            color="C0",
            alpha=0.3, label= "3$\sigma$")
    fig.autofmt_xdate()
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


def simple_training_loop(
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    model: gpflow.models.GPModel,
    optimizer: tf.optimizers.Optimizer,
    maxiter: int = 2000,
    logging_freq: int = 10,
):
    ## Optimization functions - train the model for the given maxiter
    def optimization_step(model: gpflow.models.GPR, x_train, y_train):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            obj = -model.elbo((x_train, y_train))
            grads = tape.gradient(obj, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


    tf_optimization_step = tf.function(optimization_step)
    for epoch in range(maxiter):
        tf_optimization_step(model, x_train, y_train)

        epoch_id = epoch + 1
        if epoch_id % logging_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo((x_train, y_train))}")