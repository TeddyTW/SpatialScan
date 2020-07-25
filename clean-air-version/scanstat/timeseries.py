"""Module to contain implementations of various time-series methods to be used
within the scan statistic framework. Currently only contains the Holt-Winters
exponentially smoothed method."""

import numpy as np
import pandas as pd

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

import gpflow


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
                "99_upper": baseline,
                "99_lower": baseline,
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

        try:
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=100),
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
                "99_upper": 3 * np.sqrt(test_var.flatten()) + test_predict.flatten(),
                "99_lower": 3 * np.sqrt(test_var.flatten()) - test_predict.flatten(),
            }
        )

        framelist.append(forecast_df)

        del model

    return pd.concat(framelist)
