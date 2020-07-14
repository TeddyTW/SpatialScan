"""Module containing Time Series Forecast functionality for JamCam data"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy.optimize import minimize
import gpflow
from gpflow.utilities import print_summary


def holt_wintersJ(
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    alpha: float = 0.05,
    beta: float = 0.05,
    gamma: float = 0.4,
    display: bool = False,
    detectors: list = None,
) -> pd.DataFrame:

    """Average forecast using Holt-Winters method for JamCam 16h cycles

    Args: 
        df: Dataframe of JamCam data
        detectors: List of detectors to look at
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        display: boolean which determines whether to plot forecast
        alpha, beta, gamma: optimisation parameters

    Returns:
        Dataframe forecast in same format as JamCam input dataframe

        """

    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()

    framelist = []
    for detector in detectors:
        S = 1
        T = 1
        I = np.ones(16)
        one_D = df[df["detector_id"] == detector]
        one_D = one_D.sort_values(by=["measurement_end_utc"])
        past = one_D.tail(n=16 * days_in_past)
        for i in range(0, len(past)):
            h = i % 16
            c = past["n_vehicles_in_interval"].iloc[i]
            Snew = (alpha * (c / I[h])) + (1 - alpha) * (S + T)
            T = beta * (Snew - S) + (1 - beta) * T
            I[h] = gamma * (c / Snew) + (1 - gamma) * I[h]
            S = Snew

        baseline = []
        endtime = []
        starttime = []
        shift = 1
        for j in range(0, days_in_future * 16):
            h = j % 16
            if h == 0:
                shift += 8
            end = df["measurement_end_utc"].to_numpy()[-1] + np.timedelta64(
                j + shift, "h"
            )
            start = df["measurement_start_utc"].to_numpy()[-1] + np.timedelta64(
                j + shift, "h"
            )

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


def count_baselineJ(
    df: pd.DataFrame,
    days_in_past: int,
    days_in_future: int,
    method: str = "HW",
    detectors: list = None,
    alpha: float = 0.1,
    beta: float = 0.1,
    gamma: float = 0.2,
) -> pd.DataFrame:

    """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args:
        df: Dataframe of JamCam data
        days_in_past: Integer past days to train forecast one
        days_in_future: Days in future produce a baseline too and record count for
        method: Forecast method to use for baseline, default is "HW" for Holt-Winters, option for GP
        detectors: List of detectors to look at

    Returns:
        Dataframe of counts and baseline along with detector data

        """

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

    avail_past_days = int(len(train_data["measurement_end_utc"].unique()) / 16)
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
        y = holt_wintersJ(
            train_data,
            days_in_past,
            days_in_future,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            detectors=detectors,
        )

    if method == "GP":
        y = GP_forecast(
            train_data,
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

    T = pd.date_range(
        start=Y["measurement_end_utc"].min() - np.timedelta64(3, "h"),
        end=Y["measurement_end_utc"].max() + np.timedelta64(5, "h"),
        freq="H",
    )
    dets = Y["detector_id"].unique()
    mux = pd.MultiIndex.from_product(
        [dets, T], names=("detector_id", "measurement_end_utc")
    )
    Y = Y.set_index(["detector_id", "measurement_end_utc"])
    Y = Y.reindex(mux)

    Y = Y.reset_index()

    return Y


def CB_plotJ(df: pd.DataFrame):

    """Function that plots Counts/Baseline as a 3D plot  with detector locations. Counts are 
        shown by size of point, and C/B is shown using a colourmap
        
        Args:
            Dataframe with Time, Count and Baseline columns
            """

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


def GP_forecast(
    df: pd.DataFrame,
    days_in_past: int = 2,
    days_in_future: int = 1,
    detectors: list = None,
) -> pd.DataFrame:

    """Forecast using Gaussian Processes 
    Args: 
        df: Dataframe of JamCam data
        days_in_past: Integer number of previous days to use for forecast
        days_in_future: Days in future produce a for forecast for
        detectors: List of detectors to look at


    Returns:
        Dataframe forecast in same format as JamCam input dataframe

        """

    # extract numpy array of detector ID's
    if detectors is None:
        detectors = df["detector_id"].drop_duplicates().to_numpy()
    framelist = []

    i = 0
    for detector in detectors:
        i += 1

        dataset = df[df["detector_id"] == detector].tail(n=16 * days_in_past)

        Y = dataset["n_vehicles_in_interval"].to_numpy().reshape(-1, 1)
        Y = Y.astype(float)
        X = np.arange(1, len(Y) + 1, dtype=float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        kern_pD = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
        kern_pW = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
        kern_SE = gpflow.kernels.SquaredExponential()
        kern_W = gpflow.kernels.White()
        # kern_M = gpflow.kernels.Matern52()

        kern_pD.period.assign(16.0)
        # kern_pD.base_kernel.variance.assign(10)
        kern_pW.period.assign(112.0)
        # kern_pW.base_kernel.variance.assign(10)

        k = kern_pD * kern_pW + kern_SE + kern_W

        m = gpflow.models.GPR(data=(X, y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=100)
        )

        print("please wait: ", i, "/", len(detectors), end="\r")

        ## generate test points for prediction
        xx = np.linspace(
            len(Y) + 1, len(Y) + (days_in_future * 16) + 1, (days_in_future * 16)
        ).reshape(
            (days_in_future * 16), 1
        )  # test points must be of shape (N, D)

        ## predict mean and variance of latent GP at test points
        mean, var = m.predict_f(xx)

        # reverse min_max scaler
        testPredict = scaler.inverse_transform(mean)
        testVar = scaler.inverse_transform(var)

        # find the time period for our testPredictions
        start_date = dataset["measurement_end_utc"].max() + np.timedelta64(8, "h")
        end_date = (start_date + np.timedelta64(16 + (24 * (days_in_future - 1)), "h"),)

        # print(start_date, end_date)
        N_days = days_in_future
        T = pd.date_range(start=start_date, periods=16, freq="H")
        start_of_day = start_date + np.timedelta64(1, "D")
        for d in range(0, N_days - 1):
            t = pd.date_range(
                start=start_of_day, end=start_of_day + np.timedelta64(15, "h"), freq="H"
            ).to_numpy()

            T = np.append(T, t)
            start_of_day = start_of_day + np.timedelta64(1, "D")

        T = np.array(T)

        # organise data into dataframe similar to the SCOOT outputs
        df2 = pd.DataFrame(
            {
                "detector_id": detector,
                "lon": df[df["detector_id"] == detector]["lon"].iloc[0],
                "lat": df[df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_start_utc": T,
                "measurement_end_utc": T + np.timedelta64(1, "h"),
                "n_vehicles_in_interval": testPredict.flatten(),
                "prediction_variance": testVar.flatten(),
            }
        )

        framelist.append(df2)

    return pd.concat(framelist)
