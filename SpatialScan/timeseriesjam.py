"""Module containing Time Series Forecast functionality for JamCam data"""
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
        Dataframe forecast in same format as SCOOT input dataframe

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
        shift=1
        for j in range(0, days_in_future * 16):
            h = j % 16
            if h==0:
                shift+=8
            end = df["measurement_end_utc"].to_numpy()[-1] + np.timedelta64(j + shift, "h")
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
    alpha: float = 0.05, 
    beta: float = 0.05, 
    gamma: float = 0.4,
) -> pd.DataFrame:

    """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args:
        df: Dataframe of JamCam data
        days_in_past: Integer past days to train forecast one
        days_in_future: Days in future produce a baseline too and record count for
        method: Forecast method to use for baseline, default is "HW" for Holt-Winters, option for MLAD
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
    
    T = pd.date_range(start=Y["measurement_end_utc"].min() - np.timedelta64(3, "h"), end=Y["measurement_end_utc"].max() + np.timedelta64(5, "h"), freq="H",)
    dets=Y["detector_id"].unique()
    mux = pd.MultiIndex.from_product(
            [dets, T], names=("detector_id", "measurement_end_utc")
        )
    Y = Y.set_index(["detector_id", "measurement_end_utc"])
    Y=Y.reindex(mux)

    Y = Y.reset_index()
    
    return Y