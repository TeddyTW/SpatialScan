from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sbn
from typing import Any, List, Type
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from datetime import datetime

def MA(df: pd.DataFrame, detector: str, past_days: int)-> float:
    """Function calculates the average number of vehicles over the last past_days

    Args:
        df: Dataframe of SCOOT data
        detector: string representing detector ID
        past_days: integer number of days to look back and average over

    Returns:
        Average number of vehicles measured at this detector in the previous past_days days

        """

    one_D = df[df["detector_id"] == detector]
    one_D = one_D.sort_values(by=['measurement_end_utc'])
    return one_D.tail(n=24*past_days)['n_vehicles_in_interval'].mean()

def MALD(df: pd.DataFrame, detector: str, past_days: int, hour: int)-> float:

    """Proportion of counts at this hour, looking back at all historical data

    Args:
        df: Dataframe of SCOOT data
        detector: string representing detector ID
        past_days: integer number of days to look back and average over
        hour: integer hour of the day for which to calculate MALD

    Returns:
        Proportion of vehicles expected at this hour

        """

    one_D = df[df["detector_id"] == detector].tail(n=24*past_days)
    beta = (one_D[one_D['measurement_end_utc'].dt.hour == hour]['n_vehicles_in_interval'].sum())/(one_D['n_vehicles_in_interval'].sum())
    return beta

def MALDforecast(df: pd.DataFrame, detectors: list, days_in_past: int,
                 days_in_future: int, display: bool = False)-> pd.DataFrame:

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

    framelist = []
    for detector in detectors:
        meanV = MA(df, detector, days_in_past)
        pred = []
        endtime = []
        starttime = []
        for i in range(1, 24*days_in_future +1):
            end = df['measurement_end_utc'].to_numpy()[-1] + np.timedelta64(i, 'h')
            hour = (df['measurement_end_utc'].dt.hour.to_numpy()[-1] + i)% 24
            start = df['measurement_start_utc'].to_numpy()[-1] + np.timedelta64(i, 'h')
            beta = MALD(df, detector, days_in_past, hour)
            endtime.append(end)
            starttime.append(start)
            pred.append(beta*24*meanV)

        df2 = pd.DataFrame({"detector_id" : detector, "lon" : df[df["detector_id"] == detector]["lon"].iloc[0], "lat" : df[df["detector_id"] == detector]["lat"].iloc[0], 'measurement_start_utc': starttime, 'measurement_end_utc':endtime, "n_vehicles_in_interval": pred})
        framelist.append(df2)
    DF = pd.concat(framelist)

    if(display):
        df_plot = DF.set_index('measurement_end_utc')
        for detector in detectors:
            df_plot[df_plot["detector_id"] == detector]["n_vehicles_in_interval"].plot()

    return DF


def holt_winters(df: pd.DataFrame, detectors: list, days_in_past: int, 
                days_in_future: int, alpha: float = 0.1, beta: float = 0.1, 
                gamma: float = 0.1, display: bool = False)-> pd.DataFrame:

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

    framelist=[]
    for detector in detectors:
        S=1
        T=1
        I=np.ones(24)
        one_D=df[df["detector_id"]==detector]
        one_D=one_D.sort_values(by=['measurement_end_utc'])
        past=one_D.tail(n=24*days_in_past)
        for i in range(0, len(past)):
            h=i%24
            c = past["n_vehicles_in_interval"].iloc[i]
            Snew = (alpha*(c/I[h])) + (1-alpha)*(S + T)
            T = beta*(Snew - S) + (1-beta)*T
            I[h] = gamma*(c/Snew) + (1-gamma)*I[h]
            S=Snew

        baseline = []
        endtime = []
        starttime = []
        for j in range(0, days_in_future*24):
            end=df['measurement_end_utc'].to_numpy()[-1] + np.timedelta64(j, 'h')
            start=df['measurement_start_utc'].to_numpy()[-1] + np.timedelta64(j, 'h')
            h=j%24
            b=(S + T)*I[h]
            baseline.append(b)
            endtime.append(end)
            starttime.append(start)

            Snew = (alpha*(b/I[h])) + (1-alpha)*(S + T)
            T = beta*(Snew - S) + (1-beta)*T
            I[h] = gamma*(b/Snew) + (1-gamma)*I[h]
            S=Snew

        df2 = pd.DataFrame({"detector_id" : detector, "lon" : df[df["detector_id"]==detector]["lon"].iloc[0], "lat" : df[df["detector_id"]==detector]["lat"].iloc[0], 'measurement_start_utc': starttime, 'measurement_end_utc':endtime, "n_vehicles_in_interval": baseline})
        framelist.append(df2)
    DF=pd.concat(framelist)
    
    if(display):
        df_plot=DF.set_index('measurement_end_utc')
        for detector in detectors:
            df_plot[df_plot["detector_id"]==detector]["n_vehicles_in_interval"].plot()

    return DF

def count_baseline(df: pd.DataFrame, detectors: list, days_in_past: int, days_in_future: int, method: str ="HW"):

    """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args: 
        df: Dataframe of SCOOT data
        detectors: List of detectors to look at
        days_in_past: Integer past days to train forecast one
        days_in_future: Days in future produce a baseline too and record count for
        method: Forecast method to use for baseline, default is "HW" for Holt-Winters, option for MLAD

    Returns:
        Dataframe of counts and baseline along with detector data

        """

    prediction_start = df["measurement_end_utc"].iloc[-1] - np.timedelta64(days_in_future*24, 'h')

    train_data=df[df["measurement_end_utc"]<=prediction_start]
    test_data=df[df["measurement_end_utc"]>prediction_start]

    if(method=="HW"):
        y=holt_winters(train_data, detectors, days_in_past, days_in_future, alpha=0.05, beta=0.05, gamma=0.2)
    if(method=="MALD"):
        y=MALDforecast(train_data, detectors, days_in_past, days_in_future)
    sd=[]

    for detector in detectors:

        sd.append(test_data[test_data['detector_id']==detector])

    sample_test_data=pd.concat(sd)

    Y=y.merge(sample_test_data, on=["lon", "lat","measurement_end_utc", "detector_id", "measurement_start_utc"], how='left')
    Y=Y.rename(columns={"n_vehicles_in_interval_x": "baseline", "n_vehicles_in_interval_y": "count"})

    return Y