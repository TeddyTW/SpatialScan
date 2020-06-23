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

def MA(df, detector, past_days):
    one_D=df[df["detector_id"]==detector]
    one_D=one_D.sort_values(by=['measurement_end_utc'])
    return(one_D.tail(n=24*past_days)['n_vehicles_in_interval'].mean())

def MALD(df, detector, hour):
    one_D=df[df["detector_id"]==detector]
    beta=(one_D[one_D['measurement_end_utc'].dt.hour == hour]['n_vehicles_in_interval'].sum())/(one_D['n_vehicles_in_interval'].sum())
    return beta

def forecast(df, detectors, days_in_past, days_in_future, display=False):
    framelist=[]
    for detector in detectors:
        meanV=MA(df, detector, days_in_past)
        pred=[]
        endtime=[]
        starttime=[]
        for i in range(1, 24*days_in_future +1):
            end=df['measurement_end_utc'].to_numpy()[-1] + np.timedelta64(i, 'h')
            hour=(df['measurement_end_utc'].dt.hour.to_numpy()[-1] + i)% 24
            start=df['measurement_start_utc'].to_numpy()[-1] + np.timedelta64(i, 'h')
            beta=MALD(df, detector, hour)
            endtime.append(end)
            starttime.append(start)
            pred.append(beta*24*meanV)

        df2 = pd.DataFrame({"detector_id" : detector, "lon" : df[df["detector_id"]==detector]["lon"].iloc[0], "lat" : df[df["detector_id"]==detector]["lat"].iloc[0], 'measurement_start_utc': starttime, 'measurement_end_utc':endtime, "n_vehicles_in_interval": pred})
        
        framelist.append(df2)
    DF=pd.concat(framelist)
    
    if(display):
        df_plot=DF.set_index('measurement_end_utc')
        for detector in detectors:
            df_plot[df_plot["detector_id"]==detector]["n_vehicles_in_interval"].plot()
    
    return DF


def holt_winters(df, detectors, days_in_past, days_in_future, alpha=0.1, beta=0.1, gamma=0.1, display=False):
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
    print(I)
    
    if(display):
            df_plot=DF.set_index('measurement_end_utc')
            for detector in detectors:
                df_plot[df_plot["detector_id"]==detector]["n_vehicles_in_interval"].plot()

    return DF

    
    