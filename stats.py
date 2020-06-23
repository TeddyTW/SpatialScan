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

def dataframe_org(df):
    df2=df.sort_values(by=['measurement_end_utc'])
    df2.index=pd.to_datetime(df2['measurement_end_utc'])
    df2=df2.drop(["Unnamed: 0", "measurement_start_utc", "measurement_end_utc"], axis=1)
    return df2  

def MA(df, detector, past_days):
    one_D=df[df["detector_id"]==detector]
    return(one_D.tail(n=24*past_days)['n_vehicles_in_interval'].mean())

def MALD(df, detector, hour):
    one_D=df[df["detector_id"]==detector]
    hour=f'{hour:02}'
    beta=(one_D.loc[one_D.index.strftime("%H") == hour]['n_vehicles_in_interval'].sum())/(one_D['n_vehicles_in_interval'].sum())
    return beta

def forecast(df, detectors, days_in_past, days_in_future, display=False):
    framelist=[]
    for detector in detectors:
        meanV=MA(df, detector, days_in_past)
        pred=[]
        index=[]
        for i in range(1, 24*days_in_future +1):
            time=df.index[-1]+ np.timedelta64(i, 'h')
            beta=MALD(df, detector, time.hour)
            index.append(time)
            pred.append(beta*24*meanV)

        df2 = pd.DataFrame({"measurement_end_utc":index, "n_vehicles_in_interval": pred})
        df2["detector_id"]=detector
        df2["lon"] = df[df["detector_id"]==detector]["lon"][0]
        df2["lat"] = df[df["detector_id"]==detector]["lat"][0]
        
        framelist.append(df2)
    DF=pd.concat(framelist)
    DF=DF.set_index('measurement_end_utc')
    
    if(display):
        for detector in detectors:
            DF[DF["detector_id"]==detector]["n_vehicles_in_interval"].plot()
    
    return DF


def holt_winters(df, detectors, days_in_past, days_in_future, alpha=0.1, beta=0.1, gamma=0.1, display=False):
    framelist=[]
    for detector in detectors:
        S=1; T=1; I=np.ones(24);
        past=df[df["detector_id"]==detector].tail(n=24*days_in_past)
        for i in range(0, len(past)):
            h=i%24
            c = past["n_vehicles_in_interval"][i]
            Snew = (alpha*(c/I[h])) + (1-alpha)*(S + T)
            T = beta*(Snew - S) + (1-beta)*T
            I[h] = gamma*(c/Snew) + (1-gamma)*I[h]
            S=Snew

        baseline=[]
        index = []
        for j in range(0, days_in_future*24):
            time=df.index[-1]+ np.timedelta64(j, 'h')
            h=j%24
            b=(S + T)*I[h]
            baseline.append(b)
            index.append(time)

            Snew = (alpha*(b/I[h])) + (1-alpha)*(S + T)
            T = beta*(Snew - S) + (1-beta)*T
            I[h] = gamma*(b/Snew) + (1-gamma)*I[h]
            S=Snew

        df2 = pd.DataFrame({"measurement_end_utc":index, "n_vehicles_in_interval": baseline})
        df2["detector_id"]=detector
        df2["lon"] = df[df["detector_id"]==detector]["lon"][0]
        df2["lat"] = df[df["detector_id"]==detector]["lat"][0]
        framelist.append(df2)
    DF=pd.concat(framelist)
    DF=DF.set_index('measurement_end_utc')
    
    if(display):
        for detector in detectors:
            DF[DF["detector_id"]==detector]["n_vehicles_in_interval"].plot()

    return DF

    
    