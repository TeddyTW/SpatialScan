"""Module containing Time Series Forecast functionality"""
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.colors as colors
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


class LSTM_Forecast:
    def __init__(self, days_in_past, days_in_future, detectors, LSTM_blocks=8, **kwargs):

        self.N = len(detectors)
        self.detectors = detectors
        self.model=[]
        for i in range(0, len(detectors)):
            self.model.append(Sequential())
            self.model[i].add(LSTM(8, input_shape=(1, 24 * days_in_past)))
            self.model[i].add(Dense(24 * days_in_future))
            self.model[i].compile(loss="mean_squared_error", optimizer="adam")
        self.days_in_future = days_in_future
        self.days_in_past = days_in_past

    def create_dataset(dataset: pd.DataFrame):
        dataX, dataY = [], []
        for i in range(0, len(dataset) - self.days_in_past - self.days_in_future + 1, self.days_in_future):
            # a will be one set of X data to be used as inputs to the NN
            a = dataset[i : (i + self.days_in_past), 0]
            dataX.append(a)
            # by will be one set of Y data, to be predicted by the NN and then
            # compared to for means of training
            b = dataset[(i + self.days_in_past) : (i + self.days_in_past + self.days_in_future), 0]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)

    def data_test_train_org(df: pd.DataFrame):

        df=interpolator(df)

        for detector in self.detectors:

            dataset = df[df["detector_id"] == detector]

            # split up dataset into train and test. The test dataset is the final
            # days in future + days in the past (days in past to feed into the inputs
            # to make predictions on days in the future). the training days are used to
            # train the weights of the NN
            prediction_start = df["measurement_end_utc"].max() - np.timedelta64(
                (self.days_in_future + self.days_in_past) * 24, "h"
            )
            train = dataset[dataset["measurement_end_utc"] <= prediction_start]
            test = dataset[dataset["measurement_end_utc"] > prediction_start]

            # datasets converted to arrays then rescaled with a MinMax scaler
            train = train["n_vehicles_in_interval"].to_numpy()
            test = test["n_vehicles_in_interval"].to_numpy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            train = scaler.fit_transform(train.reshape(-1, 1))
            test = scaler.fit_transform(test.reshape(-1, 1))

            # use create to reshape the data into sucessive sets of X and Y features and
            # labels that can be fed into the LSTM NN
            X_train, Y_train = create_dataset(
                train, look_back=self.days_in_past * 24, look_forward=24 * self.days_in_future
            )
            X_test, Y_test = create_dataset(
                test, look_back=self.days_in_past * 24, look_forward=24 * self.days_in_future
            )

            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            return (X_train, Y_train, X_test, Y_test)

    def fit(X_train, Y_train, epochs=50, batch_size=1, verbose=0):
        checkpoint_cb = ModelCheckpoint("lstm_forecast.h5")
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[checkpoint_cb],
        )

def interpolator(df):
    detectors = df["detector_id"].drop_duplicates().to_numpy()
    df_list=[]
    for detector in detectors:
        dataset=df[df["detector_id"]==detector]
        dataset.index=dataset["measurement_end_utc"]
        
        T=pd.date_range(start=df["measurement_end_utc"].min(), end=df["measurement_end_utc"].max(), freq='H')
        dataset = dataset.reindex(T)
        dataset["n_vehicles_in_interval"]=dataset["n_vehicles_in_interval"].interpolate(method='linear', limit_direction='forward', axis=0)
        dataset["detector_id"]=dataset["detector_id"].interpolate(method='pad', limit_direction='forward', axis=0)
        dataset["lon"]=dataset["lon"].interpolate(method='pad', limit_direction='forward', axis=0)
        dataset["lat"]=dataset["lat"].interpolate(method='pad', limit_direction='forward', axis=0)
        dataset["measurement_end_utc"]=dataset.index
        dataset["measurement_start_utc"]=dataset["measurement_end_utc"]-+ np.timedelta64(1, "h")
        df_list.append(dataset)
        
    DF = pd.concat(df_list)
    
    return DF.reset_index(drop=True)