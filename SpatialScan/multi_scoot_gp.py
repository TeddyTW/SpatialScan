from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import gpflow
from gpflow.utilities import print_summary


class MultiGP:

    def __init__(self):
        self.model = None
        self.model_training_info = None
        self.scaler = None

    def train(self, training_scoot_df, detectors=None, kern=None):
        
        # set up kernels
        if kern is None:
            kern_pd = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_pw = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_se = gpflow.kernels.SquaredExponential()

            kern_pd.period.assign(24.0)
            kern_pw.period.assign(168.0)
            kern = kern_pd * kern_pw + kern_se

        if detectors is None:
            detectors = training_scoot_df["detector_id"].unique()
        

        Y = []
        X = []
        starts = []
        ends = []
        for detector in detectors:
            dataset = training_scoot_df[training_scoot_df["detector_id"] == detector]
            
            X = (
                (dataset["measurement_end_utc"] - dataset["measurement_end_utc"].min())
                .astype("timedelta64[h]")
                .to_numpy()
                .reshape(-1, 1)
            )

            starts.append(dataset["measurement_end_utc"].min())
            ends.append(dataset["measurement_end_utc"].max())

            Y.append(dataset["n_vehicles_in_interval"].tolist())

        self.model_training_info = pd.DataFrame({"detector_id" : detectors, "training_start" : starts, "training_end" : ends})

        Y = np.array(Y)
        Y = Y.T 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        # fit our GP to X & y
        model = gpflow.models.GPR(data=(X, y), kernel=kern, mean_function=None)
        opt = gpflow.optimizers.Scipy()

        print("BEFORE OPTIMISATION")
        print_summary(model)

        opt.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=10000))

        print("AFTER OPTIMISATION")
        print_summary(model)

        self.model = model
        self.scaler = scaler

    def forecast(self, forecast_scoot_df, detectors: list = None):

        pd.options.mode.chained_assignment = None

        if detectors is None:
            detectors = forecast_scoot_df["detector_id"].drop_duplicates().to_numpy()

        detectors_in=np.intersect1d(detectors, self.model_training_info["detector_id"].to_numpy())

        if(detectors_in!=detectors):
            print("Model not trained for: ", np.setdiff1d(detectors, detectors_in))
            print("Calculating for remaining detectors...")
            detectors=detectors_in

        prediction_start = (
            forecast_scoot_df["measurement_end_utc"].min() - self.model_training_info["training_start"].min()
        ) / np.timedelta64(1, "h")
        prediction_end = (
            forecast_scoot_df["measurement_end_utc"].max() - self.model_training_info["training_start"].min()
        ) / np.timedelta64(1, "h")

        prediction_range = np.arange(prediction_start, prediction_end + 1).reshape(-1, 1)

        mean, var = self.model.predict_f(prediction_range)
        mean = self.scaler.inverse_transform(mean)
        var = self.scaler.inverse_transform(var)

        mean = mean.T
        var = var.T

        framelist = []
        for d, detector in enumerate(detectors, 0):
            dataset = forecast_scoot_df[forecast_scoot_df["detector_id"] == detector]
            dataset["baseline"] = mean[d]
            dataset["prediction_variance"] = var[d]
            dataset["upper_99"] = mean[d] + 3 * np.sqrt(var[d])
            dataset["lower_99"] = mean[d] - 3 * np.sqrt(var[d])
            framelist.append(dataset)

        output = pd.concat(framelist)
        output = output.rename(columns={"n_vehicles_in_interval": "count"})
        return output

        



def multi_gp1(train, forecasting, detectors=None, kern=None):

    if kern is None:
        kern_pd = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
        kern_pw = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
        kern_se = gpflow.kernels.SquaredExponential()

        kern_pd.period.assign(24.0)
        kern_pw.period.assign(168.0)
        # kern_SE.lengthscales.assign(100)

        kern = kern_pd * kern_pw + kern_se
    if detectors is None:
        detectors = train["detector_id"].unique()
    Y = []
    X = []
    for detector in detectors:
        dataset = train[train["detector_id"] == detector]
        X = (
            (dataset["measurement_end_utc"] - dataset["measurement_end_utc"].min())
            .astype("timedelta64[h]")
            .to_numpy()
            .reshape(-1, 1)
        )
        Y.append(dataset["n_vehicles_in_interval"].tolist())
    Y = np.array(Y)
    Y = Y.T

    print(X.shape, Y.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(Y)

    # fit our GP to X & y
    model = gpflow.models.GPR(data=(X, y), kernel=kern, mean_function=None)
    opt = gpflow.optimizers.Scipy()

    print("BEFORE OPTIMISATION")
    print_summary(model)
    
    # optimise GP performance
    opt.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=10000)
    )

    prediction_start = (
        forecasting["measurement_end_utc"].min() - dataset["measurement_end_utc"].min()
    ) / np.timedelta64(1, "h")
    prediction_end = (
        forecasting["measurement_end_utc"].max() - dataset["measurement_end_utc"].min()
    ) / np.timedelta64(1, "h")
    prediction_range = np.arange(prediction_start, prediction_end + 1).reshape(-1, 1)
    mean, var = model.predict_f(prediction_range)
    mean = scaler.inverse_transform(mean)
    var = scaler.inverse_transform(var)

    mean = mean.T
    var = var.T

    framelist = []
    for d, detector in enumerate(detectors, 0):
        dataset = forecasting[forecasting["detector_id"] == detector]
        dataset["baseline"] = mean[d]
        dataset["prediction_variance"] = var[d]
        framelist.append(dataset)
    output = pd.concat(framelist)

    print("AFTER OPTIMISATION")
    print_summary(model)

    output = output.rename(columns={"n_vehicles_in_interval": "count"})
    return output
