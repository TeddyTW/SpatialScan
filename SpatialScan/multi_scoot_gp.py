from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import gpflow


def multi_gp(train, forecasting, detectors=None, kern=None):

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

    scaler = MinMaxScaler(feature_range=(-1, 1))
    y = scaler.fit_transform(Y)

    # fit our GP to X & y
    model = gpflow.models.GPR(data=(X, y), kernel=kern, mean_function=None)
    opt = gpflow.optimizers.Scipy()

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

    output = output.rename(columns={"n_vehicles_in_interval": "count"})
    return output
