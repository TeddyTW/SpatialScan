"""Module to contain the GP Model Class"""

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import gpflow

# from gpflow.utilities import print_summary, set_trainable

import joblib


class GPLandscape:
    """Class which manages the training, saving and loading of GP models for an array of detectors"""

    def __init__(self):
        self.models = None
        self.model_last_update = None
        self.model_detector_id = None
        self.scalers = None

    def train_save_detector(
        self, scoot_df: pd.DataFrame, days_in_past: int, detector: str, kern=None
    ):
        """Trains GP a to one detector, using a SCOOT dataframe format. The trained model is
        then saved in a directory along with other useful model data such as scalers
        Args:
            df: SCOOT dataframe of one detector used for training
            days_in_past: how many most recent days of past dataframe should be used for training
            detector: detector_id as string
            kern: Optional kernel choice
        Returns:
            last_update: date for which the detector was trained
            det: name of detector
        """

        # set Y and X to our days_in_past used for fitting, reshape, and scale
        det = scoot_df["detector_id"].unique()[0]
        Y = (
            scoot_df["n_vehicles_in_interval"]
            .tail(n=24 * days_in_past)
            .to_numpy()
            .reshape(-1, 1)
        )
        last_update = scoot_df["measurement_end_utc"].tail(n=24 * days_in_past).min()
        Y = Y.astype(float)
        X = np.arange(1, len(Y) + 1, dtype=float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        # iniialise kernel
        if kern is None:

            kern_pd = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_pw = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_se = gpflow.kernels.SquaredExponential()

            kern_pd.period.assign(24.0)
            kern_pw.period.assign(168.0)
            # kern_SE.lengthscales.assign(100)

            k = kern_pd * kern_pw + kern_se
        else:
            k = kern

        # fit our GP to X & y
        model = gpflow.models.GPR(data=(X, y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()

        # optimise GP performance
        opt.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=100)
        )

        # save model as TF module under detector name
        frozen_model = gpflow.utilities.freeze(model)
        module_to_save = tf.Module()
        predict_fn = tf.function(
            frozen_model.predict_f,
            input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
        )
        module_to_save.predict = predict_fn

        # save path
        save_dir = str("gp_models/" + detector + "/")
        tf.saved_model.save(module_to_save, save_dir)

        scaler_filename = str(save_dir + "scaler.gz")
        joblib.dump(scaler, scaler_filename)

        return last_update, det

    def train_save_landscape(self, scoot_df: pd.DataFrame, days_in_past: int):
        """Trains GPs to multiple detectors passed to it in SCOOT dataframe format. Trained models are
        saved in directories

        Args:
            scoot_df: SCOOT dataframe used to train multiple detector models
            days_in_past: how manyt most recent days of past dataframe should be used for training
        """

        detectors = scoot_df["detector_id"].unique()

        last_updates = []
        saved_detectors = []

        for i, detector in enumerate(detectors, 1):
            single_detector_df = scoot_df[scoot_df["detector_id"] == detector]

            try:
                date, det = self.train_save_detector(
                    single_detector_df, days_in_past, detector
                )
            except:
                print(detector, " Matrix not invertible")
                continue

            last_updates.append(date)
            saved_detectors.append(det)
            print("please wait: ", i, "/", len(detectors), end="\r")

        pd.DataFrame(
            {"detectors": saved_detectors, "last_update": last_updates}
        ).to_csv("gp_models/det_date.csv", index=False)

    def load_landscape(self):
        """Loads array of pre-trained models and model data as arays from file, and
        sets them to the class variable"""

        models = []
        scalers = []

        det_date = pd.read_csv("gp_models/det_date.csv", index_col=False)
        detectors = det_date["detectors"].to_numpy()
        dates = det_date["last_update"].astype("datetime64[h]").to_numpy()

        for i, detector in enumerate(detectors, 1):

            save_dir = str("gp_models/" + detector + "/")
            scaler_filename = str(save_dir + "scaler.gz")
            models.append(tf.saved_model.load(save_dir))
            scalers.append(joblib.load(scaler_filename))
            print("please wait: ", i, "/", len(detectors), end="\r")

        self.models = models
        self.model_detector_id = detectors
        self.model_last_update = dates
        self.scalers = scalers

    def count_baseline(
        self, scoot_df: pd.DataFrame, detectors: list = None
    ) -> pd.DataFrame:
        """Produces a DataFrame where the count and baseline can be compared for use
        in scan statistics

    Args:
        scoot_df: Dataframe of processed SCOOT data which we want to compare to model
        detectors: List of detectors to compare to forecasts. Default behaviour
                   retrieves forecasts for all detectors present in input dataframe.

    Returns:
        forecast_df: Dataframe of SCOOT vehicle counts and baseline estimates"""

        pd.options.mode.chained_assignment = None

        if detectors is None:
            detectors = scoot_df["detector_id"].drop_duplicates().to_numpy()

        framelist = []

        for i, detector in enumerate(detectors, 1):
            print("please wait: ", i, "/", len(detectors), end="\r")

            one_detector_df = scoot_df.loc[scoot_df["detector_id"] == detector]

            start_of_trained_data = self.model_last_update[
                np.where(self.model_detector_id == detector)
            ]

            baseline_range = (
                (one_detector_df["measurement_end_utc"] - start_of_trained_data[0])
                .to_numpy()
                .astype("timedelta64[h]")
            )
            baseline_range = baseline_range + np.timedelta64(1, "h")

            loc = np.where(self.model_detector_id == detector)

            baseline_range = baseline_range.reshape(-1, 1)
            model = self.models[loc[0][0]]
            scaler = self.scalers[loc[0][0]]

            mean, var = model.predict(baseline_range)
            mean = scaler.inverse_transform(mean)
            var = scaler.inverse_transform(var)

            one_detector_df.rename(
                columns={"n_vehicles_in_interval": "count"}, inplace=True,
            )

            one_detector_df = one_detector_df.assign(baseline=mean.flatten().tolist())
            one_detector_df = one_detector_df.assign(
                upper_99=(3 * np.sqrt(var.flatten()) + mean.flatten()).tolist()
            )
            one_detector_df = one_detector_df.assign(
                lower_99=(mean.flatten() - 3 * np.sqrt(var.flatten())).tolist()
            )
            one_detector_df = one_detector_df.assign(
                prediction_variance=var.flatten().tolist()
            )

            framelist.append(one_detector_df)

        return pd.concat(framelist)
