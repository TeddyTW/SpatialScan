from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import gpflow
from gpflow.utilities import print_summary, set_trainable, to_default_float
from sklearn.preprocessing import MinMaxScaler

class JamCamMVGP:
    
    def __init__(self):
        self.model = None
        self.model_training_info = None
        self.scaler = None

    def create_dataset(self, single_det_df, target, days):
        x = [single_det_df["n_vehicles_in_interval_car"].to_numpy()[:-16*days],
            single_det_df["n_vehicles_in_interval_person"].to_numpy()[:-16*days], 
            single_det_df["n_vehicles_in_interval_bus"].to_numpy()[:-16*days]]
        if target == "car":
            y = single_det_df["n_vehicles_in_interval_car"].to_numpy()[16*days:]
        if target == "person":
            y = single_det_df["n_vehicles_in_interval_person"].to_numpy()[16*days:]
        if target == "bus":
            y = single_det_df["n_vehicles_in_interval_bus"].to_numpy()[16*days:]
        return(x, y)    

    def train(self, scoot_df, detector, target, days = 3):
        single_det_df = scoot_df[scoot_df["detector_id"]==detector]
        x, y = self.create_dataset(single_det_df, target, days=days)
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X = np.array(x).T
        X = scaler_x.fit_transform(X)

        Y = y.reshape(-1, 1)
        Y = Y.astype(float)
        Y = scaler_y.fit_transform(Y)

        #print(Y)
        kern_w = gpflow.kernels.White(1e-5)
        set_trainable(kern_w.variance, False)
        kern = gpflow.kernels.Linear() + gpflow.kernels.SquaredExponential(lengthscales=[1, 1, 1])

        model = gpflow.models.GPR(data=(X, Y), kernel=kern, mean_function=None)
        opt = gpflow.optimizers.Scipy()       
        opt.minimize(
                    model.training_loss,
                    model.trainable_variables,
                    options=dict(maxiter=100),
                )

        self.model = model
        return model, scaler_x, scaler_y


    def count_baseline(self, train_df, count_df, detectors, target, days=2):
        frame_list=[]
        for i, detector in enumerate(detectors, 1):
            print("please wait: ", i, "/", len(detectors), end="\r")
            try:
                model, scaler_x, scaler_y = self.train(train_df, detector, target, days=days)
            except:
                print("uninvertable: ", detector)
                continue
            single_detector_df = count_df[count_df["detector_id"] == detector]
            #print(single_detector_df)
            x, y = self.create_dataset(single_detector_df, target, days=days)
            X = np.array(x).T
            X = scaler_x.fit_transform(X)
            mean, var = model.predict_f(X)
            # mean = mean.numpy().flatten()
            # var = var.numpy().flatten()
            mean = scaler_y.inverse_transform(mean).flatten()
            var = scaler_y.inverse_transform(var).flatten()
            forecast_period = (single_detector_df[single_detector_df["measurement_end_utc"] >= 
                                single_detector_df["measurement_end_utc"].min() 
                                                  + np.timedelta64(days*24, "h")]["measurement_end_utc"]).to_numpy()
            forecast_df = pd.DataFrame(
                {
                "count": y,
                "baseline": mean,
                "prediction_variance": var,
                "baseline_upper":mean
                + 3 * np.sqrt(var),
                "baseline_lower": mean
                - 3 * np.sqrt(var),
                "detector_id": detector,
                "lon": single_detector_df[single_detector_df["detector_id"] == detector]["lon"].iloc[0],
                "lat": single_detector_df[single_detector_df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_end_utc": forecast_period,
                }
             )
            frame_list.append(forecast_df)

            del model

        return pd.concat(frame_list)

class JamCamMOGP:

    def __init__(self):
        self.model = None
        self.model_training_info = None
        self.scaler = None

    def train(self, dataset, kern=None, method="GPR", num_induce=50):
        """method"""
        # set up kernels
        if kern is None:
            
            kern_pD = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_pW = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
            kern_SE = gpflow.kernels.SquaredExponential()
            kern_W = gpflow.kernels.White()
            kern_M = gpflow.kernels.Matern32()

            kern_pD.period.assign(24.0)
            # kern_pD.base_kernel.variance.assign(10)
            kern_pW.period.assign(168.0)
            # kern_pW.base_kernel.variance.assign(10)

            kern = kern_pD + kern_pW + kern_M
            
        X = (
            (dataset["measurement_end_utc"] - dataset["measurement_end_utc"].min())
            .astype("timedelta64[h]")
            .to_numpy()
            .reshape(-1, 1)
        )

        starts = dataset["measurement_end_utc"].min()
        ends = dataset["measurement_end_utc"].max()

        Y = [dataset["n_vehicles_in_interval_car"].to_numpy(), dataset["n_vehicles_in_interval_person"].to_numpy(), dataset["n_vehicles_in_interval_bus"].to_numpy()]

        #self.model_training_info = pd.DataFrame({"detector_id" : detectors, "training_start" : starts, "training_end" : ends})

        Y = np.array(Y)
        Y = Y.T 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(Y)

        # fit our GP to X & y
        if method == "GPR":
            model = gpflow.models.GPR(data=(X, y), kernel=kern, mean_function=None)
            opt = gpflow.optimizers.Scipy()
            #print("BEFORE OPTIMISATION")
            #print_summary(model)

            opt.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=10000))
        
        if method == "sharedkern":
            kern = gpflow.kernels.SharedIndependent(kern, output_dim=3)
            Zinit = np.linspace(X.min(), X.max(), num_induce)[:, None]
            # initialization of inducing input locations (M random points from the training inputs)
            Z = Zinit.copy()
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z))
            # create SVGP model as usual and optimize
            model = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=3)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                options={"disp": True, "maxiter": 10000},)

        if method == "seperatekern":
            kern_list = [kern for _ in range(3)]
            kern = gpflow.kernels.SeparateIndependent(kern_list)
            Zinit = np.linspace(X.min(), X.max(), num_induce)[:, None]
            # initialization of inducing input locations (M random points from the training inputs)
            Z = Zinit.copy()
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z))
            # create SVGP model as usual and optimize
            model = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=3)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                options={"disp": True, "maxiter": 10000},)
            
        if method == "coregional":
            kern_list = [kern for _ in range(3)]
            kern = gpflow.kernels.LinearCoregionalization(kern_list, W=np.random.randn(3, 3))
            Zinit = np.linspace(X.min(), X.max(), num_induce)[:, None]
            # initialization of inducing input locations (M random points from the training inputs)
            Z = Zinit.copy()
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z))
            # initialize mean of variational posterior to be of shape MxL
            q_mu = np.zeros((num_induce, 3))
            # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
            q_sqrt = np.repeat(np.eye(num_induce)[None, ...], 3, axis=0) * 1.0
            model = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                options={"disp": True, "maxiter": 10000},)

        #print("AFTER OPTIMISATION")
        #print_summary(model)

        self.model = model
        self.scaler = scaler
        return model, scaler
    
    def count_baseline(self, train_df, count_df, detectors: list = None, method = "GPR"):
        
        pd.options.mode.chained_assignment = None

        if detectors is None:
            detectors = forecast_scoot_df["detector_id"].drop_duplicates().to_numpy()
        
        frame_list=[]
        for i, detector in enumerate(detectors, 1):
            
            single_detector_train = train_df[train_df["detector_id"]==detector]
            print("please wait: ", i, "/", len(detectors), end="\r")
            
            model, scaler = self.train(single_detector_train, method=method)
            if model is None:
                print("uninvertable: ", detector)
                continue
                
            single_detector_count = count_df[count_df["detector_id"] == detector]

            X = ((single_detector_count["measurement_end_utc"] - single_detector_train["measurement_end_utc"].min())
            .astype("timedelta64[h]")
            .to_numpy()
            .reshape(-1, 1))
            
            car = single_detector_count["n_vehicles_in_interval_car"].to_numpy() 
            person = single_detector_count["n_vehicles_in_interval_person"].to_numpy()
            
            mean, var = model.predict_f(X)
            mean = scaler.inverse_transform(mean)
            var = scaler.inverse_transform(var)
            mean = mean.T
            var = var.T

            forecast_df = pd.DataFrame(
                {
                "count": car,
                "baseline": mean[0],
                "prediction_variance": var[0],
                "baseline_upper":mean[0]
                + 3 * np.sqrt(var[0]),
                "baseline_lower": mean[0]
                - 3 * np.sqrt(var[0]),
                "detector_id": detector,
                "lon": single_detector_count[single_detector_count["detector_id"] == detector]["lon"].iloc[0],
                "lat": single_detector_count[single_detector_count["detector_id"] == detector]["lat"].iloc[0],
                "measurement_end_utc": single_detector_count["measurement_end_utc"]
                }
             )
            frame_list.append(forecast_df)

            del model

        return pd.concat(frame_list)


class MultiVariateGP:
    
    def __init__(self):
        self.model = None
        self.model_training_info = None
        self.scaler = None

    def create_dataset(self, scoot_df, detectors, target, days):
        x=[]
        for i, detector in enumerate(detectors, 0):
            dataset=scoot_df[scoot_df["detector_id"]==detector]
            dataset = dataset["n_vehicles_in_interval"].to_numpy()
            if detector == target:
                y = dataset[24*days:]
            x.append(dataset[:-24*days])
        return(x, y)    

    def train(self, scoot_df, detectors, target, days = 7):
        x, y = self.create_dataset(scoot_df, detectors, target, days=days)
        X = np.array(x).T
        Y = y.reshape(-1, 1)
        Y = Y.astype(float)
        kern = gpflow.kernels.Linear()
        model = gpflow.models.GPR(data=(X, Y), kernel=kern, mean_function=None)
        
        opt = gpflow.optimizers.Scipy()
        
        opt.minimize(
                    model.training_loss,
                    model.trainable_variables,
                    options=dict(maxiter=500),
                )
        
        self.model = model
        return model

    def count_baseline(self, train_df, count_df, detectors, days=7):
        frame_list=[]
        for i, detector in enumerate(detectors, 1):
            print("please wait: ", i, "/", len(detectors), end="\r")
            model = self.train(train_df, detectors, detector,days=days)
            x, y = self.create_dataset(count_df, detectors, detector, days=days)
            X = np.array(x).T
            mean, var = model.predict_f(X)
            mean = mean.numpy().flatten()
            var = var.numpy().flatten()
            single_detector_df = count_df[count_df["detector_id"] == detector]
            forecast_period = (single_detector_df[single_detector_df["measurement_end_utc"] >= single_detector_df["measurement_end_utc"].min() + np.timedelta64(days*24, "h")]["measurement_end_utc"]).to_numpy()
            print(len(y), len(forecast_period))
            forecast_df = pd.DataFrame(
                {
                "count": y,
                "baseline": mean,
                "prediction_variance": var,
                "baseline_upper":mean
                + 3 * np.sqrt(var),
                "baseline_lower": mean
                - 3 * np.sqrt(var),
                "detector_id": detector,
                "lon": single_detector_df[single_detector_df["detector_id"] == detector]["lon"].iloc[0],
                "lat": single_detector_df[single_detector_df["detector_id"] == detector]["lat"].iloc[0],
                "measurement_end_utc": forecast_period,
                }
             )
            frame_list.append(forecast_df)

            del model

        return pd.concat(frame_list)


class MultiOutputGP:

    def __init__(self):
        self.model = None
        self.model_training_info = None
        self.scaler = None

    def train(self, training_scoot_df, detectors=None, kern=None, method="GPR", num_induce=50):
        """method"""
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
        if method == "GPR":
            model = gpflow.models.GPR(data=(X, y), kernel=kern, mean_function=None)
            opt = gpflow.optimizers.Scipy()
            print("BEFORE OPTIMISATION")
            print_summary(model)

            opt.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=10000))
        
        if method == "sharedkern":
            kern = gpflow.kernels.SharedIndependent(kern, output_dim=len(detectors))
            Zinit = np.linspace(X.min(), X.max(), num_induce)[:, None]
            # initialization of inducing input locations (M random points from the training inputs)
            Z = Zinit.copy()
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z))
            # create SVGP model as usual and optimize
            model = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=len(detectors))
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                options={"disp": True, "maxiter": 10000},)

        if method == "seperatekern":
            kern_list = [kern for _ in range(len(detectors))]
            kern = gpflow.kernels.SeparateIndependent(kern_list)
            Zinit = np.linspace(X.min(), X.max(), num_induce)[:, None]
            # initialization of inducing input locations (M random points from the training inputs)
            Z = Zinit.copy()
            iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z))
            # create SVGP model as usual and optimize
            model = gpflow.models.SVGP(kern, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=len(detectors))
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss_closure((X, y)),
                variables=model.trainable_variables,
                method="l-bfgs-b",
                options={"disp": True, "maxiter": 10000},)

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

