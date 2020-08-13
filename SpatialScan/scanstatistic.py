"""Main Class for Scan Statistics"""

import logging
import pandas as pd

from SpatialScan.preprocessing import data_preprocessor
from SpatialScan.timeseries import count_baseline
from SpatialScan.scan import scan
from SpatialScan.results import database_results, visualise_results_from_database
from SpatialScan.region import make_region_from_res, plot_region_time_series, plot_region_by_rank


class ScanStatistic:
    """Simple helper class to reduce number of function calls when modelling"""

    def __init__(
        self,
        readings,
        grid_resolution=8,
        percentage_missing=20,
        max_anom_per_day=1,
        N_sigma=3,
        repeats=1,
        rolling_hours=24,
        fap_threshold=1e-40,
        consecutive_missing_threshold=3,
        global_threshold=False,
        drop_sparse=True,
        drop_anomalous=True,
        drop_aperiodic=True,
        drop_consecutives=True,
        data_type="scoot",
        days_in_past=28,
        days_in_future=1,
        ts_method="HW",
        alpha=0.03869791,
        beta=0.0128993,
        gamma=0.29348953,
        kernel=None,
    ):
        self.readings = readings
        self.grid_resolution = grid_resolution
        self.percentage_missing = percentage_missing
        self.max_anom_per_day = max_anom_per_day
        self.N_sigma = N_sigma
        self.repeats = repeats
        self.rolling_hours = rolling_hours
        self.fap_threshold = fap_threshold
        self.consecutive_missing_threshold = consecutive_missing_threshold
        self.global_threshold = global_threshold
        self.drop_sparse = drop_sparse
        self.drop_anomalous = drop_anomalous
        self.drop_aperiodic = drop_aperiodic
        self.drop_consecutives = drop_consecutives
        self.data_type = data_type
        self.days_in_past = days_in_past
        self.days_in_future = days_in_future
        self.ts_method = ts_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kernel = kernel

        # results at each stage of pipeline
        self.processed = None
        self.forecast = None
        self.all_results = None
        self.grid_results = None

    def run(self):
        """Build scan results"""
        self.processed = data_preprocessor(
            self.readings,
            self.percentage_missing,
            self.max_anom_per_day,
            self.N_sigma,
            self.repeats,
            self.rolling_hours,
            self.fap_threshold,
            self.consecutive_missing_threshold,
            self.global_threshold,
            self.drop_sparse,
            self.drop_anomalous,
            self.drop_aperiodic,
            self.drop_consecutives,
        )
        self.forecast = count_baseline(
            self.processed,
            self.days_in_past,
            self.days_in_future,
            self.ts_method,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            kern=self.kernel,
        )
        self.all_results = scan(self.forecast, self.grid_resolution)
        self.grid_results = database_results(self.all_results)

    def plot(self, metric="av_lhd_score_EBP"):
        """Plot animation plot from results"""
        if isinstance(self.grid_results, pd.DataFrame):
            visualise_results_from_database(self.grid_results, metric=metric)
        else:
            logging.info(" Results not populated. Call `run()` first.")

    def highest_region(self, metric='l_score_EBP'):
        """Return highest region"""
        if isinstance(self.all_results, pd.DataFrame):
            data = self.all_results.sort_values(metric, ascending=False)
            return data.iloc[0]
        logging.info("Results not populated. Call `run()` first.")

    def plot_region_time_series(self, rank=0, legend=False, metric='l_score_EBP'):
        if not isinstance(self.all_results, pd.DataFrame):
            raise TypeError('Run the scan first')
        data = self.all_results.sort_values(metric, ascending=False)
        region = make_region_from_res(data, rank=rank)
        plot_region_time_series(region, self.forecast, add_legend=legend)
    
    def plot_region_by_rank(self, rank=0, legend=False, metric='l_score_EBP'):
        if not isinstance(self.all_results, pd.DataFrame):
            raise TypeError('Run the scan first')
        data = self.all_results.sort_values(metric, ascending=False)
        plot_region_by_rank(rank, data, self.forecast, add_legend=legend)

    def model_settings(self):
        settings = self.__dict__.copy()
        del settings['readings']
        del settings['processed']
        del settings['forecast']
        del settings['all_results']
        del settings['grid_results']
        print(settings)

    def rerun_forecast(self):
        self.forecast = count_baseline(
            self.processed,
            self.days_in_past,
            self.days_in_future,
            self.ts_method,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            kern=self.kernel,
        )

    def rerun_scan(self):
        # Assumes everything remains the same up to scanning
        print('Using cached processed and forecast data to rebuild scan')
        self.all_results = scan(self.forecast, self.grid_resolution)
        self.grid_results = database_results(self.all_results)
