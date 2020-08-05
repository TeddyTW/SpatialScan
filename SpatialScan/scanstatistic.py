"""Main Class for Scan Statistics"""

import logging
import pandas as pd

from SpatialScan.preprocessing import Preprocessor
from SpatialScan.timeseries import Forecast
from SpatialScan.scan import scan
from SpatialScan.results import database_results, visualise_results_from_database


class ScanStatistic(Preprocessor, Forecast):
    """Simple helper class to reduce number of function calls when modelling"""

    def __init__(
        self,
        readings,
        grid_resolution,
        days_in_past,
        days_in_future,
        ts_method,
        *args,
        **kwargs
    ):
        Preprocessor.__init__(self)
        Forecast.__init__(self, days_in_past, days_in_future, ts_method)
        self.grid_resolution = grid_resolution
        self.readings = readings
        self.processed = None
        self.forecast = None
        self.all_results = None
        self.grid_results = None

    def run(self):
        """Build scan results"""
        self.processed = self.process(self.readings)
        self.forecast = self.predict(self.processed)
        self.all_results = scan(self.forecast, self.grid_resolution)
        self.grid_results = database_results(self.all_results)

    def plot(self, metric="av_lhd_score_EBP"):
        """Plot animation plot from results"""
        if isinstance(self.grid_results, pd.DataFrame):
            visualise_results_from_database(self.grid_results, metric=metric)
        else:
            logging.info(" Results not populated. Call `run()` first.")

    def highest_region(self):
        """Return highest region"""
        if isinstance(self.all_results, pd.DataFrame):
            return self.all_results.iloc[0]
        logging.info("Results not populated. Call `run()` first.")
