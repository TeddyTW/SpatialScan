"""Main Class for Scan Statistics"""
import pandas as pd

from SpatialScan.preprocessing import data_preprocessor
from SpatialScan.timeseries import count_baseline
from SpatialScan.scan import scan
from SpatialScan.results import database_results, visualise_results_from_database


class ScanStatistic:
    """Simple helper class to reduce number of function calls when modelling"""
    def __init__(self, data, grid_resolution, days_in_past, days_in_future, ts_method):
        self.data = data
        self.grid_resolution = grid_resolution
        self.days_in_past = days_in_past
        self.days_in_future = days_in_future
        self.ts_method = ts_method
        self.all_results = None
        self.grid_results = None

    def run(self):
        """Build scan results"""
        proc_df = data_preprocessor(self.data)
        forecast = count_baseline(
            proc_df, self.days_in_past, self.days_in_future, self.ts_method
        )
        self.all_results = scan(forecast, self.grid_resolution)
        self.grid_results = database_results(self.all_results)

    def plot(self):
        """Plot animation plot from results"""
        if isinstance(self.grid_results, pd.DataFrame):
            visualise_results_from_database(self.grid_results)
        else:
            print("Call `run()` first.")
