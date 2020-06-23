"""Module to contain all region and grid-based construction code."""

from typing import Type
import pandas as pd
from datetime import datetime

class Region:
    """Class to represent space-time region"""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                       t_min: datetime, t_max: datetime) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max
        self.label = None

    def __str__(self):
        return "({}, {}) x ({}, {}) x ({}, {})"\
         .format(self.x_min, self.x_max, self.y_min, self.y_max,
                 self.t_min, self.t_max)

    def add_label(self, label):
        self.label = label


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Check columns are in here.
    copy_df = df
    copy_df['measurement_start_utc'] = pd.to_datetime(df['measurement_start_utc'])
    copy_df['measurement_end_utc'] = pd.to_datetime(df['measurement_end_utc'])
    return copy_df

def region_event_count(S: Type[Region], data: pd.DataFrame) -> float:

    """Function to calculate the number of count (vehicles) within a given
    space-time region S. Can be used to calculate both the baseline count variable
    B and the actual count variable C in the likelihood ratio.
    Args:
        S: Space-Time Region to count events in
        data: Usual format SCOOT dataframe
    Returns: (float) Event count within region
    """
    region_mask = (data['lon'].between(S.x_min, S.x_max)) &\
                  (data['lat'].between(S.y_min, S.y_max)) &\
                  (data['measurement_end_utc'] > S.t_min) &\
                  (data['measurement_end_utc'] <= S.t_max)
    S_df = data.loc[region_mask]
    if S_df.empty:
        return 0
    return S_df['n_vehicles_in_interval'].sum()

