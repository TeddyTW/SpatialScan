import numpy as np
import pandas as pd
from numpy.random import normal


def synthetic_detector(
    Y: pd.Series, noise_percentage: float = 10, dow_percentage: float = 5
) -> np.array:
    """
    Creates a synthetic count based on real counts for the means of simualting outbreak. Takes
    in the counts for a single detectors, and outputs a 24 hour periodic trace, with amplitude equal 
    to the 90th percentile of actual count data with added noise. Also includes day of the week component
    
    Args:
        Y: Series of count data for a given detector
        noise_percentage: percentage of random noise to add to the snthetic data, default 10%
        dow_percentage: strength of day of week component. 10% default means it varies trace by 10%

    Returns: np.array of synthetic counts
    """

    Y = Y.to_numpy()
    X = np.arange(0, len(Y))
    noise = normal(0, noise_percentage / 100, len(Y))
    S = (
        np.percentile(Y, 90)
        * abs((np.sin((np.pi * X / 24)) ** 2 + noise))
        * ((dow_percentage / 100) * np.sin((np.pi * X / 168)) ** 2 + 1)
    + np.percentile(Y, 3)).astype(int)
    return S


def synthetic_SCOOT(
    df: pd.DataFrame, noise_percentage: float = 10, dow_percentage: float = 10
) -> pd.DataFrame:
    """
    Creates a synthetic SCOOT dataframe based on a real input dataframe.
    Args:
        df: input SCOOT dataframe (preprocessed for best results)
        noise_percentage: percentage of random noise to add to the snthetic data, default 10%
        dow_percentage: strength of day of week component. 10% default means it varies trace by 10%

    Returns: Dataframe in SCOOT format, with synthetic data"""

    DF = df.set_index(["detector_id", "measurement_end_utc"])
    X = DF.groupby(level="detector_id")["n_vehicles_in_interval"].apply(
        lambda x: synthetic_detector(
            x, noise_percentage=noise_percentage, dow_percentage=dow_percentage
        )
    )
    DF["n_vehicles_in_interval"] = np.hstack(X.to_numpy())
    DF = DF.reset_index()
    return DF

