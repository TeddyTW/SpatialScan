"""Module to contain all functionality required to display results from the
main Expectation-Based Scan Statistic Loop"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def average_likelihood(res_df: pd.DataFrame) -> tuple:
    """Functionality to calculate the average likelihood values used to plot
    results from the main scan. 

    Args:
        res_df: pd.DataFrame from the main EBP function.

    Returns:
        scores_array: np.ndarray of shape (W x N x N) - i.e. containing 2D grid
                      information for each forecasted timestep
        x_ticks: Labels for x array values (longitude)
        y_ticks: Labels for y array values (Latitude)
        t_ticks: Labels for dates (Time)
        v_max: Maximum average likelihood Score from whole search region. USe this
               to calibrate the color bar when plotting.
    """

    # Infer grid partition form the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    # Infer Spatial Grid extent from resulting dataframe
    x_min = res_df["x_min"].min()
    x_max = res_df["x_max"].max()
    y_min = res_df["y_min"].min()
    y_max = res_df["y_max"].max()

    # Re-create the grid used
    x_ticks = np.linspace(x_min, x_max, grid_partition + 1)
    y_ticks = np.linspace(y_min, y_max, grid_partition + 1)

    # Use these to explicitly return labels
    x_labels = [
        "{0:.3f}".format((x_ticks[i] + x_ticks[i + 1]) / 2)
        for i in range(len(x_ticks) - 1)
    ]
    y_labels = [
        "{0:.3f}".format((y_ticks[i] + y_ticks[i + 1]) / 2)
        for i in reversed(range(len(y_ticks) - 1))
    ]

    # Find array of ordered dates over prediction period and format pretty for
    # plot.
    res_df = res_df.sort_values(by="t_max")
    res_df["datetime_nice"] = res_df["t_max"].apply(
        lambda x: x.strftime("%I%p, %d %b %y")
    )
    dates = res_df["t_max"].unique()
    t_labels = res_df["datetime_nice"].unique()

    scores_array = []
    global_max = 0
    for _, date in enumerate(dates):
        x_array = []
        for j in range(len(y_ticks) - 1):
            y_array = []
            for i in range(len(x_ticks) - 1):

                sub_df = res_df[
                    (res_df["x_min"] <= x_ticks[i])
                    & (res_df["x_max"] >= x_ticks[i + 1])
                    & (res_df["y_min"] <= y_ticks[j])
                    & (res_df["y_max"] >= y_ticks[j + 1])
                    & (res_df["t_max"] == date)
                ]

                l_score = sub_df["likelihood_score"].mean()

                if l_score > global_max:
                    global_max = l_score

                y_array.append(l_score)
            x_array.insert(0, y_array)
        scores_array.append(x_array)

    return np.array(scores_array), x_labels, y_labels, t_labels, global_max


def visualise_results(
    scores_array: np.ndarray,
    x_labels: np.ndarray,
    y_labels: np.ndarray,
    t_labels: np.ndarray,
    global_max: float,
    smooth: bool = False,
):
    """Functionality which plots the animated results of the Spatial Scan. To
    be used in conjunction with `average_likelihood`.

    Args:
        scores_array: np.ndarray of shape (W x N x N) - i.e. containing 2D grid
                      information for each forecasted timestep
        x_labels: Labels for x array values (longitude)
        y_labels: Labels for y array values (Latitude)
        t_labels: Labels for dates (Time)
        global_max: Maximum average likelihood Score from whole search region.
                    Used to calibrate the color bar when plotting. 
        smooth: Boolean which decides whether to smooth the spatial region in
                the animation or not.
    """

    zsmooth = "best" if smooth else None

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=scores_array[0],
                x=x_labels,
                y=y_labels,
                zmin=1,
                zmax=global_max,
                zsmooth=zsmooth,
                colorbar={"title": "Average Likelihood Ratio Score"},
            )
        ],
        layout=go.Layout(
            title=t_labels[0],
            width=800,
            height=500,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", args=[None]),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=scores_array[i])],
                layout=go.Layout(title=t_labels[i]),
            )
            for i in range(1, len(t_labels))
        ],
    )
    fig.update_layout(
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    fig.show()
