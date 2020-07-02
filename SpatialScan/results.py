"""Module to contain all functionality required to display results from the
main Expectation-Based Scan Statistic Loop"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def visualise_results(res_df, metric: str = "l_score_basic", smooth: bool = False, c_min: float = None, c_max: float = None) -> None:
    """Functionality which plots the animated results of the Spatial Scan.

    Args:
        res_df: resulting dataframe from `EBP()`
        metric: Which metric to plot from res_df
        smooth: Boolean which decides whether to smooth the spatial region in
                the animation or not.
        c_min: Minimum value to set the color bar
        c_max: Maximum value to set the color bar.
    """
    assert(set([metric])) <= set(res_df.columns)
    
        # Infer grid partition form the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    # Infer Spatial Grid extent from resulting dataframe
    x_min = res_df["x_min"].min()
    x_max = res_df["x_max"].max()
    y_min = res_df["y_min"].min()
    y_max = res_df["y_max"].max()
    t_min = res_df["t_min"].min()
    t_max = res_df["t_max"].max()

    # Re-create the grid used
    x_ticks = np.linspace(x_min, x_max, grid_partition + 1)
    y_ticks = np.linspace(y_min, y_max, grid_partition + 1)
    t_ticks = pd.date_range(start=t_min, end=t_max, freq="H")

    # Use these to explicitly return labels
    x_labels = [
        "{0:.3f}".format((x_ticks[i] + x_ticks[i + 1]) / 2)
        for i in range(len(x_ticks) - 1)
    ]
    y_labels = [
        "{0:.3f}".format((y_ticks[i] + y_ticks[i + 1]) / 2)
        for i in reversed(range(len(y_ticks) - 1))
    ]

    t_labels = [x.strftime("%I%p, %d %b %y") for x in t_ticks]

    
    # Find array of ordered dates over prediction period and format pretty for
    # plot.
    res_df = res_df.sort_values(by=["t_max"])

    scores_array = []
    global_max = -np.inf
    global_min = np.inf
    for t in range(len(t_ticks) - 1):
        x_array = []
        for j in range(len(y_ticks) - 1):
            y_array = []
            for i in range(len(x_ticks) - 1):

                sub_df = res_df[
                    (res_df["x_min"] <= x_ticks[i])
                    & (res_df["x_max"] >= x_ticks[i + 1])
                    & (res_df["y_min"] <= y_ticks[j])
                    & (res_df["y_max"] >= y_ticks[j + 1])
                    & (res_df["t_min"] == t_ticks[0])
                    & (res_df["t_max"] == t_ticks[t + 1])
                ]

                l_score = sub_df[metric].mean()

                if l_score > global_max:
                    global_max = l_score
                if l_score < global_min:
                    global_min = l_score

                y_array.append(l_score)
            x_array.insert(0, y_array)
        scores_array.append(x_array)
    
    
    zsmooth = "best" if smooth else None
    c_min = global_min if c_min is None else c_min
    c_max = global_max if c_max is None else c_max

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=scores_array[0],
                x=x_labels,
                y=y_labels,
                zmin=c_min,
                zmax=c_max,
                zsmooth=zsmooth,
                colorbar={"title": "Average Likelihood Ratio Score"},
            )
        ],
        layout=go.Layout(
            title="{} to {}".format(t_labels[0], t_labels[1]),
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
                data=[go.Heatmap(z=scores_array[i], zmin=c_min, zmax=c_max)],
                layout=go.Layout(title="{} to {}".format(t_labels[0], t_labels[i])),
            )
            for i in range(1, len(t_labels) - 1)
        ],
    )
    fig.update_layout(
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    fig.show()


def database_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Functionality to produce a dataframe in the correct format for storage 
    in the database. It may not be what we want perfectly atm, but template
    functionality is in place.
    Args:
        res_df: Resulting dataframe from `EBP()`
    Returns:
        pd.DataFrame in format for storage.
    """

    # Infer grid partition form the resulting dataframe
    grid_partition = len(res_df["x_min"].unique())

    # Infer Spatial Grid extent from resulting dataframe
    x_min = res_df["x_min"].min()
    x_max = res_df["x_max"].max()
    y_min = res_df["y_min"].min()
    y_max = res_df["y_max"].max()
    t_min = res_df["t_min"].min()
    t_max = res_df["t_max"].max()

    # Re-create the grid used
    x_ticks = np.linspace(x_min, x_max, grid_partition + 1)
    y_ticks = np.linspace(y_min, y_max, grid_partition + 1)
    t_ticks = pd.date_range(start=t_min, end=t_max, freq="H")

    return_dict = {}
    num_regions = 0
    for t in range(len(t_ticks) - 1):
        num_spatial_regions = 0
        for j in range(len(y_ticks) - 1):
            for i in range(len(x_ticks) - 1):

                sub_df = res_df[
                    (res_df["x_min"] <= x_ticks[i])
                    & (res_df["x_max"] >= x_ticks[i + 1])
                    & (res_df["y_min"] <= y_ticks[j])
                    & (res_df["y_max"] >= y_ticks[j + 1])
                    & (res_df["t_min"] == t_ticks[0])
                    & (res_df["t_max"] == t_ticks[t + 1])
                ]

              
                B, C = sub_df[["B_in", "C_in"]].sum()
            
                means = sub_df[["l_score_basic",
                                "l_score_000",
                                "l_score_025",
                                "l_score_050",
                                "l_score_075",
                                "l_score_100",
                               ]].mean()
              
                return_dict[num_regions] = {
                    
                    "start_time_utc": t_ticks[0],
                    "end_time_utc": t_ticks[t + 1],
                    "point_id": num_spatial_regions,
                    "observed_count": C,
                    "forecasted_count": B,
                    "lhd_score_basic_av": means['l_score_basic'],
                    "lhd_score_eps_000_av": means['l_score_000'],
                    "lhd_score_eps_025_av": means['l_score_025'],
                    "lhd_score_eps_050_av": means['l_score_050'],
                    "lhd_score_eps_075_av": means['l_score_075'],
                    "lhd_score_eps_100_av": means['l_score_100'],
                }
                
                num_spatial_regions += 1
                num_regions += 1
    return pd.DataFrame.from_dict(return_dict, "index")