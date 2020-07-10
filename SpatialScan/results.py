"""Module to contain all functionality required to display results from the
main Expectation-Based Scan Statistic Loop"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def title_generator(scan_type: str, i: int, t_labels: np.ndarray) -> str:
    """Quick utility function to create labels based on the tye of search.
    Accounts for the varying/non-varying t_max in both regimes.
    Args:
        scan_type: "normal" or "exhaustive"
        i: loop iteration
        t_labels: Array of t_tick labels for plotting
    Returns:
        String representing the appropiate label for graph.
    """
    if scan_type == "normal":
        return "{} to {}".format(t_labels[i], t_labels[len(t_labels) - 1])
    elif scan_type == "exhaustive":
        return "{} to {}".format(t_labels[i], t_labels[i + 1])
    return None

def visualise_results(
    res_df,
    metric: str = "l_score_basic",
    smooth: bool = False,
    c_min: float = None,
    c_max: float = None,
) -> None:
    """Functionality which plots the animated results of the Spatial Scan.

    Args:
        res_df: resulting dataframe from `EBP()`
        metric: Which metric to plot from res_df
        smooth: Boolean which decides whether to smooth the spatial region in
                the animation or not.
        c_min: Minimum value to set the color bar
        c_max: Maximum value to set the color bar.
    """
    assert (set([metric])) <= set(res_df.columns)

    # What type of scan was it? Normal or Exhaustive?
    # Only way to tell is by the number of unique t_maxs in res_df
    num_t_maxs = len(res_df["t_max"].unique())

    # If more than one t_max, scan was exhaustive
    if num_t_maxs > 1:
        scan_type = "exhaustive"
    else:
        scan_type = "normal"

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
                    & (res_df["t_min"] == t_ticks[t])
                    & (res_df["t_max"] == t_ticks[len(t_ticks) - 1  if scan_type == "normal" else t + 1])
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
            title="{} to {}".format(t_labels[0], t_labels[len(t_labels) - 1]),
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
                layout=go.Layout(title=title_generator(scan_type, i, t_labels)),
            )
            for i in range(0, len(t_labels) - 1)
        ],
    )
    fig.update_layout(
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    fig.show()
    return {"max": c_max, "min": c_min}


def database_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Functionality to produce a dataframe in the correct format for storage
    in the database. It may not be what we want perfectly atm, but template
    functionality is in place. Calculates the average likelihood per grid cell.
    Beware: this means different things depending on which function (`EBP()` or
    `EBP_exhaustive()`) was called.

    Args:
        res_df: Resulting dataframe from `EBP()` OR `EBP_exhaustive()`
    Returns:
        pd.DataFrame in format for storage.
    """

    # What type of scan was it? Normal or Exhaustive?
    # Only way to tell is by the number of unuique t_maxs in res_df
    num_t_maxs = len(res_df["t_max"].unique())

    # If more than one t_max, scan was exhaustive
    if num_t_maxs > 1:
        scan_type = "exhaustive"
    else:
        scan_type = "normal"

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
                    & (res_df["t_min"] == t_ticks[t])
                    & (res_df["t_max"] == t_ticks[len(t_ticks) - 1  if scan_type == "normal" else t + 1])
                ]

                B, C = sub_df[["B_in", "C_in"]].sum()

                means = sub_df[
                    [
                        "l_score_basic",
                        "l_score_000",
                        "l_score_025",
                        "l_score_050",
                        "l_score_075",
                        "l_score_100",
                    ]
                ].mean()

                return_dict[num_regions] = {
                    "start_time_utc": t_ticks[t],
                    "end_time_utc": t_ticks[len(t_ticks) - 1  if scan_type == "normal" else t + 1],
                    "point_id": num_spatial_regions,
                    "x_min": x_ticks[i],
                    "x_max": x_ticks[i + 1],
                    "y_min": y_ticks[j],
                    "y_max": y_ticks[j + 1],
                    "observed_count": C,
                    "forecasted_count": B,
                    "av_lhd_score_basic": means["l_score_basic"],
                    "av_lhd_score_eps_000": means["l_score_000"],
                    "av_lhd_score_eps_025": means["l_score_025"],
                    "av_lhd_score_eps_050": means["l_score_050"],
                    "av_lhd_score_eps_075": means["l_score_075"],
                    "av_lhd_score_eps_100": means["l_score_100"],
                }

                num_spatial_regions += 1
                num_regions += 1
    return pd.DataFrame.from_dict(return_dict, "index")


def visualise_results_from_database(
    database_df,
    metric: str = "av_lhd_score_basic",
    smooth: bool = False,
    c_min: float = None,
    c_max: float = None):
    
    """Allows reconstruction of the plot from any time slice of database data.
    the above make the plots directly from the output of `scan()`

    Args:
        database_df: dataframe from database storage
        metric: Which metric to plot from database_df
        smooth: Boolean which decides whether to smooth the spatial region in
                the animation or not.
        c_min: Minimum value to set the color bar
        c_max: Maximum value to set the color bar.
    """

    assert (set([metric])) <= set(database_df.columns)
    
    times = database_df['start_time_utc'].unique()
    grid_partition = len(database_df["x_min"].unique())
    
    x_min = database_df["x_min"].min()
    x_max = database_df["x_max"].max()
    y_min = database_df["y_min"].min()
    y_max = database_df["y_max"].max()
    t_min = database_df["start_time_utc"].min()
    t_max = database_df["end_time_utc"].max()
    
    print("Dataframe contains data from the database spanning {} to {}.".format(t_min, t_max))

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
    
    # Pick random time series to get correct dates
    # XXX - point Id will need to be changed eventually.
    example_df = database_df[database_df['point_id'] == 0]
  
    t_min_ticks = example_df['start_time_utc']
    t_max_ticks = example_df['end_time_utc']
    t_min_labels = [x.strftime("%I%p, %d %b %y") for x in t_min_ticks]
    t_max_labels = [x.strftime("%I%p, %d %b %y") for x in t_max_ticks]

    scores_array = []
    for time in times:
        scores = database_df[database_df['start_time_utc'] == time][metric].to_numpy()
        scores = np.reshape(scores, (grid_partition, grid_partition))
        
        # Reverse the array for plotting convention (low y is high)
        scores = scores[::-1]
        
        scores_array.append(scores)
        
    global_min = np.min(scores_array)
    global_max = np.max(scores_array)


    # Below is all plotting
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
            title="{} to {}".format(t_min_labels[0], t_max_labels[0]),
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
                layout=go.Layout(title="{} to {}".format(t_min_labels[i], t_max_labels[i])),
            )
            for i in range(0, len(t_min_labels))
        ],
    )
    fig.update_layout(
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    fig.show()
    return {"max": c_max, "min": c_min}