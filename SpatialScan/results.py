"""Module to contain all functionality required to display results from the
main Expectation-Based Scan Statistic Loop"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd


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
    metric: str = "l_score_EBP",
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
                    & (
                        res_df["t_max"]
                        == t_ticks[len(t_ticks) - 1 if scan_type == "normal" else t + 1]
                    )
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
                    & (
                        res_df["t_max"]
                        == t_ticks[len(t_ticks) - 1 if scan_type == "normal" else t + 1]
                    )
                ]

                B, C = sub_df[["B_in", "C_in"]].sum()

                means = sub_df[
                    [
                        "l_score_EBP_lower",
                        "l_score_EBP",
                        "l_score_EBP_upper",
                        "l_score_000_lower",
                        "l_score_000",
                        "l_score_000_upper",
                        "l_score_025_lower",
                        "l_score_025",
                        "l_score_025_upper",
                        "l_score_050_lower",
                        "l_score_050",
                        "l_score_050_upper",
                        "posterior_bbayes",
                    ]
                ].mean()

                return_dict[num_regions] = {
                    "start_time_utc": t_ticks[t],
                    "end_time_utc": t_ticks[
                        len(t_ticks) - 1 if scan_type == "normal" else t + 1
                    ],
                    "point_id": num_spatial_regions,
                    "x_min": x_ticks[i],
                    "x_max": x_ticks[i + 1],
                    "y_min": y_ticks[j],
                    "y_max": y_ticks[j + 1],
                    "observed_count": C,
                    "forecasted_count": B,
                    "av_lhd_score_EBP_lower": means["l_score_EBP_lower"],
                    "av_lhd_score_EBP": means["l_score_EBP"],
                    "av_lhd_score_EBP_upper": means["l_score_EBP_upper"],
                    "av_lhd_score_eps_000_lower": means["l_score_000_lower"],
                    "av_lhd_score_eps_000": means["l_score_000"],
                    "av_lhd_score_eps_000_upper": means["l_score_000_upper"],
                    "av_lhd_score_eps_025_lower": means["l_score_025_lower"],
                    "av_lhd_score_eps_025": means["l_score_025"],
                    "av_lhd_score_eps_025_upper": means["l_score_025_upper"],
                    "av_lhd_score_eps_050_lower": means["l_score_050_lower"],
                    "av_lhd_score_eps_050": means["l_score_050"],
                    "av_lhd_score_eps_050_upper": means["l_score_050_upper"],
                    "av_posterior_bbayes": means["posterior_bbayes"],
                }

                num_spatial_regions += 1
                num_regions += 1
    return pd.DataFrame.from_dict(return_dict, "index")


def visualise_results_from_database(
    database_df,
    metric: str = "av_lhd_score_EBP",
    smooth: bool = False,
    c_min: float = None,
    c_max: float = None,
):

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

    times = database_df["start_time_utc"].unique()
    grid_partition = len(database_df["x_min"].unique())

    x_min = database_df["x_min"].min()
    x_max = database_df["x_max"].max()
    y_min = database_df["y_min"].min()
    y_max = database_df["y_max"].max()
    t_min = database_df["start_time_utc"].min()
    t_max = database_df["end_time_utc"].max()

    print(
        "Dataframe contains data from the database spanning {} to {}.".format(
            t_min, t_max
        )
    )

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
    example_df = database_df[database_df["point_id"] == 0]

    t_min_ticks = example_df["start_time_utc"]
    t_max_ticks = example_df["end_time_utc"]
    t_min_labels = [x.strftime("%I%p, %d %b %y") for x in t_min_ticks]
    t_max_labels = [x.strftime("%I%p, %d %b %y") for x in t_max_ticks]

    scores_array = []
    for time in times:
        scores = database_df[database_df["start_time_utc"] == time][metric].to_numpy()
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
                layout=go.Layout(
                    title="{} to {}".format(t_min_labels[i], t_max_labels[i])
                ),
            )
            for i in range(0, len(t_min_labels))
        ],
    )
    fig.update_layout(
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    fig.show()
    return {"max": c_max, "min": c_min}


class MapboxPlot:
    def __init__(self, database_df, london_gpd, borough="all"):
        self.database_df = database_df
        self.london_gpd = london_gpd
        self.borough = borough
        # Mapbox Token
        self.token = "pk.eyJ1IjoiY2hhbmNlaGF5Y29jayIsImEiOiJja2Q0d25iNjMxYTgxMnNudzUzdm9veG5xIn0.MNo8CDYOo6z_g1lWiRM3vg"
        self.masked_results = None

    def _preprocess_database_results(self):
        # Add centroid
        self.database_df["lon"] = (
            self.database_df["x_min"] + self.database_df["x_max"]
        ) / 2
        self.database_df["lat"] = (
            self.database_df["y_min"] + self.database_df["y_max"]
        ) / 2

        # Shapely formatting
        self.database_df["location"] = self.database_df.apply(
            lambda x: Point(x.lon, x.lat), axis=1
        )

        # Project the london boroughs to standard
        self.london_gpd = self.london_gpd.to_crs(epsg=4326)

        if self.borough in ["All", "all", "London", "london"]:
            boundary = gpd.GeoSeries(unary_union(self.london_gpd.geometry))
        else:
            boundary = self.london_gpd[self.london_gpd["NAME"] == self.borough]

        # Does all time steps at the mo - inefficient
        self.database_df["in_borough"] = self.database_df.apply(
            lambda x: boundary.contains(x.location), axis=1
        )

        self.masked_results = self.database_df[self.database_df["in_borough"]]

    def display(self, metric, zmin=None, zmax=None):
        if not isinstance(self.masked_results, pd.DataFrame):
            self._preprocess_database_results()

        unique_times = self.masked_results.drop_duplicates(
            subset=["start_time_utc", "end_time_utc"]
        )

        t_min_ticks = unique_times["start_time_utc"]
        t_max_ticks = unique_times["end_time_utc"]
        t_min_labels = [x.strftime("%I%p, %d %b %y") for x in t_min_ticks]
        t_max_labels = [x.strftime("%I%p, %d %b %y") for x in t_max_ticks]

        global_min = self.masked_results[metric].min()
        global_max = self.masked_results[metric].max()

        zmin = global_min if zmin is None else zmin
        zmax = global_max if zmax is None else zmax

        start_time = unique_times["start_time_utc"].min()

        fig = go.Figure(
            data=go.Densitymapbox(
                lon=self.masked_results[
                    self.masked_results["start_time_utc"] == start_time
                ]["lon"],
                lat=self.masked_results[
                    self.masked_results["start_time_utc"] == start_time
                ]["lat"],
                z=self.masked_results[
                    self.masked_results["start_time_utc"] == start_time
                ][metric],
                colorscale="viridis",
                radius=100,
                opacity=0.5,
                showscale=True,
                zmin=global_min,
                zmax=global_max,
                colorbar={
                    "tickcolor": "white",
                    "tickfont": {"color": "white"},
                    "title": {"text": "Score", "font": {"color": "white"}},
                },
            ),
            layout=go.Layout(
                title="{} to {}".format(t_min_labels[0], t_max_labels[0]),
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
                    ),
                ],
            ),
            frames=[
                go.Frame(
                    data=[
                        go.Densitymapbox(
                            lon=self.masked_results[
                                self.masked_results["start_time_utc"]
                                == start_time + np.timedelta64(i, "h")
                            ]["lon"],
                            lat=self.masked_results[
                                self.masked_results["start_time_utc"]
                                == start_time + np.timedelta64(i, "h")
                            ]["lat"],
                            z=self.masked_results[
                                self.masked_results["start_time_utc"]
                                == start_time + np.timedelta64(i, "h")
                            ][metric],
                        )
                    ],
                    layout=go.Layout(
                        title="{} to {}".format(t_min_labels[i], t_max_labels[i])
                    ),
                )
                for i in range(0, len(t_min_labels))
            ],
        )
        fig.update_layout(
            mapbox_style="dark",
            mapbox_accesstoken=self.token,
            margin={"l": 0, "r": 0, "b": 0, "t": 60},
            mapbox={"center": {"lon": -0.09, "lat": 51.495}, "zoom": 9.7},
            autosize=False,
            width=1600,
            height=900,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # paper_bgcolor='black',
            title_font_color="white",
        )
        fig.show()
        return {"max": zmax, "min": zmin}
