"""Network Scan functionality"""
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import Point
import networkx as nx

from SpatialScan.network.path_generator import PathGenerator
from SpatialScan.likelihood import likelihood_ratio


class NetworkScan(PathGenerator):

    """Functionality to carry out Scan on Road Networks"""

    def __init__(
        self,
        forecast,
        borough="Westminster",
        min_path_length=50,
        max_path_length=500,
        **kwargs
    ):
        self.forecast = forecast
        self.borough = borough
        self.borough_file_path = "../../data/ESRI/London_Borough_Excluding_MHW.shp"

        logging.info("Setting up Network Scan")
        self.borough_polygon = get_borough_polygon(self.borough, self.borough_file_path)
        self.network = get_network_from_polygon(self.borough_polygon)
        logging.info("Restricting readings to network")
        self.network_forecast = restrict_readings_to_network(
            self.network, self.forecast
        )
        detector_edges = self.get_detector_edges()
        super().__init__(
            self.network, min_path_length, max_path_length, detector_edges, **kwargs
        )
        logging.info(
            "Calculating paths of lengths between %f and %f",
            self.min_path_length,
            self.max_path_length,
        )
        self.generate_paths()

        self.t_min = forecast["measurement_start_utc"].min()
        self.t_max = forecast["measurement_end_utc"].max()
        self.agg_df = None
        self.results = None
        self.agg_results = None
        logging.info("Setup complete.")

    def get_detector_edges(self):
        """Return edges of network with active detectors"""
        detector_edges = self.network_forecast.drop_duplicates(
            subset=["nearest_u", "nearest_v"]
        )[["nearest_u", "nearest_v"]]

        detector_edges = detector_edges.astype(int)

        return list(zip(detector_edges.nearest_u, detector_edges.nearest_v))

    def aggregate_to_edges(self):
        """Sum vehicle counts per edge"""
        self.network, self.agg_df = aggregate_edge_event_count(
            self.network, self.network_forecast, self.detector_edges
        )

    def scan(self):
        """Carry out Network Scan, and calculte metrics"""

        t_ticks = pd.date_range(start=self.t_min, end=self.t_max, freq="H")

        return_dict = {}
        path_count = 0

        for time, tick in enumerate(t_ticks[:-1]):
            for source in self.network_paths.keys():
                targets = self.network_paths[source].keys()
                for target in targets:

                    base, count = path_event_count(
                        self.network,
                        self.network_paths[source][target],
                        time,
                        len(t_ticks) + 1,
                    )

                    likelihood = likelihood_ratio(base, count)

                    return_dict[path_count] = {
                        "source": source,
                        "target": target,
                        "base": base,
                        "count": count,
                        "score": likelihood,
                        "path": self.network_paths[source][target],
                        "path_graph": self.network.subgraph(
                            nodes=self.network_paths[source][target]
                        ),
                        "measurement_start_utc": tick,
                        "measurement_end_utc": self.t_max,
                    }
                    path_count += 1

            if time % 8 == 0:
                logging.info("Scan progress: %.2f%%", (time + 1) * 100 / len(t_ticks))

        logging.info("%d space-time paths searched", path_count)

        self.results = pd.DataFrame.from_dict(return_dict, "index")

    def highest_path(self):
        """Path with highest EBP score"""
        return self.results.iloc[0]["path"]

    def aggregate_results_to_edges(self):
        """Calculate mean EBP score per edge"""

        t_ticks = pd.date_range(start=self.t_min, end=self.t_max, freq="H")

        agg_network = self.network.copy()

        num_aggs = 0

        # Set Defaults
        for time in t_ticks[:-1]:
            nx.set_edge_attributes(agg_network, 1, "score_t={}".format(time))

        agg_dict = {}
        for i, (source, target) in enumerate(self.detector_edges):

            spatial_cond = (
                self.results["path"]
                .astype(str)
                .str.contains("{}, {}".format(source, target))
            ) | (
                self.results["path"]
                .astype(str)
                .str.contains("{}, {}".format(target, source))
            )

            for tick in t_ticks[:-1]:

                cond = spatial_cond & (self.results["measurement_start_utc"] <= tick)
                sub_df = self.results[cond]

                if sub_df.empty:  # TODO - if done properly this shouldnt happen
                    continue

                score = sub_df["score"].mean()

                agg_dict[num_aggs] = {
                    "source": source,
                    "target": target,
                    "measurement_start_utc": tick,
                    "measurement_end_utc": self.t_max,
                    "score": score,
                }
                num_aggs += 1

                nx.set_edge_attributes(
                    agg_network,
                    {(source, target, 0): score},
                    name="score_t={}".format(tick),
                )

            if i % 50 == 0:
                logging.info(
                    "Aggregation progress: %.2f%%", i * 100 / len(self.detector_edges)
                )

        self.agg_results = pd.DataFrame.from_dict(agg_dict, "index")
        self.network = agg_network


def get_borough_polygon(
    borough="Westminster",
    boroughs_file="../../data/ESRI/London_Borough_Excluding_MHW.shp",
):
    """Fetch borough polygon information"""

    if not os.path.isfile(boroughs_file):
        raise ValueError("File does not exist")

    london_df = gpd.read_file(boroughs_file)
    london_df = london_df.to_crs(epsg=4326)
    return london_df[london_df["NAME"] == borough].iloc[0]["geometry"]


def get_network_from_polygon(polygon):
    """Fetch network within polygon bounds from osmnx"""

    # Roads of Interest
    roi = '["highway"~"motorway|motorway_link|primary|'
    roi += 'primary_link|secondary|secondary_link|trunk|trunk_link|tertiary|tertiary_link"]'
    network = ox.graph_from_polygon(
        polygon, network_type="drive", simplify=True, custom_filter=roi
    )
    return nx.MultiGraph(network)  # Note - not directional


def restrict_readings_to_polygon(forecast, polygon):

    """"Keep readings which lie within polygon"""

    detectors = forecast.drop_duplicates(subset=["lon", "lat"], keep="first").copy()
    detectors["location"] = detectors.apply(lambda x: Point(x.lon, x.lat), axis=1)
    detectors["geom"] = polygon

    intersect_dets = detectors[
        detectors.apply(lambda x: x.geom.contains(x.location), axis=1)
    ]["detector_id"]
    return forecast[forecast["detector_id"].isin(intersect_dets)]


def restrict_readings_to_network(network, readings, max_dist=5e-4):

    """Keep readings which lie close to network of interest"""

    detectors = readings.drop_duplicates(subset=["lon", "lat"], keep="first").copy()

    arr = detectors.apply(
        lambda x: ox.distance.get_nearest_edge(
            network, (x.lat, x.lon), return_dist=True
        ),
        axis=1,
    )

    detectors["nearest_u"] = arr.apply(lambda x: x[0])
    detectors["nearest_v"] = arr.apply(lambda x: x[1])
    detectors["dist"] = arr.apply(lambda x: x[3])

    detectors = detectors[detectors["dist"] <= max_dist]
    detectors = detectors[
        ["detector_id", "lon", "lat", "nearest_u", "nearest_v", "dist"]
    ]

    network_readings = readings.merge(
        detectors, how="left", on=["detector_id", "lon", "lat"]
    )
    return network_readings.dropna()


def aggregate_edge_event_count(network, network_df, non_zero_edges):

    """Aggregate sum of vehicle counts to each edge"""

    # Aggregates from detector level to grid=cell level
    # stores in the graph and in the dataframe
    # Update to be class G method

    agg_network = network.copy()

    edge_count_dict = {}
    num_edges = 0

    t_min = network_df["measurement_start_utc"].min()
    t_max = network_df["measurement_end_utc"].max()

    t_ticks = pd.date_range(start=t_min, end=t_max, freq="H")

    nx.set_edge_attributes(agg_network, [], "counts")
    nx.set_edge_attributes(agg_network, [], "baselines")

    for i, (source, target) in enumerate(non_zero_edges):
        baselines = []
        counts = []
        for tick in t_ticks[:-1]:
            edge_df = network_df[
                (network_df["nearest_u"] == source)
                & (network_df["nearest_v"] == target)
                & (network_df["measurement_start_utc"] == tick)
                & (network_df["measurement_end_utc"] == tick + np.timedelta64(1, "h"))
            ]

            edge_count = edge_df["count"].sum() / 1e6
            edge_baseline = edge_df["baseline"].sum() / 1e6

            baselines.append(edge_baseline)
            counts.append(edge_count)

            edge_count_dict[num_edges] = {
                "source": source,
                "target": target,
                "measurement_start_utc": tick,
                "measurement_end_utc": t_max,
                "count": edge_count,
                "baseline": edge_baseline,
            }

            num_edges += 1

        nx.set_edge_attributes(
            agg_network, {(source, target, 0): baselines}, name="baselines"
        )
        nx.set_edge_attributes(
            agg_network, {(source, target, 0): counts}, name="counts"
        )

        if i % 50 == 0:
            logging.info("Aggregation progress: %.2f%%", i * 100 / len(non_zero_edges))
    return agg_network, pd.DataFrame.from_dict(edge_count_dict, "index")


def path_event_count(network, path: list, t_min, t_max):

    """Sum vehicle counts on a given path"""

    baseline = 0
    count = 0

    for i, _ in enumerate(path[:-1]):

        try:
            edge = network[path[i]][path[i + 1]]
        except KeyError:
            edge = network[path[i + 1]][path[i]]

        baseline += sum(edge[0]["baselines"][t_min:t_max])
        count += sum(edge[0]["counts"][t_min:t_max])
    return baseline, count
