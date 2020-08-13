"""Generate paths on a osmnx network"""

import numpy as np
import networkx as nx


class PathGenerator:

    """Base class for generating paths on a network using networkx/dijkstra"""

    def __init__(
        self,
        network: nx.MultiGraph,
        min_path_length: float,
        max_path_length: float,
        detector_edges: list,
        drop_duplicates=True,
        drop_detectorless=True,
    ):
        self.network = network
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.detector_edges = detector_edges
        self.drop_duplicates = drop_duplicates
        self.drop_detectorless = drop_detectorless
        self.network_paths = {}

    def generate_paths(self):
        """Generate Paths"""
        self.network_paths = self._find_all_paths()
        if self.drop_duplicates:
            self._drop_duplicate_paths()
        if self.drop_detectorless:
            self._drop_detectorless_paths()

    def _find_all_paths(self) -> dict:
        """Find all possible paths with appropiate lengths on the network"""

        upper_paths = dict(
            nx.all_pairs_dijkstra_path(
                self.network, cutoff=self.max_path_length, weight="length"
            )
        )
        lower_paths = dict(
            nx.all_pairs_dijkstra_path(
                self.network, cutoff=self.min_path_length, weight="length"
            )
        )

        useful_paths = upper_paths.copy()

        for source in upper_paths.copy().keys():
            for target in upper_paths[source].copy().keys():
                if (
                    source in lower_paths.keys()
                    and target in lower_paths[source].keys()
                ):
                    del useful_paths[source][target]
        return useful_paths

    def _drop_detectorless_paths(self):
        """Drop paths without detectors on them"""

        for source in self.network_paths.copy().keys():
            for target in self.network_paths[source].copy().keys():

                path = self.network_paths[source][target]
                path_graph = self.network.subgraph(path)

                if not any(
                    path_graph.has_edge(u, v) or path_graph.has_edge(v, u)
                    for (u, v) in self.detector_edges
                ):
                    del self.network_paths[source][target]

        for source in self.network_paths.copy().keys():
            if not self.network_paths[source]:
                del self.network_paths[source]

    def _drop_duplicate_paths(self):
        """Drop duplicate paths ie. u -> v == v -> u"""

        seen_paths = set()
        for source in self.network_paths.copy().keys():
            for target in self.network_paths[source].copy().keys():

                if (source, target) in seen_paths:
                    del self.network_paths[target][source]
                elif source == target:
                    del self.network_paths[target][source]

                seen_paths.add((source, target))
                seen_paths.add((target, source))

        for source in self.network_paths.copy().keys():
            if not self.network_paths[source]:
                del self.network_paths[source]

    def num_paths(self):
        """Number of paths on the network"""
        if len(self.network_paths) == 0:
            raise TypeError("No Paths have been generated yet")
        count = 0
        for source in self.network_paths.keys():
            count += len(self.network_paths[source])
        return count

    def random_path(self):
        """Choose random path"""
        source = int(np.random.choice(list(self.network_paths.keys())))
        target = int(np.random.choice(list(self.network_paths[source].keys())))
        return self.network_paths[source][target]
