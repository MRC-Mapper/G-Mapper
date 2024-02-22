import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from abc import ABC
from cover import Cover
from oracle import _check_clustering_object, cluster_points, map_overlap_cluster_to_interval
from graph import EnhancedGraph, Graph, AbstractGraph
from node import Sign, EnhancedNode, Node


def generate_mapper_graph(X: np.ndarray, lens: np.ndarray, cover: Cover, clusterer: object, enhanced: bool = False, cover_min: float = None, cover_max: float = None, refit_cover=True):
    # Quick checks to fail if input is malformed
    _check_clustering_object(clusterer)
  
    if refit_cover:
        cover.force_refit()
    if cover_min is not None and cover_max is not None and refit_cover:
        cover.compute_intervals(cover_min, cover_max)
    
    interval_clusterings: List[List[int]] = []
    graph: AbstractGraph = EnhancedGraph() if enhanced else Graph()
    interval_clusterings = []

    for i, interval_members in enumerate(cover.fit_intervals(lens)):
        interval_clusterings.append([])
        if len(interval_members) == 0:
            continue
        assignments: np.ndarray = cluster_points(X[interval_members], clusterer)
        num_clusters: int = assignments.max() + 1 # if 3 is a cluster, then there are 4 clusters: see dbscan sklearn docs
        for cluster in range(num_clusters):
            cluster_members = interval_members[assignments == cluster]
            interval_clusterings[i].append(cluster_members)
            if len(cluster_members) == 0:
                continue
            if enhanced:
                graph: EnhancedGraph
                positive_node = EnhancedNode(interval_index=i, cluster_index=cluster, sign=Sign.PLUS, members=cluster_members)
                negative_node = EnhancedNode(interval_index=i, cluster_index=cluster, sign=Sign.MINUS, members=cluster_members)
                graph.add_node(positive_node)
                graph.add_node(negative_node)
                graph.add_edge(positive_node, negative_node)

                # assign positive node value
                if i+1 < cover.num_intervals:
                    graph.set_function_val(positive_node, cover[i+1][0])
                else:
                    graph.set_function_val(positive_node, cover[i][1])

                # assign negative node value
                if 0 <= i-1:
                    graph.set_function_val(negative_node, cover[i-1][1])
                else:
                    graph.set_function_val(negative_node, cover[i][0])
            else:
                graph: Graph
                node = Node(i, cluster, cluster_members)
                graph.add_node(node)
                # Modified - add graph.set_mean_val.
                graph.set_mean_val(node, np.mean(lens[cluster_members]))
                if i > 0: # beyond first interval
                    for j in range(0,i):
                        lower_interval_clusters = interval_clusterings[j]
                        map_down = map_overlap_cluster_to_interval(cluster_members, lower_interval_clusters)
                        for down in map_down:
                            node2 = graph.get_node(j, down)
                            graph.add_edge(node, node2)

                            

    if enhanced:
        graph: EnhancedGraph
        for i, overlap_members in enumerate(cover.fit_overlaps(lens)):
            if len(overlap_members) == 0:
                continue
            assignments = cluster_points(X[overlap_members], clusterer)
            lower_interval_clusters = interval_clusterings[i]
            upper_interval_clusters = interval_clusterings[i+1]
            num_clusters = assignments.max() + 1
            for cluster in range(num_clusters):
                cluster_members = overlap_members[assignments == cluster]
                map_down = map_overlap_cluster_to_interval(cluster_members, lower_interval_clusters)
                map_up = map_overlap_cluster_to_interval(cluster_members, upper_interval_clusters)
                for down in map_down:
                    down_node = graph.get_node(i, down, Sign.PLUS)
                    for up in map_up:
                        up_node = graph.get_node(i+1, up, Sign.MINUS)
                        graph.add_edge(down_node, up_node)

    return graph


# For mapper interactive
def generate_lens(X, proj, scale = "MinMax"):
    if scale == 'MinMax':
        scaler = MinMaxScaler((0, 1), copy=True)
    else:
        scaler = None

    if proj == "sum":
        ret = np.sum(X, axis=1)
    elif proj == "mean":
        ret = np.mean(X, axis=1)
    elif proj == "median":
        ret = np.median(X, axis=1)
    elif proj == "max":
        ret = np.max(X, axis=1)
    elif proj == "min":
        ret = np.min(X, axis=1)
    elif proj == "std":
        ret = np.std(X, axis=1)
    elif proj == "l2norm":
        ret = np.linalg.norm(X, axis=1)

    print("Scaler term", scaler)

    return ret if scaler is None else scaler.fit_transform(np.reshape(ret, (-1, 1)))

# 2-dimensional lens
def fit_cubes(lens: np.ndarray,  cover0: Cover, cover1: Cover):
    indices = np.arange(len(lens))
    for i in range(len(cover0.intervals)):
        for j in range(len(cover1.intervals)):
            yield indices[(cover0.intervals[i,0] <= lens[:,0]) & (lens[:,0] <= cover0.intervals[i,1]) & (cover1.intervals[j,0] <= lens[:,1]) & (lens[:,1] <= cover1.intervals[j,1])]
            
def generate_mapper_graph_2d(X: np.ndarray, lens: np.ndarray, cover0: Cover, cover1: Cover, clusterer: object, enhanced: bool = False, cover_min: float = None, cover_max: float = None, refit_cover=True):

    cube_clusterings: List[List[int]] = []
    graph: AbstractGraph = EnhancedGraph() if enhanced else Graph()
    cube_clusterings = []
    
    for i, cube_members in enumerate(fit_cubes(lens,cover0,cover1)):
        cube_clusterings.append([])
        if len(cube_members) == 0:
            continue
        assignments: np.ndarray = cluster_points(X[cube_members], clusterer)
        num_clusters: int = assignments.max() + 1 # if 3 is a cluster, then there are 4 clusters: see dbscan sklearn docs
        for cluster in range(num_clusters):
            cluster_members = cube_members[assignments == cluster]
            cube_clusterings[i].append(cluster_members)
            if len(cluster_members) == 0:
                continue
            if enhanced:
                graph: EnhancedGraph
                positive_node = EnhancedNode(cube_index=i, cluster_index=cluster, sign=Sign.PLUS, members=cluster_members)
                negative_node = EnhancedNode(cube_index=i, cluster_index=cluster, sign=Sign.MINUS, members=cluster_members)
                graph.add_node(positive_node)
                graph.add_node(negative_node)
                graph.add_edge(positive_node, negative_node)

                # assign positive node value
                if i+1 < cover.num_intervals:
                    graph.set_function_val(positive_node, cover[i+1][0])
                else:
                    graph.set_function_val(positive_node, cover[i][1])

                # assign negative node value
                if 0 <= i-1:
                    graph.set_function_val(negative_node, cover[i-1][1])
                else:
                    graph.set_function_val(negative_node, cover[i][0])
            else:
                graph: Graph
                node = Node(i, cluster, cluster_members)
                graph.add_node(node)
                # Modified - add graph.set_mean_val.
                graph.set_mean_val(node, np.mean(lens[cluster_members]))
                if i>0:
                    for j in range(0,i):
                        lower_cube_clusters = cube_clusterings[j]
                        map_down = map_overlap_cluster_to_interval(cluster_members, lower_cube_clusters)
                        for down in map_down:
                            node2 = graph.get_node(j, down)
                            graph.add_edge(node, node2)

    if enhanced:
        graph: EnhancedGraph
        for i, overlap_members in enumerate(cover.fit_overlaps(lens)):
            if len(overlap_members) == 0:
                continue
            assignments = cluster_points(X[overlap_members], clusterer)
            lower_interval_clusters = interval_clusterings[i]
            upper_interval_clusters = interval_clusterings[i+1]
            num_clusters = assignments.max() + 1
            for cluster in range(num_clusters):
                cluster_members = overlap_members[assignments == cluster]
                map_down = map_overlap_cluster_to_interval(cluster_members, lower_interval_clusters)
                map_up = map_overlap_cluster_to_interval(cluster_members, upper_interval_clusters)
                for down in map_down:
                    down_node = graph.get_node(i, down, Sign.PLUS)
                    for up in map_up:
                        up_node = graph.get_node(i+1, up, Sign.MINUS)
                        graph.add_edge(down_node, up_node)

    return graph



