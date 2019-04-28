"""
Train-Test-Split
"""
import networkx as nx
import numpy as np

from .util import edgelist_sorted_by_timestamp, largest_component
from sklearn.utils import check_random_state


class SplitNotPossibleException(Exception):
    """indicates split was not possible"""
    pass


def feature_and_label_graph(G, start_edge, n_edges_feature, n_edges_label):
    """
    Sample feature and label graph from original graph (implementation of
    algorithm XY)

    Parameters
    ----------
    G : nx.DiGraph
        Input Graph to sample feature and label graph from
    start_edge : int
        Edge to start sampling from. Must satisfy 0 <= `start_edge`
        < nE(G)
    n_edges_feature: int
        Number of edges to sample into feature graph
    n_edges_label:
        Number of edges to sample into label graph

    Returns
    -------
    tuple(Graph, Graph)
        Tuple of sampled feature and label graph

    Raises
    -------
    SplitNotPossibleExcecption
        if split is not possible (due to badly chosen parameters)

    """

    Gf = nx.DiGraph()
    edges_sorted = edgelist_sorted_by_timestamp(G)
    edge_nr = start_edge

    while largest_component(Gf).number_of_edges() < n_edges_feature:
        n_missing = n_edges_feature - largest_component(
            Gf).number_of_edges()
        if edge_nr + n_missing > len(edges_sorted):
            raise SplitNotPossibleException(
                'Not enough edges left to put in feature graph. '
                'Try earlier start edge or sampling less edges.')

        Gf.add_edges_from(edges_sorted[edge_nr:(edge_nr + n_missing)])
        edge_nr += n_missing

    Gf = largest_component(Gf)
    Gl = nx.DiGraph()
    Gl.add_nodes_from(Gf)

    Gl_edges = []

    while len(Gl_edges) < n_edges_label:
        u, v = edges_sorted[edge_nr]

        if u in Gf and v in Gf:
            Gl_edges.append((u, v))

        edge_nr += 1
        if edge_nr >= len(edges_sorted):
            raise SplitNotPossibleException(
                'Not enough edges left to put in label graph. '
                'Try earlier start edge or sampling less edges.')
    Gl_edges = [(u, v, dict(order=i)) for (i, (u, v)) in enumerate(
        Gl_edges)]  # retaining insertion order of edges (for train-test-split)
    Gl.add_edges_from(Gl_edges)

    return Gf, Gl


def graph_split(G, nE_feature, nE_label, offset, how):
    out_dict = dict.fromkeys(how)
    for i, key in enumerate(out_dict):
        offset_ = offset * i
        try:
            feature, label = feature_and_label_graph(G, offset_, nE_feature,
                                                     nE_label)
        except SplitNotPossibleException:
            raise SplitNotPossibleException(f'{key}-split not possible.')
        else:
            out_dict[key] = feature, label

    return out_dict


def nested_edge_split(G, pe, ne, sizes, rs=None):
    """ split positive and negative edges into train-val and test edges

    Parameters
    ----------
    G : Graph
        input label graph. Must have edges with `order` attribute that indicates
        temporal ordering of edges
    pe : edgelist
        list of positive edges
    ne : edgelist
        list of negative edges
    sizes : list
        relative sizes of splits.
    rs
        random state seed

    Returns
    -------
    pe_splits : list
        list (same length as sizes) of splits of positive edges. Split is done
        acc. to timestamp
    ne_splits : list
        list (same length as sizes) of splits of negative edges. Split is done
        randomly
    """

    rs = check_random_state(rs)

    sizes = [s / sum(sizes) for s in sizes]

    pe_sorted = sorted(pe, key=lambda x: G.get_edge_data(*x)['order'])
    ne_shuffled = ne.copy()

    ne_shuffled = [ne_shuffled[i] for i in
                   rs.choice(np.arange(len(ne)), size=len(ne), replace=False)]

    breaks = np.cumsum([0] + sizes)

    pe_splits = []
    ne_splits = []

    for i in range(len(sizes)):
        pe_from = int(breaks[i] * len(pe))
        pe_to = int(breaks[i + 1] * len(pe))

        ne_from = int(breaks[i] * len(ne))
        ne_to = int(breaks[i + 1] * len(ne))

        pe_splits.append(pe_sorted[pe_from:pe_to])
        ne_splits.append(ne_shuffled[ne_from:ne_to])

    return pe_splits, ne_splits
