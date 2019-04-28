"""
calculation of similarity indices
"""
import networkx as nx
from scipy import sparse
import numpy as np


class SimilarityIndex:
    """
    Container for Similarity Indices of a Graph
    """

    def __init__(self, sim_dict):
        self.sim_dict = sim_dict

    def to_features_labels(self, pos_edges, neg_edges):
        """
        Convert to features (link probability) and labels (true links)
        :param pos_edges: set of positive future edges (list of two-tuples)
        :param neg_edges: set of negative future edges
        :return: X (normalized similarity index), y (true links)
        """
        n_edges = len(pos_edges) + len(neg_edges)

        X = np.empty(shape=n_edges, dtype=np.float64)
        y = np.empty_like(X, dtype=np.bool)

        for i, edge in enumerate(pos_edges + neg_edges):
            X[i] = self.sim_dict[edge]
            y[i] = (i < len(pos_edges))

        denom = X.max() if X.max() == X.min() else (X.max() - X.min())
        X = (X - X.min()) / denom  # scale X to be between 0 and 1
        return X, y

    @classmethod
    def from_matrix(cls, mat, edges, nodes):
        """

        :param mat:
        :param edges:
        :param nodes:
        :return:
        """
        out = dict()

        for i, j in edges:
            i_idx = nodes.index(i)
            j_idx = nodes.index(j)

            out[i, j] = mat[i_idx, j_idx]

        return cls(out)


def common_neighbors(G, edge_list):
    """
    CN index
    Parameters
    ----------
    G
        Input Graph
    edge_list
        List of edges for which CN index should be calculated

    Returns
    -------
        SimilarityIndex

    """
    node_list = list(G.nodes)
    A = nx.adj_matrix(G)
    A_sq = A.dot(A)
    return SimilarityIndex.from_matrix(A_sq, edge_list, node_list)


def local_path(G, epsilon, degree, edge_list):
    node_list = list(G.nodes())
    A = nx.adj_matrix(G)
    out = sparse.csr_matrix(A.shape)
    for i in range(2, degree + 1):
        out += epsilon ** (i - 2) * (A ** i)

    return SimilarityIndex.from_matrix(out, edge_list, node_list)


def pref_attach(G, edge_list):
    out = dict()
    for i, j in edge_list:
        pa = G.out_degree(i) * G.in_degree(j)

        out[i, j] = pa

    return SimilarityIndex(out)


def _directed_common_neighbors(G, n1, n2):
    """
    returns set of common neighbors of n1 (successors) and n2 (predecessors) of a DiGraph
    """

    n1_neighbors = G.neighbors(n1)
    n2_neighbors = G.predecessors(n2)

    common_neighbors = set(n1_neighbors).intersection(set(n2_neighbors))

    return common_neighbors


def adamic_adar(G, edge_list):
    out = dict()

    for i, j in edge_list:

        aa = 0

        common_neighbors = _directed_common_neighbors(G, i, j)
        for neighbor in common_neighbors:
            log_degree = np.log(G.degree(neighbor))
            aa += (1 / log_degree)

        out[i, j] = aa

    return SimilarityIndex(out)


SIMILARITY_INDICES = {'CN': common_neighbors,
                      'LP': local_path,
                      'PA': pref_attach,
                      'AA': adamic_adar}
