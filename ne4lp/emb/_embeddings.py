"""
Containers for Embeddings
"""
from collections import Sequence

import numpy as np
import networkx as nx


class NodeEmbeddings(object):
    """Container for NodeEmbeddings"""

    def __init__(self, G, d):
        self.d = d
        self._embs = dict()
        self.G = G

    def __getitem__(self, item):
        return self._embs[item]

    def __setitem__(self, key, value):
        assert key in self.G, ValueError(f'Node {key} not in the graph')
        self._validate_embedding(value)
        value = value.reshape(-1)  # flatten
        self._embs[key] = value

    def _is_complete(self):
        return set(self.G.nodes) == set(self._embs.keys())

    def _validate_embedding(self, embedding):
        assert isinstance(embedding, np.ndarray), TypeError(
            'Embedding is not a numpy array')
        assert embedding.size == self.d, ValueError(
            f'Wrong embedding size (expected {self.d}, got{embedding.size}')
        assert np.issubdtype(embedding.dtype, np.floating), TypeError(
            'Wrong dtype (expected float-type, got {embedding.dtype}')

    def combine_embeddings(self, node_1, node_2, combiner):
        """Combine embeddings of `node_1` and `node_2` using `combiner`"""
        return combiner(self[node_1], self[node_2])

    def to_features_labels(self, combiner, pos_edges, neg_edges):
        """
        Transform node embeddings to machine learner input X and target y

        Parameters
        ----------
        combiner: callable
            function to combine embeddings. Needs to take two d-dimensional
            vectors as input and return a one-dimendional d-dimensional vector
        pos_edges: edgelist
            list of positive edges
        neg_edges: edgelist
            list of negative edges

        Returns
        -------
        X: array of shape (len(pos_edges) + len(neg_edges), d)
            Machine learner input (array of combined embedding vectors)
        y: array of shape (len(pos_edges) + len(neg_edges),), dtype bool
            Machine learner targets

        Notes
        ------
        X and y are sorted according to the order of pos_edges and neg_edges,
        with pos_edges appearing first
        """

        assert self._is_complete(), "There are non-embedded nodes in the graph."

        n_edges = len(pos_edges) + len(neg_edges)

        X = np.empty(shape=(n_edges, self.d), dtype=np.float64)
        y = np.empty(shape=(n_edges,), dtype=np.bool)

        for i, (node_i, node_j) in enumerate(pos_edges + neg_edges):
            X[i, :] = self.combine_embeddings(node_i, node_j, combiner)

            if i < len(pos_edges):
                y[i] = True
            else:
                y[i] = False

        return X, y

    @classmethod
    def from_array(cls, G, array):
        assert array.shape[0] == G.number_of_nodes()

        d = array.shape[1]
        emb = cls(d=d, G=G)
        for i, node in enumerate(emb.G.nodes):
            emb[node] = array[i]

        return emb

    @classmethod
    def from_dict(cls, G, dikt):
        assert len(dikt) == G.number_of_nodes()

        d = next(iter(dikt.values())).shape[0]
        emb = cls(G, d)
        for node, embedding in dikt:
            emb[node] = embedding

    def to_array(self):
        arr = np.empty((self.G.number_of_nodes(), self.d), dtype=np.float64)
        for i, node in enumerate(self.G):
            arr[i, :] = self[node]
        return arr


class DirectedNodeEmbeddings(NodeEmbeddings):

    def __init__(self, G, d):

        if not nx.is_directed(G):
            raise TypeError(
                'Cannot make directed embedding for undirected graph')

        super().__init__(G, d)

    def __setitem__(self, key, value):
        assert key in self.G
        self._validate_embedding(value)
        embeddings = tuple(v.reshape(-1) for v in value)
        self._embs[key] = embeddings

    def _validate_embedding(self, embedding):

        assert isinstance(embedding, Sequence)
        assert len(embedding) == 2
        for e in embedding:
            super()._validate_embedding(e)

    def combine_embeddings(self, node_1, node_2, combiner):
        return combiner(self[node_1][0], self[node_2][1])

    @classmethod
    def from_array(cls, G, array):
        assert array.shape[0] == G.number_of_nodes()
        assert array.shape[2] == 2

        d = array.shape[1]
        emb = cls(G, d)
        for i, node in enumerate(emb.G.nodes):
            emb[node] = np.split(array[i], 2, 1)

        return emb

    @classmethod
    def from_dict(cls, G, dikt, d=None):
        assert len(dikt) == G.number_of_nodes()

        d = next(iter(dikt.values()))[0].shape[0] if not d else d
        emb = cls(G, d)
        for node, embedding in dikt:
            emb[node] = embedding
