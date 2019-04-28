from functools import reduce
import os

import keras
import networkx as nx
import numpy as np
import tensorflow as tf

jp = os.path.join

l12reg = keras.regularizers.l1_l2


def get_encoder(nV, d, layer_sizes, l1_reg, l2_reg):
    encoder_in = keras.layers.Input(shape=(nV,), name='encoder-in')
    layers = []

    for i, size in enumerate(layer_sizes):
        layers.append(keras.layers.Dense(
            units=size,
            activation='relu',
            kernel_regularizer=l12reg(l1_reg, l2_reg),
            name=f'encoder-{i}'
        ))

    # final layer (where embeddings live)
    layers.append(keras.layers.Dense(
        units=d,
        activation='tanh',  # embeddings between -1 and 1
        kernel_regularizer=l12reg(l1_reg, l2_reg),
        name='encoder-final'
    )
    )
    encoder_out = reduce(lambda arg, f: f(arg), layers, encoder_in)

    return keras.models.Model(
        inputs=encoder_in,
        outputs=encoder_out,
        name='encoder'
    )


def get_decoder(nV, d, layer_sizes, l1_reg, l2_reg):
    decoder_in = keras.layers.Input(shape=(d,), name='decoder-in')
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.append(keras.layers.Dense(
            units=size,
            activation='relu',
            kernel_regularizer=l12reg(l1_reg, l2_reg),
            name=f'decoder-{i}'
        ))
    layers.append(keras.layers.Dense(
        units=nV,
        activation='relu',
        kernel_regularizer=l12reg(l1_reg, l2_reg),
        name='decoder-final'
    )
    )
    decoder_out = reduce(lambda arg, f: f(arg), layers, decoder_in)

    return keras.models.Model(
        inputs=decoder_in,
        outputs=decoder_out,
        name='decoder'
    )


def get_autoencoder(encoder, decoder):
    encoder_input_shape = encoder.layers[0].input_shape[1]
    ae_in = keras.Input((encoder_input_shape,))
    encoded = encoder(ae_in)
    decoded = decoder(encoded)
    return keras.models.Model(
        inputs=ae_in,
        outputs=[encoded, decoded],
        name='autoencoder'
    )


def get_link_predictor(d, layer_sizes, l1_reg, l2_reg):
    lp_in = keras.Input((d,), name='link-prediction-in')
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.append(
            keras.layers.Dense(
                units=size,
                activation='relu',
                kernel_regularizer=l12reg(l1_reg, l2_reg),
                name=f'link-prediction-{i}'

            )
        )
    layers.append(
        keras.layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_regularizer=l12reg(l1_reg, l2_reg),
            name='link-prediction-out'
        )
    )
    lp_out = reduce(lambda arg, f: f(arg), layers, lp_in)
    return keras.models.Model(
        inputs=lp_in,
        outputs=lp_out,
        name='link-predictor'
    )


def get_lp_autoencoder(autoencoder, link_predictor):
    autoencoder_input_shape = autoencoder.layers[0].input_shape[1]

    lp_in_1 = keras.Input((autoencoder_input_shape,))
    lp_in_2 = keras.Input((autoencoder_input_shape,))

    encoded_1, decoded_1 = autoencoder(lp_in_1)
    encoded_2, decoded_2 = autoencoder(lp_in_2)

    emb_average = keras.layers.Average()([encoded_1, encoded_2])

    lp_out = link_predictor(emb_average)

    return keras.models.Model(
        inputs=[lp_in_1, lp_in_2],
        outputs=[lp_out, decoded_1, decoded_2],
        name='lp-autoencoder'
    )


def sdne_2_loss(beta):
    def keras_sdne_2_loss(y_true, y_pred):
        B = tf.multiply(y_true, beta - 1.) + 1.
        weighted_diff = (y_true - y_pred) * B
        squared_diff = tf.square(weighted_diff)
        summ_diff = tf.reduce_sum(squared_diff, axis=1)
        return tf.reduce_mean(summ_diff)

    return keras_sdne_2_loss


class LPAutoencoderGenerator(keras.utils.Sequence):
    """

    """

    def __init__(self, G, pos_edges, neg_edges, batch_size):
        """Batch data generator for link prediction autoencoder.

        The generator returns for each batch a tuple of input and output data.
        The input data consists of list of rows of the adjacency matrix of G.
        To obtain this data, the list of positive and negative edges is iterated
        over. For each edge $i$ and $j$, the corresponding row of the adjacency
        matrix $a_i$, and $a_j$ is returned as input tuple.

        The output is a three tuple of link prediction target (i.e.~will there
        be an edge between $v_i$ and $v_j$) and inputs to the model (to super-
        vise the autoencoder).

        In each batch, the distribution of positive and negative edges is
        approx. the same.

        Parameters
        ----------
        G: nx.DiGraph
            Input feature Graph
        pos_edges: edgelist
            list of positive instances (edges)
        neg_edges : edgelist
            list of negative instances (edges)
        batch_size : int
            size of data batches
        """
        self.nodelist = list(G.nodes)
        A = nx.adj_matrix(G, self.nodelist)
        self.A = (A + A.T) / 2
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.batch_size = batch_size
        self.n_pos_edges = len(self.pos_edges)
        self.n_neg_edges = len(self.neg_edges)
        self.n_edges = self.n_pos_edges + self.n_neg_edges

        self.pos_batch_size = round(
            self.batch_size * (self.n_pos_edges / self.n_edges))
        self.neg_batch_size = self.batch_size - self.pos_batch_size

    def __len__(self):
        n_possible_batches = np.ceil(self.n_edges / self.batch_size)
        return int(n_possible_batches)

    def __getitem__(self, idx):
        pos_edges_slice = slice(idx * self.pos_batch_size,
                                (idx + 1) * self.pos_batch_size)
        neg_edges_slice = slice(idx * self.neg_batch_size,
                                (idx + 1) * self.neg_batch_size)

        batch_pos_edges = self.pos_edges[pos_edges_slice]
        batch_neg_edges = self.neg_edges[neg_edges_slice]

        target = np.repeat((1, 0),
                           (len(batch_pos_edges),
                            len(batch_neg_edges)),
                           ).astype(np.bool)
        all_edges = batch_pos_edges + batch_neg_edges
        _batch_size = len(all_edges)

        A_from = np.empty(shape=(_batch_size, self.A.shape[1]),
                          dtype=np.float64)
        A_to = np.empty_like(A_from)

        for i, (u, v) in enumerate(all_edges):
            A_from[i, :] = self._lookup_adj(u)
            A_to[i, :] = self._lookup_adj(v)

        return [A_from, A_to], [target, A_from, A_to]

    def _lookup_adj(self, node):
        nodelist_idx = self.nodelist.index(node)
        return self.A[nodelist_idx].toarray()

    def all_lp_targets(self):
        batch_lp_targets = []
        for i in range(len(self)):
            batch_lp_targets.append(self[i][1][0].flatten())
        return np.concatenate(batch_lp_targets, axis=0)

    def _get_batch_edgelist(self, idx):
        pos_edges_slice = slice(idx * self.pos_batch_size,
                                (idx + 1) * self.pos_batch_size)
        neg_edges_slice = slice(idx * self.neg_batch_size,
                                (idx + 1) * self.neg_batch_size)

        return self.pos_edges[pos_edges_slice] + self.neg_edges[neg_edges_slice]

    def get_edgelist(self):
        out = []
        for i in range(len(self)):
            out += self._get_batch_edgelist(i)
        return out


class AutoencoderGenerator(keras.utils.Sequence):
    """
    Data Generator for Autoencoder (pretraining).
    Returns rows of the adjacency matrix of `G` in batches of `batch_size`
    """

    def __init__(self, G, batch_size):
        self.G = G
        self.batch_size = batch_size
        self.nodelist = list(G.nodes)
        A = nx.adj_matrix(self.G, self.nodelist)
        self.A = (A + A.T) / 2

    def __len__(self):
        nV = self.G.number_of_nodes()
        n_possible_batches = nV / self.batch_size
        return int(np.ceil(n_possible_batches))

    def __getitem__(self, item):
        bs = self.batch_size
        A_slice = slice(item * bs, (item + 1) * bs)
        A_batch = self.A[A_slice].toarray()
        return A_batch, A_batch

    def all_data(self):
        batches = []
        for i in range(len(self)):
            batches.append(self[i][0])
        return np.concatenate(batches, axis=0)
