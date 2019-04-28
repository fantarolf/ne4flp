"""
Model R
(Hou, Y., Holder, L.B., 2017. Deep learning approach to link weight prediction)
"""
import keras
import numpy as np
from ._embeddings import DirectedNodeEmbeddings

from ..util import subsampled_edges


def model_r(G, d, n_hidden_layers, imb_ratio, n_epochs):

    # construct neural network
    hidden_layer_sizes = [d] * n_hidden_layers
    nV = G.number_of_nodes()
    model, source_emb, target_emb = _model_r_nn(int(d / 2), hidden_layer_sizes,
                                                nV)

    # extract features and labels from graph
    (X, y), node_id_mappings = _extract_features_labels(G, imb_ratio)

    # fit
    model.fit(X, y, epochs=n_epochs)

    node_ids = np.array(list(node_id_mappings.keys())).reshape(-1, 1)
    source_embeddings = source_emb.predict(node_ids)
    target_embeddings = target_emb.predict(node_ids)

    out = DirectedNodeEmbeddings(G, int(d / 2))

    # return embeddings
    for node, se, te in zip(node_ids, source_embeddings, target_embeddings):
        original_node_id = node_id_mappings[int(node)]
        out[original_node_id] = se, te

    return out


def node_ids_to_asc_ints(G):
    return {i: n for (i, n) in enumerate(G.nodes)}


def _model_r_nn(d, hidden_layer_sizes, nV):
    # inputs
    i1 = keras.layers.Input(shape=(1,), dtype=np.int32, name='from-node-ids')
    i2 = keras.layers.Input(shape=(1,), dtype=np.int32, name='to-node-ids')

    e1 = keras.layers.Embedding(input_dim=nV, output_dim=d, input_length=1)(i1)
    e2 = keras.layers.Embedding(input_dim=nV, output_dim=d, input_length=1)(i2)

    conc = keras.layers.Concatenate()([e1, e2])

    hidden = keras.layers.Dense(hidden_layer_sizes[0], activation='relu')(conc)

    for hl_size in hidden_layer_sizes[1:]:
        hidden = keras.layers.Dense(hl_size, activation='relu')(hidden)

    f = keras.layers.Flatten()(hidden)

    out = keras.layers.Dense(1, activation='sigmoid')(f)

    model = keras.models.Model(inputs=[i1, i2], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    source_emb = keras.models.Model(i1, e1)
    target_emb = keras.models.Model(i2, e2)

    return model, source_emb, target_emb


def _extract_features_labels(G, imb_ratio):
    node_id_mapping = node_ids_to_asc_ints(G)
    pos_edges, neg_edges = subsampled_edges(G, imb_ratio)
    n_instances = len(pos_edges) + len(neg_edges)

    y = np.repeat([True, False], [len(pos_edges), len(neg_edges)])
    X_from = np.empty((n_instances,), np.int32)
    X_to = np.empty_like(X_from)

    node_id_mapping_rev = {
        v: k for k, v in node_id_mapping.items()
    }

    for i, (edge_from, edge_to) in enumerate(pos_edges + neg_edges):
        X_from[i] = node_id_mapping_rev[edge_from]
        X_to[i] = node_id_mapping_rev[edge_to]

    return ([X_from, X_to], y), node_id_mapping
