"""
DNGR (Cao et al. 2016)

taken from https://github.com/MdAsifKhan/DNGR-Keras/blob/master/DNGR.py
"""
from keras.layers import Input, Dense, noise
from keras.models import Model
import numpy as np
import networkx as nx
from ._embeddings import NodeEmbeddings


def scale_sim_mat(mat):
    # Scale Matrix by rowsum (=out-degree)
    mat = mat - np.diag(np.diag(mat))
    rowsum = mat.sum(axis=1, dtype=np.float32)
    rowsum[rowsum == 0] = 1.0

    D_inv = np.diag(np.reciprocal(rowsum))
    mat = np.dot(D_inv, mat)

    return mat


def random_surfing(adj_matrix, max_step, alpha):
    # Random Surfing
    nm_nodes = len(adj_matrix)
    adj_matrix = scale_sim_mat(adj_matrix)
    P0 = np.eye(nm_nodes, dtype='float32')
    M = np.zeros((nm_nodes, nm_nodes), dtype='float32')
    P = np.eye(nm_nodes, dtype='float32')
    for i in range(0, max_step):
        P = alpha * np.dot(P, adj_matrix) + (1 - alpha) * P0
        M = M + P

    return M


def PPMI_matrix(M):
    M = scale_sim_mat(M)
    nm_nodes = len(M)

    col_s = np.sum(M, axis=0).reshape(1, nm_nodes)
    row_s = np.sum(M, axis=1).reshape(nm_nodes, 1)
    D = np.sum(col_s)
    rowcol_s = np.dot(row_s, col_s)
    PPMI = np.log(np.divide(D * M, rowcol_s))
    PPMI[np.isnan(PPMI)] = 0.0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI < 0] = 0.0

    return PPMI


def model(data, hidden_layer_sizes, noise_level, n_epochs):

    input_sh = Input(shape=(data.shape[1],))
    encoded = noise.GaussianNoise(noise_level)(input_sh)

    for size in hidden_layer_sizes:
        encoded = Dense(size, activation='relu')(encoded)

    decoded = Dense(hidden_layer_sizes[-2], activation='relu')(encoded)
    for size in hidden_layer_sizes[-3::-1]:
        decoded = Dense(size, activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_sh, outputs=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')

    autoencoder.fit(data, data,
                    epochs=n_epochs)

    enco = Model(inputs=input_sh, outputs=encoded)
    enco.compile(optimizer='adadelta', loss='mse')
    reprsn = enco.predict(data)
    return reprsn


def dngr(G, d, alpha, k, n_hidden_layers, noise_level, n_epochs):
    """

    Parameters
    ----------
    G
        input graph
    d
        dimension
    alpha
        1 - return probability
    k
        max walk length
    n_hidden_layers
        # hidden layers. sizes will be [d * (n_hidden_layers + 1),
                                        d * n_hidden_layers,
                                        d * (n_hidden_layers - 1), ...]
    noise_level
        gamma
    n_epochs
        # of epochs to train model

    Returns
    -------

    """

    nodes = list(G.nodes)

    sim_mat = nx.adj_matrix(G).toarray()
    A = random_surfing(sim_mat, k, alpha)

    data = PPMI_matrix(A)

    hidden_layer_sizes = [d * i for i in range(n_hidden_layers, 0, -1)]
    emb_mat = model(data, hidden_layer_sizes, noise_level, n_epochs)

    out = NodeEmbeddings(G, d)

    for n in nodes:
        n_emb = emb_mat[nodes.index(n), :]
        out[n] = n_emb

    return out
