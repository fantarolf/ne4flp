from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.sdne import SDNE as SDNE_
import numpy as np

from ._embeddings import NodeEmbeddings


def laplacian_eigenmaps(G, d):
    lam = LaplacianEigenmaps(d=d)
    lam.learn_embedding(G)

    X = lam.get_embedding()
    X = X.astype(np.float64)

    ne = NodeEmbeddings.from_array(G, X)
    return ne


def locally_linear_embeddings(G, d):
    lle = LocallyLinearEmbedding(d=d)
    lle.learn_embedding(G)

    X = lle.get_embedding()

    return NodeEmbeddings.from_array(G, X)


def SDNE(G,
         d,
         beta,
         alpha,
         nu1,
         nu2,
         n_hidden_layers,
         n_epochs,
         xeta,
         batch_size):
    # precalculate n hidden units
    n_units = [d * i for i in range(n_hidden_layers + 1, 1, -1)]

    sdne_ = SDNE_(
        d=d,
        beta=beta,
        alpha=alpha,
        nu1=nu1,
        nu2=nu2,
        K=n_hidden_layers,
        n_units=n_units,
        n_iter=n_epochs,
        xeta=xeta,
        n_batch=batch_size
    )

    sdne_.learn_embedding(G)

    X = sdne_.get_embedding()

    return NodeEmbeddings.from_array(G, X)
