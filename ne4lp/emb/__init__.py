from ._n2v import node2vec
from ._dngr import dngr
from ._modelr import model_r
from ._dummy import random_embedding
from ._gem import laplacian_eigenmaps, locally_linear_embeddings, SDNE
from ._embeddings import NodeEmbeddings, DirectedNodeEmbeddings

EMBEDDING_METHODS = {
    'node2vec': node2vec,
    'dngr': dngr,
    'model_r': model_r,
    'lap': laplacian_eigenmaps,
    'random': random_embedding,
    'sdne': SDNE
}
