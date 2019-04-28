from ._embeddings import NodeEmbeddings
import numpy as np


def random_embedding(G, d):
    """
    calculates dummy (i.e. randomly sampled from standard normal distribution)
    """

    emb = np.random.normal(size=(G.number_of_nodes(), d))

    return NodeEmbeddings.from_array(array=emb, G=G)
