"""
node2vec
"""
import subprocess
import tempfile
import networkx as nx
import numpy as np

from ._embeddings import NodeEmbeddings


def node2vec(G, d, p, q, **kwargs):
    """function for node2vec embedding algorithm

    Parameters
    ----------
    G : DiGraph

    d : int
        Embedding Dimension
    p : float
        In-Out-Parameter
    q : float
        Return Parameter
    kwargs:
        Further args passed to node2vec

    Returns
    -------
    NodeEmbeddings
        NodeEmbeddings for G found with node2vec algorithm
    """
    # set node ids to ints and remember old ids
    G_processed, node_id_mapping = _preprocess_graph(G)

    with tempfile.NamedTemporaryFile('w+b', delete=False) as tf_el:
        nx.write_edgelist(G_processed, tf_el, data=False)
    n2v_args = ['node2vec', f'-i:{tf_el.name}', f'-d:{d}',
                f'-p:{p}', f'-q:{q}']
    if G.is_directed():
        n2v_args.append('-dr')
    tf_emb = tempfile.NamedTemporaryFile('w+b', delete=False)
    n2v_args.append(f'-o:{tf_emb.name}')
    for kwarg in kwargs:
        n2v_args.append(f'-{kwarg}:{kwargs[kwarg]}')
    subprocess.run(n2v_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _read_in_embedding(G, d, tf_emb.name, node_id_mapping)


def _read_in_embedding(G, d, file, node_id_mapping):
    embeddings = NodeEmbeddings(G, d)
    with open(file) as f:
        for line_num, line in enumerate(f.readlines()):
            if line_num == 0:
                V, d_ = line.strip().split()
                print(f"dimension of embedding: {V}x{d_}")
            else:
                embedding = line.strip().split()
                node_id = int(embedding.pop(0))
                if node_id == 0:  # is a dummy node indicating end of walk
                    continue
                embedding = np.array(embedding, dtype=np.float64)
                original_node_id = node_id_mapping[node_id]
                embeddings[original_node_id] = embedding
    return embeddings


def _preprocess_graph(G):
    # make node ids ascending, starting from 1
    node_id_mapping = {i + 1: node for i, node in enumerate(G.nodes)}
    G_new = nx.relabel_nodes(G, {v: k for (k, v) in node_id_mapping.items()})

    return G_new, node_id_mapping
