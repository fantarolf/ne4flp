from sklearn.metrics import precision_recall_curve, auc, make_scorer
import yaml
disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')

import pickle as pkl

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import datetime as dt
import warnings

from hashlib import md5
from itertools import product

from scipy.sparse import csr_matrix, save_npz
from sklearn.utils import check_random_state
from networkx.exception import NetworkXNoPath


def get_timestamps(G, ts_identifier='timestamp'):
    """get edge timestamps of a graph"""
    return nx.get_edge_attributes(G, ts_identifier)


def date_to_datetime(d):
    return dt.datetime.fromordinal(d.toordinal())


def number_of_non_edges(G):
    nV = G.number_of_nodes()
    nE = G.number_of_edges()

    if nx.is_directed(G):
        n_poss = nV * (nV - 1)
    else:
        n_poss = nV * (nV - 1) / 2

    return n_poss - nE


def parse_timestamp(s):
    return dt.datetime.fromtimestamp(int(s))


def text_timestamps_to_datetime(G, ts_key='timestamp'):
    ts = get_timestamps(G, ts_key)
    ts_parsed = {k: parse_timestamp(v) for(k, v) in ts.items()}
    nx.set_edge_attributes(G, ts_parsed, ts_key)


def edgelist_sorted_by_timestamp(G):
    """Return list of edges sorted by timestamp (ascending order)"""
    ts = nx.get_edge_attributes(G, 'timestamp')
    return sorted(ts, key=lambda x: (ts[x], x))


def subsampled_edges(G, imb_ratio, known_edges=None, random_state=None):
    """Subsample negative edges from a graph

    Subsampling refers to sampling all positive, but only a fraction of
    non-edges (randomly) from a graph. Number of sampled edges is controlled
    by `imb_ratio`, i.e. we sample nE(G) * imb_ratio edges.

    Parameters
    ----------
    G
        Graph to subsample from
    imb_ratio
        Imbalance ratio, i.e. ratio of resulting negative to positive edges
    known_edges: edgelist
        Edges that are excluded from subsampling (because e.g.~we know they
        already formed in the feature graph)
    random_state
        seed to control randomness of sampling

    Returns
    -------
    pos_edges: edgelist
        list of positive edges
    neg_edges: edgelist
        list of subsampled non-edges

    """
    random_state = check_random_state(random_state)
    pos_edges = list(G.edges)

    if G.number_of_edges() == 0:
        raise ValueError(f'Graph does not have any edges. Cannot subsample.')
    neg_edges = subsample_negatives(G, imb_ratio, known_edges, random_state)
    return pos_edges, neg_edges


def _sample_zero_forever(mat, directed, rs):
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    rs = check_random_state(rs)
    while True:
        t = tuple(rs.randint(0, mat.shape[0], 2))
        if not directed and t[0] > t[1]:
            continue
        if t not in nonzero_or_sampled and t[0] != t[1]:
            yield t
            nonzero_or_sampled.add(t)


def _sample_zero_n(mat, n, directed, random_state):
    """
    taken from
    https://stackoverflow.com/questions/50665681/finding-n-random-zero-elements-from-a-scipy-sparse-matrix
    """
    itr = _sample_zero_forever(mat, directed, random_state)
    return [next(itr) for _ in range(n)]


def subsample_negatives(G, imb_ratio, known_edges, random_state=None):
    known_edges = [] if known_edges is None else known_edges

    n_inst_to_sample = G.number_of_edges() * imb_ratio

    if (number_of_non_edges(G) - len(known_edges)) <= n_inst_to_sample:
        warnings.warn(
            f'Subsampling with imbalance ratio {imb_ratio} not possible, '
            f'as G does not have enough non-edges.'
            f' Returning all non-edges instead.')
        return list(nx.non_edges(G))

    nodes = list(G.nodes)
    A = nx.adj_matrix(G).tolil()

    for i, j in known_edges:
        A[nodes.index(i), nodes.index(j)] = 1

    subs_adj_idx = _sample_zero_n(A,
                                  n_inst_to_sample,
                                  G.is_directed(),
                                  random_state)

    return [(nodes[i], nodes[j]) for (i, j) in subs_adj_idx]


def edge_list_checksum(edge_list):
    h = md5()
    h.update(str(tuple(edge_list)).encode())
    return h.hexdigest()


def largest_component(G):
    if G.number_of_edges() == 0:
        return G

    G_comps = nx.weakly_connected_components(G)

    largest_cc = sorted(G_comps, key=len, reverse=True)[0]
    return G.subgraph(largest_cc).copy()


def cart_prod_from_dict(d):
    keys = d.keys()
    vals = d.values()
    for prod in product(*vals):
        yield dict(zip(keys, prod))


def sample_hyperparameters(param_grid, n_params, random_state=None):
    """sample hyperparameter sets from a combination grid"""
    r = check_random_state(random_state)
    all_combs = list(cart_prod_from_dict(param_grid))

    if n_params >= len(all_combs):
        return all_combs

    sampled_combs = r.choice(np.arange(len(all_combs)), size=n_params,
                             replace=False)

    return [all_combs[i] for i in sampled_combs]


def shortest_path_length(G, e):  # TODO check need
    """wrapper around nx-fct that returns -1 if no path."""
    try:
        L = nx.shortest_path_length(G, *e)
    except NetworkXNoPath:
        L = -1

    return L


def name_to_seed(name):
    """get random seed from name (useful to control randomness)"""
    return int(md5(name.encode()).hexdigest()[:8], 16)


def average_degree(G):
    """average (out)-degree of G"""
    return nx.to_numpy_matrix(G).sum(axis=0).mean()


def plot_embedding(emb, path=None):
    fig, ax = plt.subplots()
    a = ax.imshow(emb, aspect='auto', cmap='nipy_spectral')
    fig.colorbar(a)
    if path is None:
        return fig
    else:
        fig.savefig(path)
        plt.close(fig)


def save_embedding(emb, sparse, path):
    if sparse:
        sparse_emb = csr_matrix(emb)
        save_npz(file=path, matrix=sparse_emb)
    else:
        np.savez(file=path, arr=emb)


def best_args_from_tuning_result(result):
    result = result.copy()
    result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
    best_result = result_sorted[0]
    best_result = {k: best_result[k] for k in best_result if k != 'score'}
    return best_result


def dir_status(path):
    """
    0: path does not exist
    1: path exists, is empty directory
    2: else
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                return 1
    else:
        return 0
    return 2


def write_yaml(file, d):
    """write dict d to file as yaml string"""
    with open(file, 'w') as f:
        yaml.safe_dump(d, f)


def write_pkl(path, obj):
    """write obj to pickle-file"""
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def write_score(path, score, fmt='.5f'):
    with open(path, 'w') as f:
        f.write(format(score, fmt))


def read_yaml(path):
    with open(path) as f:
        return yaml.load(f)


def precision_at_k(y_true, y_pred, k):
    argsort_preds = np.argsort(y_pred)[::-1]
    top_k_preds = y_true[argsort_preds][:k]
    return top_k_preds.mean()


def _aupr_score_func(y_true, y_pred):
    pr, rec, _ = precision_recall_curve(y_true, y_pred)
    return auc(rec, pr)


aupr_score = make_scorer(_aupr_score_func, needs_proba=True)
