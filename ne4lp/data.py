import os

import networkx as nx

from . import conf
from . import util
from .tts import graph_split, nested_edge_split
from collections import defaultdict
import numpy as np

_REF_DIR = conf.get_relpath('data/original')
DATASETS = {ds[:-11] for ds in os.listdir(_REF_DIR)}


def read_in_raw_graph(dataset):
    """reads in raw (i.e. preprocessed) graph from name"""
    if dataset not in DATASETS:
        raise ValueError(f'Datensatz {dataset} nicht bekannt.')
    G_path = os.path.join(conf.RAW_DATA_DIR, dataset + '.gpkl')
    G = nx.read_gpickle(G_path)
    return G


def read_in_splits(graph):
    """read in splits (D_train, D_test) of a graph

    Parameters
    ----------
    graph : str
        name of the dataset to be read in

    Returns
    -------
    tuple
        Tuple of D_train and D_test, which each consists of feature Graph,
        list of positive edges and list of negative edges.

    """
    split_data = []

    for split in ['train', 'test']:
        Gf = read_in_split_graph(graph, split, 'f')
        p_e, n_e = read_in_split_edges(graph, split)
        split_data.append((Gf, p_e, n_e))

    return split_data


def delete_false_timestamps(G, ts_filter):
    edges_to_delete = []
    for f, t, d in G.edges(data=True):
        ts = d['timestamp']
        if ts_filter(ts):
            edges_to_delete.append((f, t))
    G.remove_edges_from(edges_to_delete)
    return G


def make_split(dataset, out_dir=None):
    """split data according to split sizes defined in `conf` module"""

    raw_filename = dataset + '.gpkl'
    if raw_filename not in os.listdir(conf.RAW_DATA_DIR):
        raise ValueError(f'Cannot find raw dataset {dataset}')

    split_dir = os.path.join(conf.DATA_DIR, 'splits', dataset)

    if out_dir is None:
        out_dir = split_dir

    G = read_in_raw_graph(dataset)

    nE_feature, nE_label, offset = conf.SPLIT_SIZES[dataset]

    splits = graph_split(G, nE_feature, nE_label, offset, how=['train', 'test'])

    seed = util.name_to_seed(dataset)
    rs = np.random.RandomState(seed=seed)

    for s in splits:
        for G_, name in zip(splits[s], ['f', 'l']):
            fname = f'{s}_{name}.gpkl'
            nx.write_gpickle(G_, os.path.join(split_dir, fname))

        feature, label = splits[s]

        p_e, n_e = util.subsampled_edges(G=label, imb_ratio=conf.IMB_RATIO,
                                         known_edges=list(feature.edges),
                                         random_state=rs)

        nested_splits = nested_edge_split(label, p_e, n_e, conf.NESTED_SIZES,
                                          rs=rs)

        with open(os.path.join(out_dir, s + '.edges'), 'w') as f:
            for cls, cls_split in zip([1, 0], nested_splits):
                for split, edges in enumerate(cls_split):
                    for i, j in edges:
                        f.write(', '.join([str(i),
                                           str(j),
                                           str(cls),
                                           str(split)]) + '\n')


def read_in_split_graph(graph, split, feature_or_label):
    """read in graph (feature or label) from split (train or test)

    Parameters
    ----------
    graph : str
        name of the dataset
    split : {'train', 'test'}
        name of the split, one of
    feature_or_label : {'f', 'l'}
        read in feature or label graph

    Returns
    -------

    """
    split_dir = os.path.join(conf.DATA_DIR, 'splits', graph)
    fname = split + '_' + feature_or_label + '.gpkl'
    return nx.read_gpickle(os.path.join(split_dir, fname))


def read_in_split_edges(graph, split, conv=str):
    """read in labeled edges for split.

    Parameters
    ----------
    graph: str
        name of dataset
    split: {'train', 'test'}
    conv: callable
        conversion function applied to textual node ids read in from file.
        Default is `str` (which means no conversion, node ids are left as
        strings)

    Returns
    -------
    positive edges: tuple of edgelists
        Each member of the tuple represents the split of positive future edges,
        e.g. \tilde{E}_train, \tilde{E}_val, \tilde{E}_test.
    negative edges: tuple of edgelists
        Same as above, only for subsampled non-edges

    """
    split_dir = os.path.join(conf.DATA_DIR, 'splits', graph)

    edges = defaultdict(list)

    n_splits = 0

    with open(os.path.join(split_dir, split + '.edges')) as f:
        for line in f.readlines():
            i, j, cls, split = line.strip().split(', ')
            edges[int(cls), int(split)].append((conv(i), conv(j)))
            n_splits = max(int(split), n_splits)

    out = []

    for cls_key in [1, 0]:
        out.append([edges[cls_key, i] for i in range(n_splits + 1)])

    return tuple(out)


def clean_datasets():
    """ Clean original datasets, i.e. delete false timestamps and converts
    format from graphml to gpkl.

    Returns
    -------
    None

    """
    dataset_filters = {
        'DiggFriends': lambda ts: ts.year == 1970,
        'Enron': lambda ts: ts.year < 1999 or ts.year > 2002
    }
    for ds in DATASETS:
        G = nx.read_graphml(os.path.join(_REF_DIR, ds + '.graphml.gz'))
        util.text_timestamps_to_datetime(G)
        if ds in dataset_filters:
            G = delete_false_timestamps(G, dataset_filters[ds])
        nx.write_gpickle(G, os.path.join(conf.RAW_DATA_DIR, ds + '.gpkl'))
