# Node Embeddings for Link Prediction

## Install

We recommend using a seperate virtual environment to install ne4lp.

To install the source code, run the following from this directory.

```bash
pip install .
```

Set an environment variable `NE4LP_DIR` to the location where this folder is 
stored on your machine. This is needed in order for the module to locate the
data.

node2vec implementation needs to be installed from snap (https://github.com/snap-stanford/snap).
A compiled binary is included in the DVD for the thesis. It may or may not work
on your system. If it does not work, compile node2vec from snap and put the binary
somewhere on your path. node2vec implementation in the source code relies on the existence
of this binary.

## Module Structure

``
ne4lp/
├── conf.py:                Configurations for experiments and folder structure
├── data.py                 utilities for data i/o
├── emb
│   ├── combiners.py        Combiner functions
│   ├── _dngr.py            DNGR (taken from Khan 2019)
│   ├── _dummy.py           Random dummy embeddings
│   ├── _embeddings.py      Containers for Embeddings
│   ├── _gem.py             Embedding functions from GEM library (Goyal & Ferrara)
│   ├── __init__.py
│   ├── lp_autoencoder.py   Link-prediction Autoencoder
│   ├── _modelr.py          Model R
│   ├── _n2v.py             node2vec
├── exp                     Functions to run experiments
│   ├── ae.py
│   ├── emb.py
│   ├── __init__.py
│   └── sim.py
├── __init__.py
├── sim.py                  Containers and functions for similarity indices
├── tts.py                  train-test-split of graphs and edges
└── util.py                 utilitiy functions
```
