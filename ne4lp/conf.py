"""
Configurations
"""

import os
import logging.config

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
import yaml

average_precision_score_clf = make_scorer(average_precision_score,
                                          needs_proba=True,
                                          greater_is_better=True)


# ---- DIRECTORIES ----
if 'NE4LP_DIR' in os.environ:
    PROJ_PATH = os.path.abspath(os.environ['NE4LP_DIR'])
else:
    PROJ_PATH = os.path.abspath('.')
DATA_DIR = os.path.join(PROJ_PATH, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
CONF_DIR = os.path.join(PROJ_PATH, 'conf')

LOG_CONF_FILE = os.path.join(CONF_DIR, 'logging.yaml')


# ---- LOGGING SETUP ----
def _get_logging_config():
    with open(LOG_CONF_FILE) as f:
        config = yaml.safe_load(f)
    return config


def _setup_logging():
    conf = _get_logging_config()
    logging.config.dictConfig(conf)


def get_default_logger():
    _setup_logging()
    return logging.getLogger('ne4lp')


def get_logger(name, level):
    _setup_logging()
    l = logging.getLogger(name)
    l.setLevel(level)
    return l


def get_relpath(relpath):
    return os.path.join(PROJ_PATH, relpath)


# ---- DATASET CONFIGS ---- #

# all datasets for experiments
ALL_DATASETS = ['CollegeMsg', 'DiggFriends', 'Enron', 'fbWosn', 'wikiSimple',
                'wikiVote']

# Sizes of \tilde{E} splits (train, val, test)
NESTED_SIZES = [.4, .2, .4]

# n_f, n_l and offset for split in G_train and G_test
SPLIT_SIZES = {
    'CollegeMsg': (6000, 3000, 3000),
    'DiggFriends': (40000, 20000, 10000),
    'Enron': (30000, 15000, 10000),
    'fbWosn': (50000, 25000, 20000),
    'wikiSimple': (50000, 25000, 10000),
    'wikiVote': (1500, 750, 500)
}

# Experiment configs
IMB_RATIO = 30  # imbalance ratio
D = 128  # embedding dimension TODO use in AE-Experiment
# number of iterations for (existing) embedding hyperparameter search
N_ITER_EMB_GRID_SEARCH = 10
SCORE_NAME = 'Average Precision Score'
SCORE_FUNC = average_precision_score  # Scoring function TODO use in AE-Experiment
SCORE_FUNC_CLF = average_precision_score_clf  # sklearn grid-search compatible


# grid of classifiers for link prediction from embeddings
CLASSIFIERS = {
    'RandomForest':
        (RandomForestClassifier(n_estimators=50),
         [
             {'max_depth': 3, 'min_samples_split': 20},
             {'max_depth': 3, 'min_samples_split': 2},
             {'max_depth': 5, 'min_samples_split': 20},
             {'max_depth': 5, 'min_samples_split': 2},
             {'max_depth': 7, 'min_samples_split': 2}
         ]),
    'LogisticRegression':
        (LogisticRegression(solver='lbfgs'),
         [
             {'C': 0.01},
             {'C': 0.1},
             {'C': 1},
             {'C': 10},
             {'C': 100}
         ])
}

# number of runs for stability experiment
N_ITER_STABILITY = 5
