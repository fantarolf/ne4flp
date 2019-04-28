import logging
from time import time

from sklearn.base import clone
import numpy as np

from ..conf import CLASSIFIERS
from ..emb.combiners import avg, hadamard
from ..util import sample_hyperparameters, name_to_seed

logger = logging.getLogger(__name__)


def search_embedding_parameter(data, func, d, tuning_grid, args, score_func,
                               n_tunings):
    """embedding hyperparameter search and test score calculation

    Parameters
    ----------
    data : tuple
        tuple of train and test data as returned by `data.read_in_splits
    func : callable
        embedding function
    d : int
        dimension of embedding
    tuning_grid : dict
        grid to sample hyperparameters from
    args : dict
        fixed arguments for embedding function
    score_func : callable
        function to calculate score (should be sklearn-grid-search compatible=
    n_tunings : int
        number of hyperperameter sets to try

    Returns
    -------
    score : float
        test score
    e_test : Node Embeddings
        final set of node embeddings on Gf of D_test
    best_settings : dict
        best hyperparameter setting
    (params, scores) : list of tuples, dict, float
        hyperparameter sets and corresponding train scores (i.e. result of
        hyperparameter search)
    time_spent: float
        time for whole procedure
    """
    t1 = time()
    # split the data
    train, test = data
    Gf_tr, pe_tr, ne_tr = train
    Gf_te, pe_test, ne_test = test

    if len(tuning_grid) > 0:
        logger.info('Start Hyperparameter-tuning')
        # hyperparameter search for embeddings on train data
        params, scores = tune_emb(data=train, func=func, d=d,
                                  tuning_grid=tuning_grid, args=args,
                                  score_func=score_func, n_tunings=n_tunings)

        best_idx = scores.index(max(scores))
        best_params = params[best_idx]

    else:
        params = []
        scores = []
        best_params = {}

    logger.info('Get Test-Embedding and calculate final score')
    # calc embedding on Gf from D_test
    e_test = calc_embedding(Gf_te, func, d, {**args, **best_params})

    # search best classifier for embedding
    score, best_settings = tune_clf_emb(emb=e_test, pos_edges=pe_test,
                                        neg_edges=ne_test,
                                        score_func=score_func)

    time_spent = time() - t1

    return score, e_test, best_settings, (params, scores), time_spent


def tune_emb(data, func, d, tuning_grid, args, score_func, n_tunings):
    """embedding hyperparameter search

    Parameters
    ----------
    data : tuple
        tuple of feature graph, list of positive edge splits and negative
        edge splits
    func : callable
        embedding function
    d : int
        embedding dimension
    tuning_grid : dict
        tuning grid
    args : dict
        fixed arguments
    score_func : callable
        function to calculate score
    n_tunings : int
        # of parameter candidates to try

    Returns
    -------
    params: list of dict
        list of hyperparemeter sets evaluated
    scores: list of floats
        list of corresponding scores

    """
    Gf, pe, ne = data

    # sample hyperparameters with seed calculated from embedding method
    # name ---> always the same for a specific method
    params_to_try = sample_hyperparameters(tuning_grid,
                                           n_tunings,
                                           random_state=name_to_seed(
                                               func.__name__))

    scores = []

    logger.info('Starting Parameter Search')
    for i, param in enumerate(params_to_try):
        logger.info(
            f'Trying parameters ({i + 1}/{len(params_to_try)}: {param})')
        train_emb = calc_embedding(Gf, func, d, {**param, **args})

        score, _ = tune_clf_emb(train_emb, pe, ne, score_func)
        scores.append(float(score))
    return params_to_try, scores


def tune_clf_emb(emb, pos_edges, neg_edges, score_func):
    """ tune classifier with embedding as input and link prediction target

    Parameters
    ----------
    emb : NodeEmbeddings
        input embeddings
    pos_edges : list
        list of split input edgelist for positive class of length three
        where first edgelist is list of positive train edges, second edgelist
        is list of positive val edges and third edgelist is list of positive
        test edges
    neg_edges : list
        same as above, only for negative instances
    score_func : callable
        score function

    Returns
    -------
    score: float
        test score
    best_hyperparameters : tuple
        tuple of best combiner, best classifier and best classifier hyper-
        parameters

    """
    pe_train, pe_val, pe_test = pos_edges
    ne_train, ne_val, ne_test = neg_edges

    scores = list()
    settings = list()

    logger.info('Starting classifier tuning.')

    # for each combiner
    for combiner in [avg, hadamard]:

        # combine embeddings using combiner for val and train data
        X_train, y_train = emb.to_features_labels(combiner, pe_train, ne_train)
        X_val, y_val = emb.to_features_labels(combiner, pe_val, ne_val)
        #

        for classifier_name, (estimator, grid) in CLASSIFIERS.items():
            # iterate over classifiers and classifier hyperparameters
            for param in grid:
                logger.info(
                    f'Trying combination '
                    f'{combiner.__name__, classifier_name, param}')
                estimator_ = clone(estimator)
                estimator_.set_params(n_jobs=-1, **param)

                # fit on train data
                estimator_.fit(X_train, y_train)
                # score on cal data
                score = score_func(estimator_, X_val, y_val)
                # record used settings and obtained scores
                settings.append((combiner, classifier_name, param))
                scores.append(score)
    # get position of best score
    best_idx = scores.index(max(scores))
    # and corresponding settings
    best_combiner, best_clf, best_clf_params = settings[best_idx]

    # recombine node embeddings using best combiner
    X_train, y_train = emb.to_features_labels(best_combiner, pe_train, ne_train)
    X_val, y_val = emb.to_features_labels(best_combiner, pe_val, ne_val)
    X_test, y_test = emb.to_features_labels(best_combiner, pe_test, ne_test)

    # fit best classifier with best settings on train- and val data together
    best_estimator = clone(CLASSIFIERS[best_clf][0])
    best_estimator.set_params(n_jobs=-1, **best_clf_params, )
    best_estimator.fit(np.concatenate([X_train, X_val]),
                       np.concatenate([y_train, y_val]))

    # and score on test data
    test_score = score_func(best_estimator, X_test, y_test)

    return float(test_score), (best_combiner.__name__,
                               best_clf,
                               best_clf_params)


def calc_embedding(G, func, d, hparams):
    """worker function to calculate embeddings"""
    emb = func(G, d, **hparams)
    return emb


def emb_for_dims(data, func, d_list, args, score_func):
    """run embedding for different dimensions and get obtained test score

    Parameters
    ----------
    data
        test data (as returned by `data.read_in_splits([DS])[1]`)
    func
        embedding function
    d_list
        list of dimensions to calculate
    args
        dict of embedding args
    score_func
        score function

    Returns
    -------
    out
        dict, with key = d and value = score obtained with dimension d

    """
    Gf, pe, ne = data

    out = dict.fromkeys(d_list)

    for d in d_list:
        logger.info(f'd={d}')
        emb = calc_embedding(Gf, func, d, args)
        score, _ = tune_clf_emb(emb, pe, ne, score_func)

        out[d] = score

    return out


def result_stability(data, d, func, args, n_iter, score_func):
    """Run embedings n_iter times and get corr. test scores"""
    G, pe, ne = data
    scores = []
    for i in range(n_iter):
        logger.info(f'Stability run {i + 1} / {n_iter}')
        emb = calc_embedding(G, func, d, args)
        score, _ = tune_clf_emb(emb, pe, ne, score_func)
        scores.append(score)
    return scores
