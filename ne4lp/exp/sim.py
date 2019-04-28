from ..util import sample_hyperparameters
from .. import conf
from time import time


def search_simind_parameter(train_data,
                            test_data,
                            func,
                            args,
                            tuning_grid,
                            score_func):
    """

    Parameters
    ----------
    train_data:
        train data, tuple of feature graph, positive edge splits and negative
        edge splits
    test_data
        test data, see above
    func
        sim ind function
    args
        args for sim ind function
    tuning_grid
        tuning grid
    score_func
        score function

    Returns
    -------
    test_score : float
        score obtained on test split
    test_si : SimilarityIndex
        SimilarityIndex on test feature graph
    (params, scores): list, list
        list of evaluated hyperparameter sets and corresponding scores
    time : float
        time for whole procedure
    """
    t1 = time()
    params, scores = tune_similarity_index(train_data,
                                           func,
                                           tuning_grid,
                                           args,
                                           score_func)

    best_idx = scores.index(max(scores))
    best_params = params[best_idx]

    Gf_test, pe_test, ne_test = test_data

    all_test_edges = pe_test[0] + pe_test[1] + pe_test[2] + \
        ne_test[0] + ne_test[1] + ne_test[2]

    test_si = calc_similarity_index(G=Gf_test,
                                    func=func,
                                    args={**args, **best_params},
                                    edge_list=all_test_edges)

    test_score = score_similarity_index(test_si, pe_test[2], ne_test[2],
                                        score_func)

    time_spent = time() - t1
    return test_score, test_si, (params, scores), time_spent


def calc_similarity_index(G, func, args, edge_list):
    return func(G=G, edge_list=edge_list, **args)


def predict_similarity_index(sim_ind, pos_edges, neg_edges):
    return sim_ind.to_features_labels(pos_edges, neg_edges)


def score_similarity_index(sim_ind, pos_edges, neg_edges, score):
    y_pred, y_true = predict_similarity_index(sim_ind, pos_edges, neg_edges)
    return score(y_true, y_pred)


def tune_similarity_index(data, func, tuning_grid, args, score_func):
    """ parameter search for similarity index

    Parameters
    ----------
    data
        tuple of feature graph, positive and negative edge splits
    func
        sim ind function
    tuning_grid
        tuning grid
    args
        dict of fixed args for simind function
    score_func
        score function

    Returns
    -------
    params: list
        evaluated hyperparameter sets
    scores: list
        corresponding scores

    """

    G_train, pe_train, ne_train = data

    params_to_try = sample_hyperparameters(tuning_grid,
                                           conf.N_ITER_EMB_GRID_SEARCH)

    params = []
    scores = []

    for param in params_to_try:
        params.append(param)
        si_train = calc_similarity_index(G_train, func, {**args, **param},
                                         pe_train[-1] + ne_train[-1])

        score = score_similarity_index(si_train,
                                       pe_train[-1],
                                       ne_train[-1],
                                       score_func)
        scores.append(score)

    return params, scores
