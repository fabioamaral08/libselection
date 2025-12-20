import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.model_selection import KFold


def score_fn(Pd, D, terms, y, norm_y):
    """
    Calculate the score based on the provided data.

    Parameters:
    ----------
    Pd : numpy.ndarray
        Projection matrix over dictionary.
    Pd_sub : numpy.ndarray
        Projection matrix over dictionary without D_sub.
    y : numpy.ndarray
        The target data.

    Returns:
    -------
    float
        The calculated score.
    """
    # D_sub = D[:, [i for i in range(D.shape[1]) if i not in terms]]
    D_sub = D[:, terms]
    Pd_sub = D_sub @ np.linalg.pinv(D_sub)

    # Calculate the score for each dimension
    score = (np.linalg.norm((Pd - Pd_sub)@y,axis=0) / norm_y).sum()
    return score


def iter_scores(D, y,n_terms, stepwise=False, stepsize=1, replace=False, backwards = True):
    """
    Calculate scores for each column in the dictionary.

    Parameters:
    ----------
    D : numpy.ndarray
        The dictionary matrix.
    y : numpy.ndarray
        The target data.
    n_terms : int
        Number of terms to consider.
    stepwise : bool, optional
        If True, iteratively remove terms until reaching n_terms. Default is False.


    Returns:
    -------
    numpy.ndarray
        Array of scores for each column in the dictionary.
    """
    Pd = D@ np.linalg.pinv(D)
    norm_y = np.linalg.norm(y,axis=0)
    

    all_itens = list(range(D.shape[1]))
    if stepwise:
        if backwards:
            history_values = -1 * np.ones((D.shape[1]-n_terms,D.shape[1]))
            removed_order = np.zeros(D.shape[1], dtype=int)
            survivors = list(np.arange(D.shape[1], dtype=int))
            n_itens = len(all_itens)
            count = 0
            while n_itens > n_terms:
                lib_size = np.maximum(n_itens - stepsize, n_terms)
                combinations_list = list(combinations(all_itens, lib_size))
                all_scores = Parallel(n_jobs=-1)(
                    delayed(score_fn)(Pd, D, comb, y, norm_y)
                    for comb in combinations_list
                )
                for scr, itens in zip(all_scores, combinations_list):
                    itm = [i for i in survivors if i not in itens]
                    history_values[count, itm] = scr

                best_score = np.argmin(all_scores)
                all_itens = list(combinations_list[best_score])
                rmvd = [i for i in survivors if i not in all_itens][0]
                removed_order[count] = rmvd
                survivors.remove(rmvd)

                n_itens = len(all_itens)
                if replace:
                    survivors = np.zeros(D.shape[1], dtype=bool)
                    survivors[all_itens] = True
                    D[:, ~survivors] = 0
                    Pd = D @ np.linalg.pinv(D)
                count += stepsize
            best_combination = all_itens
            removed_order[count:] = best_combination
        else:
            history_values = -1 * np.ones((D.shape[1]-n_terms,D.shape[1]))
            removed_order = np.zeros(D.shape[1], dtype=int)
            survivors = []
            n_itens = len(all_itens)
            count = 0
            while n_itens > n_terms:
                lib_size = np.maximum(n_itens - stepsize, n_terms)
                combinations_list = list(combinations(all_itens, stepsize))
                all_scores = Parallel(n_jobs=-1)(
                    delayed(score_fn)(Pd, D, survivors + list(comb), y, norm_y)
                    for comb in combinations_list
                )

                best_score = np.argmin(all_scores)
                rmvd = combinations_list[best_score]
                for r in rmvd:
                    all_itens.remove(r)
                removed_order[count:count+stepsize] = rmvd
                survivors += rmvd
                for itm in rmvd:
                    history_values[:history_values.shape[0]-count, itm] = all_scores[best_score]

                n_itens = len(all_itens)
                count += stepsize
                if len(all_itens) < stepsize:
                    stepsize = len(all_itens)
            best_combination = all_itens
            removed_order[count:] = best_combination

        return best_combination, removed_order, history_values
    else:
        combinations_list = list(combinations(all_itens, n_terms))
        all_scores = Parallel(n_jobs=-1)(
            delayed(score_fn)(Pd, D, comb, y, norm_y)
            for comb in combinations_list
        )
        best_score = np.argmin(all_scores)
        best_combination = combinations_list[best_score]
    return best_combination, best_score, all_scores, combinations_list, 


def get_pareto_scores(D, x_dot):
    '''
    Index_survivor = bool type data
    '''
    No_D = D.shape[1]
    Index_survivor = np.ones(No_D, dtype=bool)
    scores = np.zeros((No_D))
    index_order = np.zeros((No_D), dtype=int)
    norm_y = np.linalg.norm(x_dot)
    for count in range(No_D, 0, -1):
        indices = np.where(Index_survivor )[0] # [0, 1, 3]
        score_min = np.inf
        for ii in range(count):
            Index_survivor[indices[ii]] = 0 # if count =1, then Index_survivor_temp = [1, 0, 1, 1]
            D_new = D[:,Index_survivor]

            # Pareto
            score = np.linalg.norm( x_dot - np.dot(D_new, np.linalg.pinv(D_new)) @ x_dot ) /norm_y
            if score < score_min:
                score_min = score
                index_order[No_D - count] = indices[ii]
            Index_survivor[indices[ii]] = True # if count =1, then Index_survivor_temp = [1, 0, 1, 1]
        scores[No_D - count] = score_min
        Index_survivor[index_order[No_D - count]] = False


    return scores, index_order


def Cross_validation_score(D, x_dot, k_fold=5, random_state = 42):
    SSR_original_score = np.zeros(k_fold)
    kf_SSR_TEST = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(kf_SSR_TEST.split(D)):
        SSR_original_score[fold] = np.linalg.norm(x_dot[test_index] - D[test_index] @ np.dot(np.linalg.pinv(D[train_index]), x_dot[train_index]))
        # SSR_original_score[fold] = our_score(D,x)
    return np.mean(SSR_original_score)
def get_cv_scores(D, x_dot):
    '''
    Index_survivor = bool type data
    '''
    No_D = D.shape[1]
    Index_survivor = np.ones(No_D, dtype=bool)
    scores = np.zeros((No_D))
    index_order = np.zeros((No_D), dtype=int)
    norm_y = np.linalg.norm(x_dot)
    for count in range(No_D, 0, -1):
        indices = np.where(Index_survivor)[0] # [0, 1, 3]
        score_min = np.inf
        for ii in range(count):
            Index_survivor[indices[ii]] = False # if count =1, then Index_survivor_temp = [1, 0, 1, 1]
            D_new = D[:,Index_survivor]

            # Pareto
            score = Cross_validation_score(D_new, x_dot) /norm_y
            if score < score_min:
                score_min = score
                index_order[No_D - count] = indices[ii]
            Index_survivor[indices[ii]] = True # if count =1, then Index_survivor_temp = [1, 0, 1, 1]
        scores[No_D - count] = score_min
        Index_survivor[index_order[No_D - count]] = False


    return scores, index_order

from sklearn.model_selection import KFold

def SSR(D, x, n_iter=None):
    if n_iter is None:
        n_iter = D.shape[1]
    removed = np.zeros((x.shape[1], n_iter), dtype=int)
    survivors = np.ones((x.shape[1], D.shape[1]), dtype=bool)
    for j in range(x.shape[1]):
        Y = x[...,j].copy()
        D_temp = D.copy()
        index = np.arange(D.shape[1])
        for i in range(n_iter):
            C = np.linalg.lstsq(D_temp, Y, rcond=None)[0]
            idx = np.argmin(np.abs(C))
            removed[j, i] = index[idx]
            survivors[j, index[idx]] = False
            D_temp = np.delete(D_temp, idx, axis=1)
            index = np.delete(index, idx,axis=0)
            # Y = D[:, survivors[j]]@C[np.arange(len(C))!=idx]
    return removed, survivors


def CV_SSR(D,x, k_fold = 5, random_state = 42, name_tag=None):
    n_iter = D.shape[1]
    scores = np.zeros((x.shape[1], D.shape[1]), dtype=float)
    CV_Kfolds = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    for i in range(n_iter):
        for j, (train_index, test_index) in enumerate(CV_Kfolds.split(D)):
            X_train = x[train_index]
            X_test = x[test_index]
            D_train = D[train_index]
            D_test = D[test_index]
            _, survivors = SSR(D_train, X_train, n_iter=i+1)
            for var in range(x.shape[1]):
                if i == n_iter - 5 and name_tag is not None:
                    print(name_tag[survivors[var]])
                scr = np.linalg.norm(X_test[:,var] - D_test[:,survivors[var]] @ np.linalg.pinv(D_train[:,survivors[var]]) @ X_train[:,var])
                scores[var, i] += scr
    scores = scores / k_fold
    return scores
