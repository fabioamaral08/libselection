# Imports
import numpy as np

import pysindy as ps
import scipy.io as sio
from itertools import combinations, product
from scipy import integrate

import warnings
from scipy.linalg import LinAlgWarning, pascal
from sklearn.linear_model import ridge_regression
import re
from joblib import Parallel, delayed
import os
import argparse



# Força bibliotecas internas a 1 thread por processo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



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

def iter_scores(D, y,n_terms, stepwise=False, stepsize=1, replace=False):
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
        n_itens = len(all_itens)
        while n_itens > n_terms:
            lib_size = np.maximum(n_itens - stepsize, n_terms)
            combinations_list = list(combinations(all_itens, lib_size))
            all_scores = Parallel(n_jobs=-1)(
                delayed(score_fn)(Pd, D, comb, y, norm_y)
                for comb in combinations_list
            )

            best_score = np.argmin(all_scores)
            all_itens = list(combinations_list[best_score])
            n_itens = len(all_itens)
            if replace:
                survivors = np.zeros(D.shape[1], dtype=bool)
                survivors[all_itens] = True
                D[:, ~survivors] = 0
                Pd = D @ np.linalg.pinv(D)
        best_combination = all_itens
    else:
        combinations_list = list(combinations(all_itens, n_terms))
        all_scores = Parallel(n_jobs=-1)(
            delayed(score_fn)(Pd, D, comb, y, norm_y)
            for comb in combinations_list
        )
        best_score = np.argmin(all_scores)
        best_combination = combinations_list[best_score]
    return best_combination, best_score,



def inner_prod(fx,gx,x_dot,t_train, p = False):
    '''
    line integral
    each column is a function
    '''
    fx = fx.flatten()
    gx = gx.flatten()
    if p:
        print(">>",fx.shape, gx.shape, x_dot.shape, t_train.shape, np.sqrt(np.sum(x_dot ** 2, axis=1)).shape)
        print((fx * gx * np.sqrt(np.sum(x_dot ** 2, axis=1))).shape)

    inner = integrate.trapezoid(fx * gx * np.sqrt(np.sum(x_dot ** 2, axis=1)), t_train).sum()
    return float(inner)

def Gram(A):
    """
    Perform Gram-Schmidt orthonormalization.
    """
    n, k = A.shape
    Q = np.zeros_like(A)
    for ii in range(k):
        Q[:, ii] = A[:, ii]
        for jj in range(ii):
            # Q[:, ii] -= np.dot(Q[:, jj], Q[:, ii]) / np.dot(Q[:, jj], Q[:, jj]) * Q[:, jj]
            Q[:, ii] -= np.dot(Q[:, jj], Q[:, ii]) * Q[:, jj]
        Q[:, ii] /= np.linalg.norm(Q[:, ii])
    return Q

def Gram_L2(A, x_dot, t_train):
    """
    Perform Gram-Schmidt orthonormalization with line integration.
    """
    n, k = A.shape
    Q = np.zeros_like(A)
    for ii in range(k):
        Q_ii = A[:, ii].copy()
        for jj in range(ii):
            # projection = (inner_prod(Q[:, jj], Q[:, ii], x_dot, t_train, False) / inner_prod(Q[:, jj], Q[:, jj], x_dot, t_train, False))
            projection = inner_prod(Q[:, jj], A[:, ii], x_dot, t_train, False) / inner_prod(Q[:, jj], Q[:, jj], x_dot, t_train, False)
            Q_ii -= projection * Q[:, jj]
        norm = np.sqrt(inner_prod(Q_ii, Q_ii, x_dot, t_train, False))
        if norm > 1e-10:
            Q[:, ii] = Q_ii/norm
        else:
            print(f"Function {ii} nearly linearly dependent, skipped.")
        
    return Q

def mixining(D, i):
    """
    Move the ith column (starting from 0) to the last position.
    """
    D1 = np.hstack((D, D[:, i][:, None]))
    D1 = np.delete(D1, i, axis=1)
    return D1

def calculate_score(y, D, num_trajectories, t_train, dim, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True):

    """
    Calculate the score based on the given parameters.
    """
    y_score_normalization = np.linalg.norm(y)

    if Gram_Schmidt:
        if Line_integration:
            return _calculate_score_with_line_integration(y, D, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization)
        else:
            return _calculate_score_without_line_integration(y, D, num_trajectories, dim, Trajectorywise_score, y_score_normalization)
    else:
        return _calculate_score_non_gram(y, D, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization)

def _calculate_score_with_line_integration(y, D, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization):

    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    score_c = 0
    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        Q_temp = Gram_L2(part, y1, t_train)
        for iii in range(dim):
            score_c += np.abs(inner_prod(Q_temp[:, -1], y1[:, iii], y1, t_train)) / inner_prod(Q_temp[:, -1], Q_temp[:, -1], y1, t_train)
    return score_c / y_score_normalization

def _calculate_score_without_line_integration(y, D, num_trajectories, dim, Trajectorywise_score, y_score_normalization):


    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    score_c = 0
    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        Q_temp = Gram(part)
        
        #[Gram] (default) score: projection - projection
        # Q_pinv = np.linalg.pinv(Q_temp)
        # Q_one_delete_pinv = np.linalg.pinv(Q_temp[:,:-1])
        # score_c += np.linalg.norm(np.dot(Q_temp, Q_pinv) @ y1 - np.dot(Q_temp[:,:-1], Q_one_delete_pinv) @ y1)

        #[Gram] if you are using Gram, it is alternative calculation of the score
        score_c += np.linalg.norm(np.dot(Q_temp, Q_temp.T) @ y1 - np.dot(Q_temp[:,:-1], Q_temp[:,:-1].T) @ y1)

        # for iii in range(dim):
        #     print(np.linalg.matrix_rank(Q_temp).shape, y1[:, iii].shape)
        #     score_c += np.abs(np.inner(np.linalg.matrix_rank(Q_temp), y1[:, iii]))
    return score_c / y_score_normalization

def _calculate_score_non_gram(y, D, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization):

    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    score_c = 0
    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        Q_temp = part
        Q_pinv = np.linalg.pinv(Q_temp)
        Q_one_delete_pinv = np.linalg.pinv(Q_temp[:, :-1])
        score_c += np.linalg.norm(np.dot(Q_temp, Q_pinv) @ y1 - np.dot(Q_temp[:, :-1], Q_one_delete_pinv) @ y1)
    return score_c / y_score_normalization

def lowest_score(y, D, num_trajectories, t_train, dim, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True):


    """
    Find the column with the lowest score.
    """
    No_of_dictionary = D.shape[1]
    history_temp = np.zeros(No_of_dictionary)
    for i in range(No_of_dictionary):
        scr =  calculate_score(y, mixining(D, i), num_trajectories, t_train, dim, Gram_Schmidt, Line_integration, Trajectorywise_score)
        history_temp[i] = scr
    argmin = np.argmin(history_temp)
    return argmin, history_temp

def low_score(y, D, num_trajectories, t_train, dim, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True):

    """
    Calculate scores for all columns.
    """
    No_of_dictionary = D.shape[1]
    history_temp = np.zeros(No_of_dictionary)
    for i in range(No_of_dictionary):
        history_temp[i] = calculate_score(y, mixining(D, i), num_trajectories, t_train, dim, Gram_Schmidt, Line_integration, Trajectorywise_score)
    return history_temp

def mixining_multi(D, index_vec):

    """
    Rearrange columns of D based on index_vec.
    """
    selected_cols = D[:, index_vec == 0]
    remaining_cols = D[:, index_vec == 1]
    return np.hstack([remaining_cols, selected_cols])

def score_multi(y,y_full, D, how_many_vanished, num_trajectories, t_train, dim, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True):

    """
    Calculate score for multiple columns.
    """
    y_score_normalization = np.linalg.norm(y)

    if Gram_Schmidt:
        if Line_integration:
            return _score_multi_with_line_integration(y,y_full, D, how_many_vanished, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization)
        else:
            return _score_multi_without_line_integration(y, D, how_many_vanished, num_trajectories, dim, Trajectorywise_score, y_score_normalization)
    else:
        return _score_multi_non_gram(y, D, how_many_vanished, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization)

def _score_multi_with_line_integration(y,y_full, D, how_many_vanished, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization):

    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    split_y_full = np.array_split(y_full, k, axis=0)
    score_c = 0

    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        y1_full = split_y_full[ii]
        Q_temp = Gram_L2(part, y1_full, t_train)
        #y1_norm = np.sqrt(inner_prod(y1, y1, y1, t_train))
        for iii in range(dim):
            for jjj in range(-how_many_vanished,0):
                score_c += np.abs(inner_prod(Q_temp[:, jjj], y1[:, iii], y1_full, t_train)) #/ y1_norm

            # score_c += np.abs(inner_prod(Q_temp[:, -how_many_vanished:], y1[:, iii], y1, t_train)) / inner_prod(Q_temp[:, -how_many_vanished:], Q_temp[:, -how_many_vanished:], y1, t_train)
    return score_c / y_score_normalization

def _score_multi_without_line_integration(y, D, how_many_vanished, num_trajectories, dim, Trajectorywise_score, y_score_normalization):


    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    score_c = 0
    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        Q_temp = Gram(part)
        score_c += np.linalg.norm(np.dot(Q_temp, Q_temp.T) @ y1 - np.dot(Q_temp[:, :-how_many_vanished], Q_temp[:, :-how_many_vanished].T) @ y1)
    return score_c / y_score_normalization

def _score_multi_non_gram(y, D, how_many_vanished, num_trajectories, t_train, dim, Trajectorywise_score, y_score_normalization):

    if not Trajectorywise_score:
        k = 1
    else:
        k = num_trajectories
    split_matrices = np.array_split(D, k, axis=0)
    split_y = np.array_split(y, k, axis=0)
    score_c = 0
    for ii, part in enumerate(split_matrices):
        y1 = split_y[ii]
        Q_temp = part
        Q_pinv = np.linalg.pinv(Q_temp)
        Q_one_delete_pinv = np.linalg.pinv(Q_temp[:, :-how_many_vanished])
        score_c += np.linalg.norm(np.dot(Q_temp, Q_pinv) @ y1 - np.dot(Q_temp[:, :-how_many_vanished], Q_one_delete_pinv) @ y1)
    return score_c / y_score_normalization

def lowest_score_union(y,y_full, D,history_index, num_trajectories, t_train, dim, how_many_vanished, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True,ensamble_count=0):
    """
    Find the column with the lowest score, considering the union of removed columns and one column from the remaining D.
    """
    valid_index = np.argwhere(history_index != 0)
    No_of_dictionary = valid_index.shape[0]
    emsamble = ensamble_count > 0

    iteration_count = ensamble_count if emsamble else 1
    m = y.shape[0]
    data_index = np.arange(m)


    def compute_score(count_ensemble):
        history = np.zeros(( No_of_dictionary))
        if emsamble:
            samples = np.random.choice(data_index, m,replace=True)
            U = y[samples,:]
            U_full = y_full[samples,:]
            Theta = D[samples, :]
        else:
            U = y
            U_full = y_full
            Theta = D
        for ii, index in enumerate(valid_index):
            hist_aux = history_index.copy()
            hist_aux[index] = 0
            scr = score_multi(
                        U,U_full, mixining_multi(Theta, hist_aux), how_many_vanished, num_trajectories, 
                        t_train, dim=dim, Gram_Schmidt=Gram_Schmidt, Line_integration=Line_integration, 
                        Trajectorywise_score=True)
            history[ii] += scr
        return history
    history_temp = Parallel(n_jobs=-1)(delayed(compute_score)(p) for p in range(iteration_count))
    history_temp = np.array(history_temp)

    history_temp = np.median(history_temp, axis=0) if emsamble else history_temp
    argmin_v = np.argmin(history_temp)
    return argmin_v, history_temp

def filtering(y,y_full, D, q, name_tag, num_trajectories, t_train, Gram_Schmidt=True, Line_integration=False, Trajectorywise_score=True, ensamble_count = 0, verbose=True, memory = 0):
    """
    Perform filtering on the dictionary `D` based on the given parameters.

    Parameters:
    ----------
    y : numpy.ndarray
        The target data, typically x_dot.
    D : numpy.ndarray
        The dictionary matrix where columns represent items.
    q : int
        The number of iterations for filtering (used for optimal sparsity level).
    name_tag : list
        List of names corresponding to the columns of the dictionary `D`.
    num_trajectories : int
        Number of trajectories in the data.
    t_train : numpy.ndarray
        Training time data.
    Gram_Schmidt : bool, optional
        Whether to use the Gram-Schmidt process for orthonormalization. Default is True.
    Line_integration : bool, optional
        Whether to use line integration. Default is False.
    Trajectorywise_score : bool, optional
        Whether to calculate scores trajectory-wise. Default is True.
    verbose : bool, optional
        Whether to print detailed information during execution. Default is True.

    Returns:
    -------
    history_value : numpy.ndarray
        History of scores for each iteration.
    history_index : numpy.ndarray
        History of indices indicating which columns are retained.
    Subdictionary : numpy.ndarray
        Additional scores for the subdictionary at each iteration.
    """
    if not Gram_Schmidt and Line_integration:
        raise NotImplementedError("When 'Gram_Schmidt' is False and 'Line_integration' is True, this feature is not implemented yet.")

    if verbose:
        print('It is Gram-Schmidt version.' if Gram_Schmidt else 'You are not using the Gram-Schmidt process.')
        if Line_integration:
            print('You are using line integration.')

    dim = y.shape[1]
    No_of_dictionary = D.shape[1]
    if verbose:
        print('starting_sparsity_dictionary =', No_of_dictionary)

    D_update = D.copy()
    history_value = -1 * np.ones((No_of_dictionary, No_of_dictionary))
    history_index = np.ones((No_of_dictionary, No_of_dictionary))
    Subdictionary = np.ones((No_of_dictionary, 1))

    for ii in range(1, q + 1):
        argmin, history_value[ii, history_index[ii - 1, :].astype(bool)] = lowest_score_union(
            y,y_full, D, history_index[ii-1,:], num_trajectories, t_train, dim, ii, Gram_Schmidt=Gram_Schmidt, 
            Line_integration=Line_integration, Trajectorywise_score=Trajectorywise_score, ensamble_count=ensamble_count)

        # Update the history index
        # Remove the column with the lowest score from the history index
        A = history_index[ii - 1, :].copy()
        indices_of_ones = A[history_index[ii - 1, :].astype(bool)]
        indices_of_ones[argmin] = 0
        A[history_index[ii - 1, :].astype(bool)] = indices_of_ones
        history_index[ii, :] = A

        # Update the dictionary with the remaining columns
        D_update = D[:, history_index[ii, :].astype(bool)]
        deleted_index = np.where(history_index[ii - 1, :] - history_index[ii, :] == 1)[0]


        if verbose:
            print('D_update.shape=', D_update.shape, 'filtered=', name_tag[int(deleted_index.item())], 
                    'score_filtered=', history_value[ii, int(deleted_index.item())])

            if ii > 1:
                print('\tcurrent/previous=', history_value[ii, int(deleted_index.item())] / 
                    history_value[ii - 1, int(deleted_index_previous.item())])
        
        
        deleted_index_previous = deleted_index.copy()
        how_many_vanished = np.sum(history_index[ii, :] == 0)

        # Calculate the additional score for the subdictionary
        Additional_score_multi = score_multi(
            y, y_full, mixining_multi(D, history_index[ii, :]), how_many_vanished, num_trajectories, 
            t_train, dim=len(y[0, :]), Gram_Schmidt=Gram_Schmidt, Line_integration=Line_integration, 
            Trajectorywise_score=Trajectorywise_score)
        if verbose:
            print('\tFYI=', Additional_score_multi, '(Similarity) score_filtered/FYI', 
                    history_value[ii, int(deleted_index.item())] / Additional_score_multi)
        Subdictionary[ii] = Additional_score_multi

    history_value = np.delete(history_value, 0, axis=0)
    history_index = np.delete(history_index, 0, axis=0)
    Subdictionary = Subdictionary[1:, :]
    if verbose:
        print('Last =', name_tag[int(np.where(history_index[-1, :] == 1)[0].item())])
    return history_value, history_index, Subdictionary



def get_der_order(max_dx, n, allow_combinations=False):
    combinations = []
    if allow_combinations:
        for indexes in product(range(max_dx+1), repeat=n-1):
            indexes = list(indexes)
            if sum(indexes) <= max_dx:  
                combinations.append(indexes + [0])
    else:
        combinations.append(np.zeros(n, dtype=int))  # Include the zero derivative case
        for i in range(n-1):
            der_ind = np.zeros(n, dtype=int)
            for j in range(1,max_dx+1):
                der_ind[i] = j
                combinations.append(der_ind.copy())
    return np.array(combinations)

def phi_int_weights(m, d, tol):
    """
    Compute the integration weights for basis functions (phi) and their derivatives
    for use in weak formulations or other approximation techniques.

    Parameters:
    - m: number of grid intervals (defines resolution)
    - d: number of derivatives (order)
    - tol: tolerance for decay; if negative, sets p directly
    - phi_class: type of basis (only class 1 is considered here)

    Returns:
    - Cfs: array of shape (d+1, 2m+1), containing values of basis function and its derivatives
           at the integration grid points
    - p: power parameter that controls decay of basis function
    """

    # Choose p to ensure decay of the basis near boundaries,
    # unless explicitly set by negative tol
    if tol < 0:
        p = -tol
    else:
        # Estimate p so that basis function value near boundary ~ tol
        p = int(np.ceil(max(np.log(tol) / np.log((2 * m - 1) / m**2), d + 1)))

    # Define the normalized grid: t in [0, 1] with m intervals
    t = np.arange(0, m + 1) / m

    # t_L and t_R will hold values of (1 ± t)^q for q = p−d to p
    t_L = np.zeros((d + 1, m + 1))
    t_R = np.zeros((d + 1, m + 1))

    for j in range(m + 1):
        powers = np.flip(np.arange(p - d, p + 1))  # descending powers
        t_L[:, j] = (1 + t[j]) ** powers
        t_R[:, j] = (1 - t[j]) ** powers

    # Coefficients for derivatives of (1 ± t)^p
    ps = np.ones(d + 1)
    for q in range(1, d + 1):
        ps[q] = (p - q + 1) * ps[q - 1]

    # Multiply each row by its derivative coefficient
    t_L = ps[:, None] * t_L
    t_R = ((-1) ** np.arange(d + 1))[:, None] * ps[:, None] * t_R

    # Initialize output array: (d+1) rows, (2m+1) grid points
    Cfs = np.zeros((d + 1, 2 * m + 1))

    # Compute the function value (0th derivative) as product of left and right
    Cfs[0, :] = np.concatenate((
        np.flip(t_L[0, :] * t_R[0, :]),       # mirror left half
        t_L[0, 1:] * t_R[0, 1:]               # right half (excluding center overlap)
    ))

    # Generate Pascal matrix for binomial coefficients
    P = np.fliplr(pascal(d + 1))  # flipped for correct diagonal extraction
    # Compute higher-order derivatives
    for k in range(1, d + 1):
        binoms = np.diag(P, d - k)  # extract diagonal entries
        Cfs_temp = np.zeros(m + 1)
        for j in range(k + 1):
            Cfs_temp += binoms[j] * t_L[k - j, :] * t_R[j, :]
        # Concatenate mirrored and forward halves
        Cfs[k, :] = np.concatenate((
            (-1) ** k * np.flip(Cfs_temp),
            Cfs_temp[1:]
        ))

    return Cfs, int(p)

def convNDfft(X, cols, sub_inds, ver=2):
    """
    Perform separable N-dimensional convolution using FFT along each dimension,
    restricted to 'valid' points (non-padded regions).

    Parameters:
    - X: input multidimensional array (e.g. the nonlinear term u^2)
    - cols: list of 1D filters (FFT-transformed or raw) for each dimension
    - sub_inds: list of slicing indices for each dimension, specifying the valid region
    - ver: version flag; if ver==1, perform FFT on filter inside this function.
           if ver==2, assume cols[k] are already FFT-transformed

    Returns:
    - X: the convolved result, restricted to the subdomain defined by sub_inds
    """
    Ns = X.shape
    dim = len(Ns)
    for k in range(dim):
        # Determine FFT of the kernel along dimension k
        if ver == 1:
            col = cols[k].reshape(-1)
            n = len(col)
            padded = np.concatenate([np.zeros(Ns[k] - n), col])
            col_fft = np.fft.fft(padded)
        else:
            col_fft = cols[k].reshape(-1)

        # Axes reordering to bring axis-k to front
        shift = np.roll(np.arange(dim), -(k+1))
        shift_back = np.roll(np.arange(dim),k+1)
        # Nss = [Ns[o] for o in shift]

        # Permute so that the convolution axis is first
        X_perm = np.transpose(X, shift)

        # FFT along first axis, multiply, then inverse FFT
        X_fft = np.fft.fft(X_perm, axis=-1)
        X_filtered = np.fft.ifft(col_fft * X_fft, axis=-1)
        # X_filtered = np.reshape(X_filtered, Nss)
        # Slice to valid region (sub_inds[k])
        # Create indexing tuple like (slice(start, end), :, :, ...)
        inds = [slice(None)] * dim
        inds[-1] = sub_inds[k]
        X = np.transpose(X_filtered[tuple(inds)], shift_back)
    return np.real(X)

def gen_weaky_lib(U,lib_gen,dx, dt, supp_size_x, supp_size_t, max_dx, max_dt = 1, tol=1e-10, allow_combinations=False):
    if not isinstance(U, list):
        U = [U]

    Cfs_x,px = phi_int_weights(supp_size_x, max_dx, tol)
    Cfs_t,pt = phi_int_weights(supp_size_t, max_dt, tol)

    dim = U[0].ndim
    n = len(U)
    der_index = get_der_order(max_dx, dim, allow_combinations=allow_combinations)
    Ns = U[0].shape
    U_arr = np.stack(U, axis=-1)  # shape: (Nx, Nt, n)
    lib_gen = lib_gen.fit(U_arr.reshape(-1, n))  # Fit the library generator to the data
    D = lib_gen.transform(U_arr)  # shape: (Nx*Nt, num_terms)
    # D = np.array(lib_gen.fit_transform(U_arr.reshape(-1, n)))  # shape: (Nx*Nt, num_terms)
    # D = D.reshape((*U_arr.shape[:-1], -1))  # (Nx, Nt, num_terms)
    name_tag = lib_gen.get_feature_names(['u{}'.format(i+1) for i in range(n)])
    name_tag_der = [x for x in name_tag]

    if lib_gen.include_bias:
        name_tag = name_tag[1:]  # Exclude the bias term from the names
    
    # if scales is None:
    scales = np.ones(dim)


    Cfs_ffts = [None] * dim
    sub_inds = [None] * dim
    # Spatial dimensions
    mm, nn = Cfs_x.shape
    for k in range(dim - 1):
        if max_dx > 0:
            sub_inds[k] = np.arange(0, Ns[k] - 2 * supp_size_x + 1, supp_size_x)
            pad = np.zeros((mm, Ns[k] - nn))
            scaled = (supp_size_x * dx * scales[k]) ** (-np.arange(mm))[:, None]
            padded = np.hstack([
                pad,
                scaled * Cfs_x / nn
            ])
        else:
            sub_inds[k] = np.arange(1)
            padded = np.ones((mm, 1))
        Cfs_ffts[k] = np.fft.fft(padded, axis=1)

    # Temporal dimension
    mm, nn = Cfs_t.shape
    scaled = (supp_size_t * dt * scales[dim - 1]) ** (-np.arange(mm))[:, None]
    padded = np.hstack([
        np.zeros((mm, Ns[dim - 1] - nn)),
        scaled * Cfs_t / nn
    ])
    Cfs_ffts[-1] = np.fft.fft(padded, axis=1)
    sub_inds[-1] = np.arange(1, Ns[-1] - 2 * supp_size_t + 1, supp_size_t)
    Theta_pdx = []
    fcn = D  # Use precomputed nonlinear field  
    for der_tag in der_index:
        test_conv_cell = []
        for k in range(dim-1):
            order = int(der_tag[k])
            test_conv_cell.append(Cfs_ffts[k][order, :])
        test_conv_cell.append(Cfs_ffts[-1][0, :])

        # Perform separable convolution in each dimension
        for j in range(D.shape[-1]):
            if lib_gen.include_bias and j == 0 and not np.all(der_tag == 0):
                # Skip the bias term
                continue
            fcn_conv = convNDfft(fcn[...,j], test_conv_cell, sub_inds)

            # Store flattened column
            Theta_pdx.append(fcn_conv.ravel())
        if np.all(der_tag == 0):
            continue
        der_nortation = '{' +''.join([f'{k}'*i for k,i in enumerate(der_tag)]) + '}'
        name_tag_der += [f"∂_{der_nortation} " + nt for nt in name_tag]

    b = []
    test_conv_cell = []
    for k in range(dim-1):
        order = 0
        test_conv_cell.append(Cfs_ffts[k][order, :])
    test_conv_cell.append(Cfs_ffts[-1][max_dt, :])

    # Perform separable convolution in each dimension
    for u in U:
        u_t = convNDfft(u, test_conv_cell, sub_inds)

        # Store flattened column
        b.append(u_t.ravel())  # Ensure b is a column vector

    return np.column_stack(Theta_pdx), np.column_stack(b), name_tag_der

def root_mean_squared_error(x):
    """
    Calculate the root mean squared error between x and zero.
    """
    return np.sqrt(np.mean(x**2))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lorenz system simulation with SINDy filtering.")
    parser.add_argument("--file", type=str, default="burgers.mat", help="File to load data from.")
    parser.add_argument("--nl", type=float, default=0.1, help="Noise Level. Default: 0.1")
    parser.add_argument("--var", type=int, default=0, help="Variable index to use for SINDy. Default = 0")

    args = parser.parse_args()


    # Lorenz system parameters

    var = args.var
    filename = args.file

    num_trajectories = 1
    ini = 1


    data = sio.loadmat(f'data/{filename}')
    case = filename.split('.')[0]
    if filename == 'burgers.mat':
        u = data['usol'].real
        U = [u]
        x = data['x'][0]
        t = data['t'][:,0]
        n_terms = 1
    elif filename == 'NLS.mat':
        u = data['U_exact'][...,0].item()
        v = data['U_exact'][...,1].item()
        U = [u,v]
        x = data['x'][:,0]
        t = data['t'][0]    

        n_terms = 3
    elif filename == 'RD.mat':
        u = data['u']
        v = data['v']
        x = data['x'][0]
        t = data['t'][0]
        U = [u, v]
        n_terms = 7
    elif filename == 'KS.mat':
        u = data['U_exact'].item()
        U = [u]
        x = data['x'][:,0]
        t = data['t'][0]
        n_terms = 3

    dx = x[1] - x[0]
    dt = t[1] - t[0]


    save_dir = f"Weak_PDE"

    os.makedirs(save_dir, exist_ok=True)
    noise_levels = [args.nl]
    num_samples = 100

    for noise_level in noise_levels:
        if os.path.exists(f"{save_dir}/noise_{case}_nl_{noise_level:g}_.npy"):
            print(f"Noise data for noise level {noise_level} already exists. Skipping generation.")
            continue
        noisy_list = []
        for i in range(num_samples):
            noise_u = []
            for ui in U:
                rmse = root_mean_squared_error(ui)
                noise = np.random.normal(0, rmse * noise_level, ui.shape)
                x_train = ui + noise
                noise_u.append(x_train)
            noisy_list.append(noise_u)
            if noise_level == 0:
                break  # save only 1 example of no noise
        np.save(f"{save_dir}/noise_{case}_nl_{noise_level:g}_.npy", noisy_list)
        # Iterate over ensemble counts



    library_gen = ps.PolynomialLibrary(degree=3, include_bias=True)

    for i, noise_level in enumerate(noise_levels):
        print(f"Noise level {noise_level}:", flush=True)
        noisy_array = np.load(f"{save_dir}/noise_{case}_nl_{noise_level:g}_.npy")
        best_all = []
        hist_val_all = []
        hist_ind_all = []
        for u_data in noisy_array:
            x_train = [u for u in u_data]
            G,b, name_tag = gen_weaky_lib(x_train,library_gen, dx, dt, 20, 20, max_dx=4, max_dt = 1, tol=1e-10, allow_combinations=False)
            No_D = G.shape[1]  # Number of dictionary terms
            # if case == 'NLS':
            #     best_combination, best_score = iter_scores(G, b[...,var:var+1], n_terms=n_terms, stepwise=True, stepsize=1, replace=False)
            # else:
            #     best_combination, best_score = iter_scores(G, b, n_terms=n_terms, stepwise=False, stepsize=1, replace=False)
            # best_all.append(best_combination)
            # print(f"Best combination for noise level {noise_level}: {best_combination}, score: {best_score}")
            if var >= 0:
                b = b[...,var:var+1]
            history_value, history_index, _ = filtering(b, b, G, No_D-1, name_tag, num_trajectories, t,
                                    Gram_Schmidt = False,
                                    Line_integration = False,
                                    Trajectorywise_score = False,
                                    verbose= False)
            hist_val_all.append(history_value)
            hist_ind_all.append(history_index)

        #save the best terms for this noise level
        
        # best_all = np.array(best_all)
        hist_val_all = np.array(hist_val_all)
        hist_ind_all = np.array(hist_ind_all)
        if var >= 0:
            # np.save(f"{save_dir}/best_terms_exhaustion_nl_{noise_level:g}_nf_{n_terms}_{case}_var_{var}.npy", best_all)
            np.save(f"{save_dir}/hist_val_stepwise_nl_{noise_level:g}_{case}_var_{var}.npy", hist_val_all)
            np.save(f"{save_dir}/hist_ind_stepwise_nl_{noise_level:g}_{case}_var_{var}.npy", hist_ind_all)
        else:
            # np.save(f"{save_dir}/best_terms_exhaustion_nl_{noise_level:g}_nf_{n_terms}_{case}.npy", best_all)
            np.save(f"{save_dir}/hist_val_stepwise_nl_{noise_level:g}_{case}.npy", hist_val_all)
            np.save(f"{save_dir}/hist_ind_stepwise_nl_{noise_level:g}_{case}.npy", hist_ind_all)
