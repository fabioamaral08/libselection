# Imports
import numpy as np

import pysindy as ps
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.io as sio
from itertools import combinations


import warnings
from scipy.linalg import LinAlgWarning
from scipy.integrate import solve_ivp
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.linear_model import ridge_regression
from sklearn.metrics import root_mean_squared_error
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

def SINDy(x_dot, D, eps = 1e-2, alpha = 0):
    n = x_dot.shape[-1]
    nD = D.shape[-1]

    Xi = np.zeros((nD, n))

    for i in range(n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            Ei = ridge_regression(D,x_dot[...,i],alpha)
        nnz = np.count_nonzero(Ei)
        old_nnz = nD+1
        while nnz != old_nnz:
        # for k in range(10): # test: sparify 20x
            small_ind = np.abs(Ei) < eps
            Ei[small_ind] = 0
            bi = ~small_ind

            old_nnz = nnz
            nnz = np.count_nonzero(Ei)
            if nnz == 0:
                break
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                Ei[bi] = ridge_regression(D[..., bi],x_dot[...,i],alpha)


        Xi[:,i] = Ei
    return Xi
            

def databag_sindy(x_dot, D, q,tol, eps = 1e-2, alpha = 0):
    m, n = x_dot.shape
    _, nD = D.shape
    Ce = np.zeros((nD, n,q)) # Coefficients matrix
    
    index = np.arange(m)
    for i in range(q):
        samples = np.random.choice(index, m,replace=True)
        U = x_dot[samples,:]
        Theta = D[samples, :]

        Ce[:,:,i] = SINDy(U, Theta, eps, alpha)


    ip = np.mean(Ce != 0,2)
    C = np.mean(Ce,2)
    C[ip < tol ] = 0

    #optional
    small_ind = np.abs(C) < eps
    C[small_ind] = 0

    return C

def libbag_sindy(x_dot, D, ql,l, tol = 0.1, eps = 1e-2, alpha = 0):

    _, n = x_dot.shape
    _, nD = D.shape
    Cb = np.zeros((nD, n,ql)) # Coefficients matrix

    index_lib = np.arange(nD)
    Cs = SINDy(x_dot, D, eps, alpha)
    score = np.zeros(nD)
    for i in range(ql):
        sample_cols = np.random.choice(index_lib, l,replace=False)
        removed_cols  = [x for x in index_lib if x not in sample_cols]
        Db = D[:, sample_cols]
        Cb[sample_cols,:,i] = SINDy(x_dot, Db, eps, alpha)
        scr = np.linalg.norm((Cb[...,i] - Cs))
        score[removed_cols] += scr
    ip = np.mean(Cb != 0,2)
    score = score
    C = np.mean(Cb, 2)
    C[ip<tol] = 0
    keep = ip.max(1) > tol
    Dt = D[:, keep]
    return C, Dt, keep, score, ip

def print_model(C, f_names, precision = 3):
    d = C.shape[1]

    for i in range(d):
        msg = f"x{i}' = " 
        ind = np.argwhere(C[:,i]) .flatten()
        if len(ind) == 0:
            msg += '0'
        else:
            for j in ind[:-1]:
                msg += f"{C[j,i]:0.{precision}f} {f_names[j]} + "
            msg += f"{C[ind[-1],i]:0.{precision}f} {f_names[ind[-1]]} "
        print(msg)


def FUN_SINDy(t, x, solu,  lib,  z_original_part_max = 1, sel_ind =  None, u=None):
    # Assume Dict_gen_3d is a function that you've defined elsewhere
    if u is not None:
        x = np.concatenate((x, u), axis=0)
    D_temp = lib.transform(x / z_original_part_max)
    if sel_ind is not None:
        D_temp = D_temp[sel_ind]
    d = x.shape[0]
    if u is not None:
        d = d - u.shape[0]
    f = np.zeros(d)  # Initialize f as a 3x1 vector
    for i in range(d):
        f[i] = z_original_part_max * D_temp @ solu[:, i]
    return f

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

def non_uniform_diff(X,t):
    h0 = (t[1:-1] - t[:-2])[:, None]
    h1 = (t[2:] - t[1:-1])[:, None]
    x_dot = np.zeros_like(X)
    x_dot[0] = (X[1] - X[0]) / h0[0]
    x_dot[-1] = (X[-1] - X[-2]) / h1[-1]
    x_dot[1:-1] = (h1 - h0) / (h0 * h1) * X[1:-1] + (1 / (h0 + h1)) * ((h0 / h1) * X[2:] - (h1 / h0) * X[:-2])
    return x_dot

def create_lambda_functions(expressions, inputs=['x', 'y', 'w']):
    """
    Create a list of lambda functions from a list of mathematical expressions.

    Parameters:
    ----------
    expressions : list of str
        List of mathematical expressions as strings.

    Returns:
    -------
    list of lambda functions
        Each lambda function corresponds to an expression in the input list.
    """
    expressions = [x.replace('^', '**') for x in expressions]
    expressions = [x.replace(' ', '*') for x in expressions]
    expressions = [x.replace('sin', 'np.sin') for x in expressions]
    expressions = [x.replace('cos', 'np.cos') for x in expressions]

    expressions_names = [x.replace('**', '+"^"+') for x in expressions]
    expressions_names = [x.replace('*', '+ " " +') for x in expressions_names]
    expressions_names = [re.sub(r'(?<!\w)(\d+)(?!\w)', r"'\1'", x) for x in expressions_names]

    print(expressions_names)
    lambda_functions = []
    lambda_functions_names = []
    for expr, expr_nm in zip(expressions, expressions_names):
        # Create a lambda function for each expression
        func = eval(f"lambda {','.join(inputs)}: {expr}")
        func_name = eval(f"lambda {','.join(inputs)}: {expr_nm}")
        lambda_functions.append(func)
        lambda_functions_names.append(func_name)
    return lambda_functions, lambda_functions_names


def compute_phi_and_derivative(t, a, b, p, q):
    C = 1.0/(p**p*q**q)*((p+q)/(b-a))**(p+q)  #normalization
    phi = np.zeros_like(t)
    dphi = np.zeros_like(t)
    
    mask = (t > a) & (t < b)
    ta = t[mask] - a
    bt = b - t[mask]
    
    phi[mask] = C * ta**p * bt**q
    if p > 0 and q > 0:
        dphi[mask] = C * (p * ta**(p - 1) * bt**q - q * ta**p * bt**(q - 1))
    
    return phi, dphi

def generate_overlapping_test_function_matrices(t, k, l, p, q=None):
    if q is None:
        q = p
    t_min, t_max = np.min(t), np.max(t)
    total_range = t_max - t_min
    centers = np.linspace(t_min, t_max, k + 1)
    
    V = np.zeros((k+1, len(t)))
    V_prime = np.zeros((k+1, len(t)))
    
    for i, c in enumerate(centers):
        a_i = np.maximum(c - l/2, t_min)
        b_i = np.minimum(c + l/2, t_max)
        # b_i = c + l/2
        phi, dphi = compute_phi_and_derivative(t, a_i, b_i, p, q)
        V[i, :] = phi
        V_prime[i, :] = dphi
        
    return V, V_prime

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lorenz system simulation with SINDy filtering.")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step for simulation.")
    parser.add_argument("--var", type=int, default=-1, help="Variable to match (-1 for all variables). Default: -1")
    parser.add_argument("--nl", type=float, default=0.1, help="Noise Level. Default: 0.1")
    parser.add_argument("--nf", type=float, default=0.5, help="Fraction of remaining library terms. Default = 0.5" )
    parser.add_argument("--bl", type=bool, default=False, help="Use Big Library. Default = False" )

    args = parser.parse_args()

    # Use the parsed dt value
    time_step = args.dt
    # Lorenz system parameters
    sigma = 10.0
    rho = 26.0
    beta = 8.0 / 3.0
    nf = args.nf
    big_lib = args.bl
    print(f"Using Big Library: {big_lib}, Noise Level: {args.nl}, Fraction of remaining library terms: {nf}", flush=True)
    # Lorenz system equations
    def lorenz_system(state, t):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    # Time range for simulation
    t = np.arange(0, 10 + 2*time_step, time_step) # HERE
    # t = np.linspace(0, 10, 2001)
    dt =  t[1] - t[0]
    # Initial condition
    initial_state = [-8, 8, 27]

    # Solve the Lorenz system
    lorenz_data = integrate.odeint(lorenz_system, initial_state, t)

    x_train = lorenz_data
    t_train = t
    var_names = ['x', 'y', 'z']

    num_trajectories = 1
    var = args.var
    ini = 1

    if var == 0:
        n_terms = 2
    elif var == 1:
        n_terms = 3
    elif var == 2:
        n_terms = 2
    else:
        n_terms = 5
    if nf>0:
        if big_lib:
            n_terms = np.maximum(int(32 * nf), n_terms)
        else:
            n_terms = np.maximum(int(19 * nf), n_terms)

    if var > 0:
        save_dir = f"ensamble_mult_noise_dt_{time_step}_Lorenz_{var_names[var]}"
    else:
        save_dir = f"ensamble_mult_noise_dt_{time_step}_Lorenz_all"

    os.makedirs(save_dir, exist_ok=True)
    noise_levels = [args.nl]
    num_samples = 100
    rmse = root_mean_squared_error(lorenz_data, np.zeros_like(lorenz_data))
    for noise_level in noise_levels:
        noisy_list = []
        if os.path.exists(f"{save_dir}/noise_data_nl_{noise_level:g}_.npy"):
            print(f"Noise data for noise level {noise_level} already exists. Skipping generation.")
            continue
        for i in range(num_samples):
            noise = np.random.normal(0, rmse * noise_level, lorenz_data.shape)
            x_train = lorenz_data + noise
            noisy_list.append(x_train)
            if noise_level == 0:
                break  # save only 1 example of no noise
        np.save(f"{save_dir}/noise_data_nl_{noise_level:g}_.npy", noisy_list)
    # Iterate over ensemble counts

    phi_degrees = [16]
    # phi_degrees = [0]

    # results = {}

    # Ensure the directory exists
    if big_lib:
        library_gen = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary(n_frequencies=2)
    else:
        library_gen = ps.PolynomialLibrary(degree=3, include_bias=False)

    for phi_degree in phi_degrees:
        print(f"Processing phi_degree count: {phi_degree}")
        V, dV = generate_overlapping_test_function_matrices(t_train, 64, l=0.5, p=phi_degree)
        for i, noise_level in enumerate(noise_levels):
            print(f"Noise level {noise_level}:", flush=True)
            noisy_array = np.load(f"{save_dir}/noise_data_nl_{noise_level:g}_.npy")
            best_all = []
            for x_train in noisy_array:
                D = np.array(library_gen.fit_transform(x_train))
                _, No_D = D.shape
                name_tag = library_gen.get_feature_names(var_names)
                if var >= 0:
                    y = x_train[:, var:var+1]
                else:
                    y = x_train
                
                if phi_degree == 0:
                    x_dot = non_uniform_diff(y, t_train)
                    best_combination, best_score = iter_scores(D[ini:-1], x_dot[ini:-1], num_trajectories, n_terms)
                else:

                    G = V@D
                    b = -dV@y
                    print(f"Shape of G: {G.shape}, Shape of b: {b.shape}, n_terms: {n_terms}", flush=True)
                    best_combination, best_score = iter_scores(G, b, n_terms=n_terms, stepwise=False, stepsize=1, replace=False)
                    # best_combination, best_score = iter_scores(G, b, num_trajectories, n_terms)
                # Save Best Terms
                best_all.append(best_combination)
                print(f"Best combination for noise level {noise_level}: {best_combination}, score: {best_score}")
                print(f"{save_dir}/best_terms_exhaustion_phi_{phi_degree}_nl_{noise_level:g}_nf_{n_terms}_big_lib_{big_lib}.npy")
            #save the best terms for this noise level
            best_all = np.array(best_all)
            np.save(f"{save_dir}/best_terms_exhaustion_phi_{phi_degree}_nl_{noise_level:g}_nf_{n_terms}_big_lib_{big_lib}.npy", best_all)
