import numpy as np
import warnings
from scipy.linalg import LinAlgWarning, pascal
from sklearn.linear_model import ridge_regression
from itertools import product

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

        phi, dphi = compute_phi_and_derivative(t, a_i, b_i, p, q)
        V[i, :] = phi
        V_prime[i, :] = dphi
        
    return V, V_prime

def SINDy(x_dot, D, eps = 1e-2, alpha = 0):
    _, n = x_dot.shape
    _, nD = D.shape

    Xi = np.zeros((nD, n))

    for i in range(n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            Ei = ridge_regression(D,x_dot[:,i],alpha)
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
                Ei[bi] = ridge_regression(D[:, bi],x_dot[:,i],alpha)


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

def gen_weaky_lib(U,lib_gen,dx, dt, supp_size_x, supp_size_t, max_dx, max_dt = 1, tol=1e-10, allow_combinations=False, var_names=None):
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
    # # Calculate scale_x similar to the MATLAB expression
    # max_dx_half_floor = int(np.floor(max_dx / 2))
    # max_dx_half_ceil = int(np.ceil(max_dx / 2))

    # p_x = px  # px is already computed above

    # numerator = prod([p_x - (i) for i in range(max_dx_half_floor)])
    # denominator = prod(range(1, max_dx_half_ceil + 1)) * prod(range(1, max_dx + 1))
    # scale_x = (numerator / denominator) ** (1 / max_dx) / (supp_size_x * dx)

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

    if var_names is not None:
        for i,vn in enumerate(var_names):
            name_tag_der = [n.replace(f'u{i+1}',vn) for n in name_tag_der]
    return np.column_stack(Theta_pdx), np.column_stack(b), name_tag_der