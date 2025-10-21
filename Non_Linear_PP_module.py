# -*- coding: utf-8 -*-
"""
This is an module that implements the nonlinear PP calculations of the article:
Malý, P., Lüttig, J., Rose, P.A. et al. Separating single- from multi-particle 
dynamics in nonlinear spectroscopy. Nature 616, 280–287 (2023). 
https://doi.org/10.1038/s41586-023-05846-7


@author: Alessandro
"""

import numpy as np
from math import comb  # for binomial coefficients

# calculates w and I_p with functions
def w(n, N, p):
    """
    Compute w_p^(nQ) as defined by:
        w_p^(nQ) = (1 / (2N)) * ((2 - δ_{p,1}) / (1 + δ_{n,N})) * cos(n * (p - 1) * 2π / (2N))
    
    Parameters
    ----------
    n : int or float
        Index n
    N : int or float
        Parameter N (total count or normalization factor)
    p : int or float
        Index p
    
    Returns
    -------
    float
        Value of w_p^(nQ)
    """
    delta_p1 = 1 if p == 1 else 0
    delta_nN = 1 if n == N else 0

    w_value = (1 / (2 * N)) * ((2 - delta_p1) / (1 + delta_nN)) * np.cos(n * (p - 1) * 2 * np.pi / (2 * N))
    return w_value

def I_p_vector(I0, N):
    """
    Compute I_p = 4 I0 cos^2((p-1) * pi / (2N)), for p=1..N
    Returns an array of length N (p index 0..N-1 corresponds to p=1..N).
    """
    p = np.arange(1, N+1)                     # 1..N
    angles = (p - 1) * np.pi / (2 * N)        # (p-1)*pi/(2N)
    Ip = 4.0 * I0 * (np.cos(angles) ** 2)
    return Ip

#calculates PP_nQ given N and PP_I_p

def PP_nQ(I_values, n, N):
    """
    Compute PP^(nQ)(I0) = sum_{p=1}^{N} w_p^(nQ) * PP(I_p)

    Parameters
    ----------
    I_values : array-like
        Sequence of PP(I_p) values, length N (for p = 1..N)
    n : int
        Index n (order)
    N : int
        Total number of points (same as len(I_values))

    Returns
    -------
    float
        Value of PP^(nQ)(I0)
    """
    I_values = np.asarray(I_values, dtype=float)
    assert len(I_values) == N, "Length of I_values must equal N"

    total = 0.0
    for p in range(1, N + 1):
        total += w(n, N, p) * I_values[p - 1]

    return total

def PP_all_nQ(I_values):
    N = len(I_values)
    I_values = np.asarray(I_values, dtype=float)
    W = np.zeros((N, N))

    for n in range(1, N + 1):
        for p in range(1, N + 1):
            W[n - 1, p - 1] = w(n, N, p)

    return W @ I_values  # matrix multiplication: gives all PP^(nQ)(I0)

# Computes PP_nQ but in another way


# Gives relations between PP_nQ_from_PPodd and Inverse Matrix

def build_C_matrix(l):
    """
    Build matrix C (l x l) with C[n-1, r-1] = binom(2r, r-n) for r>=n, else 0.
    Indices: n, r start at 1 in math; arrays are 0-based.
    """
    C = np.zeros((l, l), dtype=float)
    for n in range(1, l+1):
        for r in range(n, l+1):
            C[n-1, r-1] = comb(2*r, r-n)
    return C

def compute_PP_nQ_from_PPodd(PP_odd, I0):
    """
    Compute vector w = [PP^(1Q)(I0), PP^(2Q)(I0), ..., PP^(lQ)(I0)]
    given PP_odd = [PP^(3), PP^(5), PP^(7), ..., PP^(2l+1)]
    and scalar I0. Length l = len(PP_odd).
    
    Returns
    -------
    w : numpy.ndarray, shape (l,)
    """
    PP_odd = np.asarray(PP_odd, dtype=float)
    l = len(PP_odd)
    # b_r = PP^(2r+1) * I0^r with r from 1..l
    r_indices = np.arange(1, l+1)
    b = PP_odd * (I0 ** r_indices)
    
    C = build_C_matrix(l)
    w = C.dot(b)   # w_n = sum_{r=n}^l binom(2r, r-n) * b_r
    return w

def build_inverse_A(l):
    """
    Build A = C^{-1}. This reproduces the matrix you posted (with alternating signs).
    Use with care for large l (numerical conditioning).
    """
    C = build_C_matrix(l)
    A = np.linalg.inv(C)
    return A

#Define matrix w, the matrix to invert and the transformation between PP(Ip)

def w_matrix(N):
    """Compute w_p^(nQ) for all n,p (both 1..N)."""
    n = np.arange(1, N+1)[:, np.newaxis]  # shape (N,1)
    p = np.arange(1, N+1)[np.newaxis, :]  # shape (1,N)
    delta_p1 = (p == 1).astype(float)
    delta_nN = (n == N).astype(float)
    w = (1/(2*N)) * (2 - delta_p1) / (1 + delta_nN) * np.cos(n * (p-1) * 2*np.pi / (2*N))
    return w

def extract_PP_non_lin_orders(PP_measured):
    """
    Reconstruct first N nonlinear coefficients PP^(2r+1) from measured PP(I_p).
    
    Parameters
    ----------
    PP_measured : array-like of shape (N,)
        Measured PP(I_p) datasets (one per intensity)
        
    Returns
    -------
    PP_odd : array of shape (N,)
        Estimated nonlinear coefficients PP^(3), PP^(5), ...
    """
    PP_measured = np.asarray(PP_measured, dtype=float)
    N = PP_measured.size
    
    # Step 1: compute matrices
    W = w_matrix(N)            # shape (N,N)
    L_inv = build_inverse_A(N)  # shape (N,N)
    
    # Step 2: compute PP^(nQ)(I0) = W @ PP(I_p)
    PP_nQ = W @ PP_measured     # shape (N,)
    
    # Step 3: compute PP^(2r+1)*I0^r = L_inv @ PP^(nQ)(I0)
    PP_odd_times_I0r = L_inv @ PP_nQ  # shape (N,)
    
    """
    # Step 4: divide by I0^r to get PP^(2r+1)
    r = np.arange(1, N+1)
    PP_odd = PP_odd_times_I0r / (I0**r)
    """
    PP_odd = PP_odd_times_I0r
    
    return PP_odd, PP_nQ, W, L_inv

#%% Do the trasformation vectorially

def calculate_transformation_matrix(N):
    W = w_matrix(N)            # shape (N,N)
    #L_inv = Lambda_inverse(N)  # shape (N,N)
    L_inv = build_inverse_A(N)  # shape (N,N)
    
    total_mat = L_inv @ W;
    
    return total_mat
    
def stack_matrices(*matrices: np.ndarray) -> np.ndarray:
    """
    Stack a variable number of 2D numpy arrays (each of shape n×m)
    into a single 3D array of shape (N, n, m),
    where N is the number of input matrices.

    Works both for:
        stack_matrices(A, B, C)
    and
        stack_matrices([A, B, C])

    Parameters
    ----------
    *matrices : np.ndarray or list of np.ndarray
        Variable number of 2D numpy arrays of the same shape.

    Returns
    -------
    stacked : np.ndarray
        3D numpy array of shape (N, n, m)
    """
    # If user passed a list or tuple of arrays, unpack it
    if len(matrices) == 1 and isinstance(matrices[0], (list, tuple, np.ndarray)):
        # If it's a list/tuple of arrays, use that list
        if all(isinstance(x, np.ndarray) for x in matrices[0]):
            matrices = matrices[0]

    # Verify all inputs are numpy arrays
    matrices = [np.asarray(mat) for mat in matrices]

    # Check shapes
    shapes = [mat.shape for mat in matrices]
    if len(set(shapes)) != 1:
        raise ValueError(f"All input matrices must have the same shape, got {shapes}")

    # Stack along a new axis (shape = N x n x m)
    stacked = np.stack(matrices, axis=0).astype(np.float64)
    return stacked

def apply_matrix_to_stack(M, V, use_reshape=True):
    """
    Apply square matrix M (N,N) to every vector in V (N, nx, m) producing (N, nx, m).

    Parameters
    ----------
    M : array-like, shape (N, N)
        Square matrix.
    V : array-like, shape (N, nx, m)
        Stack of vectors; each vector is V[:, i, j].
    use_reshape : bool, optional
        If True, use reshape + 2D matmul (often fastest).
        If False, use np.einsum (clear syntax).

    Returns
    -------
    R : np.ndarray, shape (N, nx, m)
        Result of applying M to each vector.
    """
    M = np.asarray(M)
    V = np.asarray(V)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D array (N, N).")
    if V.ndim != 3 or V.shape[0] != M.shape[0]:
        raise ValueError("V must have shape (N, nx, m) with same N as M.")

    N = M.shape[0]
    # ensure float64 for numerical consistency and performance
    M = M.astype(np.float64, copy=False)
    V = V.astype(np.float64, copy=False)

    if use_reshape:
        # reshape to 2D, do BLAS matmul, then reshape back
        nx, m = V.shape[1], V.shape[2]
        V2 = V.reshape(N, nx * m)         # (N, nx*m)
        R2 = M @ V2                       # (N, nx*m)
        R = R2.reshape(N, nx, m)          # (N, nx, m)
    else:
        # einsum: 'ij,jkm->ikm' => R[i,k,m] = sum_j M[i,j]*V[j,k,m]
        R = np.einsum('ij,jkm->ikm', M, V)

    return R

