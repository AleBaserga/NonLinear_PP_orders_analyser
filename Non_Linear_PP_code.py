# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 14:33:32 2025

@author: matte
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

print(w(n=2, N=4, p=1))
print(w(n=3, N=4, p=2))

I0 = 4
N = 3   # number of p points

Ip = I_p_vector(I0, N)
print("I_p (p=1..N):", np.round(Ip, 3))

#%% calculates PP_nQ given N and PP_I_p

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

# Example PP(I_p) values
PP_I = [0.9, 0.7, 0.4, 0.1]
N = len(PP_I)

# Compute for a few n values
for n in range(1, N + 1):
    result = PP_nQ(PP_I, n, N)
    print(f"PP^({n}Q)(I0) = {result:.5f}")

def PP_all_nQ(I_values):
    N = len(I_values)
    I_values = np.asarray(I_values, dtype=float)
    W = np.zeros((N, N))

    for n in range(1, N + 1):
        for p in range(1, N + 1):
            W[n - 1, p - 1] = w(n, N, p)

    return W @ I_values  # matrix multiplication: gives all PP^(nQ)(I0)

PP_vector = PP_all_nQ(PP_I)
print(PP_vector)


#%% Computes PP_nQ but in another way

def PP_nQ(I0, n, N, PP_values):
    """
    Compute PP^(nQ)(I0) = sum_{r=n}^{N} [ C(2r, r-n) * PP^(2r+1) * I0^r ]
    
    Parameters
    ----------
    I0 : float
        Input intensity (I₀)
    n : int
        Lower summation limit
    N : int
        Upper summation limit
    PP_values : dict or list
        A dictionary or list containing PP^(2r+1) values.
        For example:
            {3: value_for_2r+1=3, 5: value_for_2r+1=5, ...}
        or a list such that PP_values[r] gives PP^(2r+1)
    
    Returns
    -------
    float
        The computed PP^(nQ)(I0)
    """
    total = 0.0
    for r in range(n, N + 1):
        # get PP^(2r+1)
        if isinstance(PP_values, dict):
            PP_term = PP_values.get(2 * r + 1, 0)
        else:
            PP_term = PP_values[r] if r < len(PP_values) else 0
        
        term = comb(2 * r, r - n) * PP_term * (I0 ** r)
        total += term
    return total

# Example PP^(2r+1) values for r = 1..5
PP_vals = {
    3: 1.0,
    5: 0.8,
    7: 0.5,
    9: 0.3,
    11: 0.2
}

I0 = 0.5
n = 2
N = 5

result = PP_nQ(I0, n, N, PP_vals)
print(f"PP^({n}Q)({I0}) = {result:.5f}")

#%% Gives relations between PP_nQ_from_PPodd and Inverse Matrix

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

# example PP^(2r+1) values for r=1..6 (so l=6)
PP_odd = [1.0, 0.8, 0.5, 1, 5, 6]   # PP^(3), PP^(5), PP^(7), ...
I0 = 0.5

w = compute_PP_nQ_from_PPodd(PP_odd, I0)
print("PP^(nQ)(I0) for n=1..6:\n", w)

# If you want to see the inverse matrix (your displayed matrix):
A = build_inverse_A(len(PP_odd))
print("First 6x6 block of A (rounded):\n", np.round(A[:6,:6],0))


#%% Define matrix w, the matrix to invert and the transformation between PP(Ip)

def w_matrix(N):
    """Compute w_p^(nQ) for all n,p (both 1..N)."""
    n = np.arange(1, N+1)[:, np.newaxis]  # shape (N,1)
    p = np.arange(1, N+1)[np.newaxis, :]  # shape (1,N)
    delta_p1 = (p == 1).astype(float)
    delta_nN = (n == N).astype(float)
    w = (1/(2*N)) * (2 - delta_p1) / (1 + delta_nN) * np.cos(n * (p-1) * 2*np.pi / (2*N))
    return w

def Lambda_inverse(N):
    """Build Λ^{-1} matrix using binomial coefficients (upper triangular)."""
    L_inv = np.zeros((N, N))
    for r in range(1, N+1):
        for n in range(r, N+1):
            L_inv[r-1, n-1] = comb(2*n, n-r)  # note: n ≥ r
    return L_inv

def extract_PP_non_lin_orders(PP_measured, I0):
    """
    Reconstruct first N nonlinear coefficients PP^(2r+1) from measured PP(I_p).
    
    Parameters
    ----------
    PP_measured : array-like of shape (N,)
        Measured PP(I_p) datasets (one per intensity)
    I0 : float
        Reference intensity I0
        
    Returns
    -------
    PP_odd : array of shape (N,)
        Estimated nonlinear coefficients PP^(3), PP^(5), ...
    """
    PP_measured = np.asarray(PP_measured, dtype=float)
    N = PP_measured.size
    
    # Step 1: compute matrices
    W = w_matrix(N)            # shape (N,N)
    #L_inv = Lambda_inverse(N)  # shape (N,N)
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

# Example usage
N = 3
I0 = 0.5
# measured PP(I_p) values (example numbers)
PP_measured = np.array([4.00, 3.00, 1.00])

PP_odd, PP_nQ, W, L_inv = extract_PP_non_lin_orders(PP_measured, I0)

print("w matrix (W):\n", np.round(W, 4))
print("\nLambda inverse matrix:\n", L_inv)
print("\nTotal inverse matrix:\n", L_inv @ W)
print("\nPP^(nQ)(I0):\n", np.round(PP_nQ, 6))
print("\nRecovered nonlinear coefficients PP^(3), PP^(5), ...:\n", np.round(PP_odd, 6))


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

    Parameters
    ----------
    *matrices : np.ndarray
        Variable number of 2D numpy arrays of the same shape.

    Returns
    -------
    stacked : np.ndarray
        3D numpy array of shape (N, n, m)
    """
    # Check that all inputs have the same shape
    shapes = [mat.shape for mat in matrices]
    if len(set(shapes)) != 1:
        raise ValueError(f"All input matrices must have the same shape, got {shapes}")

    # Stack along a new first axis
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

