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
from typing import Sequence, Optional, Tuple, List, Text
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append(r"C:\Users\aless\Documents\Python Scripts\Reader_PP")

import PP_utils_module as utilsPP

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

# Plotting functions

def compute_clims_auto_single(map_mat: np.ndarray, factor: float = 0.6) -> Tuple[float, float]:
    """
    Compute symmetric color limits from a single 2D matrix.
    Default: ±factor * max(abs(map_mat)).
    """
    a = np.nanmax(np.abs(map_mat))
    v = float(a) if np.isfinite(a) else 0.0
    vmax = v * factor
    vmin = -vmax
    return vmin, vmax

def compute_clims_auto_global(stack: np.ndarray, factor: float = 0.6) -> Tuple[float, float]:
    """
    Compute symmetric color limits across the whole stack array (N, wl, t).
    """
    a = np.nanmax(np.abs(stack))
    v = float(a) if np.isfinite(a) else 0.0
    vmax = v * factor
    vmin = -vmax
    return vmin, vmax

def plot_stack_maps(
    t: np.ndarray,
    wl: np.ndarray,
    stack: np.ndarray,
    rows: int = 2,
    cols: int = 4,
    cmap: str = "PuOr_r",
    clims: Text or Sequence[float] = "auto",
    clims_mode: str = "per_map",   # "per_map" or "global"
    titles: Optional[Sequence[str]] = None,
    figsize_per_subplot: Tuple[float,float] = (4.0, 3.0),
    show_colorbar: bool = True,
    colorbar_pad: float = 0.02,
    colorbar_size: float = "5%",
    tight: bool = True
) -> List[plt.Figure]:
    """
    Plot a stack of maps (N, n_wl, n_t) into one or more figures with row x col subplots per figure.

    Parameters
    ----------
    t : 1D array length n_t
        Horizontal axis (time/delay).
    wl : 1D array length n_wl
        Vertical axis (wavelength).
    stack : 3D array shape (N, n_wl, n_t)
        Stack of maps to plot.
    rows, cols : int
        Number of subplot rows and columns per figure.
    cmap : str
        Colormap for pcolormesh.
    clims : "auto" or [vmin, vmax]
        If "auto", clims computed either per-map or globally depending on clims_mode.
        If array-like of length 2, used as [vmin, vmax] for all maps.
    clims_mode : {"per_map", "global"}
        If "per_map": compute auto clims separately for each map.
        If "global": compute one clims across all maps and use same for all.
    titles : optional sequence of strings length N
        Titles for each subplot.
    figsize_per_subplot : (w, h)
        Size of each subplot; total figure size = (cols*w, rows*h).
    show_colorbar : bool
        Add a colorbar to the right of each subplot.
    colorbar_pad : float
        Padding between axes and colorbar.
    colorbar_size : str or float
        Size of colorbar (accepted by make_axes_locatable.append_axes).
    tight : bool
        Call tight_layout on figure at the end.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of created figure objects (one per page).
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be 3D with shape (N, n_wl, n_t).")
    N, n_wl, n_t = stack.shape

    # Titles
    if titles is None:
        titles = [f"map {i}" for i in range(N)]
    else:
        if len(titles) < N:
            # pad with default titles
            titles = list(titles) + [f"map {i}" for i in range(len(titles), N)]

    # Prepare clims
    if isinstance(clims, str) and clims.lower() == "auto":
        if clims_mode == "global":
            vmin_global, vmax_global = compute_clims_auto_global(stack, factor=0.6)
            clims_all = [(vmin_global, vmax_global)] * N
        elif clims_mode == "per_map":
            clims_all = [compute_clims_auto_single(stack[i], factor=0.6) for i in range(N)]
        else:
            raise ValueError("clims_mode must be 'per_map' or 'global'")
    else:
        clims_arr = np.asarray(clims, dtype=float).flatten()
        if clims_arr.size != 2:
            raise ValueError("clims must be 'auto' or a sequence of two numbers [vmin, vmax].")
        vmin_use, vmax_use = np.sort(clims_arr)
        clims_all = [(vmin_use, vmax_use)] * N

    figs = []
    per_fig = rows * cols
    n_pages = (N + per_fig - 1) // per_fig

    # compute figure size
    fig_w = cols * figsize_per_subplot[0]
    fig_h = rows * figsize_per_subplot[1]

    map_index = 0
    for page in range(n_pages):
        fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        # flatten axes list for easy indexing
        axs_flat = np.array(axs).reshape(-1)
        # For subplots that will remain empty, we will turn off axis later
        for slot in range(per_fig):
            if map_index >= N:
                break
            ax = axs_flat[slot]
            m = stack[map_index]
            vmin, vmax = clims_all[map_index]

            # pcolormesh: t (n_t), wl (n_wl) must correspond map shape (n_wl, n_t)
            pcm = ax.pcolormesh(t, wl, m, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(titles[map_index])
            ax.set_xlabel("Delay (fs)")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_xlim([np.min(t), np.max(t)])
            ax.set_ylim([np.min(wl), np.max(wl)])

            if show_colorbar:
                # place a colorbar to the right of this axis without overlapping the next subplot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
                fig.colorbar(pcm, cax=cax)
            map_index += 1

        # turn off remaining axes (if any)
        for i_empty in range(map_index % per_fig, per_fig):
            # if map_index % per_fig == 0 this loop will turn off all axes on last page only when needed
            ax_empty = axs_flat[i_empty]
            ax_empty.axis("off")

        if tight:
            plt.tight_layout()

        figs.append(fig)

    return figs


    
    