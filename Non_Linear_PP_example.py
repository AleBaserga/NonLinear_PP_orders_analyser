# -*- coding: utf-8 -*-
"""
This is an example script of usage of Non_Linear_PP_module

@author: Alessandro
"""

import numpy as np

# here is our non_linear module
import Non_Linear_PP_module as nlPP

# add other PP library
import sys
# change this path to match where the module PP_utils_module will be 
sys.path.append(r"C:\Users\aless\Documents\Python Scripts\Reader_PP")

import PP_utils_module as utilsPP


# calculates w and I_p with functions

print(nlPP.w(n=2, N=4, p=1))
print(nlPP.w(n=3, N=4, p=2))

I0 = 4
N = 3   # number of p points

Ip = nlPP.I_p_vector(I0, N)
print("I_p (p=1..N):", np.round(Ip, 3))

#%% calculates PP_nQ given N and PP_I_p

# Example PP(I_p) values
PP_I = [0.9, 0.7, 0.4, 0.1]
N = len(PP_I)

# Compute for a few n values
for n in range(1, N + 1):
    result = nlPP.PP_nQ(PP_I, n, N)
    print(f"PP^({n}Q)(I0) = {result:.5f}")

PP_vector = nlPP.PP_all_nQ(PP_I)
print(PP_vector)


#%% Gives relations between PP_nQ_from_PPodd and Inverse Matrix

# example PP^(2r+1) values for r=1..6 (so l=6)
PP_odd = [1.0, 0.8, 0.5, 1, 5, 6]   # PP^(3), PP^(5), PP^(7), ...
I0 = 0.5

w = nlPP.compute_PP_nQ_from_PPodd(PP_odd, I0)
print("PP^(nQ)(I0) for n=1..6:\n", w)

# If you want to see the inverse matrix (your displayed matrix):
A = nlPP.build_inverse_A(len(PP_odd))
print("First 6x6 block of A (rounded):\n", np.round(A[:6,:6],0))


#%% Define matrix w, the matrix to invert and the transformation between PP(Ip)

# Example usage
N = 3
I0 = 0.5
# measured PP(I_p) values (example numbers)
PP_measured = np.array([4.00, 3.00, 1.00])

PP_odd, PP_nQ, W, L_inv = nlPP.extract_PP_non_lin_orders(PP_measured)

print("w matrix (W):\n", np.round(W, 4))
print("\nLambda inverse matrix:\n", L_inv)
print("\nTotal inverse matrix:\n", L_inv @ W)
print("\nPP^(nQ)(I0):\n", np.round(PP_nQ, 6))
print("\nRecovered nonlinear coefficients PP^(3), PP^(5), ...:\n", np.round(PP_odd, 6))


#%% Do the trasformation vectorially

path = r"C:\Users\aless\OneDrive - Politecnico di Milano\PhD_backup\Experiments\NonLinear_PP\Data\AleMatteo Stratus Long\d251008\N3"
fileNames = ["\d25100801", "\d25100802", "\d25100803"]
list_maps = []

for fileName in fileNames:
    loadPath = path + fileName + "_average" + ".dat"
    
    data_s = utilsPP.load_dat(loadPath, asClass= False)
    t = data_s[0]
    wl = data_s[1]
    map_data = data_s[2]
    
    
    # Cut spectra
    wl_l = [600, 745]
    wl_cut, map_cut = utilsPP.cut_spectra(wl, map_data, wl_l)

    # bkg cleaning
    t_bkg = -900
    map_cut_2 = utilsPP.remove_bkg(t, map_cut, t_bkg)

    # plot maps
    fig, ax, c = utilsPP.plot_map(t, wl_cut, map_cut_2)
    ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="black")
    ax.set_title(f"{fileName}")
    
    # plot spectra
    delays_to_plot = [1000, 5000, 10000, 100000]
    fig, ax = utilsPP.plot_spectra(t, wl_cut, map_cut_2, delays_to_plot)
    ax.set_title(f"{fileName}")

    
    list_maps.append(map_cut_2)

PP_data_stack_2 = nlPP.stack_matrices(list_maps)

M_inv = nlPP.calculate_transformation_matrix(PP_data_stack_2.shape[0])

non_lin_maps = nlPP.apply_matrix_to_stack(M_inv, PP_data_stack_2)

for i in range(non_lin_maps.shape[0]):
    map_nl = non_lin_maps[i]
    
    fig, ax, c = utilsPP.plot_map(t, wl_cut, map_nl)
    ax.set_title(f"nl: {i}")
    
    # plot spectra
    delays_to_plot = [1000, 5000, 10000, 100000]
    fig, ax = utilsPP.plot_spectra(t, wl_cut, map_nl, delays_to_plot)
    ax.set_title(f"nl: {i}")

