# -*- coding: utf-8 -*-
"""
This is an example script of usage of Non_Linear_PP_module

@author: Alessandro
"""

import numpy as np
import matplotlib.pyplot as plt

# here is our non_linear module
import Non_Linear_PP_module as nlPP

# add other PP library
import sys
# change this path to match where the module PP_utils_module will be 
sys.path.append(r"C:\Users\aless\Documents\Python Scripts\Reader_PP")

import PP_utils_module as utilsPP

# Close all previous plots
plt.close("all")

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
    ax.set_title(f"non linear order: {2*(i+1) +1}")
    
    # plot spectra
    delays_to_plot = [1000, 5000, 10000, 100000]
    fig, ax = utilsPP.plot_spectra(t, wl_cut, map_nl, delays_to_plot)
    ax.set_title(f"non linear order: {2*(i+1) +1}")

#%% Let's do it with functions

def load_n_clean_maps(path_folder, file_name_vector, wl_l, t_bkg):
    
    for i in range(len(file_name_vector)):
        
        base_file = file_name_vector[i]
        
        # Load and stack
        t, wl, stacked, files_used = utilsPP.load_and_stack_related_maps(path_folder, base_file)

        # Cut
        wl_cut, stacked_cut = utilsPP.cut_spectra_stacked(wl, stacked, wl_l)

        # find the spikes
        spike_mask, detected_indices, wl_idx = utilsPP.detect_spikes_stack_at_wl(stacked_cut, wl_cut, 530.0, window=11, thresh=15.0, min_distance=1)
        
        """
        # plot overlay with spikes marked
        fig2, ax2 = utilsPP.plot_spike_mask_overlay(t, wl_cut, stacked_cut, spike_mask, wl_choice=630)
        """
        
        # clean the spikes
        cleaned = utilsPP.replace_spikes_stack_with_median_spectrum(stacked_cut, spike_mask)        
        
        map_cut = utilsPP.mean_stack(cleaned)
        
        #bkg
        map_cut = utilsPP.remove_bkg(t, map_cut, t_bkg)
        
        #smoothing
        map_cut = utilsPP.smooth_along_axis(map_cut, axis=0, method="gaussian", window=3, sigma=2)
        
        if i == 0:
            map_cut_mat = np.zeros((len(file_name_vector), *map_cut.shape), dtype=np.float64)
        
        #append the results
        map_cut_mat[i, :] = map_cut
        
    return t, wl_cut, map_cut_mat

# define a list of your files and powers
path_folder = r"C:\Users\aless\OneDrive - Politecnico di Milano\PhD_backup\Experiments\NonLinear_PP\Data\AleMatteo Stratus Long\d251009\PM6"
#file_name_vector = ["d25100909", "d25100913", "d25100913", "d25100913","d25100913"]
file_seed = "d251009"
#file_nums = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24]
#powers = [2.35, 1.55, 4.0, 10, 16, 12, 4, 13.657, 8.0, 2.343, 1.072, 14.928, 6.0, 2.0, 0.4]

file_nums = [13, 14, 15]
powers = [16, 12, 4]
#file_nums = [13, 16, 17, 18]
#powers = [16, 13.657, 8, 2.343]
file_nums = [13, 20, 14, 17, 15, 19]
powers = [16, 14.928, 12, 8, 4, 1.072]
#file_nums = [17, 22, 23]
#powers = [8, 6, 2]

file_name_vector = utilsPP.generate_string_list(file_seed, file_nums)

# call the loading function
t_bkg = -1000
wl_l = [500, 740]

# load all
t, wl, stacked_maps = load_n_clean_maps(path_folder, file_name_vector, wl_l, t_bkg)

list_maps = []
for i in range(stacked_maps.shape[0]):
    map_cut = stacked_maps[i]

    # plot maps
    """
    fig, ax, c = utilsPP.plot_map(t, wl, map_cut)
    ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="black")
    ax.set_title(f"{file_name_vector[i]}")
    
    # plot spectra
    delays_to_plot = [1000, 5000, 10000, 100000]
    fig, ax = utilsPP.plot_spectra(t, wl, map_cut, delays_to_plot)
    ax.set_title(f"{file_name_vector[i]}")
    """
    
    list_maps.append(map_cut)

PP_data_stack_2 = nlPP.stack_matrices(list_maps)

M_inv = nlPP.calculate_transformation_matrix(PP_data_stack_2.shape[0])

non_lin_maps = nlPP.apply_matrix_to_stack(M_inv, PP_data_stack_2)

for i in range(non_lin_maps.shape[0]):
    map_nl = non_lin_maps[i]
    
    map_nl = utilsPP.smooth_along_axis(map_nl, axis=0, method="uniform", window=3)
    
    # plot map
    fig, ax, c = utilsPP.plot_map(t, wl, map_nl)
    ax.set_title(f"non linear order: {2*(i+1) +1}")
    
    # plot spectra
    delays_to_plot = [1000, 5000, 10000, 100000]
    fig, ax = utilsPP.plot_spectra(t, wl, map_nl, delays_to_plot)
    ax.set_title(f"non linear order: {2*(i+1) +1}")
    
    # plot dynamics
    wls_to_plot = [580, 635]
    fig, ax = utilsPP.plot_dynamics(t, wl, map_nl, wls_to_plot)
    ax.set_title(f"non linear order: {2*(i+1) +1}")
    
#%% plot nl orders

def smooth_moving_average(y, window_size=5):
    """
    Smooth a 1D array using a centered moving average.
    """
    if window_size < 2:
        return y  # no smoothing

    kernel = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, kernel, mode='same')
    return y_smooth

        
def plot_dynamics_nl_orders(
        t: np.ndarray,
        wl: np.ndarray,
        stack: np.ndarray,
        wls_to_plot,
        nl_order_index,
        normalize = True,
        i_to_norm = -1):
    
    """
    TODO: fix this description
    """
    
    figs = []
    axs = []
    
    cmap_name = "plasma"
    colors = utilsPP.create_diverging_colormap(len(nl_order_index), cmap_name)
    
    for wl_to_plot in wls_to_plot:
        for i in range(len(nl_order_index)):
            nl_index = nl_order_index[i]
            if i == 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 3))
                figs.append(fig)
                axs.append(ax)
            
            # --- Extract dynamics ---
            dynamic, i_taken = utilsPP.extract_dyns(wl, stack[nl_index], wl_to_plot)
            wl_c = wl[i_taken]
            
            dynamic = np.squeeze(dynamic)
            
            dynamic = smooth_moving_average(dynamic, 1)
            
            if normalize:
                if i_to_norm == -1:
                    i_max = np.argmax(np.abs(dynamic))
                else:
                    #i_max = i_to_norm
                    i_max = np.arange(i_to_norm - 5, i_to_norm + 5, 1)
                    
                dynamic = dynamic / (np.mean(dynamic[i_max]))
                        
            ax.plot(t, dynamic,
                    label=f'nl order: {2*(nl_index + 1) +1} ',
                    color=colors[i])
    
        # --- Labels and formatting ---
        ax.set_xlabel("Delay (fs)")
        
        if normalize:
            ax.set_ylabel("ΔT/T (norm.)")
        else:
            ax.set_ylabel("ΔT/T (%)")
            
        ax.set_xlim([np.min(t), np.max(t)])
        ax.set_title(f'{wl_c[0]:.2f} nm')
        ax.legend(fontsize='small')

        plt.tight_layout()
        plt.show()
    
    return figs, axs

            
def load_low_fluence(path_folder, file_name_vector, wl_l, t_bkg):
    
    for i in range(len(file_name_vector)):
        
        base_file = file_name_vector[i]
        
        # Load and stack
        t, wl, stacked, files_used = utilsPP.load_and_stack_related_maps(path_folder, base_file)

        # Cut
        wl_cut, stacked_cut = utilsPP.cut_spectra_stacked(wl, stacked, wl_l)

        # find the spikes
        spike_mask, detected_indices, wl_idx = utilsPP.detect_spikes_stack_at_wl(stacked_cut, wl_cut, 530.0, window=11, thresh=15.0, min_distance=1)
        
        """
        # plot overlay with spikes marked
        fig2, ax2 = utilsPP.plot_spike_mask_overlay(t, wl_cut, stacked_cut, spike_mask, wl_choice=630)
        """
        
        # clean the spikes
        cleaned = utilsPP.replace_spikes_stack_with_median_spectrum(stacked_cut, spike_mask)        
        
        map_cut = utilsPP.mean_stack(cleaned)
        
        #bkg
        map_cut = utilsPP.remove_bkg(t, map_cut, t_bkg)
        
        #smoothing
        map_cut = utilsPP.smooth_along_axis(map_cut, axis=0, method="gaussian", window=3, sigma=2)
        
        if i == 0:
            map_cut_mat = np.zeros((len(file_name_vector), *map_cut.shape), dtype=np.float64)
        
        #append the results
        map_cut_mat[i, :] = map_cut
        
    map_cut_mat = np.squeeze(map_cut_mat)
    return t, wl_cut, map_cut_mat


def plot_dynamics_comparison(
        t: np.ndarray,
        wl: np.ndarray,
        stack: np.ndarray,
        map_lf: np.ndarray,
        wls_to_plot,
        normalize = True,
        i_to_norm = -1):
    
    """
    TODO: fix this description
    """
    
    figs = []
    axs = []
    
    cmap_name = "inferno"
    colors = utilsPP.create_diverging_colormap(2, cmap_name)
    
    for wl_to_plot in wls_to_plot:
        nl_index = 0
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        figs.append(fig)
        axs.append(ax)
    
        # --- Extract dynamics ---
        dynamic_nl, i_taken = utilsPP.extract_dyns(wl, stack[nl_index], wl_to_plot)
        wl_c = wl[i_taken]
        
        dynamic = np.squeeze(dynamic_nl)
        
        dynamic = smooth_moving_average(dynamic, 1)
        
        if normalize:
            if i_to_norm == -1:
                i_max = np.argmax(np.abs(dynamic))
            else:
                #i_max = i_to_norm
                i_max = np.arange(i_to_norm - 5, i_to_norm + 5, 1)
                
            dynamic = dynamic / (np.mean(dynamic[i_max]))
                    
        ax.plot(t, dynamic,
                label=f'nl order: {2*(nl_index + 1) +1} ',
                color=colors[0])
        
        # --- Extract dynamics ---
        dynamic_lf, i_taken = utilsPP.extract_dyns(wl, map_lf, wl_to_plot)
        wl_c = wl[i_taken]
        
        dynamic = np.squeeze(dynamic_lf)
        
        dynamic = smooth_moving_average(dynamic, 1)
        
        if normalize:
            if i_to_norm == -1:
                i_max = np.argmax(np.abs(dynamic))
            else:
                #i_max = i_to_norm
                i_max = np.arange(i_to_norm - 5, i_to_norm + 5, 1)
                
            dynamic = dynamic / (np.mean(dynamic[i_max]))
                    
        ax.plot(t, dynamic,
                label='low fluece',
                color=colors[1])
    
        # --- Labels and formatting ---
        ax.set_xlabel("Delay (fs)")
        
        if normalize:
            ax.set_ylabel("ΔT/T (norm.)")
        else:
            ax.set_ylabel("ΔT/T (%)")
            
        ax.set_xlim([np.min(t), np.max(t)])
        ax.set_title(f'{wl_c[0]:.2f} nm')
        ax.legend(fontsize='small')

        plt.tight_layout()
        plt.show()
    
    return figs, axs
    
wls_to_plot = [635]
nl_order_index = [0, 1]
plot_dynamics_nl_orders(t, wl, non_lin_maps, wls_to_plot, nl_order_index, normalize = True, i_to_norm=60)

file_name_vector = ["d25100924.dat"]
t, wl, map_lf = load_low_fluence(path_folder, file_name_vector, wl_l, t_bkg)

plot_dynamics_comparison(t, wl, non_lin_maps, map_lf, wls_to_plot, normalize = True, i_to_norm=60)