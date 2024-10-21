import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import os
from matplotlib.gridspec import GridSpec

#file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-18_10-00-14_lx06/XGB/XGB_55.pkl'  # best model
#file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-18_15-30-30_lx06/XGB/XGB_55.pkl'
#file_path  = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-18_16-45-51_lx06/XGB/XGB_42.pkl'  # Found the best model at epoch [42] with validation loss = 0.6295
#file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-21_11-13-51_lx06/XGB/XGB_34.pkl'  # Found the best model at epoch [34] with validation loss = 0.6295
file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-21_15-52-53_lx06/XGB/XGB_44.pkl'  # Found the best model at epoch [44] with validation loss = 0.6308

with open(file_path, 'rb') as file:
    data = pickle.load(file)

if 'model' in data:
    model = data['model']
    
    X_test_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_mc_events.root'
    single_muon_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_data_events.root'
    
    with uproot.open(X_test_path) as file:
        tree = file["tree"]
    print(f"Number of entries in the MC tree: {tree.num_entries}")
    
    branches = ['decayMode_1', 'decayMode_2', 'jpt_pt_1', 'jpt_pt_2', 'pt_1', 'pt_2', 'eta_1', 'eta_2', 'charge_1', 'charge_2', 'n_jets', 'n_prebjets', 'wt_sf']
    df = tree.arrays(branches, library="pd")
    
    feature_names = ['decayMode_1', 'decayMode_2', 'jpt_pt_1', 'jpt_pt_2', 'pt_1', 'pt_2', 'eta_1', 'eta_2', 'charge_1', 'charge_2', 'n_jets', 'n_prebjets',]
    original_weights = df['wt_sf']
    X_test = df[feature_names]
    
    with uproot.open(single_muon_path) as file:
        tree_single_muon = file["tree"]
    print(f"Number of entries in the data tree: {tree_single_muon.num_entries}")
    df_single_muon = tree_single_muon.arrays(feature_names, library="pd")
    
    def process_batch(batch_df, batch_index):
        dtest = xgb.DMatrix(batch_df[feature_names], feature_names=feature_names)
        probabilities = model.predict(dtest)
        probabilities = 1 - probabilities
        new_weights = probabilities / (1 - probabilities)
        print(f"Processed batch {batch_index + 1}")
        return new_weights

    batch_size = 10000
    num_batches = len(df) // batch_size + 1

    new_weights_list = []
    for i in range(num_batches):
        batch_df = df.iloc[i * batch_size:(i + 1) * batch_size]
        new_weights = process_batch(batch_df, i)
        new_weights_list.append(new_weights)

    new_weights = np.concatenate(new_weights_list)

    # Multiply original weights with new weights
    combined_weights = original_weights * new_weights

    # Normalize combined weights
    combined_weights /= np.sum(combined_weights) / np.sum(original_weights)
    print(combined_weights)

    hep.style.use("CMS")

    # Define the output directory relative to the script directory
    output_dir = "ff_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Define plotting ranges and number of bins for each feature
    plotting_ranges = {  # 'charge_1', 'charge_2'
        'decayMode_1': (0, 12),
        'decayMode_2': (0, 12),
        'jpt_pt_1': (0, 15),
        'jpt_pt_2': (0, 15),
        'pt_1': (0, 200),
        'pt_2': (0, 200),
        'eta_1': (-3, 3),
        'eta_2': (-3, 3),
        'charge_1': (-2, 2),
        'charge_2': (-2, 2),
        'n_jets': (0, 10),
        'n_prebjets': (0, 10),
    }

    num_bins = {
        'decayMode_1': 4,  # Discrete bins: 0, 1, 10, 11
        'decayMode_2': 4,  # Discrete bins: 0, 1, 10, 11
        'jpt_pt_1': 15,
        'jpt_pt_2': 15,
        'pt_1': 50,
        'pt_2': 50,
        'eta_1': 30,
        'eta_2': 30,
        'charge_1': 4,
        'charge_2': 4,
        'n_jets': 10,
        'n_prebjets': 10,
    }

    # Define bins for discrete features
    discrete_bins = {
        'decayMode_1': np.array([0, 1, 2, 10, 11, 12]),
        'decayMode_2': np.array([0, 1, 2, 10, 11, 12])
    }

    for i, branch in enumerate(feature_names):
        if branch in discrete_bins:
            bin_edges = discrete_bins[branch]
        else:
            bin_edges = np.linspace(plotting_ranges[branch][0], plotting_ranges[branch][1], num_bins[branch] + 1)

        original_hist_counts = np.zeros(len(bin_edges) - 1)
        combined_hist_counts = np.zeros(len(bin_edges) - 1)
        single_muon_hist_counts = np.zeros(len(bin_edges) - 1)

        for j in range(num_batches):
            batch_df = df.iloc[j * batch_size:(j + 1) * batch_size]
            batch_original_weights = original_weights[j * batch_size:(j + 1) * batch_size]
            batch_combined_weights = combined_weights[j * batch_size:(j + 1) * batch_size]

            original_counts, _ = np.histogram(batch_df[branch], bins=bin_edges, weights=batch_original_weights)
            combined_counts, _ = np.histogram(batch_df[branch], bins=bin_edges, weights=batch_combined_weights)
            single_muon_counts, _ = np.histogram(df_single_muon[branch].iloc[j * batch_size:(j + 1) * batch_size], bins=bin_edges)

            original_hist_counts += original_counts
            combined_hist_counts += combined_counts
            single_muon_hist_counts += single_muon_counts

            print(f"Updated histogram for batch {j + 1} for feature {branch}")

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the ratio for original weights
        ratio_original = np.divide(single_muon_hist_counts, original_hist_counts, out=np.zeros_like(single_muon_hist_counts), where=original_hist_counts != 0)

        # Calculate the ratio for combined weights
        ratio_combined = np.divide(single_muon_hist_counts, combined_hist_counts, out=np.zeros_like(single_muon_hist_counts), where=combined_hist_counts != 0)

        # Calculate chi-squared for original weights
        chi_squared_original = np.sum(((ratio_original - 1) ** 2) / 1)  # TODO: Propagate uncerts, square root of N!!!!, for MC do sum of weights squared
        ndf_original = len(bin_centers) - 1
        reduced_chi_squared_original = chi_squared_original / ndf_original

        # Calculate chi-squared for combined weights
        chi_squared_combined = np.sum(((ratio_combined - 1) ** 2) / 1)
        ndf_combined = len(bin_centers) - 1
        reduced_chi_squared_combined = chi_squared_combined / ndf_combined

        fig = plt.figure(figsize=(10, 15))
        gs = GridSpec(4, 1, height_ratios=[3, 1, 3, 1])

        # Plot the original weights histogram
        ax0 = fig.add_subplot(gs[0])
        ax0.hist(bin_edges[:-1], bin_edges, weights=original_hist_counts, alpha=0.7, label='Original', histtype='step', linewidth=2)
        ax0.plot(bin_centers, single_muon_hist_counts, 'o', label='Iso Data Points')
        ax0.set_title(f'{branch} (Original Weights)')
        ax0.set_xlabel(branch)
        ax0.set_ylabel('Counts')
        ax0.set_xlim(plotting_ranges[branch])
        ax0.legend()

        # Plot the ratio for original weights
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax1.plot(bin_centers, ratio_original, 'o')
        ax1.set_ylabel('Ratio')
        ax1.set_ylim(0., 1.5)
        ax1.axhline(1, color='r', linestyle='--')
        ax1.text(0.95, 0.95, f'Reduced Chi-squared: {reduced_chi_squared_original:.2f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

        # Plot the combined weights histogram
        ax2 = fig.add_subplot(gs[2])
        ax2.hist(bin_edges[:-1], bin_edges, weights=combined_hist_counts, alpha=0.7, label='Combined Weights', histtype='step', linewidth=2)
        ax2.plot(bin_centers, single_muon_hist_counts, 'o', label='Iso Data Points')
        ax2.set_title(f'{branch} (Combined Weights)')
        ax2.set_xlabel(branch)
        ax2.set_ylabel('Counts')
        ax2.set_xlim(plotting_ranges[branch])
        ax2.legend()

        # Plot the ratio for combined weights
        ax3 = fig.add_subplot(gs[3], sharex=ax2)
        ax3.plot(bin_centers, ratio_combined, 'o')
        ax3.set_ylabel('Ratio')
        ax3.set_ylim(0, 1.5)
        ax3.axhline(1, color='r', linestyle='--')
        ax3.text(0.95, 0.95, f'Reduced Chi-squared: {reduced_chi_squared_combined:.2f}', transform=ax3.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{branch}_with_ratio_new_vars.pdf'))
        plt.close(fig)
        print(f'{branch} plot with ratio saved as a PDF file.')

else:
    print("No model found in the .pkl file.")