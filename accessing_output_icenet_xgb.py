# source /vols/cms/ia2318/MLTools/Evaluation/python/myenv/bin/activate

import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
import os
from scipy.stats import ks_2samp
from tqdm import tqdm

# File path to the best model
file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-15_11-08-01_lx06/XGB/XGB_44.pkl'  # best model - latest training (val_loss: 0.6226)

# Load the model
with open(file_path, 'rb') as file:
    data = pickle.load(file)

if 'model' in data:
    model = data['model']
    
    # Paths to the test and single muon data
    X_test_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DY_all_events.root'
    single_muon_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/SingleMuon_all_events.root'
    
    # Load the test data
    with uproot.open(X_test_path) as file:
        tree = file["tree"]
    print(f"Number of entries in the DY tree: {tree.num_entries}")
    
    branches = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'wt_sf']
    df = tree.arrays(branches, library="pd")
    
    feature_names = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj']
    latex_feature_names = {
        'Z_mass': r'$Z \, \text{mass}$',
        'Z_pt': r'$Z \, p_T$',
        'n_jets': r'$\text{Number of jets}$',
        'n_deepbjets': r'$\text{Number of deep b-jets}$',
        'mjj': r'$m_{jj}$'
    }
    original_weights = df['wt_sf']
    X_test = df[feature_names]
    
    # Load the single muon data
    with uproot.open(single_muon_path) as file:
        tree_single_muon = file["tree"]
    print(f"Number of entries in the Single Muon tree: {tree_single_muon.num_entries}")
    df_single_muon = tree_single_muon.arrays(feature_names, library="pd")
    
    def process_batch(batch_df, batch_index):
        dtest = xgb.DMatrix(batch_df[feature_names], feature_names=feature_names)
        probabilities = model.predict(dtest)
        probabilities = 1 - probabilities
        new_weights = probabilities / (1 - probabilities)
        return new_weights

    batch_size = 10000
    num_batches = len(df) // batch_size + 1

    new_weights_list = []
    for i in tqdm(range(num_batches), desc="Processing batches"):
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
    output_dir = "zmm_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Define plotting ranges and number of bins for each feature
    plotting_ranges = {
        'Z_mass': (0, 200),
        'Z_pt': (0, 200),
        'n_jets': (0, 10),
        'n_deepbjets': (0, 2),
        'mjj': (0, 200),
    }

    num_bins = {
        'Z_mass': 60,
        'Z_pt': 40,
        'n_jets': 10,
        'n_deepbjets': 2,
        'mjj': 20,
    }

    # Define bins for discrete features (if any)
    discrete_bins = {
        'n_jets': np.arange(0, 11, 1),
        'n_deepbjets': np.arange(0, 3, 1)
    }

    for i, branch in enumerate(feature_names):
        # Calculate bin edges using the entire dataset
        if (branch in discrete_bins):
            bin_edges = discrete_bins[branch]
        else:
            bin_edges = np.linspace(plotting_ranges[branch][0], plotting_ranges[branch][1], num_bins[branch] + 1)

        original_hist_counts = np.zeros(len(bin_edges) - 1)
        combined_hist_counts = np.zeros(len(bin_edges) - 1)
        single_muon_hist_counts = np.zeros(len(bin_edges) - 1)

        original_hist_uncerts = np.zeros(len(bin_edges) - 1)
        combined_hist_uncerts = np.zeros(len(bin_edges) - 1)
        single_muon_hist_uncerts = np.zeros(len(bin_edges) - 1)

        for j in tqdm(range(num_batches), desc=f"Updating histograms for {branch}"):
            batch_df = df.iloc[j * batch_size:(j + 1) * batch_size]
            batch_original_weights = original_weights[j * batch_size:(j + 1) * batch_size]
            batch_combined_weights = combined_weights[j * batch_size:(j + 1) * batch_size]

            original_counts, _ = np.histogram(batch_df[branch], bins=bin_edges, weights=batch_original_weights)
            combined_counts, _ = np.histogram(batch_df[branch], bins=bin_edges, weights=batch_combined_weights)
            single_muon_counts, _ = np.histogram(df_single_muon[branch].iloc[j * batch_size:(j + 1) * batch_size], bins=bin_edges)

            original_hist_counts += original_counts
            combined_hist_counts += combined_counts
            single_muon_hist_counts += single_muon_counts

            original_hist_uncerts += np.sqrt(np.sum(batch_original_weights * batch_original_weights))
            combined_hist_uncerts += np.sqrt(np.sum(batch_combined_weights * batch_combined_weights))
            single_muon_hist_uncerts += np.sqrt(np.sum(single_muon_counts))

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the ratio for original weights
        ratio_original = np.divide(single_muon_hist_counts, original_hist_counts, out=np.zeros_like(single_muon_hist_counts), where=original_hist_counts != 0)

        # Calculate the ratio for combined weights
        ratio_combined = np.divide(single_muon_hist_counts, combined_hist_counts, out=np.zeros_like(single_muon_hist_counts), where=combined_hist_counts != 0)

        # Calculate statistical uncertainties
        original_uncert = original_hist_uncerts
        combined_uncert = combined_hist_uncerts
        single_muon_uncert = single_muon_hist_uncerts

        # Calculate chi-squared for original weights
        chi_squared_original = np.sum((single_muon_hist_counts - original_hist_counts) ** 2 / (original_uncert ** 2 + single_muon_uncert ** 2))
        print(f"Chi-squared for original weights for feature {branch}: {chi_squared_original}")
        print(f"Reduced chi-squared for original weights for feature {branch}: {chi_squared_original / (num_bins[branch] - 1)}")

        # Calculate chi-squared for combined weights
        chi_squared_combined = np.sum((single_muon_hist_counts - combined_hist_counts) ** 2 / (combined_uncert ** 2 + single_muon_uncert ** 2))
        print(f"Chi-squared for combined weights for feature {branch}: {chi_squared_combined}")
        print(f"Reduced chi-squared for combined weights for feature {branch}: {chi_squared_combined / (num_bins[branch] - 1)}")

        # Perform K-S test for original weights
        ks_stat_original, ks_pvalue_original = ks_2samp(single_muon_hist_counts, original_hist_counts)
        print(f"K-S test for original weights for feature {branch}: statistic={ks_stat_original}, p-value={ks_pvalue_original}")

        # Perform K-S test for combined weights
        ks_stat_combined, ks_pvalue_combined = ks_2samp(single_muon_hist_counts, combined_hist_counts)
        print(f"K-S test for combined weights for feature {branch}: statistic={ks_stat_combined}, p-value={ks_pvalue_combined}")

        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 2, height_ratios=[3, 1])

        # Plot the original weights histogram
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(bin_edges[:-1], bin_edges, weights=original_hist_counts, alpha=0.7, label='Original weights', histtype='step', linewidth=2)
        ax0.errorbar(bin_centers, single_muon_hist_counts, yerr=single_muon_uncert, fmt='o', label='SingleMuon Data')
        ax0.set_title(f'{latex_feature_names[branch]} (Original)')
        ax0.set_xlabel(latex_feature_names[branch])
        ax0.set_ylabel('Counts')
        ax0.set_xlim(plotting_ranges[branch])
        ax0.legend()

        # Plot the ratio for original weights
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.errorbar(bin_centers, ratio_original, yerr=single_muon_uncert / original_hist_counts, fmt='o')
        ax1.set_ylabel('Ratio')
        ax1.set_ylim(0., 1.5)
        ax1.axhline(1, color='r', linestyle='--')

        # Plot the combined weights histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(bin_edges[:-1], bin_edges, weights=combined_hist_counts, alpha=0.7, label='New weights', histtype='step', linewidth=2)
        ax2.errorbar(bin_centers, single_muon_hist_counts, yerr=single_muon_uncert, fmt='o', label='SingleMuon Data')
        ax2.set_title(f'{latex_feature_names[branch]} (Reweighting)')
        ax2.set_xlabel(latex_feature_names[branch])
        ax2.set_ylabel('Counts')
        ax2.set_xlim(plotting_ranges[branch])
        ax2.legend()

        # Plot the ratio for combined weights
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax3.errorbar(bin_centers, ratio_combined, yerr=single_muon_uncert / combined_hist_counts, fmt='o')
        ax3.set_ylabel('Ratio')
        ax3.set_ylim(0, 1.5)
        ax3.axhline(1, color='r', linestyle='--')

        plt.savefig(os.path.join(output_dir, f'{branch}_with_ratio.pdf'))
        plt.close(fig)
        print(f'{branch} plot with ratio saved as a PDF file.')

else:
    print("No model found in the .pkl file.")