# source /vols/cms/ia2318/MLTools/Evaluation/python/myenv/bin/activate

import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec

#file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-14_15-30-57_lx06/XGB/XGB_44.pkl'  # best model
file_path = 'checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-15_11-08-01_lx06/XGB/XGB_44.pkl'  # best model - latest training (val_loss: 0.6226)

with open(file_path, 'rb') as file:
    data = pickle.load(file)

if 'model' in data:
    model = data['model']
    
    X_test_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DY_all_events.root'
    single_muon_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/SingleMuon_all_events.root'
    
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
    
    with uproot.open(single_muon_path) as file:
        tree_single_muon = file["tree"]
    print(f"Number of entries in the Single Muon tree: {tree_single_muon.num_entries}")
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

    for i, branch in enumerate(feature_names):
        # Calculate bin edges using the entire dataset
        bin_number = 500
        _, bin_edges = np.histogram(df[branch], bins=bin_number)

        original_hist_counts = np.zeros(bin_number)
        combined_hist_counts = np.zeros(bin_number)
        single_muon_hist_counts = np.zeros(bin_number)

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
        chi_squared_original = np.sum(((ratio_original - 1) ** 2) / 1)

        # Calculate chi-squared for combined weights
        chi_squared_combined = np.sum(((ratio_combined - 1) ** 2) / 1)

        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 2, height_ratios=[3, 1])

        # Plot the original weights histogram
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(bin_edges[:-1], bin_edges, weights=original_hist_counts, alpha=0.7, label='Original weights', histtype='step', linewidth=2)
        ax0.plot(bin_centers, single_muon_hist_counts, 'o', label='SingleMuon Data')
        ax0.set_title(f'{latex_feature_names[branch]} (Original)')
        ax0.set_xlabel(latex_feature_names[branch])
        ax0.set_ylabel('Counts')
        ax0.set_xlim(0, 200)
        ax0.legend()

        # Plot the ratio for original weights
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(bin_centers, ratio_original, 'o')
        ax1.set_ylabel('Ratio')
        ax1.set_ylim(0., 1.5)
        ax1.axhline(1, color='r', linestyle='--')

        # Plot the combined weights histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(bin_edges[:-1], bin_edges, weights=combined_hist_counts, alpha=0.7, label='New weights', histtype='step', linewidth=2)
        ax2.plot(bin_centers, single_muon_hist_counts, 'o', label='SingleMuon Data')
        ax2.set_title(f'{latex_feature_names[branch]} (Reweighting)')
        ax2.set_xlabel(latex_feature_names[branch])
        ax2.set_ylabel('Counts')
        ax2.set_xlim(0, 200)
        ax2.legend()

        # Plot the ratio for combined weights
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
        ax3.plot(bin_centers, ratio_combined, 'o')
        ax3.set_ylabel('Ratio')
        ax3.set_ylim(0, 1.5)
        ax3.axhline(1, color='r', linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{branch}_with_ratio.pdf')
        plt.close(fig)
        print(f'{branch} plot with ratio saved as a PDF file.')

else:
    print("No model found in the .pkl file.")