#TODO: Do feature importance!

import argparse
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp
import os

# Activate the environment
# source /vols/cms/ia2318/MLTools/Evaluation/python/myenv/bin/activate
#deactivate

# TODO: FIX THIS!!!!! - add tau lead/sublead and fix folders

# Argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--era', type=str, choices=['Run3_2022', 'Run3_2022EE'], required=True, help='Era to process: Run3_2022 or Run3_2022EE')
args = parser.parse_args()

# Set paths based on the era
if args.era == 'Run3_2022':
    file_path = 'best_models/Run3_2022/lead/no_global/best_model_tau1_all_var.pkl'
    X_test_path = 'travis-stash/input/icebrkprime/data/Tau_mc_events_lead_Run3_2022.root'
    single_muon_path = 'travis-stash/input/icebrkprime/data/Tau_data_events_lead_Run3_2022.root'
    output_dir = "ff_plots_tau_1_vars_mybdt_ss_Run3_2022_unbalanced"
elif args.era == 'Run3_2022EE':
    file_path = 'best_models/best_model_tau1_all_var_Run3_2022EE.pkl'
    X_test_path = 'travis-stash/input/icebrkprime/data/Tau_mc_events_Run3_2022EE.root'
    single_muon_path = 'travis-stash/input/icebrkprime/data/Tau_data_events_Run3_2022EE.root'
    output_dir = "ff_plots_tau_1_vars_mybdt_ss_Run3_2022EE"

# TODO: FIXT THIS!!!!!
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the model
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# # Paths to the test and single muon data
# X_test_path = 'travis-stash/input/icebrkprime/data/Tau_mc_events_Run3.root'#'test_data_with_label_0.root'
# single_muon_path = 'travis-stash/input/icebrkprime/data/Tau_data_events.root'#'test_data_with_label_1.root'

# Load the test data
with uproot.open(X_test_path) as file:
    tree = file["tree"]
print(f"Number of entries in the MC tree: {tree.num_entries}")

branches = ['decayMode_1', 'decayMode_2', 'jpt_pt_1', 'jpt_pt_2', 'pt_1', 'pt_2', 'eta_1', 'eta_2', 'charge_1', 'charge_2', 'phi_1', 'phi_2','n_jets', 'n_prebjets', 'n_bjets','met_pt', 'met_phi', 'met_dphi_1', 'met_dphi_2', 'decayModePNet_1', 'decayModePNet_2', 'wt_sf']

df = tree.arrays(branches, library="pd")

feature_names = ['decayMode_1', 'jpt_pt_1', 'pt_1', 'eta_1', 'charge_1', 'phi_1', 'decayModePNet_1'] # feature_names = model.feature_names

feature_names_to_plot = feature_names + ['decayMode_2', 'jpt_pt_2', 'pt_2', 'eta_2', 'charge_2', 'phi_2','met_pt','met_phi', 'met_dphi_1', 'met_dphi_2', 'decayModePNet_2', 'n_jets', 'n_prebjets', 'n_bjets']
latex_feature_names = {
    'decayMode_1': r'$\text{Decay Mode 1}$',
    'decayMode_2': r'$\text{Decay Mode 2}$',
    'jpt_pt_1': r'$\text{Jet Pt 1}$',
    'jpt_pt_2': r'$\text{Jet Pt 2}$',
    'pt_1': r'$\text{Pt 1}$',
    'pt_2': r'$\text{Pt 2}$',
    'eta_1': r'$\eta 1$',
    'eta_2': r'$\eta 2$',
    'phi_1': r'$\phi 1$',
    'phi_2': r'$\phi 2$',
    'charge_1': r'$\text{Charge 1}$',
    'charge_2': r'$\text{Charge 2}$',
    'n_jets': r'$\text{Number of Jets}$',
    'n_prebjets': r'$\text{Number of pre-bjets}$',
    'n_bjets': r'$\text{Number of b-jets}$',
    'met_pt': r'$\text{MET pT}$',
    'met_phi': r'$\text{MET phi}$',
    'met_dphi_1': r'$\text{MET dphi 1}$',
    'met_dphi_2': r'$\text{MET dphi 2}$',
    'decayModePNet_1': r'$\text{Decay Mode_1 PNet}$',
    'decayModePNet_2': r'$\text{Decay Mode_2 PNet}$'
}
original_weights = df['wt_sf']
X_test = df[feature_names]

# Load the single muon data
with uproot.open(single_muon_path) as file:
    tree_single_muon = file["tree"]
print(f"Number of entries in the data tree: {tree_single_muon.num_entries}")
df_single_muon = tree_single_muon.arrays(branches, library="pd")

def process_batch(batch_df, batch_index):
    dmatrix = xgb.DMatrix(batch_df[feature_names])
    probabilities = model.predict(dmatrix)
    new_weights = probabilities / (1 - probabilities)
    print(f"Processed batch {batch_index + 1}")
    return new_weights

batch_size = 10000
num_batches = len(df) // batch_size + 1

new_weights_list = []
pt_1_list = []
dm_list = []
for i in range(num_batches):
    batch_df = df.iloc[i * batch_size:(i + 1) * batch_size]
    new_weights = process_batch(batch_df, i)
    new_weights_list.append(new_weights)
    pt_1_list.append(batch_df['pt_1'].values)
    dm_list.append(batch_df['decayMode_1'].values)

new_weights = np.concatenate(new_weights_list)
pt_1 = np.concatenate(pt_1_list)
dm = np.concatenate(dm_list)

# Multiply original weights with new weights
combined_weights = original_weights * new_weights

# Create a DataFrame with pt_1 and new_weight
df_weights = pd.DataFrame({'pt_1': pt_1, 'combined_weight': combined_weights, 'dm': dm})

# Save the DataFrame to a CSV file
df_weights.to_csv(os.path.join(output_dir, 'pt_1_with_combined_weights.csv'), index=False)

print("Saved pt_1 and new weights to pt_1_with_combined_weights.csv")

# # Normalize combined weights
# combined_weights /= np.sum(combined_weights) / np.sum(original_weights)
# print(combined_weights)
# import sys
# sys.exit()

df_dm_0_1 = df_weights[(df_weights['dm'] == 0) | (df_weights['dm'] == 1)]
df_dm_10_11 = df_weights[(df_weights['dm'] == 10) | (df_weights['dm'] == 11)]


hep.style.use("CMS")

# Define the output directory relative to the script directory
# output_dir = "ff_plots_tau_1_all_vars_mybdt_ss"
# os.makedirs(output_dir, exist_ok=True)

# Plot the distribution of the new weights
plt.hist(combined_weights, bins=50, alpha=0.75, label='All DMs')
plt.hist(df_dm_0_1['combined_weight'], bins=50, alpha=0.75, label='DM0 and DM1')
plt.hist(df_dm_10_11['combined_weight'], bins=50, alpha=0.75, label='DM10 and DM11')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Distribution of New Weights')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, 'new_weights_distribution.pdf'))


# Define plotting ranges and number of bins for each feature
plotting_ranges = {
    'decayMode_1': (0, 12),
    'decayMode_2': (0, 12),
    'jpt_pt_1': (0, 15),
    'jpt_pt_2': (0, 15),
    'pt_1': (35, 800),
    'pt_2': (35, 800),
    'eta_1': (-3, 3),
    'eta_2': (-3, 3),
    'phi_1': (-3.2, 3.2),
    'phi_2': (-3.2, 3.2),
    'charge_1': (-2, 2),
    'charge_2': (-2, 2),
    'n_jets': (0, 10),
    'n_prebjets': (0, 10),
    'n_bjets': (0, 4),
    'met_pt': (0, 200),
    'met_phi': (-3.2, 3.2),
    'met_dphi_1': (-3.2, 3.2),
    'met_dphi_2': (-3.2, 3.2),
    'decayModePNet_1': (0, 12),
    'decayModePNet_2': (0, 12)
}

num_bins = {
    'decayMode_1': 4,  # Discrete bins: 0, 1, 10, 11
    'decayMode_2': 4,  # Discrete bins: 0, 1, 10, 11
    'jpt_pt_1': 15,
    'jpt_pt_2': 15,
    'pt_1': 100,
    'pt_2': 100,
    'eta_1': 30,
    'eta_2': 30,
    'phi_1': 30,
    'phi_2': 30,
    'charge_1': 4,
    'charge_2': 4,
    'n_jets': 10,
    'n_prebjets': 10,
    'n_bjets': 4,
    'met_pt': 40,
    'met_phi': 30,
    'met_dphi_1': 30,
    'met_dphi_2': 30,
    'decayModePNet_1': 5,  # Discrete bins: 0, 1, 2, 10, 11
    'decayModePNet_2': 5,  # Discrete bins: 0, 1, 2, 10, 11
}

# Define bins for discrete features
discrete_bins = {
    'decayMode_1': np.array([0, 1, 2, 10, 11, 12]),
    'decayMode_2': np.array([0, 1, 2, 10, 11, 12]),
    'decayModePNet_1': np.array([0, 1, 2, 3, 10, 11, 12]),
    'decayModePNet_2': np.array([0, 1, 2, 3, 10, 11, 12])
}

def rebin_histogram_errors(hist_counts, hist_errors, bin_edges, uncertainty_threshold=0.35):
    """Rebins the histogram to ensure the uncertainty in each bin is not more than 35% of the bin value."""
    new_bin_edges = [bin_edges[0]]
    current_bin_count = 0
    current_bin_error_sum = 0

    for i in range(len(hist_counts)):
        current_bin_count = hist_counts[i]
        current_bin_error_sum = hist_errors[i]
        uncertainty = np.sqrt(current_bin_error_sum)
        print(f"Bin {i}, Count: {current_bin_count}, Uncertainty: {uncertainty}")
        if uncertainty < uncertainty_threshold * current_bin_count:
            new_bin_edges.append(bin_edges[i + 1])
            current_bin_count = 0
            current_bin_error_sum = 0

    if new_bin_edges[-1] != bin_edges[-1]:
        new_bin_edges.append(bin_edges[-1])

    return np.array(new_bin_edges)

for i, branch in enumerate(feature_names_to_plot):
    # Calculate bin edges using the entire dataset
    if (branch in discrete_bins):
        bin_edges = discrete_bins[branch]
    else:
        bin_edges = np.linspace(plotting_ranges[branch][0], plotting_ranges[branch][1], num_bins[branch] + 1)

    original_hist_counts = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    combined_hist_counts = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    single_muon_hist_counts = np.zeros(len(bin_edges) - 1, dtype=np.float64)

    original_hist_errors = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    combined_hist_errors = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    single_muon_hist_errors = np.zeros(len(bin_edges) - 1, dtype=np.float64)

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

        original_hist_errors += np.histogram(batch_df[branch], bins=bin_edges, weights=batch_original_weights**2)[0]
        combined_hist_errors += np.histogram(batch_df[branch], bins=bin_edges, weights=batch_combined_weights**2)[0]
        single_muon_hist_errors += np.histogram(df_single_muon[branch].iloc[j * batch_size:(j + 1) * batch_size], bins=bin_edges)[0]

        print(f"Updated histogram for batch {j + 1} for feature {branch}")

    # Rebin the histograms based on the histogram errors
    if branch not in discrete_bins:
        print(bin_edges)
        bin_edges = rebin_histogram_errors(original_hist_counts, original_hist_errors, bin_edges)
        print(bin_edges)
        original_hist_counts, _ = np.histogram(df[branch], bins=bin_edges, weights=original_weights)
        combined_hist_counts, _ = np.histogram(df[branch], bins=bin_edges, weights=combined_weights)
        single_muon_hist_counts, _ = np.histogram(df_single_muon[branch], bins=bin_edges)

        original_hist_errors = np.histogram(df[branch], bins=bin_edges, weights=original_weights**2)[0]
        combined_hist_errors = np.histogram(df[branch], bins=bin_edges, weights=combined_weights**2)[0]
        single_muon_hist_errors = np.histogram(df_single_muon[branch], bins=bin_edges)[0]

    # Convert histogram counts to masked arrays to hide zero counts
    original_hist_counts = np.ma.masked_where(original_hist_counts == 0, original_hist_counts)
    combined_hist_counts = np.ma.masked_where(combined_hist_counts == 0, combined_hist_counts)
    single_muon_hist_counts = np.ma.masked_where(single_muon_hist_counts == 0, single_muon_hist_counts)

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate the ratio for original weights
    ratio_original = np.divide(single_muon_hist_counts, original_hist_counts, out=np.zeros_like(single_muon_hist_counts, dtype=np.float64), where=original_hist_counts != 0)

    # Calculate the ratio for combined weights
    ratio_combined = np.divide(single_muon_hist_counts, combined_hist_counts, out=np.zeros_like(single_muon_hist_counts, dtype=np.float64), where=combined_hist_counts != 0)

    # Calculate errors for the ratios
    ratio_original_errors = ratio_original * np.sqrt(
        (single_muon_hist_errors / single_muon_hist_counts**2) +
        (original_hist_errors / original_hist_counts**2)
    )
    ratio_combined_errors = ratio_combined * np.sqrt(
        (single_muon_hist_errors / single_muon_hist_counts**2) +
        (combined_hist_errors / combined_hist_counts**2)
    )

    # # Calculate chi-squared for original weights
    # chi_squared_original = np.sum(((ratio_original - 1) ** 2) / 1)

    # # Calculate chi-squared for combined weights
    # chi_squared_combined = np.sum(((ratio_combined - 1) ** 2) / 1)

    # Chi-Squared for original weights
    chi_squared_original = np.sum((single_muon_hist_counts - original_hist_counts) ** 2 / 
                                  (original_hist_errors + single_muon_hist_errors))
    reduced_chi_squared_original = chi_squared_original / (len(bin_centers) - 1)
    print(f"Chi-squared for original weights for feature {branch}: {chi_squared_original}")
    print(f"Reduced chi-squared for original weights for feature {branch}: {reduced_chi_squared_original}")

    # Chi-Squared for combined weights
    chi_squared_combined = np.sum((single_muon_hist_counts - combined_hist_counts) ** 2 / 
                                  (combined_hist_errors + single_muon_hist_errors))
    reduced_chi_squared_combined = chi_squared_combined / (len(bin_centers) - 1)
    print(f"Chi-squared for combined weights for feature {branch}: {chi_squared_combined}")
    print(f"Reduced chi-squared for combined weights for feature {branch}: {reduced_chi_squared_combined}")

    # K-S test for original weights
    ks_stat_original, ks_pvalue_original = ks_2samp(single_muon_hist_counts, original_hist_counts)
    print(f"K-S test for original weights for feature {branch}: statistic={ks_stat_original}, p-value={ks_pvalue_original}")

    # K-S test for combined weights
    ks_stat_combined, ks_pvalue_combined = ks_2samp(single_muon_hist_counts, combined_hist_counts)
    print(f"K-S test for combined weights for feature {branch}: statistic={ks_stat_combined}, p-value={ks_pvalue_combined}")

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, height_ratios=[3, 1])

    # Plot the original weights histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(bin_edges[:-1], bin_edges, weights=original_hist_counts, alpha=0.7, label='Original weights', histtype='step', linewidth=2)
    ax0.plot(bin_centers, single_muon_hist_counts, 'o', label='Tau Iso')
    ax0.set_title(f'{latex_feature_names[branch]} (Original)')
    ax0.set_xlabel(latex_feature_names[branch])
    ax0.set_ylabel('Counts')
    ax0.set_xlim(plotting_ranges[branch])
    ax0.legend()

    # Plot the ratio for original weights
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.errorbar(bin_centers, ratio_original, yerr=ratio_original_errors, fmt='o')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0., 2.5)
    ax1.axhline(1, color='r', linestyle='--')

    # Plot the combined weights histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bin_edges[:-1], bin_edges, weights=combined_hist_counts, alpha=0.7, label='New weights', histtype='step', linewidth=2)
    ax2.plot(bin_centers, single_muon_hist_counts, 'o', label='Tau Iso')
    ax2.set_title(f'{latex_feature_names[branch]} (Reweighting)')
    ax2.set_xlabel(latex_feature_names[branch])
    ax2.set_ylabel('Counts')
    ax2.set_xlim(plotting_ranges[branch])
    ax2.legend()

    # Plot the ratio for combined weights
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.errorbar(bin_centers, ratio_combined, yerr=ratio_combined_errors, fmt='o')
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0, 2.5)
    ax3.axhline(1, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{branch}_with_ratio_new_vars.pdf'))
    plt.close(fig)
    print(f'{branch} plot with ratio saved as a PDF file.')