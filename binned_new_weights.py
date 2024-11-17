import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the bin edges for pt_1
pt_bins = [40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160, 200, 240, 300, 400]

# Read the data from the CSV file
df = pd.read_csv('ff_plots_tau_1_vars_mybdt_ss_Run3_2022_unbalanced/pt_1_with_combined_weights.csv')

# Bin the data
df['pt_bin'] = pd.cut(df['pt_1'], bins=pt_bins, right=False)

# Calculate the average value and statistical uncertainty for each bin
bin_means = df.groupby('pt_bin')['combined_weight'].mean()
bin_counts = df.groupby('pt_bin')['combined_weight'].count()
bin_std = df.groupby('pt_bin')['combined_weight'].std()

# Calculate the statistical uncertainty (standard error of the mean)
bin_uncert = bin_std / np.sqrt(bin_counts)

# Print the results
print("Bin edges:", pt_bins)
print("Bin means:", bin_means)
print("Bin counts:", bin_counts)
print("Bin standard deviations:", bin_std)
print("Bin uncertainties:", bin_uncert)

# Save the results to a CSV file
results = pd.DataFrame({
    'bin_edges': pt_bins[:-1],
    'bin_means': bin_means.values,
    'bin_counts': bin_counts.values,
    'bin_std': bin_std.values,
    'bin_uncert': bin_uncert.values
})

# results.to_csv('binned_pt_1_with_new_weights.csv', index=False)
# print("Saved binned results to binned_pt_1_with_new_weights.csv")

# Plot the binned averages and their statistical uncertainties
bin_centers = (np.array(pt_bins[:-1]) + np.array(pt_bins[1:])) / 2

plt.errorbar(bin_centers, bin_means, yerr=bin_uncert, fmt='o', capsize=5, label='Binned Averages')
plt.xlabel('pT of Leading Tau (GeV)')
plt.ylabel('Average Combined Weight')
plt.title('Binned Averages of Combined Weights with Statistical Uncertainties')
plt.grid(True)
plt.legend()
plt.savefig('ff_plots_tau_1_vars_mybdt_ss_Run3_2022_unbalanced/binned_combined_averages_with_uncertainties.pdf')