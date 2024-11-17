import numpy as np
import torch
import uproot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

# Define the number of samples
n_samples = 1000

# Load MC and data features using uproot (same as before)
file_mc = uproot.open("travis-stash/input/icebrkprime/data/sampled/DY_all_events.root")
tree_mc = file_mc["tree"]
branches = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj']
data_mc = tree_mc.arrays(branches, library="np")
mc_features = np.column_stack([data_mc[branch] for branch in branches])

file_data = uproot.open("travis-stash/input/icebrkprime/data/sampled/SingleMuon_all_events.root")
tree_data = file_data["tree"]
data_data = tree_data.arrays(branches, library="np")
data_features = np.column_stack([data_data[branch] for branch in branches])

# Sample 1000 entries from both datasets
mc_indices = np.random.choice(mc_features.shape[0], n_samples, replace=False)
data_indices = np.random.choice(data_features.shape[0], n_samples, replace=False)
mc_features = mc_features[mc_indices]
data_features = data_features[data_indices]

# Check for NaNs in the data
if np.isnan(mc_features).any() or np.isnan(data_features).any():
    raise ValueError("Data contains NaNs")

# Normalize the data
mc_mean = mc_features.mean(axis=0)
mc_std = mc_features.std(axis=0)
mc_features_normalized = (mc_features - mc_mean) / mc_std
data_features_normalized = (data_features - mc_mean) / mc_std

# Convert to PyTorch tensors
X_MC = torch.tensor(mc_features_normalized, dtype=torch.float32)
X_data = torch.tensor(data_features_normalized, dtype=torch.float32)

# Ensure tensors have at least one dimension
if X_MC.ndim == 0:
    X_MC = X_MC.unsqueeze(0)
if X_data.ndim == 0:
    X_data = X_data.unsqueeze(0)

# Prepare the training data dictionary with correct keys
n_samples = X_MC.shape[0]
train_data = {
    'prior_draws': X_MC.numpy(),  # Convert tensors to numpy arrays
    'sim_data': X_data.numpy(),  # Use 'sim_data' as the key for simulations
    'summary_conditions': np.ones((n_samples, 1), dtype=np.float32),  # Example condition
    'direct_conditions': np.ones((n_samples, 1), dtype=np.float32)  # Example condition, ensure it's not None
}

# Debug: Print the contents of train_data
print("train_data contents:")
for key, value in train_data.items():
    print(f"{key}: {value.shape}, dtype: {value.dtype}")

# Define the flow architecture using InvertibleNetwork
flow = InvertibleNetwork(
    num_params=X_MC.shape[1],  # Number of parameters (assumed to be the same as the number of input features)
    num_coupling_layers=6,  # Number of coupling layers
    coupling_design="affine",  # Coupling design
    permutation="fixed",  # Permutation type
    use_act_norm=True,  # Use activation normalization
    use_soft_flow=False  # Use soft flow
)

# Wrap the InvertibleNetwork flow with AmortizedPosterior for inference
posterior = AmortizedPosterior(inference_net=flow)

# Define the trainer
trainer = Trainer(amortizer=posterior)

# Train the model using the train_offline method
trainer.train_offline(
    simulations_dict=train_data,
    epochs=50,  # Number of training epochs
    batch_size=64
)

# Save the trained model for future use
torch.save(posterior.state_dict(), 'best_bayesflow_model.pth')

# Load the model if needed later
posterior.load_state_dict(torch.load('best_bayesflow_model.pth'))
posterior.eval()

# Generate reweighted MC samples
with torch.no_grad():
    reweighted_samples = posterior.sample(
        context=X_MC,  # Using the MC samples as context to generate reweighted samples
        n_samples=n_samples
    ).numpy()
    
# Reverse normalization for visualization
reweighted_samples_original_scale = reweighted_samples * mc_std + mc_mean

# Plot the original, data, and reweighted distributions
plt.figure(figsize=(12, 8))

# 2D scatter plot of the original MC and data
plt.subplot(2, 2, 1)
plt.scatter(mc_features[:, 0], mc_features[:, 1], alpha=0.5, label='Original MC', color='blue')
plt.scatter(data_features[:, 0], data_features[:, 1], alpha=0.5, label='Data', color='orange')
plt.title('Original Distributions')
plt.legend()

# 2D scatter plot of the reweighted MC and data
plt.subplot(2, 2, 2)
plt.scatter(reweighted_samples_original_scale[:, 0], reweighted_samples_original_scale[:, 1], alpha=0.5, label='Reweighted MC', color='green')
plt.scatter(data_features[:, 0], data_features[:, 1], alpha=0.5, label='Data', color='orange')
plt.title('Reweighted MC using BayesFlow')
plt.legend()

# Histograms for Feature 1
plt.subplot(2, 2, 3)
plt.hist(mc_features[:, 0], bins=30, alpha=0.5, label='Original MC', color='blue', density=True)
plt.hist(data_features[:, 0], bins=30, alpha=0.5, label='Data', color='orange', density=True)
plt.hist(reweighted_samples_original_scale[:, 0], bins=30, alpha=0.5, label='Reweighted MC', color='green', density=True)
plt.title('Feature 1 Distribution')
plt.legend()

# Histograms for Feature 2
plt.subplot(2, 2, 4)
plt.hist(mc_features[:, 1], bins=30, alpha=0.5, label='Original MC', color='blue', density=True)
plt.hist(data_features[:, 1], bins=30, alpha=0.5, label='Data', color='orange', density=True)
plt.hist(reweighted_samples_original_scale[:, 1], bins=30, alpha=0.5, label='Reweighted MC', color='green', density=True)
plt.title('Feature 2 Distribution')
plt.legend()

# Save and show the plots
plt.tight_layout()
plt.savefig('bayesflow_reweighted_distributions.png')
plt.show()