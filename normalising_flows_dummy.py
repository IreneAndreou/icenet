import torch
import uproot
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

# Define the number of samples
n_samples = 1000

# Open the ROOT file and extract specific branches
file = uproot.open("travis-stash/input/icebrkprime/data/sampled/DY_all_events.root")
tree = file["tree"]

# Specify the branches you want to extract
branches = ['Z_mass','Z_pt','n_jets','n_deepbjets', 'mjj']

# Extract the data from the branches
data = tree.arrays(branches, library="np")

# Convert the data to a numpy array
mc_features = np.column_stack([data[branch] for branch in branches])

# Open the ROOT file and extract specific branches
file = uproot.open("travis-stash/input/icebrkprime/data/sampled/SingleMuon_all_events.root")
tree = file["tree"]

# Specify the branches you want to extract
branches = ['Z_mass','Z_pt','n_jets','n_deepbjets', 'mjj']

# Extract the data from the branches
data = tree.arrays(branches, library="np")

# Convert the data to a numpy array
data_features = np.column_stack([data[branch] for branch in branches])

# Sample 1000 entries from both datasets
mc_indices = np.random.choice(mc_features.shape[0], n_samples, replace=False)
data_indices = np.random.choice(data_features.shape[0], n_samples, replace=False)
mc_features = mc_features[mc_indices]
data_features = data_features[data_indices]

# Check for NaNs in the data
if np.isnan(mc_features).any() or np.isnan(data_features).any():
    raise ValueError("Data contains NaNs")

# Normalize the data
mc_features = (mc_features - mc_features.mean(axis=0)) / mc_features.std(axis=0)
data_features = (data_features - data_features.mean(axis=0)) / data_features.std(axis=0)

# Convert features to PyTorch tensors
X_MC = torch.tensor(mc_features, dtype=torch.float32)
X_data = torch.tensor(data_features, dtype=torch.float32)

# Create datasets and dataloaders
dataset = TensorDataset(X_MC)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameter optimization using Optuna (commented out)
# def objective(trial):
#     # Define hyperparameters to be optimized
#     num_layers = trial.suggest_int('num_layers', 3, 10)
#     hidden_units = trial.suggest_int('hidden_units', 64, 256)
#     learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # Adjusted learning rate range
    
#     # Define the flow architecture
#     base_distribution = StandardNormal(shape=[X_MC.shape[1]])
#     transforms = [MaskedAffineAutoregressiveTransform(features=X_MC.shape[1], hidden_features=hidden_units)
#                   for _ in range(num_layers)]
#     transform = CompositeTransform(transforms)
#     flow = Flow(transform, base_distribution)
    
#     # Define the optimizer
#     optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    
#     # Training loop (with early stopping)
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         flow.train()
#         epoch_loss = 0.0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             x_batch = batch[0]
#             loss = -flow.log_prob(x_batch).mean()
#             if torch.isnan(loss):
#                 raise ValueError("Loss is NaN")
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # Gradient clipping
#             optimizer.step()
#             epoch_loss += loss.item()
#         print(f'Hyperopt Training - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')
    
#     # Evaluate using log-probability on the target data as a measure of how well the flow matches the data
#     flow.eval()
#     with torch.no_grad():
#         val_loss = -flow.log_prob(X_data).mean().item()
    
#     # Save the model if it is the best one
#     if trial.should_prune():
#         raise optuna.exceptions.TrialPruned()
#     return val_loss

# # Optimize hyperparameters with Optuna
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Save the best hyperparameters
# best_trial = study.best_trial
# best_num_layers = best_trial.params['num_layers']
# best_hidden_units = best_trial.params['hidden_units']
# best_learning_rate = best_trial.params['lr']

# # Print the best parameters
# print("Best parameters found: ", study.best_params)
# print("Best validation loss: ", study.best_value)

# Train the model for 100 epochs with predefined hyperparameters
def train_model(num_layers, hidden_units, learning_rate):
    base_distribution = StandardNormal(shape=[X_MC.shape[1]])
    transforms = [MaskedAffineAutoregressiveTransform(features=X_MC.shape[1], hidden_features=hidden_units)
                  for _ in range(num_layers)]
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_distribution)

    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    num_epochs = 20
    loss_values = []  # List to store loss values
    for epoch in range(num_epochs):
        flow.train()
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            x_batch = batch[0]
            loss = -flow.log_prob(x_batch).mean()
            if torch.isnan(loss):
                raise ValueError("Loss is NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_epoch_loss)  # Append the average loss for the epoch
        print(f'Training - Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss}')

    # Save the final model
    torch.save(flow.state_dict(), 'best_flow_model_mydata.pth')
    return flow, loss_values

# Train the model with predefined hyperparameters
num_layers = 5
hidden_units = 128
learning_rate = 0.001
best_flow, loss_values = train_model(num_layers, hidden_units, learning_rate)

# Plot the loss curve
plt.figure()
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss_curve.png')
plt.show()

# Load the best model for plotting
best_flow.load_state_dict(torch.load('best_flow_model_mydata.pth'))
best_flow.eval()

# Visualize the results
with torch.no_grad():
    flow_samples = best_flow.sample(1000).numpy()

# Plotting the distributions for comparison
plt.figure(figsize=(12, 8))

# 2D scatter plots
plt.subplot(2, 2, 1)
plt.scatter(mc_features[:, 0], mc_features[:, 1], alpha=0.5, label='Original MC', color='blue')
plt.scatter(data_features[:, 0], data_features[:, 1], alpha=0.5, label='Data', color='orange')
plt.title('Original Distributions')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(flow_samples[:, 0], flow_samples[:, 1], alpha=0.5, label='Reweighted MC', color='green')
plt.scatter(data_features[:, 0], data_features[:, 1], alpha=0.5, label='Data', color='orange')
plt.title('Reweighted MC using Best Normalizing Flow')
plt.legend()

# 1D histograms for feature 1
plt.subplot(2, 2, 3)
plt.hist(mc_features[:, 0], bins=30, alpha=0.5, label='Original MC', color='blue', density=True)
plt.hist(data_features[:, 0], bins=30, alpha=0.5, label='Data', color='orange', density=True)
plt.hist(flow_samples[:, 0], bins=30, alpha=0.5, label='Reweighted MC', color='green', density=True)
plt.title('Feature 1 Distribution')
plt.legend()

# Save the plot for Feature 1 Distribution
plt.savefig('feature_1_distribution_mydata.png')

# 1D histograms for feature 2
plt.subplot(2, 2, 4)
plt.hist(mc_features[:, 1], bins=30, alpha=0.5, label='Original MC', color='blue', density=True)
plt.hist(data_features[:, 1], bins=30, alpha=0.5, label='Data', color='orange', density=True)
plt.hist(flow_samples[:, 1], bins=30, alpha=0.5, label='Reweighted MC', color='green', density=True)
plt.title('Feature 2 Distribution')
plt.legend()

# Save the plot for Feature 2 Distribution
plt.savefig('feature_2_distribution_mydata.png')

plt.tight_layout()
plt.show()