import torch
import torch.nn as nn
import torch.nn.functional as F
import uproot
import pandas as pd
import numpy as np
from tqdm import tqdm

class DMLP(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size, activation='relu', batch_norm=False, dropout=0.01):
        super(DMLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Define the activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Input layer
        self.mlp.append(nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        ))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.mlp.append(nn.Sequential(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]) if batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout)
            ))

        # Output layer
        self.mlp.append(nn.Linear(hidden_dims[-1], output_size))

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

# Replace 'your_file.pth' with the path to your .pth file
file_path = '/vols/cms/ia2318/icenet/checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-14_15-30-57_lx06/DMLP/DMLP_41.pth'

# Load the checkpoint
checkpoint = torch.load(file_path)

# Inspect the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# Extract the model configuration from the checkpoint
if 'param' in checkpoint:
    config = checkpoint['param']
    model_param = config['model_param']
    input_size = 5  # Number of input features (adjust as needed)
    hidden_dims = model_param['mlp_dim']
    output_size = 2  # Assuming binary classification with two logits
    activation = model_param['activation']
    batch_norm = model_param['batch_norm']
    dropout = model_param['dropout']
    
    print("Model Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
else:
    raise KeyError("No model configuration found in the checkpoint")

# Extract the state dictionary
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    
    # Define the model architecture based on the extracted configuration
    model = DMLP(input_size, hidden_dims, output_size, activation, batch_norm, dropout)
    
    # Rename keys in state_dict to match the model's expected keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('layer', '0')
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
else:
    raise KeyError("No state_dict found in the checkpoint")

model.eval()  # Set the model to evaluation mode

# Identify the final layer and the layer before it
final_layer = list(model.mlp)[-1]
layer_before_final = list(model.mlp)[-2]

print("Final layer:", final_layer)
print("Layer before final:", layer_before_final)

# Print the size of the output of the layer before the final ReLU activation
if isinstance(layer_before_final, nn.Sequential):
    for sublayer in reversed(layer_before_final):
        if isinstance(sublayer, nn.Linear):
            print("Output size of the layer before final ReLU:", sublayer.out_features)
            break
else:
    print("Output size of the layer before final ReLU:", layer_before_final.out_features)

# Load some test data from a ROOT file
X_test_path = '/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DY1JetsToLL-LO_zmm_2018_all_events.root'

# Open the ROOT file and extract the tree
with uproot.open(X_test_path) as file:
    tree = file["tree"]  # Replace "tree" with the actual name of the tree in the ROOT file

# Define the branches to load
branches = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj']

# Convert the tree to a pandas DataFrame, loading only the necessary branches
df = tree.arrays(branches, library="pd")

# Sample random events from the DataFrame
sample_size = 100  # Number of random events to sample
X_test = df.sample(n=sample_size)

# Ensure the DataFrame has the correct feature names
feature_names = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj']
X_test = X_test[feature_names]

# Convert the test data to a PyTorch tensor
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    raw_scores = model(X_test_tensor)
    #probabilities = F.softmax(raw_scores, dim=1).numpy()  # Apply softmax for multi-class probabilities
    probabilities = torch.sigmoid(raw_scores).numpy()  # Apply sigmoid for binary classification probabilities

# Print raw scores to 3 decimal places
print("Raw scores:")
print(np.around(raw_scores.numpy(), decimals=3))

# Print probabilities to 3 decimal places
print("Probabilities:")
print(np.around(probabilities, decimals=3))

# Calculate fake factors using probabilities (if applicable)
# Note: Fake factors calculation may need adjustment for multi-class classification
ff = probabilities / (1 - probabilities)

# Print fake factors to 3 decimal places
print("Fake factors:")
print(np.around(ff, decimals=3))