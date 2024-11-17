import torch
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

# Data preparation
X_MC = torch.tensor(mc_features).float()
X_data = torch.tensor(data_features).float()

# Create datasets and dataloaders
dataset = TensorDataset(X_MC)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define flow architecture
base_distribution = StandardNormal(shape=[X_MC.shape[1]])
transforms = [MaskedAffineAutoregressiveTransform(features=X_MC.shape[1]) for _ in range(5)]
transform = CompositeTransform(transforms)
flow = Flow(transform, base_distribution)

# Optimizer and training loop
optimizer = torch.optim.Adam(flow.parameters(), lr=0.001)

for epoch in range(100):
    flow.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        x_batch = batch[0]
        loss = -flow.log_prob(x_batch).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}')
