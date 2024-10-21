import torch

# Replace 'your_file.pth' with the path to your .pth file
file_path = '/vols/cms/ia2318/icenet/checkpoint/brkprime/config__tune0.yml/modeltag__None/2024-10-03_10-58-45_lx06/DMLP/DMLP_49.pth'

# Load the checkpoint
checkpoint = torch.load(file_path)

# Inspect the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# Extract the model configuration from the checkpoint
if 'param' in checkpoint:
    config = checkpoint['param']
    print("Model Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
else:
    print("No model configuration found in the checkpoint")

# Extract the state dictionary
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    
    # Find the final layer's weight tensor
    final_layer_weight = None
    for key, value in state_dict.items():
        if 'weight' in key and len(value.shape) == 2:
            final_layer_weight = value
            break
    
    if final_layer_weight is not None:
        output_size = final_layer_weight.shape[0]
        print(f"Extracted Output Size: {output_size}")
    else:
        print("Could not determine the output size from the state dictionary")
else:
    print("No state_dict found in the checkpoint")