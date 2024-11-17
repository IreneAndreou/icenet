import yaml

# Load the YAML file
with open('params_2022_preEE.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract the lumi value
lumi = data.get('lumi')

# Print the lumi value
print(f"Lumi: {lumi}")