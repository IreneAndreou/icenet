import uproot
import numpy as np

# Load the ROOT file
input_file = uproot.open("travis-stash/input/icebrkprime/data/Tau_all_events.root")

# Get the tree from the file
tree = input_file["tree"]

# Define the criteria for data and MC
data_criteria = tree["idDeepTau2018v2p5VSjet_1"].array() > 5
mc_criteria = (tree["idDeepTau2018v2p5VSjet_1"].array() <= 5) & (tree["idDeepTau2018v2p5VSjet_1"].array() >= 0)

# Create new dictionaries to store the data and MC events
data_events = {key: tree[key].array()[data_criteria] for key in tree.keys()}
mc_events = {key: tree[key].array()[mc_criteria] for key in tree.keys()}

# Randomly sample 1,000,000 events for each
data_indices = np.random.choice(len(data_events["idDeepTau2018v2p5VSjet_1"]), size=200000, replace=False)
mc_indices = np.random.choice(len(mc_events["idDeepTau2018v2p5VSjet_1"]), size=200000, replace=False)

sampled_data_events = {key: data_events[key][data_indices] for key in data_events.keys()}
sampled_mc_events = {key: mc_events[key][mc_indices] for key in mc_events.keys()}

# Calculate the weight (wt) and add it as a new branch named wt_sf
sampled_data_events["wt_sf"] = sampled_data_events["weight"]
sampled_mc_events["wt_sf"] = sampled_mc_events["weight"]

sampled_data_events["jpt_pt_1"] = sampled_data_events["jpt_1"] / sampled_data_events["pt_1"]
sampled_data_events["jpt_pt_2"] = sampled_data_events["jpt_2"] / sampled_data_events["pt_2"]
sampled_mc_events["jpt_pt_1"] = sampled_mc_events["jpt_1"] / sampled_mc_events["pt_1"]
sampled_mc_events["jpt_pt_2"] = sampled_mc_events["jpt_2"] / sampled_mc_events["pt_2"]

# Save the new trees to new ROOT files
with uproot.recreate("travis-stash/input/icebrkprime/data/Tau_data_events.root") as data_file:
    data_file["tree"] = sampled_data_events

with uproot.recreate("travis-stash/input/icebrkprime/data/Tau_mc_events.root") as mc_file:
    mc_file["tree"] = sampled_mc_events

print("Data and MC files prepared and saved as Tau_data_events.root and Tau_mc_events.root")