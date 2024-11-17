# Example command: python make_input_rootfiles.py --tau leading

# TODO: Add a way to process both leading and subleading

import argparse
import uproot
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser(description="Process ROOT files with specified tau selection (leading or subleading).")
parser.add_argument("--tau", choices=["leading", "subleading"], default="leading", help="Specify whether to process leading or subleading tau.")

# Parse arguments
args = parser.parse_args()
tau_index = "1" if args.tau == "leading" else "2"
file_suffix = "lead" if args.tau == "leading" else "sublead"

# Load the ROOT file
input_files = ["travis-stash/input/icebrkprime/data/Tau_all_events_Run3_2022.root",
                "travis-stash/input/icebrkprime/data/Tau_all_events_Run3_2022EE.root"]


# Loop over each input file and process it
for input_file_path in input_files:
    print(f"Processing file: {input_file_path}")
    input_file = uproot.open(input_file_path)
    # Get the tree from the file
    tree = input_file["tree"]

    # Define the criteria for data and MC
    data_criteria = (tree[f"idDeepTau2018v2p5VSjet_{tau_index}"].array() >5) & (tree["os"].array() == 0)
    mc_criteria = (tree[f"idDeepTau2018v2p5VSjet_{tau_index}"].array() <= 5) & (tree[f"idDeepTau2018v2p5VSjet_{tau_index}"].array() >= 0) & (tree["os"].array() == 0)

    # Create new dictionaries to store the data and MC events
    data_events = {key: tree[key].array()[data_criteria] for key in tree.keys()}
    mc_events = {key: tree[key].array()[mc_criteria] for key in tree.keys()}

    # Randomly sample 1,000,000 events for each
    data_len = len(data_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    mc_len = len(mc_events[f"idDeepTau2018v2p5VSjet_{tau_index}"])
    min_len = min(data_len, mc_len)
    print(f"Data length: {data_len}, MC length: {mc_len}, Minimum length: {min_len}")
    data_indices = np.random.choice(data_len, size=data_len, replace=False)
    mc_indices = np.random.choice(mc_len, size=mc_len, replace=False)

    sampled_data_events = {key: data_events[key][data_indices] for key in data_events.keys()}
    sampled_mc_events = {key: mc_events[key][mc_indices] for key in mc_events.keys()}

    # Calculate the weight (wt) and add it as a new branch named wt_sf
    sampled_data_events["wt_sf"] = sampled_data_events["weight"]
    sampled_mc_events["wt_sf"] = sampled_mc_events["weight"]

    sampled_data_events["jpt_pt_1"] = sampled_data_events["jpt_1"] / sampled_data_events["pt_1"]
    sampled_data_events["jpt_pt_2"] = sampled_data_events["jpt_2"] / sampled_data_events["pt_2"]
    sampled_mc_events["jpt_pt_1"] = sampled_mc_events["jpt_1"] / sampled_mc_events["pt_1"]
    sampled_mc_events["jpt_pt_2"] = sampled_mc_events["jpt_2"] / sampled_mc_events["pt_2"]

    # Determine the output file name based on the input file name
    if "Run3_2022EE" in input_file_path:
        output_data_file_name = f"travis-stash/input/icebrkprime/data/Tau_data_events_{file_suffix}_Run3_2022EE.root"
        output_mc_file_name = f"travis-stash/input/icebrkprime/data/Tau_mc_events_{file_suffix}_Run3_2022EE.root"
    else:
        output_data_file_name = f"travis-stash/input/icebrkprime/data/Tau_data_events_{file_suffix}_Run3_2022.root"
        output_mc_file_name = f"travis-stash/input/icebrkprime/data/Tau_mc_events_{file_suffix}_Run3_2022.root"
    
    # Save the new trees to new ROOT files
    with uproot.recreate(output_data_file_name) as data_file:
        data_file["tree"] = sampled_data_events

    with uproot.recreate(output_mc_file_name) as mc_file:
        mc_file["tree"] = sampled_mc_events

    print(f"Data and MC files prepared and saved as {output_data_file_name} and {output_mc_file_name}")