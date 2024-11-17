# I need to implement the iso and anti-iso selections for the background subtraction as well ending up with 4 root files essentially

import uproot
import pandas as pd
import os
import subprocess

# List of ROOT files and their corresponding output names
MCFILES = [
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_10to50_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_1J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_2J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_3J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_4J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M-50_madgraphMLM_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_t-channel_antitop_4f_InclusiveDecays_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_t-channel_top_4f_InclusiveDecays_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_2L2Nu_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_2L2Nu_ext1_all_events.root",
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_4Q_all_events.root",
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_4Q_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_LNu2Q_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_LNu2Q_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_2L2Nu_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_2L2Nu_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_4Q_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_4Q_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_LNu2Q_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_LNu2Q_ext1_all_events.root",
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TT_all_events.root",
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TT_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto2L2Nu_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto2L2Nu_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto4Q_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto4Q_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTtoLNu2Q_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTtoLNu2Q_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_1J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_2J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_3J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_4J_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_madgraphMLM_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_madgraphMLM_ext1_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WW_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WZ_all_events.root",
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ZZ_all_events.root"
]

# Directory to save the filtered files
output_dir = "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/filtered_files"
os.makedirs(output_dir, exist_ok=True)

filtered_files = []

# Process each file
for file_path in MCFILES:
    output_name = os.path.basename(file_path)
    
    # Open the ROOT file
    with uproot.open(file_path) as file:
        # Assuming the tree name is "tree"
        tree = file["tree"]
        
        # Convert the tree to a pandas DataFrame
        df = tree.arrays(library="pd")
        
        # Apply the filter
        df_filtered = df[df['genPartFlav_1'] != 0].copy()

        # Set wt_sf to -wt_sf
        df_filtered.loc[:, 'wt_sf'] = -df_filtered['wt_sf']
        
        # Save the filtered DataFrame to a new ROOT file
        output_file_path = os.path.join(output_dir, output_name)
        with uproot.recreate(output_file_path) as new_file:
            new_file["tree"] = df_filtered.to_dict(orient="list")
        
        print(f"Processed and saved filtered data to {output_file_path}")
        filtered_files.append(output_file_path)

# Combine all filtered ROOT files into a single ROOT file using hadd
combined_output_path = os.path.join(output_dir, "mc_subtraction.root")
subprocess.run(["hadd", combined_output_path] + filtered_files)

print(f"All filtered files have been combined into {combined_output_path}")