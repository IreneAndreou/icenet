import os
import sys
import uproot
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def load_params(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def rename_tree(input_file, params, old_tree_name="ntuple", new_tree_name="tree", chunk_size=100000):
    output_file = input_file.replace(".root", "_all_events.root")
    pd.set_option('display.max_columns', None)

    with uproot.open(input_file) as file:
        print(file.keys())  
        tree = file[old_tree_name]

        # TODO: automatically get the lumi value from the file name
        # Get the lumi value from the SingleMuon entry
        single_muon_lumi = params.get("SingleMuon", {}).get("lumi", 1)
        print(single_muon_lumi)
        # Get the lumi value from the Tau entry
        single_muon_lumi = params.get("Tau", {}).get("lumi", 1)

        # Process the data in chunks
        with uproot.recreate(output_file) as new_file:
            # Initialize the progress bar
            total_entries = tree.num_entries
            with tqdm(total=total_entries, desc=f"Processing {os.path.basename(input_file)}", unit="entries") as pbar:
                for chunk in tree.iterate(library="pd", step_size=chunk_size):

                    # Create a new weight column for scaling MC'
                    if os.path.basename(input_file).startswith('DY'):
                        base_name = os.path.splitext(os.path.basename(input_file))[0]
                        base_name = base_name.replace('zmm_2018', '').rstrip('-_')
                        print(base_name)
                        xs = params.get(base_name, {}).get("xs", 1)
                        evt = params.get(base_name, {}).get("evt", 1)
                        print(single_muon_lumi, xs, evt)
                        wt_sf = single_muon_lumi * xs / evt
                        print(wt_sf)
                        chunk['wt_sf'] = wt_sf * chunk['wt']

                    # if os.path.basename(input_file).startswith('SingleMuon'):
                    else:
                        base_name = os.path.splitext(os.path.basename(input_file))[0]
                        # base_name = base_name.replace('zmm_2018', '').rstrip('-_')
                        print(base_name)
                        print(chunk['weight'])  # in Run 3 IT IS CALLED WEIGHT ARGHHHHH
                        chunk['wt_sf'] = chunk['weight']

                    if new_tree_name in new_file:
                        # Append to the existing tree
                        new_file[new_tree_name].extend(chunk)
                    else:
                        # Create a new tree
                        new_file[new_tree_name] = chunk
                    # Update the progress bar
                    pbar.update(len(chunk))

        print(f"Renamed tree in {input_file} from {old_tree_name} to {new_tree_name} and saved to {output_file}")

def process_file(file_path, params):
    rename_tree(file_path, params)

def main(file_paths, params_file):
    params = load_params(params_file)
    # Use multiprocessing to process files in parallel
    with Pool(cpu_count()) as pool:
        pool.starmap(process_file, [(file_path, params) for file_path in file_paths])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rename_trees.py <params_file.json> <file1> <file2> ...")
        sys.exit(1)
    
    params_file = sys.argv[1]
    file_paths = sys.argv[2:]
    main(file_paths, params_file)