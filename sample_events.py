import uproot
import pandas as pd
import sys

def sample_events(input_file, output_file, tree_name="tree", sample_size=1000000, chunk_size=1000000):
    print(f"Sampling {sample_size} events from {input_file} and saving to {output_file}", flush=True)
    with uproot.open(input_file) as file:
        print(f"Opened file: {input_file}", flush=True)
        print(file.keys(), flush=True)  
        tree = file[tree_name]

        # Initialize an empty DataFrame to store the sampled data
        df_sampled = pd.DataFrame()

        # Process the data in chunks
        chunk_count = 0
        for arrays in tree.iterate(step_size=chunk_size, library="pd"):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}", flush=True)
            
            # Randomly sample from the current chunk
            df_chunk_sampled = arrays.sample(frac=sample_size / tree.num_entries, random_state=42)
            # Append the sampled chunk to the DataFrame
            df_sampled = pd.concat([df_sampled, df_chunk_sampled], ignore_index=True)

        # Print the sampled DataFrame to verify
        print(f"Sampled DataFrame:\n{df_sampled}", flush=True)

        # Save the sampled DataFrame back into a ROOT file
        with uproot.recreate(output_file) as new_file:
            new_file[tree_name] = df_sampled

    print(f"Sampled {sample_size} events from {input_file} and saved to {output_file}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_events.py <input_file> <output_file>", flush=True)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    sample_events(input_file, output_file)