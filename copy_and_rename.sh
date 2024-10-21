#!/bin/bash

# Define the source files
SOURCE_FILES=(
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonC_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonA_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonB_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonD_zmm_2018.root"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/Tau_Run2022C/nominal/merged.root Tau_Run2022C.root"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/Tau_Run2022D/nominal/merged.root Tau_Run2022D.root"
)

# Define the mcfile paths
MCFILES=(
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY2JetsToLL-LO_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL_M-10to50-LO_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY3JetsToLL-LO_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY4JetsToLL-LO_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL-LO_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL-LO-ext1_zmm_2018.root"
    # "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY1JetsToLL-LO_zmm_2018.root"
)

# Define the destination directory
DEST_DIR="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# # Copy the source files to the destination directory
# for entry in "${SOURCE_FILES[@]}"; do
#     src_file=$(echo $entry | awk '{print $1}')
#     new_filename=$(echo $entry | awk '{print $2}')
#     cp "$src_file" "$DEST_DIR/$new_filename"
#     echo "Copied $src_file to $DEST_DIR/$new_filename"
# done

# # Copy the mcfiles to the destination directory
# for file in "${MCFILES[@]}"; do
#    cp "$file" "$DEST_DIR"
# done

# # TODO: Update for Run3!!! - and change from SinlgeMuon to Tau 

# # Run the Python script to rename the trees for each file individually

# for entry in "${SOURCE_FILES[@]}"; do
#     new_filename=$(echo $entry | awk '{print $2}')
#     python rename_trees.py params_2022_preEE.json "$DEST_DIR/$new_filename"
# done

# # Merge the processed SOURCE files using hadd
# hadd -f $DEST_DIR/Tau_all_events.root $DEST_DIR/Tau*_all_events.root
# Define the target and source files
TARGET_FILE="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_all_events.root"
SOURCE_FILES=(
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_Run2022C_all_events.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_Run2022D_all_events.root"
)

# Check if the target file exists and is not empty
if [ -f "$TARGET_FILE" ]; then
    if [ ! -s "$TARGET_FILE" ]; then
        echo "Warning: Target file $TARGET_FILE is empty. It will be overwritten."
        rm "$TARGET_FILE"
    fi
fi

# Merge the ROOT files
echo "Merging ROOT files..."
hadd -f "$TARGET_FILE" "${SOURCE_FILES[@]}"

# Check if the merge was successful
if [ $? -eq 0 ]; then
    echo "Merge completed successfully."
else
    echo "Error: Merge failed."
    exit 1
fi

# # Merge the processed MC files using hadd
# hadd -f $DEST_DIR/DY_all_events.root $DEST_DIR/DY*_all_events.root

# Run the Python script to rename the trees for each file individually

# for file in "${SOURCE_FILES[@]}" "${MCFILES[@]}"; do
#    base_file=$(basename "$file")
#    python rename_trees.py params_UL2018.json "$DEST_DIR/$base_file"
# done

# # Merge the processed SOURCE files using hadd
# hadd -f $DEST_DIR/SingleMuon_all_events.root $DEST_DIR/SingleMuon*_all_events.root

# # Merge the processed MC files using hadd
# hadd -f $DEST_DIR/DY_all_events.root $DEST_DIR/DY*_all_events.root

# # Randomly sample 100,000 events from the merged files

# # Define the merged files
# MERGED_FILES=(
#     "$DEST_DIR/SingleMuon_all_events.root"
#     "$DEST_DIR/DY_all_events.root"
# )

# # Directory to save the sampled files
# SAMPLED_DIR="$DEST_DIR/sampled"

# # Create the sampled directory if it doesn't exist
# mkdir -p "$SAMPLED_DIR"

# # Loop through each merged file and randomly sample 100,000 events
# for file in "${MERGED_FILES[@]}"; do
#     base_name=$(basename "$file")
#     sampled_file="$SAMPLED_DIR/$base_name"
    
#     echo "Processing file: $file"
    
#     # Call the Python script to sample events
#     python sample_events.py "$file" "$sampled_file"
# done