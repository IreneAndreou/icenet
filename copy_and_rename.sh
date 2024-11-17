#!/bin/bash

# Define the source files with input type
SOURCE_FILES=(
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/Tau_Run2022C/nominal/merged.root Tau_Run2022C.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/Tau_Run2022D/nominal/merged.root Tau_Run2022D.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022EE/tt/Tau_Run2022E/nominal/merged.root Tau_Run2022E.root Run3_2022EE"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022EE/tt/Tau_Run2022F/nominal/merged.root Tau_Run2022F.root Run3_2022EE"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022EE/tt/Tau_Run2022G/nominal/merged.root Tau_Run2022G.root Run3_2022EE"
)

MCFILES=(
    #"/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-10to50_madgraphMLM/nominal/merged.root DYto2L_M_10to50_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_1J_madgraphMLM/nominal/merged.root DYto2L_M-50_1J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_2J_madgraphMLM/nominal/merged.root DYto2L_M-50_2J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_3J_madgraphMLM/nominal/merged.root DYto2L_M-50_3J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_4J_madgraphMLM/nominal/merged.root DYto2L_M-50_4J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_madgraphMLM/nominal/merged.root DYto2L_M-50_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/DYto2L_M-50_madgraphMLM_ext1/nominal/merged.root DYto2L_M-50_madgraphMLM_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_t-channel_antitop_4f_InclusiveDecays/nominal/merged.root ST_t-channel_antitop_4f_InclusiveDecays.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_t-channel_top_4f_InclusiveDecays/nominal/merged.root ST_t-channel_top_4f_InclusiveDecays.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_2L2Nu/nominal/merged.root ST_tW_antitop_2L2Nu.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_2L2Nu_ext1/nominal/merged.root ST_tW_antitop_2L2Nu_ext1.root Run3_2022"
    #"/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_4Q/nominal/merged.root ST_tW_antitop_4Q.root Run3_2022"
    #"/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_4Q_ext1/nominal/merged.root ST_tW_antitop_4Q_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_LNu2Q/nominal/merged.root ST_tW_antitop_LNu2Q.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_antitop_LNu2Q_ext1/nominal/merged.root ST_tW_antitop_LNu2Q_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_2L2Nu/nominal/merged.root ST_tW_top_2L2Nu.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_2L2Nu_ext1/nominal/merged.root ST_tW_top_2L2Nu_ext1.root Run3_2022"
    #"/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_4Q/nominal/merged.root ST_tW_top_4Q.root Run3_2022"
    #"/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_4Q_ext1/nominal/merged.root ST_tW_top_4Q_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_LNu2Q/nominal/merged.root ST_tW_top_LNu2Q.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ST_tW_top_LNu2Q_ext1/nominal/merged.root ST_tW_top_LNu2Q_ext1.root Run3_2022"
    # "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TT/nominal/merged.root TT.root Run3_2022"
    # "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TT_ext1/nominal/merged.root TT_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTto2L2Nu/nominal/merged.root TTto2L2Nu.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTto2L2Nu_ext1/nominal/merged.root TTto2L2Nu_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTto4Q/nominal/merged.root TTto4Q.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTto4Q_ext1/nominal/merged.root TTto4Q_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTtoLNu2Q/nominal/merged.root TTtoLNu2Q.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/TTtoLNu2Q_ext1/nominal/merged.root TTtoLNu2Q_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_1J_madgraphMLM/nominal/merged.root WtoLNu_1J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_2J_madgraphMLM/nominal/merged.root WtoLNu_2J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_3J_madgraphMLM/nominal/merged.root WtoLNu_3J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_4J_madgraphMLM/nominal/merged.root WtoLNu_4J_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_madgraphMLM/nominal/merged.root WtoLNu_madgraphMLM.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WtoLNu_madgraphMLM_ext1/nominal/merged.root WtoLNu_madgraphMLM_ext1.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WW/nominal/merged.root WW.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/WZ/nominal/merged.root WZ.root Run3_2022"
    "/vols/cms/ks1021/offline/HiggsDNA/IC/output/production_fixPU/Run3_2022/tt/ZZ/nominal/merged.root ZZ.root Run3_2022"
)

# Define the destination directory
DEST_DIR="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the source files to the destination directory
for entry in "${SOURCE_FILES[@]}" "${MCFILES[@]}"; do
    src_file=$(echo $entry | awk '{print $1}')
    new_filename=$(echo $entry | awk '{print $2}')
    input_type=$(echo $entry | awk '{print $3}')
    cp "$src_file" "$DEST_DIR/$new_filename"
    echo "Copied $src_file to $DEST_DIR/$new_filename"
done

# Run the Python script to rename the trees for each file individually
for entry in "${SOURCE_FILES[@]}"; do
    new_filename=$(echo $entry | awk '{print $2}')
    input_type=$(echo $entry | awk '{print $3}')

    # Print the variables
    echo "new_filename: $new_filename"
    echo "input_type: $input_type"
    
    # Check the flag to determine the configuration file
    if [[ $input_type == "Run3_2022" ]]; then
        config_file="params_2022_preEE.yaml"
    else
        config_file="params_2022_postEE.yaml"
    fi
    
    python rename_trees.py "$config_file" "$DEST_DIR/$new_filename"
done

# Run the Python script to rename the trees for each MC file individually
for entry in "${MCFILES[@]}"; do
    new_filename=$(echo $entry | awk '{print $2}')
    input_type=$(echo $entry | awk '{print $3}')
    
    # Check the flag to determine the configuration file
    if [[ $input_type == "Run3_2022" ]]; then
        config_file="params_2022_preEE.yaml"
    else
        config_file="params_2022_postEE.yaml"
    fi
    
    # Check if the destination file exists and is not a directory
    if [ -f "$DEST_DIR/$new_filename" ]; then
        python rename_trees.py "$config_file" "$DEST_DIR/$new_filename"
    else
        echo "Error: $DEST_DIR/$new_filename is not a valid file"
    fi
done

# Define the target files based on input type
TARGET_FILE_RUN3_2022="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_all_events_Run3_2022.root"
TARGET_FILE_RUN3_2022EE="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/Tau_all_events_Run3_2022EE.root"
TARGET_FILE_RUN3_2022_MC="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/MC_all_events_Run3_2022.root"
TARGET_FILE_RUN3_2022EE_MC="/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/MC_all_events_Run3_2022EE.root"

# Define the source files for each input type
SOURCE_FILES_RUN3_2022=(
                        "travis-stash/input/icebrkprime/data/Tau_Run2022C_all_events.root"
                        "travis-stash/input/icebrkprime/data/Tau_Run2022D_all_events.root"
                        )
SOURCE_FILES_RUN3_2022EE=(
                        "travis-stash/input/icebrkprime/data/Tau_Run2022E_all_events.root"
                        "travis-stash/input/icebrkprime/data/Tau_Run2022F_all_events.root"
                        "travis-stash/input/icebrkprime/data/Tau_Run2022G_all_events.root")

MCFILES_RUN3_2022=(
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_10to50_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_1J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_2J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_3J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_4J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/DYto2L_M_50_madgraphMLM_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_t_channel_antitop_4f_InclusiveDecays_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_t_channel_top_4f_InclusiveDecays_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_2L2Nu_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_2L2Nu_ext1_Run3_2022.root"
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_4Q_Run3_2022.root"
    #"/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_4Q_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_LNu2Q_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_antitop_LNu2Q_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_2L2Nu_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_2L2Nu_ext1_Run3_2022.root"
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_4Q_Run3_2022.root"
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_4Q_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_LNu2Q_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ST_tW_top_LNu2Q_ext1_Run3_2022.root"
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TT_Run3_2022.root"
    # "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TT_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto2L2Nu_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto2L2Nu_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto4Q_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTto4Q_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTtoLNu2Q_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/TTtoLNu2Q_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_1J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_2J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_3J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_4J_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_madgraphMLM_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WtoLNu_madgraphMLM_ext1_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WW_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/WZ_Run3_2022.root"
    "/vols/cms/ia2318/icenet/travis-stash/input/icebrkprime/data/ZZ_Run3_2022.root"
)
MCFILES_RUN3_2022EE=(
                    )


# Process SOURCE_FILES
for entry in "${SOURCE_FILES[@]}"; do
    new_filename=$(echo $entry | awk '{print $2}')
    input_type=$(echo $entry | awk '{print $3}')
    if [ "$input_type" == "Run3_2022" ]; then
        SOURCE_FILES_RUN3_2022+=("$DEST_DIR/$new_filename")
    elif [ "$input_type" == "Run3_2022EE" ]; then
        SOURCE_FILES_RUN3_2022EE+=("$DEST_DIR/$new_filename")
    fi
done

# Process MCFILES
for entry in "${MCFILES[@]}"; do
    new_filename=$(echo $entry | awk '{print $2}')
    input_type=$(echo $entry | awk '{print $3}')
    if [ "$input_type" == "Run3_2022" ]; then
        MCFILES_RUN3_2022+=("$DEST_DIR/$new_filename")
    elif [ "$input_type" == "Run3_2022EE" ]; then
        MCFILES_RUN3_2022EE+=("$DEST_DIR/$new_filename")
    fi
done

# Print the list of files to be merged for debugging
echo "Files to be merged for Run3_2022: ${SOURCE_FILES_RUN3_2022[@]}"
echo "Files to be merged for Run3_2022EE: ${SOURCE_FILES_RUN3_2022EE[@]}"
echo "MC Files to be merged for Run3_2022: ${MCFILES_RUN3_2022[@]}"
echo "MC Files to be merged for Run3_2022EE: ${MCFILES_RUN3_2022EE[@]}"

# # Merge the ROOT files for Run3_2022
# echo "Merging ROOT files for Run3_2022..."
# hadd -f "$TARGET_FILE_RUN3_2022" "${SOURCE_FILES_RUN3_2022[@]}"
# if [ $? -ne 0 ]; then
#     echo "Error: Merge failed for Run3_2022"
#     exit 1
# fi

# # Merge the ROOT files for Run3_2022EE
# echo "Merging ROOT files for Run3_2022EE..."
# hadd -f "$TARGET_FILE_RUN3_2022EE" "${SOURCE_FILES_RUN3_2022EE[@]}"
# if [ $? -ne 0 ]; then
#     echo "Error: Merge failed for Run3_2022EE"
#     exit 1
# fi

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