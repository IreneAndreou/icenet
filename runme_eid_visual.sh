#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"

#DATAPATH="/home/user/vols/cms/icenet/2020Oct16/"
DATAPATH="/home/user/imperial_new_trees"
#DATAPATH="/home/user/imperial_old_trees"
#DATAPATH="/vols/cms/icenet/data/2020Oct16/"


# Use * or other glob wildcards for filenames

python ./analysis/eid_visual_tensors.py --config $CONFIG --datapath $DATAPATH --datasets "output_*" #,1 #,2,3,4,5,6