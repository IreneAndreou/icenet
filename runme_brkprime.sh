#!/bin/sh
#
# Execute training and evaluation for the electron HLT trigger
#
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="./travis-stash/input/icebrkprime"
#DATAPATH="/vols/cms/mmieskol/HLT_electron_data/22112021"
#DATAPATH="/home/user/HLT_electron_data/22112021"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
mkdir "figs/brkprime/config-[$CONFIG]" -p # for output ascii dump

# tee redirect output to both a file and to screen
python analysis/brkprime.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/train_output.txt"
python analysis/brkprime.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/train_output.txt"
python analysis/brkprime.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/eval_output.txt"