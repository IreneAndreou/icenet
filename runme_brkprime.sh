#!/bin/sh
#
# Execute training and evaluation for the electron HLT trigger

# Read environment variables
export HTC_PROCESS_ID=$1
export HTC_QUEUE_SIZE=$2

# Source the Conda initialization script
source /vols/cms/ia2318/miniconda3/etc/profile.d/conda.sh

#Activate the virtual environment
conda activate icenet

source /vols/software/cuda/setup.sh 11.8.0

# Set CUDA environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run with: source runme.sh
source /vols/cms/ia2318/icenet/setenv.sh

CONFIG="tune0.yml"
DATAPATH="./travis-stash/input/icebrkprime"
#DATAPATH="/vols/cms/mmieskol/HLT_electron_data/22112021"
#DATAPATH="/home/user/HLT_electron_data/22112021"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
mkdir "figs/brkprime/fakefactors/config-[$CONFIG]" -p # for output ascii dump

# tee redirect output to both a file and to screen
python analysis/brkprime.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/train_output.txt"
python analysis/brkprime.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/train_output.txt"
python analysis/brkprime.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/brkprime/$CONFIG/eval_output.txt"