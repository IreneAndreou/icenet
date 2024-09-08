# High Level Trigger steering code
# ADAPTED FOR BRKPRIME
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from icebrkprime import common

def main():
    args = process.generic_flow(rootname='brkprime', func_loader=common.load_root_file, func_factor=common.splitfactor)

if __name__ == '__main__' :
    main()