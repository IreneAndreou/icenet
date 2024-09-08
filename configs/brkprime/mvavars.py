
# Generator level variables
KINEMATIC_GEN_VARS = [
]

# Variables to plot etc.
KINEMATIC_VARS = [
]

# Use here only variables available in real data.
MVA_SCALAR_VARS = [
'Z_mass',
'Z_pt',
'wt',
'n_jets',
'n_deepbjets', 
'mjj',
]

# Mutual information regularization targets
MI_VARS = [
]

# Variables we use with name replacement, need to be found in both MC and DATA
NEW_VARS = [
'Z_mass',
'Z_pt',
'wt',
'n_jets',
'n_deepbjets', 
'mjj',
]

# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
LOAD_VARS = ['.*'] # all
