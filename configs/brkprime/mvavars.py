
# Generator level variables
KINEMATIC_GEN_VARS = [
]

# Variables to plot etc.
KINEMATIC_VARS = [
]

# Use here only variables available in real data.
MVA_SCALAR_VARS = [
  'decayMode_1',
  'decayMode_2',
  'jpt_pt_1',
  'jpt_pt_2',
  'pt_1',
  'pt_2',
  'eta_1',
  'eta_2',
  'charge_1',
  'charge_2',
  'n_jets',
  'n_prebjets',
# 'Z_mass',
# 'Z_pt',
# 'n_jets',
# 'n_deepbjets', 
# 'mjj',
]

# Mutual information regularization targets
MI_VARS = [
]

# Variables we use with name replacement, need to be found in both MC and DATA
NEW_VARS = [
  'decayMode_1',
  'decayMode_2',
  'jpt_pt_1',
  'jpt_pt_2',
  'pt_1',
  'pt_2',
  'eta_1',
  'eta_2',
  'charge_1',
  'charge_2',
  'n_jets',
  'n_prebjets',
# 'Z_mass',
# 'Z_pt',
# 'n_jets',
# 'n_deepbjets', 
# 'mjj',
]

# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
LOAD_VARS = ['.*'] # all
