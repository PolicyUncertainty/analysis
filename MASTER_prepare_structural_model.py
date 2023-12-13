# Gathers everything necessary for structural model estimation.
# This script executes the following steps:
# 0. Set paths and parameters.
# 1. Get choice and state variables from SOEP core and SOEP RV VSKT.
# 2. Estimates wage equation parameters.
# 3. Estimates policy expectation process parameters.

# locations:
# dependencies: original data saved on local machines, functions for three steps in src folder
# output: data (decisions and outside_parameters) saved in output folder


#----------------------------------------------------------------------------------------------
# Step 0: Set paths and parameters
#----------------------------------------------------------------------------------------------

# Set file paths
paths_dict = {