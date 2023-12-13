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
# %%
# Set file paths
paths_dict = {
    # "soep_c38": "C:\Users\bruno\papers\soep\soep38",
    # "soep_rv": "C:\Users\bruno\papers\soep\soep_rv",
    "soep_c38": "/home/maxbl/Uni/pol_uncetainty/data/soep38",
    "soep_rv": "/home/maxbl/Uni/pol_uncetainty/data/soep_rv"
}

# Set options for data preparation
min_SRA = 65
min_ret_age = min_SRA - 4
exp_cap = 40

data_options = {
    "min_ret_age": min_ret_age,
    "start_year": 2010,
    "end_year": 2021,
    "start_age": 25,
    "exp_cap": 40
}
from src.gather_decision_data import gather_decision_data
dec_data = gather_decision_data(paths_dict, data_options, load_data=False)
# Add checks on our model choice sets. I.e. no retirement before min_ret_age,
# no working/unemployment after max_ret_age.
# Also our syear is from the soep. We could check, that single observations,
# have only two consecutive observations in soep.

# %%


#----------------------------------------------------------------------------------------------
# Step 1: Get choice and state variables from SOEP core and SOEP RV VSKT
#----------------------------------------------------------------------------------------------


# %%
