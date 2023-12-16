# Gathers everything necessary for structural model estimation.
# This script executes the following steps:
# 0. Set paths and parameters.
# 1. Get choice and state variables from SOEP core and SOEP RV VSKT.
# 2. Estimates wage equation parameters.
# 3. Estimates policy expectation process parameters.

# locations:
# dependencies: original data saved on local machines, functions for steps 0-3 in src folder
# output: data (decisions and outside_parameters) saved in output folder


# %%
# Step 0: Set paths and parameters
#----------------------------------------------------------------------------------------------

# Set file paths
paths_dict = {
    # SOEP Core and SOEP RV are saved locally
    "soep_c38": "C:/Users/bruno/papers/soep/soep38",
    "soep_rv": "C:/Users/bruno/papers/soep/soep_rv",
    #"soep_c38": "/home/maxbl/Uni/pol_uncetainty/data/soep38",
    #"soep_rv": "/home/maxbl/Uni/pol_uncetainty/data/soep_rv",
    # SOEP IS (our own questions) is saved on the server
    "soep_is": "../../data/dataset_main_SOEP_IS.dta"
}

# Set recurring parameters 
min_SRA = 65
min_ret_age = min_SRA - 4
exp_cap = 40 # maximum number of periods of exp accumulation
start_year = 2010 # start year of estimation sample
end_year = 2021 # end year of estimation sample

# Set options for data preparation
data_options = {
    "start_year": start_year,
    "end_year": end_year,
    "start_age": 25,
    "min_ret_age": min_ret_age,
    "exp_cap": exp_cap
}

# Set options for estimation of wage equation parameters
wage_eq_options = {
    "start_year": start_year,
    "end_year": end_year,
    "exp_cap": exp_cap,
    "wage_dist_truncation_percentiles": [0.01, 0.99]
}

# Set options for estimation of policy expectation process parameters
policy_expectation_options = {
    start_year: start_year
}

# %%
# Step 1: Get choice and state variables from SOEP core and SOEP RV VSKT
#----------------------------------------------------------------------------------------------

from src.gather_decision_data import gather_decision_data
dec_data = gather_decision_data(paths_dict, data_options, load_data=False)
# Add checks on our model choice sets. I.e. no retirement before min_ret_age,
# no working/unemployment after max_ret_age.
# Also our syear is from the soep. We could check, that single observations,
# have only two consecutive observations in soep.


# %%
# Step 2: Estimates wage equation parameters
#----------------------------------------------------------------------------------------------
from src.wage_equation import estimate_wage_parameters
wage_params = estimate_wage_parameters(paths_dict, wage_eq_options)

# %%
# Step 3: Estimates policy expectation process parameters
#----------------------------------------------------------------------------------------------
from src.policy_expectation_process import estimate_policy_expectation_parameters
policy_expectation_params = estimate_policy_expectation_parameters(paths_dict,_policy_expectation_options)

# %%