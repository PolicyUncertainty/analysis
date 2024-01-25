# Gathers everything necessary for structural model estimation.
# This script executes the following steps:
# 0. Set paths and parameters.
# 1. Estimates policy expectation process parameters.
# 2. Estimates wage equation parameters.
# 3. Get choice and other state variables (lagged choice, policy state, retirement age id, experience, wealth) from SOEP core and SOEP RV VSKT.

# locations:
# dependencies: original data saved on local machines, functions for steps 0-3 in src folder
# output: data (decisions and outside_parameters) saved in output folder

# %%
# Step 0: Set paths and parameters
# ----------------------------------------------------------------------------------------------
USER = "max"
LOAD_DATA = False  # if True, load data from pickle files instead of generating it

# Set data paths according to user.
if USER == "bruno":
    data_path = "C:/Users/bruno/papers/soep/"
elif USER == "max":
    data_path = "/home/maxbl/Uni/pol_uncetainty/data/"
else:
    raise ValueError("Please specify valid USER in "
                     "MASTER_prepare_structural_model.py.")

import os
analysis_path = os.getcwd() + "/"

# Set paths
paths_dict = {
    "soep_c38": data_path + "soep38",
    "soep_rv": data_path + "soep_rv",
    "soep_is": data_path + "soep_is_2022/dataset_main_SOEP_IS.dta",
    "project_path": analysis_path
}

# Set recurring parameters
min_SRA = 65
min_ret_age = min_SRA - 4
max_ret_age = 72
exp_cap = 40  # maximum number of periods of exp accumulation
start_year = 2010  # start year of estimation sample
end_year = 2021  # end year of estimation sample


options = {
    # Set options for estimation of policy expectation process parameters
    # limits for truncation of the normal distribution
    "lower_limit": 66.5,
    "upper_limit": 80,
    # points at which the CDF is evaluated from survey data
    "first_cdf_point": 67.5,
    "second_cdf_point": 68.5,
    # cohorts for which process parameters are estimated
    "min_birth_year": 1947,
    "max_birth_year": 2000,
    # lowest policy state
    "min_policy_state": 65,
    "start_age": 25,
    "min_ret_age": min_ret_age,
    "max_ret_age": max_ret_age,
    # Set options for estimation of wage equation parameters
    "start_year": start_year,
    "end_year": end_year,
    "exp_cap": exp_cap,
    "wage_dist_truncation_percentiles": [0.01, 0.99],
}


# %%
# Step 1: Estimates policy expectation process parameters
# ----------------------------------------------------------------------------------------------
from src.ret_age_expectations import estimate_policy_expectation_parameters

policy_expectation_params = estimate_policy_expectation_parameters(
    paths_dict, options, load_data=True
)

# %%
# Step 2: Estimates wage equation parameters
# ----------------------------------------------------------------------------------------------
from src.wage_equation import estimate_wage_parameters

wage_params = estimate_wage_parameters(paths_dict, options, load_data=LOAD_DATA)

# %%
# Step 3: Get choice and state variables from policy_step_sizeSOEP core and SOEP RV VSKT
# ----------------------------------------------------------------------------------------------
policy_step_size = policy_expectation_params.iloc[1, 0]

from src.gather_decision_data import gather_decision_data

dec_data = gather_decision_data(
    paths_dict,
    options,
    policy_step_size,
    load_data=LOAD_DATA,
)

# %%
