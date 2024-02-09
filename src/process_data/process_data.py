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
    raise ValueError(
        "Please specify valid USER in " "MASTER_prepare_structural_model.py."
    )

import os
import yaml

analysis_path = os.path.abspath(os.getcwd() + "/../../") + "/"

import sys

sys.path.insert(0, analysis_path + "src/")

# Set paths
paths_dict = {
    "soep_c38": data_path + "soep38",
    "soep_rv": data_path + "soep_rv",
    "soep_is": data_path + "soep_is_2022/dataset_main_SOEP_IS.dta",
    "project_path": analysis_path,
    "output_path": analysis_path + "output/",
}

# Load options and generate auxiliary options
from derive_specs import generate_derived_specs

project_specs = yaml.safe_load(open(analysis_path + "src/spec.yaml"))
project_specs = generate_derived_specs(project_specs)

# %%
# Step 1: Get choice and state variables from policy_step_sizeSOEP core and SOEP RV
# VSKT
# ----------------------------------------------------------------------------------------------
from process_data.steps.gather_decision_data import gather_decision_data

dec_data = gather_decision_data(
    paths_dict,
    project_specs,
    load_data=LOAD_DATA,
)

# %%


# %%
# Step 2: Estimates policy expectation process parameters
# Step 2a: Estimate for each individual truncated normals
# --------------------------------------------------------------------------------------
from process_data.steps.est_ret_age_expectations import estimate_truncated_normal

df_estimated_ret_age_expectations = estimate_truncated_normal(
    paths_dict, project_specs, load_data=True
)

# %%
#
# Step 2b: Estimate regressions and create plots
# ---------------------------------------------------------------------------------------
from process_data.steps.regression_and_plots import (
    gen_exp_val_params_and_plot,
    gen_var_params_and_plot,
)

policy_expectation_alpha = gen_exp_val_params_and_plot(
    paths=paths_dict, df=df_estimated_ret_age_expectations
)

policy_expectation_sigma_sq = gen_var_params_and_plot(
    paths=paths_dict, df=df_estimated_ret_age_expectations
)

# %%

# Step 3: Estimates wage equation parameters
# ----------------------------------------------------------------------------------------------
from process_data.steps.est_wage_equation import estimate_wage_parameters

wage_params = estimate_wage_parameters(paths_dict, project_specs, load_data=LOAD_DATA)
