# %%
# Set paths of project

from set_paths import create_path_dict

paths_dict = create_path_dict()

import pickle
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import yaml

jax.config.update("jax_enable_x64", True)

from set_paths import create_path_dict

path_dict = create_path_dict()
from estimation.estimate_setup import create_job_offer_params_from_start

# %%
params = yaml.safe_load(open(path_dict["start_params"], "rb"))

job_sep_params = create_job_offer_params_from_start(path_dict)
params.update(job_sep_params)
# params["dis_util_work"] = 1.4911161193847658e-09
# params["dis_util_unemployed"] = 50


from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from model_code.specify_model import specify_model

# Generate model_specs
model, params = specify_model(
    path_dict=paths_dict,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    params=params,
    load_model=True,
)


from estimation.estimate_setup import load_and_prep_data
data_decision = load_and_prep_data(paths_dict)

from estimation.estimate_setup import create_ll_from_paths
individual_likelihood = create_ll_from_paths(
    params, paths_dict, load_model=True
)
ll_value, ll_contribution = individual_likelihood(params)
data_decision["ll_contribution"] = -ll_contribution
from model_code.utility_functions import utility_func
breakpoint()