import os

import numpy as np
import pandas as pd

analysis_path = os.path.abspath(os.getcwd() + "/../../") + "/"

import sys
import yaml

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")


data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")

# Retirees don't have any choice and therefore no information
data_decision = data_decision[data_decision["lagged_choice"] != 2]

# Load data specs
from derive_specs import generate_derived_and_data_derived_options

project_paths = {
    "project_path": analysis_path,
}
project_specs = yaml.safe_load(open(analysis_path + "src/spec.yaml"))
project_specs = generate_derived_and_data_derived_options(
    project_specs, project_paths, load_data=True
)

from model_code.specify_model import specify_model

model, start_params, options = specify_model(project_specs)

# Create dummy exog column to handle in the model
data_decision["dummy_exog"] = np.zeros(len(data_decision), dtype=np.int8)
oberved_states_dict = {
    name: data_decision[name].values for name in model["state_space_names"]
}

observed_wealth = data_decision["wealth"].values
observed_choices = data_decision["choice"].values

savings_grid = np.arange(start=0, stop=100, step=0.5)

from dcegm.likelihood import create_individual_likelihood_function_for_model

individual_likelihood = create_individual_likelihood_function_for_model(
    model=model,
    options=options,
    observed_states=oberved_states_dict,
    observed_wealth=observed_wealth,
    observed_choices=observed_choices,
    exog_savings_grid=savings_grid,
    params_all=start_params,
)
