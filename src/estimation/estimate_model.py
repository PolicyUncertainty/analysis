import os

import numpy as np
import pandas as pd

analysis_path = os.path.abspath(os.getcwd() + "/../../") + "/"

import sys

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")


data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")


# Filters and datatypes changes which should be done in pre cleaning
data_decision.rename(columns={"w011ha": "wealth"}, inplace=True)
data_decision = data_decision[data_decision["wealth"].notna()]
data_decision[data_decision["wealth"] < 0] = 0

# We count in the model in 1000s of euros
data_decision["wealth"] = data_decision["wealth"] / 1000

# Retirees don't have any choice
data_decision = data_decision[data_decision["lagged_choice"] != 2]

data_decision = data_decision.astype(
    {
        "choice": "int8",
        "lagged_choice": "int8",
        "policy_state": "int8",
        "retirement_age_id": "int8",
        "experience": "int8",
        "wealth": "float32",
        "period": "int8",
    }
)

from model_code.specify_model import specify_model

model, start_params, options = specify_model()

from dcegm.likelihood import create_individual_likelihood_function_for_model

# Create dummy exog column to handle in the model
data_decision["dummy_exog"] = np.zeros(len(data_decision), dtype=np.int8)
oberved_states_dict = {
    name: data_decision[name].values for name in model["state_space_names"]
}

observed_wealth = data_decision["wealth"].values
observed_choices = data_decision["choice"].values

savings_grid = np.arange(start=0, stop=100, step=0.5)

individual_likelihood = create_individual_likelihood_function_for_model(
    model=model,
    options=options,
    observed_states=oberved_states_dict,
    observed_wealth=observed_wealth,
    observed_choices=observed_choices,
    exog_savings_grid=savings_grid,
    params_all=start_params,
)
