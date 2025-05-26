# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from model_code.stochastic_processes.job_offers import job_offer_process_transition
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
model_name = "wo_disability"


params_est = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_start = load_and_set_start_params(path_dict)
# params_start["job_finding_logit_const_men"] += 1

data_decision = pd.read_csv(path_dict["struct_est_sample"])
data_decision = data_decision[data_decision["lagged_choice"] == 1]
data_decision["work_start"] = data_decision["choice"].isin([2, 3]).astype(int)

data_decision["job_offer_prob_start"] = job_offer_process_transition(
    params=params_start,
    sex=data_decision["sex"].values,
    model_specs=specs,
    education=data_decision["education"].values,
    period=data_decision["period"].values,
    choice=data_decision["lagged_choice"].values,
)[1, :]

data_decision["job_offer_prob_est"] = job_offer_process_transition(
    params=params_est,
    sex=data_decision["sex"].values,
    model_specs=specs,
    education=data_decision["education"].values,
    period=data_decision["period"].values,
    choice=data_decision["lagged_choice"].values,
)[1, :]

fig, axs = plt.subplots(2, 2)
for sex_var, sex_label in enumerate(specs["sex_labels"]):
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        ax = axs[sex_var, edu_var]
        df_type = data_decision[
            (data_decision["sex"] == sex_var) & (data_decision["education"] == edu_var)
        ]
        df_type.groupby("period")["job_offer_prob_start"].mean().plot(
            ax=ax, label="Start", ls=":"
        )
        df_type.groupby("period")["job_offer_prob_est"].mean().plot(ax=ax, label="Est")
        df_type.groupby("period")["work_start"].value_counts(normalize=True).loc[
            (slice(None), 1)
        ].plot(ls="--", ax=ax, label="Observed")
        # Set title
        ax.set_title(f"{sex_label}; {edu_label}")

axs[0, 0].legend()
plt.show()
