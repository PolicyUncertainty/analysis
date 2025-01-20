import pandas as pd
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()

specs = generate_derived_and_data_derived_specs(path_dict)


data_sim = pd.read_pickle(
    path_dict["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
).reset_index()

from estimation.first_step_estimation.est_job_sep import est_job_for_sample

# restrict data to below max ret age, to lagged_choice ==
df_working = data_sim[
    (data_sim["age"] <= specs["max_ret_age"]) & (data_sim["lagged_choice"] == 3)
]
# Generate job_sep as job offer equal 0
df_working["job_sep"] = 1 - df_working["job_offer"]

job_sep_probs_sim, job_sep_params_sim = est_job_for_sample(df_working, specs)

# Estimate job offer probabilities for working agents
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    est_job_offer_params_full_obs,
)

df_working_age = data_sim[data_sim["age"] <= specs["max_ret_age"]]
df_working_age = df_working_age[df_working_age["age"] > 32]
job_offer_params_sim = est_job_offer_params_full_obs(
    df_working_age, specs, sex_append=["men"]
)
breakpoint()
