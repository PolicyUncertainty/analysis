import os

import pandas as pd

from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

path_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(path_dict["specs"])

# create pension belief dataset
from beliefs.soep_is.process_soep_is import add_covariates, load_and_filter_soep_is

df = load_and_filter_soep_is(paths=path_dict)
df = add_covariates(df, paths=path_dict, specs=specs)
df.to_csv(path_dict["intermediate_data"] + "beliefs/" + "soep_is_clean.csv")

from beliefs.sra_beliefs.random_walk import est_SRA_params, est_alpha_heterogeneity

# estimate SRA belief parameters (fitting truncated normals takes a while, load_data=True to speed up)
from beliefs.sra_beliefs.truncated_normals import estimate_truncated_normal

df_truncated_normal = estimate_truncated_normal(
    df, paths=path_dict, options=specs, load_data=False
)
df_truncated_normal.to_csv(path_dict["intermediate_data"] + "beliefs/" + "soep_is_truncated_normals.csv")
sra_params_df = est_SRA_params(path_dict, df=df_truncated_normal, print_summary=True)
alpha_heterogeneity_df = est_alpha_heterogeneity(path_dict, df=df_truncated_normal)

# estimate ERP belief parameters (bootstrapping SE takes a while, calculate_se=False to speed up)
from beliefs.erp_beliefs.informed_state_transition import (
    calibrate_uninformed_hazard_rate_with_se,
)

uninformed_params_df = calibrate_uninformed_hazard_rate_with_se(
    df, specs, calculate_se=False
)

# save results
params_df = pd.concat([sra_params_df, uninformed_params_df], ignore_index=True)
params_df.to_csv(path_dict["intermediate_data"] + "beliefs/" + "beliefs_parameters.csv", index=False)

# save heterogeneity results
alpha_heterogeneity_df.to_csv(path_dict["intermediate_data"] + "beliefs/" + "alpha_heterogeneity_results.csv", index=False)