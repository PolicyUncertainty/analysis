import pandas as pd
from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs
path_dict = create_path_dict(define_user = True, user = "b")
specs = read_and_derive_specs(path_dict["specs"])

# create pension belief dataset
from beliefs.soep_is.process_soep_is import load_and_filter_soep_is, add_covariates
df = load_and_filter_soep_is(paths=path_dict)
df = add_covariates(df, paths=path_dict, specs=specs)

# estimate SRA belief parameters
from beliefs.sra_beliefs.truncated_normals import estimate_truncated_normal
from beliefs.sra_beliefs.random_walk import est_SRA_params
df_truncated_normal = estimate_truncated_normal(df, paths=path_dict, options=specs, load_data=True)
sra_params_df = est_SRA_params(path_dict, df=None)

# estimate ERP belief parameters
from beliefs.erp_beliefs.informed_state_transition import calibrate_uninformed_hazard_rate_with_se
uninformed_params_df = calibrate_uninformed_hazard_rate_with_se(df, specs)

# save results
params_df = pd.concat([sra_params_df, uninformed_params_df], ignore_index=True)
params_df.to_csv(path_dict["intermediate_data"] + "beliefs/beliefs_parameters.csv", index=False)