from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs
path_dict = create_path_dict(define_user = True, user = "b")
specs = read_and_derive_specs(path_dict["specs"])

# create pension belief dataset
from beliefs.process_soep_is import load_and_filter_soep_is, add_covariates
df = load_and_filter_soep_is(path_dict)
df = add_covariates(df, path_dict)

# estimate SRA belief parameters
from analysis.src.beliefs.sra_beliefs.truncated_normals import estimate_truncated_normal
from analysis.src.beliefs.sra_beliefs.random_walk import est_SRA_params
df_truncated_normal = estimate_truncated_normal(df, paths=path_dict, options=specs)
sra_params_df = est_SRA_params(path_dict, df=None)

# estimate ERP belief parameters


