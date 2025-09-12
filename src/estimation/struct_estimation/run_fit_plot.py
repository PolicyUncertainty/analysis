# %% Set paths of project
import pickle as pkl

from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = specs["model_name"]
print(f"Running model: {model_name}")
load_sol_model = True
load_solution = None
load_data_from_sol = True

# Load start params
start_params_all = load_and_set_start_params(path_dict)

# Low edu men
start_params_all["disutil_unemployed_low_good_men"] = 0.3982895233324046
start_params_all["disutil_unemployed_low_bad_men"] = 0.35630092790781426
start_params_all["disutil_ft_work_low_good_men"] = 0.6491840265554566
start_params_all["disutil_ft_work_low_bad_men"] = 0.945899297378813
start_params_all["disutil_partner_low_retired_men"] = -0.14767339974551175
start_params_all["SRA_firing_logit_intercept_men_low"] = 4.558180110702362


create_fit_plots(
    path_dict=path_dict,
    specs=specs,
    params=start_params_all,
    model_name=model_name,
    load_sol_model=load_sol_model,
    load_solution=load_solution,
    load_data_from_sol=load_data_from_sol,
    sex_type="men",
    edu_type="low",
)


# %%
