# %%
import os
import pickle as pkl

import pandas as pd

# paths and specs
from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)
import jax

jax.config.update("jax_enable_x64", True)

# sim tools
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from simulation.sim_tools.simulation_print import start_simulation_print
from simulation.tables.debias_table import (
    aggregate_comparison_baseline_cf,
)

# %%
# Set specifications
seeed = 123
model_name = specs["model_name"]
util_type = specs["util_type"]
sra_at_63 = 67.0

base_label = "Baseline with Misinformed"
cf_label = "Only Informed"

load_unc_model = True  # informed state as type
load_unc_solution = None  # baseline solution conntainer
model_solution = None  # actual baseline model solution object (None = create new)
load_df_baseline = None  # True = load existing df, False = create new df, None = create but do not save
load_df_unbiased = None  # same as above


# # Load params
# params = pkl.load(
#     open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
# )

# # Simulate baseline
# start_simulation_print(
#     model_name=model_name,
#     sra_63=sra_at_63,
#     uncertainty=True,
#     misinformation=True,
#     load_model=load_unc_model,
#     load_solution=load_unc_solution,
#     load_df=load_df_baseline,
# )

# df_base, _ = solve_and_simulate_scenario(
#     path_dict=path_dict,
#     params=params,
#     subj_unc=True,
#     custom_resolution_age=None,
#     announcement_age=None,
#     SRA_at_retirement=sra_at_63,
#     SRA_at_start=67,
#     model_name=model_name,
#     df_exists=load_df_baseline,
#     only_informed=False,
#     solution_exists=load_unc_solution,
#     sol_model_exists=load_unc_model,
#     model_solution=model_solution,
#     util_type=util_type,
# )
# df_base = df_base.reset_index()

save_folder = path_dict["sim_results"] + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# df_base_plot = df_base[(df_base["choice"] == 0) & (df_base["lagged_choice"] != 0)]
# df_base_plot.to_csv(save_folder + "df_bunching_base.csv", index=False)

# del df_base_plot
# del df_base

model_name_informed = model_name + "_inf"
params_informed = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name_informed}.pkl", "rb")
)

load_unc_solution_informed = None  # baseline solution conntainer
model_solution_informed = (
    None  # actual baseline model solution object (None = create new)
)
load_df_informed = None  # same as above

# Simulate counterfactual without misinformation
start_simulation_print(
    model_name=model_name_informed,
    sra_63=sra_at_63,
    uncertainty=True,
    misinformation=False,
    load_model=load_unc_model,
    load_solution=load_unc_solution_informed,
    load_df=load_df_informed,
)

df_cf, _ = solve_and_simulate_scenario(
    path_dict=path_dict,
    params=params_informed,
    subj_unc=True,
    custom_resolution_age=None,
    announcement_age=None,
    SRA_at_retirement=sra_at_63,
    SRA_at_start=67,
    model_name=model_name_informed,
    df_exists=load_df_informed,
    only_informed=True,
    solution_exists=load_unc_solution_informed,
    sol_model_exists=load_unc_model,
    model_solution=model_solution_informed,  # use same solution as baseline
    util_type=util_type,
)
df_cf = df_cf.reset_index()

df_cf_plot = df_cf[(df_cf["choice"] == 0) & (df_cf["lagged_choice"] != 0)]
df_cf_plot.to_csv(save_folder + "df_bunching_cf.csv", index=False)

# mask_func_m = lambda df_int: df_int["sex"] == 0
# mask_func_f = lambda df_int: df_int["sex"] == 1
# mask_func_all = lambda df_int: df_int["sex"].isin([0, 1])

# label_dict = {
#     "all": mask_func_all,
#     "men": mask_func_m,
#     "women": mask_func_f,
# }


# for group_label, mask_func in label_dict.items():
#     df_base_group = df_base[mask_func(df_base)].copy()
#     df_cf_group = df_cf[mask_func(df_cf)].copy()

#     # calculate aggregate and lifetime results
#     result_df = pd.DataFrame(index=[0])
#     for df, label in zip([df_base_group, df_cf_group], ["baseline", "cf"]):
#         result_df = add_overall_results(
#             result_df=result_df, df_scenario=df, index=0, pre_name=label, specs=specs
#         )

#     cv_cf = calc_compensated_variation(
#         df_base=df_base_group,
#         df_cf=df_cf_group,
#         params=params,
#         specs=specs,
#     )
#     result_df.at[0, "cv_cf"] = cv_cf

#     filename = f"debias_{group_label}.csv"
#     save_folder = path_dict["sim_results"] + model_name + "/"
#     result_df.to_csv(save_folder + filename)
#     print(f"Saved: {filename}")

#     # Create aggregate comparison table
#     aggregate_comparison_baseline_cf(
#         path_dict=path_dict,
#         model_name=model_name,
#         file_append=group_label,
#         result_df=result_df,
#     )

# df_lc_baseline = calc_life_cycle_detailed(df_base)
# df_lc_cf = calc_life_cycle_detailed(df_cf)

# plot_retirement_share(
#     path_dict=path_dict,
#     specs=specs,
#     df_base=df_base,
#     df_cf=df_cf,
#     final_SRA=sra_at_63,
#     model_name=model_name,
#     left_difference=-6,
#     right_difference=2,
#     base_label=base_label,
#     cf_label=cf_label,
# )

# plot_detailed_lifecycle_results(
#     path_dict=path_dict,
#     specs=specs,
#     model_name=model_name,
#     comparison_name="all_informed",
#     df_baseline=df_lc_baseline,
#     df_comparison=df_lc_cf,
#     show=False,
#     save=True,
#     end_age=80,
# )
