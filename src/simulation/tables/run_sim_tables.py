# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: This file has been moved from export_results to simulation. Check if still needed
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Set paths of project
import pickle

import pandas as pd
from set_paths import create_path_dict

path_dict = create_path_dict()
params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
from simulation.tables.effects_table import create_effects_table

sim_dir = path_dict["intermediate_data"] + "sim_data/"
###############################################
# Effects of uncertainty
###############################################
data_no_unc = pd.read_pickle(sim_dir + "data_real_scale_1.pkl")
data_unc = pd.read_pickle(sim_dir + "data_subj_scale_1.pkl")
create_effects_table(
    df_base=data_unc,
    df_cf=data_no_unc,
    params=params,
    path_dict=path_dict,
    scenario_name="no_uncertainty",
)
print("Uncertainty done")
###############################################
# Policy trajectories
###############################################
data_no_inc = pd.read_pickle(sim_dir + "data_no_increase.pkl")
data_005 = pd.read_pickle(sim_dir + "data_incr_005.pkl")
data_01 = pd.read_pickle(sim_dir + "data_incr_01.pkl")

create_effects_table(
    df_base=data_no_inc,
    df_cf=data_005,
    params=params,
    path_dict=path_dict,
    scenario_name="0_05_inc",
)
print("0.05 done")
create_effects_table(
    df_base=data_no_inc,
    df_cf=data_01,
    params=params,
    path_dict=path_dict,
    scenario_name="0_1_inc",
)

# from export_results.tables.cv import create_real_utility, create_disc_sum
# def print_av_ret_age(df):
#     print(
#         df[(df["choice"] == 2) & (df["lagged_choice"] != 2)]["age"].mean()
#     )
#     df["policy_state_val"] = df["policy_state"] * 0.25 + 65
#     print(
#         df[(df["choice"] == 2) & (df["lagged_choice"] != 2)][
#             "policy_state_val"
#         ].mean()
#     )
#
#
# print_av_ret_age(data_no_inc)
# # print_av_ret_age(data_005)
# # print_av_ret_age(data_01)
