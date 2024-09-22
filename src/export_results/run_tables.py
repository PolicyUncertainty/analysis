# Set paths of project
import pickle

import pandas as pd
from set_paths import create_path_dict

path_dict = create_path_dict()
params = pickle.load(open(path_dict["est_results"] + "est_params_all.pkl", "rb"))

from export_results.tables.effects_table import create_effects_table

sim_dir = path_dict["intermediate_data"] + "sim_data/"
###############################################
# Effects of uncertainty
###############################################
data_no_unc = pd.read_pickle(sim_dir + "data_real_scale_1.pkl")
data_unc = pd.read_pickle(sim_dir + "data_subj_scale_1.pkl")
create_effects_table(
    df_base=data_no_unc,
    df_cf=data_unc,
    params=params,
    path_dict=path_dict,
    scenario_name="no_uncertainty",
)

###############################################
# Policy trajectories
###############################################
data_no_inc = pd.read_pickle(sim_dir + "data_no_increase.pkl")
data_005 = pd.read_pickle(sim_dir + "data_incr_005.pkl")
create_effects_table(
    df_base=data_no_inc,
    df_cf=data_005,
    params=params,
    path_dict=path_dict,
    scenario_name="0_05_inc",
)
data_005 = pd.read_pickle(sim_dir + "data_incr_05.pkl")
create_effects_table(
    df_base=data_no_inc,
    df_cf=data_005,
    params=params,
    path_dict=path_dict,
    scenario_name="0_5_inc",
)
