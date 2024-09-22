# Set paths of project
import pickle

import pandas as pd
from set_paths import create_path_dict

path_dict = create_path_dict()
params = pickle.load(open(path_dict["est_results"] + "est_params_all.pkl", "rb"))

from export_results.tables.effects_table import create_effects_table

data_no_unc = pd.read_pickle(
    path_dict["intermediate_data"] + "sim_data/data_real_scale_1.pkl"
)
data_unc = pd.read_pickle(
    path_dict["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
)
create_effects_table(
    df_base=data_no_unc,
    df_cf=data_unc,
    params=params,
    path_dict=path_dict,
    scenario_name="no_uncertainty",
)
