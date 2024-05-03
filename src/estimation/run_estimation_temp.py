#%% Set paths of project
import sys
from pathlib import Path
import pandas as pd

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

paths_dict = create_path_dict(analysis_path, define_user="y")


# Estimate wage parameter
from estimation.first_step_estimation.est_wage_equation_new import (
    estimate_wage_parameters,
)

wage_data = pd.read_pickle(paths_dict["intermediate_data"] + "wage_estimation_sample.pkl")
#breakpoint()
wage_parameters = estimate_wage_parameters(paths_dict, wage_data)
#breakpoint()

# %%
