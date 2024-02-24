# dependencies:
#   - original data saved on local machines,
#   - functions for steps 0-3 in src\steps folder
#   - src\spec.yaml
# output: data (decisions and outside_parameters) saved in output folder
# %%
# Step 0: Set paths and parameters
# ----------------------------------------------------------------------------------------------
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

paths_dict = create_path_dict(analysis_path, define_user=True)

# Specify load data from pickle files
custom_load_data = False
load_data_prompt = input("Load data from pickle files? (y/n/[c]ustom for every step): ")
if load_data_prompt == "y":
    LOAD_DATA = True
elif load_data_prompt == "n":
    LOAD_DATA = False
else:
    custom_load_data = True

# Load options and generate auxiliary options
from model_code.derive_specs import read_and_derive_specs


specs = read_and_derive_specs(paths_dict["specs"])

# %%
# Step 1: Get choice and state variables from policy_step_sizeSOEP core and SOEP RV VSKT
# ----------------------------------------------------------------------------------------------
from process_data.gather_decision_data import gather_decision_data

if custom_load_data:
    LOAD_DATA = input("Load choice & decision data from pickle files? (y/n): ") == "y"

gather_decision_data(
    paths_dict,
    specs,
    load_data=LOAD_DATA,
)

# %%
# Step 2: Prepare expectation data
# --------------------------------------------------------------------------------------
from process_data.est_SRA_expectations import estimate_truncated_normal

if custom_load_data:
    LOAD_DATA = (
        input("Load policy expectation parameters from pickle files? (y/n): ") == "y"
    )

estimate_truncated_normal(paths_dict, specs, load_data=LOAD_DATA)
