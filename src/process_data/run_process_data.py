# %%
# set paths and parameters
# --------------------------------------------------------------------------------------
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

paths_dict = create_path_dict(analysis_path, define_user=True)

# Load options and generate auxiliary options
from model_code.derive_specs import read_and_derive_specs


specs = read_and_derive_specs(paths_dict["specs"])

# %%
# process SOEP core data, generate decision data and wage data
# --------------------------------------------------------------------------------------
from process_data.process_soep_core import process_soep_core
from process_data.derive_datasets import gather_decision_data
from process_data.derive_datasets import gather_wage_data

LOAD_DATA = False	
soep_core_df = process_soep_core(paths_dict, specs, load_data=LOAD_DATA)
gather_decision_data(paths_dict, df=soep_core_df, load_data=LOAD_DATA)
gather_wage_data(paths_dict, df=soep_core_df, load_data=LOAD_DATA)

# %%
# process SOEP IS, generate SRA data
# --------------------------------------------------------------------------------------
#from process_data.process_soep_is import process_soep_is

