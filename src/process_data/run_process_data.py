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
from process_data.create_structural_est_sample import create_structural_est_sample
from process_data.create_wage_est_sample import create_wage_est_sample

LOAD_DATA = False
structural_est_df = create_structural_est_sample(
    paths_dict, load_data=LOAD_DATA, options=specs
)
wage_est_df = create_wage_est_sample(paths_dict, load_data=LOAD_DATA, options=specs)

# %%
# process SOEP IS, generate SRA data
# --------------------------------------------------------------------------------------
# from process_data.process_soep_is import process_soep_is
