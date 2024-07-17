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
string_in = input(
    """Which dataset should be created? \n\n- all (a)\n- structural (s)\n- wage (w)\n"""
    """- job separation (j)\n Please write the according letter:"""
)


# %%
# Create relevant datasets. Ask for first if all schould be created or just some
# --------------------------------------------------------------------------------------
from process_data.create_structural_est_sample import create_structural_est_sample
from process_data.create_wage_est_sample import create_wage_est_sample
from process_data.create_job_sep_sample import create_job_sep_sample

if string_in == "a" or string_in == "s":
    create_structural_est_sample(paths_dict, options=specs, load_data=False)
if string_in == "a" or string_in == "w":
    create_wage_est_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "j":
    create_job_sep_sample(paths_dict, specs=specs, load_data=False)
