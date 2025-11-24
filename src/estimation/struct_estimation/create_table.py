from estimation.struct_estimation.scripts.param_table import create_latex_tables
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

create_latex_tables(path_dict, specs["model_name"])
