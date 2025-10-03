from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(path_dict["specs"])
save_tables = True

# SRA belief parameters table
from beliefs.sra_beliefs.params_table import create_sra_params_table_latex
create_sra_params_table_latex(path_dict, save=save_tables)

print("Table generation completed.")