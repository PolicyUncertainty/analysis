from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
save_tables = True

# Dataset description table
from process_data.tables.structural_estimation_sample import create_dataset_description_table_latex
create_dataset_description_table_latex(path_dict, specs, save=save_tables)

print("Table generation completed.")