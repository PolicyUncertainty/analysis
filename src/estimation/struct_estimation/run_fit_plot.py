# %% Set paths of project
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = specs["model_name"]
print(f"Running model: {model_name}")
load_sol_model = True
load_solution = None
load_data_from_sol = True

create_fit_plots(
    path_dict=path_dict,
    specs=specs,
    model_name=model_name,
    load_sol_model=load_sol_model,
    load_solution=load_solution,
    load_data_from_sol=load_data_from_sol,
)


# %%
