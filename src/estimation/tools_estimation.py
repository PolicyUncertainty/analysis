import estimagic as em
import numpy as np
import yaml
from dcegm.likelihood import create_individual_likelihood_function_for_model
from derive_specs import generate_derived_and_data_derived_specs
from model_code.specify_model import specify_model


def prepare_estimation(data_decision, project_paths, start_params_all, load_model):
    analysis_path = project_paths["project_path"]
    model_path = project_paths["model_path"]

    # Generate model_specs
    project_specs = yaml.safe_load(open(analysis_path + "src/spec.yaml"))
    project_specs = generate_derived_and_data_derived_specs(
        project_specs, project_paths, load_data=True
    )
    # Assign income shock scale to start_params_all
    start_params_all["sigma"] = project_specs["income_shock_scale"]

    # Generate dcegm model for project specs
    model, options = specify_model(
        project_specs=project_specs, model_data_path=model_path, load_model=load_model
    )
    print("Model specified.")
    # Prepare data for estimation with information from dcegm model. Retirees don't
    # have any choice and therefore no information
    data_decision = data_decision[data_decision["lagged_choice"] != 2]
    # Now transform for dcegm
    oberved_states_dict = {
        name: data_decision[name].values for name in model["state_space_names"]
    }
    observed_wealth = data_decision["wealth"].values
    observed_choices = data_decision["choice"].values

    # Specifiy savings wealth grid
    savings_grid = np.arange(start=0, stop=100, step=0.5)

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        options=options,
        observed_states=oberved_states_dict,
        observed_wealth=observed_wealth,
        observed_choices=observed_choices,
        exog_savings_grid=savings_grid,
        params_all=start_params_all,
    )
    return individual_likelihood


def visualize_em_database(db_path):
    fig = em.criterion_plot(db_path)
    fig.show()

    fig = em.params_plot(db_path)
    fig.show()
