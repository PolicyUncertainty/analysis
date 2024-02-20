import estimagic as em
import numpy as np
import yaml
from dcegm.likelihood import calc_choice_probs_for_observed_states
from dcegm.likelihood import create_individual_likelihood_function_for_model
from dcegm.likelihood import create_observed_choice_indexes
from dcegm.solve import get_solve_func_for_model
from derive_specs import generate_derived_and_data_derived_specs
from model_code.specify_model import specify_model


def process_data_and_model(
    data_decision, project_paths, start_params_all, load_model, output="likelihood"
):
    model, options = specify_model_and_options(
        project_paths=project_paths,
        start_params_all=start_params_all,
        load_model=load_model,
        step="estimation",
    )

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

    if output == "likelihood":
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
    elif output == "solved_model":
        solve_func = get_solve_func_for_model(model, savings_grid, options)
        value, policy_left, policy_right, endog_grid = solve_func(start_params_all)
        observed_state_choice_indexes = create_observed_choice_indexes(
            oberved_states_dict, model
        )
        choice_probs_observations = calc_choice_probs_for_observed_states(
            value_solved=value,
            endog_grid_solved=endog_grid,
            params=start_params_all,
            observed_states=oberved_states_dict,
            state_choice_indexes=observed_state_choice_indexes,
            oberseved_wealth=observed_wealth,
            choice_range=np.arange(options["model_params"]["n_choices"], dtype=int),
            compute_utility=model["model_funcs"]["compute_utility"],
        )
        return choice_probs_observations, value, policy_left, policy_right, endog_grid
    else:
        raise ValueError("Output must be either 'likelihood' or 'probabilities'")


def specify_model_and_options(project_paths, start_params_all, load_model, step):
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
        project_specs=project_specs,
        model_data_path=model_path,
        load_model=load_model,
        step=step,
    )
    print("Model specified.")
    return model, options


def visualize_em_database(db_path):
    fig = em.criterion_plot(db_path)
    fig.show()

    fig = em.params_plot(db_path)
    fig.show()
