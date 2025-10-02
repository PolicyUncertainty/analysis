import numpy as np

from model_code.specify_model import specify_and_solve_model
from simulation.sim_tools.simulate_scenario import create_additional_variables
from specs.derive_specs import (
    generate_derived_and_data_derived_specs,
)


def simulate_exp(
    initial_state,
    n_multiply,
    path_dict,
    params,
    subj_unc,
    model_name,
    custom_resolution_age=None,
    solution_exists=True,
    sol_model_exists=True,
    model_solution=None,
    util_type="add",
):

    if model_solution is None:
        model_solved = specify_and_solve_model(
            path_dict=path_dict,
            params=params,
            file_append=model_name,
            custom_resolution_age=custom_resolution_age,
            subj_unc=subj_unc,
            load_model=sol_model_exists,
            load_solution=solution_exists,
            sim_specs=None,
            sex_type="all",
            edu_type="all",
            util_type=util_type
        )
    else:
        model_solved = model_solution
    initial_states = {
        key: np.ones(n_multiply) * value for key, value in initial_state.items()
    }
    specs = generate_derived_and_data_derived_specs(path_dict)

    df = model_solved.simulate(
        states_initial=initial_states,
        seed=specs["seed"],
    )
    # Kick out dead people
    df = df[df["health"] != 3].copy()
    # Reset index to avoid issues
    df = df.reset_index()
    df = create_additional_variables(df, specs)
    return df, model_solved
