import numpy as np

from model_code.specify_model import specify_and_solve_model
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
):

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
    )
    initial_states = {
        key: np.ones(n_multiply) * value for key, value in initial_state.items()
    }
    specs = generate_derived_and_data_derived_specs(path_dict)

    df = model_solved.simulate(
        states_initial=initial_states,
        seed=specs["seed"],
    )
    return df
