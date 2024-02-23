import pickle

import estimagic as em
from dcegm.solve import get_solve_func_for_model
from model_code.budget_equation import create_savings_grid
from model_code.specify_model import specify_model


def solve_model(
    path_dict,
    file_append,
    params,
    update_spec_for_policy_state,
    policy_state_trans_func,
    load_model,
    load_solution,
):
    solution_file = path_dict["intermediate_data"] + f"model_solution_{file_append}.pkl"

    if load_solution:
        solution_est = pickle.load(open(solution_file, "rb"))
        return solution_est

    # Generate model_specs
    model, options, params = specify_model(
        path_dict=path_dict,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_trans_func,
        params=params,
        load_model=load_model,
    )
    savings_grid = create_savings_grid()

    solve_func = get_solve_func_for_model(model, savings_grid, options)
    value, policy_left, policy_right, endog_grid = solve_func(params)

    solution = {
        "value": value,
        "policy_left": policy_left,
        "policy_right": policy_right,
        "endog_grid": endog_grid,
    }

    pickle.dump(solution, open(solution_file, "wb"))

    return solution
