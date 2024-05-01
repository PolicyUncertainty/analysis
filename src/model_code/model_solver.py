import pickle

from dcegm.solve import get_solve_func_for_model
from model_code.specify_model import specify_model
from model_code.wealth_and_budget.savings_grid import create_savings_grid


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    update_spec_for_policy_state,
    policy_state_trans_func,
    load_model,
    load_solution,
):
    """Solve the model and save the solution as well as specifications to a file."""
    solution_file = path_dict["intermediate_data"] + (
        f"solved_models/model_solution" f"_{file_append}.pkl"
    )

    # Generate model_specs
    model, options, params = specify_model(
        path_dict=path_dict,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_trans_func,
        params=params,
        load_model=load_model,
    )

    if load_solution:
        solution_est = pickle.load(open(solution_file, "rb"))
        return solution_est, model, options, params

    savings_grid = create_savings_grid()

    solve_func = get_solve_func_for_model(model, savings_grid, options)
    value, policy, endog_grid = solve_func(params)

    solution = {
        "value": value,
        "policy": policy,
        "endog_grid": endog_grid,
    }

    pickle.dump(solution, open(solution_file, "wb"))

    return solution, model, options, params
