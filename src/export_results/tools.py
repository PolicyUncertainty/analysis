import numpy as np


def create_realized_taste_shock(df, specs):
    df.loc[:, "real_taste_shock"] = np.nan
    for choice in range(specs["n_choices"]):
        df.loc[df["choice"] == choice, "real_taste_shock"] = df.loc[
            df["choice"] == choice, f"taste_shocks_{choice}"
        ]
    return df


def create_discounted_sum_utilities(df, beta, utility_col="real_util"):
    mean_utility = df.groupby("period")[utility_col].mean().sort_index().values

    max_period = df["period"].max()
    # reverse loop over range
    for i in range(max_period - 1, -1, -1):
        mean_utility[i] += mean_utility[i + 1] * beta

    return mean_utility


def create_step_function_values(specs, base_policy_state, plot_span):
    new_value_periods = specs["policy_step_periods"] + 1

    step_function_vals = np.zeros(plot_span) + base_policy_state
    for i in range(1, plot_span):
        if np.isin(i, new_value_periods):
            step_function_vals[i] = step_function_vals[i - 1] + 1
        else:
            step_function_vals[i] = step_function_vals[i - 1]

    step_function_vals = step_function_vals * specs["SRA_grid_size"] + specs["min_SRA"]
    return step_function_vals
