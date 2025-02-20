# Description: This file estimates the parameters of the partner transition matrix using the SOEP panel data.
# For each sex, education level and age bin (10 years), we estimate P(partner_state | lagged_partner_state) non-parametrically.
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
import statsmodels.api as sm
from export_results.figures.color_map import JET_COLOR_MAP
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from specs.derive_specs import read_and_derive_specs
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.discrete.discrete_model import Logit
from statsmodels.discrete.discrete_model import MNLogit

# import warnings
# warnings.filterwarnings("error")


def estimate_partner_transitions(paths_dict, specs, load_data):
    """Estimate the partner state transition matrix."""
    est_data = create_partner_transition_sample(paths_dict, specs, load_data=load_data)

    # Assume that everybody stays single with 76 onwards.
    est_data = est_data[est_data["age"] <= specs["end_age_transition_estimation"]]

    # Determine relevant ages
    all_ages = np.arange(specs["start_age"], specs["end_age"])
    est_ages = np.arange(specs["start_age"], specs["end_age_transition_estimation"] + 1)

    # Labels
    all_partner_labels = specs["partner_labels"]
    param_name_states = ["single", "working_age", "retirement"]

    full_index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            specs["education_labels"],
            all_ages,
            specs["partner_labels"],
            specs["partner_labels"],
        ],
        names=["sex", "education", "age", "partner_state", "lead_partner_state"],
    )
    full_df = pd.Series(index=full_index, data=np.nan, name="proportion")

    partner_state_vals = list(range(specs["n_partner_states"]))

    col_count = 0
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig, axs2 = plt.subplots(nrows=3, ncols=3)

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            df_reduced = est_data[
                (est_data["sex"] == sex_var) & (est_data["education"] == edu_var)
            ].copy()

            empirical_counts = df_reduced.groupby("age")["partner_state"].value_counts()
            mulitindex = pd.MultiIndex.from_product(
                [est_ages, partner_state_vals], names=["age", "partner_state"]
            )
            empirical_counts = empirical_counts.reindex(mulitindex, fill_value=0)
            n_obs_per_age = empirical_counts.groupby("age").transform("sum")
            empirical_shares = empirical_counts / n_obs_per_age

            initial_shares = empirical_shares.loc[
                (est_ages[0], partner_state_vals)
            ].values

            params_start = {}
            for current_state in param_name_states[:2]:
                for next_state in param_name_states[1:]:
                    params_start[f"const_{current_state}_to_{next_state}"] = 0
                    params_start[f"age_{current_state}_to_{next_state}"] = 0
                    if next_state == "working_age":
                        params_start[f"age_squared_{current_state}_to_{next_state}"] = 0
                        params_start[f"age_cubic_{current_state}_to_{next_state}"] = 0

            # params_start = pkl.load(open(f"result_{sex_label}_{edu_label}.pkl", "rb"))
            # for param_name in param_name_states[1:]:
            #     params_start.pop(f"const_retirement_to_{param_name}")
            #     params_start.pop(f"age_retirement_to_{param_name}")
            # params_result = params_start

            # Set upper bounds to 500 and lower bounds to -inf
            upper_bounds = {key: 5 for key in params_start.keys()}
            lower_bounds = {key: -5 for key in params_start.keys()}
            bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

            kwargs = {
                "age_grid": est_ages,
                "empirical_shares": empirical_shares,
                "start_shares": initial_shares,
                "weights": n_obs_per_age / n_obs_per_age.sum(),
            }

            result = om.minimize(
                fun=method_of_moments,
                params=params_start,
                fun_kwargs=kwargs,
                algorithm="scipy_neldermead",
                # algo_options={
                #     "n_cores": 7,
                # },
                bounds=bounds,
            )

            params_result = result.params

            pkl.dump(
                params_result,
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}.pkl",
                    "wb",
                ),
            )

            for age in est_ages:
                trans_mat = calc_trans_mat(params_result, age)
                for current_partner_state, partner_label in enumerate(
                    all_partner_labels
                ):
                    full_df.loc[
                        (sex_label, edu_label, age, partner_label, all_partner_labels)
                    ] = trans_mat[current_partner_state, :]

            for current_partner_state, partner_label in enumerate(all_partner_labels):
                for next_partner_state, next_partner_label in enumerate(
                    all_partner_labels
                ):
                    axs2[current_partner_state, next_partner_state].plot(
                        est_ages,
                        full_df.loc[
                            (
                                sex_label,
                                edu_label,
                                est_ages,
                                partner_label,
                                next_partner_label,
                            )
                        ],
                        label=f"{sex_label}; {edu_label}",
                        color=JET_COLOR_MAP[col_count],
                    )

            pred_shares = predicted_shares(
                params=params_result,
                est_ages=est_ages,
                start_shares=initial_shares,
            )

            ax = axs[col_count // 2, col_count % 2]
            for current_partner_state in partner_state_vals:
                ax.plot(
                    est_ages,
                    pred_shares.loc[(est_ages, current_partner_state)],
                    label=f"{current_partner_state}",
                    color=JET_COLOR_MAP[current_partner_state],
                )
                ax.plot(
                    est_ages,
                    empirical_shares.loc[(est_ages, current_partner_state)],
                    linestyle="--",
                    color=JET_COLOR_MAP[current_partner_state],
                    label=f"{current_partner_state}",
                )

            ax.legend()
            col_count += 1

    for age in range(specs["end_age_transition_estimation"] + 1, specs["end_age"]):
        for current_partner_state, partner_label in enumerate(all_partner_labels):
            full_df.loc[(slice(None), slice(None), age, slice(None), slice(None))] = 0
            # Assign 1 to diagonal
            full_df.loc[
                (slice(None), slice(None), age, partner_label, partner_label)
            ] = 1
    axs2[0, 0].legend()
    plt.show()
    out_file_path = paths_dict["est_results"] + "partner_transition_matrix.csv"
    full_df.to_csv(out_file_path)


def calc_trans_mat(params, age):
    """Compute the transition matrix for a given age using a multinomial logit
    specification."""

    above_40 = age > 40

    param_state_names = ["single", "working_age", "retirement"]
    age_sq = age**2 / 1_000
    age_cub = age**3 / 100_000

    trans_mat = np.zeros((3, 3), dtype=float)
    for i, current_state_name in enumerate(param_state_names):
        if current_state_name == "retirement":
            exp_vals = np.array([0, 0, 1], dtype=float)
        else:
            exp_vals = [1]
            for next_state_name in param_state_names[1:]:
                val = (
                    params[f"const_{current_state_name}_to_{next_state_name}"]
                    + params[f"age_{current_state_name}_to_{next_state_name}"] * age
                )
                if next_state_name == "working_age":
                    val += (
                        params[f"age_squared_{current_state_name}_to_{next_state_name}"]
                        * age_sq
                    )
                    val += (
                        params[f"age_cubic_{current_state_name}_to_{next_state_name}"]
                        * age_cub
                    )
                    exp_vals += [np.exp(val)]
                else:
                    exp_vals += [np.exp(val)]
            else:
                exp_vals[2] *= above_40
                exp_vals = np.array(exp_vals, dtype=float)

        # exp_vals = np.exp([
        #     1, params[f"{age_bin_name}_{current_state_name}_to_working_age"], params[f"{age_bin_name}_{current_state_name}_to_retirement"]
        # ])
        trans_mat[i, :] = exp_vals / exp_vals.sum()

    return trans_mat


def predicted_shares(params, est_ages, start_shares):
    """Compute the predicted shares over age using the Markov process."""
    n_states = start_shares.shape[0]

    index = pd.MultiIndex.from_product(
        [est_ages, range(n_states)],
        names=["age", "partner_state"],
    )
    shares = pd.Series(index=index, name="proportion", dtype=float, data=0)

    shares.loc[(est_ages[0], slice(None))] = start_shares

    for age in est_ages[1:]:
        trans_mat = calc_trans_mat(params, age - 1)
        shares.loc[(age, slice(None))] = shares.loc[(age - 1, slice(None))] @ trans_mat
    return shares


def method_of_moments(params, age_grid, empirical_shares, start_shares, weights):
    """Objective function for method of moments: minimize the squared difference."""
    predicted = predicted_shares(params, age_grid, start_shares)
    crit_val = np.sum(((predicted - empirical_shares) ** 2) * weights)
    print(crit_val)
    return crit_val


def estimate_nb_children(paths_dict, specs):
    """Estimate the number of children in the household for each individual conditional
    on sex, education and age bin."""
    # load data, filter, create period and has_partner state
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]
    end_age = specs["end_age"]

    df = df[df["age"] >= start_age]

    # Filter out individuals below 60 for better estimation(we should set this in specs)
    df = df[df["age"] <= 60]
    df["period"] = df["age"] - start_age
    df["period_sq"] = df["period"] ** 2
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    # estimate OLS for each combination of sex, education and has_partner

    edu_states = list(range(specs["n_education_types"]))
    sexes = [0, 1]
    partner_states = [0, 1]

    sub_group_names = ["sex", "education", "has_partner"]

    multiindex = pd.MultiIndex.from_product(
        [sexes, edu_states, partner_states],
        names=sub_group_names,
    )

    columns = ["const", "period", "period_sq"]
    estimates = pd.DataFrame(index=multiindex, columns=columns)
    for sex in sexes:
        for education in edu_states:
            for has_partner in partner_states:
                df_reduced = df[
                    (df["sex"] == sex)
                    & (df["education"] == education)
                    & (df["has_partner"] == has_partner)
                ]
                X = df_reduced[columns[1:]]
                X = sm.add_constant(X)
                Y = df_reduced["children"]
                model = sm.OLS(Y, X).fit()
                estimates.loc[(sex, education, has_partner), columns] = model.params

    out_file_path = paths_dict["est_results"] + "nb_children_estimates.csv"
    estimates.to_csv(out_file_path)
    # plot results
    return estimates
