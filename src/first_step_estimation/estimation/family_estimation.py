# Description: This file contains estimation functions for family transitions.
# For each sex, education level and age bin (10 years), we estimate P(partner_state | lagged_partner_state) non-parametrically.
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd

from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)


def estimate_partner_transitions(paths_dict, specs, load_data=True):
    """Two-step estimation: first single/working_age, then working_age/retirement."""
    est_data = create_partner_transition_sample(paths_dict, specs, load_data=load_data)
    est_data = est_data[
        (est_data["age"] >= specs["start_age"]) & (est_data["age"] <= specs["end_age"])
    ]

    all_ages = np.arange(specs["start_age"], specs["end_age"])
    old_ages = np.arange(specs["end_age_transition_estimation"] + 1, specs["end_age"])

    param_name_states = ["single", "working_age", "retirement"]
    partner_state_vals = list(range(specs["n_partner_states"]))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            df = est_data[
                (est_data["sex"] == sex_var) & (est_data["education"] == edu_var)
            ].copy()

            # Calculate empirical shares
            empirical_counts = df.groupby("age")["partner_state"].value_counts()
            mulitindex = pd.MultiIndex.from_product(
                [all_ages, [0, 1, 2]], names=["age", "partner_state"]
            )
            empirical_counts = empirical_counts.reindex(mulitindex, fill_value=0)
            n_obs = empirical_counts.groupby("age").transform("sum")
            empirical_shares = empirical_counts / n_obs
            # We manipulate the empirical shares and assign to all ages above end_age_transition_estimation
            # and older the empirical single share of the end_age_transition_estimation. Marriage shares,
            # we set to zero and put the whole share to retirement.
            empirical_shares.loc[(old_ages, 0)] = empirical_shares.loc[
                (specs["end_age_transition_estimation"], 0)
            ]
            empirical_shares.loc[(old_ages, 1)] = 0.0
            empirical_shares.loc[(old_ages, 2)] = (
                1 - empirical_shares.loc[(specs["end_age_transition_estimation"], 0)]
            )

            initial_shares = empirical_shares.loc[
                (all_ages[0], partner_state_vals)
            ].values
            # Step 1: Estimate single/working_age transition
            params_step1 = estimate_single_working_age(df)

            # Step 2: Estimate working_age/retirement transition using full likelihood
            params_step2 = estimate_working_retirement_full(
                df, empirical_shares, params_step1
            )

            # Combine parameters
            params_combined = {**params_step1, **params_step2}

            print(calc_exp_ret_age_difference(params_combined, specs, initial_shares))
            print(params_combined)

            plot_trans_probs(
                all_ages,
                sra=67,
                params=params_combined,
                param_state_names=param_name_states,
            )

            sra_weights = df.groupby("age")["SRA"].value_counts(normalize=True)
            unique_sras = df["SRA"].unique()
            sra_weights_dict = {
                sra: sra_weights.xs(sra, level="SRA")
                .reindex(all_ages, fill_value=0)
                .values
                for sra in unique_sras
            }
            pred_shares = predicted_shares_for_sample(
                params_combined, all_ages, initial_shares, sra_weights_dict
            )
            plot_predicted_vs_empirical_shares(all_ages, empirical_shares, pred_shares)

            pkl.dump(
                params_combined,
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}.pkl",
                    "wb",
                ),
            )


def estimate_single_working_age(df):
    """Step 1: Estimate transitions between single and working_age."""
    df_est = df[df["age"] < 75].copy()
    df_est.loc[df_est["partner_state"] == 2, "partner_state"] = 1

    empirical_shares = df_est.groupby("age")["partner_state"].value_counts(
        normalize=True
    )
    df_est = df_est[df_est["age"] > 30]

    params_start = {
        "const_single_to_working_age": 0.0,
        "age_single_to_working_age": 0.0,
        "age_squared_single_to_working_age": 0.0,
        "age_cubic_single_to_working_age": 0.0,
        "const_working_age_to_working_age": 0.0,
        "age_working_age_to_working_age": 0.0,
        "age_squared_working_age_to_working_age": 0.0,
        "age_cubic_working_age_to_working_age": 0.0,
    }

    def ll_function(params):
        df_int = df_est.copy()

        trans_probs = calc_trans_mat_step1(
            params=params,
            age=(df_int["age"].values - 1).astype(float),
        )

        df_int["prob_to_observe"] = 0.0

        for current_state in [0, 1]:
            mask = df_int["partner_state"] == current_state
            if mask.sum() > 0:
                age_last_period = df_int.loc[mask, "age"].values - 1

                for state_last_period in [0, 1]:
                    shares_possible_last_period = empirical_shares.xs(
                        state_last_period, level="partner_state"
                    )
                    shares_last_period = shares_possible_last_period.loc[
                        age_last_period
                    ].values

                    df_int.loc[mask, "prob_to_observe"] += (
                        trans_probs[mask.values, state_last_period, current_state]
                        * shares_last_period
                    )

        log_lik = (
            np.log(np.maximum(df_int["prob_to_observe"].values, 1e-10))
            / df_int.shape[0]
        )
        ll_val = -np.sum(log_lik)
        print(ll_val)
        return ll_val

    bounds = om.Bounds(
        lower={k: -20 for k in params_start}, upper={k: 20 for k in params_start}
    )

    result = om.minimize(
        fun=ll_function,
        params=params_start,
        algorithm="scipy_bfgs",
        bounds=bounds,
    )
    print(result)

    return result.params


def estimate_working_retirement_full(df, empirical_shares, params_step1):
    """Step 2: Estimate working_age/retirement using full 3-state likelihood with fixed params from step 1."""
    param_name_states = ["single", "working_age", "retirement"]

    df_est = df.copy()
    df_est = df_est[df_est["age"] < 75]
    df_est = df_est[df_est["age"] > 30]
    df_est = df_est[~((df_est["partner_state"] == 2) & (df_est["age"] <= 40))]

    params_start = {
        "const_working_age_to_retirement": 0.0,
        "age_working_age_to_retirement": 0.0,
        "age_squared_working_age_to_retirement": 0.0,
        "age_cubic_working_age_to_retirement": 0.0,
        "SRA_age_diff_effect_working_age_to_retirement": 0.0,
    }

    def ll_function(params):
        # Combine fixed params from step 1 with current retirement params
        params_full = {**params_step1, **params}

        df_int = df_est.copy()
        df_int["prob_to_observe"] = 0.0

        trans_probs = calc_trans_mat_vectorized(
            params=params_full,
            age=(df_int["age"].values - 1).astype(float),
            sra=df_int["SRA"].values.astype(float),
        )
        trans_probs = np.asarray(trans_probs)

        for current_state_var, current_state_label in enumerate(param_name_states):
            mask = df_int["partner_state"] == current_state_var
            if mask.sum() > 0:
                age_last_period = df_int.loc[mask, "age"].values - 1
                for state_var_last_period, state_label_last_period in enumerate(
                    param_name_states
                ):
                    shares_possible_last_period = empirical_shares.loc[
                        (slice(None), state_var_last_period)
                    ]
                    shares_last_period = shares_possible_last_period.loc[
                        age_last_period
                    ].values

                    df_int.loc[mask, "prob_to_observe"] += (
                        trans_probs[
                            mask.values,
                            state_var_last_period,
                            current_state_var,
                        ]
                        * shares_last_period
                    )

        log_lik = (
            np.log(np.maximum(df_int["prob_to_observe"].values, 1e-10))
            / df_int.shape[0]
        )
        ll_val = -np.sum(log_lik)
        print(ll_val)
        return ll_val

    bounds = om.Bounds(
        lower={k: -20 for k in params_start}, upper={k: 20 for k in params_start}
    )

    result = om.minimize(
        fun=ll_function,
        params=params_start,
        algorithm="scipy_bfgs",
        bounds=bounds,
    )
    print(result)

    return result.params


def calc_trans_mat_step1(params, age):
    """Compute transition matrix for single/working_age (2x2)."""
    val_0_1 = exp_val_single_to_working_age(params, age)
    val_1_1 = exp_val_working_age_to_working_age(params, age)

    zeros = np.zeros_like(age)

    first_row = np.array([zeros, val_0_1]).T
    max_values = np.nanmax(first_row, axis=1, keepdims=True)
    first_row = np.exp(first_row - max_values)

    second_row = np.array([zeros, val_1_1]).T
    max_values = np.nanmax(second_row, axis=1, keepdims=True)
    second_row = np.exp(second_row - max_values)

    exp_vals = np.stack([first_row, second_row], axis=1)
    trans_mat = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    trans_mat_absorbing = np.zeros_like(trans_mat)
    trans_mat_absorbing[:, 0, 0] = 1.0
    trans_mat_absorbing[:, 1, 1] = 1.0
    above_75 = age >= 75
    trans_mat = (1 - above_75)[:, None, None] * trans_mat + above_75[
        :, None, None
    ] * trans_mat_absorbing

    return trans_mat


def calc_trans_mat_vectorized(params, age, sra):
    """Compute transition matrices for a particular age and sra."""
    val_0_1 = exp_val_single_to_working_age(params, age)
    val_1_1 = exp_val_working_age_to_working_age(params, age)
    val_1_2 = exp_val_working_age_to_retirement(params, age, sra)

    ones = np.ones_like(age)
    zeros = np.zeros_like(age)

    val_0_2 = np.nan * ones
    val_1_2 = np.where(age > 40, val_1_2, np.nan)

    # First row: single can only transition to working_age or stay single
    first_row = np.array([zeros, val_0_1, val_0_2]).T
    max_values = np.nanmax(first_row, axis=1, keepdims=True)
    first_row = np.exp(first_row - max_values)
    first_row_normalized = first_row / np.nansum(first_row, axis=-1, keepdims=True)
    first_row_normalized = np.nan_to_num(first_row_normalized, nan=0.0)

    # Second row: normalize single transition separately from working_age/retirement
    # First get probability to single from step 1 parameters only
    second_row_to_single = np.array([zeros, val_1_1]).T
    max_values_step1 = np.nanmax(second_row_to_single, axis=1, keepdims=True)
    second_row_to_single = np.exp(second_row_to_single - max_values_step1)
    step1_probs = second_row_to_single / np.sum(
        second_row_to_single, axis=-1, keepdims=True
    )
    prob_to_single = step1_probs[:, 0]
    prob_to_partnered = 1 - prob_to_single

    # Now split the partnered probability between working_age and retirement
    partnered_transition = np.array([val_1_1, val_1_2]).T
    max_values_partnered = np.nanmax(partnered_transition, axis=1, keepdims=True)
    partnered_transition = np.exp(partnered_transition - max_values_partnered)
    partnered_probs = partnered_transition / np.nansum(
        partnered_transition, axis=-1, keepdims=True
    )
    partnered_probs = np.nan_to_num(partnered_probs, nan=0.0)

    second_row_normalized = np.column_stack(
        [
            prob_to_single,
            prob_to_partnered * partnered_probs[:, 0],  # to working_age
            prob_to_partnered * partnered_probs[:, 1],  # to retirement
        ]
    )

    third_row_normalized = np.column_stack([zeros, zeros, ones])

    trans_mat = np.stack(
        [first_row_normalized, second_row_normalized, third_row_normalized], axis=1
    )

    trans_mat_absorbing = np.zeros_like(trans_mat)
    trans_mat_absorbing[:, 0, 0] = 1.0
    trans_mat_absorbing[:, 1, 2] = 1.0
    trans_mat_absorbing[:, 2, 2] = 1.0
    above_75 = age >= 75
    trans_mat = (1 - above_75)[:, None, None] * trans_mat + above_75[
        :, None, None
    ] * trans_mat_absorbing

    return trans_mat


def exp_val_working_age_to_retirement(params, age, sra):
    val = (
        params["const_working_age_to_retirement"]
        + params["age_working_age_to_retirement"] * age
        + params["age_squared_working_age_to_retirement"] * (age**2 / 100)
        + params["age_cubic_working_age_to_retirement"] * (age**3 / 100_000)
        + params["SRA_age_diff_effect_working_age_to_retirement"]
        * (sra - age)
        * (age > 50)
    )
    return val


def exp_val_working_age_to_working_age(params, age):
    val = (
        params["const_working_age_to_working_age"]
        + params["age_working_age_to_working_age"] * age
        + params["age_squared_working_age_to_working_age"] * (age**2 / 100)
        + params["age_cubic_working_age_to_working_age"] * (age**3 / 100_000)
    )
    return val


def exp_val_single_to_working_age(params, age):
    val = (
        params["const_single_to_working_age"]
        + params["age_single_to_working_age"] * age
        + params["age_squared_single_to_working_age"] * (age**2 / 100)
        + params["age_cubic_single_to_working_age"] * (age**3 / 100_000)
    )
    return val


def plot_trans_probs(ages, sra, params, param_state_names):
    trans_probs = calc_trans_mat_vectorized(
        params=params,
        age=ages,
        sra=sra,
    )

    n_states = len(param_state_names)
    fig, axs = plt.subplots(n_states, n_states)
    for current_state, current_state_label in enumerate(param_state_names):
        axs[current_state, 0].set_ylabel(f"Prob. from {current_state_label}")
        for next_state, next_state_label in enumerate(param_state_names):
            axs[current_state, next_state].plot(
                ages, trans_probs[:, current_state, next_state]
            )

    for next_state, next_state_label in enumerate(param_state_names):
        axs[-1, next_state].set_xlabel(f"Age")
        axs[0, next_state].set_title(f"to {next_state_label}")


def calc_exp_ret_age_difference(params, specs, initial_shares):
    all_ages = np.arange(specs["start_age"], specs["end_age"])
    pred_shares_67 = predict_shares_for_sra(params, all_ages, 67, initial_shares)
    pred_shares_68 = predict_shares_for_sra(params, all_ages, 68, initial_shares)
    ret_age_67 = calc_exp_ret_age(pred_shares_67, all_ages)
    ret_age_68 = calc_exp_ret_age(pred_shares_68, all_ages)
    return ret_age_68 - ret_age_67


def calc_exp_ret_age(pred_shares, all_ages):
    partnered_shares = pred_shares[:, 1:]
    # Normalize to 1
    partnered_shares = partnered_shares / partnered_shares.sum(axis=1, keepdims=True)
    ret_age = np.sum(
        (partnered_shares[1:, 1] - partnered_shares[:-1, 1]) * all_ages[1:]
    )
    return ret_age


def plot_predicted_vs_empirical_shares(all_ages, empirical_shares, pred_shares):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].plot(
            all_ages, empirical_shares.xs(i, level="partner_state"), label="Empirical"
        )
        axs[i].plot(all_ages, pred_shares[:, i], label="Predicted")
        axs[i].set_title(f"Partner State {i}")
        axs[i].legend()
        axs[i].set_ylim([0, 1])
    # plt.show()


def predicted_shares_for_sample(params, est_ages, start_shares, sra_weights):
    """Compute the predicted shares over age using the Markov process."""

    final_shares = np.zeros((len(est_ages), len(start_shares)))
    for sra in sra_weights.keys():
        shares_sra = (
            predict_shares_for_sra(params, est_ages, sra, start_shares)
            * sra_weights[sra][:, np.newaxis]
        )
        final_shares += shares_sra
    return final_shares


def predict_shares_for_sra(params, ages, sra, start_shares):
    """Compute the predicted shares over age using the Markov process."""

    n_ages = len(ages)
    n_states = len(start_shares)
    final_shares = np.zeros((n_ages, n_states))
    final_shares[0, :] = start_shares

    trans_mat_all_ages = calc_trans_mat_vectorized(params, ages, sra)

    for id_age in range(1, n_ages):
        final_shares[id_age, :] = (
            final_shares[id_age - 1, :] @ trans_mat_all_ages[id_age - 1, :, :]
        )

    return final_shares
