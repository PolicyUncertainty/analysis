import numpy as np
import pandas as pd
from export_reslts.figures.tools import create_discounted_sum_utilities
from export_reslts.figures.tools import create_realized_taste_shock
from export_reslts.figures.tools import create_step_function_values
from matplotlib import pyplot as plt
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.derive_specs import read_and_derive_specs
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_1,
)


def plot_full_time(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_baseline.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_no_unc.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    full_time_unc = (
        data_unc.groupby("age")["choice"].value_counts(normalize=True).reset_index()
    )
    full_time_unc = full_time_unc[full_time_unc["choice"] == 1]

    full_time_no_unc = (
        data_no_unc.groupby("age")["choice"].value_counts(normalize=True).reset_index()
    )
    full_time_no_unc = full_time_no_unc[full_time_no_unc["choice"] == 1]
    fig, ax = plt.subplots()
    ax.plot(full_time_unc["age"], full_time_unc["proportion"], label="Uncertainty")
    ax.plot(
        full_time_no_unc["age"], full_time_no_unc["proportion"], label="No Uncertainty"
    )
    ax.legend()
    ax.set_title("Proportion of full time workers by age")


def plot_values_by_age(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_baseline.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_no_unc.pkl"
    ).reset_index()

    est_params = pd.read_pickle(paths_dict["est_results"] + "est_params.pkl")

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    data_unc = create_realized_taste_shock(data_unc)
    data_no_unc = create_realized_taste_shock(data_no_unc)

    data_unc["real_util"] = data_unc["real_taste_shock"] + data_unc["utility"]
    data_no_unc["real_util"] = data_no_unc["real_taste_shock"] + data_no_unc["utility"]

    mean_disc_util_unc = create_discounted_sum_utilities(
        data_unc, est_params["beta"], utility_col="real_util"
    )
    mean_disc_util_no_unc = create_discounted_sum_utilities(
        data_no_unc, est_params["beta"], utility_col="real_util"
    )

    value_diff = mean_disc_util_unc - mean_disc_util_no_unc

    max_age = data_unc["age"].max()
    min_age = data_unc["age"].min()
    # plot average value by age
    fig, ax = plt.subplots()
    ax.plot(range(min_age, max_age + 1), mean_disc_util_unc, label="Uncertainty")
    ax.plot(range(min_age, max_age + 1), mean_disc_util_no_unc, label="No Uncertainty")
    # ax.plot(range(min_age, max_age  +1), value_diff, label="No Uncertainty")
    ax.set_title("Average value by age")
    ax.legend()


def plot_average_savings(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_baseline.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_no_unc.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    savings_unc = data_unc.groupby("age")["savings_dec"].mean()
    savings_no_unc = data_no_unc.groupby("age")["savings_dec"].mean()
    fig, ax = plt.subplots()
    ax.plot(savings_unc, label="Uncertainty")
    ax.plot(savings_no_unc, label="No Uncertainty")
    ax.set_title("Average savings by age")
    ax.legend()


def trajectory_plot(path_dict):
    specs = generate_derived_and_data_derived_specs(path_dict)

    specs = update_specs_for_step_function_scale_1(specs=specs, path_dict=path_dict)
    specs = update_specs_exp_ret_age_trans_mat(specs, path_dict)

    policy_state_67 = int((67 - specs["min_SRA"]) / specs["SRA_grid_size"])

    step_function_vals = create_step_function_values(specs, policy_state_67)
    policy_states_delta = (
        np.arange(specs["n_policy_states"]) - policy_state_67
    ) * specs["SRA_grid_size"]
    life_span = specs["end_age"] - specs["start_age"] + 1
    continous_exp_values = np.zeros(life_span)
    for i in range(1, life_span):
        trans_mat_iter = np.linalg.matrix_power(specs["beliefs_trans_mat"], i)

        continous_exp_values[i] = (
            trans_mat_iter[policy_state_67, :] @ policy_states_delta
        )

    continous_exp_values += 67

    ages = np.arange(life_span) + 30
    fig, ax = plt.subplots()
    ax.plot(ages, step_function_vals, label="Step function")
    ax.plot(ages, continous_exp_values, label="Continous expectation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "cf_no_unc_design.png", transparent=True, dpi=300)
