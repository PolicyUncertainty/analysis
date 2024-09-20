import numpy as np
import pandas as pd
from export_results.tools import create_step_function_values
from matplotlib import pyplot as plt
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_1,
)
from specs.derive_specs import generate_derived_and_data_derived_specs
from specs.derive_specs import read_and_derive_specs


def plot_full_time(paths_dict):
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data/data_real_scale_1.pkl"
    ).reset_index()
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

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


def plot_average_savings(paths_dict):
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data/data_real_scale_1.pkl"
    ).reset_index()
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()

    savings_unc = data_unc.groupby("age")["savings_dec"].mean()
    savings_no_unc = data_no_unc.groupby("age")["savings_dec"].mean()
    savings_increase = savings_unc[:35] / savings_no_unc[:35]

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
    plot_span = specs["max_SRA"] - specs["start_age"] + 1

    step_function_vals = create_step_function_values(specs, policy_state_67, plot_span)
    policy_states_delta = (
        np.arange(specs["n_policy_states"]) - policy_state_67
    ) * specs["SRA_grid_size"]

    continous_exp_values = np.zeros(plot_span)
    for i in range(1, plot_span):
        trans_mat_iter = np.linalg.matrix_power(specs["beliefs_trans_mat"], i)

        continous_exp_values[i] = (
            trans_mat_iter[policy_state_67, :] @ policy_states_delta
        )

    continous_exp_values += 67

    ages = np.arange(plot_span) + specs["start_age"]
    fig, ax = plt.subplots()
    ax.plot(ages, step_function_vals, label=r"Step function $\alpha = 0.042$")
    ax.plot(ages, continous_exp_values, label=r"Continous expectation $\alpha = 0.042$")
    ax.legend()
    ax.set_ylabel("SRA")
    ax.set_xlabel("Age")
    ax.set_ylim([67, 71])
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "cf_no_unc_design.png", transparent=True, dpi=300)
