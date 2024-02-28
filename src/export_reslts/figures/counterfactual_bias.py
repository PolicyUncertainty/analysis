import matplotlib.pyplot as plt
import pandas as pd
from export_reslts.figures.tools import create_step_function_values
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.derive_specs import read_and_derive_specs
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_05,
)
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_1,
)
from simulation.policy_state_scenarios.step_function import (
    update_specs_for_step_function_scale_2,
)


def plot_step_functions(path_dict):
    specs = generate_derived_and_data_derived_specs(path_dict)

    specs = update_specs_for_step_function_scale_1(specs=specs, path_dict=path_dict)
    policy_state_67 = int((67 - specs["min_SRA"]) / specs["SRA_grid_size"])
    step_vals_scale_1 = create_step_function_values(specs, policy_state_67)

    specs = update_specs_for_step_function_scale_05(specs=specs, path_dict=path_dict)
    step_vals_scale_05 = create_step_function_values(specs, policy_state_67)

    specs = update_specs_for_step_function_scale_2(specs=specs, path_dict=path_dict)
    step_vals_scale_2 = create_step_function_values(specs, policy_state_67)

    fig, ax = plt.subplots()
    ax.plot(step_vals_scale_1, label="SRA estimated")
    ax.plot(step_vals_scale_05, label="SRA increase half")
    ax.plot(step_vals_scale_2, label="SRA increase double")
    ax.legend()
    ax.set_ylabel("SRA")
    ax.set_xlabel("Age")
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "cf_bias_step_functions.png")


def plot_savings_over_age(paths_dict):
    data_baseline = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_baseline.pkl"
    ).reset_index()

    data_05_scale = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_05_scale.pkl"
    ).reset_index()

    data_2_scale = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_2_scale.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_baseline["age"] = data_baseline["period"] + specs["start_age"]
    data_05_scale["age"] = data_05_scale["period"] + specs["start_age"]
    data_2_scale["age"] = data_2_scale["period"] + specs["start_age"]

    mean_savings = data_baseline.groupby("age")["savings"].mean()
    mean_savings_05_scale = data_05_scale.groupby("age")["savings"].mean()
    mean_savings_2_scale = data_2_scale.groupby("age")["savings"].mean()

    fig, ax = plt.subplots()
    ax.plot(mean_savings[:34], label="Baseline")
    ax.plot(mean_savings_05_scale[:34], label="SRA increase half")
    ax.plot(mean_savings_2_scale[:34], label="SRA increase double")
    ax.legend()
    ax.set_ylabel("Mean savings by age")
    ax.set_xlabel("Age")
    fig.tight_layout()
    fig.savefig(paths_dict["plots"] + "cf_bias_mean_savings_by_age.png")
