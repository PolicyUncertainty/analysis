import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize

from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_markov_process(paths_dict):
    specs = generate_derived_and_data_derived_specs(paths_dict)

    specs = update_specs_exp_ret_age_trans_mat(specs, paths_dict)

    belief_trans_mat = specs["policy_states_trans_mat"][:-1, :-1]

    # Assuming belief_trans_mat is a 2D numpy array

    # Initialize variables
    n_periods = 35
    mean_vals = [67]
    std_vals = [0]

    SRA_values = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    for i in range(1, n_periods + 1):
        # Calculate the matrix to the power of i
        result_mat = np.linalg.matrix_power(belief_trans_mat, i)

        mean = np.dot(result_mat[8], SRA_values)
        variance = np.dot(result_mat[8], SRA_values**2) - mean**2
        std = np.sqrt(variance)

        # Append to lists
        mean_vals.append(mean)
        std_vals.append(std)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(range(0, n_periods + 1), mean_vals)
    ax.fill_between(
        range(0, n_periods + 1),
        np.array(mean_vals) - np.array(std_vals),
        np.array(mean_vals) + np.array(std_vals),
        alpha=0.3,
    )
    ax.set_xlabel("Time to retirement")
    ax.set_ylabel("SRA")
    # ax.legend()
    fig.tight_layout()
    fig.savefig(
        paths_dict["plots"] + "expectation_process.png", transparent=True, dpi=300
    )


def plot_sra_beliefs_by_cohort(paths_dict):
    df_soep_is = pd.read_stata(paths_dict["soep_is"], convert_categoricals=False)

    df_soep_is.loc[:, "expected_stat_ret_age"] = (
        df_soep_is["pol_unc_stat_ret_age_67"] * 67
        + df_soep_is["pol_unc_stat_ret_age_68"] * 68
        + df_soep_is["pol_unc_stat_ret_age_69"] * 69
    )
    relevant_columns = [
        "pol_unc_stat_ret_age_67",
        "pol_unc_stat_ret_age_68",
        "pol_unc_stat_ret_age_69",
        "gebjahr",
    ]
    exp_ret_data = df_soep_is[~df_soep_is["expected_stat_ret_age"].isnull()][
        relevant_columns
    ]
    age_bins = list(range(1957, 2001, 5))

    exp_ret_data["gebjahr_group"] = create_gebjahr_groups(
        exp_ret_data, age_bins=age_bins
    )

    exp_ret_data_grouped = exp_ret_data.groupby(["gebjahr_group"], observed=True)
    exp_ret_data_mean = exp_ret_data_grouped[
        [
            "pol_unc_stat_ret_age_67",
            "pol_unc_stat_ret_age_68",
            "pol_unc_stat_ret_age_69",
        ]
    ].mean()

    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 30,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    exp_ret_data_mean.plot(
        y=[
            "pol_unc_stat_ret_age_67",
            "pol_unc_stat_ret_age_68",
            "pol_unc_stat_ret_age_69",
        ],
        kind="bar",
        stacked=True,
        ax=ax,
        label=["67", "68", "69+"],
    )
    # Replace birth cohort integers by two lined strings first row given the start date if cohor and second row the end date of cohort
    ax.set_xticks(range(0, 8))
    # Make the strings above such that they span two lines on x axis
    ax.set_xticklabels(
        [
            "1957-\n1961 ",
            "1962-\n1966 ",
            "1967-\n1971 ",
            "1972-\n1976 ",
            "1977-\n1981 ",
            "1982-\n1986 ",
            "1987-\n1991 ",
            "1992-\n1996 ",
        ],
        rotation=0,
    )
    ax.legend(loc="lower left")
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Attributed Percentage")
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 100, 20))
    fig.tight_layout()


def create_gebjahr_groups(data, age_bins):
    return pd.cut(
        data["gebjahr"], bins=age_bins, labels=range(len(age_bins) - 1), right=False
    )


def plot_erp_beliefs_by_cohort(paths_dict):
    # Load and prepare the data
    df_soep_is = pd.read_stata(paths_dict["soep_is"], convert_categoricals=False)
    relevant_columns = [
        "belief_pens_deduct",
        "belief_pens_deduct_rob_times1_5",
        "belief_pens_deduct_rob_times0_5",
        "gebjahr",
    ]
    data_deduction = df_soep_is[~df_soep_is["belief_pens_deduct"].isnull()][
        relevant_columns
    ]
    age_bins = age_bins = [-np.inf] + list(range(1957, 2001, 5))
    data_deduction["gebjahr_group"] = create_gebjahr_groups(
        data_deduction, age_bins=age_bins
    )
    ded_data_edu_grouped = data_deduction.groupby(["gebjahr_group"], observed=True)
    ded_data_edu_mean = ded_data_edu_grouped["belief_pens_deduct"].mean()
    ded_data_edu_sem = ded_data_edu_grouped["belief_pens_deduct"].sem()
    ded_data_edu_median = ded_data_edu_grouped["belief_pens_deduct"].median()
    # Set matplotlib fontsizes
    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 30,
        }
    )
    # Make lines of plots thicker
    plt.rcParams["lines.linewidth"] = 3
    # Plot the results
    fig, ax = plt.subplots(figsize=(16, 9))
    ded_data_edu_mean.plot(
        y="belief_pens_deduct",
        ax=ax,
        label="mean ERP belief",
    )
    ded_data_edu_median.plot(
        y="belief_pens_deduct",
        ax=ax,
        #     color="grey",
        ls="--",
        label="median ERP belief",
    )
    ax.errorbar(
        x=ded_data_edu_mean.index,
        y=ded_data_edu_mean,
        yerr=ded_data_edu_sem,
        fmt="o",
        color="black",
        ecolor="grey",
        capsize=5,
    )
    # Make horizontal line at 3.6% pension deduction
    ax.axhline(y=3.6, color="gray", linestyle="--", label="true ERP")
    ax.set_xticks(range(0, 9))
    # Make the strings above such that they span two lines on x axis
    ax.set_yticks(np.arange(0, 20, 2.5))
    # # Make the strings above such that they span two lines on x axis
    ax.set_xticklabels(
        [
            "1956 &\nbefore ",
            "1957-\n1961 ",
            "1962-\n1966 ",
            "1967-\n1971 ",
            "1972-\n1976 ",
            "1977-\n1981 ",
            "1982-\n1986 ",
            "1987-\n1991 ",
            "1992-\n1996 ",
        ],
        rotation=0,
    )
    ax.legend(loc="upper left", fancybox=True, framealpha=0.5)
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Penalty in %")
    ax.set_ylim([0, 20])
    fig.tight_layout()


def plot_example_sra_evolution(alpha, sigma_sq, alpha_star, SRA_30, resolution_age):

    ages = np.arange(30, resolution_age + 1, 1)
    SRA_t = np.ones(ages.shape) * SRA_30
    SRA_t = SRA_t + (ages - 30) * alpha_star
    exp_SRA_resolution = SRA_t + (resolution_age - ages) * alpha
    ci_upper = exp_SRA_resolution + 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        resolution_age - ages
    )
    ci_lower = exp_SRA_resolution - 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        resolution_age - ages
    )
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(ages, SRA_t, label="$SRA_t$", color="red")
    ax.plot(
        ages,
        exp_SRA_resolution,
        label=f"$E[SRA_{{{resolution_age}}}|SRA_t]$",
        color="C0",
    )
    ax.plot(ages, ci_upper, label="95% CI", linestyle="--", color="C0")
    ax.plot(ages, ci_lower, label="", linestyle="--", color="C0")
    ax.set_xlabel("Age")
    ax.set_ylabel("SRA")
    ax.legend()
