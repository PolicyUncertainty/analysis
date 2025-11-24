import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors, set_plot_defaults


def plot_retirement_bunching(
    path_dict,
    specs,
    model_name,
):
    set_plot_defaults(plot_type="paper")

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    save_folder = path_dict["sim_results"] + model_name + "/"

    df_base_plot = pd.read_csv(save_folder + "df_bunching_base.csv")
    df_cf_plot = pd.read_csv(save_folder + "df_bunching_cf.csv")
    df_base_plot["SRA_diff"] = df_base_plot["policy_state_value"] - df_base_plot["age"]
    df_cf_plot["SRA_diff"] = df_cf_plot["policy_state_value"] - df_cf_plot["age"]

    ages_to_plot = np.arange(63, 69)

    # Create retirement type categories
    def categorize_retirement(df):
        condition_very_long_insured = (df["SRA_diff"] > 0) & df["very_long_insured"]
        condition_disability = (df["SRA_diff"] > 0) & (df["health"] == 2)
        condition_standard = ~condition_very_long_insured & ~condition_disability
        conditions = [
            (df["SRA_diff"] > 0) & df["very_long_insured"],  # very long insured (early)
            (df["SRA_diff"] > 0) & (df["health"] == 2),  # disability (early)
            condition_standard,
        ]
        choices = ["very_long_insured", "disability", "standard"]
        df["retirement_type"] = np.select(conditions, choices, default="standard")
        return df

    df_base_plot = categorize_retirement(df_base_plot)
    df_cf_plot = categorize_retirement(df_cf_plot)

    # Calculate shares for each retirement type by age (normalized by total population)
    def calculate_shares_by_type(df, ages):
        total_count = len(df)
        shares_dict = {}

        for ret_type in ["standard", "very_long_insured", "disability"]:
            type_age_counts = df[df["retirement_type"] == ret_type][
                "age"
            ].value_counts()
            shares = (type_age_counts / total_count).reindex(ages, fill_value=0)
            shares_dict[ret_type] = shares

        return shares_dict

    shares_base = calculate_shares_by_type(df_base_plot, ages_to_plot)
    shares_cf = calculate_shares_by_type(df_cf_plot, ages_to_plot)

    # Main plot with stacked bars
    fig, ax = plt.subplots()

    width = 0.2
    jet_color_map, _ = set_colors()

    # Use same colors as reference function
    colors = {
        "standard": jet_color_map[0],  # blue
        "very_long_insured": jet_color_map[9],  # light blue
        "disability": jet_color_map[1],  # orange
    }

    # Use same labels as reference function
    labels = {
        "standard": "old age: standard",
        "very_long_insured": "old age: long work life",
        "disability": "disability: occupation",
    }

    # Define plotting order (same as reference function)
    plot_order = ["standard", "very_long_insured", "disability"]

    # Base model bars (left)
    bottom_base = np.zeros(len(ages_to_plot))
    for ret_type in plot_order:
        ax.bar(
            ages_to_plot - 0.1,
            shares_base[ret_type].values,
            width=width,
            bottom=bottom_base,
            color=colors[ret_type],
        )
        bottom_base += shares_base[ret_type].values

    # Counterfactual bars (right) with hatching
    bottom_cf = np.zeros(len(ages_to_plot))
    for ret_type in plot_order:
        ax.bar(
            ages_to_plot + 0.1,
            shares_cf[ret_type].values,
            width=width,
            bottom=bottom_cf,
            color=colors[ret_type],
            hatch="///",
            edgecolor="white",
            linewidth=0.5,
        )
        bottom_cf += shares_cf[ret_type].values

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["standard"], label=labels["standard"]),
        Patch(facecolor=colors["very_long_insured"], label=labels["very_long_insured"]),
        Patch(facecolor=colors["disability"], label=labels["disability"]),
        Patch(
            facecolor="white",
            hatch="///",
            edgecolor="black",
            label="informed only model",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        columnspacing=0.5,
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Pension Claims")
    ax.set_ylim(0, 0.27)
    plt.tight_layout()
    fig.savefig(plot_folder + "paper_plots/" + "retirement_bunching.png")
    print("Saved plot to " + plot_folder + "paper_plots/" + "retirement_bunching.png")
    plt.close(fig)

    # Additional plot: 4 subplots (2x2) for each sex-edu combination
    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axes[sex_var, edu_var]

            mask_base = (df_base_plot["sex"] == sex_var) & (
                df_base_plot["education"] == edu_var
            )
            mask_cf = (df_cf_plot["sex"] == sex_var) & (
                df_cf_plot["education"] == edu_var
            )

            shares_base_subset = calculate_shares_by_type(
                df_base_plot[mask_base], ages_to_plot
            )
            shares_cf_subset = calculate_shares_by_type(
                df_cf_plot[mask_cf], ages_to_plot
            )

            # Base model bars (left)
            bottom_base = np.zeros(len(ages_to_plot))
            for ret_type in plot_order:
                ax.bar(
                    ages_to_plot - 0.1,
                    shares_base_subset[ret_type].values,
                    width=width,
                    bottom=bottom_base,
                    color=colors[ret_type],
                )
                bottom_base += shares_base_subset[ret_type].values

            # Counterfactual bars (right) with hatching
            bottom_cf = np.zeros(len(ages_to_plot))
            for ret_type in plot_order:
                ax.bar(
                    ages_to_plot + 0.1,
                    shares_cf_subset[ret_type].values,
                    width=width,
                    bottom=bottom_cf,
                    color=colors[ret_type],
                    hatch="///",
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottom_cf += shares_cf_subset[ret_type].values

            ax.set_xlabel("Age")
            ax.set_ylabel("Inflow into retirement share")
            ax.set_title(f"{sex_label} - {edu_label}")
            ax.set_ylim(0, 0.28)

            # Only add legend to first subplot
            if sex_var == 0 and edu_var == 0:
                ax.legend(handles=legend_elements, loc="best", ncol=2, frameon=False)

    plot_name = f"retirement_bunching_by_sex_and_edu.png"
    plt.tight_layout()
    fig.savefig(plot_folder + plot_name)
    plt.close(fig)
