import matplotlib.pyplot as plt
import numpy as np

from set_styles import get_figsize
import os

def plot_retirement_difference(
    path_dict,
    specs,
    df_base,
    df_cf,
    final_SRA,
    model_name,
    left_difference,
    right_difference,
    base_label,
    cf_label,
):
    df_base["SRA_diff"] = df_base["age"] - final_SRA
    df_cf["SRA_diff"] = df_cf["age"] - final_SRA

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    df_base_plot = df_base[
        (df_base["SRA_diff"] >= left_difference)
        & (df_base["SRA_diff"] <= right_difference)
        & (df_base["choice"] == 0)
        & (df_base["lagged_choice"] != 0)
    ]
    df_cf_plot = df_cf[
        (df_cf["SRA_diff"] >= left_difference)
        & (df_cf["SRA_diff"] <= right_difference)
        & (df_cf["choice"] == 0)
        & (df_cf["lagged_choice"] != 0)
    ]

    inflow_shares_base = (
        df_base_plot["SRA_diff"].value_counts(normalize=True).sort_index()
    )
    inflow_shares_cf = df_cf_plot["SRA_diff"].value_counts(normalize=True).sort_index()

    # Make barplot with SRA diff on x-axis and inflow shares on y-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        inflow_shares_base.index - 0.1,
        inflow_shares_base.values,
        width=0.2,
        label=base_label,
    )
    ax.bar(
        inflow_shares_cf.index + 0.1, inflow_shares_cf.values, width=0.2, label=cf_label
    )
    ax.legend()
    ax.set_xlabel("Age - SRA")
    ax.set_ylabel("Inflow into retirement share")
    ax.set_title(f"Inflow into retirement by age relative to SRA {final_SRA}")
    fig.savefig(
        plot_folder
        + f"retirement_inflow_comparison_sra_{final_SRA}_{model_name}.png"
    )
    plt.close(fig)

    # Additional plot: 4 subplots (2x2) for each sex-edu combination
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axes[sex_var, edu_var]

            mask_base = (df_base_plot["sex"] == sex_var) & (
                df_base_plot["education"] == edu_var
            )
            mask_cf = (df_cf_plot["sex"] == sex_var) & (
                df_cf_plot["education"] == edu_var
            )

            inflow_shares_base_subset = (
                df_base_plot[mask_base]["SRA_diff"]
                .value_counts(normalize=True)
                .sort_index()
            )
            inflow_shares_cf_subset = (
                df_cf_plot[mask_cf]["SRA_diff"]
                .value_counts(normalize=True)
                .sort_index()
            )

            # Plot bars
            ax.bar(
                inflow_shares_base_subset.index - 0.1,
                inflow_shares_base_subset.values,
                width=0.2,
                label=base_label,
            )
            ax.bar(
                inflow_shares_cf_subset.index + 0.1,
                inflow_shares_cf_subset.values,
                width=0.2,
                label=cf_label,
            )

            ax.set_xlabel("Age - SRA")
            ax.set_ylabel("Inflow into retirement share")
            ax.set_title(f"{sex_label} - {edu_label}")
            ax.legend()

    plt.suptitle(
        f"Inflow into retirement by education and sex relative to SRA {final_SRA}"
    )
    plt.tight_layout()
    fig.savefig(
        plot_folder
        + f"retirement_inflow_by_edu_sex_sra_{final_SRA}_{model_name}.png"
    )
    plt.close(fig)


def plot_retirement_share(
    path_dict,
    specs,
    df_base,
    df_cf,
    final_SRA,
    model_name,
    left_difference,
    right_difference,
    base_label,
    cf_label,
):
    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    all_base_share = (
        df_base.groupby("age")["choice"]
        .value_counts(normalize=True)
        .loc[(slice(None), 0)]
        .sort_index()
    )
    all_cf_share = (
        df_cf.groupby("age")["choice"]
        .value_counts(normalize=True)
        .loc[(slice(None), 0)]
        .sort_index()
    )

    old_ages = np.arange(60, 70)

    # First figure: Overall comparison
    fig_all, axs_all = plt.subplots(1, 2, figsize=get_figsize(1, 2))
    axs_all[0].plot(all_base_share.index, all_base_share.values, label=base_label)
    axs_all[0].plot(all_cf_share.index, all_cf_share.values, label=cf_label)
    axs_all[0].set_xlabel("Age")
    axs_all[0].set_ylabel("Retirement Share")
    axs_all[0].set_title("All Ages")
    axs_all[0].legend()

    axs_all[1].plot(old_ages, all_base_share.loc[old_ages], label=base_label)
    axs_all[1].plot(old_ages, all_cf_share.loc[old_ages], label=cf_label)
    axs_all[1].set_xlabel("Age")
    axs_all[1].set_ylabel("Retirement Share")
    axs_all[1].set_title("Zoom: 60-69")
    axs_all[1].legend()

    plt.tight_layout()
    fig_all.savefig(
        plot_folder
        + f"retirement_share_comparison_sra_{final_SRA}_{model_name}.png"
    )
    plt.close(fig_all)

    # Second figure: By sex (2 rows) and education levels together (2 cols for all/zoom)
    fig_sex, axes_sex = plt.subplots(2, 2, figsize=get_figsize(2, 2))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Filter data by sex
        df_base_sex = df_base[df_base["sex"] == sex_var]
        df_cf_sex = df_cf[df_cf["sex"] == sex_var]

        # Get all agents informed at 63
        base_agents_informed_at_63 = df_base_sex[
            (df_base_sex["age"] == 63) & (df_base_sex["informed"] == 1)
        ]["agent"]
        cf_agents_informed_at_63 = df_cf_sex[
            (df_cf_sex["age"] == 63) & (df_cf_sex["informed"] == 1)
        ]["agent"]

        for informed_var, informed_label in enumerate(["uninformed", "informed"]):

            if informed_label == "informed":
                df_base_sex_edu = df_base_sex[
                    df_base_sex["agent"].isin(base_agents_informed_at_63)
                ]
                df_cf_sex_edu = df_cf_sex[
                    df_cf_sex["agent"].isin(cf_agents_informed_at_63)
                ]
            else:
                df_base_sex_edu = df_base_sex[
                    ~df_base_sex["agent"].isin(base_agents_informed_at_63)
                ]
                df_cf_sex_edu = df_cf_sex[
                    ~df_cf_sex["agent"].isin(cf_agents_informed_at_63)
                ]

            # Calculate retirement shares for this sex-edu combination
            base_share_sex_edu = (
                df_base_sex_edu.groupby("age")["choice"]
                .value_counts(normalize=True)
                .loc[(slice(None), 0)]
                .sort_index()
            )
            cf_share_sex_edu = (
                df_cf_sex_edu.groupby("age")["choice"]
                .value_counts(normalize=True)
                .loc[(slice(None), 0)]
                .sort_index()
            )

            # Plot all ages (left column)
            axes_sex[sex_var, 0].plot(
                base_share_sex_edu.index,
                base_share_sex_edu.values,
                label=f"{base_label} - {informed_label}",
            )
            axes_sex[sex_var, 0].plot(
                cf_share_sex_edu.index,
                cf_share_sex_edu.values,
                label=f"{cf_label} - {informed_label}",
            )

            # Plot zoom ages (right column)
            axes_sex[sex_var, 1].plot(
                old_ages,
                base_share_sex_edu.loc[old_ages],
                label=f"{base_label} - {informed_label}",
            )
            axes_sex[sex_var, 1].plot(
                old_ages,
                cf_share_sex_edu.loc[old_ages],
                label=f"{cf_label} - {informed_label}",
            )

        # Set labels and titles for this sex
        axes_sex[sex_var, 0].set_xlabel("Age")
        axes_sex[sex_var, 0].set_ylabel("Retirement Share")
        axes_sex[sex_var, 0].set_title(f"{sex_label} - All Ages")
        axes_sex[sex_var, 0].legend()

        axes_sex[sex_var, 1].set_xlabel("Age")
        axes_sex[sex_var, 1].set_ylabel("Retirement Share")
        axes_sex[sex_var, 1].set_title(f"{sex_label} - Zoom: 60-69")
        axes_sex[sex_var, 1].legend()

    plt.tight_layout()
    fig_sex.savefig(
        plot_folder
        + f"retirement_share_by_sex_informed_sra_{final_SRA}_{model_name}.png"
    )
    plt.close(fig_sex)
