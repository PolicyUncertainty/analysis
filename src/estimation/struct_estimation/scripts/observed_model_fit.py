import matplotlib.pyplot as plt
import numpy as np

from export_results.figures.color_map import JET_COLOR_MAP


def plot_life_cycle_choice_probs(
    specs,
    data_decision,
    save_folder,
):
    df_int = data_decision[data_decision["age"] < 75].copy()

    choice_share_labels = ["Choice Share Men", "Choice Share Women"]
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        if sex_var == 0:
            n_choices = 3
        else:
            n_choices = 4
        fig, axes = plt.subplots(ncols=n_choices, figsize=(14, 8))
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            data_subset = df_int[
                (df_int["education"] == edu_var) & (df_int["sex"] == sex_var)
            ]
            all_choice_shares_obs = (
                data_subset.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )

            labels = specs["choice_labels"]
            for choice in range(specs["n_choices"]):

                choice_shares_predicted = data_subset.groupby(["age"])[
                    f"choice_{choice}"
                ].mean()

                # Only plot if we are not in men and part-time
                men_and_part_time = (sex_var == 0) and (choice == 2)
                if not men_and_part_time:
                    men_and_full_time = (sex_var == 0) and (choice == 3)
                    if men_and_full_time:
                        ax = axes[choice - 1]
                    else:
                        ax = axes[choice]

                    ax.plot(
                        choice_shares_predicted,
                        label=f"Pred. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                    )
                    ax.plot(
                        all_choice_shares_obs[choice],
                        label=f"Obs. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                    )

                    ax.set_ylim([-0.05, 1.05])
                    ax.set_title(f"{labels[choice]}")
                    ax.set_xlabel("Age")

        axes[1].legend(loc="upper left")
        axes[0].set_ylabel(choice_share_labels[sex_var])
        # Fig title
        fig.tight_layout()

        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_{sex_label}.png",
            transparent=True,
            dpi=300,
        )


def plot_retirement_fit(
    specs,
    data_decision,
    save_folder,
):
    data_decision["SRA_diff"] = (
        data_decision["age"] - data_decision["policy_state_value"]
    )

    diffs = np.arange(-2, 2.25, 0.25)
    pos = np.arange(0, len(diffs))

    not_retired_mask = data_decision["lagged_choice"] != 0
    data_subset = data_decision[not_retired_mask].copy()

    choices_shares_obs = (
        data_subset.groupby(["sex", "SRA_diff"])["choice"]
        .value_counts()
        .unstack()
        .fillna(0.0)
    )
    choices_shares_obs.columns = specs["choice_labels"]

    # Create a stacked bar plot with each choice sum on top of each other
    choice_shares_est = data_subset.groupby(["sex", "SRA_diff"])[
        [f"choice_{choice}" for choice in range(specs["n_choices"])]
    ].sum()
    choice_shares_est.columns = specs["choice_labels"]

    fig, axs = plt.subplots(nrows=2, figsize=(14, 8))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):

        choice_shares_est_to_plot = choice_shares_est.loc[sex_var].reindex(diffs)
        choices_shares_obs_to_plot = choices_shares_obs.loc[sex_var].reindex(diffs)

        # Plotting the retirement choice shares as bars (observed as hatched)
        ax = axs[sex_var]
        choice_shares_est_to_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=JET_COLOR_MAP[: specs["n_choices"]],
            width=0.4,
            position=0,
            hatch="/",
        )
        choices_shares_obs_to_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=JET_COLOR_MAP[: specs["n_choices"]],
            width=0.4,
            position=1,
            label="Observed",
        )

    # Set x-axis labels and ticks
    axs[0].set_xlabel("Difference to SRA")
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels(diffs)
    axs[1].set_xlabel("Difference to SRA")
    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(diffs)

    axs[0].set_title("Men")
    axs[1].set_title("Women")
    axs[0].set_ylabel("Choice Share")
    fig.tight_layout()
    fig.savefig(
        save_folder + f"retirement_fit.png",
        transparent=True,
        dpi=300,
    )
