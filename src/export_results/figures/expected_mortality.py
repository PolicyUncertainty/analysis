import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mortality(paths_dict, specs):
    """Plot mortality characteristics."""

    # Load the data
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )

    estimated_mortality = pd.read_csv(
        paths_dict["est_results"] + "mortality_transition_matrix.csv"
    )
    estimated_mortality["age"] = (
        estimated_mortality["period"] + specs["start_age_mortality"]
    )

    # Initialize an empty dictionary to store survival probabilities
    survival_data = {}

    # Iterate over all combinations of sex, education, and health (0 or 1)
    for sex in [0, 1]:
        for health in [0, 1]:
            for education in [0, 1]:
                # Create a unique key for the combination
                key = f"sex{sex}_health{health}_edu{education}"

                # Filter the data for the current combination
                filtered_data = estimated_mortality.loc[
                    (estimated_mortality["sex"] == sex)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == education),
                    ["death_prob", "age"],
                ]

                # Sort by age
                filtered_data = filtered_data.sort_values(by="age")

                # Calculate survival probabilities
                filtered_data["survival_prob_year"] = 1 - filtered_data["death_prob"]
                filtered_data["survival_prob"] = filtered_data[
                    "survival_prob_year"
                ].cumprod()
                filtered_data["survival_prob"] = (
                    filtered_data["survival_prob"].shift(1).fillna(1)
                )

                # Store the result in the dictionary
                survival_data[key] = filtered_data

    # Plot all 8 lines
    plt.figure(figsize=(12, 8))

    for key, data in survival_data.items():
        plt.plot(data["age"], data["survival_prob"], label=key)

    # Add labels and legend
    plt.xlabel("Age")
    plt.ylabel("Survival Probability")
    plt.title(
        "Survival Probability by Age for Different Combinations of Sex, Health, and Education"
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    # Define parameters for subplots
    sexes = ["male", "female"]
    age_settings = [False, True]
    colors = {0: "#1E90FF", 1: "#D72638"}  # Blue for male, red for female

    # Set up the figure for share of deaths
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 subplots for male and female
    fig.suptitle("Share of Deaths by Age and Subgroups", fontsize=16)

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate plain share of deaths by age
        age_death_share = (
            filtered_df.groupby("age")["event_death"]
            .mean()
            .reindex(np.arange(16, 111))
            .fillna(0)
        )

        # Plot plain share
        ax.plot(
            age_death_share.index,
            age_death_share.values,
            label="Plain Share of Deaths",
            color="black",
            linestyle="-",
        )

        # Calculate share of deaths by subgroups (education and health state)
        for edu in [0, 1]:
            for health in [0, 1]:
                subgroup = filtered_df[
                    (filtered_df["education"] == edu)
                    & (filtered_df["health"] == health)
                ]
                subgroup_death_share = (
                    subgroup.groupby("age")["event_death"]
                    .mean()
                    .reindex(np.arange(16, 111))
                    .fillna(0)
                )

                edu_label = f"Education {edu}"
                health_label = f"Health {health}"
                linestyle = "--" if edu == 0 else "-"

                ax.plot(
                    subgroup_death_share.index,
                    subgroup_death_share.values,
                    label=f"{edu_label}, {health_label}",
                    linestyle=linestyle,
                    color=colors[health],
                )

        # Set title, labels, and limits
        ax.set_title(f"{sex.capitalize()}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Share of Deaths")
        ax.set_xlim(16, 110)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(20, 101, 10))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Add legend
        ax.legend(fontsize=8, loc="upper right")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()

    # Set up the figure for stacked bar chart of total deaths
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 subplots for male and female
    fig.suptitle("Total Number of Deaths by Age and Subgroups", fontsize=16)

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate total deaths by subgroups (education and health state)
        age_range = np.arange(16, 111)
        bar_data = {
            (edu, health): filtered_df[
                (filtered_df["education"] == edu) & (filtered_df["health"] == health)
            ]
            .groupby("age")["event_death"]
            .sum()
            .reindex(age_range)
            .fillna(0)
            for edu in [0, 1]
            for health in [0, 1]
        }

        # Plot stacked bar chart
        bottom = np.zeros_like(age_range, dtype=float)
        for (edu, health), values in bar_data.items():
            edu_label = f"Education {edu}"
            health_label = f"Health {health}"
            ax.bar(
                age_range,
                values,
                bottom=bottom,
                label=f"{edu_label}, {health_label}",
                color=colors[health],
                alpha=0.7 * (1 if edu == 0 else 0.5),
            )
            bottom += values

        # Set title, labels, and limits
        ax.set_title(f"{sex.capitalize()}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Total Number of Deaths")
        ax.set_xlim(16, 110)
        ax.set_xticks(np.arange(20, 101, 10))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Add legend
        ax.legend(fontsize=8, loc="upper right")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()

    # Set up the figure for share of deaths
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 subplots for male and female
    fig.suptitle(
        "Share of Deaths by Age and Subgroups as Percentage of Total Deaths",
        fontsize=11,
    )

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate plain share of deaths by age
        age_death_share = filtered_df.groupby("age")["event_death"].count().reindex(
            np.arange(16, 111)
        ).fillna(0) / len(df[df["event_death"] == 1])

        # Plot plain share
        ax.plot(
            age_death_share.index,
            age_death_share.values,
            label=f"{sex.capitalize()}",
            color="black",
            linestyle="-",
        )

        # Calculate share of deaths by subgroups (education and health state)
        for edu in [0, 1]:
            for health in [0, 1]:
                subgroup = filtered_df[
                    (filtered_df["education"] == edu)
                    & (filtered_df["health"] == health)
                ]
                subgroup_death_share = subgroup.groupby("age")[
                    "event_death"
                ].count().reindex(np.arange(16, 111)).fillna(0) / len(
                    df[df["event_death"] == 1]
                )

                edu_label = f"Education {edu}"
                health_label = f"Health {health}"
                linestyle = "--" if edu == 0 else "-"

                ax.plot(
                    subgroup_death_share.index,
                    subgroup_death_share.values,
                    label=f"{edu_label}, {health_label}",
                    linestyle=linestyle,
                    color=colors[health],
                )

        # Set title, labels, and limits
        ax.set_title(f"{sex.capitalize()}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Deaths / Total Deaths (%)")
        ax.set_xlim(16, 110)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(20, 101, 10))
        ax.set_yticks(np.arange(0, 2.9, 0.1))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Add legend
        ax.legend(fontsize=8, loc="upper right")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()
