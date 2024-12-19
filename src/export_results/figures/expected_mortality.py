import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mortality(paths_dict, specs):
    """Plot mortality characteristics."""

    # Load the data
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )
    df["period"] = df["age"] - specs["start_age_mortality"] 
    estimated_mortality = pd.read_csv(
        paths_dict["est_results"] + "mortality_transition_matrix.csv"
    )
    df_params_male = pd.read_pickle(paths_dict["est_results"] + "est_params_mortality_male.pkl")
    df_params_female = pd.read_pickle(paths_dict["est_results"] + "est_params_mortality_female.pkl")

    combinations = [
        (1, 1, "health1_edu1"),
        (1, 0, "health1_edu0"),
        (0, 1, "health0_edu1"),
        (0, 0, "health0_edu0"),
    ]

    # plot the estimated survival function without adjustment using life tables
    for sex in ["male", "female"]:
        res = df_params_male if sex == "male" else df_params_female
        colors = {1: "#1E90FF", 0: "#D72638"} # blue, red
        fig, ax = plt.subplots(figsize=(8, 6))
        age = np.linspace(16, 110, 110 - 16 + 1)
        for health, education, param in combinations:
            edu_label = specs["education_labels"][education]
            health_label = "Bad Health" if health == 0 else "Good Health"
            linestyle = "--" if education == 0 else "-"
            ax.plot(
                age,
                survival_function(
                    age, health*education, health*(1-education), (1-health)*education, (1-health)*(1-education), res
                ),
                label=f"{edu_label}, {health_label}",
                linestyle=linestyle,
                color=colors[health],
                alpha=(1 - 0.8*education),
            )
        ax.plot(
            age,
            survival_function(age, 0, 0, 0, 0, res),
            label="Reference",
            linestyle="-.",
        )
        ax.set_xlabel("Age")
        ax.set_xlim(16, 110)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"Estimated Survival Function for {sex.capitalize()}")

        ax.legend()
        ax.grid()
        plt.show()



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
                    ["death_prob", "period"],
                ]

                # Sort by period
                filtered_data = filtered_data.sort_values(by="period")

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
        plt.plot(data["period"], data["survival_prob"], label=key)

    # Add labels and legend
    plt.xlabel("Period")
    plt.ylabel("Survival Probability")
    plt.title(
        "Survival Probability by period for Different Combinations of Sex, Health, and Education"
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    # Define parameters for subplots
    sexes = ["male", "female"]
    colors = {0: "#1E90FF", 1: "#D72638"}  # Blue for male, red for female
    # Set up the figure for share of deaths
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 subplots for male and female
    fig.suptitle("Share of Deaths by Period and Subgroups", fontsize=16)

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate plain share of deaths by period
        period_death_share = (
            filtered_df.groupby("period")["event_death"]
            .mean()
            .reindex(np.arange(16, 111))
            .fillna(0)
        )

        # Plot plain share
        ax.plot(
            period_death_share.index,
            period_death_share.values,
            label="Plain Share of Deaths",
            color="black",
            linestyle="-",
        )

        # Calculate share of deaths by subgroups (education and health state)
        for edu in [0, 1]:
            for health in [0, 1]:
                subgroup = filtered_df[
                    (filtered_df["education"] == edu)
                    & (filtered_df["health_state"] == health)
                ]
                subgroup_death_share = (
                    subgroup.groupby("period")["event_death"]
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
        ax.set_xlabel("period")
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
    fig.suptitle("Total Number of Deaths by period and Subgroups", fontsize=16)

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate total deaths by subgroups (education and health state)
        period_range = np.arange(16, 111)
        bar_data = {
            (edu, health): filtered_df[
                (filtered_df["education"] == edu)
                & (filtered_df["health_state"] == health)
            ]
            .groupby("period")["event_death"]
            .sum()
            .reindex(period_range)
            .fillna(0)
            for edu in [0, 1]
            for health in [0, 1]
        }

        # Plot stacked bar chart
        bottom = np.zeros_like(period_range, dtype=float)
        for (edu, health), values in bar_data.items():
            edu_label = f"Education {edu}"
            health_label = f"Health {health}"
            ax.bar(
                period_range,
                values,
                bottom=bottom,
                label=f"{edu_label}, {health_label}",
                color=colors[health],
                alpha=0.7 * (1 if edu == 0 else 0.5),
            )
            bottom += values

        # Set title, labels, and limits
        ax.set_title(f"{sex.capitalize()}")
        ax.set_xlabel("period")
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
        "Share of Deaths by period and Subgroups as Percentperiod of Total Deaths",
        fontsize=11,
    )

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate plain share of deaths by period
        period_death_share = filtered_df.groupby("period")["event_death"].count().reindex(
            np.arange(16, 111)
        ).fillna(0) / len(df[df["event_death"] == 1])

        # Plot plain share
        ax.plot(
            period_death_share.index,
            period_death_share.values,
            label=f"{sex.capitalize()}",
            color="black",
            linestyle="-",
        )

        # Calculate share of deaths by subgroups (education and health state)
        for edu in [0, 1]:
            for health in [0, 1]:
                subgroup = filtered_df[
                    (filtered_df["education"] == edu)
                    & (filtered_df["health_state"] == health)
                ]
                subgroup_death_share = subgroup.groupby("period")[
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
        ax.set_xlabel("period")
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


def survival_function(
    age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params
):
    """Exp(-(integral of the hazard function as a function of age from 0 to age))"""
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]

    lambda_ = np.exp(
        cons
        + health1_edu1_coef * health1_edu1
        + health1_edu0_coef * health1_edu0
        + health0_edu1_coef * health0_edu1
        + health0_edu0_coef * health0_edu0
    )
    age_contrib = np.exp(age_coef * age) - 1

    return np.exp(-lambda_ / age_coef * age_contrib)