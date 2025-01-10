import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mortality(paths_dict, specs):
    """Plot mortality characteristics."""

    ### Load the data
    # Mortality estimation sample
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )

    # Estimated mortality transition matrix (life table adjusted probabilities of death)
    estimated_mortality = pd.read_csv(
        paths_dict["est_results"] + "mortality_transition_matrix.csv"
    )

    # Estimated mortality parameters
    df_params_male = pd.read_csv(
        paths_dict["est_results"] + "est_params_mortality_men.csv"
    )
    df_params_male.set_index("Unnamed: 0", inplace=True)
    df_params_female = pd.read_csv(
        paths_dict["est_results"] + "est_params_mortality_women.csv"
    )
    df_params_female.set_index("Unnamed: 0", inplace=True)

    n_edu_types = len(specs["education_labels"])
    n_ages = specs["end_age_mortality"] - specs["start_age_mortality"] + 1
    n_alive_health_states = len(specs["health_labels"]) - 1
    alive_health_states = np.where(np.array(specs["health_labels"]) != "Death")[0]
    health_states = [specs["health_labels"][i] for i in alive_health_states]
    education_states = specs["education_labels"]
    age_range = np.arange(specs["start_age_mortality"], specs["end_age_mortality"] + 1)

    colors = {1: "#1E90FF", 0: "#D72638"}  # blue and red

    # Create a dictionary to store the survival rates for each combination for each age/age
    final_survival_est = {}
    for sex in [0, 1]:
        for health in alive_health_states:
            for education in range(n_edu_types):
                # Create a unique key for the combination
                key = (
                    specs["sex_labels"][sex]
                    + ", "
                    + specs["education_labels"][education]
                    + ", "
                    + specs["health_labels"][health]
                )

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
                filtered_data["survival_prob"] = filtered_data["survival_prob"].shift(
                    1
                )  # age = x -> survival prob up to x+1 (end of period x)
                filtered_data.loc[
                    0, "survival_prob"
                ] = 1  # Set 100% survival at beginning of first period

                # Store the result in the dictionary
                final_survival_est[key] = filtered_data

    ############################################################################
    # plot the estimated survival function without adjustment using life tables
    ############################################################################
    # By health and education (see Haan, Schaller 2024 based on Kroll, Lampert 2008)
    for sex in specs["sex_labels"]:
        res = df_params_male if sex == "Male" else df_params_female
        fig, ax = plt.subplots(figsize=(8, 6))
        age = np.linspace(
            specs["start_age_mortality"], specs["end_age_mortality"], 2 * (n_ages + 1)
        )

        reference_factors = {
            f"{specs['health_labels'][1]} {specs['education_labels'][1]}": 0,
            f"{specs['health_labels'][1]} {specs['education_labels'][0]}": 0,
            f"{specs['health_labels'][0]} {specs['education_labels'][1]}": 0,
            f"{specs['health_labels'][0]} {specs['education_labels'][0]}": 0,
        }
        ax.plot(
            age,
            survival_function(age, reference_factors, res),
            label="Reference",
            linestyle=":",
            color="black",
        )

        for health in alive_health_states:
            for education in range(n_edu_types):
                health_factors = {
                    f"{specs['health_labels'][1]} {specs['education_labels'][1]}": health
                    * education,
                    f"{specs['health_labels'][1]} {specs['education_labels'][0]}": health
                    * (1 - education),
                    f"{specs['health_labels'][0]} {specs['education_labels'][1]}": (
                        1 - health
                    )
                    * education,
                    f"{specs['health_labels'][0]} {specs['education_labels'][0]}": (
                        1 - health
                    )
                    * (1 - education),
                }
                ax.plot(
                    age,
                    survival_function(age, health_factors, res),
                    label=f"{specs['education_labels'][education]}, {specs['health_labels'][health]}",
                    linestyle="--" if education == 0 else "-",
                    color=colors[health],
                    alpha=(1 - 0.5 * education),
                )

        ax.set_xlabel("Age")
        ax.set_xlim(specs["start_age_mortality"], specs["end_age_mortality"] + 1)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"(Naive) Estimated Survival Function for {sex.capitalize()}")

        ax.legend()
        ax.grid()
        plt.show()

    ######################################################################################################
    # Plot the estimated survival function for different combinations (with adjustment using life tables)
    ######################################################################################################
    for sex in specs["sex_labels"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, data in final_survival_est.items():
            if key.startswith(sex):
                ax.plot(
                    data["age"],
                    data["survival_prob"],
                    label=key,
                    linestyle="--"
                    if key.find(specs["education_labels"][0]) > -1
                    else "-",
                    color=colors[0 if key.find(specs["health_labels"][0]) > -1 else 1],
                    alpha=1 if key.find(specs["education_labels"][0]) > -1 else 0.5,
                )

        # Set title, labels, and limits
        ax.set_xlabel("Age")
        ax.set_xlim(specs["start_age_mortality"], specs["life_table_max_age"] + 1)
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"(Model) Estimated Survival Probability by Age for {sex.capitalize()} (with Adjustment)"
        )

        ax.legend()
        ax.grid()
        plt.show()

    #################################################################################
    # Plot the in-sample death probabilities for different combinations for each age
    #################################################################################
    for sex_label in specs["sex_labels"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        filtered_df = df[df["sex"] == (0 if sex_label == "Male" else 1)]

        # cumulate plain share of deaths by age (survival probabilities)
        death_share = (
            filtered_df.groupby("age")["death event"]
            .mean()
            .reindex(age_range)
            .fillna(0)
        )
        alive = np.ones(n_ages)
        for i in range(1, n_ages):
            alive[i] = alive[i - 1] * (
                1 - death_share[i + specs["start_age_mortality"]]
            )
        ax.plot(
            death_share.index,
            alive,
            label="Reference",
            color="black",
            linestyle=":",
        )

        # cumulate share of deaths by subgroups (survival probabilities)
        for education in range(n_edu_types):
            for health in alive_health_states:
                subgroup = filtered_df[
                    (filtered_df["education"] == education)
                    & (filtered_df["health"] == health)
                ]
                subgroup_death_share = (
                    subgroup.groupby("age")["death event"]
                    .mean()
                    .reindex(age_range)
                    .fillna(0)
                )
                alive = np.ones(n_ages)
                for i in range(1, n_ages):
                    alive[i] = alive[i - 1] * (
                        1 - subgroup_death_share[i + specs["start_age_mortality"]]
                    )

                ax.plot(
                    subgroup_death_share.index,
                    alive,
                    label=specs["education_labels"][education]
                    + ", "
                    + specs["health_labels"][health],
                    linestyle="--" if education == 0 else "-",
                    color=colors[health],
                    alpha=(1 - 0.5 * education),
                )

        # Set title, labels, and limits
        ax.set_title(
            f"{sex_label.capitalize()} Data - Cumulated Product of the Share of Deaths by Age in Subgroups"
        )
        ax.set_xlabel(f"Age")
        ax.set_ylabel("Survival Probability")
        ax.set_xlim(specs["start_age_mortality"], specs["end_age_mortality"] + 1)
        ax.set_ylim(0, 1)

        ax.legend()
        ax.grid()
        plt.show()

    ##########################################################################################
    # Plot the in-sample deaths for different combinations for each age as stacked bar chart
    ##########################################################################################
    for sex_label in specs["sex_labels"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        filtered_df = df[df["sex"] == (0 if sex_label == "Male" else 1)]

        # Calculate total deaths by subgroups
        bar_data = {
            (education, health): filtered_df[
                (filtered_df["education"] == education)
                & (filtered_df["health"] == health)
            ]
            .groupby("age")["death event"]
            .sum()
            .reindex(age_range)
            .fillna(0)
            for education in range(n_edu_types)
            for health in alive_health_states
        }

        # Plot stacked bar chart
        bottom = np.zeros_like(age_range, dtype=float)
        for (education, health), values in bar_data.items():
            ax.bar(
                age_range,
                values,
                bottom=bottom,
                label=specs["education_labels"][education]
                + ", "
                + specs["health_labels"][health],
                color=colors[health],
                alpha=(1 - 0.5 * education),
            )
            bottom += values

        # Set title, labels, and limits
        ax.set_title(
            f"{sex_label.capitalize()} Data - Total Number of Deaths by Age and Subgroups"
        )
        ax.set_xlabel(f"Age")
        ax.set_ylabel("Total Number of Deaths")
        ax.set_xlim(specs["start_age_mortality"], specs["end_age_mortality"] + 1)

        ax.legend()
        ax.grid()
        plt.show()

    ##########################################################################################
    # Plot the in-sample deaths by Age for different Subgroups as Percentage of Total Deaths
    ##########################################################################################
    for sex_label in specs["sex_labels"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        filtered_df = df[df["sex"] == (0 if sex_label == "Male" else 1)]

        # plain share of deaths by Age
        period_death_share = filtered_df.groupby("age")["death event"].count().reindex(
            age_range
        ).fillna(0) / len(df[df["death event"] == 1])
        ax.plot(
            period_death_share.index,
            period_death_share.values,
            label=f"Reference",
            color="black",
            linestyle=":",
        )

        # share of deaths by subgroups
        for education in range(n_edu_types):
            for health in alive_health_states:
                subgroup = filtered_df[
                    (filtered_df["education"] == education)
                    & (filtered_df["health"] == health)
                ]
                subgroup_death_share = subgroup.groupby("age")[
                    "death event"
                ].count()  # find number of deaths by age
                subgroup_death_share = subgroup_death_share.reindex(age_range).fillna(
                    0
                ) / len(
                    df[df["death event"] == 1]
                )  # fill in missing ages with 0, divide by total number of deaths
                ax.plot(
                    subgroup_death_share.index,
                    subgroup_death_share.values,
                    label=specs["education_labels"][education]
                    + ", "
                    + specs["health_labels"][health],
                    linestyle="--" if education == 0 else "-",
                    color=colors[health],
                    alpha=(1 - 0.5 * education),
                )

        # Set title, labels, and limits
        ax.set_title(
            f"{sex_label.capitalize()} Data - Share of Deaths by Age and Subgroups as Percentage of Total Deaths"
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Deaths / Total Deaths (%)")
        ax.set_xlim(specs["start_age_mortality"], specs["end_age_mortality"] + 1)
        ax.set_ylim(0, 2)

        ax.legend()
        ax.grid()
        plt.show()


def survival_function(age, health_factors, params):
    """
    Calculates the survival function: Exp(-(integral of the hazard function as a function of age from 0 to age)).

    Parameters:
        age (float or array-like): The age(s) at which to calculate the survival function.
        health_factors (dict): A dictionary where keys are health-education categories
                               (e.g., 'health1_edu1') and values are their respective indicator variables (0 or 1).
        params (DataFrame): A DataFrame containing the model parameters, with 'intercept' and coefficient
                            names in the index and their values in a column named 'value'.

    Returns:
        float or array-like: The value(s) of the survival function.
    """
    # Extract coefficients
    coefficients = params["value"]
    intercept = coefficients["intercept"]
    age_coef = coefficients["age"]

    # Compute lambda using health factors
    lambda_ = np.exp(
        intercept
        + sum(coefficients[key] * value for key, value in health_factors.items() if key != "intercept")
    )

    # Compute age contribution
    age_contrib = np.exp(age_coef * age) - 1

    # Calculate the survival function
    return np.exp(-lambda_ / age_coef * age_contrib)
