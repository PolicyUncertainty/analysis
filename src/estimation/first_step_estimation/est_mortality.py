import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import estimagic as em
import optimagic as om
from specs.derive_specs import read_and_derive_specs


def estimate_mortality(paths_dict, specs):
    """Estimate the mortality matrix."""

    # # Load the data
    # df = pd.read_pickle(
    #     paths_dict["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    # )

    df = pd.read_stata("/Users/gregorschuler/GitProjects/analysis/src/estimation/first_step_estimation/data_mort_estim.dta", convert_categoricals=False)
    df.set_index(["pid", "syear"], inplace=True)

    #  rename columns event -> event_death
    df.rename(columns={"event": "event_death"}, inplace=True)
    df.rename(columns={"educ_final": "education"}, inplace=True)
    df.rename(columns={"health": "health_state"}, inplace=True)
    df["sex"] = 0 # only males in the sample

    # print number of observations
    print(str(len(df)) + " observations in the mortality estimation dataset.")

    # drop if the health state is missing
    df = df[(df["health_state"].notna())]
    print("Obs. after dropping missing health data:", len(df))

    # drop if the education is missing
    df = df[(df["education"].notna())]
    print("Obs. after dropping missing education data:", len(df))


    df = df.reset_index()
    df["begin_age"] = df.groupby("pid")["age"].transform("min")
    indx = df.groupby("pid")["syear"].idxmin()
    df["begin_health_state"] = 0
    df.loc[indx, "begin_health_state"] = df.loc[indx, "health_state"]
    df["begin_health_state"] = df.groupby("pid")["begin_health_state"].transform("max")
    df = df.set_index(["pid", "syear"])


    df = df[["age", "begin_age", "event_death", "education", "sex", "health_state", "begin_health_state"]]

    # set the dtype of the columns to float 
    df = df.astype(float)

    # Show data overview
    print(df.head())
    # sum the death events for the entire sample
    print("Death events in the sample:")
    print(df["event_death"].sum())

    # print the min and max age in the sample
    print("Min age in the sample:", df["age"].min())
    print("Max age in the sample:", df["age"].max())

    # print the average age in the sample
    print("Average age in the sample:", round(df["age"].mean(), 2))

    # print the number of unique individuals in the sample
    print("Number of unique individuals in the sample:", df.index.get_level_values("pid").nunique())

    # print the number of unique years in the sample (min and max)
    print("Sample Years:", df.index.get_level_values("syear").min(), "-", df.index.get_level_values("syear").max())

    # Average time spent in the sample for each individual
    print("Average time spent in the sample for each individual:", round(df.groupby("pid").size().mean(), 2))

    print(
        str(len(df))
        + " observations in the final survival transition sample.  \n ----------------"
    )

    breakpoint()

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
            filtered_df.groupby("age")["event_death"].mean().reindex(np.arange(16, 111)).fillna(0)
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
                    (filtered_df["education"] == edu) & (filtered_df["health_state"] == health)
                ]
                subgroup_death_share = (
                    subgroup.groupby("age")["event_death"].mean().reindex(np.arange(16, 111)).fillna(0)
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
                (filtered_df["education"] == edu) & (filtered_df["health_state"] == health)
            ].groupby("age")["event_death"].sum().reindex(age_range).fillna(0)
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
                alpha=0.7*(1 if edu == 0 else 0.5),
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
    fig.suptitle("Share of Deaths by Age and Subgroups as Percentage of Total Deaths", fontsize=11)

    for i, sex in enumerate(sexes):
        ax = axes[i]

        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        # Calculate plain share of deaths by age
        age_death_share = (
            filtered_df.groupby("age")["event_death"].count().reindex(np.arange(16, 111)).fillna(0)/len(df[df["event_death"] == 1])
        )

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
                    (filtered_df["education"] == edu) & (filtered_df["health_state"] == health)
                ]
                subgroup_death_share = (
                    subgroup.groupby("age")["event_death"].count().reindex(np.arange(16, 111)).fillna(0)/len(df[df["event_death"] == 1])
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

    # breakpoint()

    # Set up the figure and axes for survival function
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 subplots
    fig.suptitle("Survival Function by Sex and Age Setting", fontsize=16)
    for i, sex in enumerate(sexes):
        for j, set_age in enumerate(age_settings):
            ax = axes[i, j]

            # Filter data by sex
            filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

            

            # Placeholder for global Iteration variable (if required)
            global Iteration
            Iteration = 0

            # Define start parameters
            start_params = pd.DataFrame(
                data=[[0.087, 1e-8, 1], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf]],
                columns=["value", "lower_bound", "upper_bound"],
                index=["age", "education", "health_state", "intercept"],
            )

            # Estimate parameters
            res = em.estimate_ml(
                loglike=loglike,
                params=start_params,
                optimize_options={"algorithm": "scipy_lbfgsb"},
                loglike_kwargs={"data": filtered_df},
            )

            # Generate age range
            age = np.linspace(16, 110, 110 - 16 + 1)

            # Plot survival functions for combinations of education and health state
            for edu in filtered_df["education"].unique():
                for health in filtered_df["health_state"].unique():
                    edu_label = specs["education_labels"][int(edu)]
                    health_label = "Bad Health" if health == 0 else "Good Health"
                    linestyle = "--" if int(edu) == 0 else "-"

                    ax.plot(
                        age,
                        survival_function(age, int(edu), health, res.params, set_age=set_age),
                        label=f"{edu_label}, {health_label}",
                        color=colors[health],
                        linestyle=linestyle,
                    )

            # Set title, labels, and limits
            age_label = "Custom Set Age Coef" if set_age else "Estimated Age Coef"
            ax.set_title(f"{sex.capitalize()}, {age_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Survival Probability")
            ax.set_xlim(16, 110)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(20, 101, 10))
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

            # Add legend
            ax.legend(fontsize=8, loc="lower left")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()

    
    breakpoint()

    return res.params


def hazard_function(age, edu, health, params):
        cons = params.loc["intercept", "value"]
        age_coef = params.loc["age", "value"]
        edu_coef = params.loc["education", "value"]
        health_coef = params.loc["health_state", "value"]

        lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
        age_contrib = np.exp(age_coef * age)

        return lambda_ * age_contrib


def survival_function(age, edu, health, params, set_age=False):
    """
    exp(-(integral of the hazard function as a function of age from 0 to age)) 
    """

    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    if set_age:
        age_coef = 0.04
    edu_coef = params.loc["education", "value"]
    health_coef = params.loc["health_state", "value"]

    lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
    age_contrib = np.exp(age_coef * age) - 1

    # print lambda and the first and last age_contrib as well as the - lambda_ / age_coef * age_contrib for the first and last age and the exp of that
    print("lambda:", lambda_)
    print("age_contrib first:", age_contrib[0])
    print("age_contrib last:", age_contrib[-1])
    print("- lambda_ / age_coef * age_contrib first:", - lambda_ / age_coef * age_contrib[0])
    print("- lambda_ / age_coef * age_contrib last:", - lambda_ / age_coef * age_contrib[-1])
    print("exp(- lambda_ / age_coef * age_contrib) first:", np.exp(- lambda_ / age_coef * age_contrib[0]))
    print("exp(- lambda_ / age_coef * age_contrib) last:", np.exp(- lambda_ / age_coef * age_contrib[-1]))

    return np.exp(- lambda_ / age_coef * age_contrib)


def density_function(age, edu, health, params):
    """
    d[-S(age)]/d(age) = - dS(age)/d(age) 
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    edu_coef = params.loc["education", "value"]
    health_coef = params.loc["health_state", "value"]

    lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
    age_contrib = np.exp(age_coef*age) - 1

    return lambda_ * np.exp(age_coef*age - ((lambda_ * age_contrib) / age_coef))

def log_density_function(age, edu, health, params):

    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    edu_coef = params.loc["education", "value"]
    health_coef = params.loc["health_state", "value"]


    lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
    log_lambda_ = cons + edu_coef*edu + health_coef*health
    age_contrib = np.exp(age_coef*age) - 1

    return log_lambda_ + age_coef*age - ((lambda_ * age_contrib) / age_coef)

def log_survival_function(age, edu, health, params):
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    edu_coef = params.loc["education", "value"]
    health_coef = params.loc["health_state", "value"]

    lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
    age_contrib = np.exp(age_coef*age) - 1

    return - (lambda_ * age_contrib) / age_coef

def loglike(params, data):

        begin_age = data["begin_age"]
        begin_health_state = data["begin_health_state"]
        age = data["age"]
        edu = data["education"]
        health = data["health_state"]
        event = data["event_death"]
        death = data["event_death"].astype(bool)
        
        # initialize contributions as an array of zeros
        contributions = np.zeros_like(age)

        # calculate contributions
        contributions[death] = log_density_function(age[death], edu[death], health[death], params)
        contributions[~death] = log_survival_function(age[~death], edu[~death], health[~death], params)
        contributions -= log_survival_function(begin_age, edu, begin_health_state, params)

        # print the death and not death contributions
        print("Iteration:", globals()['Iteration'])
        print("Death contributions:", contributions[death].sum())
        print("Not death contributions:", contributions[~death].sum())
        print("Total contributions:", contributions.sum())
        
        globals()['Iteration'] += 1
        if globals()['Iteration'] % 100 == 0:
            print(params)
        
        

        return {"contributions": contributions, "value": contributions.sum()}