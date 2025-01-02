import itertools
import estimagic as em
import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
from specs.derive_specs import read_and_derive_specs


def estimate_mortality(paths_dict, specs):
    """Estimate the mortality matrix."""

    # load life table data and expand it to include all possible combinations of health, education and sex
    df = pd.read_csv(
        paths_dict["intermediate_data"] + "mortality_table_for_pandas.csv",
        sep=";",
    )
    combinations = list(itertools.product([0, 1], repeat=3))  # (health, education, sex)
    mortality_df = pd.DataFrame(
        [
            {
                "age": row["age"],
                "death_prob": row["death_prob_male"]
                if combo[2] == 0
                else row[
                    "death_prob_female"
                ],  # Use male or female death prob based on sex
                "health": combo[0],
                "education": combo[1],
                "sex": combo[2],
            }
            for _, row in df.iterrows()
            for combo in combinations
        ]
    )
    mortality_df.reset_index(drop=True, inplace=True)

    # plain life table data
    lifetable_df = mortality_df[["age", "sex", "death_prob"]]
    lifetable_df.drop_duplicates(inplace=True)

    # estimation sample - as in Kroll Lampert 2008 / Haan Schaller et al. 2024
    df = pd.read_pickle(
        paths_dict["intermediate_data"]
        + "mortality_transition_estimation_sample_duplicated.pkl"
    )

    sexes = ["male", "female"]
    combinations_health_education = [
        (1, 1, "health1_edu1"),
        (1, 0, "health1_edu0"),
        (0, 1, "health0_edu1"),
        (0, 0, "health0_edu0"),
    ]

    for i, sex in enumerate(sexes):
        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

        global Iteration
        Iteration = 0

        # Define start parameters
        start_params = pd.DataFrame(
            data=[
                [0.10, 1e-8, np.inf],
                [-0.77, -np.inf, np.inf],
                [-0.30, -np.inf, np.inf],
                [0.01, -np.inf, np.inf],
                [0.36, -np.inf, np.inf],
                [-13.21, -np.inf, np.inf],
            ],
            columns=["value", "lower_bound", "upper_bound"],
            index=[
                "age",
                "health1_edu1",
                "health1_edu0",
                "health0_edu1",
                "health0_edu0",
                "intercept",
            ],
        )

        # Estimate parameters
        res = em.estimate_ml(
            loglike=loglike,
            params=start_params,
            optimize_options={"algorithm": "scipy_lbfgsb"},
            loglike_kwargs={"data": filtered_df},
        )

        # update mortality_df with the estimated parameters
        for health, education, param in combinations_health_education:
            mortality_df.loc[
                (mortality_df["sex"] == (0 if sex == "male" else 1))
                & (mortality_df["health"] == health)
                & (mortality_df["education"] == education),
                "death_prob",
            ] *= np.exp(res.params.loc[param, "value"])

        # save the results
        print(res.summary())
        print(res.optimize_result)
        res.summary().to_pickle(paths_dict["est_results"] + f"est_params_mortality_{sex}.pkl")

    
    # export the estimated mortality table and the original life table as csv
    # restrict the age range
    lifetable_df = lifetable_df[
        (lifetable_df["age"] >= specs["start_age_mortality"])
        & (lifetable_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[
        (mortality_df["age"] >= specs["start_age_mortality"])
        & (mortality_df["age"] <= specs["end_age_mortality"])
    ]
    # order columns
    mortality_df = mortality_df[["age", "sex", "health", "education", "death_prob"]]
    lifetable_df = lifetable_df[["age", "sex", "death_prob"]]
    # save to csv
    mortality_df.to_csv(
        paths_dict["est_results"] + "mortality_transition_matrix.csv",
        sep=",",
        index=False,
    )
    lifetable_df.to_csv(
        paths_dict["est_results"] + "lifetable.csv",
        sep=",",
        index=False,
    )

def log_density_function(age, health_factors, params):
    """
    Calculate the log-density function: log of the density function. (log of d[-S(age)]/d(age) = log of - dS(age)/d(age))
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    coefficients = params["value"]

    # Compute lambda and log-lambda using health factors
    lambda_ = np.exp(
        cons + sum(coefficients[f"health{i}_edu{j}"] * factor for i, j, factor in zip([1, 1, 0, 0], [1, 0, 1, 0], health_factors))
    )
    log_lambda_ = cons + sum(
        coefficients[f"health{i}_edu{j}"] * factor for i, j, factor in zip([1, 1, 0, 0], [1, 0, 1, 0], health_factors)
    )
    age_contrib = np.exp(age_coef * age) - 1

    return log_lambda_ + age_coef * age - ((lambda_ * age_contrib) / age_coef)

def log_survival_function(age, health_factors, params):
    """
    Calculate the log-survival function.
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    coefficients = params["value"]

    # Compute lambda using health factors
    lambda_ = np.exp(
        cons + sum(coefficients[f"health{i}_edu{j}"] * factor for i, j, factor in zip([1, 1, 0, 0], [1, 0, 1, 0], health_factors))
    )
    age_contrib = np.exp(age_coef * age) - 1

    return -(lambda_ * age_contrib) / age_coef

def loglike(params, data):
    """
    Log-likelihood calculation.
    """
    start_age = data["start_age"]
    age = data["age"]
    event = data["event_death"]
    death = event.astype(bool)

    # Extract health factors
    health_factors = [data[f"health{i}_edu{j}"] for i, j in zip([1, 1, 0, 0], [1, 0, 1, 0])]
    start_health_factors = [data[f"start_health{i}_edu{j}"] for i, j in zip([1, 1, 0, 0], [1, 0, 1, 0])]

    # Initialize contributions as an array of zeros
    contributions = np.zeros_like(age)

    # Calculate contributions
    contributions[death] = log_density_function(age[death], [f[death] for f in health_factors], params)
    contributions[~death] = log_survival_function(age[~death], [f[~death] for f in health_factors], params)
    contributions -= log_survival_function(start_age, [f for f in start_health_factors], params)

    # show progress every 20 iterations
    globals()["Iteration"] += 1
    if globals()["Iteration"] % 20 == 0:
        print(
            "Iteration:",
            globals()["Iteration"],
            "Total contributions:",
            contributions.sum(),
        )
    return {"contributions": contributions, "value": contributions.sum()}

