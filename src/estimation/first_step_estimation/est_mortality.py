import itertools
import estimagic as em
import matplotlib.pyplot as plt
import numpy as np
import optimagic as om
import pandas as pd
from specs.derive_specs import read_and_derive_specs


def estimate_mortality(paths_dict, specs):
    """Estimate the mortality matrix."""

    # load life table data and expand/duplicate it to include all possible combinations of health, education and sex
    lifetable_df = pd.read_csv(
        paths_dict["intermediate_data"] + "mortality_table_for_pandas.csv",
        sep=";",
    )
    mortality_df = pd.DataFrame(
        [
            {
                "age": row["age"],
                "health": combo[0],
                "education": combo[1],
                "sex": combo[2],
                "death_prob": row["death_prob_male"] if combo[2] == 0 else row["death_prob_female"],  # male (0) or female (1) death prob
            }
            for _, row in lifetable_df.iterrows()
            for combo in list(itertools.product([0, 1], repeat=3))  # (health, education, sex)
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

    # df initial values i.e. first observations (+ sex column) 
    start_df = df[[col for col in df.columns if col.startswith("start_")] + ['sex']].copy()
    str_cols = start_df.columns.str.replace("start_", "")
    start_df.columns = str_cols


    # add a intercept column to the df and start_df
    df["intercept"] = 1
    start_df["intercept"] = 1


    for i, sex in enumerate(specs["sexes"]):
        # Filter data by sex
        filtered_df = df[df["sex"] == (0 if sex == "Male" else 1)]
        filtered_start_df = start_df[start_df["sex"] == (0 if sex == "Male" else 1)]

        global Iteration
        Iteration = 0

        # Define start parameters
        initial_params = pd.DataFrame(
            data=[
                [-13.21, -np.inf, np.inf],
                [0.10, 1e-8, np.inf],
                [-0.77, -np.inf, np.inf],
                [-0.30, -np.inf, np.inf],
                [0.01, -np.inf, np.inf],
                [0.36, -np.inf, np.inf],
            ],
            columns=["value", "lower_bound", "upper_bound"],
            index=[
                "intercept",
                "age",
                f"{specs['health_labels'][1]}_{specs['education_labels'][1]}".replace(' ', ''), # good health, high education
                f"{specs['health_labels'][1]}_{specs['education_labels'][0]}".replace(' ', ''), # good health, low education
                f"{specs['health_labels'][0]}_{specs['education_labels'][1]}".replace(' ', ''), # bad health, high education
                f"{specs['health_labels'][0]}_{specs['education_labels'][0]}".replace(' ', ''), # bad health, low education
            ],
        )

        # Estimate parameters
        res = em.estimate_ml(
            loglike=loglike,
            params=initial_params,
            optimize_options={"algorithm": "scipy_lbfgsb"},
            loglike_kwargs={"df": filtered_df, "start_df": filtered_start_df},
        )

        # update mortality_df with the estimated parameters
        for health in specs["health_labels"]:
            for education in specs["education_labels"]:
                param = f"{specs['health_labels'][1]}_{specs['education_labels'][1]}".replace(' ', '')
                mortality_df.loc[
                    (mortality_df["sex"] == (0 if sex == "Male" else 1))
                    & (mortality_df["health"] == health)
                    & (mortality_df["education"] == education),
                    "death_prob",
                ] *= np.exp(res.params.loc[param, "value"])

        # save the results
        print(res.summary())
        print(res.optimize_result)
        res.summary().to_pickle(paths_dict["est_results"] + f"est_params_mortality_{sex}.pkl")
        to_csv_summary = res.summary()
        to_csv_summary["hazard_ratio"] = np.exp(res.params["value"])
        to_csv_summary.to_csv(paths_dict["est_results"] + f"est_params_mortality_{sex}.csv")

    
    # export the estimated mortality table and the original life table as csv
    lifetable_df = lifetable_df[
        (lifetable_df["age"] >= specs["start_age_mortality"])
        & (lifetable_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[
        (mortality_df["age"] >= specs["start_age_mortality"])
        & (mortality_df["age"] <= specs["end_age_mortality"])
    ]
    mortality_df = mortality_df[["age", "sex", "health", "education", "death_prob"]]
    lifetable_df = lifetable_df[["age", "sex", "death_prob"]]
    mortality_df = mortality_df.astype(
        {
            "age": "int",
            "sex": "int",
            "health": "int",
            "education": "int",
            "death_prob": "float",
        }
    )
    lifetable_df = lifetable_df.astype(
        {
            "age": "int",
            "sex": "int",
            "death_prob": "float",
        }
    )
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

def log_density_function(data, params):
    """
    Calculate the log-density function: log of the density function. (log of d[-S(age)]/d(age) = log of - dS(age)/d(age))
    """
    age_coef = params.loc["age", "value"]
    age_contrib = np.exp(age_coef * data["age"]) - 1

    # breakpoint()

    log_lambda_ = sum([params.loc[x, "value"] * data[x] for x in params.index if x != "age"])
    lambda_ = np.exp(log_lambda_)
    
    return log_lambda_ + age_coef * data["age"] - ((lambda_ * age_contrib) / age_coef)

def log_survival_function(data, params):
    """
    Calculate the log-survival function.
    """
    age_coef = params.loc["age", "value"]
    age_contrib = np.exp(age_coef * data["age"]) - 1

    # breakpoint()

    lambda_ = np.exp(sum([params.loc[x, "value"] * data[x] for x in params.index if x != "age"]))

    return -(lambda_ * age_contrib) / age_coef

def loglike(params, df, start_df):
    """
    Log-likelihood calculation.
    params: pd.DataFrame
        DataFrame with the parameters.
    df: pd.DataFrame
        DataFrame with the data.
    param_names: list
        List with the parameter names. (includes the intercept at index 0)
    """

    event = df["event_death"]
    death = event.astype(bool)

    contributions = np.zeros_like(event)
    
    contributions[death]   = log_density_function(df[death], params)
    contributions[~death]  = log_survival_function(df[~death], params)
    contributions         -= log_survival_function(start_df, params)
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
