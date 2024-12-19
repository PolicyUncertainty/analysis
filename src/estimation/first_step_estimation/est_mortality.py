import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import estimagic as em
import optimagic as om
from specs.derive_specs import read_and_derive_specs


def estimate_mortality(paths_dict, specs):
    """Estimate the mortality matrix."""


    # load life table data and expand it to include all possible combinations of health, education and sex
    df = pd.read_csv(
        paths_dict["intermediate_data"] + "mortality_table_for_pandas.csv",
        sep=";",
    )
    combinations = list(itertools.product([0, 1], repeat=3))  # (health, education, sex)
    mortality_df = pd.DataFrame([
        {
            'age': row['age'],
            'death_prob': row['death_prob_male'] if combo[2] == 0 else row['death_prob_female'],  # Use male or female death prob based on sex
            'health': combo[0],
            'education': combo[1],
            'sex': combo[2]
        }
        for _, row in df.iterrows()
        for combo in combinations
    ])
    mortality_df.reset_index(drop=True, inplace=True)
    

    # keep a copy of the original data for plotting
    lifetable_df = mortality_df[['age', 'sex', 'death_prob']]
    lifetable_df.drop_duplicates(inplace=True)


    # Load the data for estimation of hazard ratios for different health and education levels
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "mortality_transition_estimation_sample_duplicated.pkl"
    )
   
    # Define parameters for subplots
    sexes = ["male", "female"]

    combinations = [
        (1, 1, "health1_edu1"),
        (1, 0, "health1_edu0"),
        (0, 1, "health0_edu1"),
        (0, 0, "health0_edu0")
    ]


    for i, sex in enumerate(sexes):
       
            # Filter data by sex
            filtered_df = df[df["sex"] == (0 if sex == "male" else 1)]

            # Placeholder for global Iteration variable (if required)
            global Iteration
            Iteration = 0

            # Define start parameters
            start_params = pd.DataFrame(
                data=[[0.112323, 1e-8, np.inf], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf], [0.0, -np.inf, np.inf]],
                columns=["value", "lower_bound", "upper_bound"],
                index=["age", "health1_edu1", "health1_edu0", "health0_edu1", "health0_edu0", "intercept"],
            )

            # Estimate parameters
            res = em.estimate_ml(
                loglike=loglike,
                params=start_params,
                optimize_options={"algorithm": "scipy_lbfgsb"},
                loglike_kwargs={"data": filtered_df},
            )

            # update mortality_df with the estimated parameters
            for health, education, param in combinations:
                mortality_df.loc[
                    (mortality_df["sex"] == (0 if sex == "male" else 1)) &
                    (mortality_df["health"] == health) &
                    (mortality_df["education"] == education),
                    "death_prob"
                ] *= np.exp(res.params.loc[param, "value"])

            print(res.summary())

            print(res.optimize_result)

    # export the estimated mortality table and the original life table as csv

    # restrict the age range
    lifetable_df = lifetable_df[(lifetable_df["age"] >= specs["start_age_mortality"]) & (lifetable_df["age"] <= specs["end_age_mortality"])]
    mortality_df = mortality_df[(mortality_df["age"] >= specs["start_age_mortality"]) & (mortality_df["age"] <= specs["end_age_mortality"])]
    # convert age to period
    lifetable_df["period"] = lifetable_df["age"] - specs["start_age_mortality"]
    mortality_df["period"] = mortality_df["age"] - specs["start_age_mortality"]
    # order columns
    mortality_df = mortality_df[["period", "sex", "health", "education", "death_prob"]]
    lifetable_df = lifetable_df[["period", "sex", "death_prob"]]
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

      
            

def hazard_function(age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params):
    """
    P(x<X<x+dx | X>x) / dx 
    force of mortality aka hazard function aka instantaneous rate of mortality at a certain age
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]

    lambda_ = np.exp(cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0)
    age_contrib = np.exp(age_coef * age)

    return lambda_ * age_contrib


def survival_function(age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params, set_age=False):
    """
    exp(-(integral of the hazard function as a function of age from 0 to age)) 
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]
    
    lambda_ = np.exp(cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0)
    age_contrib = np.exp(age_coef * age) - 1

    return np.exp(- lambda_ / age_coef * age_contrib)


def density_function(age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params):
    """
    d[-S(age)]/d(age) = - dS(age)/d(age) 
    """
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]

    lambda_ = np.exp(cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0)
    age_contrib = np.exp(age_coef*age) - 1

    return lambda_ * np.exp(age_coef*age - ((lambda_ * age_contrib) / age_coef))

def log_density_function(age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params):
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]

    lambda_ = np.exp(cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0)
    log_lambda_ = cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0
    age_contrib = np.exp(age_coef*age) - 1

    return log_lambda_ + age_coef*age - ((lambda_ * age_contrib) / age_coef)

def log_survival_function(age, health1_edu1, health1_edu0, health0_edu1, health0_edu0, params):
    cons = params.loc["intercept", "value"]
    age_coef = params.loc["age", "value"]
    health1_edu1_coef = params.loc["health1_edu1", "value"]
    health1_edu0_coef = params.loc["health1_edu0", "value"]
    health0_edu1_coef = params.loc["health0_edu1", "value"]
    health0_edu0_coef = params.loc["health0_edu0", "value"]
    
    lambda_ = np.exp(cons + health1_edu1_coef*health1_edu1 + health1_edu0_coef*health1_edu0 + health0_edu1_coef*health0_edu1 + health0_edu0_coef*health0_edu0)
    age_contrib = np.exp(age_coef*age) - 1

    return - (lambda_ * age_contrib) / age_coef

def loglike(params, data):

        start_age = data["start_age"]
        age = data["age"]
        event = data["event_death"]
        death = data["event_death"].astype(bool)
        health1_edu1 = data["health1_edu1"]
        health1_edu0 = data["health1_edu0"]
        health0_edu1 = data["health0_edu1"]
        health0_edu0 = data["health0_edu0"]
        start_health0_edu0 = data["start_health0_edu0"]
        start_health0_edu1 = data["start_health0_edu1"]
        start_health1_edu0 = data["start_health1_edu0"]
        start_health1_edu1 = data["start_health1_edu1"]

        
        # initialize contributions as an array of zeros
        contributions = np.zeros_like(age)

        # calculate contributions
        contributions[death] = log_density_function(age[death], health1_edu1[death], health1_edu0[death], health0_edu1[death], health0_edu0[death], params)
        contributions[~death] = log_survival_function(age[~death], health1_edu1[~death], health1_edu0[~death], health0_edu1[~death], health0_edu0[~death], params)
        contributions -= log_survival_function(start_age, start_health1_edu1, start_health1_edu0, start_health0_edu1, start_health0_edu0, params)

        # print the death and not death contributions
        print("Iteration:", globals()['Iteration'], "Total contributions:", contributions.sum())
        
        globals()['Iteration'] += 1
        if globals()['Iteration'] % 200 == 0:
            print(params)
        
        

        return {"contributions": contributions, "value": contributions.sum()}