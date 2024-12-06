# %%
import os
import estimagic as em
import optimagic as om
import pandas as pd
import numpy as np
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_by_sex
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import clean_health_create_states
from process_data.soep_vars.health import create_health_var


# %%
def create_survival_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_datasets(paths["soep_c38"], specs)

    # filter age and estmation years
    df = filter_below_age(df, 16)
    df = filter_above_age(df, 110)
    df = filter_years(df, specs["start_year_mortality"], specs["end_year_mortality"])
    # only use male data
    df = filter_by_sex(df, no_women=True)


   

    """"
    sort pid syear
	* --- fill gaps:
		forval y=1/22 {
			dis `y'
			forval x=1/20 {
				by pid: replace health = health[_n+`x'] if mi(health[_n]) & !mi(health[_n+`x']) & !mi(health[_n-1]) & health[_n-1]==health[_n+`x'] 
			}
		}	
	
	tab health_orig health, mis
	
	tab health event, mis
	tab health_orig event, mis
		
	* --- expand post-observed
		sort pid syear
		by pid: replace health = health[_n-1] if !mi(health[_n-1]) & mi(health[_n]) // & health[_n-1]==1	
    """

    # Stata code, IF: 
    # The current value of health is missing (mi(health[_n])).
    # The value x observations ahead (health[_n+x']) is not missing (!mi(health[_n+x'])). AND
    # The value immediately before the current observation (health[_n-1]) is not missing (!mi(health[_n-1])). AND
    # The values immediately before the current observation (health[_n-1]) and x observations ahead (health[_n+x']) are equal (health[_n-1] == health[_n+x']). AND
    # THEN:  the missing value of health at the current position (_n) is replaced by the value of health[_n+x']`. AND

    # # Sort the DataFrame by pid and syear
    # df = df.sort_index()

    # its = 0
    # # Loop over each individual (pid)
    # for pid, group in df.groupby(level=0):
    #     its += 1
    #     print(its)
    #     # Extract the individual's data as a series
    #     health = group["health_state"].copy()
    #     # Fill gaps for the individual
    #     for x in range(1, 5):  # Check up to 20 steps ahead
    #         for i in range(len(health)):
    #             # Apply the filling logic
    #             if pd.isna(health.iloc[i]) and i + x < len(health):
    #                 if not pd.isna(health.iloc[i - 1]) and not pd.isna(health.iloc[i + x]):
    #                     if health.iloc[i - 1] == health.iloc[i + x]:
    #                         health.iloc[i] = health.iloc[i + x]
        
    #     # Assign the updated series back to the original DataFrame
    #     df.loc[(pid, slice(None)), "health_state"] = health




    print("Obs. after filling health gaps:", len(df))
    # drop if the health state is missing
    df = df[(df["health_state"].notna())]
    print("Obs. after dropping missing health data:", len(df))


    # make the index columns
    df = df.reset_index()

    # for every pid find first year in the sample observations
    df["begin_age"] = df.groupby("pid")["age"].transform("min")
    indx = df.groupby("pid")["syear"].idxmin()
    df["begin_health_state"] = 0
    df.loc[indx, "begin_health_state"] = df.loc[indx, "health_state"]
    df["begin_health_state"] = df.groupby("pid")["begin_health_state"].transform("max")

    # make the pid and syear the index again
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



    # start estimation

    def hazard_function(age, edu, health, params):
        cons = params.loc["intercept", "value"]
        age_coef = params.loc["age", "value"]
        edu_coef = params.loc["education", "value"]
        health_coef = params.loc["health_state", "value"]

        lambda_ = np.exp(cons + edu_coef*edu + health_coef*health)
        age_contrib = np.exp(age_coef * age)

        return lambda_ * age_contrib

    
    def survival_function(age, edu, health, params):
        """
        exp(-(integral of the hazard function as a function of age from 0 to age)) 
        """
    
        cons = params.loc["intercept", "value"]
        age_coef = params.loc["age", "value"]
        # age_coef = 0.087
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

    global Iteration 
    Iteration = 0

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

    start_params = pd.DataFrame(
        data=[[0.087, 1e-8, 1], [0.0,  -np.inf, np.inf], [0.0,  -np.inf, np.inf], [0.0, -np.inf, np.inf]],
        columns=["value", "lower_bound", "upper_bound"],
        index=["age", "education", "health_state", "intercept"],
    )

    res = em.estimate_ml(
        loglike=loglike,
        params=start_params,
        optimize_options={"algorithm": "scipy_lbfgsb"},
        loglike_kwargs={"data": df},
    )

    print(res.summary())

    from matplotlib import pyplot as plt



    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure
    age = np.linspace(16, 110, 110 - 16 + 1)
    colors = {0: "#1E90FF", 1: "#D72638"}  # Blue for male, red for female

    for edu in df["education"].unique():
        for health in df["health_state"].unique():
            edu_label = specs["education_labels"][int(edu)]
            health_label = "Bad Health" if health == 0 else "Good Health"
            linestyle = "--" if int(edu) == 0 else "-"
            ax.plot(
                age,
                survival_function(age, int(edu), health, res.params),
                label=f"{edu_label}, {health_label}",
                color=colors[health],
                linestyle=linestyle,
            )

    # Adjusting axes and ticks
    ax.set_title("Survival function for different educational levels and sexes")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(16, 110)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(20, 101, 10))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Adding legend and showing plot
    ax.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()

    breakpoint()

    print(
        str(len(df))
        + " observations in the final survival transition sample.  \n ----------------"
    )

    df.to_pickle(out_file_path)
    return df


def load_and_merge_datasets(soep_c38_path, specs):
    annual_survey_data = load_and_process_soep_yearly_survey_data(soep_c38_path, specs)
    life_spell_data = load_and_process_life_spell_data(soep_c38_path, specs)
    health_data = load_and_process_soep_health(soep_c38_path, specs)
    df = pd.merge(annual_survey_data, life_spell_data, on=["pid", "syear"], how="inner")
    df = pd.merge(df, health_data, on=["pid", "syear"], how="inner")
    df = df.set_index(["pid", "syear"])
    return df 


def load_and_process_soep_yearly_survey_data(soep_c38_path, specs):
    """Load the annual data from the SOEP C38 dataset and process it."""
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(pgen_data, ppathl_data, on=["pid", "syear"], how="inner")

    merged_data.set_index(["pid", "syear"], inplace=True)

    # Pre-Filter estimation years
    df = filter_years(merged_data, specs["start_year_mortality"], specs["end_year_mortality"])
    df = filter_by_sex(df, no_women=False)
    # Create education type
    df = create_education_type(df)

    full_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [df.index.get_level_values("pid").unique(), range(1992, 2017)],
            names=["pid", "syear"],
        ),
        columns=["sex", "education", "gebjahr"],
    )
    full_df.update(df)
    full_df["education"] = full_df.groupby("pid")["education"].transform("max")
    full_df["sex"] = full_df.groupby("pid")["sex"].transform("max")
    full_df["gebjahr"] = full_df.groupby("pid")["gebjahr"].transform("max")
    full_df["age"] = full_df.index.get_level_values("syear") - full_df["gebjahr"]
    full_df.drop("gebjahr", axis=1, inplace=True)

    # Pre-Filter age and sex
    full_df = filter_below_age(full_df, 16)
    full_df = filter_above_age(full_df, 110)

    return full_df


def load_and_process_life_spell_data(soep_c38_path, specs):
    lifespell_data = pd.read_stata(
        f"{soep_c38_path}/lifespell.dta",
        convert_categoricals=False,
    ).drop(
        [
            "zensor",
            "info",
            "study1992",
            "study2001",
            "study2006",
            "study2008",
            "flag1",
            "immiyearinfo",
        ],
        axis=1,
    )
    # --- Generate spell duration and expand dataset --- lifespell data
    lifespell_data["spellduration"] = (
        lifespell_data["end"] - lifespell_data["begin"]
    ) + 1
    lifespell_data_long = lifespell_data.loc[
        lifespell_data.index.repeat(lifespell_data["spellduration"])
    ].reset_index(drop=True)
    # --- Generate syear --- lifespell data
    lifespell_data_long["n"] = (
        lifespell_data_long.groupby(["pid", "spellnr"]).cumcount() + 1
    )  # +1 since cumcount starts at 0
    lifespell_data_long["syear"] = (
        lifespell_data_long["begin"] + lifespell_data_long["n"] - 1
    )
    # --- Clean-up --- lifespell data
    lifespell_data_long = lifespell_data_long[
        lifespell_data_long["syear"] <= specs["end_year"] + 1
    ]
    columns_to_keep = ["pid", "syear", "spellnr"]
    lifespell_data_long = lifespell_data_long[columns_to_keep]
    # --- Generate death event variable --- lifespell data
    lifespell_data_long["event_death"] = (lifespell_data_long["spellnr"] == 4).astype(
        "int"
    )

    # Split into dataframes of death and not death
    not_death_idx = lifespell_data_long[lifespell_data_long["event_death"] == 0].index
    first_death_idx = (
        lifespell_data_long[lifespell_data_long["event_death"] == 1]
        .groupby("pid")["syear"]
        .idxmin()
    )

    # Final index and df
    final_index = not_death_idx.union(first_death_idx)
    lifespell_data_long = lifespell_data_long.loc[final_index]
    return lifespell_data_long


def load_and_process_soep_health(soep_c38_path, specs):

    pequiv_data = pd.read_stata(
    # m11126: Self-Rated Health Status
    # m11124: Disability Status of Individual
    f"{soep_c38_path}/pequiv.dta",
    columns=["pid", "syear", "m11126", "m11124"],
    convert_categoricals=False,
    )
    pequiv_data.set_index(["pid", "syear"], inplace=True)

    # create health states
    pequiv_data = filter_years(pequiv_data, specs["start_year_mortality"] - 1, specs["end_year_mortality"] + 1)
    pequiv_data = create_health_var(pequiv_data)
    pequiv_data = span_dataframe(pequiv_data, specs["start_year_mortality"] - 1, specs["end_year_mortality"] + 1)
    pequiv_data = clean_health_create_states(pequiv_data)

    return pequiv_data

