# %%
import os

import pandas as pd
from autograd import numpy as np
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_by_sex
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type


# %%
def create_survival_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"], specs)


    # Pre-Filter estimation years
    df = filter_years(df, 1992, 2015)

    # Pre-Filter age and sex
    df = filter_below_age(df, 16)
    df = filter_above_age(df, 100)
    df = filter_by_sex(df, no_women=False)

    # Create education type
    df = create_education_type(df)

    # only keep the last (last = first dead one) observation for dead participant
    def filter_rows(group):
        if (group['event_death'] == 1).any():
            # Find the year of the first death_event == 1
            first_death_year = group[group['event_death'] == 1].index[0][1]
            # Filter out rows with syear greater than the first death year
            return group.loc[group.index.get_level_values('syear') <= first_death_year]
        # else:
        #     # If only death_event == 0, keep only the last observation
        #     last_syear_idx = group.index[-1]
        #     return group.loc[[last_syear_idx]] 

    df = df.groupby(level='pid', group_keys=False).apply(filter_rows)
    df = df[["age", "event_death", "education", "sex"]]

    # add age another time to the dataframe and call it duration
    df["duration"] = df["age"]
    # find NaNs
    print("NaNs in the final mortality transition sample:")
    print(df.isnull().sum())
    
    from matplotlib import pyplot as plt
    from lifelines.fitters import ParametricRegressionFitter


    class GompertzFitter(ParametricRegressionFitter):

        # this class property is necessary, and should always be a non-empty list of strings.
        _fitted_parameter_names = ['lambda_', 'gamma_']

        def _cumulative_hazard(self, params, t, Xs):
            # params is a dictionary that maps unknown parameters to a numpy vector.
            # Xs is a dictionary that maps unknown parameters to a numpy 2d array
            beta_ = params['lambda_']
            gamma = params['gamma_']
            gamma_ = params['gamma_']
            X = Xs['lambda_']
            lambda_ = np.exp(np.dot(X, beta_))
            survival = gamma_ * (1 - np.exp(-lambda_ * t))
            return (lambda_ * (np.exp(gamma_ * t) - 1)) / gamma_
        def _hazard(self, params, t, Xs):
            lambda_ = np.exp(np.dot(Xs['lambda_'], params['lambda_']))
            gamma_ = params['gamma_']
            return lambda_ * np.exp(gamma_ * t)

    regressors = {
        'lambda_': "1 + sex + education",
        'gamma_': "1",
    }

    gf = GompertzFitter()
    gf.fit(df, 'duration', 'event_death', regressors=regressors, show_progress=True, initial_point=np.array([0.087, 0.0, 0.0, 0.0]))
    gf.print_summary()

    # data frame with the estimated parameters, standard errors, etc. - index is the variable name
    params = gf.summary


    def survival_function(age, edu, sex):
        cons = params.loc['lambda_', 'Intercept']['coef']
        age_coef = params.loc['gamma_', 'Intercept']['coef']
        edu_coef = params.loc['lambda_', 'education']['coef']
        sex_coef = params.loc['lambda_', 'sex']['coef']
        lambda_ = np.exp(cons + edu*edu_coef + sex*sex_coef)
        age_contrib = np.exp(age_coef*age)-1
        return np.exp(-lambda_*age_contrib/age_coef)

    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure
    age = np.linspace(16, 100, 100-16+1)
    colors = {0: "#1E90FF", 1: "#D72638"}  # Blue for male, red for female

    for edu in df['education'].unique():
        for sex in df['sex'].unique():
            edu_label = specs["education_labels"][edu]
            sex_label = 'Male' if sex == 0 else 'Female'
            linestyle = '--' if edu == 0 else '-'
            ax.plot(age, survival_function(age, edu, sex), 
                    label=f"{edu_label}, {sex_label}", 
                    color=colors[sex], 
                    linestyle=linestyle)

    # Adjusting axes and ticks
    ax.set_title("Survival function for different educational levels and sexes")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(16, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(20, 101, 10))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Adding legend and showing plot
    ax.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()


    print(
        str(len(df))
        + " observations in the final survival transition sample.  \n ----------------"
    )

    df.to_pickle(out_file_path)
    return df


def load_and_merge_soep_core(soep_c38_path, specs):
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgpsbil",
            "pgstib",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    lifespell_data = pd.read_stata(
        f"{soep_c38_path}/lifespell.dta",
        convert_categoricals=False,
    )
    # --- Generate spell duration and expand dataset --- lifespell data
    lifespell_data['spellduration'] = (lifespell_data['end'] - lifespell_data['begin']) + 1
    lifespell_data_long = lifespell_data.loc[lifespell_data.index.repeat(lifespell_data['spellduration'])].reset_index(drop=True)
    breakpoint()
    # --- Generate syear --- lifespell data
    lifespell_data_long['n'] = lifespell_data_long.groupby(['pid', 'spellnr']).cumcount() + 1 # +1 since cumcount starts at 0
    lifespell_data_long['syear'] = lifespell_data_long['begin'] + lifespell_data_long['n'] - 1
    # --- Clean-up --- lifespell data
    lifespell_data_long = lifespell_data_long[lifespell_data_long['syear'] <= specs["end_year"] + 1]
    columns_to_keep = ["pid", "syear", "spellnr"]
    lifespell_data_long = lifespell_data_long[columns_to_keep]
    # --- Generate death event variable --- lifespell data
    lifespell_data_long['event_death'] = (lifespell_data_long['spellnr'] == 4).astype('int')
    breakpoint()
    # --- Sort and drop duplicates --- lifespell data
    lifespell_data_long.sort_values(by=['pid', 'syear', 'spellnr'], inplace=True)
    lifespell_data_long = lifespell_data_long[
        ~((lifespell_data_long['syear'].shift(-1) == lifespell_data_long['syear']) & 
        (lifespell_data_long['pid'].shift(-1) == lifespell_data_long['pid']))
    ].reset_index(drop=True)
    breakpoint()

    # merge all data
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data = pd.merge(merged_data, lifespell_data_long, on=["pid", "syear"], how="inner")
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
