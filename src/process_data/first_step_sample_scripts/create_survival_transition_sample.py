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

    df = load_and_merge_datasets(paths["soep_c38"], specs)

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
        _fitted_parameter_names = ["lambda_", "gamma_"]

        def _cumulative_hazard(self, params, t, Xs):
            # params is a dictionary that maps unknown parameters to a numpy vector.
            # Xs is a dictionary that maps unknown parameters to a numpy 2d array
            beta_ = params["lambda_"]
            gamma = params["gamma_"]
            gamma_ = params["gamma_"]
            X = Xs["lambda_"]
            lambda_ = np.exp(np.dot(X, beta_))
            return (lambda_ * (np.exp(gamma_ * t) - 1)) / gamma_

        def _hazard(self, params, t, Xs):
            lambda_ = np.exp(np.dot(Xs["lambda_"], params["lambda_"]))
            gamma_ = params["gamma_"]
            return lambda_ * np.exp(gamma_ * t)

    regressors = {
        "lambda_": "1 + sex + education",
        "gamma_": "1",
    }

    gf = GompertzFitter()
    gf.fit(
        df,
        "duration",
        "event_death",
        regressors=regressors,
        show_progress=True,
        initial_point=np.array([0.087, 0.0, 0.0, 0.0]),
    )
    gf.print_summary()

    # data frame with the estimated parameters, standard errors, etc. - index is the variable name
    params = gf.summary

    def survival_function(age, edu, sex):
        cons = params.loc["lambda_", "Intercept"]["coef"]
        age_coef = params.loc["gamma_", "Intercept"]["coef"]
        edu_coef = params.loc["lambda_", "education"]["coef"]
        sex_coef = params.loc["lambda_", "sex"]["coef"]
        lambda_ = np.exp(cons + edu * edu_coef + sex * sex_coef)
        age_contrib = np.exp(age_coef * age) - 1
        return np.exp(-lambda_ * age_contrib / age_coef)

    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure
    age = np.linspace(16, 100, 100 - 16 + 1)
    colors = {0: "#1E90FF", 1: "#D72638"}  # Blue for male, red for female

    for edu in df["education"].unique():
        for sex in df["sex"].unique():
            edu_label = specs["education_labels"][edu]
            sex_label = "Male" if sex == 0 else "Female"
            linestyle = "--" if edu == 0 else "-"
            ax.plot(
                age,
                survival_function(age, edu, sex),
                label=f"{edu_label}, {sex_label}",
                color=colors[sex],
                linestyle=linestyle,
            )

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
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()

    print(
        str(len(df))
        + " observations in the final survival transition sample.  \n ----------------"
    )

    df.to_pickle(out_file_path)
    return df


def load_and_merge_datasets(soep_c38_path, specs):
    annual_survey_data = load_and_process_soep_yearly_survey_data(soep_c38_path, specs)
    life_spell_data = load_and_process_life_spell_data(soep_c38_path, specs)

    return pd.merge(
        annual_survey_data, life_spell_data, on=["pid", "syear"], how="inner"
    )


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

    # ToDo: This has to go to the specs
    start_year_mortality = 1992
    end_year_mortality = 2016
    # Pre-Filter estimation years
    df = filter_years(merged_data, start_year_mortality, end_year_mortality)
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
    full_df = filter_above_age(full_df, 100)

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
