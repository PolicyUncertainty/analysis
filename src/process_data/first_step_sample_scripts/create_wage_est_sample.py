import os

import pandas as pd

from process_data.auxiliary.filter_data import filter_data
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import sum_experience_variables
from process_data.soep_vars.work_choices import create_choice_and_employment_status


def create_wage_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["first_step_data"] + "wage_estimation_sample.csv"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Load and merge data state data from SOEP core (all but wealth)
    df = load_and_merge_soep_core(paths["soep_c38"])

    # filter data (age, sex, estimation period)
    df = filter_data(df, specs, lag_and_lead_buffer_years=False)

    # create labor choice, keep only working (2: part-time, 3: full-time)
    df = create_choice_and_employment_status(df)

    # education
    df = create_education_type(df, filter_missings=True)

    # weekly working hours. We will impute missing values later
    df.rename(columns={"pgvebzeit": "weekly_hours"}, inplace=True)
    # Also annual and monthly working hours
    df["annual_hours"] = df["weekly_hours"] * 52
    df["monthly_hours"] = df["annual_hours"] / 12

    # experience, where we use the sum of part and full time (note: unlike in
    # structural estimation, we do not round or enforce a cap on experience here)
    df = sum_experience_variables(df)

    # gross monthly wage
    df.rename(columns={"pglabgro": "monthly_wage"}, inplace=True)
    df = df[df["monthly_wage"] > 0]
    print(str(len(df)) + " observations after dropping invalid wage values.")

    # Drop unemployed(wage should be zero) and retired individuals
    df = df[df["choice"].isin([2, 3])]
    print(str(len(df)) + " observations after dropping non-working individuals.")

    # bring back indeces (pid, syear)
    df = df.reset_index()

    relevant_cols = [
        "pid",
        "syear",
        "age",
        "choice",
        "experience",
        "monthly_wage",
        "weekly_hours",
        "annual_hours",
        "monthly_hours",
        "education",
        "sex",
    ]

    # Keep relevant columns
    df = df[relevant_cols]
    print(str(len(df)) + " observations in final wage estimation dataset.")
    # save data
    df.to_csv(out_file_path)

    return df


def load_and_merge_soep_core(soep_c38_path):
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgexpft",
            "pgexppt",
            "pgstib",
            "pglabgro",
            "pgpsbil",
            "pgvebzeit",
        ],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
