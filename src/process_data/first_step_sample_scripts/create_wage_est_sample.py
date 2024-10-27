import os

import numpy as np
import pandas as pd
from process_data.aux_scripts.filter_data import filter_data
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import sum_experience_variables
from process_data.soep_vars.hours import generate_working_hours
from process_data.soep_vars.work_choices import create_choice_variable
from set_paths import create_path_dict


def create_wage_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "wage_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Load and merge data state data from SOEP core (all but wealth)
    merged_data = load_and_merge_soep_core(paths["soep_c38"])

    # filter data (age, sex, estimation period)
    merged_data = filter_data(merged_data, specs, no_women=True)

    # create labor choice, keep only working (2: part-time, 3: full-time)
    merged_data = create_choice_variable(merged_data)
    merged_data = merged_data[merged_data["choice"].isin([2, 3])]
    print(
        str(len(merged_data)) + " observations after dropping non-working individuals."
    )

    # weekly working hours
    merged_data = generate_working_hours(merged_data)

    # experience, where we use the sum of part and full time (note: unlike in
    # structural estimation, we do not round or enforce a cap on experience here)
    merged_data = sum_experience_variables(merged_data)

    # gross monthly wage
    merged_data.rename(columns={"pglabgro": "wage"}, inplace=True)
    merged_data = merged_data[merged_data["wage"] > 0]
    print(str(len(merged_data)) + " observations after dropping invalid wage values.")

    # hourly wage
    merged_data["monthly_hours"] = merged_data["working_hours"] * 52 / 12
    merged_data["hourly_wage"] = merged_data["wage"] / merged_data["monthly_hours"]

    # education
    merged_data = create_education_type(merged_data)

    # bring back indeces (pid, syear)
    merged_data = merged_data.reset_index()

    print(str(len(merged_data)) + " observations in final wage estimation dataset.")

    # save population averages (wages and hours) separately
    save_population_averages(merged_data)

    # Keep relevant columns
    merged_data = merged_data[
        [
            "pid",
            "age",
            "experience",
            "wage",
            "hourly_wage",
            "education",
            "syear",
        ]
    ]
    merged_data = merged_data.astype(
        {
            "pid": np.int32,
            "syear": np.int32,
            "age": np.int32,
            "experience": np.int32,
            "wage": np.float64,
            "hourly_wage": np.float64,
            "education": np.int32,
        }
    )

    # save data
    merged_data.to_pickle(out_file_path)

    return merged_data


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


def save_population_averages(df):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    paths_dict = create_path_dict(define_user=False)

    df["annual_wage"] = df["wage"] * 12
    pop_avg_annual_wage = df["annual_wage"].mean()
    pop_avg_hours_worked = df["working_hours"].mean() * 52
    pop_avg_hours_worked_by_edu = df.groupby("education")["working_hours"].mean() * 52
    pop_avg = pd.DataFrame({"annual_wage": [pop_avg_annual_wage]})
    for edu_level, hours in pop_avg_hours_worked_by_edu.items():
        pop_avg[f"working_hours_{edu_level}"] = hours

    pop_avg.to_csv(paths_dict["est_results"] + "population_averages.csv")
    print(
        "Population averages saved. \n Average annual wage: "
        + str(pop_avg_annual_wage)
        + "\n Average hours worked: "
        + str(pop_avg_hours_worked)
    )
    return pop_avg
