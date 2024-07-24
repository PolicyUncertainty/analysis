import os

import numpy as np
import pandas as pd
from process_data.create_structural_est_sample import create_choice_variable
from process_data.create_structural_est_sample import create_education_type
from process_data.create_structural_est_sample import filter_data
from process_data.create_structural_est_sample import load_and_merge_soep_core


def create_wage_est_sample(paths, load_data=False, options=None):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "wage_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Set file paths
    soep_c38 = paths["soep_c38"]

    # Export parameters from options
    start_year = options["start_year"]
    end_year = options["end_year"]
    start_age = options["start_age"]

    # Load and merge data state data from SOEP core (all but wealth)
    merged_data = load_and_merge_soep_core(soep_c38)

    # filter data (age, sex, estimation period)
    merged_data = filter_data(merged_data, start_year, end_year, start_age, no_women=False)

    # create labor choice, keep only working
    merged_data = create_choice_variable(
        merged_data,
    )
    merged_data = merged_data[merged_data["choice"] == 1]
    print(
        str(len(merged_data)) + " observations after dropping non-working individuals."
    )

    # experience (note: unlike in structural estimation, we do not round or enforce a cap on experience here)
    merged_data["experience"] = merged_data["pgexpft"]
    merged_data = merged_data[merged_data["experience"].notna()]
    merged_data = merged_data[merged_data["experience"] >= 0]
    print(
        str(len(merged_data))
        + " observations after dropping invalid experience values."
    )

    # gross monthly wage
    merged_data.rename(columns={"pglabgro": "wage"}, inplace=True)
    merged_data = merged_data[merged_data["wage"].notna()]
    merged_data = merged_data[merged_data["wage"] > 0]
    print(str(len(merged_data)) + " observations after dropping invalid wage values.")

    # education
    merged_data = create_education_type(merged_data)

    # bring back indeces (pid, syear)
    merged_data = merged_data.reset_index()

    # Keep relevant columns
    merged_data = merged_data[
        [
            "pid",
            "age",
            "experience",
            "wage",
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
            "education": np.int32,
        }
    )

    print(str(len(merged_data)) + " observations in final wage estimation dataset.")

    # save data
    merged_data.to_pickle(out_file_path)

    return merged_data
