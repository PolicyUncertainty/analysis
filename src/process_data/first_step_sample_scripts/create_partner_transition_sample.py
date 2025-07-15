# %%
import os

import matplotlib.pyplot as plt
import pandas as pd

from process_data.aux_and_plots.filter_data import (
    drop_missings,
    filter_below_age,
    filter_years,
    recode_sex,
)
from process_data.aux_and_plots.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.age import calc_age_at_interview
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.partner_code import create_partner_state


# %%
def create_partner_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])

    df = create_education_type(df)

    df = recode_sex(df)

    # Filter estimation years, keep one more for leading
    df = filter_years(df, specs["start_year"], specs["end_year"] + 1)

    # In this function also merging is called
    df = span_dataframe(df, specs["start_year"], specs["end_year"] + 1)

    df = create_partner_state(df, filter_missing=False)

    df["lead_partner_state"] = df.groupby(["pid"])["partner_state"].shift(-1)

    df["age"] = df.index.get_level_values("syear") - df["gebjahr"]

    # Filter age and sex
    df = filter_below_age(df, specs["start_age"])

    core_type_dict = {
        "age": "int8",
        "sex": "int8",
        "education": "int8",
        "partner_state": "int8",
        "lead_partner_state": "int8",
        "children": "int8",
    }
    # Drop observations if any of core variables are nan
    # We also delete now the observations with invalid data, which we left before to have a continuous panel
    df = drop_missings(
        df=df,
        vars_to_check=list(core_type_dict.keys()),
    )
    df = df[list(core_type_dict.keys())].astype(core_type_dict)

    print(
        str(len(df))
        + " observations in the final partner transition sample.  \n ----------------"
    )
    df.to_pickle(out_file_path)
    return df


def load_and_merge_soep_core(soep_c38_path):

    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )

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

    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(
        ppathl_data, pgen_data, on=["pid", "hid", "syear"], how="left"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    merged_data["syear"] = merged_data["syear"].astype(int)
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
