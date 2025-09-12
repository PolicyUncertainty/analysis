import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data.auxiliary.filter_data import (
    drop_missings,
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
)
from process_data.auxiliary.lagged_and_lead_vars import (
    span_dataframe,
)
from process_data.soep_vars.age import calc_age_at_interview
from process_data.soep_vars.choice_from_spell import (
    create_choice_variable_from_artkalen,
)
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import create_experience_and_working_years
from process_data.soep_vars.health import correct_health_state, create_health_var
from process_data.soep_vars.job_hire_and_fire import (
    determine_observed_job_offers,
    generate_job_separation_var,
)
from process_data.soep_vars.partner_code import (
    correct_partner_state,
    create_partner_state,
)
from process_data.soep_vars.wealth.flow_savings import create_flow_savings
from process_data.soep_vars.wealth.linear_interpolation import (
    add_wealth_interpolate_and_deflate,
)
from process_data.structural_sample_scripts.classify_reitrees import (
    add_very_long_insured_classification,
)
from process_data.structural_sample_scripts.disability_pension_health import (
    modify_health_for_disability_pension,
)
from process_data.structural_sample_scripts.informed_state import create_informed_state
from process_data.structural_sample_scripts.model_restrictions import (
    enforce_model_choice_restriction,
)
from process_data.structural_sample_scripts.policy_state import (
    create_policy_state,
    create_SRA_by_gebjahr,
)

CORE_TYPE_DICT = {
    "period": "int8",
    "choice": "int8",
    "lagged_choice": "int8",
    "informed": "int8",
    "policy_state": "int8",
    "policy_state_value": "float64",
    "lagged_health": "int8",
    "partner_state": "int8",
    "job_offer": "int8",
    "experience": "int8",
    "wealth": "float64",
    "education": "int8",
    "sex": "int8",
    "health": "int8",
}


def create_structural_est_sample(
    paths,
    specs,
    load_data=False,
    use_processed_pl=False,
    load_artkalen_choice=False,
    load_wealth=False,
):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    if load_data:
        df = pd.read_csv(paths["struct_est_sample"])
        return df

    # Load and merge data state data from SOEP core (all but wealth)
    df = load_and_merge_soep_core(path_dict=paths, use_processed_pl=use_processed_pl)

    # Create the cohort specific SRA and its enumerated policy state
    df["SRA"] = create_SRA_by_gebjahr(df["gebjahr"])

    # First start with partner state, as these could be also out of age range.
    df = create_partner_state(df, filter_missing=False)

    # Filter data to estimation years. Leave additional years for lagging and leading
    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    # Create type variables. They should not be missing anyway
    df = recode_sex(df)
    df = create_education_type(df, filter_missings=False)

    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)

    df = calc_age_at_interview(df)

    # Filter ages below
    df = filter_below_age(df, specs["start_age"] - 1)

    # Having a spanned dataframe we can correct the partner state
    # (missing partner observation in a single year).
    df = correct_partner_state(df)

    # We create the health variable and correct it
    df = create_health_var(df, filter_missings=False)
    df = correct_health_state(df)

    df = create_choice_variable_from_artkalen(
        path_dict=paths, specs=specs, df=df, load_artkalen_choice=load_artkalen_choice
    )

    # Create informed state
    df = create_informed_state(df)

    # Generare job separation variable
    df = generate_job_separation_var(df)

    # Now use this information to determine job offer state
    df["job_sep_this_year"] = df.groupby(["pid"])["job_sep"].shift(-1)
    was_fired_last_period = (df["job_sep"] == 1) | (df["job_sep_this_year"] == 1)
    df = determine_observed_job_offers(
        df, was_fired_last_period=was_fired_last_period, working_choices=[2, 3]
    )

    # We are done with lagging and leading and drop the buffer years
    df = filter_years(df, specs["start_year"], specs["end_year"])

    # Add wealth and flow savings
    df = add_wealth_interpolate_and_deflate(
        df,
        paths,
        specs,
        load_wealth=load_wealth,
        use_processed_pl=use_processed_pl,
        filter_missings=True,
    )

    df = create_flow_savings(df, specs)

    # create experience and working years
    df = create_experience_and_working_years(df.copy(), filter_missings=True)

    # Now we can also kick out the buffer age for lagging
    df = filter_below_age(df, specs["start_age"])

    # We also delete now the observations with invalid data, which we left before to have a continuous panel
    df = drop_missings(
        df=df,
        vars_to_check=[
            "health",
            "choice",
            "lagged_choice",
            "lagged_health",
            "education",
            "age",
            "wealth",
        ],
    )

    # Correct policy state
    df = create_policy_state(df, specs)
    # Now all age manipulations are done, we can also create the period
    df["period"] = df["age"] - specs["start_age"]

    df = filter_above_age(df, specs["end_age"])

    # enforce choice restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # # Modify health state for incorporation of disability pension
    df["surveyed_health"] = df["health"].copy()
    df = modify_health_for_disability_pension(df, specs)

    # Add very long insured classification
    df = add_very_long_insured_classification(
        df=df,
        path_dict=paths,
        specs=specs,
    )

    # Drop civil servants and self-employed
    df = df[~(df["self_employed"] | df["civil_servant"])]
    print(f"Dropping civil servants and self-employed, {len(df)} observations left.")

    # Rename to monthly wage
    df.rename(
        columns={
            "pglabgro": "monthly_wage",
            "pglabgro_p": "monthly_wage_partner",
            "hlc0005_v2": "hh_net_income",
            "i11102": "last_year_hh_net_income",
            "igrv1": "last_year_pension",
        },
        inplace=True,
    )

    type_dict_add = {
        "monthly_wage": "float64",
        "monthly_wage_partner": "float64",
        "hh_net_income": "float64",
        "last_year_hh_net_income": "float64",
        "savings_dec": "float64",
        "working_years": "float64",
        "children": "float64",
        # "surveyed_health": "int8",
        "last_year_pension": "float64",
    }

    df["hh_net_income"] /= specs["wealth_unit"]

    # Drop observations if any of core variables are nan
    # We also delete now the observations with invalid data, which we left before to have a continuous panel
    df = drop_missings(
        df=df,
        vars_to_check=list(CORE_TYPE_DICT.keys()),
    )

    all_type_dict = {
        **CORE_TYPE_DICT,
        **type_dict_add,
    }
    df = df[list(all_type_dict.keys())]
    df = df.astype(all_type_dict)

    print_data_description(df, detailed=True)

    # Anonymize and save data
    df["year"] = df.index.get_level_values("syear").values
    df.reset_index(drop=True, inplace=True)
    df.to_csv(paths["struct_est_sample"])

    # median wealth by age
    # df["median_wealth"] = df.groupby("age")["wealth"].transform("median")

    return df


def load_and_merge_soep_core(path_dict, use_processed_pl):

    soep_c38_path = path_dict["soep_c38"]
    # Start with ppathl. Everyone is in there even if not individually surveyed and just member
    # of surveyed household
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "parid", "rv_id", "gebjahr", "gebmonat"],
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
            "pgexpft",
            "pgexppt",
            "pgstib",
            "pgpartz",
            "pglabgro",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        ppathl_data, pgen_data, on=["pid", "hid", "syear"], how="left"
    )

    pl_intermediate_file = path_dict["struct_data"] + "pl_structural.pkl"
    if use_processed_pl:
        pl_data = pd.read_pickle(pl_intermediate_file)
    else:
        # Add pl data
        pl_data_reader = pd.read_stata(
            f"{soep_c38_path}/pl.dta",
            columns=["pid", "hid", "syear", "plb0304_h", "iyear", "pmonin", "ptagin"],
            chunksize=100000,
            convert_categoricals=False,
        )
        pl_data = pd.DataFrame()
        for itm in pl_data_reader:
            pl_data = pd.concat([pl_data, itm])

        pl_data.to_pickle(pl_intermediate_file)

    merged_data = pd.merge(merged_data, pl_data, on=["pid", "hid", "syear"], how="left")

    # get household level data
    hl_data = pd.read_stata(
        f"{soep_c38_path}/hl.dta",
        columns=["hid", "syear", "hlc0005_v2"],
        convert_categoricals=False,
    )
    hl_data["hlc0005_v2"] *= 12

    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        # d11101: age of individual
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=[
            "pid",
            "syear",
            "d11107",
            "d11101",
            "m11126",
            "m11124",
            "igrv1",
            "i11102",
        ],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def print_data_description(df, detailed=False):
    n_retirees = df.groupby("choice").size().loc[0]
    n_unemployed = df.groupby("choice").size().loc[1]
    n_part_time = df.groupby("choice").size().loc[2]
    n_full_time = df.groupby("choice").size().loc[3]
    n_fresh_retirees = (
        df.groupby(["choice", "lagged_choice"]).size().get((0, 1), 0)
        + df.groupby(["choice", "lagged_choice"]).size().get((0, 2), 0)
        + df.groupby(["choice", "lagged_choice"]).size().get((0, 3), 0)
    )
    print(str(len(df)) + " left in final estimation sample.")
    print("---------------------------")
    print(
        "Breakdown by choice:\n" + str(n_retirees) + " retirees [0] \n"
        "--"
        + str(n_fresh_retirees)
        + " thereof fresh retirees [0, lagged =!= 0] \n"
        + str(n_unemployed)
        + " unemployed [1] \n"
        + str(n_part_time)
        + " part-time [2] \n"
        + str(n_full_time)
        + " full time [3]."
    )
    print("---------------------------")
    if detailed:
        print("Detailed description of choice and state variables:")
        choice_vars = ["choice", "lagged_choice"]
        state_vars = [
            "period",
            "education",
            "sex",
            "job_offer",
            "partner_state",
            "health",
            "policy_state",
            "wealth",
        ]
        for var in choice_vars + state_vars:
            print(f"\nDescription of {var}:")
            print(df[var].describe())
            # value counts for non-float variables
            if var in choice_vars + state_vars and not pd.api.types.is_float_dtype(
                df[var]
            ):
                print(f"Value counts of {var}:")
                print(df[var].value_counts().sort_index())
    print("---------------------------")
