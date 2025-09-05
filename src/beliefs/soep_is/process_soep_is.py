import numpy as np
import pandas as pd

from process_data.auxiliary.filter_data import recode_sex
from process_data.soep_vars.age import calc_age_at_interview
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import create_health_var
from process_data.structural_sample_scripts.policy_state import (
    create_SRA_by_gebjahr,
)


def load_and_filter_soep_is(paths):
    "Load SOEP-IS data, keep only pension survey questions."
    # paths
    soep_is_path = paths["soep_is"] + "inno.dta"
    # relevant columns (ids and pension survey questions, i.e. i107_1 to i107_15)
    id_columns = ["pid", "syear", "hid", "cid", "intid", "im_107"]
    pension_survey_columns = [f"i107_{i}" for i in range(1, 16)]
    # load and filter data.
    df = pd.read_stata(soep_is_path, convert_categoricals=False)[
        id_columns + pension_survey_columns
    ].astype(float)
    df = rename_survey_columns(df)
    df = filter_df(df)
    print(f"{len(df)} observations in SOEP-IS pension beliefs survey.")
    return df


def add_covariates(df, paths, specs, filter_missings=False):
    """Add sex, age, education, health, and fweight variables to the dataframe. Df must have pid and syear columns.

    raw data to be added
      - sex: sex (ppath)
      - age:
         - soep-core: gebjahr (ppath), gebmonat (ppath), syear (already in df), pmonin (pl), ptagin (pl)
         - soep is: gebjahr (ppath), gebmonat (ppath), syear (already in df), imonth (pl), iday (pl)
      - education:
         - soep-core: pgpsbil (pgen)
         - soep is: pgsbil (pgen)
      - health:
         - soep-core: m11126 (pequiv), m11124 (pequiv)
         - soep-is: ple0008 (pl), ple0040 (pl), NOTE: different coding!]
      - fweight:
        - soep is: phrf (ppathl)
    """

    # load data
    soep_path = paths["soep_is"]
    ppath = pd.read_stata(
        f"{soep_path}/ppath.dta",
        convert_categoricals=False,
        columns=["pid", "sex", "gebjahr", "gebmonat"],
    )
    ppathl = pd.read_stata(
        f"{soep_path}/ppathl.dta",
        convert_categoricals=False,
        columns=["pid", "syear", "phrf"],
    )
    pl = pd.read_stata(
        f"{soep_path}/pl.dta",
        convert_categoricals=False,
        columns=["pid", "syear", "imonth", "iday", "ple0008", "ple0040"],
    )
    pgen = pd.read_stata(
        f"{soep_path}/pgen.dta",
        convert_categoricals=False,
        columns=["pid", "syear", "pgsbil"],
    )
    # merge data
    df = pd.merge(df, ppath, how="left", on=["pid"])
    df = pd.merge(df, ppathl, how="left", on=["pid", "syear"])
    df = pd.merge(df, pl, how="left", on=["pid", "syear"])
    df = pd.merge(df, pgen, how="left", on=["pid", "syear"])
    # modify variables
    df.set_index(["pid", "syear"], inplace=True)
    df = rename_and_reformat_is_covariates(df)
    df = recode_sex(df)
    df = calc_age_at_interview(df, drop_missing_month=filter_missings)
    df = create_education_type(df, filter_missings=filter_missings)
    df = create_health_var(df, filter_missings=filter_missings)
    df = classify_informed(df, specs)
    df["current_SRA"] = create_SRA_by_gebjahr(df["gebjahr"])
    # cleanup
    raw_columns = ["pmonin", "ptagin", "pgpsbil", "m11126", "m11124"]
    df = df.drop(columns=raw_columns)
    return df


def rename_survey_columns(df):
    """Rename the columns of the pension belief survey."""
    rename_dict = {
        "i107_1": "exp_stop_work",
        "i107_2": "exp_stop_work_rob_plus1",
        "i107_3": "exp_stop_work_rob_minus1",
        "i107_4": "exp_pens_uptake",
        "i107_5": "exp_pens_uptake_rob_plus1",
        "i107_6": "exp_pens_uptake_rob_minus1",
        "i107_7": "pol_unc_stat_ret_age_67",
        "i107_8": "pol_unc_stat_ret_age_68",
        "i107_9": "pol_unc_stat_ret_age_69",
        "i107_10": "belief_pens_deduct",
        "i107_11": "belief_pens_deduct_rob_times1_5",
        "i107_12": "belief_pens_deduct_rob_times0_5",
        "i107_13": "scen_age_66_stop_work",
        "i107_14": "scen_age_68_stop_work",
        "i107_15": "scen_age_69_stop_work",
    }
    df = df.rename(columns=rename_dict)
    return df


def filter_df(df):
    """Filter pension beliefs dataframe to include only relevant observations. Relevant observations are those for which pension questions (e.g. exp_stop_work) are > -2 (= does not apply). These are dropped.

    All observations with a value of -1 (= no answer) are set to NaN."""

    df = df[df["exp_stop_work"] > -2]
    df = df.replace(-1, np.nan)
    return df


def rename_and_reformat_is_covariates(df):
    """Rename and reformat the variables in the dataframe to match the SOEP-Core data or to make them easier to work with."""
    rename_dict = {
        "imonth": "pmonin",
        "iday": "ptagin",
        "ple0008": "m11126",
        "ple0040": "m11124",
        "pgsbil": "pgpsbil",
        "phrf": "fweights",
    }
    df = df.rename(columns=rename_dict)
    # recode variables
    # ple0040 (disabled) has 1:yes, 2:no, m11124 (disabled) has 0:no, 1:yes
    df["m11124"] = df["m11124"].replace({2: 0, 1: 1})
    return df


def classify_informed(df, specs):
    """Informed means ERP beliefs <= threshhold (e.g. 5)."""
    df["informed"] = df["belief_pens_deduct"] <= specs["informed_threshhold"]
    return df
