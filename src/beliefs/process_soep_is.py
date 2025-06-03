import pandas as pd
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import create_health_var
from process_data.soep_vars.age import calc_age_at_interview
from process_data.aux_and_plots.filter_data import recode_sex


def load_and_filter_soep_is(paths):
    " Load SOEP-IS data, keep only pension survey questions."
    # TODO: when SOEP-IS 23 is out, the name of the dataset and variables have to be changed.
    # paths
    soep_is_path = paths["soep_is"] + "\dataset_main_SOEP_IS.dta"
    out_file_path = paths["intermediate_data"] + "beliefs\soep_is_pensions.pkl"
    # relevant columns
    id_columns = ["pid", "syear", "hid", "cid", "intid"]
    pension_survey_columns = ['exp_stop_work',
       'exp_stop_work_rob_plus1', 'exp_stop_work_rob_minus1',
       'exp_pens_uptake', 'exp_pens_uptake_rob_plus1',
       'exp_pens_uptake_rob_minus1', 'pol_unc_stat_ret_age_67',
       'pol_unc_stat_ret_age_68', 'pol_unc_stat_ret_age_69',
       'belief_pens_deduct', 'belief_pens_deduct_rob_times1_5',
       'belief_pens_deduct_rob_times0_5', 'scen_age_66_stop_work',
       'scen_age_68_stop_work', 'scen_age_69_stop_work']
    # load and filter data.
    # TODO: when soep_is 23 is out, we need to filter the missing obs (negative values) for the pension beliefs questions.
    df = pd.read_stata(soep_is_path)[id_columns + pension_survey_columns].astype(float)
    df = df.dropna(subset=pension_survey_columns, how='all')
    print(f"{len(df)} observations in SOEP-IS pension beliefs survey.")
    return df

def add_covariates(df, paths, filter_missings=False):
    """ Add sex, age, education, health, and fweight variables to the dataframe. Df must have pid and syear columns.

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
    print("Extracting covariates from SOEP data. This may take a while.")
    ppath = pd.read_stata(f"{soep_path}/ppath.dta", convert_categoricals=False, columns=["pid", "sex", "gebjahr", "gebmonat"])
    ppathl = pd.read_stata(f"{soep_path}/ppathl.dta", convert_categoricals=False, columns=["pid", "syear", "phrf"])
    pl = pd.read_stata(f"{soep_path}/pl.dta", convert_categoricals=False, columns=["pid", "syear", "imonth", "iday", "ple0008", "ple0040"])
    pgen = pd.read_stata(f"{soep_path}/pgen.dta", convert_categoricals=False, columns=["pid", "syear", "pgsbil"])
    # merge data
    df = pd.merge(df, ppath, how="left", on=["pid"])
    df = pd.merge(df, ppathl, how="left", on=["pid", "syear"])
    df = pd.merge(df, pl, how="left", on=["pid", "syear"])
    df = pd.merge(df, pgen, how="left", on=["pid", "syear"])
    # modify variables
    df.set_index(["pid", "syear"], inplace=True)
    df = rename_and_reformat_is_vars(df)
    df = recode_sex(df)
    df = calc_age_at_interview(df, drop_missing_month=filter_missings)
    df = create_education_type(df, filter_missings=filter_missings)
    df = create_health_var(df, filter_missings=filter_missings)
    # cleanup
    raw_columns = ["pmonin", "ptagin", "pgpsbil", "m11126", "m11124"]
    df = df.drop(columns=raw_columns)
    return df

def rename_and_reformat_is_vars(df):
    """ Rename and reformat the variables in the dataframe to match the SOEP-Core data or to make them easier to work with."""
    rename_dict = {
        "imonth": "pmonin",
        "iday": "ptagin",
        "ple0008": "m11126",
        "ple0040": "m11124",
        "pgsbil" : "pgpsbil",
        "phrf": "fweight",
    }
    df = df.rename(columns=rename_dict)
    # recode variables
    # ple0040 (disabled) has 1:yes, 2:no, m11124 (disabled) has 0:no, 1:yes
    df["m11124"] = df["m11124"].replace({2: 0, 1: 1})
    return df    