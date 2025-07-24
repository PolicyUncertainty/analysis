import os

import pandas as pd

from process_data.aux_and_plots.filter_data import recode_sex
from process_data.aux_and_plots.plot_credited_periods_vs_exp import (
    plot_credited_periods_vs_exp,
)
from process_data.soep_rv_vars.credited_periods import create_credited_periods
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import create_experience_and_working_years
from process_data.soep_vars.partner_code import create_haspartner


def create_credited_periods_est_sample(paths, load_data=False, plot_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "credited_periods_est_sample.pkl"

    if load_data:
        df = pd.read_pickle(out_file_path)
        return df

    df = load_and_merge_soep_core(
        soep_c38_path=paths["soep_c38"], soep_rv_path=paths["soep_rv"]
    )
    df = recode_sex(df)
    df = create_education_type(df)
    df = create_experience_and_working_years(df)
    df = create_haspartner(df)
    df = create_credited_periods(df)
    df = df.sort_values(by=["rv_id", "syear"])
    df = df.drop_duplicates(subset="rv_id", keep="last")
    df = df.dropna(subset=["education"])
    print(f"Created credited periods variable with {df.shape[0]} observations.")

    if plot_data:
        plot_credited_periods_vs_exp(df)

    type_dict = {
        "sex": "int8",
        "education": "int8",
        "experience": "float",
        "has_partner": "int8",
        "credited_periods": "float",
        "pgexpft": "float",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)
    df.to_pickle(out_file_path)

    return df


def load_and_merge_soep_core(soep_c38_path, soep_rv_path):
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "hid",
            "pgexpft",
            "pgexppt",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )

    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
        convert_categoricals=False,
    )

    pequiv_data = pd.read_stata(
        f"{soep_c38_path}/pequiv.dta",
        columns=[
            "pid",
            "syear",
            "d11101",
            "ismp1",
            "iciv1",
            "iwar1",
            "iagr1",
            "iguv1",
            "iaus1",
            "ilib1",
        ],
        convert_categoricals=False,
    ).rename(columns={"d11101": "age"})

    soep_rv_data = pd.read_stata(
        f"{soep_rv_path}/rtbn/SOEP_RV_RTBN_2020.dta",
        columns=[
            "rv_id",
            "VSMO",  # "Versicherungspflichtige Monate", total months of insurance
            "AZ",  # "Anrechnungszeiten", certain credited periods (unemployment, pregnancy, etc.)
            "AUAZ",  # Anrechnungszeiten because of sickness
            "EZ",  # "Ersatzzeiten", replacement periods (military service, etc.)
            "KIMOBO",  #
            "RTAT",  # "Rentenart", type of pension (1: disability, 2: old age)
            "LEAT",  # "Leistungsart", type of pension (1: does not apply, 2: disability, 3-7: old age special cases, 8: old age)
        ],
        convert_categoricals=False,
    )
    soep_rv_vskt_data = pd.read_stata(
        f"{soep_rv_path}/vskt/SUF.SOEP-RV.VSKT.2020.fix.1-0.dta",
        columns=[
            "rv_id",
            "KBZ_TAGE",  # "Kinderberücksichtigungszeiten", child raising periods
        ],
        convert_categoricals=False,
    )

    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data = pd.merge(merged_data, soep_rv_data, on=["rv_id"], how="inner")
    merged_data = pd.merge(merged_data, soep_rv_vskt_data, on=["rv_id"], how="inner")
    merged_data = merged_data.dropna()
    n_obs = merged_data.shape[0]
    print(f"Loaded and merged SOEP core and SOEP RV data with {n_obs} observations.")
    return merged_data
