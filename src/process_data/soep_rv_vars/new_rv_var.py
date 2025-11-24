import numpy as np
import pandas as pd

# pd.options.display.max_columns = 50
# pd.options.display.max_rows = 50


data_path = "D:\gastwissenschaftler\gastw_8\Workshop_Wartezeiten"

berichtsjahr = 2022


def load_var_df_and_create_status_6(data_path, berichtsjahr):
    """
    Loads the soep_rv datasets (fix and variable) and returns the variable part with 3 new columns: KBZ for Kinder-
    berücksichtigungszeiten, KEZ for Kindererziehungszeiten, and STATUS_6 which is one of the two (see below for
    the decision rule).
    """

    # load soep rv data
    df_fix = pd.read_stata(
        data_path + "\SUF.SOEP-RV.VSKT.2022.fix.1-0.dta", convert_categoricals=False
    )
    df_var = pd.read_stata(
        data_path + "\SUF.SOEP-RV.VSKT.2022.var.1-0.dta", convert_categoricals=False
    )

    # Filter by entries with children
    df_kids = df_fix[df_fix["GBKIM1"] > 0].copy()

    # create two df's to track KBZ (Kinderberücksichtigungszeiten) and KEZ (Kindererziehungszeiten):

    # Determine start of df's = earliest birth year across all children
    start_year = df_kids["GBKIJ1"].min()
    start_month = 1

    # Determine end year of df's
    end_year = berichtsjahr
    end_month = 12

    index = pd.MultiIndex.from_product(
        [
            df_kids["rv_id"].unique(),
            np.arange(start_year, end_year + 1),
            np.arange(start_month, end_month + 1),
        ],
        names=["rv_id", "JAHR", "MONAT"],
    )

    # case "KBZ"
    df_kid_kbz = pd.Series(index=index, data=0, name="KBZ").reset_index()
    df_kid_kbz["monat_ad"] = (
        df_kid_kbz["JAHR"] * 12 + df_kid_kbz["MONAT"]
    )  # nr of months from (0000,01) to (JAHR,MONAT)

    # case "KEZ"
    df_kid_kez = pd.Series(index=index, data=0, name="KEZ").reset_index()
    df_kid_kez["monat_ad"] = (
        df_kid_kez["JAHR"] * 12 + df_kid_kez["MONAT"]
    )  # nr of months from (0000,01) to (JAHR,MONAT)

    # set max nr of children that count
    max_n_kids = 10

    # for each child count nr of months from jan 0000 to birthdate
    cols_to_add = ["rv_id"]
    for id_kid in range(1, max_n_kids + 1):
        column = f"monat_ad_kind{id_kid}"
        df_kids[column] = (
            df_kids[f"GBKIJ{id_kid}"].astype(float) * 12 + df_kids[f"GBKIM{id_kid}"]
        )
        cols_to_add += [f"GBKIJ{id_kid}", column]

    df_kid_kbz = pd.merge(df_kid_kbz, df_kids[cols_to_add], on="rv_id", how="left")
    df_kid_kez = pd.merge(df_kid_kez, df_kids[cols_to_add], on="rv_id", how="left")

    # for each child check if (JAHR,MONAT) is valid for KBZ or KEZ
    for id_kid in range(1, max_n_kids + 1):
        # KBZ (120 months, count starts over with every new child)
        mask_1 = (
            (df_kid_kbz["monat_ad"] - df_kid_kbz[f"monat_ad_kind{id_kid}"]) >= 0
        ) & ((df_kid_kbz["monat_ad"] - df_kid_kbz[f"monat_ad_kind{id_kid}"]) < 120)
        df_kid_kbz.loc[mask_1, "KBZ"] = 1

        # KEZ (different from KBZ, here 30/36 months per child, without cancellations)
        df_kid_kez["overlap_count"] = 0
        mask2 = (
            (df_kid_kbz["monat_ad"] - df_kid_kbz[f"monat_ad_kind{id_kid}"]) >= 0
        ) & (
            (
                (df_kid_kez[f"GBKIJ{id_kid}"] < 1992)
                & ((df_kid_kez["monat_ad"] - df_kid_kez[f"monat_ad_kind{id_kid}"]) < 30)
            )
            | (
                (df_kid_kez[f"GBKIJ{id_kid}"] >= 1992)
                & ((df_kid_kez["monat_ad"] - df_kid_kez[f"monat_ad_kind{id_kid}"]) < 36)
            )
        )
        df_kid_kez.loc[mask2, "KEZ"] += 1

        count_kez_overlaps = df_kid_kez[df_kid_kez["KEZ"] > 1].groupby("rv_id").size()
        count_kez_overlaps = count_kez_overlaps.reindex(
            df_kid_kez["rv_id"].unique(), fill_value=0
        )
        df_kid_kez["overlap_count"] = df_kid_kez["rv_id"].map(count_kez_overlaps)
        df_kid_kez["KEZ"] = df_kid_kez["KEZ"].replace(2, 1)
        mask2_adjusted = (
            (df_kid_kbz["monat_ad"] - df_kid_kbz[f"monat_ad_kind{id_kid}"]) >= 0
        ) & (
            (
                (df_kid_kez[f"GBKIJ{id_kid}"] < 1992)
                & (
                    (df_kid_kez["monat_ad"] - df_kid_kez[f"monat_ad_kind{id_kid}"])
                    < 30 + df_kid_kez["overlap_count"]
                )
            )
            | (
                (df_kid_kez[f"GBKIJ{id_kid}"] >= 1992)
                & (
                    (df_kid_kez["monat_ad"] - df_kid_kez[f"monat_ad_kind{id_kid}"])
                    < 36 + df_kid_kez["overlap_count"]
                )
            )
        )
        df_kid_kez.loc[mask2_adjusted, "KEZ"] = 1

    # merge into df_var
    merge_vars = ["rv_id", "JAHR", "MONAT"]
    df_kid_combo = pd.merge(df_kid_kez, df_kid_kbz, on=merge_vars, how="outer")
    df_var = pd.merge(
        df_var, df_kid_combo[merge_vars + ["KBZ", "KEZ"]], on=merge_vars, how="left"
    )

    # add STATUS_6 column to df_var:

    cond1 = (df_var["KBZ"] == 1) & (df_var["KEZ"] == 0)
    cond2 = (df_var["KBZ"] == 0) & (df_var["KEZ"] == 1)
    cond3 = (df_var["KBZ"] == 1) & (df_var["KEZ"] == 1)
    status_6 = ["KBZ", "KEZ", "KEZ"]
    df_var["STATUS_6"] = np.select([cond1, cond2, cond3], status_6, default=np.nan)
    df_var["STATUS_6"] = df_var["STATUS_6"].replace("nan", "")

    # add variable for begin of pension (in months - needed later)

    df_fix["RB_months"] = df_fix["ZTPTRTBEJ"] * 12 + df_fix["ZTPTRTBEM"]
    df_var = df_var.merge(df_fix[["rv_id", "RB_months"]], on="rv_id", how="left")

    return df_var


def add_valid_alg_flag(df):
    """
    :param df: variable part of soep_rv dataset, must have column RB_months.

    Checks whether an event "ALG" happened less than 2 years prior to pension. The result is saved in a new column
    "alg1_valide" which is True only if ALG appears as Zustand and (JAHR,MONAT) is more than 24 months before pension.
    """

    # check if (JAHR,MONAT) is in a 2 year window prior to pension begin
    check_2_year_window = (df["JAHR"].astype(float) * 12 + df["MONAT"]).between(
        df["RB_months"] - 24, df["RB_months"]
    )

    # mark rows with ALG
    alg1_receiver = (df["STATUS_2"] == "ALG") | (df["STATUS_3"] == "ALG")

    # create new column which checks both criteria
    df["alg1_valide"] = alg1_receiver & ~check_2_year_window

    return df


def add_valid_fwb_flag(df):
    """
    :param df: variable part of soep_rv dataset, must have STATUS_6 and alg1_valide column.

    The function checks if a persons months with Pflichtbeiträge sum up to 18 years. Only then, "FWB" events are valid
    (except if they appear simultaneously with ALG in a 2 year window before pension start - see next function).
    The result is saved in a new column "FWB_valide".
    """

    pflichtbeitraege_checklist = [
        "ALG",
        "WSB",
        "OSB",
        "ATZ OSB",
        "DDR OSB",
        "DDR OKN",
        "OKN",
        "WSS",
        "ATZ WSB",
        "OSS",
        "WKN",
        "ATZ OKN",
        "BRF",
        "PMU",
        "ARM",
        "AUF",
        "PFL",
        "FRG BRF",
        "NJB",
        "USV",
        "FRG BSH",
        "VRS",
        "FRG ARM",
        "BMP",
        "KEZ",
    ]

    # check Pflichtbeiträge for each relevant status
    status_1_check = df["STATUS_1"].isin(pflichtbeitraege_checklist)
    status_2_check = df["STATUS_2"].isin(pflichtbeitraege_checklist)
    status_3_check = df["STATUS_3"].isin(pflichtbeitraege_checklist)
    status_5_check = ~df["STATUS_5_EGPT"].isna()
    status_6_check = df["STATUS_6"].isin(pflichtbeitraege_checklist)

    # mark relevant rows, check if for each person sum is large enough
    rv_id_grouped_df = df.groupby("rv_id")
    df["pflichtbeitrag"] = (
        status_1_check
        | status_2_check
        | status_3_check
        | status_5_check
        | status_6_check
    )
    pflichtbeitraege_pro_person = rv_id_grouped_df["pflichtbeitrag"].sum()
    pflichtbeitraege_check = pflichtbeitraege_pro_person >= 216

    # add flag to df
    df["FWB_valide"] = df["rv_id"].map(pflichtbeitraege_check).fillna(False)

    return df


def count_valid_months(df):
    """
    :param df: variable part of soep_rv dataset, must have STATUS_6, alg1_valide, FWB_valide columns.

    Calculates Wartezeit in years. The result is saved in a new column "WARTEZEIT".
    """
    status_checklist = [
        "WSB",
        "OSB",
        "ATZ OSB",
        "DDR OSB",
        "DDR OKN",
        "OKN",
        "WSS",
        "ATZ WSB",
        "OSS",
        "WKN",
        "ATZ OKN",
        "BRF",
        "PMU",
        "ARM",
        "AUF",
        "PFL",
        "FRG BRF",
        "NJB",
        "USV",
        "FRG BSH",
        "VRS",
        "FRG ARM",
        "BMP",
        "KEZ",
        "KBZ",
    ]

    # check if row contains a relevant status (status 4 is treated separately below)
    status_1_check = df["STATUS_1"].isin(status_checklist)
    status_2_check = df["STATUS_2"].isin(status_checklist)
    status_3_check = df["STATUS_3"].isin(status_checklist)
    status_5_check = ~df["STATUS_5_TAGE"].isna()
    status_6_check = df["STATUS_6"].isin(status_checklist)

    # Freiwillige Beiträge FWB (special case: not counted if simultaneously with ALG 2 years prior to pension)
    alg1_receiver = (df["STATUS_2"] == "ALG") | (df["STATUS_3"] == "ALG")
    fwb_month = (df["STATUS_2"] == "FWB") | (df["STATUS_3"] == "FWB")
    fwb_check = (
        fwb_month
        & df["FWB_valide"]
        & ((alg1_receiver & df["alg1_valide"]) | ~alg1_receiver)
    )

    df["wartezeit_valide"] = (
        status_1_check
        | status_2_check
        | status_3_check
        | status_5_check
        | status_6_check
        | df["alg1_valide"]
        | fwb_check
    )

    # status 4 months are scaled by .33
    status_4_check = (~df["wartezeit_valide"]) & (df["STATUS_4_TAGE"] > 0)
    status_4_count = np.ceil(df[status_4_check].groupby("rv_id")["MONAT"].count() / 3)
    # add rv_ids that got dropped by status_4_check
    status_4_count = pd.Series(index=df["rv_id"].unique(), data=status_4_count).fillna(
        0
    )

    # add column for Wartezeit (in years) to df
    years_per_person = (
        df.groupby("rv_id")["wartezeit_valide"].sum() + status_4_count
    ) / 12
    df["WARTEZEIT"] = df["rv_id"].map(years_per_person).fillna(np.nan)

    return df


def clean_up_df(df):
    """
    :param df: final processed dataframe.

    Removes superfluous columns.
    """
    extra_columns = [
        "KBZ",
        "KEZ",
        "alg1_valide",
        "pflichtbeitrag",
        "FWB_valide",
        "wartezeit_valide",
        "RB_months",
    ]
    df = df.drop(columns=extra_columns)
    return df


df_final = load_var_df_and_create_status_6(
    data_path=data_path, berichtsjahr=berichtsjahr
)
df_final = add_valid_alg_flag(df_final)
df_final = add_valid_fwb_flag(df_final)
df_final = count_valid_months(df_final)

# optional: clean up columns
# df_final = clean_up_df(df_final)


# save full df
df_final.to_stata(data_path + "/Full_WZ_python.dta")

# save only Wartezeiten
df_final.groupby("rv_id")[["WARTEZEIT"]].mean().to_stata(
    data_path + "/Wartezeiten_python.dta"
)
