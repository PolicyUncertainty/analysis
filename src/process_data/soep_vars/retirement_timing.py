import numpy as np
import pandas as pd


def align_retirement_choice(path_dict, df):
    # Start by making retirement absorbing. Once choice is 0 all future choices are 0
    artkalen_data = prepare_artkalen_data(path_dict)
    # ret_spells_artkalen = artkalen_data[artkalen_data["spelltyp"] == 6]
    # pbio = pd.read_stata(
    #     f"{path_dict['soep_c38']}/pbiospe.dta", convert_categoricals=False
    # )
    df.reset_index(inplace=True)
    min_year = df["syear"].min()
    max_year = df["syear"].max()

    retired_pids = df[df["choice"] == 0]["pid"].unique()
    first_ret_year_pids_employment_data = (
        df[df["choice"] == 0].groupby("pid")["syear"].min()
    )
    # last_ret_spell_artkalen = ret_spells_artkalen.groupby("pid").tail(1).copy()

    no_ret_spell_count = 1
    i = 1
    # Assign 0 for all future years for those who retired
    for pid in retired_pids:
        # First make retirement absorbing
        first_ret_year_sample_data = first_ret_year_pids_employment_data[pid]
        mask = (df["pid"] == pid) & (df["syear"] >= first_ret_year_sample_data)
        df.loc[mask, "choice"] = 0

        # Select pid spells of arkalen
        pid_spells = artkalen_data[artkalen_data["pid"] == pid].sort_values(
            "float_begin"
        )
        pid_ret_spells = pid_spells[pid_spells["spelltyp"] == 6]
        if len(pid_ret_spells) == 0:
            df = df[df["pid"] != pid]
            no_ret_spell_count += 1
            continue
        pid_last_spell_ret = pid_ret_spells.iloc[-1]

        # First we look at the beginning of the sample
        if first_ret_year_sample_data == min_year:
            # When the first year of retirement corresponds to the first year of the sample,
            # you could be already retired in the beginning (retirement year) or just retired this year
            # As we are using an extra year here for leading, we do not need to take care of individuals
            # retiring in the same year. So let's look at people who are by spell data retired later.
            if pid_last_spell_ret["begin_year"] > min_year:
                first_ret_year_spell_data = pid_ret_spells["begin_year"].min()
                # If there is another retirement spell before the first year, continue
                if first_ret_year_spell_data <= min_year:  #
                    continue
                else:
                    first_ret_spell_idx = np.where(pid_spells == 6)[0][0]
                    last_spell_before_ret_idx = first_ret_spell_idx - 1
                    last_spell_before_ret = pid_spells.iloc[last_spell_before_ret_idx]
                    # If last spell before unemployment or house wife/husband
                    if last_spell_before_ret["spelltyp"] in [10, 15]:
                        year_unemployed_start = last_spell_before_ret["begin_year"]
                        if year_unemployed_start <= min_year:
                            mask = (
                                (df["pid"] == pid)
                                & (df["syear"] >= year_unemployed_start)
                                & (df["syear"] < first_ret_year_spell_data)
                            )
                            df.loc[mask, "choice"] = 1
                        else:
                            raise ValueError(
                                "Last spell before retirement is not unemployment or housewife/househusband,"
                                "or is not long enough. Haven't found a solution for this case yet."
                            )
        else:
            # Now we are in the case where the first retirement year is after the first year of the sample
            if first_ret_year_sample_data == pid_last_spell_ret["begin_year"]:
                breakpoint()

    breakpoint()


def prepare_artkalen_data(path_dict):
    artkalen_data = pd.read_stata(
        f"{path_dict['soep_c38']}/artkalen.dta", convert_categoricals=False
    )

    # extract the month and year of retirement
    artkalen_data["begin_month"] = artkalen_data["begin"] % 12
    artkalen_data["begin_year"] = artkalen_data["begin"] // 12 + 1983
    artkalen_data["float_begin"] = (
        artkalen_data["begin_year"] + artkalen_data["begin_month"] / 12
    )

    df_birth = create_float_birth_year(path_dict)
    artkalen_data = pd.merge(artkalen_data, df_birth, on="pid", how="left")
    # compute actual retirement age
    artkalen_data.loc[:, "float_begin_age"] = (
        artkalen_data["float_begin"] - artkalen_data["float_birth_year"]
    )
    return artkalen_data


def create_float_birth_year(path_dict):
    ppath_data = pd.read_stata(
        f"{path_dict['soep_c38']}/ppath.dta",
        columns=["pid", "gebjahr", "gebmonat"],
        convert_categoricals=False,
    )
    # drop when gebjahr <0 in ppath (missing birth year)
    ppath_data = ppath_data[ppath_data["gebjahr"] >= 0]
    # whenever gebmonat is <0 in ppath, we assume that the person was born in June
    ppath_data["gebmonat"] = ppath_data["gebmonat"].apply(lambda x: x if x >= 0 else 6)

    ppath_data["float_birth_year"] = ppath_data["gebjahr"] + ppath_data["gebmonat"] / 12
    return ppath_data

    # # Generate this years pension payments by assigning the pensions
    # # from next year. It is assumed that the dataframe is sorted by pid and syear.
    # # and covers all years
    # data["this_year_pensions"] = data.groupby("pid")["igrv1"].shift(-1)
    # received_pensions = data["this_year_pensions"] > 0
    # data.loc[received_pensions, "choice"] = 0
    # # merged_data.loc[rv_ret_choice == "RTB"] = 2
    #
    # # If there is a 0 inbetween two other values. Delete 0 and assign the value of the next year
    # # to the previous year.
    # lead_choice = data["choice"].shift(-1)
    # data.loc[(data["choice"] == 0) & (lead_choice != 0), "choice"] = lead_choice
