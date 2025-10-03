import numpy as np
import pandas as pd


def prepare_artkalen_data(path_dict, relevant_pids, start_year, end_year):
    artkalen_data = pd.read_stata(
        f"{path_dict['soep_c38']}/artkalen.dta", convert_categoricals=False
    )
    # pbiospe_data = pd.read_stata(
    #     f"{path_dict['soep_c38']}/pbiospe.dta", convert_categoricals=False
    # )
    # past_2000 = pbiospe_data[pbiospe_data["beginy"] > 2000]
    # ret_pids = past_2000[past_2000["spelltyp"] == 8]["pid"].unique()
    # # get list of artkalen pids retired after 2000
    # art_past_2000 = artkalen_data[artkalen_data["begin_year"] > 2000]
    # art_pids = art_past_2000[art_past_2000["spelltyp"] == 6]["pid"].unique()
    # # check if lists are the same
    # np.equal(art_pids, ret_pids)
    # # check if all ret_pids are in art_pids
    # ret_pids[~np.isin(ret_pids, art_pids)]

    artkalen_data = calc_begin_or_end_year_and_mont(artkalen_data, "begin")
    artkalen_data = calc_begin_or_end_year_and_mont(artkalen_data, "end")

    # # Span dataframe with index of relevant pids and syears
    # span_df_index = pd.MultiIndex.from_product(
    #     [relevant_pids, range(start_year, end_year + 1)],
    #     names=["pid", "syear"],
    # )
    # span_df = pd.DataFrame(
    #     index=span_df_index,
    # )
    # artkalen_data = artkalen_data[artkalen_data["pid"].isin(relevant_pids)]
    #
    # for spelltype in range(1, 16):
    #     spells = artkalen_data[artkalen_data["spelltyp"] == spelltype]
    #
    #     span_df[f"float_begin_{spelltype}"] = np.nan
    #     span_df[f"float_end_{spelltype}"] = np.nan
    #     for year in range(start_year, end_year + 1):
    #         cov_spells = (spells["begin_year"] <= year) & (spells["end_year"] >= year)
    #         year_spells = spells[cov_spells]
    #
    #         try:
    #             pids = year_spells["pid"].tolist()
    #             span_df.loc[(pids, year), f"float_begin_{spelltype}"] = year_spells["float_begin"].values
    #             span_df.loc[(pids, year), f"float_end_{spelltype}"] = year_spells["float_end"].values
    #         except:
    #             breakpoint()
    return artkalen_data


def calc_begin_or_end_year_and_mont(data, col_name):

    # extract the month and year of retirement
    data[f"{col_name}_month"] = (data[f"{col_name}"] - 1) % 12
    data[f"{col_name}_year"] = (data[f"{col_name}"] - 1) // 12 + 1983
    data[f"float_{col_name}"] = (
        data[f"{col_name}_year"] + data[f"{col_name}_month"] / 12
    )
    return data
