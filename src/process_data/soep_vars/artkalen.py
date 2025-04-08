import pandas as pd


def prepare_artkalen_data(path_dict):
    artkalen_data = pd.read_stata(
        f"{path_dict['soep_c38']}/artkalen.dta", convert_categoricals=False
    )
    artkalen_data = calc_begin_or_end_year_and_mont(artkalen_data, "begin")
    artkalen_data = calc_begin_or_end_year_and_mont(artkalen_data, "end")

    return artkalen_data

def calc_begin_or_end_year_and_mont(data, col_name):

    # extract the month and year of retirement
    data[f"{col_name}_month"] = (data[f"{col_name}"] - 1) % 12
    data[f"{col_name}_year"] = (data[f"{col_name}"] - 1) // 12 + 1983
    data[f"float_{col_name}"] = (
        data[f"{col_name}_year"] + data[f"{col_name}_month"] / 12
    )
    return data

