import pandas as pd


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
    return artkalen_data
