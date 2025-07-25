import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)


def plot_retiree_classification(paths):
    fresh_retirees = classify_retirees(paths)
    fresh_retirees.groupby("age")[
        [
            "ret_after_SRA",
            "ret_before_SRA_over_45_years",
            "ret_before_SRA_under_45_years",
        ]
    ].sum().plot(kind="bar", stacked=True)
    plt.xlabel("Age")
    plt.ylabel("Number of Individuals")
    plt.title("Retiree Types")


def classify_retirees(paths):
    struct_est_sample = pd.read_csv(paths["struct_est_sample"])
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]

    fresh_retirees = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ]

    ret_age_artkalen_df = align_retirement_choice(paths)
    fresh_retirees = pd.merge(
        fresh_retirees,
        ret_age_artkalen_df[
            ["pid", "float_birth_year", "actual_retirement_age"]
        ].rename(columns={"actual_retirement_age": "actual_retirement_age_artkalen"}),
        on="pid",
        how="left",
    )
    # ret_age_pl_df = get_actual_retirement_age_from_pl(paths)
    # fresh_retirees = pd.merge(
    #    fresh_retirees,
    #    ret_age_pl_df["actual_retirement_age"].rename(columns={"actual_retirement_age": "actual_retirement_age_pl"}),
    #    on="pid", how="left")

    # plot difference between actual retirement age and age
    fresh_retirees["age_diff"] = (
        fresh_retirees["actual_retirement_age_artkalen"] - fresh_retirees["age"]
    )
    fresh_retirees["age_diff"].plot(kind="hist", bins=20)

    # 1. age to be replaced with actual retirement age and 2. working years to be replaced with credited periods
    fresh_retirees.loc[:, "ret_after_SRA"] = (
        fresh_retirees["age"] >= fresh_retirees["policy_state_value"]
    ).astype(int)
    fresh_retirees.loc[:, "ret_before_SRA_over_45_years"] = (
        (fresh_retirees["age"] < fresh_retirees["policy_state_value"])
        & (fresh_retirees["working_years"] >= 45)
    ).astype(int)
    fresh_retirees.loc[:, "ret_before_SRA_under_45_years"] = (
        (fresh_retirees["age"] < fresh_retirees["policy_state_value"])
        & (fresh_retirees["working_years"] < 45)
    ).astype(int)

    return fresh_retirees


def get_actual_retirement_age_from_pl(paths):

    soep_c38_path = paths["soep_c38"]
    pl_data_reader = pd.read_stata(
        f"{soep_c38_path}/pl.dta",
        columns=["pid", "syear", "plc0235"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.DataFrame()
    for itm in pl_data_reader:
        pl_data = pd.concat([pl_data, itm])

    ppath_data = pd.read_stata(
        f"{soep_c38_path}/ppath.dta",
        columns=["pid", "gebjahr", "gebmonat"],
        convert_categoricals=False,
    )

    # for each person, get the retirement year (i.e. the row before the first occurrence of plc0235 == 12)
    pl_orig = pl_data.copy()
    pl_data = pl_orig.copy()
    result_rows = pl_data.groupby("pid").apply(get_prev_row)
    df = result_rows.dropna().reset_index(drop=True)
    df = df.apply(get_year_of_retirement, axis=1)
    # compute float actual retirement year
    df["actual_retirement_year"] = (
        df["jahr_renteneintr_pl"] + df["monat_renteneintr_pl"] / 12
    )
    # get float birth year
    # df = create_float_birth_year(df, ppath_data)
    # compute actual retirement age
    df["actual_retirement_age"] = df["actual_retirement_year"] - df["float_birth_year"]
    return df


def get_prev_row(group):
    """Find the previous row in a group where plc0235 == 12 and the years are consecutive."""
    # Ensure the group is sorted by syear
    group = group.sort_values("syear")
    # Find the index of the first occurrence of plc0235 == 12
    first_12_idx = group.index[group["plc0235"] == 12]
    # If there is no occurrence of plc0235 == 12, return None
    if first_12_idx.empty:
        return None
    # Get the position of the first occurrence of plc0235 == 12
    first_12_idx = first_12_idx[0]
    pos = group.index.get_loc(first_12_idx)
    # If the first occurrence is in the first row, return None
    if pos == 0:
        return None
    # Get the previous row
    prev_row = group.iloc[pos - 1]
    # Check if the years are consecutive
    current_year = group.loc[first_12_idx, "syear"]
    prev_year = prev_row["syear"]
    if current_year - prev_year == 1:
        return prev_row
    else:
        return None


def get_year_of_retirement(row):
    """Calculate the year and month of retirement"""
    # Case 1 plc is negative
    # We assume that the person retired in January of the current year
    if row["plc0235"] < 0:
        row["monat_renteneintr_pl"] = 1
        row["jahr_renteneintr_pl"] = row["syear"]
    # Case 2 plc is positive
    # plc corresponds to the number of months the person was retired in the previous year
    else:
        row["monat_renteneintr_pl"] = 13 - row["plc0235"]
        row["jahr_renteneintr_pl"] = row["syear"] - 1
    return row
