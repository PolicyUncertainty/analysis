import os

import numpy as np
import pandas as pd
from process_data.soep_vars import create_choice_variable
from process_data.soep_vars import create_education_type
from process_data.soep_vars import create_experience_variable_with_cap


def create_structural_est_sample(paths, load_data=False, options=None):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "structural_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Export parameters from options
    start_year = options["start_year"]
    end_year = options["end_year"]
    min_ret_age = options["min_ret_age"]
    start_age = options["start_age"]
    exp_cap = options["exp_cap"]

    # Load and merge data state data from SOEP core (all but wealth)
    merged_data = load_and_merge_soep_core(soep_c38_path=paths["soep_c38"])

    # wealth data from SOEP C38 (hwealth.dta)
    merged_data = gather_wealth_data(paths["soep_c38"], merged_data, options)
    merged_data = create_hh_cons_equivalence_data(merged_data)
    # Wealth hack. Until we have family context
    merged_data["wealth"] = merged_data["wealth"] / merged_data["cons_equiv"]
    merged_data = deflate_wealth(merged_data, paths)

    # (labor) choice
    merged_data = create_choice_variable(merged_data)

    # filter data
    merged_data = filter_data(merged_data, start_year, end_year, start_age)

    # period
    merged_data["period"] = merged_data["age"] - start_age

    # lagged choice
    merged_data = create_lagged_choice_variable(
        merged_data, start_year, end_year, start_age
    )

    # policy_state
    merged_data["policy_state"] = create_policy_state(merged_data["gebjahr"])
    (
        merged_data["policy_state_value"],
        merged_data["policy_state"],
    ) = modify_policy_state(merged_data["policy_state"], options)

    # retirement_age_id (dummy 0 for now)
    merged_data["retirement_age_id"] = 0

    # experience
    merged_data = create_experience_variable_with_cap(merged_data, exp_cap)

    # education
    merged_data = create_education_type(merged_data)

    # additional filters based on model setup
    merged_data = enforce_model_choice_restriction(
        merged_data, min_ret_age, options["max_ret_age"]
    )

    # Keep relevant columns (i.e. state variables)
    merged_data = merged_data[
        [
            "choice",
            "period",
            "lagged_choice",
            "policy_state",
            "policy_state_value",
            "retirement_age_id",
            "experience",
            "wealth",
            "education",
        ]
    ]
    merged_data = merged_data.astype(
        {
            "choice": "int8",
            "lagged_choice": "int8",
            "policy_state": "int8",
            "retirement_age_id": "int8",
            "experience": "int8",
            "wealth": "float32",
            "period": "int8",
            "education": "int8",
        }
    )

    print(
        str(len(merged_data)) + " observations in final structural estimation dataset."
    )

    # Anonymize and save data
    merged_data.reset_index(drop=True, inplace=True)
    merged_data.to_pickle(out_file_path)

    # save data
    merged_data.to_pickle(out_file_path)

    return merged_data


def load_and_merge_soep_core(soep_c38_path):
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
    pathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr", "rv_id"],
        convert_categoricals=False,
    )
    hl_data = pd.read_stata(
        f"{soep_c38_path}/hl.dta",
        columns=["hid", "syear", "hlc0043"],
        convert_categoricals=False,
    )

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data, hl_data
    print(str(len(merged_data)) + " observations in SOEP C38 core.")

    # Calculate age and filter out missing rv_id values for people older than minimum retirement age

    # merged_data = merged_data[
    #     (merged_data["rv_id"] >= 0)
    #     | ((merged_data["rv_id"] < 0) & (merged_data["age"] < min_ret_age))
    # ]
    # merged_data["rv_id"].replace(-2, pd.NA, inplace=True)
    #
    # print(
    #     str(len(merged_data))
    #     + " left after dropping over "
    #     + str(min_ret_age)
    #     + " year olds w/o SOEP-RV ID."
    # )
    #
    # # Load SOEP RV VSKT data
    # rv_data = pd.read_stata(
    #     f"{soep_rv}/vskt/SUF.SOEP-RV.VSKT.2020.var.1-0.dta",
    #     columns=["rv_id", "JAHR", "STATUS_2", "MONAT"],
    # )
    # # Check if rv_id is in rv data
    # merged_data = merged_data[
    #     (merged_data["age"] < min_ret_age)
    #     | merged_data["rv_id"].isin(rv_data["rv_id"].unique())
    # ]
    #
    # print(
    #     str(len(merged_data))
    #     + " left after dropping people with missing SOEP-RV ID in SOEP-RV VSKT."
    # )
    #
    # # Prepare merge data
    # merged_data["MONAT"] = 12
    # rv_data["syear"] = rv_data["JAHR"]
    #
    # # Merge with SOEP core data
    # merged_data = merged_data.merge(rv_data, on=["rv_id", "syear", "MONAT"], how="left")
    return merged_data


def gather_wealth_data(soep_c38_path, merged_data, options):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)

    # for each household, create a row for each year between min and max syear
    min_max_syear = wealth_data.groupby("hid")["syear"].agg(["min", "max"])
    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(row["min"], row["max"] + 1)})
            for hid, row in min_max_syear.iterrows()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )

    # Set 'hid' and 'syear' as the index
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full.sort_index(inplace=True)

    # Interpolate the missing values for each household
    wealth_data_full["w011ha"] = wealth_data_full.groupby("hid")["w011ha"].transform(
        lambda group: group.interpolate(method="linear")
    )

    # rename to "wealth" and change unit to 1000s of euros
    wealth_data_full.rename(columns={"w011ha": "wealth"}, inplace=True)
    wealth_data_full["wealth"] = wealth_data_full["wealth"] / options["wealth_unit"]

    merged_data = merged_data.merge(wealth_data_full, on=["hid", "syear"], how="left")

    merged_data = merged_data[(merged_data["wealth"].notna())]

    merged_data.loc[merged_data["wealth"] < 0, "wealth"] = 0

    print(str(len(merged_data)) + " left after dropping people with missing wealth.")

    return merged_data


def create_hh_cons_equivalence_data(merged_data):
    """This function creates the household consumption equivalence scale following the
    OECD-modified equivalence scale."""
    # partner (>0 means has partner)
    merged_data["has_partner"] = 0
    merged_data.loc[merged_data["pgpartz"] >= 1, "has_partner"] = 1

    # number of children (<0 means 0 or "no info", which is treated as 0)
    merged_data["n_children"] = merged_data["hlc0043"]
    merged_data.loc[merged_data["n_children"] < 0, "n_children"] = 0

    # consumption equivalence scale
    merged_data["cons_equiv"] = (
        1 + 0.5 * merged_data["has_partner"] + 0.3 * merged_data["n_children"]
    )

    # drop missing values
    merged_data = merged_data[merged_data["cons_equiv"].notna()]

    print(
        str(len(merged_data))
        + " left after dropping people with missing consumption equivalence scale."
    )
    return merged_data


def deflate_wealth(merged_data, path_dict):
    """This function deflates the wealth variable using the consumer price index."""
    cpi_data = pd.read_csv(
        path_dict["intermediate_data"] + "cpi_base_2010.csv", index_col=0
    )
    merged_data = merged_data.merge(cpi_data, left_on="syear", right_index=True)
    merged_data["wealth"] = merged_data["wealth"] / merged_data["cpi"]
    return merged_data


def filter_data(merged_data, start_year, end_year, start_age):
    """This function filters the data according to the model setup.

    Specifically, it filters out young people, women, and years outside of estimation
    range. It leaves one year younger and one year below in the sample to construct
    lagged_choice.

    """

    # Set pid and syear as index
    merged_data.set_index(["pid", "syear"], inplace=True)

    # filter out young people, women, and years outside of estimation range
    merged_data = merged_data[merged_data["age"] >= start_age - 1]
    print(
        str(len(merged_data))
        + " left after dropping people under "
        + str(start_age)
        + " years old."
    )
    merged_data = merged_data[(merged_data["sex"] == 1)]
    print(str(len(merged_data)) + " left after dropping women.")
    merged_data = merged_data.loc[
        ((slice(None), range(start_year - 1, end_year + 1))), :
    ]
    print(
        str(len(merged_data))
        + " left after dropping people outside of estimation years."
    )
    return merged_data


def create_lagged_choice_variable(merged_data, start_year, end_year, start_age):
    """This function creates the lagged choice variable and drops missing lagged
    choices."""
    # Create full index with all possible combinations of pid and syear
    full_index = pd.MultiIndex.from_product(
        [merged_data.index.levels[0], range(start_year - 1, end_year + 1)],
        names=["pid", "syear"],
    )
    full_container = pd.DataFrame(
        index=full_index, data=np.nan, dtype=float, columns=merged_data.columns
    )
    full_container.update(merged_data)
    full_container["lagged_choice"] = full_container.groupby(["pid"])["choice"].shift()
    merged_data = full_container[full_container["lagged_choice"].notna()]

    # Delete entries of persons missing 2021, but observed in 2020.
    merged_data = merged_data[merged_data["choice"].notna()]

    # Delete people who are start_age - 1 years old
    merged_data = merged_data[merged_data["age"] >= start_age]

    print(str(len(merged_data)) + " left after filtering missing lagged choices.")
    return merged_data


def create_policy_state(gebjahr):
    """This function creates the policy state according to the 2007 reform."""
    # Default state is 67
    policy_state = pd.Series(index=gebjahr.index, data=67, dtype=float)
    # Create masks for everyone born before 1964
    mask1 = (gebjahr <= 1964) & (gebjahr >= 1958)
    mask2 = (gebjahr <= 1958) & (gebjahr >= 1947)
    mask3 = gebjahr < 1947
    policy_state.loc[mask1] = 67 - 2 / 12 * (1964 - gebjahr[mask1])
    policy_state.loc[mask2] = 66 - 1 / 12 * (1958 - gebjahr[mask2])
    policy_state.loc[mask3] = 65
    return policy_state


def modify_policy_state(policy_states, options):
    """This function rounds policy state to the closest multiple of the policy
    expectations process grid size.

    min_SRA is set by the assumption of the belief process in the model.

    """
    min_SRA = options["min_SRA"]
    SRA_grid_size = options["SRA_grid_size"]
    policy_states = policy_states - min_SRA
    policy_id = np.around(policy_states / SRA_grid_size).astype(int)
    policy_states_values = min_SRA + policy_id * SRA_grid_size
    return policy_states_values, policy_id


def enforce_model_choice_restriction(merged_data, min_ret_age, max_ret_age):
    """This function filters the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """
    # Filter out people who are retired before min_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] == 2) & (merged_data["age"] < min_ret_age))
    ]
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] == 2) & (merged_data["age"] <= min_ret_age))
    ]

    # Filter out people who are working after max_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] != 2) & (merged_data["age"] >= max_ret_age))
    ]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] != 2) & (merged_data["age"] > max_ret_age))
    ]
    print(
        str(len(merged_data))
        + " left after dropping people who are retired before "
        + str(min_ret_age)
        + " or working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who come back from retirement
    merged_data = merged_data[
        (merged_data["lagged_choice"] != 2) | (merged_data["choice"] == 2)
    ]

    print(
        str(len(merged_data))
        + " left after dropping people who come back from retirement."
    )
    return merged_data


def print_n_retirees(merged_data):
    n_retirees = merged_data.groupby("choice").size().loc[2]
    print(str(n_retirees) + " retirees in sample.")


def print_n_fresh_retirees(merged_data):
    n_fresh_retirees = (
        merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 1]
        + merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 0]
    )
    print(str(n_fresh_retirees) + " fresh retirees in sample.")


def print_sample_by_choice(merged_data, string):
    n_unemployed = merged_data.groupby(["choice"]).size().loc[0]
    n_workers = merged_data.groupby(["choice"]).size().loc[1]
    n_retirees = merged_data.groupby(["choice"]).size().loc[2]
    n_fresh_retirees = (
        merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 1]
        + merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 0]
    )
    print(
        "Left in sample:"
        + " unemployed: "
        + str(n_unemployed)
        + ", workers: "
        + str(n_workers)
        + ", retirees: "
        + str(n_retirees)
        + ", fresh retirees: "
        + str(n_fresh_retirees)
        + ", after"
        + string
    )
