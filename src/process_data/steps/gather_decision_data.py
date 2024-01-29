import numpy as np
import pandas as pd


def gather_decision_data(paths, options, policy_step_size, load_data=False):
    out_file_path = paths["project_path"] + "output/decision_data.pkl"
    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Set file paths
    soep_c38 = paths["soep_c38"]
    soep_rv = paths["soep_rv"]

    # Export parameters from options
    start_year = options["start_year"]
    end_year = options["end_year"]
    min_ret_age = options["min_ret_age"]
    start_age = options["start_age"]
    exp_cap = options["exp_cap"]

    # Load and merge data state data from SOEP core and SOEP RV VSKT (all but wealth)
    merged_data = load_and_merge_data(soep_c38, soep_rv, min_ret_age)

     # wealth data from SOEP C38 (hwealth.dta)
    wealth_data = gather_wealth_data(soep_c38, start_year, end_year)
    merged_data = merged_data.merge(wealth_data, on=["hid", "syear"], how="left")

    merged_data = merged_data[merged_data["wealth"].notna()]
    merged_data[merged_data["wealth"] < 0] = 0

    # Filter data
    merged_data = filter_data(merged_data, start_year, end_year, start_age, exp_cap)

    # (labor) choice
    merged_data["choice"] = create_choice_variable(
        rv_ret_choice=merged_data["STATUS_2"], soep_empl_choice=merged_data["pgemplst"]
    )
    merged_data = merged_data[merged_data["choice"].notna()]

    # period
    merged_data["period"] = merged_data["age"] - start_age

    # lagged choice
    merged_data = create_lagged_choice_variable(merged_data, start_year, end_year)

    # policy_state
    merged_data["policy_state"] = create_policy_state(merged_data["gebjahr"])
    (
        merged_data["policy_state_value"],
        merged_data["policy_state"],
    ) = modify_policy_state(merged_data["policy_state"], policy_step_size, options)

    # retirement_age_id (dummy 0 for now)
    merged_data["retirement_age_id"] = 0

    # experience
    merged_data["experience"] = merged_data["pgexpft"].astype(float).round()

    # additional filters based on model setup
    merged_data = enforce_model_work_and_ret_conditions(
        merged_data, min_ret_age, options["max_ret_age"], start_age
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
    }
)

    print(str(len(merged_data)) + " in final sample.")

    # Save data
    merged_data.to_pickle(out_file_path)
    return merged_data


def load_and_merge_data(soep_c38, soep_rv, min_ret_age):
    # Load SOEP core data
    core_data = pd.read_stata(
        f"{soep_c38}/pgen.dta",
        columns=["syear", "pid", "hid", "pgemplst", "pgexpft"],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        f"{soep_c38}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr", "rv_id"],
        convert_categoricals=False,
    )

    # Merge core data with pathl data
    merged_data = pd.merge(
        core_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    print(str(len(merged_data)) + " observations in SOEP C38 core.")

    # Calculate age and filter out missing rv_id values for people older than minimum retirement age
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data = merged_data[
        (merged_data["rv_id"] >= 0)
        | ((merged_data["rv_id"] < 0) & (merged_data["age"] < min_ret_age))
    ]
    merged_data["rv_id"].replace(-2, pd.NA, inplace=True)

    print(
        str(len(merged_data))
        + " left after dropping over "
        + str(min_ret_age)
        + " year olds w/o SOEP-RV ID."
    )

    # Load SOEP RV VSKT data
    rv_data = pd.read_stata(
        f"{soep_rv}/vskt/SUF.SOEP-RV.VSKT.2020.var.1-0.dta",
        columns=["rv_id", "JAHR", "STATUS_2", "MONAT"],
    )

    # Prepare merge data
    merged_data["MONAT"] = 12
    rv_data["syear"] = rv_data["JAHR"]

    # Merge with SOEP core data
    merged_data = merged_data.merge(rv_data, on=["rv_id", "syear", "MONAT"], how="left")
    return merged_data


def gather_wealth_data(soep_c38, start_year, end_year):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38}/hwealth.dta",
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
    wealth_data_full["wealth"] = wealth_data_full["wealth"] / 1000

    return wealth_data_full


def filter_data(merged_data, start_year, end_year, start_age, exp_cap):
    # Set pid and syear as index
    merged_data.set_index(["pid", "syear"], inplace=True)

    # filter out young people, women, and years outside of estimation range
    merged_data = merged_data[merged_data["age"] >= start_age]
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

    # Filter out invalid experience values
    merged_data = merged_data[
        (merged_data["pgexpft"] >= 0) & (merged_data["pgexpft"] <= exp_cap)
    ]
    print(
        str(len(merged_data))
        + " left after dropping people with invalid experience values."
    )

    # drop missing wealth values and set negative wealth to 0
    merged_data = merged_data[merged_data["wealth"].notna()]
    merged_data[merged_data["wealth"] < 0] = 0
    print(str(len(merged_data)) + " left after dropping people with missing wealth.")

    return merged_data


def create_lagged_choice_variable(merged_data, start_year, end_year):
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

    print(str(len(merged_data)) + " left after filtering missing lagged choices.")
    return merged_data


def create_choice_variable(rv_ret_choice, soep_empl_choice):
    """This function creates the choice variable for the structural model.

    TODO: This function assumes retirees with part-time employment as full-time retirees.

    """
    choice = pd.Series(index=rv_ret_choice.index, data=np.nan, dtype=float)
    choice.loc[soep_empl_choice == 5] = 0
    choice.loc[soep_empl_choice == 1] = 1
    choice.loc[rv_ret_choice == "RTB"] = 2
    return choice


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


def modify_policy_state(policy_states, policy_step_size, options):
    """This function rounds policy state to closest multiple of the policy expectations
    process step size.

    65 is hard coded b/c of reference to law.

    """
    min_policy_state = options["min_policy_state"]
    policy_states = policy_states - min_policy_state
    policy_id = np.around(policy_states / policy_step_size).astype(int)
    policy_states = min_policy_state + policy_id * policy_step_size
    return policy_states, policy_id


def enforce_model_work_and_ret_conditions(
    merged_data, min_ret_age, max_ret_age, start_age
):
    """This function filters the choice data according to the model setup."""
    # Filter out people who are retired before min_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] == 2) & (merged_data["age"] < min_ret_age))
    ]

    # Filter out people who are working after max_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] != 2) & (merged_data["age"] > max_ret_age))
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
