import numpy as np
import pandas as pd


def gather_decision_data(paths, options, load_data=False):
    if load_data:
        data = pd.read_pickle("output/decision_data.pkl")
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

    print(str(len(merged_data))+" observations in SOEP C38 core.")

    # Calculate age and filter out missing rv_id values for people older than minimum retirement age
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data = merged_data[
        (merged_data["rv_id"] >= 0)
        | ((merged_data["rv_id"] < 0) & (merged_data["age"] < min_ret_age))
    ]
    merged_data["rv_id"].replace(-2, pd.NA, inplace=True)

    print(str(len(merged_data))+" left after dropping over "+str(min_ret_age)+" year olds w/o SOEP-RV ID.")

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

    # Create choice variable
    merged_data["choice"] = create_choice_variable(
        rv_ret_choice=merged_data["STATUS_2"], soep_empl_choice=merged_data["pgemplst"]
    )
    merged_data = merged_data[merged_data["choice"].notna()]

    # Calculate period and filter out young people
    merged_data["period"] = merged_data["age"] - start_age
    merged_data = merged_data[merged_data["period"] >= 0]

    # Set pid and syear as index
    merged_data.set_index(["pid", "syear"], inplace=True)


    # Filter out women and years outside of estimation range
    merged_data = merged_data[(merged_data["sex"] == 1)]
    merged_data = merged_data.loc[((slice(None), range(start_year - 1, end_year + 1))), :]

    print(str(len(merged_data))+" left after dropping women, men under "+str(start_age)+" years old, and people outside of estimation years.")


    # Create lagged choice variable
    full_index = pd.MultiIndex.from_product(
        [merged_data.index.levels[0], range(start_year - 1, end_year + 1)],
        names=["pid", "syear"],
    )
    full_container = pd.DataFrame(index=full_index, data=np.nan, dtype=float, columns=merged_data.columns)
    full_container.update(merged_data)
    full_container["lagged_choice"] = full_container.groupby(["pid"])["choice"].shift()
    merged_data = full_container[full_container["lagged_choice"].notna()]
    # Delete entries of persons missing 2021, but observed in 2020.
    merged_data = merged_data[merged_data["choice"].notna()]

    print(str(len(merged_data))+" left after filtering missing lagged choices.")

    # Calculate policy_state according to 2007 reform
    merged_data["policy_state"] = create_policy_state(merged_data["gebjahr"])

    # Create retirement_age_id (empty for now)
    merged_data["retirement_age_id"] = np.nan

    # Filter out invalid experience values
    merged_data = merged_data[
        (merged_data["pgexpft"] >= 0) & (merged_data["pgexpft"] <= exp_cap)
    ]

    # Round experience values
    merged_data["experience"] = merged_data["pgexpft"].astype(float).round()

    # Keep relevant columns (i.e. state variables)
    merged_data = merged_data[
        [
            "choice",
            "period",
            "lagged_choice",
            "policy_state",
            "retirement_age_id",
            "experience",
        ]
    ]

    print(str(len(merged_data))+" in final sample, after dropping invalid experience values.")


    # Save data
    merged_data.to_pickle("output/decision_data.pkl")
    return merged_data


def create_choice_variable(rv_ret_choice, soep_empl_choice):
    """This function creates the choice variable for the structural model.
    TODO: This function assumes reti    rees with part-time employment as full-time retirees.

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




