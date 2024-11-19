import numpy as np


def create_experience_variable_with_cap(data, exp_cap):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience.

    It als enforces an experience cap.

    """
    # Create experience variable
    data = create_experience_variable(data)
    # Enforce experience cap
    data.loc[data["experience"] > exp_cap, "experience"] = exp_cap
    return data


def create_experience_variable(data):
    """This function creates an experience variable as the sum of full-time and 0.5
    weighted part-time experience and rounds the sum."""
    data = sum_experience_variables(data)
    # Round experience
    data.loc[:, "experience"] = data["experience"].round()
    return data


def sum_experience_variables(data):
    """This function sums the experience variables pgexpft and pgexppt.

    Part time experience is weighted by 0.5. The function returns a new column
    experience.

    """
    invalid_ft_exp = data["pgexpft"] < 0
    invalid_pt_exp = data["pgexppt"] < 0

    # Initialize empty experience column
    data["experience"] = np.nan

    # Check if years of part plus full time exceed age minus 14 (not allowed to work before)
    max_exp = data["age"] - 14
    exp_exceeding = ((data["pgexpft"] + data["pgexppt"]) - max_exp).clip(lower=0)
    # Deduct exceeding experience from part time experience. Assume if worked both, you worked full
    data.loc[:, "pgexppt"] -= exp_exceeding

    # If both are valid use the sum
    data.loc[~invalid_ft_exp & ~invalid_pt_exp, "experience"] = (
        data.loc[~invalid_ft_exp & ~invalid_pt_exp, "pgexpft"]
        + 0.5 * data.loc[~invalid_ft_exp & ~invalid_pt_exp, "pgexppt"]
    )
    # If only one is valid use the valid one
    data.loc[invalid_ft_exp & ~invalid_pt_exp, "experience"] = (
        0.5 * data.loc[invalid_ft_exp & ~invalid_pt_exp, "pgexppt"]
    )
    data.loc[~invalid_ft_exp & invalid_pt_exp, "experience"] = data.loc[
        ~invalid_ft_exp & invalid_pt_exp, "pgexpft"
    ]
    # If both are invalid drop observations
    data = data[data["experience"].notna()]
    print(
        str(len(data)) + " left after dropping people with invalid experience values."
    )
    return data
