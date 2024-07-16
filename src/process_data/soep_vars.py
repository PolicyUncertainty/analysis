import numpy as np


def create_education_type(data):
    """This function creates a education type from pgpsbil in soep-pgen.

    The function uses a two category split of the population, encoding 1 if an
    individual has at least Fachhochschulreife.

    """
    data = data[data["pgpsbil"].notna()]
    data["education"] = 0
    data.loc[data["pgpsbil"] == 3, "education"] = 1  # Fachhochschulreife
    data.loc[data["pgpsbil"] == 4, "education"] = 1  # Abitur
    print(str(len(data)) + " left after dropping people with missing education values.")
    return data


def create_choice_variable(data):
    """This function creates the choice variable for the structural model.

    TODO: This function assumes retirees with part-time employment as full-time retirees.

    """
    data["choice"] = np.nan
    soep_empl_choice = data["pgemplst"]
    soep_empl_status = data["pgstib"]
    # rv_ret_choice = merged_data["STATUS_2"]

    # Now assign emploayment choices
    data.loc[soep_empl_choice == 5, "choice"] = 0
    data.loc[soep_empl_choice == 1, "choice"] = 1

    # Finally retirement choice
    data.loc[soep_empl_status == 13, "choice"] = 2
    # merged_data.loc[rv_ret_choice == "RTB"] = 2
    merged_data = data[data["choice"].notna()]
    return merged_data


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
    # If both are valid use the sum
    data.loc[~invalid_ft_exp & ~invalid_pt_exp, "experience"] = (
        data["pgexpft"] + 0.5 * data["pgexppt"]
    )
    # If only one is valid use the valid one
    data.loc[invalid_ft_exp & ~invalid_pt_exp, "experience"] = 0.5 * data["pgexppt"]
    data.loc[~invalid_ft_exp & invalid_pt_exp, "experience"] = data["pgexpft"]
    # If both are invalid drop observations
    data = data[data["experience"].notna()]
    print(
        str(len(data)) + " left after dropping people with invalid experience values."
    )
    return data
