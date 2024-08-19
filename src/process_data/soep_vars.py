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

    # assign emploayment choices
    data.loc[soep_empl_choice == 5, "choice"] = 0
    data.loc[soep_empl_choice == 1, "choice"] = 1

    # assign retirement choice
    data.loc[soep_empl_status == 13, "choice"] = 2
    # merged_data.loc[rv_ret_choice == "RTB"] = 2
    data = data[data["choice"].notna()]
    return data


def create_choice_variable_with_part_time(data):
    """This function creates the choice variable for the structural model.

    It includes part-time employment as a separate choice. (0: unemployed, 1: full-time, 2: retired, 3: part-time)
    """
    data["choice"] = np.nan
    soep_empl_choice = data["pgemplst"]
    soep_empl_status = data["pgstib"]

    # assign emploayment choices
    data.loc[soep_empl_choice == 5, "choice"] = 0
    data.loc[soep_empl_choice == 1, "choice"] = 1
    data.loc[soep_empl_choice == 2, "choice"] = 3

    # assign retirement choice
    data.loc[soep_empl_status == 13, "choice"] = 2
    # merged_data.loc[rv_ret_choice == "RTB"] = 2
    data = data[data["choice"].notna()]
    return data


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


def generate_job_separation_var(data):
    """This function generates a job separation variable.

    The function creates a new column job_sep which is 1 if the individual got fired
    from the last job. It uses plb0304_h from the soep pl data.

    """
    data["job_sep"] = 0
    data.loc[data["plb0304_h"].isin([1, 3, 5]), "job_sep"] = 1
    return data
