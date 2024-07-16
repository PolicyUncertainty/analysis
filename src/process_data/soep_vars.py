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
