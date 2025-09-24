import numpy as np


def create_education_type(data, filter_missings=False):
    """This function creates a education type from pgpsbil in soep-pgen.

    The function uses a two category split of the population, encoding 1 if an
    individual has at least Fachhochschulreife.

    """
    data["education"] = 0
    data.loc[data["pgpsbil"] < 0, "education"] = np.nan
    data.loc[data["pgpsbil"] == 3, "education"] = 1  # Fachhochschulreife
    data.loc[data["pgpsbil"] == 4, "education"] = 1  # Abitur

    if filter_missings:
        print(
            str(len(data))
            + " left after dropping people with missing education values."
        )
        data = data[data["education"].notna()]
    return data
