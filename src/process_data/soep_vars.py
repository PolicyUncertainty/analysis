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
