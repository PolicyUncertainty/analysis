def filter_est_years(df, start_year, end_year):
    df = df.loc[(slice(None), range(start_year - 1, end_year + 2)), :]
    print(
        str(len(df)) + " left after dropping people outside of estimation years (+-1)."
    )
    return df


def filter_below_age(df, age):
    # filter out young people, women, and years outside of estimation range
    df = df[df["age"] >= age]
    print(
        str(len(df)) + " left after dropping people under " + str(age) + " years old."
    )
    return df


def filter_by_sex(df, no_women):
    df.loc[:, "sex"] = df["sex"] - 1
    if no_women:
        df = df[(df["sex"] == 0)]
        print(str(len(df)) + " left after dropping women.")
    else:
        df = df[(df["sex"] >= 0)]
    return df


def filter_data(merged_data, specs, no_women=True):
    """This function filters the data according to the model setup.

    Specifically, it filters out young people, women (if no_women=True), and years
    outside of estimation range. It leaves one year younger and one year below in the
    sample to construct lagged_choice.

    """
    merged_data = filter_below_age(merged_data, specs["start_age"] - 1)

    merged_data = filter_by_sex(merged_data, no_women=no_women)

    merged_data = filter_est_years(merged_data, specs["start_year"], specs["end_year"])
    return merged_data
