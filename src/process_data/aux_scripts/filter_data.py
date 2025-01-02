def filter_years(df, start_year, end_year):
    df = df.loc[(slice(None), range(start_year, end_year + 1)), :]
    print(
        str(len(df))
        + f" left after dropping people outside of estimation years {start_year} - {end_year}."
    )
    return df


def filter_below_age(df, age):
    df = df[df["age"] >= age]
    print(
        str(len(df)) + " left after dropping people under " + str(age) + " years old."
    )
    return df


def filter_above_age(df, age):
    df = df[df["age"] <= age]
    print(str(len(df)) + " left after dropping people over " + str(age) + " years old.")
    return df


def recode_sex(df):
    """Recode sex to 0(men) and 1(women), from SOEP definition 1(men) and 2(women)."""
    df.loc[:, "sex"] = df["sex"] - 1
    return df


def filter_data(merged_data, specs, lag_and_lead_buffer_years=True):
    """This function filters the data according to the model setup.

    Specifically, it filters out young people, women (if no_women=True), and years
    outside of estimation range. It leaves one year younger and one year below in the
    sample to construct lagged_choice.

    """
    merged_data = filter_below_age(merged_data, specs["start_age"] - 1)

    merged_data = recode_sex(merged_data)

    if lag_and_lead_buffer_years:
        start_year = specs["start_year"] - 1
        end_year = specs["end_year"] + 1
    else:
        start_year = specs["start_year"]
        end_year = specs["end_year"]

    merged_data = filter_years(merged_data, start_year, end_year)
    return merged_data
