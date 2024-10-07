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
