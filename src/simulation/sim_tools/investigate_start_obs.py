from model_code.transform_data_from_model import load_scale_and_correct_data


def investigate_start_obs(
    path_dict,
    model_class,
):

    observed_data = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    # Get table of median and mean for continous variables
    # Get table of median and mean wealth
    median = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period"]
    ].median()
    mean = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period"]
    ].mean()
    rename_median = {col: col + "_median" for col in median.columns}
    rename_mean = {col: col + "_mean" for col in mean.columns}
    median = median.rename(columns=rename_median)
    mean = mean.rename(columns=rename_mean)
    initial_obs_table = median.merge(mean, left_index=True, right_index=True)
    initial_obs_table.to_csv(path_dict["data_tables"] + "initial_obs_table.csv")
    return initial_obs_table
