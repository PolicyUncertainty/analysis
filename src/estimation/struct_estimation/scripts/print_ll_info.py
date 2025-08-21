def print_choice_probs_by_group(df, specs):
    not_retired = df["lagged_choice"] != 0

    df = df[not_retired].copy()
    df = df[df["sex"] == 0]
    #
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #
    #     breakpoint()
