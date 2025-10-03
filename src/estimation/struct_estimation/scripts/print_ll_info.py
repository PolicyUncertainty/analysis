import matplotlib.pyplot as plt

from process_data.structural_sample_scripts.classify_reitrees import (
    add_very_long_insured_classification,
)


def print_choice_probs_by_group(df, specs, path_dict):
    df = df[df["sex"] == 0]
    # df = df[df["age"] >= 63]
    # df = df[df["age"] < 70]
    not_retired = df["lagged_choice"] != 0
    df = df[not_retired].copy()
    df["experience_float"] = df["experience"].copy()
    df["experience"] = df["experience_years"].copy()
    df = add_very_long_insured_classification(df=df, path_dict=path_dict, specs=specs)

    # df.groupby("age")["choice"].value_counts(normalize=True).loc[(slice(None), 0)].plot(label="observed")
    # df.groupby("age")["choice_0"].mean().plot(label="predicted")
    # plt.legend()
    # plt.show()

    #    df = df[not_retired].copy()

    #
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #
