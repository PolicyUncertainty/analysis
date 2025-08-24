import matplotlib.pyplot as plt


def print_choice_probs_by_group(df, specs):
    df = df[df["sex"] == 0]
    df = df[df["age"] >= 63]
    df = df[df["age"] < 70]

    # df.groupby("age")["choice"].value_counts(normalize=True).loc[(slice(None), 0)].plot(label="observed")
    # df.groupby("age")["choice_0"].mean().plot(label="predicted")
    # plt.legend()
    # plt.show()
    not_retired = df["lagged_choice"] != 0

    #
    # breakpoint()
    df = df[not_retired].copy()
    #
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #
    #     breakpoint()
