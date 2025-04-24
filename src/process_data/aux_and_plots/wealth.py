import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_average_wealth_by_type(path_dict):
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])

    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]

    wealth_by_type = struct_est_sample.groupby(["sex", "education", "age"])[
        "wealth"
    ].median()

    fig, axs = plt.subplots(ncols=specs["n_education_types"])
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_var]
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax.plot(
                wealth_by_type.loc[(sex_var, edu_var, slice(None))],
                label=f"Median {sex_label}",
            )
        ax.set_title(f"{edu_label}")
        ax.legend()
