import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors, get_figsize
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_average_wealth_by_type(path_dict, specs=None, show=False, save=False):
    """Plot average wealth by education type and sex.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict, optional
        Model specifications. If None, will be loaded from file.
    show : bool, default False
        Whether to display plots
    save : bool, default False  
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])

    if specs is None:
        specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]

    wealth_by_type = struct_est_sample.groupby(["sex", "education", "age"])[
        "wealth"
    ].median()

    fig, axs = plt.subplots(ncols=specs["n_education_types"], figsize=get_figsize(nrows=1, ncols=specs["n_education_types"]))
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        ax = axs[edu_var]
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax.plot(
                wealth_by_type.loc[(sex_var, edu_var, slice(None))],
                label=f"Median {sex_label}",
                color=colors[sex_var],
            )
        ax.set_title(f"{edu_label}")
        ax.legend()
        ax.set_xlabel("Age")
        ax.set_ylabel("Wealth")
    
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["data_plots"] + "wealth_by_type.pdf", bbox_inches="tight")
        fig.savefig(path_dict["data_plots"] + "wealth_by_type.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
