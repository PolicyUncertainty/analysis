import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from set_styles import set_colors, get_figsize
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_state_by_age_and_type(path_dict, state_vars, specs=None, show=False, save=False):
    """Plot state variables by age and education type.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    state_vars : list
        List of state variables to plot
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

    # age restriction
    struct_est_sample["age"] = struct_est_sample["period"] + specs["start_age"]
    start = specs["start_age"]
    end = specs["max_ret_age"]

    struct_est_sample = struct_est_sample[struct_est_sample["age"] < end]
    plot_ages = np.arange(start, end + 1)

    n_education_types = specs["n_education_types"]
    n_state_vars = len(state_vars)
    #fig, axs = plt.subplots(
    #    nrows=n_education_types, ncols=n_state_vars, figsize=(15, 5 * n_education_types)
    #)
    fig, axs = plt.subplots(
        nrows=n_education_types, ncols=n_state_vars, figsize=get_figsize(nrows=n_education_types, ncols=n_state_vars)
    )


    for edu in range(n_education_types):
        edu_df = struct_est_sample[struct_est_sample["education"] == edu]
        for idx, state_var in enumerate(state_vars):
            if "median" in state_var:
                col_name = state_var.split(" ")[1]
                state_values = (
                    edu_df.groupby("age")[col_name]
                    .median()
                    .reindex(plot_ages, fill_value=np.nan)
                    .values
                )
            elif "mean" in state_var:
                col_name = state_var.split(" ")[1]
                state_values = (
                    edu_df.groupby("age")[col_name]
                    .mean()
                    .reindex(plot_ages, fill_value=np.nan)
                    .values
                )
            else:
                raise ValueError("Only median and mean are supported")
            ax = axs[edu, idx] if n_education_types > 1 else axs[idx]
            ax.plot(plot_ages, state_values, label=state_var, color=colors[edu])
            #ax.legend()
            ax.set_title(f"{specs['education_labels'][edu]}: {state_var}")
            ax.set_xlabel("Age")
            ax.set_ylabel(state_var)
            ax.set_ylim(bottom=0)  # Adjust as needed

    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["data_plots"] + "states_by_age_and_type.pdf", bbox_inches="tight")
        fig.savefig(path_dict["data_plots"] + "states_by_age_and_type.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
