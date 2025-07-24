# Set data paths according to user.
import os
from pathlib import Path

import jax
import matplotlib.pyplot as plt


def create_path_dict(define_user=False, user=None):
    # Set jax to 64 bit
    jax.config.update("jax_enable_x64", True)

    set_standard_matplotlib_specs()

    if define_user:
        if user is None:
            user = input("Enter user name ([b]runo / [m]ax / [g]regor): ")
        else:
            pass

        if user == "b":
            data_path = "C:/Users/bruno/papers/soep/"
        elif user == "m":
            data_path = "/home/maxbl/Uni/pol_uncertainty/data/"
        elif user == "g":
            data_path = "/Users/gregorschuler/GitProjects/soep/"
        else:
            raise ValueError(
                "Please specify valid USER in " "MASTER_prepare_structural_model.py."
            )
        # Set user specified paths
        paths_dict = {
            "soep_c38": data_path + "soep38/",
            "soep_rv": data_path + "soep_rv/",
            "soep_is": data_path + "soep_is/soep-is.2023_stata_de/",
        }
    else:
        paths_dict = {}

    analysis_path = str(Path(__file__).resolve().parents[1]) + "/"
    paper_path = str(Path(__file__).resolve().parents[2]) + "/paper/"

    # Assign input folders
    paths_dict = {
        **paths_dict,
        "intermediate_data": analysis_path + "output/intermediate_data/",
        "open_data": analysis_path + "output/open_access_data/",
    }
    # Assign result folders
    paths_dict["est_results"] = analysis_path + "output/est_results/"
    paths_dict["first_step_results"] = analysis_path + "output/est_results/first_step/"
    paths_dict["struct_results"] = analysis_path + "output/est_results/struct_results/"
    paths_dict["sim_results"] = analysis_path + "output/sim_results/"

    # Assign output folders
    paths_dict["tables"] = analysis_path + "output/tables/"
    paths_dict["plots"] = analysis_path + "output/plots/"

    # Assign folders directly in paper
    paths_dict["paper_plots"] = analysis_path + "output/paper_plots/"
    paths_dict["paper_tables"] = paper_path + "tables/"

    # Assign model specification file
    paths_dict["specs"] = analysis_path + "src/spec.yaml"

    # Assign start params and bounds folder for structural estimation
    paths_dict["start_params_and_bounds"] = (
        analysis_path + "src/estimation/struct_estimation/start_params_and_bounds/"
    )

    # Assign name of structural estimation sample
    paths_dict["struct_est_sample"] = (
        paths_dict["intermediate_data"] + "structural_estimation_sample.csv"
    )

    return paths_dict


def get_model_resutls_path(paths_dict, model_name):
    model_folder = paths_dict["intermediate_data"] + "model_" + model_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    folder_dict = {
        "model_results": model_folder,
    }

    for folder in ["solution", "simulation"]:
        folder_name = model_folder + "/" + folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        folder_dict[folder] = folder_name + "/"

    return folder_dict


def set_standard_matplotlib_specs():
    # Set matplotlib fontsizes
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    # Make lines of plots thicker
    plt.rcParams["lines.linewidth"] = 3
