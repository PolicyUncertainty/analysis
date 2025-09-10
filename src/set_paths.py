# Set data paths according to user.
import os
from pathlib import Path

import jax




def create_path_dict(define_user=False, user=None):
    # Set jax to 64 bit
    jax.config.update("jax_enable_x64", True)

    # Assign raw data paths (only if define_user is True)
    if define_user:
        if user is None:
            detected_user = detect_user_from_path()
            if detected_user:
                user = detected_user
            else:
                user = input("Enter user name ([b]runo / [m]ax / [g]regor): ")
        else:
            pass

        if user == "b":
            data_path = "C:/Users/bruno/papers/soep/"
        elif user == "m":
            data_path = "/home/maxbl/Uni/data/"
        elif user == "g":
            data_path = "/Users/gregorschuler/GitProjects/soep/"
        else:
            raise ValueError(
                "Please specify valid USER in " "MASTER_prepare_structural_model.py."
            )
        # Set user specified paths
        paths_dict = {
            "soep_c38": data_path + "soep38",
            "soep_rv": data_path + "soep_rv",
            "soep_is": data_path + "soep_is/soep-is.2023_stata_de/",
        }
    else:
        paths_dict = {}

    analysis_path = str(Path(__file__).resolve().parents[1]) + "/"

    # Assign input folders
    paths_dict = {
        **paths_dict,
        "intermediate_data": analysis_path + "output/intermediate_data/",
        "open_data": analysis_path + "output/open_access_data/",
    }
    # Assign est result folders
    paths_dict["est_results"] = analysis_path + "output/est_results/"
    paths_dict["first_step_results"] = (
        analysis_path + "output/est_results/first_step/"
    )  # legacxy path
    paths_dict["first_step_incomes"] = (
        analysis_path + "output/est_results/incomes/"
    )  # legacy path
    paths_dict["struct_results"] = analysis_path + "output/est_results/struct_results/"

    paths_dict["sim_results"] = analysis_path + "output/sim_results/"

    # Assign plot and table folders
    paths_dict["tables"] = analysis_path + "output/tables/"
    paths_dict["plots"] = analysis_path + "output/plots/"

    # Assign plot and table subdolders
    for subfolder in [
        "beliefs",
        "data",
        "model",
        "first_step",
        "struct",
        "validation",
        "simulation",
        "misc",
    ]:
        folder_name_plots = paths_dict["plots"] + subfolder
        folder_name_tables = paths_dict["tables"] + subfolder
        paths_dict[subfolder + "_plots"] = folder_name_plots + "/"
        paths_dict[subfolder + "_tables"] = folder_name_tables + "/"

    # Assign model specification file
    paths_dict["specs"] = analysis_path + "src/spec.yaml"

    # Assign start params and bounds folder for structural estimation
    paths_dict["start_params_and_bounds"] = (
        analysis_path + "src/estimation/struct_estimation/start_params_and_bounds/"
    )

    # Check if entries of paths_dict exist, if not create them
    for key, path in paths_dict.items():
        if not os.path.exists(path):
            os.makedirs(path)

    # Assign name of structural estimation sample
    paths_dict["struct_est_sample"] = (
        paths_dict["intermediate_data"] + "structural_estimation_sample.csv"
    )
    return paths_dict


def detect_user_from_path():
    """Detect user from current working directory path."""
    current_path = str(Path.cwd()).lower()
    user_mapping = {
        'bruno': 'b',
        'maxbl': 'm', 
        'gregorschuler': 'g'
    }
    for username, user_key in user_mapping.items():
        if username in current_path:
            print(f"Auto-detected user: {username}")
            return user_key
    return None

def get_model_results_path(paths_dict, model_name):
    model_folder = paths_dict["intermediate_data"] + "model_" + model_name + "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    folder_dict = {
        "model_results": model_folder,
    }

    for folder in ["solution", "simulation"]:
        folder_name = model_folder + folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        folder_dict[folder] = folder_name + "/"

    return folder_dict
