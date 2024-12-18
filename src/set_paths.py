# Set data paths according to user.
from pathlib import Path

import jax


def create_path_dict(define_user=False, user=None):
    jax.config.update("jax_enable_x64", True)
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
            "soep_c38": data_path + "soep38",
            "soep_rv": data_path + "soep_rv",
            "soep_is": data_path + "soep_is_2022/dataset_main_SOEP_IS.dta",
        }
    else:
        paths_dict = {}

    analysis_path = str(Path(__file__).resolve().parents[1]) + "/"

    paths_dict = {
        **paths_dict,
        "intermediate_data": analysis_path + "output/intermediate_data/",
        "est_results": analysis_path + "output/est_results/",
        "tables": analysis_path + "output/tables/",
        "specs": analysis_path + "src/spec.yaml",
        "start_params_and_bounds": analysis_path
        + "src/estimation/struct_estimation/start_params_and_bounds/",
        "est_params": analysis_path + "output/est_results/est_params.pkl",
        "plots": analysis_path + "output/plots/",
    }
    paths_dict["struct_est_sample"] = (
        paths_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    return paths_dict
