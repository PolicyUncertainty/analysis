# Set data paths according to user.
import jax


def create_path_dict(analysis_path, define_user=False, user=None):
    jax.config.update("jax_enable_x64", True)
    if define_user:
        if user is None:
            user = input("Enter user name ([b]runo/ [m]ax): ")
        else:
            pass

        if user == "b":
            data_path = "C:/Users/bruno/papers/soep/"
        elif user == "m":
            data_path = "/home/maxbl/Uni/pol_uncetainty/data/"
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

    paths_dict = {
        **paths_dict,
        "intermediate_data": analysis_path + "output/intermediate_data/",
        "est_results": analysis_path + "output/est_results/",
        "specs": analysis_path + "src/spec.yaml",
        "start_params": analysis_path + "src/estimation/start_params.yaml",
        "est_params": analysis_path + "output/est_results/est_params.pkl",
        "plots": analysis_path + "output/plots/",
    }
    return paths_dict
