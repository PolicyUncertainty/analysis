
def create_path_dict(user, analysis_path):
    if user == "bruno":
        data_path = "C:/Users/bruno/papers/soep/"
    elif user == "max":
        data_path = "/home/maxbl/Uni/pol_uncetainty/data/"
    else:
        raise ValueError(
            "Please specify valid USER in " "MASTER_prepare_structural_model.py."
        )

    # Set paths
    paths_dict = {
        "soep_c38": data_path + "soep38",
        "soep_rv": data_path + "soep_rv",
        "soep_is": data_path + "soep_is_2022/dataset_main_SOEP_IS.dta",
        "project_path": analysis_path,
        "output_path": analysis_path + "output/",
    }
    return paths_dict