from process_data.create_job_sep_sample import create_job_sep_sample


def est_job_sep(paths_dict, specs, load_data=False):
    df_job = create_job_sep_sample(paths_dict, specs, load_data)
