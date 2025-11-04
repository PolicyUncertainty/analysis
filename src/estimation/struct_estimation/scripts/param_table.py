import os
import pickle


def create_latex_tables(path_dict, model_name):
    """
    Create three separate LaTeX tables from model parameters and standard errors.
    Each table includes the full tabular environment with headers.

    Parameters:
    -----------
    path_dict : dict
        Dictionary containing paths for input and output directories
    model_name : str
        The name of the model (used to construct filenames)

    Returns:
    --------
    dict : Dictionary containing the three table filenames with keys:
           'disutility', 'job_offer', 'disability'
    """

    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
    std_errors_men = pickle.load(
        open(path_dict["struct_results"] + f"std_errors_{model_name}_men.pkl", "rb")
    )
    std_errors_women = pickle.load(
        open(path_dict["struct_results"] + f"std_errors_{model_name}_women.pkl", "rb")
    )

    out_dir = path_dict["estimation_tables"] + f"{model_name}/"
    os.makedirs(out_dir, exist_ok=True)

    def format_param_row(label, param_key_men, param_key_women):
        """
        Format a parameter row with standard errors below in parentheses.
        Two columns: one for men, one for women.
        Each column shows: value on first line, (std_error) on second line.
        """

        # Get men's parameter and standard error
        param_men = params.get(param_key_men)
        se_men = std_errors_men.get(param_key_men)

        # Get women's parameter and standard error
        param_women = params.get(param_key_women)
        se_women = std_errors_women.get(param_key_women)

        # Format men's column
        if param_men is not None:
            men_value = f"{param_men:.4f}"
            men_se = f"({se_men:.4f})" if se_men is not None else ""
        else:
            men_value = "{}"
            men_se = "{}"

        # Format women's column
        if param_women is not None:
            women_value = f"{param_women:.4f}"
            women_se = f"({se_women:.4f})" if se_women is not None else ""
        else:
            women_value = "{}"
            women_se = "{}"

        # Create two rows: parameter values, then standard errors
        param_row = f"    {label} & {men_value} & {women_value} \\\\"
        se_row = f"    {{}} & {men_se} & {women_se} \\\\"

        return param_row + "\n" + se_row

    def create_table_wrapper(content):
        """Wrap table content in LaTeX tabular environment"""
        table = []
        table.append("\\begin{tabular}{llc}")
        table.append("    \\toprule")
        table.append("     Parameter Name  & \\multicolumn{2}{c}{Estimates} \\\\")
        table.append("     {}              & {Men} &{Women}   \\\\")
        table.append("    \\midrule")
        table.append(content)
        table.append("    \\bottomrule")
        table.append("\\end{tabular}")
        return "\n".join(table)

    # ========================================================================
    # TABLE 1: DISUTILITY PARAMETERS
    # ========================================================================
    disutil_rows = []

    # Unemployed
    disutil_rows.append(
        format_param_row(
            "Unemployed; Bad Health",
            "disutil_unemployed_bad_men",
            "disutil_unemployed_bad_women",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Unemployed; Good Health",
            "disutil_unemployed_good_men",
            "disutil_unemployed_good_women",
        )
    )

    # Full-time
    disutil_rows.append(
        format_param_row(
            "Full-time; Bad Health",
            "disutil_ft_work_bad_men",
            "disutil_ft_work_bad_women",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Full-time; Good Health",
            "disutil_ft_work_good_men",
            "disutil_ft_work_good_women",
        )
    )

    # Part-time (women only)
    disutil_rows.append(
        format_param_row(
            "Part-time; Bad Health",
            None,  # No men parameter
            "disutil_pt_work_bad_women",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Part-time; Good Health",
            None,  # No men parameter
            "disutil_pt_work_good_women",
        )
    )

    # Children effects (women only)
    disutil_rows.append(
        format_param_row(
            "Children; Part-time; Low Education",
            None,  # No men parameter
            "disutil_children_pt_work_low",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Children; Part-time; High Education",
            None,  # No men parameter
            "disutil_children_pt_work_high",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Children; Full-time; Low Education",
            None,  # No men parameter
            "disutil_children_ft_work_low",
        )
    )

    disutil_rows.append(
        format_param_row(
            "Children; Full-time; High Education",
            None,  # No men parameter
            "disutil_children_ft_work_high",
        )
    )

    # Partner retired
    disutil_rows.append(
        format_param_row(
            "Partner Retired",
            "disutil_partner_retired_men",
            "disutil_partner_retired_women",
        )
    )

    # Taste shock scale
    disutil_rows.append(
        format_param_row(
            "Taste Shock Scale", "taste_shock_scale_men", "taste_shock_scale_women"
        )
    )

    disutil_content = "\n".join(disutil_rows)
    disutil_table = create_table_wrapper(disutil_content)
    disutil_filename = out_dir + f"table_disutility_{model_name}.tex"

    with open(disutil_filename, "w") as f:
        f.write(disutil_table)

    print(f"Disutility table saved to: {disutil_filename}")

    # ========================================================================
    # TABLE 2: JOB OFFER PROBABILITY (LOGIT)
    # ========================================================================
    job_rows = []

    job_rows.append(
        format_param_row(
            "Constant", "job_finding_logit_const_men", "job_finding_logit_const_women"
        )
    )

    job_rows.append(
        format_param_row(
            "High Education",
            "job_finding_logit_high_educ_men",
            "job_finding_logit_high_educ_women",
        )
    )

    job_rows.append(
        format_param_row(
            "Age Above 50",
            "job_finding_logit_above_50_men",
            "job_finding_logit_above_50_women",
        )
    )

    job_rows.append(
        format_param_row(
            "Age Above 55",
            "job_finding_logit_above_55_men",
            "job_finding_logit_above_55_women",
        )
    )

    job_rows.append(
        format_param_row(
            "Age Above 60",
            "job_finding_logit_above_60_men",
            "job_finding_logit_above_60_women",
        )
    )

    job_content = "\n".join(job_rows)
    job_table = create_table_wrapper(job_content)
    job_filename = f"{out_dir}table_job_offer_{model_name}.tex"

    with open(job_filename, "w") as f:
        f.write(job_table)

    print(f"Job offer table saved to: {job_filename}")

    # ========================================================================
    # TABLE 3: DISABILITY PROBABILITY (LOGIT)
    # ========================================================================
    disability_rows = []

    disability_rows.append(
        format_param_row(
            "Constant", "disability_logit_const_men", "disability_logit_const_women"
        )
    )

    disability_rows.append(
        format_param_row(
            "Age Above 50",
            "disability_logit_above_50_men",
            "disability_logit_above_50_women",
        )
    )

    disability_rows.append(
        format_param_row(
            "Age Above 55",
            "disability_logit_above_55_men",
            "disability_logit_above_55_women",
        )
    )

    disability_rows.append(
        format_param_row(
            "Age Above 60",
            "disability_logit_above_60_men",
            "disability_logit_above_60_women",
        )
    )

    disability_content = "\n".join(disability_rows)
    disability_table = create_table_wrapper(disability_content)
    disability_filename = f"{out_dir}table_disability_{model_name}.tex"

    with open(disability_filename, "w") as f:
        f.write(disability_table)

    print(f"Disability table saved to: {disability_filename}")

    # Return dictionary of filenames
    return {
        "disutility": disutil_filename,
        "job_offer": job_filename,
        "disability": disability_filename,
    }
