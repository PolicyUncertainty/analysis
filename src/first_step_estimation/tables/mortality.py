"""
Create a LaTeX tabular from mortality estimation results.
Reports hazard ratios and standard errors for all health x education combinations by sex.
"""

import pandas as pd


def create_mortality_params_latex_table(paths_dict, specs):
    """
    Load mortality parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_results' and 'first_step_tables'
    specs : dict
        Dictionary containing model specifications
    """
    # Load the parameters for both men and women
    params_men = pd.read_csv(
        paths_dict["first_step_results"] + "est_params_mortality_men.csv",
        index_col=0,
    )
    params_women = pd.read_csv(
        paths_dict["first_step_results"] + "est_params_mortality_women.csv",
        index_col=0,
    )

    # Get labels from specs
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    observed_health_labels = specs["observed_health_labels"]

    # Define parameter order for the table (excluding intercept and age)
    param_order = []
    for health_label in observed_health_labels:
        for edu_label in edu_labels:
            param_order.append(f"{health_label} {edu_label}")

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - 4 columns (2 education x 2 health states)
    n_cols = len(edu_labels) * len(observed_health_labels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # First row: Education levels (only once at the top)
    header1 = "    "
    for edu_label in edu_labels:
        header1 += (
            f" & \\multicolumn{{{len(observed_health_labels)}}}{{c}}{{{edu_label}}}"
        )
    header1 += r" \\"
    latex_lines.append(header1)

    # Second row: Health state (only once at the top)
    header2 = "    "
    for edu_label in edu_labels:
        for health_label in observed_health_labels:
            header2 += f" & {health_label}"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"    \midrule")

    # Add sections for each sex
    for sex_idx, (sex_label, params_sex) in enumerate(
        [(sex_labels[0], params_men), (sex_labels[1], params_women)]
    ):
        # Add sex label row
        latex_lines.append(f"    {sex_label} & & & & \\\\")
        latex_lines.append(r"    \midrule")

        # Add hazard ratio row
        row = "    Hazard Ratio"
        for health_label in observed_health_labels:
            for edu_label in edu_labels:
                param_name = f"{health_label} {edu_label}"
                try:
                    hr = params_sex.loc[param_name, "hazard_ratio"]
                    if pd.notna(hr):
                        row += f" & {hr:.3f}"
                    else:
                        row += " & ---"
                except KeyError:
                    row += " & ---"

        row += r" \\"
        latex_lines.append(row)

        # Add standard errors in the next row
        row_se = "    "
        for health_label in observed_health_labels:
            for edu_label in edu_labels:
                param_name = f"{health_label} {edu_label}"
                hr_se = params_sex.loc[param_name, "hazard_ratio_std_error"]
                if pd.notna(hr_se):
                    row_se += f" & ({hr_se:.3f})"
                else:
                    row_se += " & "

        row_se += r" \\"
        latex_lines.append(row_se)

        # Add midrule after men (before women)
        if sex_idx < len(sex_labels) - 1:
            latex_lines.append(r"    \midrule")

    # Close the tabular
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Join all lines
    latex_table = "\n".join(latex_lines)

    output_path = paths_dict["first_step_tables"] + "mortality_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
