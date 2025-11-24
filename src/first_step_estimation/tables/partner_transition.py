"""
Create a LaTeX tabular from partner transition estimation results.
Reports parameters for all sex x education combinations.
"""

import pickle as pkl

import pandas as pd


def create_partner_transition_params_latex_table(paths_dict, specs):
    """
    Load partner transition parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_results' and 'first_step_tables'
    specs : dict
        Dictionary containing model specifications
    """

    # Define parameter labels
    param_labels = {
        "const_single_to_working_age": "Constant",
        "age_single_to_working_age": "Age",
        "age_squared_single_to_working_age": "Age Squared",
        "age_cubic_single_to_working_age": "Age Cubed",
        "const_working_age_to_working_age": "Constant",
        "age_working_age_to_working_age": "Age",
        "age_squared_working_age_to_working_age": "Age Squared",
        "age_cubic_working_age_to_working_age": "Age Cubed",
        "const_working_age_to_retirement": "Constant",
        "age_working_age_to_retirement": "Age",
        "age_squared_working_age_to_retirement": "Age Squared",
        "age_cubic_working_age_to_retirement": "Age Cubed",
        "SRA_age_diff_effect_working_age_to_retirement": "SRA Age Difference",
    }

    # Group parameters by transition type
    param_groups = {
        "Single to Working Age": [
            "const_single_to_working_age",
            "age_single_to_working_age",
            "age_squared_single_to_working_age",
            "age_cubic_single_to_working_age",
        ],
        "Working Age to Working Age": [
            "const_working_age_to_working_age",
            "age_working_age_to_working_age",
            "age_squared_working_age_to_working_age",
            "age_cubic_working_age_to_working_age",
        ],
        "Working Age to Retirement": [
            "const_working_age_to_retirement",
            "age_working_age_to_retirement",
            "age_squared_working_age_to_retirement",
            "age_cubic_working_age_to_retirement",
            "SRA_age_diff_effect_working_age_to_retirement",
        ],
    }

    # Get education and sex labels
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    # Load all parameter files
    all_params = {}
    all_se = {}

    for sex_var, sex_label in enumerate(sex_labels):
        for edu_var, edu_label in enumerate(edu_labels):
            # Load parameters
            params = pkl.load(
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}.pkl",
                    "rb",
                )
            )
            # Load standard errors
            se = pkl.load(
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}_se.pkl",
                    "rb",
                )
            )
            all_params[(sex_label, edu_label)] = params
            all_se[(sex_label, edu_label)] = se

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - 4 columns (2 sex x 2 education)
    n_cols = len(sex_labels) * len(edu_labels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # First row: Sex levels
    header1 = "    "
    for sex_label in sex_labels:
        header1 += f" & \\multicolumn{{{len(edu_labels)}}}{{c}}{{{sex_label}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Add cline under sex headers
    cline = "    "
    col_start = 2
    for sex_label in sex_labels:
        col_end = col_start + len(edu_labels) - 1
        cline += f"\\cline{{{col_start}-{col_end}}} "
        col_start = col_end + 1
    latex_lines.append(cline)

    # Second row: Education levels
    header2 = "    Parameter"
    for sex_label in sex_labels:
        for edu_label in edu_labels:
            header2 += f" & {edu_label}"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"    \midrule")

    # Add parameter rows grouped by transition type
    for group_name, param_names in param_groups.items():
        # Add group header
        latex_lines.append(
            f"    \\multicolumn{{{n_cols + 1}}}{{l}}{{{group_name}}} \\\\"
        )
        latex_lines.append(r"    \midrule")

        for param in param_names:
            row = f"    {param_labels[param]}"

            for sex_label in sex_labels:
                for edu_label in edu_labels:
                    try:
                        coef = all_params[(sex_label, edu_label)][param]
                        se = all_se[(sex_label, edu_label)][param]

                        if pd.notna(coef) and pd.notna(se):
                            row += f" & {coef:.3f}"
                        else:
                            row += " & ---"
                    except KeyError:
                        row += " & ---"

            row += r" \\"
            latex_lines.append(row)

            # Add standard errors in the next row
            row_se = "    "
            for sex_label in sex_labels:
                for edu_label in edu_labels:
                    try:
                        se = all_se[(sex_label, edu_label)][param]
                        if pd.notna(se):
                            row_se += f" & ({se:.3f})"
                        else:
                            row_se += " & "
                    except KeyError:
                        row_se += " & "

            row_se += r" \\"
            latex_lines.append(row_se)

        # Add spacing after each group except the last
        if group_name != list(param_groups.keys())[-1]:
            latex_lines.append(r"    \midrule")

    # Close the tabular
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Join all lines
    latex_table = "\n".join(latex_lines)

    output_path = (
        paths_dict["first_step_tables"] + "partner_transition_params_table.tex"
    )
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
