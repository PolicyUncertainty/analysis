"""
Create a LaTeX tabular from health transition estimation results.
Reports parameters for all sex x education x health state combinations.
"""

import pandas as pd


def create_health_transition_params_latex_table(paths_dict, specs):
    """
    Load health transition parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_results' and 'first_step_tables'
    specs : dict
        Dictionary containing model specifications
    """
    # Load the parameters
    params = pd.read_csv(
        paths_dict["first_step_results"] + "health_transition_params.csv",
        index_col=[0, 1, 2],
    )

    # Define the parameters we want to display
    param_names = ["const", "age"]

    param_labels = {
        "const": "Constant",
        "age": "Age",
    }

    # Get unique levels
    sex_levels = params.index.get_level_values("sex").unique()
    edu_levels = params.index.get_level_values("education").unique()
    health_levels = params.index.get_level_values("health").unique()

    # Get labels from specs
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    observed_health_labels = specs["observed_health_labels"]

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - 4 columns (2 education x 2 health states)
    n_cols = len(edu_levels) * len(health_levels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # First row: Education levels (only once at the top)
    header1 = "    "
    for edu_label in edu_labels:
        header1 += f" & \\multicolumn{{{len(health_levels)}}}{{c}}{{{edu_label}}}"
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

    # Create sections for each sex
    for sex_idx, sex_label in enumerate(sex_labels):
        # Add sex label row
        latex_lines.append(f"    {sex_label} & & & & \\\\")
        latex_lines.append(r"    \midrule")

        # Add parameter rows for this sex
        for param in param_names:
            row = f"    {param_labels[param]}"

            for edu_label in edu_labels:
                for health_label in observed_health_labels:
                    try:
                        coef = params.loc[(sex_label, edu_label, health_label), param]
                        se = params.loc[
                            (sex_label, edu_label, health_label), param + "_ser"
                        ]

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
            for edu_label in edu_labels:
                for health_label in observed_health_labels:
                    try:
                        se = params.loc[
                            (sex_label, edu_label, health_label), param + "_ser"
                        ]
                        if pd.notna(se):
                            row_se += f" & ({se:.3f})"
                        else:
                            row_se += " & "
                    except KeyError:
                        row_se += " & "

            row_se += r" \\"
            latex_lines.append(row_se)

        # Add midrule after each sex section (except after the last one)
        if sex_idx < len(sex_labels) - 1:
            latex_lines.append(r"    \midrule")

    # Close the tabular
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Join all lines
    latex_table = "\n".join(latex_lines)

    output_path = paths_dict["first_step_tables"] + "health_transition_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
