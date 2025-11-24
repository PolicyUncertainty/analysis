"""
Create a LaTeX tabular from number of children estimation results.
Reports parameters for all sex x education x partner status combinations.
"""

import pandas as pd


def create_nb_children_params_latex_table(paths_dict, specs):
    """
    Load number of children parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_results' and 'first_step_tables'
    specs : dict
        Dictionary containing model specifications
    """
    # Load the parameters
    params = pd.read_csv(
        paths_dict["first_step_results"] + "nb_children_estimates.csv",
        index_col=[0, 1, 2],
    )

    # Define the parameters we want to display
    param_names = ["const", "period", "period_sq"]

    param_labels = {
        "const": "Constant",
        "period": "Period",
        "period_sq": "Period Squared",
    }

    # Get unique levels
    sex_levels = params.index.get_level_values("sex").unique()
    edu_levels = params.index.get_level_values("education").unique()
    partner_levels = params.index.get_level_values("has_partner").unique()

    # Get education labels from specs
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    partner_labels = ["No Partner", "Has Partner"]

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - 2 columns per education level (no partner, has partner)
    n_cols = len(edu_levels) * len(partner_levels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # Create separate tables for each sex
    for sex_idx, sex_val in enumerate(sex_levels):
        if sex_idx > 0:
            # Add spacing between sex groups
            latex_lines.append(r"    \midrule")

        # Add sex header
        latex_lines.append(
            f"    \\multicolumn{{{len(edu_levels) * len(partner_levels) + 1}}}{{c}}{{{sex_labels[sex_val]}}} \\\\"
        )
        latex_lines.append(r"    \midrule")

        # First row: Education levels (with multicolumn)
        header1 = "    "
        for edu_val in edu_levels:
            header1 += f" & \\multicolumn{{{len(partner_levels)}}}{{c}}{{{edu_labels[edu_val]}}}"
        header1 += r" \\"
        latex_lines.append(header1)

        # Second row: Partner status
        header2 = "    Parameter"
        for edu_val in edu_levels:
            for partner_val in partner_levels:
                header2 += f" & {partner_labels[partner_val]}"
        header2 += r" \\"
        latex_lines.append(header2)
        latex_lines.append(r"    \midrule")

        # Add parameter rows for this sex
        for param in param_names:
            row = f"    {param_labels[param]}"

            for edu_val in edu_levels:
                for partner_val in partner_levels:
                    try:
                        coef = params.loc[(sex_val, edu_val, partner_val), param]
                        se = params.loc[(sex_val, edu_val, partner_val), param + "_ser"]

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
            for edu_val in edu_levels:
                for partner_val in partner_levels:
                    try:
                        se = params.loc[(sex_val, edu_val, partner_val), param + "_ser"]
                        if pd.notna(se):
                            row_se += f" & ({se:.3f})"
                        else:
                            row_se += " & "
                    except KeyError:
                        row_se += " & "

            row_se += r" \\"
            latex_lines.append(row_se)

    # Close the tabular
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"\end{tabular}")

    # Join all lines
    latex_table = "\n".join(latex_lines)

    output_path = paths_dict["first_step_tables"] + "nb_children_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
