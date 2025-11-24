"""
Create a LaTeX tabular from partner wage equation estimation results.
Reports parameters for all sex x education combinations.
"""

import pandas as pd


def create_partner_wage_params_latex_table(paths_dict):
    """
    Load partner wage equation parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_incomes' and 'first_step_tables'
    """
    # Load the parameters for both men and women
    params_men = pd.read_csv(
        paths_dict["first_step_incomes"] + "partner_wage_eq_params_men.csv",
        index_col=0,
    )
    params_women = pd.read_csv(
        paths_dict["first_step_incomes"] + "partner_wage_eq_params_women.csv",
        index_col=0,
    )

    # Define the parameters we want to display
    param_names = ["constant", "period", "period_sq", "period_cub"]

    param_labels = {
        "constant": "Constant",
        "period": "Period",
        "period_sq": "Period Squared",
        "period_cub": "Period Cubed",
    }

    # Get unique education levels
    edu_levels = params_men.index.tolist()

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - 2 columns per education level (men and women)
    n_cols = len(edu_levels) * 2
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # Create header rows
    # First row: Education levels (with multicolumn for men and women)
    header1 = "    "
    for edu in edu_levels:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{edu}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Second row: Sex levels
    header2 = "    Parameter"
    for edu in edu_levels:
        header2 += " & Men & Women"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"    \midrule")

    # Add parameter rows
    for param in param_names:
        row = f"    {param_labels[param]}"

        for edu in edu_levels:
            # Men
            try:
                coef_men = params_men.loc[edu, param]
                se_men = params_men.loc[edu, param + "_ser"]

                if pd.notna(coef_men) and pd.notna(se_men):
                    row += f" & {coef_men:.3f}"
                else:
                    row += " & ---"
            except KeyError:
                row += " & ---"

            # Women (no cubic term)
            if param == "period_cub":
                row += " & ---"
            else:
                try:
                    coef_women = params_women.loc[edu, param]
                    se_women = params_women.loc[edu, param + "_ser"]

                    if pd.notna(coef_women) and pd.notna(se_women):
                        row += f" & {coef_women:.3f}"
                    else:
                        row += " & ---"
                except KeyError:
                    row += " & ---"

        row += r" \\"
        latex_lines.append(row)

        # Add standard errors in the next row
        row_se = "    "
        for edu in edu_levels:
            # Men
            try:
                se_men = params_men.loc[edu, param + "_ser"]
                if pd.notna(se_men):
                    row_se += f" & ({se_men:.3f})"
                else:
                    row_se += " & "
            except KeyError:
                row_se += " & "

            # Women (no cubic term)
            if param == "period_cub":
                row_se += " & "
            else:
                try:
                    se_women = params_women.loc[edu, param + "_ser"]
                    if pd.notna(se_women):
                        row_se += f" & ({se_women:.3f})"
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

    output_path = paths_dict["first_step_tables"] + "partner_wage_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
