"""
Create a LaTeX tabular from wage equation estimation results.
Reports parameters for all sex x education combinations.
"""

import pandas as pd


def create_wage_params_latex_table(paths_dict):
    """
    Load wage equation parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_incomes' and 'first_step_tables'
    """
    # Load the parameters
    params = pd.read_csv(
        paths_dict["first_step_incomes"] + "wage_eq_params.csv", index_col=[0, 1, 2]
    )

    # Define the parameters we want to display (excluding IMR)
    param_names = ["constant", "ln_exp", "above_50_age", "income_shock_std"]
    param_labels = {
        "constant": "Constant",
        "ln_exp": "Return to Experience",
        "above_50_age": "Age Trend above 50",
        "income_shock_std": r"Income Shock Std.",
    }

    # Get unique education and sex levels (excluding 'all')
    edu_levels = params.index.get_level_values("education").unique()
    sex_levels = params.index.get_level_values("sex").unique()

    # Filter out 'all' category
    edu_levels = [e for e in edu_levels if e != "all"]
    sex_levels = [s for s in sex_levels if s != "all"]

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification - one column per type combination
    n_cols = len(edu_levels) * len(sex_levels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # Create header rows
    # First row: Education levels (with multicolumn)
    header1 = "    "
    for edu in edu_levels:
        header1 += f" & \\multicolumn{{{len(sex_levels)}}}{{c}}{{{edu}}}"
    header1 += r" \\"
    latex_lines.append(header1)

    # Second row: Sex levels
    header2 = "    Parameter"
    for edu in edu_levels:
        for sex in sex_levels:
            header2 += f" & {sex}"
    header2 += r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"    \midrule")

    # Add parameter rows
    for param in param_names:
        row = f"    {param_labels[param]}"

        for edu in edu_levels:
            for sex in sex_levels:
                try:
                    # Get coefficient value
                    coef = params.loc[(edu, sex, param), "value"]
                    # Get standard error
                    se = params.loc[(edu, sex, param + "_ser"), "value"]

                    # Format the cell with coefficient and standard error
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
        for edu in edu_levels:
            for sex in sex_levels:
                try:
                    se = params.loc[(edu, sex, param + "_ser"), "value"]
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

    output_path = paths_dict["first_step_tables"] + "wage_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
