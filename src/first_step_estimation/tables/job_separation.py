"""
Create a LaTeX tabular from job separation estimation results.
Reports parameters for both sexes.
"""

import pandas as pd


def create_job_sep_params_latex_table(paths_dict):
    """
    Load job separation parameters and create a formatted LaTeX tabular.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths, including 'first_step_results' and 'first_step_tables'
    """
    # Load the parameters
    params = pd.read_csv(
        paths_dict["first_step_results"] + "job_sep_params.csv",
        index_col=0,
    )

    # Define the parameters we want to display
    param_names = [
        "const",
        "high_educ",
        "good_health",
        "above_50",
        "above_55",
        "above_60",
    ]

    param_labels = {
        "const": "Constant",
        "high_educ": "High Education",
        "good_health": "Good Health",
        "above_50": "Age $\\geq$ 50",
        "above_55": "Age $\\geq$ 55",
        "above_60": "Age $\\geq$ 60",
    }

    # Get sex levels
    sex_levels = params.index.tolist()

    # Start building the LaTeX tabular
    latex_lines = []

    # Create column specification
    n_cols = len(sex_levels)
    col_spec = "l" + "c" * n_cols
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"    \toprule")

    # Create header row
    header = "    Parameter"
    for sex in sex_levels:
        header += f" & {sex}"
    header += r" \\"
    latex_lines.append(header)
    latex_lines.append(r"    \midrule")

    # Add parameter rows
    for param in param_names:
        row = f"    {param_labels[param]}"

        for sex in sex_levels:
            try:
                coef = params.loc[sex, param]
                se = params.loc[sex, param + "_ser"]

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
        for sex in sex_levels:
            try:
                se = params.loc[sex, param + "_ser"]
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

    output_path = paths_dict["first_step_tables"] + "job_sep_params_table.tex"
    # Save to file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX tabular saved to: {output_path}")

    return latex_table
