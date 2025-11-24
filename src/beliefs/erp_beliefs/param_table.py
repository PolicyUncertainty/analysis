import pandas as pd


def create_erp_params_table_latex(path_dict, save=False):
    """
    Create a LaTeX table for ERP belief parameters from beliefs_parameters.csv.

    This function reads the estimated ERP belief parameters (initial_informed_share,
    hazard_rate, and erp_uninformed_belief) and formats them into a LaTeX table
    with separate columns for Low and High Education.
    Only returns the tabular environment content without notes.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    save : bool, optional
        Whether to save the LaTeX table to file (default: False)

    Returns
    -------
    str
        LaTeX tabular environment for ERP belief parameters
    """

    # Load the beliefs parameters
    params = pd.read_csv(path_dict["beliefs_est_results"] + "beliefs_parameters.csv")

    # Filter for ERP parameters (exclude alpha and sigma_sq)
    erp_params = params[~params["parameter"].isin(["alpha", "sigma_sq"])].copy()

    # Define parameter order and display names
    param_order = ["hazard_rate", "erp_uninformed_belief"]
    param_names = {
        "hazard_rate": "Hazard Rate",
        "erp_uninformed_belief": "ERP Misinformed Belief",
    }
    param_symbols = {
        "hazard_rate": r"$\lambda$",
        "erp_uninformed_belief": r"$\tilde{ERP}$",
    }

    # Start LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    # Add multi-level header
    latex_lines.append(
        "Parameter Name & Parameter & \\multicolumn{2}{c}{Estimates} \\\\"
    )
    latex_lines.append(" & & Low Educ. & High Educ. \\\\")
    latex_lines.append("\\midrule")

    # Process each parameter
    for param in param_order:
        # Get Low Education values
        low_row = erp_params[
            (erp_params["parameter"] == param) & (erp_params["type"] == "Low Education")
        ]
        low_est = low_row["estimate"].values[0]
        low_se = low_row["std_error"].values[0]

        # Get High Education values
        high_row = erp_params[
            (erp_params["parameter"] == param)
            & (erp_params["type"] == "High Education")
        ]
        high_est = high_row["estimate"].values[0]
        high_se = high_row["std_error"].values[0]

        # Format estimates
        low_est_str = f"{low_est:.3f}" if pd.notna(low_est) else ""
        high_est_str = f"{high_est:.3f}" if pd.notna(high_est) else ""

        # Format standard errors
        low_se_str = f"({low_se:.3f})" if pd.notna(low_se) else ""
        high_se_str = f"({high_se:.3f})" if pd.notna(high_se) else ""

        # Add parameter name and estimates
        param_display_name = param_names.get(param, param)
        param_symbol = param_symbols.get(param, param)
        latex_lines.append(
            f"  {param_display_name} & {param_symbol} & {low_est_str} & {high_est_str} \\\\"
        )

        # Add standard errors row (only if at least one SE exists)
        if low_se_str or high_se_str:
            latex_lines.append(f"    {{}} & {{}} & {low_se_str} & {high_se_str} \\\\")

    # Close table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    # Join all lines
    latex_table = "\n".join(latex_lines)

    # Save to file if requested
    if save:
        output_path = path_dict["beliefs_tables"] + "erp_belief_parameters.tex"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"ERP belief parameters table saved to: {output_path}")

    return latex_table
