import pandas as pd
from set_paths import create_path_dict


def create_sra_params_table_latex(path_dict, save=False):
    """
    Create a LaTeX table for SRA belief parameters from beliefs_parameters.csv.
    
    This function reads the estimated SRA belief parameters (alpha and sigma_sq)
    and formats them into a LaTeX table matching Table 1 from Chapter 3.
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
        LaTeX tabular environment for SRA belief parameters
    """
    
    # Load the beliefs parameters
    beliefs_params_df = pd.read_csv(
        path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
    )
    
    # Filter for SRA parameters and sort them (alpha first, then sigma_sq)
    sra_params = beliefs_params_df[
        beliefs_params_df["parameter"].isin(["alpha", "sigma_sq"])
    ].copy()
    
    # Sort to ensure alpha comes first, then sigma_sq
    param_order = {"alpha": 0, "sigma_sq": 1}
    sra_params = sra_params.sort_values(
        by="parameter", key=lambda x: x.map(param_order)
    )
    
    # Check that we have both parameters
    if len(sra_params) != 2:
        raise ValueError(
            f"Expected 2 SRA parameters (alpha, sigma_sq), found {len(sra_params)}"
        )
    
    # Create parameter mapping for display names and LaTeX symbols
    param_mapping = {
        "alpha": {
            "name": "Drift",
            "symbol": r"$\alpha$"
        },
        "sigma_sq": {
            "name": "Variance of belief process", 
            "symbol": r"$\sigma_{SRA}^2$"
        }
    }
    
    # Start LaTeX table
    latex_table = (
        r"\begin{tabular}{llc}" + "\n" +
        r"\toprule" + "\n" +
        r" Parameter Name & Parameter & Estimate \\" + "\n" +
        r"\midrule" + "\n"
    )
    
    # Add parameter rows
    for _, row in sra_params.iterrows():
        param_name = row["parameter"]
        estimate = row["estimate"] 
        std_error = row["std_error"]
        
        param_info = param_mapping.get(param_name)
        if param_info:
            display_name = param_info["name"]
            symbol = param_info["symbol"]
        else:
            display_name = param_name
            symbol = param_name
        
        latex_table += f"  {display_name} & {symbol} & {estimate:.3f} \\\\\n"
        latex_table += f"    {{}} & {{}} & ({std_error:.4f}) \\\\\n"
    
    # Close table
    latex_table += r"    \bottomrule" + "\n"
    latex_table += r"\end{tabular}"
    
    # Save to file if requested
    if save:
        output_path = path_dict["beliefs_tables"] + "sra_belief_parameters.tex"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"SRA belief parameters table saved to: {output_path}")
    
    return latex_table

