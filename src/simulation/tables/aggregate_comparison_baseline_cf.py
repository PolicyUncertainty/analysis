import os
import pandas as pd
import numpy as np


def aggregate_comparison_baseline_cf(result_df, base_label, cf_label, path_dict, model_name):
    """
    Create LaTeX table comparing baseline and counterfactual aggregate results.
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        DataFrame with columns like 'baseline_ret_age', 'cf_ret_age', etc.
    base_label : str
        Label for baseline scenario
    cf_label : str
        Label for counterfactual scenario
    path_dict : dict
        Dictionary with paths including 'simulation_tables'
    model_name : str
        Model name for file naming
    """
    
    # Define the metrics we want to display
    metrics = {
        # Work Life (<63)
        'working_hours_below_63': 'Annual Labor Supply (hrs)',
        'consumption_below_63': 'Annual Consumption',
        'savings_below_63': 'Annual Savings',
        # Retirement
        'ret_age': 'Retirement Age',
        'ret_age_excl_disabled': 'Retirement Age (excl. Disability)',
        'pension_wealth_at_ret': 'Pension Wealth (PV at Retirement)',
        'private_wealth_at_ret': 'Financial Wealth at Retirement',
        # Lifecycle (30+)
        'lifecycle_working_hours': 'Annual Labor Supply (hrs)',
        'lifecycle_avg_wealth': 'Average Financial Wealth',
    }
    
    # Section definitions
    sections = {
        'Work Life (<63)': ['working_hours_below_63', 'consumption_below_63', 'savings_below_63'],
        'Retirement': ['ret_age', 'ret_age_excl_disabled', 'pension_wealth_at_ret', 'private_wealth_at_ret'],
        'Lifecycle (30+)': ['lifecycle_working_hours', 'lifecycle_avg_wealth'],
    }
    
    # Extract values and calculate percentage differences
    table_rows = []
    
    for section_name, section_metrics in sections.items():
        # Add section header
        table_rows.append({
            'section': section_name,
            'outcome': '',
            'baseline': '',
            'cf': '',
            'diff': ''
        })
        
        # Add metrics for this section
        for metric_key in section_metrics:
            outcome_name = metrics[metric_key]
            
            baseline_val = result_df.loc[0, f'baseline_{metric_key}']
            cf_val = result_df.loc[0, f'cf_{metric_key}']
            
            # Calculate percentage difference
            if baseline_val != 0:
                pct_diff = ((cf_val - baseline_val) / baseline_val) * 100
            else:
                pct_diff = np.nan
            
            table_rows.append({
                'section': '',
                'outcome': outcome_name,
                'baseline': baseline_val,
                'cf': cf_val,
                'diff': pct_diff
            })
    
    # Create LaTeX table
    latex_lines = []
    
    # Table header
    latex_lines.append(r'  \begin{tabular}{lccc}')
    latex_lines.append(r'    \toprule')
    latex_lines.append(f'    Outcome & {base_label} & {cf_label} & Difference (\\%) \\\\')
    latex_lines.append(r'    \midrule')
    
    # Table body
    for row in table_rows:
        if row['outcome'] == '':  # Section header
            latex_lines.append(r'    \midrule')
            latex_lines.append(f"    \\multicolumn{{4}}{{l}}{{\\textit{{{row['section']}}}}} \\\\")
        else:  # Data row
            # Format numbers appropriately
            if pd.notna(row['baseline']):
                baseline_str = f"{row['baseline']:.2f}"
                cf_str = f"{row['cf']:.2f}"
                
                if pd.notna(row['diff']):
                    diff_str = f"{row['diff']:+.2f}"
                else:
                    diff_str = "---"
                
                latex_lines.append(f"    {row['outcome']} & {baseline_str} & {cf_str} & {diff_str} \\\\")
    
    # Table footer
    latex_lines.append(r'    \bottomrule')
    latex_lines.append(r'  \end{tabular}')
    
    # Join lines
    latex_table = '\n'.join(latex_lines)
    
    # Save to file
    table_dir = path_dict["simulation_tables"]
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)
    
    output_path = os.path.join(table_dir, f"cf_debias_aggregate_results_{model_name}.tex")
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"Table saved to: {output_path}")
    
    return latex_table