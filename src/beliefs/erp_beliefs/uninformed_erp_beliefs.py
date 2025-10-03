"""
Calculate uninformed ERP belief parameters.

This script calculates the conditional average ERP beliefs for uninformed agents
by education level: E[belief_pens_deduct | informed=False, education=edu]
"""

import numpy as np
import pandas as pd


def calculate_uninformed_erp_beliefs(df, specs):
    """
    Calculate conditional average ERP beliefs for uninformed agents by education.
    
    Returns DataFrame with parameter estimates for uninformed ERP beliefs.
    """
    # Filter data: remove missing values and invalid beliefs
    df_clean = df.copy()
    df_clean = df_clean[df_clean["belief_pens_deduct"] >= 0]
    df_clean = df_clean[df_clean["education"].notna()]
    df_clean = df_clean[df_clean["informed"].notna()]
    
    results = pd.DataFrame(columns=["parameter", "type", "estimate", "std_error"])
    
    # Calculate for each education group
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        df_edu = df_clean[df_clean["education"] == edu_val]
        df_uninformed = df_edu[df_edu["informed"] == False]
        
        if len(df_uninformed) == 0:
            avg_belief = 0.0
            std_error = np.nan
        else:
            avg_belief = df_uninformed["belief_pens_deduct"].mean()
            std_error = df_uninformed["belief_pens_deduct"].sem()
        
        # Add results
        results = pd.concat([
            results,
            pd.DataFrame({
                "parameter": ["erp_uninformed_belief"],
                "type": [edu_label], 
                "estimate": [avg_belief],
                "std_error": [std_error],
            })
        ], ignore_index=True)
    
    return results