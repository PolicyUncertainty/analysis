import numpy as np
import pandas as pd


def calc_life_cycle_detailed(df):
    """
    Calculate detailed life cycle statistics by age and demographic groups.

    Computes choice rates, savings rate, and average wealth by age,
    both in aggregate and split by sex, education, informed status,
    health, and partner state.
    """

    # Define grouping combinations: aggregate + each demographic split
    group_combinations = [
        (["age"], "aggregate"),
        (["age", "sex"], "sex"),
        (["age", "education"], "education"),
        (["age", "initial_informed"], "initial_informed"),
        (["age", "health"], "health"),
        (["age", "partner_state"], "partner_state"),
    ]

    results = []

    for group_cols, group_name in group_combinations:
        grouped = df.groupby(group_cols)

        # Choice rates for all 4 choices (0=retirement, 1=unemployment, 2=part-time, 3=full-time)
        choice_rates = (
            grouped["choice"].value_counts(normalize=True).unstack(fill_value=0)
        )

        # Ensure all choice columns exist (add missing ones as zeros with proper index)
        for choice in [0, 1, 2, 3]:
            if choice not in choice_rates.columns:
                choice_rates[choice] = 0

        # Rename columns to be more descriptive
        choice_rates_df = choice_rates[[0, 1, 2, 3]].copy()
        choice_rates_df.columns = [f"choice_{i}_rate" for i in [0, 1, 2, 3]]

        # Savings rate (aggregate method), average wealth, and income variables
        other_stats = pd.DataFrame(
            {
                "savings_rate": grouped["savings_dec"].sum()
                / grouped["total_income"].sum(),
                "avg_wealth": grouped["savings"].mean(),
                "consumption": grouped["consumption"].mean(),
                "gross_own_income": (
                    grouped["gross_own_income"].mean()
                    if "gross_own_income" in df.columns
                    else 0
                ),
                "net_hh_income": (
                    grouped["net_hh_income"].mean()
                    if "net_hh_income" in df.columns
                    else 0
                ),
            }
        )

        # Combine all statistics
        stats = other_stats.join(choice_rates_df)

        # Add group identifiers
        stats = stats.reset_index()
        stats["group_type"] = group_name
        stats["group_value"] = "all" if group_name == "aggregate" else stats[group_name]

        results.append(stats)

    # Combine and set multi-index
    final_df = pd.concat(results, ignore_index=True)
    return final_df.set_index(["group_type", "group_value", "age"])
