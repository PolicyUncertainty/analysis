import numpy as np
import pandas as pd


def compute_bunching_shares(df, specs, ages_to_plot=None):
    """
    Reduce a simulated panel to pension-claim shares by age, retirement type,
    and sex x education group.

    This is the only aggregate the retirement bunching plots need, so the raw
    simulated dataframe never has to be written to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated panel already filtered to the claiming rows
        (choice == 0 & lagged_choice != 0).
    specs : dict
        Model specifications, used for sex/education group counts.
    ages_to_plot : array-like, optional
        Ages to report shares for. Defaults to 63-68.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns sex, education, age, retirement_type,
        share. sex == -1 and education == -1 denote the pooled (all groups)
        rows.
    """
    if ages_to_plot is None:
        ages_to_plot = np.arange(63, 69)

    df = df.copy()
    df["SRA_diff"] = df["policy_state_value"] - df["age"]

    conditions = [
        (df["SRA_diff"] > 0) & df["very_long_insured"],
        (df["SRA_diff"] > 0) & (df["health"] == 2),
    ]
    choices = ["very_long_insured", "disability"]
    df["retirement_type"] = np.select(conditions, choices, default="standard")

    groups = {(-1, -1): df}
    for sex_var in range(len(specs["sex_labels"])):
        for edu_var in range(len(specs["education_labels"])):
            mask = (df["sex"] == sex_var) & (df["education"] == edu_var)
            groups[(sex_var, edu_var)] = df[mask]

    records = []
    for (sex_var, edu_var), group_df in groups.items():
        total_count = len(group_df)
        for ret_type in ["standard", "very_long_insured", "disability"]:
            counts = group_df.loc[
                group_df["retirement_type"] == ret_type, "age"
            ].value_counts()
            shares = (
                (counts / total_count).reindex(ages_to_plot, fill_value=0)
                if total_count
                else pd.Series(0.0, index=ages_to_plot)
            )
            for age, share in shares.items():
                records.append(
                    {
                        "sex": sex_var,
                        "education": edu_var,
                        "age": age,
                        "retirement_type": ret_type,
                        "share": share,
                    }
                )

    return pd.DataFrame.from_records(records)
