import matplotlib.pyplot as plt


def plot_retirement_difference(
    df_base, df_cf, final_SRA, left_difference, right_difference, base_label, cf_label
):

    df_base["SRA_diff"] = df_base["age"] - final_SRA
    df_cf["SRA_diff"] = df_cf["age"] - final_SRA

    df_base_plot = df_base[
        (df_base["SRA_diff"] >= left_difference)
        & (df_base["SRA_diff"] <= right_difference)
        & (df_base["choice"] == 0)
        & (df_base["lagged_choice"] != 0)
    ]
    df_cf_plot = df_cf[
        (df_cf["SRA_diff"] >= left_difference)
        & (df_cf["SRA_diff"] <= right_difference)
        & (df_cf["choice"] == 0)
        & (df_cf["lagged_choice"] != 0)
    ]

    inflow_shares_base = (
        df_base_plot["SRA_diff"].value_counts(normalize=True).sort_index()
    )
    inflow_shares_cf = df_cf_plot["SRA_diff"].value_counts(normalize=True).sort_index()

    # Make barplot with SRA diff on x-axis and inflow shares on y-axis
    fig, ax = plt.figure(figsize=(10, 6))
    ax.bar(
        inflow_shares_base.index - 0.1,
        inflow_shares_base.values,
        width=0.2,
        label=base_label,
    )
    ax.bar(
        inflow_shares_cf.index + 0.1, inflow_shares_cf.values, width=0.2, label=cf_label
    )
    ax.legend()
    ax.set_xlabel("Age - SRA")
    ax.set_ylabel("Inflow into retirement share")
    ax.set_title(f"Inflow into retirement by age relative to SRA {final_SRA}")
    return fig
