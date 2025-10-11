import os

import pandas as pd


def sra_increase_table(path_dict, model_name):
    """Generate LaTeX table comparing SRA 67 vs 69 under uncertainty scenarios."""

    het_labels = ["men", "women", "overall"]
    # Load results
    result_dfs = {}
    for scenario in ["unc", "no_unc"]:
        result_dfs[scenario] = {}
        for het in het_labels:
            filename = f"sra_increase_aggregate_{scenario}_{het}_{model_name}.csv"
            result_dfs[scenario][het] = pd.read_csv(
                path_dict["sim_results"] + filename, index_col=0
            )

    metrics = {
        "working_hours_below_63": "Annual Labor Supply (hrs)",
        "consumption_below_63": "Annual Consumption",
        "savings_below_63": "Annual Savings",
        "sra_at_63": "SRA at 63",
        "ret_age": "Retirement Age",
        "ret_age_excl_disabled": "Retirement Age (excl. Disability)",
        "pension_wealth_at_ret": "Pension Wealth (PV at Retirement)",
        "pension_wealth_at_ret_excl_disability": "Pension Wealth (excl. Disability)",
        "private_wealth_at_ret": "Financial Wealth at Retirement",
        "private_wealth_at_ret_excl_disability": "Financial Wealth (excl. Disability)",
        "pensions": "Annual Pension Income",
        "pensions_excl_disability": "Annual Pension Income (excl. Disability)",
        "share_disability_pensions": "Share with Disability Pension",
        "pensions_share_below_63": "Share with Pension before 63",
        "lifecycle_working_hours": "Annual Labor Supply (hrs)",
        "lifecycle_avg_wealth": "Average Financial Wealth",
        "cv": "Compensated Variation (\\%)",
    }

    sections = {
        "Retirement": [
            "ret_age",
            "ret_age_excl_disabled",
            "pension_wealth_at_ret",
            "pension_wealth_at_ret_excl_disability",
            "private_wealth_at_ret",
            "private_wealth_at_ret_excl_disability",
            "pensions",
            "pensions_excl_disability",
            "share_disability_pensions",
            "pensions_share_below_63",
        ],
        "Work Life ($<63$)": [
            "working_hours_below_63",
            "consumption_below_63",
            "savings_below_63",
        ],
        "Lifecycle (30+)": ["lifecycle_working_hours", "lifecycle_avg_wealth"],
    }

    for het_label in het_labels:

        latex_lines = []
        latex_lines.append("\\begin{tabular}{lcccc}")
        latex_lines.append("    \\toprule")
        latex_lines.append(
            "    \\multirow{2}{*}{Expected Outcome} & "
            "\\multicolumn{2}{c}{SRA 67} & "
            "\\multicolumn{2}{c}{SRA 68.25} \\\\"
        )
        latex_lines.append("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
        latex_lines.append("     & No Unc. & Unc. & No Unc. & Unc. \\\\")
        latex_lines.append("     & (1) & (2) & (3) & (4) \\\\")
        latex_lines.append("    \\midrule")

        for section_name, section_metrics in sections.items():
            latex_lines.append("    \\midrule")
            latex_lines.append(
                f"    \\multicolumn{{5}}{{l}}{{\\textit{{{section_name}}}}} \\\\"
            )

            for metric_key in section_metrics:
                outcome_name = metrics[metric_key]
                row_data = [outcome_name]

                unc_df = result_dfs["unc"][het_label]
                no_unc_df = result_dfs["no_unc"][het_label]
                mask_unc = unc_df["sra_at_63"] == 68.25
                mask_no_unc = no_unc_df["sra_at_63"] == 68.25

                # No Uncertainty SRA 67 (base_)
                row_data.append(
                    f"{no_unc_df.loc[mask_no_unc, f'base_{metric_key}'].values[0]:.2f}"
                )
                # Uncertainty SRA 67 (base_)
                row_data.append(
                    f"{unc_df.loc[mask_unc, f'base_{metric_key}'].values[0]:.2f}"
                )
                # No Uncertainty SRA 68.25 (cf_)
                row_data.append(
                    f"{no_unc_df.loc[mask_no_unc, f'cf_{metric_key}'].values[0]:.2f}"
                )
                # Uncertainty SRA 68.25 (cf_)
                row_data.append(
                    f"{unc_df.loc[mask_unc, f'cf_{metric_key}'].values[0]:.2f}"
                )

                latex_lines.append("    " + " & ".join(row_data) + " \\\\")

        latex_lines.append("    \\bottomrule")
        latex_lines.append("\\end{tabular}")

        latex_table = "\n".join(latex_lines)

        # Save to file
        table_dir = path_dict["simulation_tables"] + model_name + "/"
        if not os.path.exists(table_dir):
            os.makedirs(table_dir)

        output_path = os.path.join(table_dir, f"sra_increase_baseline_{het_label}.tex")

        with open(output_path, "w") as f:
            f.write(latex_table)

        print(f"Table saved to: {output_path}")
