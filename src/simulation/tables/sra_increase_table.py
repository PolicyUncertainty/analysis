import os

import pandas as pd


def sra_increase_table(path_dict, model_name):
    """Generate LaTeX table comparing SRA 67 vs 69 under uncertainty scenarios."""
    het_labels = ["overall"]

    # Load results
    result_dfs = {}
    for scenario in ["unc", "no_unc"]:
        result_dfs[scenario] = {}
        for het in het_labels:
            filename = f"sra_increase_aggregate_{scenario}_{het}.csv"
            df = pd.read_csv(
                path_dict["sim_results"] + model_name + "/" + filename, index_col=0
            )
            if scenario == "unc":
                df["base_exp_sra_at_63"] = 68.32
                df["cf_exp_sra_at_63"] = 68.32
            else:
                df["base_exp_sra_at_63"] = df["base_sra_at_63"]
                df["cf_exp_sra_at_63"] = df["cf_sra_at_63"]
            result_dfs[scenario][het] = df

    metrics = {
        "working_hours_below_63": "Annual Labor Supply (hrs)",
        "consumption_below_63": "Annual Consumption",
        "savings_below_63": "Annual Savings",
        "sra_at_63": "True SRA",
        "exp_sra_at_63": "Expected SRA",
        # "ret_age": "Retirement Age",
        "ret_age_excl_disabled": "Retirement Age",
        # "pension_wealth_at_ret": "Pension Wealth (PV at Retirement)",
        "pension_wealth_at_ret_excl_disability": "Pension Wealth",
        # "private_wealth_at_ret": "Financial Wealth at Retirement",
        "private_wealth_at_ret_excl_disability": "Financial Wealth",
        # "pensions": "Annual Pension Income",
        "pensions_excl_disability": "Annual Pension Income",
        "share_disability_pensions": "Share with Disability Pension",
        "pensions_share_below_63": "Share with Pension before 63",
        "share_below_SRA": "Share Retiring below SRA",
        "share_very_long_insured": "Share with Very Long Insured",
        "lifecycle_working_hours": "Annual Labor Supply (hrs)",
        "lifecycle_avg_wealth": "Average Financial Wealth",
    }

    sections = {
        "SRA": [
            "sra_at_63",
            "exp_sra_at_63",
        ],
        "Retirement": [
            # "ret_age",
            "ret_age_excl_disabled",
            # "pension_wealth_at_ret",
            "pension_wealth_at_ret_excl_disability",
            # "private_wealth_at_ret",
            "private_wealth_at_ret_excl_disability",
            # "pensions",
            # "pensions_excl_disability",
            # "share_disability_pensions",
            # "pensions_share_below_63",
            # "share_below_SRA",
            # "share_very_long_insured",
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
        latex_lines.append("\\begin{tabular}{lcccccc}")
        latex_lines.append("    \\toprule")
        latex_lines.append(
            "    \\multirow{2}{*}{Outcome} & "
            "\\multicolumn{3}{c}{Uncertainty} & "
            "\\multicolumn{3}{c}{No Uncertainty} \\\\"
        )
        latex_lines.append("    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
        latex_lines.append(
            "     & SRA 67 & SRA 69 & Diff. \\% & SRA 67 & SRA 69 & Diff. \\% \\\\"
        )
        latex_lines.append("     & (1) & (2) & (3) & (4) & (5) & (6) \\\\")
        latex_lines.append("    \\midrule")

        for section_name, section_metrics in sections.items():
            latex_lines.append("    \\midrule")
            latex_lines.append(
                f"    \\multicolumn{{7}}{{l}}{{\\textit{{{section_name}}}}} \\\\"
            )

            for metric_key in section_metrics:
                outcome_name = metrics[metric_key]
                row_data = [outcome_name]

                unc_df = result_dfs["unc"][het_label]
                no_unc_df = result_dfs["no_unc"][het_label]
                mask_unc = unc_df["sra_at_63"] == 69
                mask_no_unc = no_unc_df["sra_at_63"] == 69

                # Uncertainty SRA 67 (base_)
                unc_base = unc_df.loc[mask_unc, f"base_{metric_key}"].values[0]
                row_data.append(f"{unc_base:.2f}")

                # Uncertainty SRA 69 (cf_)
                unc_cf = unc_df.loc[mask_unc, f"cf_{metric_key}"].values[0]
                row_data.append(f"{unc_cf:.2f}")

                # Uncertainty Diff % (skip for SRA section)
                if section_name == "SRA":
                    row_data.append("")
                else:
                    unc_diff_pct = (
                        ((unc_cf - unc_base) / unc_base) * 100 if unc_base != 0 else 0
                    )
                    row_data.append(f"{unc_diff_pct:+.2f}")

                # No Uncertainty SRA 67 (base_)
                no_unc_base = no_unc_df.loc[mask_no_unc, f"base_{metric_key}"].values[0]
                row_data.append(f"{no_unc_base:.2f}")

                # No Uncertainty SRA 69 (cf_)
                no_unc_cf = no_unc_df.loc[mask_no_unc, f"cf_{metric_key}"].values[0]
                row_data.append(f"{no_unc_cf:.2f}")

                # No Uncertainty Diff % (skip for SRA section)
                if section_name == "SRA":
                    row_data.append("")
                else:
                    no_unc_diff_pct = (
                        ((no_unc_cf - no_unc_base) / no_unc_base) * 100
                        if no_unc_base != 0
                        else 0
                    )
                    row_data.append(f"{no_unc_diff_pct:+.2f}")

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


def welfare_table(path_dict, model_name):
    """Generate LaTeX table with welfare (CV) for SRA 67-70 under uncertainty scenarios."""

    het_label = "overall"

    # Load results
    result_dfs = {}
    for scenario in ["unc", "no_unc"]:
        filename = f"sra_increase_aggregate_{scenario}_{het_label}.csv"
        result_dfs[scenario] = pd.read_csv(
            path_dict["sim_results"] + model_name + "/" + filename, index_col=0
        )

    # SRA levels
    sra_levels = [67, 68, 69, 70]

    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append("    Scenario & SRA 67 & SRA 68 & SRA 69 & SRA 70 \\\\")
    latex_lines.append("    \\midrule")

    # No Uncertainty row
    row_data = ["No Uncertainty"]
    no_unc_df = result_dfs["no_unc"]
    # Reformat nans to 0
    no_unc_df = no_unc_df.fillna(0)
    for sra in sra_levels:
        mask = no_unc_df["sra_at_63"] == sra
        cv_value = no_unc_df.loc[mask, "cv"].values[0]
        row_data.append(f"{cv_value:.2f}")
    latex_lines.append("    " + " & ".join(row_data) + " \\\\")

    # Uncertainty row
    row_data = ["Uncertainty"]
    unc_df = result_dfs["unc"]
    # Reformat nans to 0
    unc_df = unc_df.fillna(0)
    for sra in sra_levels:
        mask = unc_df["sra_at_63"] == sra
        cv_value = unc_df.loc[mask, "cv"].values[0]
        row_data.append(f"{cv_value:.2f}")
    latex_lines.append("    " + " & ".join(row_data) + " \\\\")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("\\end{tabular}")

    latex_table = "\n".join(latex_lines)

    # Save to file
    table_dir = path_dict["simulation_tables"] + model_name + "/"
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    output_path = os.path.join(table_dir, "welfare_table_overall.tex")

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"Welfare table saved to: {output_path}")


def welfare_table_2(path_dict, model_name):
    """Generate LaTeX table with welfare (CV) for SRA 67-70 under uncertainty scenarios."""

    # Load results for all heterogeneity groups
    result_dfs = {}
    het_labels = [
        "overall",
        "initial_informed",
        "initial_uninformed",
        "low_men",
        "high_men",
        "low_women",
        "high_women",
    ]

    for het_label in het_labels:
        for scenario in ["unc", "no_unc"]:
            filename = f"sra_increase_aggregate_{scenario}_{het_label}.csv"
            key = f"{scenario}_{het_label}"
            result_dfs[key] = pd.read_csv(
                path_dict["sim_results"] + model_name + "/" + filename, index_col=0
            ).fillna(0)

    # SRA levels
    sra_levels = [67, 68, 69, 70]

    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append("    Group & SRA 67 & SRA 68 & SRA 69 & SRA 70 \\\\")
    latex_lines.append("    \\midrule")

    # Benchmark: Uncertainty (overall)
    latex_lines.append("    \\multicolumn{5}{l}{\\textit{Whole Sample}} \\\\")
    row_data_unc = ["Benchmark: Uncertainty"]
    row_data_no_unc = ["No Uncertainty"]
    unc_df = result_dfs["unc_overall"]
    now_unc_df = result_dfs["no_unc_overall"]
    for sra in sra_levels:
        cv_value_unc = unc_df.loc[unc_df["sra_at_63"] == sra, "cv"].values[0]
        row_data_unc.append(f"{cv_value_unc:.2f}")
        cv_value_no_unc = now_unc_df.loc[now_unc_df["sra_at_63"] == sra, "cv"].values[0]
        row_data_no_unc.append(f"{cv_value_no_unc:.2f}")
    latex_lines.append("    " + " & ".join(row_data_unc) + " \\\\")
    latex_lines.append("    " + " & ".join(row_data_no_unc) + " \\\\")
    latex_lines.append("    \\midrule")

    # Information heterogeneity (under uncertainty)
    latex_lines.append("    \\multicolumn{5}{l}{\\textit{By Initial Information}} \\\\")

    for het_label, het_name in [
        ("initial_informed", "Initially Informed"),
        ("initial_uninformed", "Initially Misinformed"),
    ]:
        row_data = [het_name]
        het_df = result_dfs[f"unc_{het_label}"]
        for sra in sra_levels:
            mask = het_df["sra_at_63"] == sra
            cv_value = het_df.loc[mask, "cv"].values[0]
            row_data.append(f"{cv_value:.2f}")
        latex_lines.append("    " + " & ".join(row_data) + " \\\\")

    latex_lines.append("    \\midrule")

    # Education-gender types (under uncertainty)
    latex_lines.append("    \\multicolumn{5}{l}{\\textit{By Type}} \\\\")

    type_labels = [
        ("low_men", "Low Edu Men"),
        ("high_men", "High Edu Men"),
        ("low_women", "Low Edu Women"),
        ("high_women", "High Edu Women"),
    ]

    for het_label, het_name in type_labels:
        row_data = [het_name]
        het_df = result_dfs[f"unc_{het_label}"]
        for sra in sra_levels:
            mask = het_df["sra_at_63"] == sra
            cv_value = het_df.loc[mask, "cv"].values[0]
            row_data.append(f"{cv_value:.2f}")
        latex_lines.append("    " + " & ".join(row_data) + " \\\\")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("\\end{tabular}")

    latex_table = "\n".join(latex_lines)

    # Save to file
    table_dir = path_dict["simulation_tables"] + model_name + "/"
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    output_path = os.path.join(table_dir, "welfare_table_extensive.tex")

    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"Extended welfare table saved to: {output_path}")

    return latex_table
