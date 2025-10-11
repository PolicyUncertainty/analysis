import os

import pandas as pd


def generate_baseline_expectation_table_for_all_types(path_dict, specs, model_name):
    edu_append = ["low", "high"]
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            file_append = sex_label + edu_append[edu_var]
            sim_results_folder = path_dict["sim_results"] + model_name + "/"

            res_df = pd.read_csv(
                sim_results_folder + f"ex_ante_expected_margins_{file_append}.csv",
                index_col=0,
            )
            table_folder = path_dict["simulation_tables"] + model_name + "/"
            os.makedirs(table_folder, exist_ok=True)
            table = generate_baseline_expectation_table(res_df)
            with open(
                table_folder + f"ex_ante_expected_margins_{file_append}.tex", "w"
            ) as f:
                f.write(table)


def generate_baseline_expectation_table(res_df):
    """Generate LaTeX table comparing informed vs uninformed under different uncertainty scenarios."""

    # Define metrics matching add_overall_results
    metrics = {
        # Work Life (<63)
        "working_hours_below_63": "Annual Labor Supply (hrs)",
        "consumption_below_63": "Annual Consumption",
        "savings_below_63": "Annual Savings",
        # Retirement
        # "ret_age": "Retirement Age",
        "sra_at_63": "SRA at 63",
        "ret_age_excl_disabled": "Retirement Age (excl. Disability)",
        "pension_wealth_at_ret": "Pension Wealth (PV at Retirement)",
        "pension_wealth_at_ret_excl_disability": "Pension Wealth (excl. Disability)",
        "private_wealth_at_ret": "Financial Wealth at Retirement",
        "private_wealth_at_ret_excl_disability": "Financial Wealth (excl. Disability)",
        "pensions": "Annual Pension Income",
        "pensions_excl_disability": "Annual Pension Income (excl. Disability)",
        "share_disability_pensions": "Share with Disability Pension",
        "pensions_share_below_63": "Share with Pension before 63",
        # Lifecycle (30+)
        "lifecycle_working_hours": "Annual Labor Supply (hrs)",
        "lifecycle_avg_wealth": "Average Financial Wealth",
        "cv": "Compensated Variation (\\%)",
    }

    sections = {
        "Retirement": [
            # "ret_age",
            "sra_at_63",
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
        "Welfare": ["cv"],
    }
    # Column order (now used as index)
    col_order = [
        "Informed_unc_False",
        "Informed_unc_True",
        "Uninformed_unc_False",
        "Uninformed_unc_True",
    ]

    latex_lines = []
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append(
        "    \\multirow{2}{*}{Expected Outcome} & "
        "\\multicolumn{2}{c}{Informed} & "
        "\\multicolumn{2}{c}{Misinformed} \\\\"
    )
    latex_lines.append("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    latex_lines.append("     & No Unc. & Unc. & No Unc. & Unc. \\\\")
    latex_lines.append("  {}   & (1) & (2) & (3) & (4) \\\\")
    latex_lines.append("    \\midrule")

    for section_name, section_metrics in sections.items():
        latex_lines.append("    \\midrule")

        latex_lines.append(
            f"    \\multicolumn{{5}}{{l}}{{\\textit{{{section_name}}}}} \\\\"
        )

        for metric_key in section_metrics:
            outcome_name = metrics[metric_key]
            row_data = [outcome_name]

            for col in col_order:
                val = res_df.loc[col, f"_{metric_key}"]
                row_data.append(f"{val:.2f}")

            latex_lines.append("    " + " & ".join(row_data) + " \\\\")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("\\end{tabular}")

    return "\n".join(latex_lines)
