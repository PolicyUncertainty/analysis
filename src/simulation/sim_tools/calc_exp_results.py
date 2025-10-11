# %%
# Set paths of project
import pandas as pd
from matplotlib import pyplot as plt

from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)
from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.simulate_exp import simulate_exp
from simulation.tables.cv import calc_compensated_variation


def calc_exp_results(
    path_dict,
    specs,
    sex,
    education,
    params,
    model_name,
    load_solution,
    load_sol_model,
    util_type,
):
    initial_obs_table = pd.read_csv(
        path_dict["data_tables"] + "initial_obs_table.csv", index_col=[0, 1]
    )
    fixed_states = {
        "period": 0,
        "sex": sex,
        "education": education,
        "lagged_choice": 1,
        "health": 0,
        "assets_begin_of_period": initial_obs_table.loc[
            (sex, education), "assets_begin_of_period_median"
        ],
        "experience": initial_obs_table.loc[(sex, education), "experience_median"],
        "partner_state": 1,
        "job_offer": 1,
        "policy_state": 8,  # SRA = 67
    }

    # Initialize empty DataFrame
    res_df = pd.DataFrame()

    df_base = None
    for subj_unc in [False, True]:
        model_solution = None
        for informed_label in ["Informed", "Uninformed"]:
            informed = int(informed_label == "Informed")
            col_prefix = f"{informed_label}_unc_{subj_unc}"
            print(
                "Eval expectation: ",
                subj_unc,
                informed_label,
                flush=True,
            )
            state = {
                **fixed_states,
                "informed": informed,
            }

            df, model_solution = simulate_exp(
                initial_state=state,
                n_multiply=1_000,
                path_dict=path_dict,
                params=params,
                subj_unc=subj_unc,
                custom_resolution_age=None,
                model_name=model_name,
                solution_exists=load_solution,
                sol_model_exists=load_sol_model,
                model_solution=model_solution,
                util_type=util_type,
            )
            if (informed_label == "Informed") and (not subj_unc):
                df_base = df.copy()
                res_df.loc[col_prefix, "_cv"] = 0.0
            else:
                res_df.loc[col_prefix, "_cv"] = (
                    calc_compensated_variation(
                        df_base=df_base,
                        df_cf=df,
                        params=params,
                        specs=specs,
                    )
                    * 100
                )

            # Use col_prefix as index, empty string as pre_name
            res_df = add_overall_results(
                result_df=res_df,
                df_scenario=df,
                index=col_prefix,
                pre_name="",
                specs=specs,
            )
            if subj_unc:
                res_df.loc[col_prefix, "_sra_at_63"] = 68.23

    return res_df


def generate_latex_table(res_df):
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
