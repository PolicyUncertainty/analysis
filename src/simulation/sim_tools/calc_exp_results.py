# %%
# Set paths of project
import pandas as pd
from matplotlib import pyplot as plt

from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)
import numpy as np

from simulation.sim_tools.calc_aggregate_results import (
    calc_average_retirement_age,
    expected_lifetime_income,
    pension_wealth_at_retirement,
    private_wealth_at_retirement,
)
from simulation.sim_tools.simulate_exp import simulate_exp


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
        "partner_state": 0,
        "job_offer": 1,
        "policy_state": 8,  # SRA = 67
    }

    res_df = pd.DataFrame(
        index=[
            "ExpRetAge",
            "ExpLifetimeIncome",
            "PrivWealthRet",
            "PensWealthRet",
            "FT-Prob",
            "Consumption",
        ],
    )
    for subj_unc in [True, False]:
        model_solution = None
        for informed, informed_label in enumerate(["Uninformed", "Informed"]):
            col_name = f"{informed_label}_unc_{subj_unc}"
            res_df[col_name] = np.nan
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
            res_df.loc["ExpRetAge", col_name] = calc_average_retirement_age(df)
            res_df.loc["ExpLifetimeIncome", col_name] = (
                expected_lifetime_income(df, specs) * 10
            )
            res_df.loc["PrivWealthRet", col_name] = (
                private_wealth_at_retirement(df) * 10
            )
            res_df.loc["PensWealthRet", col_name] = (
                pension_wealth_at_retirement(df, specs) * 10
            )
            # Calc FT prob and consumption
            choice_probs = model_solution.choice_probabilities_for_states(states=state)
            res_df.loc["FT-Prob", col_name] = choice_probs[0, 3] * 100
            choice_policies = model_solution.choice_policies_for_states(states=state)

            res_df.loc["Consumption", col_name] = (
                np.nansum(choice_probs[0, :] * choice_policies[0, :]) * 10
            )

    return res_df


def generate_latex_table(res_df):
    """
    Generate a nicely formatted LaTeX table from the results DataFrame.

    Parameters
    ----------
    res_df : pd.DataFrame
        DataFrame with outcomes in rows and scenarios in columns

    Returns
    -------
    str
        LaTeX table code
    """
    # Create mapping for expressive row names
    row_names = {
        "ExpRetAge": "Expected Retirement Age",
        "ExpLifetimeIncome": "Expected Lifetime Income (€)",
        "PrivWealthRet": "Private Wealth at Retirement (€)",
        "PensWealthRet": "Pension Wealth at Retirement (€)",
        "FT-Prob": "Full-Time Work Probability",
        "Consumption": "Consumption (€)",
    }

    # Create mapping for expressive column names
    col_names = {
        "Uninformed_unc_True": "Uninformed\\newline w/ Uncertainty",
        "Informed_unc_True": "Informed\\newline w/ Uncertainty",
        "Uninformed_unc_False": "Uninformed\\newline w/o Uncertainty",
        "Informed_unc_False": "Informed\\newline w/o Uncertainty",
    }

    # Rename columns and index
    df_latex = res_df.copy()
    df_latex.index = df_latex.index.map(lambda x: row_names.get(x, x))
    df_latex.columns = df_latex.columns.map(lambda x: col_names.get(x, x))

    # Format numbers appropriately
    def format_value(val, row_name):
        if pd.isna(val):
            return "--"
        if "Age" in row_name:
            return f"{val:.2f}"
        elif "Probability" in row_name:
            return f"{val:.3f}"
        elif "€" in row_name:
            return f"{val:,.0f}".replace(",", "\\,")
        else:
            return f"{val:.2f}"

    # Start building LaTeX table
    latex_code = []
    latex_code.append("\\begin{table}[htbp]")
    latex_code.append("    \\centering")
    latex_code.append(
        "    \\caption{Retirement Decisions Under Different Information and Uncertainty Scenarios}"
    )
    latex_code.append("    \\label{tab:retirement_scenarios}")
    latex_code.append("    \\begin{tabular}{l" + "c" * len(df_latex.columns) + "}")
    latex_code.append("        \\toprule")

    # Header with scenario groups
    latex_code.append(
        "        \\multirow{2}{*}{\\textbf{Outcome}} & "
        + "\\multicolumn{2}{c}{\\textbf{Expecting Reform}} & "
        + "\\multicolumn{2}{c}{\\textbf{Not expecting reform}} \\\\"
    )
    latex_code.append("        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")

    # Column headers
    headers = [""] + ["Uninformed", "Informed", "Uninformed", "Informed"]
    latex_code.append(
        "        "
        + " & ".join(f"\\textbf{{{h}}}" if h else h for h in headers)
        + " \\\\"
    )
    latex_code.append("        \\midrule")

    # Data rows
    for idx, row_name in enumerate(df_latex.index):
        row_data = [row_name]
        for col in df_latex.columns:
            val = df_latex.loc[row_name, col]
            row_data.append(format_value(val, row_name))

        # Add extra spacing after certain rows
        line = "        " + " & ".join(row_data) + " \\\\"
        if idx == 0:  # After retirement age
            line += " \\addlinespace"
        elif idx == 3:  # After pension wealth
            line += " \\addlinespace"

        latex_code.append(line)

    latex_code.append("        \\bottomrule")
    latex_code.append("    \\end{tabular}")
    latex_code.append("    \\begin{tablenotes}[flushleft]")
    latex_code.append("        \\small")
    latex_code.append(
        "        \\item \\textit{Notes:} This table presents expected outcomes for different combinations of "
    )
    latex_code.append(
        "        information states and uncertainty. Informed individuals know their health status, while "
    )
    latex_code.append(
        "        uninformed individuals face uncertainty. Monetary values are in Euros."
    )
    latex_code.append("    \\end{tablenotes}")
    latex_code.append("\\end{table}")

    return "\n".join(latex_code)
