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
    calc_consumption_below_63,
    calc_working_hours_below_63,
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
            "ExpWorkHours",
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
            res_df.loc["ExpWorkHours", col_name] = calc_working_hours_below_63(df)
            res_df.loc["Consumption", col_name] = calc_consumption_below_63(df) * 10

    return res_df


def generate_latex_table(res_df):
    """
    Generate LaTeX tabular code from the results DataFrame.
    Only produces the tabular environment content (no table wrapper, caption, or notes).

    Parameters
    ----------
    res_df : pd.DataFrame
        DataFrame with outcomes in rows and scenarios in columns

    Returns
    -------
    str
        LaTeX tabular code
    """

    # Create mapping for expressive row names
    row_names = {
        "ExpWorkHours": "Expected Work Hours",
        "PrivWealthRet": "Private Wealth at Retirement (Tsd.)",
        "PensWealthRet": "Pension Wealth at Retirement (Tsd.)",
        "ExpLifetimeIncome": "Expected Lifetime Income (Tsd.)",
        "ExpRetAge": "Expected Retirement Age",
        "Consumption": "Consumption (Tsd.)",
    }

    # Reorder rows: ExpRetAge first, then wealth variables, then reaction margins
    row_order = [
        "ExpRetAge",
        "PrivWealthRet",
        "PensWealthRet",
        "ExpLifetimeIncome",
        "ExpWorkHours",
        "Consumption",
    ]

    # Add any remaining rows that might exist
    for idx in res_df.index:
        if idx not in row_order:
            row_order.append(idx)

    # Reorder columns: No Reform (Informed, Uninformed), then Expected Reform (Informed, Uninformed)
    col_order = [
        "Informed_unc_False",
        "Uninformed_unc_False",
        "Informed_unc_True",
        "Uninformed_unc_True",
    ]

    # Reorder dataframe
    df_latex = res_df.copy()
    df_latex = df_latex.reindex(index=row_order, columns=col_order)

    # Rename index
    df_latex.index = df_latex.index.map(lambda x: row_names.get(x, x))

    # Start building LaTeX tabular
    latex_code = []
    latex_code.append("\\begin{tabular}{lcccc}")
    latex_code.append("    \\toprule")

    # Header with scenario groups
    latex_code.append(
        "    \\multirow{2}{*}{\\textbf{Outcome}} & "
        + "\\multicolumn{2}{c}{\\textbf{No Expected Reform}} & "
        + "\\multicolumn{2}{c}{\\textbf{Expected Reform}} \\\\"
    )
    latex_code.append("    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")

    # Column headers
    latex_code.append(
        "     & \\textbf{Informed} & \\textbf{Uninformed} & \\textbf{Informed} & \\textbf{Uninformed} \\\\"
    )
    latex_code.append("  {}   & (1) & (2) & (3) & (4) \\\\")
    latex_code.append("    \\midrule")

    # Data rows
    for idx, row_name in enumerate(df_latex.index):
        row_data = [row_name]
        for col in df_latex.columns:
            val = df_latex.loc[row_name, col]
            row_data.append(f"{val:.2f}")

        # Add extra spacing after ExpRetAge (row 0) and after PensWealthRet (row 2)
        line = "    " + " & ".join(row_data) + " \\\\"
        if idx == 0:  # After ExpRetAge
            line += " \\addlinespace"
        elif idx == 2:  # After PensWealthRet
            line += " \\addlinespace"

        latex_code.append(line)

    latex_code.append("    \\bottomrule")
    latex_code.append("\\end{tabular}")

    return "\n".join(latex_code)
