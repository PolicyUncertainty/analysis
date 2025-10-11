# %%
# Set paths of project
import pandas as pd

from model_code.transform_data_from_model import load_scale_and_correct_data
from set_paths import create_path_dict

path_dict = create_path_dict()
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)
from simulation.sim_tools.calc_aggregate_results import add_overall_results
from simulation.sim_tools.cv import calc_compensated_variation
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
    initial_obs_table = investigate_start_obs(
        path_dict=path_dict, model_class=None, load=True
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


def investigate_start_obs(
    path_dict,
    model_class,
    load=False,
):
    if load:
        initial_obs_table = pd.read_csv(
            path_dict["data_tables"] + "initial_obs_table.csv", index_col=[0, 1]
        )
        return initial_obs_table

    observed_data = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()
    # Get table of median and mean for continous variables
    # Get table of median and mean wealth
    median = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period", "wealth"]
    ].median()
    mean = start_period_data.groupby(["sex", "education"])[
        ["experience_years", "experience", "assets_begin_of_period", "wealth"]
    ].mean()
    rename_median = {col: col + "_median" for col in median.columns}
    rename_mean = {col: col + "_mean" for col in mean.columns}
    median = median.rename(columns=rename_median)
    mean = mean.rename(columns=rename_mean)
    initial_obs_table = median.merge(mean, left_index=True, right_index=True)
    initial_obs_table.to_csv(path_dict["data_tables"] + "initial_obs_table.csv")
    return initial_obs_table
