import pandas as pd
from process_data.first_step_sample_scripts.create_health_transition_sample import (
    create_health_transition_sample,
)
from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from process_data.first_step_sample_scripts.create_partner_wage_est_sample import (
    create_partner_wage_est_sample,
)
from process_data.first_step_sample_scripts.create_survival_transition_sample import (
    create_survival_transition_sample,
)
from process_data.first_step_sample_scripts.create_wage_est_sample import (
    create_wage_est_sample,
)
from process_data.structural_sample_scripts.create_structural_est_sample import (
    create_structural_est_sample,
)
from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs


def create_table_describing_datasets(paths_dict, specs):
    df_struct, df_beliefs = import_main_datasets(paths_dict, specs)
    main_datasets = {
        "Structural Estimation Data": df_struct,
        "Policy Belief Data": df_beliefs,
    }
    df_description = describe_datasets(main_datasets, specs)
    return df_description


def import_main_datasets(paths_dict, specs):
    df_struct = create_structural_est_sample(paths_dict, specs=specs, load_data=True)
    df_struct["age"] = df_struct["period"] + specs["start_age"]

    df_beliefs = pd.read_stata(paths_dict["soep_is"])
    df_beliefs["education"] = df_beliefs["education"].replace({1: 0, 2: 0, 3: 1})
    df_beliefs["sex"] = df_beliefs["sex"].astype(str)
    df_beliefs["sex"] = (
        df_beliefs["sex"].replace({"Männlich": 0, "Weiblich": 1}).astype(int)
    )
    df_beliefs = df_beliefs[
        (df_beliefs["belief_pens_deduct"] >= 0)
        | (df_beliefs["pol_unc_stat_ret_age_67"] >= 0)
    ]
    return df_struct, df_beliefs


def describe_datasets(datasets, specs):
    rows = [
        "Years",
        "Nb. of Observations",
        "Age Range",
        "Median Age",
        "Female Share",
        "Highschool Degree Share",
    ]

    df_description = pd.DataFrame(
        {"": rows, **{name: [""] * len(rows) for name in datasets.keys()}}
    )
    # Populate the "Years" column. Note: For df_beliefs , "years" is 2022, otherwise it is between start year and end year.
    start_year = specs["start_year"]
    end_year = specs["end_year"]
    for name, df in datasets.items():
        if name == "Structural Estimation Data":
            df_description.loc[
                df_description[""] == "Years", name
            ] = f"{start_year}-{end_year}"
        else:
            df_description.loc[df_description[""] == "Years", name] = "2022"

    for name, df in datasets.items():
        df_description.loc[df_description[""] == "Nb. of Observations", name] = len(df)
        df_description.loc[
            df_description[""] == "Age Range", name
        ] = f"{df['age'].min()}-{df['age'].max()}"
        df_description.loc[df_description[""] == "Median Age", name] = df[
            "age"
        ].median()
        df_description.loc[df_description[""] == "Female Share", name] = (
            df["sex"] == 1
        ).mean()
        # Use the column fweights to calculate the weighted share of highschool degree holders
        if "fweights" in df.columns:
            fweights_sum = df["fweights"].sum()
            edu_high_fweights = df["education"] * df["fweights"]
            mean_high = edu_high_fweights.sum() / fweights_sum
        else:
            mean_high = (df["education"] == 1).mean()

        df_description.loc[
            df_description[""] == "Highschool Degree Share", name
        ] = mean_high
    return df_description
