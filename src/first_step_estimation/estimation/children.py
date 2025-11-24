import pandas as pd
from statsmodels import api as sm


def estimate_nb_children(paths_dict, specs):
    """Estimate the number of children in the household for each individual conditional
    on sex, education and age bin."""
    # load data, filter, create period and has_partner state
    df = pd.read_pickle(
        paths_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]

    df = df[df["age"] >= start_age]

    # Filter out individuals below 60 for better estimation(we should set this in specs)
    df = df[df["age"] <= 60]
    df["period"] = (df["age"] - start_age).astype(float)
    df["period_sq"] = df["period"] ** 2
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    # estimate OLS for each combination of sex, education and has_partner

    edu_states = list(range(specs["n_education_types"]))
    sexes = [0, 1]
    partner_states = [0, 1]

    sub_group_names = ["sex", "education", "has_partner"]

    multiindex = pd.MultiIndex.from_product(
        [sexes, edu_states, partner_states],
        names=sub_group_names,
    )

    columns = ["const", "period", "period_sq"]
    # Add columns for standard errors
    all_columns = columns + [f"{col}_ser" for col in columns]
    estimates = pd.DataFrame(index=multiindex, columns=all_columns)

    for sex in sexes:
        for education in edu_states:
            for has_partner in partner_states:
                df_reduced = df[
                    (df["sex"] == sex)
                    & (df["education"] == education)
                    & (df["has_partner"] == has_partner)
                ].copy()
                X = df_reduced[columns[1:]].astype(float)
                X = sm.add_constant(X)
                Y = df_reduced["children"]
                model = sm.OLS(Y, X).fit()
                # Save parameters
                estimates.loc[(sex, education, has_partner), columns] = model.params
                # Save standard errors
                for col in columns:
                    estimates.loc[(sex, education, has_partner), f"{col}_ser"] = (
                        model.bse[col]
                    )

    out_file_path = paths_dict["first_step_results"] + "nb_children_estimates.csv"
    estimates.to_csv(out_file_path)

    return estimates
