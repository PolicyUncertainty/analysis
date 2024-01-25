import pandas as pd
from linearmodels.panel.model import PanelOLS


def estimate_wage_parameters(paths, options, load_data=False):
    out_file_path = paths["project_path"] + "output/wage_eq_params.csv"

    if load_data:
        coefficients = pd.read_csv(out_file_path)
        return coefficients

    # unpack path to SOEP core
    soep_c38 = paths["soep_c38"]

    # unpack options
    start_year = options["start_year"]
    end_year = options["end_year"]
    exp_cap = options["exp_cap"],
    truncation_percentiles = options["wage_dist_truncation_percentiles"]

    # get relevant data (sex, employment status, gross income, full time experience) from SOEP core
    pgen_df = pd.read_stata(f"{soep_c38}/pgen.dta", columns=['syear', 'hid', 'pid', 'pgemplst', 'pglabgro', 'pgexpft'], convert_categoricals=False)
    ppathl_df = pd.read_stata(f"{soep_c38}/ppathl.dta", columns=['pid', 'hid', 'syear', 'sex', 'gebjahr'])
    merged_df = pd.merge(pgen_df, ppathl_df[['pid', 'hid', 'syear', 'sex', 'gebjahr']], on=['pid', 'hid', 'syear'], how='inner')

    # restrict sample
    merged_df = merged_df[merged_df['sex'] == '[1] maennlich']  # only men
    merged_df = merged_df[(merged_df['syear'] >= start_year) & (merged_df['syear'] <= end_year)]
    merged_df = merged_df[merged_df['pgemplst'] == 1]  # only full time
    merged_df = merged_df[(merged_df['pgexpft'] >= 0) & (merged_df['pgexpft'] <= exp_cap)]
    pglabgro_percentiles = merged_df['pglabgro'].quantile(truncation_percentiles)
    merged_df = merged_df[(merged_df['pglabgro'] >= pglabgro_percentiles.iloc[0]) & (merged_df['pglabgro'] <= pglabgro_percentiles.iloc[1])]

    # Prepare estimation
    merged_df = merged_df.set_index(['syear', 'pid'])
    merged_df = merged_df.rename(columns={'pgexpft': 'full_time_exp', 'pglabgro': 'wage'})
    merged_df['full_time_exp_2'] = merged_df['full_time_exp'] ** 2

    # estimate parametric regression, save parameters
    model = PanelOLS(
                dependent=merged_df['wage'],
                exog=merged_df[['full_time_exp', 'full_time_exp_2']],
                entity_effects=True, time_effects=True
            )
    coefficients = model.fit().params
    print("Estimated wage equation coefficients:\n{}".format(coefficients.to_string()))

    # Export regression coefficients
    coefficients.to_csv(out_file_path)
    return coefficients