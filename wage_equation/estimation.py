import pandas as pd
from linearmodels import PanelOLS

# source folder for SOEP core
SOEP_C38 = r"C:\Users\bruno\papers\soep\soep38"

# get relevant data (sex, employment status, gross income, full time experience) from SOEP core
pgen_df = pd.read_stata(f"{SOEP_C38}/pgen.dta", columns=['syear', 'hid', 'pid', 'pgemplst', 'pglabgro', 'pgexpft'], convert_categoricals=False)
ppathl_df = pd.read_stata(f"{SOEP_C38}/ppathl.dta", columns=['pid', 'hid', 'syear', 'sex', 'gebjahr'])
merged_df = pd.merge(pgen_df, ppathl_df[['pid', 'hid', 'syear', 'sex', 'gebjahr']], on=['pid', 'hid', 'syear'], how='inner')

# restrict sample
merged_df = merged_df[merged_df['sex'] == '[1] maennlich']  # only men
merged_df = merged_df[(merged_df['syear'] >= 2010) & (merged_df['syear'] <= 2020)]
merged_df = merged_df[merged_df['pgemplst'] == 1]  # only full time
merged_df = merged_df[(merged_df['pgexpft'] >= 0) & (merged_df['pgexpft'] <= 40)]
pglabgro_percentiles = merged_df['pglabgro'].quantile([0.01, 0.99])
merged_df = merged_df[(merged_df['pglabgro'] >= pglabgro_percentiles.iloc[0]) & (merged_df['pglabgro'] <= pglabgro_percentiles.iloc[1])]

# Prepare estimation
merged_df = merged_df.set_index(['syear', 'pid'])
merged_df = merged_df.rename(columns={'pgexpft': 'full_time_exp', 'pglabgro': 'wage'})
merged_df['full_time_exp_2'] = merged_df['full_time_exp'] ** 2

# Estimate parametric regression
PanelOLS(
            dependent=merged_df['wage'],
            exog=merged_df[['full_time_exp', 'full_time_exp_2']],
            entity_effects=True, time_effects=True
        ).fit().summary