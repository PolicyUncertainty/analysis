print("Running: gather_decision_data.py")

import pandas as pd

# Set file paths
SOEP_C38 = r"C:\Users\bruno\papers\soep\soep38"
SOEP_RV = r"C:\Users\bruno\papers\soep\soep_rv"

# Load SOEP core data
core_data = pd.read_stata(f"{SOEP_C38}/pgen.dta", columns=['syear', 'pid', 'hid', 'pgemplst', 'pgexpft'])
pathl_data = pd.read_stata(f"{SOEP_C38}/ppathl.dta", columns=['pid', 'hid', 'syear', 'sex', 'gebjahr', 'rv_id'])

# Merge core data with pathl data
merged_data = pd.merge(core_data, pathl_data, on=['pid', 'hid', 'syear'], how='inner')

# Calculate age and filter out unwanted records
merged_data['age'] = merged_data['syear'] - merged_data['gebjahr']
merged_data = merged_data[(merged_data['rv_id'] >= 0) | ((merged_data['rv_id'] < 0) & (merged_data['age'] < 60))]
merged_data['rv_id'].replace(-1, pd.NA, inplace=True)

# Load SOEP RV data
rv_data = pd.read_stata(f"{SOEP_RV}/vskt/SUF.SOEP-RV.VSKT.2020.var.1-0.dta", columns=['rv_id', 'JAHR', 'STATUS_2'])

# Merge with SOEP core data
merged_data = pd.merge(merged_data, rv_data, left_on=['rv_id', 'syear'], right_on=['rv_id', 'JAHR'], how='left')

# Filter out unwanted employment status
merged_data = merged_data[merged_data['pgemplst'].isin([0, 1])]

# Rename and replace values
merged_data.rename(columns={'pgemplst': 'choice'}, inplace=True)
merged_data['choice'].replace({5: 0}, inplace=True)
merged_data['choice'].replace({'RTB': 2}, inplace=True)

# Calculate period and filter out records
merged_data['period'] = merged_data['age'] - 25
merged_data = merged_data[merged_data['period'] >= 0]

# Set index for time series analysis
merged_data.set_index(['pid', 'syear'], inplace=True)

# Create lagged choice variable
merged_data['lagged_choice'] = merged_data.groupby('pid')['choice'].shift()

# Filter the sample
merged_data = merged_data[(merged_data['sex'] == 1) & (merged_data['syear'] >= 2010) & (merged_data['syear'] <= 2020)]

# Calculate policy_state
merged_data['policy_state'] = 67
mask1 = (merged_data['gebjahr'] <= 1964) & (merged_data['gebjahr'] >= 1958)
mask2 = (merged_data['gebjahr'] <= 1958) & (merged_data['gebjahr'] >= 1947)
merged_data.loc[mask1, 'policy_state'] = 67 - 2/12 * (1964 - merged_data['gebjahr'])
merged_data.loc[mask2, 'policy_state'] = 66 - 1/12 * (1958 - merged_data['gebjahr'])
merged_data.loc[merged_data['gebjahr'] < 1947, 'policy_state'] = 65

# Create retirement_age_id
merged_data['retirement_age_id'] = pd.NA

# Filter out invalid experience values
merged_data = merged_data[(merged_data['pgexpft'] >= 0) & (merged_data['pgexpft'] <= 40)]

# Round experience values
merged_data['experience'] = merged_data['pgexpft'].round()

# Keep relevant columns
merged_data = merged_data[['choice', 'period', 'lagged_choice', 'policy_state', 'retirement_age_id', 'experience']]

# Save data
merged_data.to_pickle('output/decision_data.pkl')