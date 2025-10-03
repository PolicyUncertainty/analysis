# Structural Estimation Sample Variable Documentation

This document describes all variables in the structural estimation sample, their construction, and the sample restrictions applied during data processing.

## State and Choice Variables

### 1. **choice** ($d_t$) - Labor Market Choice
- **Possible values**: 
  - 0: Retired
  - 1: Unemployed
  - 2: Working part-time
  - 3: Working full-time
- **Original variable(s)**: 
  - Primary: Employment spells from SOEP artkalen dataset  
  - Fallback: `pgemplst` (employment status) and `pgstib` (job type) from SOEP pgen
- **Modifications**:
  - Created using `create_choice_variable_from_artkalen()` function
  - Uses employment spell data to determine dominant employment type during interview period
  - For missing spell data, fills with pgen-based choice if consistent with previous choice
  - Age corrections applied when retirement spells start mid-year
- **Sample restrictions**:
  - Men with choice=2 (part-time) are dropped
  - Retirement is absorbing state (no transitions from choice=0)

### 2. **lagged_choice** ($d_{t-1}$) - Previous Period Labor Market Choice
- **Possible values**: 
  - 0: Retired (lagged)
  - 1: Unemployed (lagged)
  - 2: Part-time (lagged)
  - 3: Full-time (lagged)
- **Original variable(s)**: Derived from choice variable
- **Modifications**:
  - One-period lag of choice variable
  - First observation per individual has lagged_choice based on employment status in previous year
- **Sample restrictions**:
  - Men with lagged_choice=2 are dropped (no male part-time)
  - Individuals leaving retirement (lagged_choice=0 but choice≠0) are dropped

### 3. **period** ($t$) - Model Period
- **Possible values**: 0 to 42 (corresponding to ages 30-72)
- **Original variable(s)**: Derived from age and start_age specification
- **Modifications**:
  - Calculated as: `age - specs["start_age"]`
- **Sample restrictions**:
  - None directly on period

### 4. **education** ($\tau$) - Education Type
- **Possible values**:
  - 0: Low education (below Fachhochschulreife)
  - 1: High education (Fachhochschulreife or Abitur)
- **Original variable(s)**: `pgpsbil` (SOEP pgen dataset)
- **Modifications**:
  - Binary classification based on pgpsbil values 3 (Fachhochschulreife) and 4 (Abitur)
  - Missing values set to None initially, then dropped
- **Sample restrictions**:
  - Individuals with missing education are dropped

### 5. **sex** ($\tau$) - Gender
- **Possible values**:
  - 0: Men
  - 1: Women
- **Original variable(s)**: `sex` (SOEP ppathl dataset)
- **Modifications**:
  - Recoded from original SOEP coding to 0/1 binary
- **Sample restrictions**:
  - None directly

### 6. **job_offer** ($o_t$) - Job Offer State
- **Possible values**:
  - 0: No job offer
  - 1: Job offer
  - -99: Unobserved
- **Original variable(s)**: Derived from employment status and job separation data
- **Modifications**:
  - Set to 1 if currently working (choice in [2,3])
  - Set to -99 (unobserved) for non-working individuals
- **Sample restrictions**:
  - None directly (unobserved states handled in estimation)

### 7. **partner_state** ($p_t$) - Partner Employment Status
- **Possible values**:
  - 0: Single
  - 1: Partner working age
  - 2: Partner retired
- **Original variable(s)**: `parid` (partner ID) and partner's work status
- **Modifications**:
  - Created by merging couples based on parid
  - Partner's work status determined from partner's employment variables
  - Corrected for missing single-year observations using lead/lag values
- **Sample restrictions**:
  - Individuals with missing partner_state are dropped

### 8. **health** ($h_t$) - Health State
- **Possible values**:
  - 0: Good health
  - 1: Bad health (set to -99 in final sample)
  - 2: Disabled
  - 3: Dead (not in estimation sample)
- **Original variable(s)**: `m11126` (self-rated health), `m11124` (disability status) from SOEP pequiv
- **Modifications**:
  - Good health: m11126 in [1,2,3] and m11124=0
  - Bad health: all other valid observations
  - Corrected using lead/lag values (bad health between two good health years → good)
  - Modified for disability pension: fresh retirees below min_long_insured_age set to disabled
  - All bad health (1) set to unobserved (-99)
- **Sample restrictions**:
  - Individuals with missing health are dropped

### 9. **informed** ($i_t$) - Information State
- **Possible values**: 
  - 0: uninformed
  - 1: informed
- **Origin and Modifications**:
  - only placeholder, variable is predicted from SOEP-IS sample

### 10. **policy_state** ($\pi_t$) - Pension Policy State
- **Possible values**: [PLACEHOLDER - discrete values based on SRA grid]
- **Original variable(s)**: Birth year (`gebjahr`) and policy parameters
- **Modifications**:
  - SRA calculated by birth year using `create_SRA_by_gebjahr()`
  - Rounded to nearest policy grid point
  - Age adjusted (+1) when SRA < min_SRA
- **Sample restrictions**:
  - None directly

### 11. **wealth** ($w_t$) - Household Wealth
- **Possible values**: ℝ≥0 (scaled units)
- **Original variable(s)**: Multiple wealth components from SOEP
- **Modifications**:
  - Linear interpolation between observed wealth points
  - Deflated to base year prices
  - Negative wealth set to 0
  - Divided by wealth_unit for scaling
- **Sample restrictions**:
  - None directly

### 12. **experience** ($e_t$) - Work Experience
- **Possible values**: ℝ≥0 (scaled)
- **Original variable(s)**: `pgexpft`, `pgexppt` from SOEP pgen
- **Modifications**:
  - Sum of full-time and 0.5*part-time experience
  - Scaled differently for retired vs. working individuals
  - Updated based on current choice and lagged experience
- **Sample restrictions**:
  - Individuals with invalid experience are dropped

### 13. **SRA** - Statutory Retirement Age
- **Possible values**: Continuous values around 65-67
- **Original variable(s)**: Birth year (`gebjahr`) and pension law
- **Modifications**:
  - Calculated from birth year using pension policy rules
  - Rounded to nearest grid point for policy_state creation
- **Sample restrictions**:
  - None directly

## Other Variables

### **min_SRA** - Minimum SRA for Period
**Original variable(s)**: Model specification parameter  
**Modifications**: None

### **age** - Age in Years  
**Original variable(s)**: `syear` - `gebjahr`  
**Modifications**: Calculated at interview date; adjusted (+1) when SRA < min_SRA to maintain valid policy state

### **monthly_wage** - Monthly Gross Wage
**Original variable(s)**: `pglabgro` from SOEP pgen  
**Modifications**: Renamed from pglabgro

### **monthly_wage_partner** - Partner's Monthly Gross Wage
**Original variable(s)**: `pglabgro_p` (partner's wage after merge)  
**Modifications**: Created through couple merge based on parid

### **hh_net_income** - Household Net Income
**Original variable(s)**: `hlc0005_v2` from SOEP hl dataset  
**Modifications**: Multiplied by 12 (to annualize); divided by wealth_unit for scaling

### **working_years** - Total Working Years
**Original variable(s)**: `pgexpft` + `pgexppt`  
**Modifications**: Sum of full-time and part-time experience years

### **children** - Number of Children in Household
**Original variable(s)**: `d11107` from SOEP pequiv  
**Modifications**: Renamed from d11107

### **very_long_insured** - Very Long Insured Classification
**Original variable(s)**: Derived from experience and retirement age  
**Modifications**: Classification based on experience years and retirement age difference; applied only to fresh retirees

### **job_sep** - Job Separation Indicator
**Original variable(s)**: `plb0304_h` from SOEP pl dataset  
**Modifications**: Set to 1 if plb0304_h in [1, 3, 5] (various firing reasons); set to 0 otherwise

## General Sample Restrictions

1. **Temporal restrictions**:
   - Years: 2013-2020 (with buffer years for lagging/leading)
   - Ages: 30-72

2. **Model restrictions**:
   - Retirement is absorbing (choice=0 → lagged_choice must be 0 in next period)
   - No work after max_ret_age (72)
   - No unemployment after SRA
   - Men cannot work part-time

3. **Data completeness**:
   - All core state variables must be non-missing
   - Continuous panel required (gaps filled through spanning)