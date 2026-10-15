# Retirement timing: pension claiming

In the model, **retirement is the claim of a statutory pension**: choice 0 starts in
the period in which the first old-age or disability pension is paid, and the
retirement age is the age at the start of that payment. When people stop working is
not the object of interest. This document describes how the data processing
approximates the claim and its timing, how well that works against administrative
pension records (SOEP-RV), and what a correction could look like.

## 1. Current derivation in the pipeline

### Interview age
`soep_vars/age.py`: `float_age = float_interview - float_birth_date`,
`age = floor(float_age)`.

### Claim indicator and claim date from artkalen spells
`soep_vars/choice_from_spell.py::select_spell_for_pid` loops over the yearly
interviews of each person. The pension spell (`spelltyp == 6`) is the claim
indicator.

- **Dominant spell** (`select_dominant_spelltyp_and_recode`): among the spells
  covering the interview date, the priority is education → NaN; full-time (1, 2) →
  3; pension (6) → 0; part-time (3) → 2; unemployment-like (5, 7, 10, 14) → 1;
  else NaN. Full-time dominates a pension spell, so people who claim and keep
  working full-time are not coded as claimants.
- **Claim date**: in the first pension observation,
  `corrected_age = start of the pension spell - birth date`. The start is the
  `float_begin` of the first new type-6 spell covering the interview. If there is
  none, it is the end of the previous interview's dominant spell (fallback).
- **Claim at or after the SRA** (interview `age >= SRA`): absorbing; all later
  years are set to 0 without looking at the spells.
- **Pension spell ends before the SRA**: all earlier pension observations are
  recoded to unemployment (1), and `corrected_age` of the first one is reset to the
  integer age. Earlier pension spells stay marked as seen, so a second claim takes
  its date from the fallback.
- The result is cached in `output/intermediate_data/struct/art_choice.pkl`
  (`art_choice`, `corrected_age`). `load_artkalen_choice=True` (the default in
  `run_process_data.py`) reloads it without checking whether it matches the code.

### Survey fill and pension-receipt check
`choice_from_spell.py::create_choice_variable_from_artkalen`:

- A missing spell-based choice is filled with the pgen choice only if that equals
  the previous year's spell-based choice. A first claim can therefore never come
  from pgen.
- `choice == 0` is kept only if the person reports current pension receipt
  (`plc0232_v1 == 1`) or next year's `igrv1 > 0`; otherwise it is set to NaN.
- `lagged_choice` is the one-year lag of this final choice.

### Mapping the claim date to the integer-age model
`structural_sample_scripts/policy_state.py::create_policy_state`:

1. `age_diff = age - corrected_age`. It is non-zero only in the claim year.
2. Persons with `|age_diff| > 1` are dropped entirely.
3. For fresh claimants (`choice == 0 & lagged_choice != 0`): `SRA += age_diff`.
   This keeps the distance between the claim age and the SRA while using the
   integer `age`.
4. If `SRA < min_SRA` (65): `SRA += 1` and `age += 1`.
5. Rounding to the 0.25 grid gives `policy_state` / `policy_state_value`.

### Downstream uses of the claim timing
- `disability_pension_health.py`: fresh claimants below `min_long_insured_age`
  (63) are set to disabled (`health = 2`).
- `classify_reitrees.py`: the very-long-insured flag uses
  `policy_state_value - age`.
- `model_restrictions.py`: no non-claim choice at `age >= max_ret_age` (72), and
  no unemployment at `age >= policy_state_value`.
- `data_plots/retirement_timing.py`: distance to the SRA (`plot_retirement_timing_data`)
  and the monthly claim-age distribution (`plot_derived_retirement_age_months`).

### Issues in the code
- **Policy state jumps.** Only the claim year gets the shifted SRA. The years
  before and after keep the cohort SRA, so `policy_state` changes within a person.
- **The SRA < 65 fix adds 1 to the age of a single row.** The next year can then
  have the same age / period.
- **The pension-receipt check runs after the spell logic.** A claim row set to NaN
  keeps its `corrected_age`, and the next row can look like a fresh claim with an
  unshifted date.
- **Absorbing claims are not enforced by a filter.**
  `structural_estimation_sample_description.md` says that persons leaving
  retirement are dropped, but no code does this. Absorption comes only from the
  recoding in the spell loop (checked in `test_decision_data.py`).
- **Claim indicator and claim date are built in the same loop**, and the date is
  carried by the implicit column `corrected_age`.

## 2. Monthly claim-age distribution (spell-based)

`plot_derived_retirement_age_months` plots `corrected_age`, floored to months, for
fresh claimants (`art_choice == 0` after 1–3) in 2013–2020, from the cached
`art_choice.pkl` (N = 1,585 with a claim date).

- 24% claim at age 63 and 26% at age 65.
- There are spikes at exactly 63 (5.5% in a single month) and from 65 to 65y8m
  (the cohort SRAs of the 1948–1954 cohorts).
- Only 14% claim in a month that is a full year of age.

## 3. Validation against administrative pension records (SOEP-RV)

### Data
Linked via `rv_id` in `ppathl` (only respondents who consented). Used: older
releases `rtbn/SOEP_RV_RTBN_2020.dta` and `vskt/SUF.SOEP-RV.VSKT.2020.{fix,var}`.
Codebooks of the current releases: RTBN 2023 V2, VSKT 2022 V2.

- **Admin claim date**: RTBN `RTBEJ` / `RTBEM`, the start of the first
  uninterrupted pension payment. A later change of pension type (e.g. disability →
  old age) does not reset it. Fallbacks: the start of the current pension in RTBN,
  then in VSKT fix (`ZTPTRTBEJ` / `ZTPTRTBEM`). Dates use the spell convention
  `year + (month - 1) / 12`. The birth month comes from SOEP, since the RV data
  have only the birth year.
- **Pension type**: `LEAT`. 1 = disability, 2 = unemployment, 3 = partial
  retirement, 4 = women, 5 = severely disabled, 6 = long-insured,
  7 = very-long-insured, 8 = regular. The 2020 release uses the same coding
  (`LEAT == 1` coincides with `RTAT == 1`). The comment in
  `create_credited_periods_est_sample.py` describes a different coding; the
  estimation there only uses `RTAT == 2` and is not affected.
- **VSKT variable part**: a monthly status history. Pension receipt (RTB) carries
  no pension points and can be crowded out of STATUS_2/3, so it is only a check,
  not the claim date.

### Coverage
- 21% of the 60,475 persons interviewed 2013–2020 are linked; 3,276 of them have
  an admin claim date.
- 43% of the 2,159 spell-based fresh claimants 2013–2020 are linked; 791 have an
  admin claim date (741 from the RTBN first start).
- 44% of the respondents asked about linkage consented.

### Claim date: spells vs admin
Constructed minus admin claim age, in months, N = 791:

| Group | N | Exact | Within 1 | Within 12 | Median |
|---|---|---|---|---|---|
| All | 791 | 46% | 56% | 82% | 0 |
| Spell start | 597 | 53% | 64% | 82% | 0 |
| Fallback start | 194 | 25% | 31% | 80% | 0 |
| Old-age pensions | 669 | 51% | 61% | 86% | 0 |
| Disability pensions | 118 | 17% | 25% | 59% | +6 |
| Very-long-insured (`LEAT` 7) | 174 | – | 76% | 92% | 0 |

Disability claims are recorded late in the spells: the median gap is +6 months,
and p90 is +86 months.

The admin claim date lies relative to the survey interviews as follows:
- 67% between the previous and the claim interview (consistent);
- 15% before the previous interview (claim assigned one survey year too late);
- 13% after the claim interview (claim assigned too early);
- 5% have a missing interview date.

### Selection into linkage
Among persons aged 50–70, all standardized differences between linked and
non-linked persons are below 0.12. Linked persons are slightly older, more often
female and pensioners, less often self-employed, and have lower earnings.

The spell-based claim age differs strongly by linkage: mean 61.8 for linked vs
58.8 for non-linked persons, with 18% vs 30% claiming below 60. This is driven by
574 fresh claimants whose cached `corrected_age` is missing. Inside the pipeline
these are filled with the interview age (they are the bulk of the fallback cases);
they are mostly not linked and young. Without them the means were equal (62.1).
The cache likely predates the current interview-date imputation and should be
rebuilt before drawing conclusions.

## 4. Where admin claims are lost in the pipeline

Population: 1,317 linked persons whose admin claim falls in a survey year
2013–2020. The expected claim year is the first year with an interview on or after
the admin claim date.

Only **33%** appear as a fresh claim in that year, 7% in another year, and **60%
never**. Each person is counted at the first step that loses the claim:

| Step | Share | Comment |
|---|---|---|
| Already a claimant in the previous year | 26% | Mostly a survey gap, see below |
| Spells give no choice in the claim year | 11% | Mostly no spell covers the interview; pgen reports a pension for 37% |
| Spells say no pension in the claim year | 10% | One third of all disability claims; mostly unemployment spells |
| Previous-year choice missing | 8% | Transition cannot be observed |
| Pension-receipt check | 0.5% | Not an issue |
| Later sample filters | ~11% | Wealth 3.2%, missing core variables 2.6%, final missing check 2.1%, civil servants / self-employed 1.7%, model restrictions 1.3% |

Disability claims are found in the right year in 19% of cases, old-age claims in
37%.

### Why
- **"Already a claimant in the previous year" is mostly a survey gap**, not an early
  claim.
  - For 75% of these persons the first spell-based claim lies exactly one year
    before the expected claim year.
  - For 53% that year has no interview. The spells place the claim in a
    non-interviewed year, whose row is dropped later, and the next year then has
    lagged choice 0.
  - For 68% the year before has no choice either.
  - Where a constructed claim date exists, the pension spell starts 3–24 months
    before the admin claim. Respondents report a pension spell before statutory
    payment begins.
- **"Spells say no pension"** coincides with unemployment spells (76 of 133).
  - The first pension spell starts within 3 months of the admin claim for about
    a third of the two spell-related groups.
  - It starts 3–24 months after the admin claim for about a quarter.
  - This fits disability pensions that are granted retroactively.
- **"Spells give no choice"**: no spell covers the interview for 106 of 139.

## 5. Proposed correction and evaluation

A correction of the claim indicator and claim date, applied right after the
spell-based choice:

1. **Admin claim date** (linked persons only):
   - The claim year is the first interviewed year on or after the admin claim
     date; choice = 0 from then on.
   - Earlier pension observations are recoded to unemployment, because no
     statutory pension was paid yet.
   - Non-interviewed years after the claim date get a missing choice.
   - `corrected_age` is the admin claim date minus the birth date.
2. **Survey-reported claim** (persons without an admin date). If all of the
   following hold, choice = 0:
   - spells give no choice or unemployment;
   - pgen reports being a pensioner;
   - a pension is received (`plc0232_v1` or next year's `igrv1`);
   - the person does not work in any later year.
3. **Bridge single missing years**: a fresh claim with a missing previous-year
   choice uses the choice from two years before.

Variants: *survey_only* = rules 2–3, *full* = rules 1–3. Evaluated on the linked
population of section 4 (for *survey_only* the admin dates are not used, so it is a
holdout test of the rules that apply to everyone else):

| | current | survey_only | full |
|---|---|---|---|
| Claim found in the expected year | 33% | 37% | 56% |
| Claim never found | 60% | 54% | 44% |
| Disability claims found in the expected year | 19% | 24% | 54% |
| Fresh claimants in the final sample (all persons) | 1,178 | 1,426 | 1,534 |

- **full** removes all spell-related losses. The remaining losses are:
  - missing previous-year choice from survey gaps longer than one year (21%);
  - the general sample filters (about 22%).
- **survey_only**, compared with the admin dates:
  - exact agreement falls from 58% to 53%, and agreement within 12 months is
    unchanged (92–93%);
  - 80 more linked claimants are kept;
  - "found one year early" rises from 23 to 42. The pension-receipt questions
    likely also capture non-statutory pensions (survivor, civil-servant or
    occupational pensions), so the rule needs tightening before use.
- The claim-age and SRA-distance distributions of the final sample barely change
  across variants.

## 6. Next steps

1. **Use the admin claim date for linked persons** (rule 1).
2. **Fix the pre-claim status after survey gaps.**
   - For linked persons: the VSKT monthly status in the months before the claim.
   - For everyone else: the last observed status over gaps longer than one year.
3. **Tighten the survey-reported claim.** Require a statutory pension (if the
   survey allows distinguishing it) and an age floor for old-age claims.
4. **Rebuild `art_choice.pkl`** (`LOAD_ARTKALEN_CHOICE = False` once) and recheck
   the fallback cases.
5. **Check which variables drop claim-year rows** in the wealth and missing-value
   filters.

Diagnostics behind sections 3–5 are local and not versioned (`reports/`,
gitignored). They write only aggregate tables with small cells suppressed.
