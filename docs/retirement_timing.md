# Retirement timing in the data processing

How the structural estimation sample derives the retirement choice and the
(true) retirement age, and how that timing is mapped to the integer-age model.
Written as a basis for reworking this part of the processing.

## Pipeline

### 1. Interview age
`soep_vars/age.py`: `float_age = float_interview - float_birth_date`,
`age = floor(float_age)`.

### 2. Retirement choice and true retirement age (artkalen spells)
`soep_vars/choice_from_spell.py::select_spell_for_pid`, a loop over the yearly
interviews of each pid:

- **Dominant spell** (`select_dominant_spelltyp_and_recode`): among spells
  covering the interview date, priority is education → NaN; full-time (1, 2) → 3;
  retirement (6) → 0; part-time (3) → 2; unemployment-like (5, 7, 10, 14) → 1;
  else NaN. Full-time dominates a retirement spell, so working pensioners are
  full-time.
- **First retirement observation**: `corrected_age = start of retirement - birth date`.
  The start is the `float_begin` of the first *new* type-6 spell covering the
  interview. If there is none, it falls back to the end of the previous
  interview's dominant spell.
- **Retired at/after SRA** (interview `age >= SRA`): retirement becomes absorbing;
  all later years are set to 0 without looking at spells.
- **Un-retirement before SRA**: all earlier retirement observations are recoded to
  unemployment (1), the first retirement row's `corrected_age` is reset to the
  integer `age`, and the retired flag is cleared. `past_ret_spells` is *not*
  cleared, so a later re-retirement takes its start from the fallback.
- Output is cached in `output/intermediate_data/struct/art_choice.pkl`
  (`art_choice`, `corrected_age`); `load_artkalen_choice=True` reloads it.

### 3. Merge with pgen and pension-receipt check
`choice_from_spell.py::create_choice_variable_from_artkalen`:

- Missing artkalen choices are filled with the pgen choice only if it equals the
  lagged artkalen choice.
- `choice == 0` is kept only if the person reports current pension receipt
  (`plc0232_v1 == 1`) or next year's `igrv1 > 0`; otherwise it is set to NaN.
- `lagged_choice` is the shift of this final choice.

### 4. SRA correction for integer model ages
`structural_sample_scripts/policy_state.py::create_policy_state`, after the year,
age and missing-value filters:

1. `age_diff = age - corrected_age`. Non-zero only for first-retirement rows;
   positive means retirement started before the interview age.
2. Whole pids with `|age_diff| > 1` are dropped.
3. Fresh retirees (`choice == 0 & lagged_choice != 0`): `SRA += age_diff`. This
   keeps the true distance `corrected_age - SRA` while using the integer `age`.
4. If `SRA < min_SRA` (65): `SRA += 1` and `age += 1`.
5. Rounding to the 0.25 grid gives `policy_state` / `policy_state_value`.

### 5. Downstream uses of the corrected timing
- `disability_pension_health.py`: fresh retirees below `min_long_insured_age`
  (63) get `health = 2`.
- `classify_reitrees.py`: very-long-insured flag uses `policy_state_value - age`.
- `model_restrictions.py`: no work at `age >= max_ret_age` (72), no unemployment
  at `age >= policy_state_value` (the shifted value for fresh retirees).
- `data_plots/retirement_timing.py::plot_retirement_timing_data`: rebuilds the
  distance to the SRA from `period` and `policy_state`.

## Issues to address in the rework

- **Policy state jumps within a person.** Only the fresh-retiree row gets a
  shifted SRA. Rows before and after keep the cohort SRA, so `policy_state` can
  change in the retirement year.
- **The SRA < 65 fix can create duplicate ages.** `age += 1` is applied to that
  single row, so the following year can have the same age / period. This affects
  cohorts with SRA 65 who retired in the same age-year as the interview
  (`age_diff < 0`). Not yet checked against the data.
- **Pension check runs after the spell logic.** A first-retirement row whose
  choice is set to NaN keeps its `corrected_age`. The next row can then look like
  a fresh retiree that has the wrong (unshifted) timing.
- **Absorbing retirement is not enforced by a filter.**
  `structural_estimation_sample_description.md` says people leaving retirement
  are dropped, but no code does this. Retirement stays absorbing only through the
  recoding in the spell loop; `test_decision_data.py` checks it.
- **Mixed responsibilities.** Spell selection and timing correction both happen
  in `select_spell_for_pid`, and the correction is spread over three files and
  carried by the implicit `corrected_age` column.
  `soep_vars/retirement_timing.py` is empty and would be the natural single home.

## Derived retirement age in months

`data_plots/retirement_timing.py::plot_derived_retirement_age_months` plots the
distribution of `corrected_age`, floored to full months, for fresh retirees
(`art_choice == 0` and previous year's `art_choice` in {1, 2, 3}), 2013–2020,
pooled over sex. Output: `output/plots/data/derived_retirement_age_months.{pdf,png}`.

Based on the cached `art_choice.pkl`, so before the pension-receipt check, the
`|age_diff| > 1` drop and the other sample filters. N = 1,585; about 10% lie
outside the plotted 55–72 window (mostly below 55).

Shares by year of age:

| Age   | <59 | 59  | 60  | 61  | 62  | 63   | 64   | 65   | 66  | 67+ |
|-------|-----|-----|-----|-----|-----|------|------|------|-----|-----|
| Share | 4.5% | 3.0% | 4.9% | 5.1% | 7.4% | 24.0% | 11.4% | 25.5% | 2.2% | 1.7% |

- Two clear spikes: exactly 63 (5.5% in the single month; long-insured early
  retirement) and 65 to 65y8m (the cohort-specific SRA for those born 1948–1954,
  which rises one month per cohort).
- Smaller bunching at exact 60 and 62.
- Only 14% retire in a month that is a full year of age. Monthly timing is real
  information that the integer-age model has to absorb through the SRA shift.
