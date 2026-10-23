# Monte Carlo: parameter recovery

We simulate data from the structural model with known ("true") params,
estimate the model on the simulated data, and check whether the estimates are
close to the true params. We start with an easy setup and add the features of
the full model one at a time.

Run from `src/`:

```
python estimation/monte_carlo/run_fake_data_est.py --informed <coding> --sample <sample> --job-offer <coding>
```

`--informed`: `all`, `data_coding`, `unobserved`, `true`, `informed_from_63`.
`--sample`: `full`, `data_ratio`. `--job-offer`: `working_only`, `data_coding`.
Further options: `--seed`, `--n-agents` (default 1,000), `--true-params`
(default `alg1_sparse_inf`), `--disutil-shift`, `--load-model`,
`--load-solution`.

## Runs

Common to all runs:

- Easy model (`specify_easy_model.py`): low educated men only, everybody starts
  with SRA 67 and keeps it, no SRA uncertainty. Built from the model functions
  of the full model.
- True params: `alg1_sparse_inf` with the disutility of full-time work and
  unemployment (good and bad health) shifted by −0.9375
  (`calibrate_disutil_shift.py`), so that 27.8% of fresh retirements are at 63
  (data: 27.5%; unshifted: 76%). The differences between the disutilities, and
  with them the labor shares of working-age individuals, stay the same.
- Estimation: mean negative log-likelihood, L-BFGS-B from perturbed start
  values (25% noise), standard errors from the outer product of the scores,
  clustered by individual.
- 1,000 simulated individuals, seed 123. 13 estimated params (disutilities,
  disability, job finding); all other params fixed at the truth.
- Health coded as in the real data: bad and disabled unobserved, fresh retirees
  before 63 observed as disabled. Already retired observations are dropped.
- Runs with the same informed setup use the same simulated panel: runs 1, 2, 8;
  runs 3, 4, 5, 7, 9; run 6 differs only from age 63 on.

**Run 1: baseline**
- Everybody informed, and informed observed.
- Full simulated panel: 33,555 observations, 875 fresh retirements.
- Job offer observed only for individuals working.

**Run 2: data ratio of fresh retirements**
- As run 1, but panels censored at random such that fresh retirements per
  working-life observation match the data (0.019 instead of 0.026).
- 26,590 observations, 516 fresh retirements.
- The censoring is independent of choices (no choice-based sampling).

**Run 3: uninformed individuals, real-data informed coding**
- Uninformed individuals as in the full model: informed is a type in the
  solution and evolves with `informed_transition` in the simulation (hazard, and
  informed after retiring).
- Informed coded as in the real data: retirement from 63 on = informed, all
  other observations (including disability pensioners) unobserved and weighted
  with `probability_informed`.
- 449 of 735 retirees from 63 on are uninformed in the simulation but coded
  informed.

**Run 4: run 3 with data ratio**
- Real-data informed coding as in run 3, panels censored as in run 2.
- 27,432 observations, 533 fresh retirements.

**Run 5: informed unobserved**
- Same simulated panel as run 3.
- Informed unobserved for every observation, including retirees from 63 on.

**Run 6: correct informed labels**
- Simulation changed: everybody becomes informed by 63, so the real-data coding
  (retirees from 63 on = informed) is correct.
- `probability_informed` = 1 from 63 on.

**Run 7: informed observed (benchmark)**
- Same simulated panel as run 3.
- True informed state observed for every observation.

**Run 8: run 1 with real-data job offer coding**
- As run 1, but the job offer of individuals who worked last period and stop
  working is observed, as in the real data: 0 if fired, 1 otherwise.
- 1,872 individuals stop working, 644 of them fired (34%; data 35%).

**Run 9: run 3 with real-data job offer coding**
- Real-data coding of both informed (as run 3) and job offers (as run 8).
- 1,883 individuals stop working, 690 of them fired.

## Results

Estimate (t vs. truth). Bold: |t| ≥ 2.

| parameter | true | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 | run 7 | run 8 | run 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| disutil_ft_work_good | 0.616 | 0.614 (-0.09) | 0.611 (-0.14) | 0.616 (-0.01) | 0.631 (0.62) | 0.622 (0.33) | 0.619 (0.10) | 0.612 (-0.14) | 0.616 (-0.01) | 0.624 (0.48) |
| disutil_ft_work_bad | 0.867 | 0.870 (0.10) | 0.877 (0.26) | 0.865 (-0.07) | 0.877 (0.36) | 0.847 (-0.98) | 0.838 (-1.01) | 0.843 (-0.82) | 0.876 (0.36) | 0.867 (-0.03) |
| disutil_unemployed_good | 0.055 | 0.062 (0.25) | 0.075 (0.52) | 0.070 (0.44) | 0.100 (1.90) | 0.079 (1.32) | 0.085 (1.00) | 0.066 (0.39) | 0.061 (0.19) | 0.077 (1.38) |
| disutil_unemployed_bad | 0.105 | 0.128 (0.79) | 0.132 (0.69) | 0.119 (0.41) | 0.143 (1.57) | 0.105 (-0.02) | 0.105 (-0.00) | 0.100 (-0.18) | 0.128 (0.84) | 0.119 (0.64) |
| disutil_partner_retired | -0.513 | -0.504 (0.32) | -0.491 (0.55) | -0.506 (0.21) | -0.485 (1.13) | -0.509 (0.19) | -0.525 (-0.36) | -0.540 (-0.87) | -0.499 (0.49) | -0.498 (0.89) |
| disability_const | -4.143 | -4.027 (0.64) | -4.312 (-0.73) | -3.931 (1.20) | -4.255 (-0.50) | -3.954 (1.07) | -3.959 (1.05) | -4.002 (0.81) | -4.063 (0.44) | -3.942 (1.15) |
| disability_above_50 | 0.171 | 0.025 (-0.45) | 0.230 (0.15) | 0.013 (-0.49) | 0.190 (0.05) | 0.000 (-0.53) | 0.000 (-0.53) | 0.014 (-0.49) | 0.055 (-0.36) | 0.000 (-0.54) |
| disability_above_55 | 0.962 | 0.972 (0.03) | 0.916 (-0.12) | 0.832 (-0.43) | 0.761 (-0.53) | 0.933 (-0.11) | 0.957 (-0.02) | 0.959 (-0.01) | 1.002 (0.13) | 0.852 (-0.36) |
| disability_above_60 | 0.961 | 0.704 (-1.13) | 0.459 (-1.56) | **0.431 (-2.47)** | **0.230 (-2.35)** | 0.681 (-1.94) | 0.765 (-0.90) | 0.732 (-1.16) | 0.738 (-0.99) | **0.465 (-2.33)** |
| job_finding_const | -0.777 | -0.766 (0.30) | -0.773 (0.09) | -0.733 (1.23) | -0.714 (1.66) | -0.728 (1.41) | -0.712 (1.81) | -0.729 (1.36) | -0.762 (0.51) | -0.734 (1.63) |
| job_finding_above_50 | -0.077 | -0.130 (-0.58) | -0.224 (-1.25) | -0.192 (-1.53) | -0.215 (-1.31) | -0.182 (-1.20) | -0.170 (-1.07) | -0.131 (-0.63) | -0.147 (-1.07) | -0.187 (-1.71) |
| job_finding_above_55 | -0.253 | -0.112 (1.27) | 0.000 (1.80) | -0.061 (1.95) | -0.051 (1.50) | -0.072 (1.66) | -0.084 (1.56) | -0.155 (0.91) | -0.100 (1.77) | -0.087 (1.90) |
| job_finding_above_60 | -0.231 | **-0.457 (-2.23)** | **-0.512 (-2.20)** | -0.312 (-0.79) | -0.249 (-0.15) | -0.336 (-1.04) | -0.320 (-0.85) | -0.295 (-0.84) | **-0.432 (-2.31)** | -0.286 (-0.62) |
| **identified** | | (✓) | (✓) | (✓) | (✓) | ✓ | ✓ | ✓ | (✓) | (✓) |

Identified: ✓ all |t| < 2; (✓) one param with 2 ≤ |t| < 2.5, as expected by
chance with 13 params; ✗ otherwise. In all runs, the criterion at the true
params is lower than at the start values, and the criterion at the estimate is
lower than at the true params, so the optimizer converged.

Implied disability probability from 60 (true 0.114): run 1: 0.089, 2: 0.063,
3: 0.066, 4: 0.044, 5: 0.088, 6: 0.096, 7: 0.091, 8: 0.094, 9: 0.068.

## Findings

**1. The pipeline works.** In all nine runs the true params fit better than the
perturbed start values, and the estimate fits slightly better than the true
params. So the likelihood, the data coding, and the optimizer are consistent:
the remaining distance to the truth is sampling noise or identification, not a
bug. No run fails the identification flag; every miss is a single parameter
with |t| between 2 and 2.5.

**2. The disutilities are well identified.** They are recovered in every run
(|t| ≤ 1.9) with standard errors of about 0.02–0.04, also with uninformed
individuals, the real-data informed coding, and the censored samples. This
needed the realistic true params: with the unshifted params, 76% of retirements
were at 63, the disutility of work and of unemployment were almost perfectly
correlated (0.96 and 0.94), and their common level was only weakly identified
(standard errors about twice as large, three params with |t| > 2). With
retirement spread over more ages, the correlation falls to 0.66–0.92 and the
level is pinned down. For the real estimation this means that the level of the
disutilities is identified from the timing of retirement, and their differences
from the choice between work and unemployment.

**3. Disability from 60 is the weak spot.** The age 60 step is too low in every
run (estimates 0.23–0.77 vs. 0.96), so the implied disability probability from
60 is too low (4.4–9.6% vs. 11.4%). Disability is only observed for fresh
retirees before 63; everybody else in bad health is unobserved and enters as a
mixture of bad health and disability. The age 60 step therefore rests on the
retirements at 60–62, 44–49 per full sample, which explains the large standard
errors (0.14–0.32). The step also governs disability from 63 on, where it is
never observed but still changes the value of retiring through the disability
pension. I checked the weighting of unobserved health against the
health transition and found no mismatch, so with one seed this can still be
noise.

**4. The real-data informed coding is the one that hurts.** In the simulation,
61% of the individuals retiring from 63 on are uninformed when they decide, but
the real-data coding labels all of them informed. On the same simulated panel:

| informed coding | run | disability step 60 | disability probability from 60 |
|---|---|---|---|
| observed (benchmark) | 7 | 0.73 | 9.1% |
| unobserved everywhere | 5 | 0.68 | 8.8% |
| real-data coding | 3 | 0.43 | 6.6% |
| real-data coding + job offers | 9 | 0.47 | 6.8% |
| real-data coding, censored | 4 | 0.23 | 4.4% |

With correct labels (run 6, everybody informed by 63) the real-data coding is
as good as the benchmark (9.6%). So the problem is not the coding rule itself,
but that it is wrong for uninformed retirees. A plausible channel: uninformed
individuals expect a different early retirement penalty, so more of them retire
through the disability pension (68 disabled fresh retirees from 63 on in the
panel with uninformed individuals, 46 in the all-informed panel). From 63 on,
disability is unobserved, and the retirement decisions of these individuals,
labeled informed, are partly explained by the age 60 step, which also sets the
disability probability from 63 on. The disutilities are not affected. Treating
informed as unobserved for everybody (run 5) loses almost nothing relative to
the benchmark on this panel, which makes it a candidate for the real data.

**5. Coding job offers as in the real data adds information on job finding.**
Revealing the job offer of everybody who stops working (runs 8, 9; 34% of them
fired, 35% in the data) changes no estimate noticeably and lowers the standard
errors of the job finding params by 10–30% (age 50 step: 0.091 → 0.065). It
does not change the disability problem of the informed coding (run 9 vs. run 3).

**6. The age steps of job finding are only jointly identified.** The age 55
step is at or near its upper bound of 0 in most runs, and the age 60 step is
too negative to compensate (|t| up to 2.3 in runs 1, 2, 8). The implied job
finding probabilities are within 3 percentage points of the truth at all ages.
Job finding is only informative for unemployed individuals, of whom there are
few above 55, so the split between the 55 and 60 steps is poorly determined,
while their combined effect is.

**7. Fewer observed retirements cost precision, not bias.** Censoring the panels
to the retirement ratio of the data (runs 2, 4) removes about 40% of the fresh
retirements. Standard errors grow by 20–40% (run 2 vs. 1), and the disability
step 60 moves further down, but there is no new systematic bias. This holds
because our censoring is independent of the choices; in the real data,
attrition and cohort composition may not be.

**Caveats.**

- All results are from one seed: runs 1, 2, 8 share one simulated panel, and
  runs 3, 4, 5, 7, 9 another. Differences between runs on the same panel are
  driven by the coding, differences across panels may be noise.
- The criterion levels are not comparable between informed codings. The
  likelihood uses `use_probability_of_observed_states=True`, so an observed
  informed state multiplies the likelihood by `probability_informed` (or one
  minus it). This does not depend on the estimated params and only shifts the
  level (0.826 at the truth in run 7, 0.270 in run 3, 0.254 in run 5).
- The standard errors from a single sample are noisy: on the same panel they
  differ between codings in ways that do not follow the information content
  (e.g. 0.018 for the disutilities in run 5, 0.028 in run 7). Coverage has to
  come from several seeds.

## Next steps

1. Several seeds for runs 1, 3, 5, 7, 9: bias, RMSE, and coverage, in
   particular for disability from 60 and the informed coding.
2. Sample size and age structure as in the real data (about 21,000
   observations of low educated men).
3. Open issue informed: the real data codes retirees from 63 on as informed,
   while in the model the retirement decision is made in the current informed
   state and individuals become informed only after retiring.
4. Open issue job offers: the docstring of `determine_observed_job_offers` says
   the "not fired = job offer" rule is excluded, but the code applies it.
5. Subjective SRA uncertainty and heterogeneous initial SRA.
6. Other types (high education, women) and the full model.
