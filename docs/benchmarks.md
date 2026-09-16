# Benchmarks

Summary of the two benchmark investigations that informed the model's grids. The
scripts themselves are not kept in the repo; this captures their purpose, method,
findings, and how to redo them if needed.

---

## 1. Experience-grid discontinuities

**Purpose.** Locate the value-function jump along the experience axis caused by the
very-long-insured (VLI) pension, to decide where the experience grid needs dense
nodes. Motivates the design in `docs/experience_grid.md`.

**Setup.** Single type (low-education men — men accrue experience in whole years),
SRA fixed at 67, no subjective uncertainty (the sharpest possible jump). Solved on a
dense real-year experience grid (1-year spacing 30–60y, 2-year filler 0–28y), with
an attainability mask so nodes beyond a period's cap are excluded from the jump
search. One asset slice, one fixed discrete state. The dense grid is injected by
overriding `experience_grid_working_by_sex_period` / `experience_grid_cap_by_period`
in the specs.

**The mechanism.** The VLI pension pays deduction-free early retirement to workers
who reach the ~42-year credited threshold *and* retire within 2 years of the SRA, so
the retirement payoff jumps at the threshold. A still-working man can *reach* the
threshold by working more full-time years, so the jump appears at lower experience
and slides down as age falls.

**Key finding.** The reachability jump in the full-time (choice 3) value function
travels backwards **continuously**, not confined to a 2-year window: its location
sits almost exactly on `experience = threshold − (65 − age)`, and its size grows as
age approaches the retirement window.

| age | jump location (exp-yrs) | raw ΔV (1y cell) |
|---|---|---|
| 55 | 31.5 | 0.38 |
| 60 | 36.5 | 0.45 |
| 62 | 38.5 | 0.54 |
| 63 | 39.5 | 0.71 |
| **64** | **40.5** | **0.93** (largest) |
| 65 | 40.5 | 0.64 |

Below ~age 55 the true location drifts below 30 (out of the tested band), so that
range is a coverage limit, not evidence of "no jump." The retirement (choice 0)
value shows the same clean kink at the threshold for retiring at SRA−2.

**Implication for the production grid.** Because the jump's location is
**age-dependent** (slides down linearly with age), no single *fixed* bracket
resolves it at every age. This is why the production grid places the dense VLI
bracket per period at each sex's threshold and spreads the remaining resolution
across the attainable range (see `docs/experience_grid.md`).

**Caveats.** Fixed SRA and no belief uncertainty (a deliberately sharp best case;
the estimated model averages over SRA beliefs and smooths this). Solving *on* a
1-year grid smears the sharp kink across ~2 cells, so ΔV magnitudes are comparable
only to each other.

---

## 2. Druedahl–Jørgensen `assets_begin_of_period` grid selection

**Purpose.** Choose a good begin-of-period wealth grid for the Druedahl–Jørgensen
(DJ) upper-envelope method — dense enough near the credit constraint to be accurate,
small enough to be cheap.

**Method.**
- **Ground truth:** a `fues` solve on production's real `assets_end_of_period` /
  experience grids. `fues` needs no begin-of-period grid, so it is untainted by the
  quantity under test. It yields (a) its own solved endogenous wealth grid — dcegm's
  own answer to where density is needed — used as the *shape source* for candidates,
  and (b) a forward simulation, used to *weight* errors by how often agents actually
  visit each wealth region.
- **Candidates** (two shape families, built from a wealth sample, not from
  `assets_end_of_period`): `quantile` (evenly spaced quantiles — self-tuning to where
  mass is) and `power` (spacing `(i/(n−1))**power`, concentrating points near the
  credit constraint), at 10/15/20/28/40 points.
- **Accuracy:** solve each candidate with DJ on the same grids, simulate with the
  same seed/initial states, compare per-(period, agent) consumption and choice to the
  fues reference (errors implicitly weighted by visitation; broken out by wealth
  decile; a choice-mismatch rate flags where consumption errors are confounded by a
  different discrete choice). This is a closed-loop check of whether reported
  simulated moments would change.
- **Cost:** solve time and peak GPU memory over the identical candidate set (solve
  only; peak memory is a contaminated high-water mark, useful only as a rough
  ceiling).

Read accuracy and cost together as one accuracy-vs-cost picture. The sweep was run
on an H100 via SLURM (~9 min accuracy + ~5 min timing).

**Status.** No results were committed, so there is no recorded winning candidate —
rerun the sweep to pick one if the DJ method is (re)adopted.

---

## Reproducing

Both were standalone scripts under `src/benchmarks/` (removed). To redo:

- **Discontinuities:** build a dense real-year experience grid, inject it via the
  `experience_grid_working_by_sex_period` / `experience_grid_cap_by_period` spec
  keys, solve low-education men at fixed SRA with `subj_unc=False`, and scan the
  choice-specific value function for the largest cell-to-cell change per age over
  attainable nodes.
- **DJ grid:** solve a `fues` reference on the production grids, pool its
  `endog_grid` into a positive wealth sample, build quantile/power candidate grids
  from it, and compare DJ solves (accuracy via closed-loop simulation, cost via
  solve time/memory) against the fues reference.
