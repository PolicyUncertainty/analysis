# Experience-grid discontinuities and how they travel backwards

**Script:** `src/benchmarks/inspect_experience_discontinuities.py`
**Outputs:** `src/benchmarks/output_experience_discontinuities/` (plots + `summary.txt`)
**Type inspected:** low-education men, SRA fixed at 67, no subjective uncertainty.

**Correction (this version).** An earlier version of this report solved on a grid
built the way production builds it: one fixed node *shape* (fractions of the
*overall* max, 59y) rescaled per period by that period's own real-year cap. That
compresses a fixed "dense band" onto the wrong real years at every age below 63
(the cap only equals 59 from age 63 on; before that it's smaller and grows with
age). The earlier report's central claim -- "the reachability jump washes out
within ~2 years of the window; ages 50-62 show no interior jump" -- was an
**artifact of that compression hiding the jump's true (lower) location at younger
ages, not a real economic finding.** This version fixes the grid construction and
overturns that claim: see "Corrected finding" below.

## Question

The very-long-insured (VLI) pension pays a **deduction-free** early retirement to
workers who reach the ~42 model-year credited threshold *and* retire within 2 years
of the SRA. That makes the retirement payoff jump at the threshold. Does the solved
**value function** inherit that jump along the experience dimension, and — since a
still-working man can *reach* the threshold by working more full-time years — do
extra jumps appear at lower experience and **travel backwards** as we move to
younger ages?

## Grid construction

Each period's row is built by **clipping absolute real-year targets to that
period's own attainable cap** — no rescaling. Every attainable target year keeps
its exact requested real-year location at every age; only years beyond a period's
own cap (genuinely unattainable at that age, e.g. 50 years of experience at age 50)
are collapsed into a tiny strictly-increasing sequence just below the cap (needed
only so the row stays valid for interpolation) and are excluded from the
discontinuity search via a `genuine` attainability mask.

- **1-year spacing, 30–60y** (`FINE_YEARS`) — brackets the ~42y threshold and the
  whole range the reachability jump can travel through for ages 50–65.
- **2-year spacing, 0–28y** (`COARSE_YEARS`) — filler for the rest of the range.
- 46 nodes total. Verified: at age 62 (cap 48y) all 8 fine nodes in [37, 44] land
  exactly on 37, 38, ..., 44 (previously, under the rescaled-shape grid, only 5
  compressed/mislabelled points fell in that window, topping out at 40.7 instead
  of 44).

## Method / design decisions (unchanged from before)

- **Model:** single type (men, low edu). Men can only work full time, so
  experience moves in whole years.
- **Fixed, known SRA (67), `subj_unc=False`** — sharpest possible jump.
- **"From age 50 onwards":** full backward induction (`start_age` unchanged),
  inspecting ages ≥ 50 only. Backward induction makes the age-50+ solution
  independent of earlier ages, so this needs no period truncation.
- We read the *choice-specific* value of continuing full time (choice 3) and, in
  the window, the value of retiring (choice 0). A discontinuity search looks for
  the largest cell-to-cell value change within the fine band, restricted at each
  age to `genuine` (attainable) nodes.

## Corrected finding: the reachability jump travels backwards continuously, not just for 2 years

`jump_location_by_age.png` and `fulltime_slope_by_age.png` now show a clean,
continuous relationship for **ages 55–65**: the largest interior jump in the
full-time (choice 3) value sits almost exactly on the theoretical line
`experience = threshold - (65 - age)` (42 at the window, one year lower per year
further from it), and its *size* grows monotonically as age approaches the window:

| age | jump location (exp-years) | raw ΔV (1y cell) | at band edge? |
|---|---|---|---|
| 55 | 31.5 | 0.384 | no |
| 58 | 34.5 | 0.413 | no |
| 60 | 36.5 | 0.446 | no |
| 62 | 38.5 | 0.541 | no |
| 63 | 39.5 | 0.707 | no |
| **64** | **40.5** | **0.932** | **no (largest overall)** |
| 65 | 40.5 | 0.640 | no |
| 66–67 | ~30–31 | 0.33 | edge (no real jump; in-window without needing to "reach") |
| 50–54 | ~30.5 | 0.31–0.38 | **edge** — true location (`threshold-(65-age)` = 27–31) falls at or below the fine band's start, so the detector can't resolve it there, not because it isn't present |

So the jump is present and *growing* continuously from at least age 55 through
64, not confined to a 2-year window before age 65 as the earlier (compressed-grid)
report claimed. Below age 55 the true location drifts below 30, out of the fine
band's reach, so this run can't confirm or rule out a jump there — that's a grid
coverage limit, not a finding of "no jump."

**Why the previous report missed this:** on the old rescaled-shape grid, ages
50–62 had their dense nodes compressed into real years well below the jump's true
location (e.g. at age 60, cap = 46 but the dense band's fractions were calibrated
to land on 25–46 only at cap = 59, so they actually landed around 19–37) — the
detector simply never had a node near the true jump location (36.5) to see it.

**A second, expected effect of the coarser (1-year) solve grid:** the kink is
visibly *smeared* across roughly two fine cells (e.g. at age 64, elevated slope at
both the 39–40 and 40–41 cells, 0.44 and 0.93 raw ΔV vs. a smooth background of
~0.30–0.33) rather than resolved in one sharp cell as it was in an earlier,
0.2-year-spaced version of this grid. This is expected: the model is *solved* on
this coarser grid, so the backward induction itself, not just the plotting, blurs
a sharp kink across whatever cells bracket it. Finer spacing right around the
threshold would sharpen this further; that's a separate axis from the 1y/2y
real-year-fidelity fix above.

**Retirement value** (`retirement_value_window.png`): still shows the same clean
kink at the threshold for retiring at 65 (`retirement_age_difference = 2`), and
essentially none for 64 (outside the window) or 66/67 (already at/near full
pension without VLI) — unaffected by the grid fix, since this plot only ever
needed nodes bracketing the threshold, which both grid versions had.

## Implication for the production experience grid

The evidence now supports a **wider** bracket than either previous
recommendation, because the reachability jump is not confined to 2 years before
the window:

- A single fixed bracket near the threshold (e.g. `{40, 41, 41.5, 42}`) would
  resolve the sharp, large jumps at ages 63–65 (0.6–0.9 raw ΔV) but miss the
  smaller, still-real ones at ages 55–62 (0.38–0.54), which sit progressively
  lower (down to ~31.5 at age 55).
- Because the jump's location is *age-dependent* (it slides down linearly with
  age), no single *fixed* bracket resolves it at every age — this is inherent to
  the reachability mechanism, not a grid-density problem. A genuinely
  age-dependent placement (bracket nodes at `threshold - (65 - age)` per period,
  on top of the existing per-period real-year grid) would be needed to resolve it
  at every age; a fixed bracket can only target the ages that matter most
  (typically the ones closest to the window, where the jump is largest).
  Confirming this trade-off further from age 50 requires extending `FINE_YEARS`
  below 30 to see whether the effect stays material into the 40s (see caveats).

## Caveats

- Fixed SRA and no belief uncertainty — a deliberately sharp, best-case setting;
  the estimated model averages over SRA beliefs and would smooth this further.
- `FINE_YEARS` starts at 30, so ages 50–54 (true jump location 27–31) are only
  partially resolved; extending the fine band down to ~20 would be needed to
  trace the jump (or confirm it has become negligible) below age 55.
- Single asset slice (begin-of-period assets = 25) and one fixed discrete state
  (good health, working partner, job offer, informed).
- This run solves the model *on* the coarser 1-year grid, which smears the sharp
  near-window kink across ~2 cells relative to a finer solve grid — the raw ΔV
  numbers above are not directly comparable to a finer grid's cell-level ΔV, only
  to each other (same cell width throughout).

How to reproduce:
```
cd src && python3 benchmarks/inspect_experience_discontinuities.py
# delete output_experience_discontinuities/sol_cache.pkl to force a re-solve
```
