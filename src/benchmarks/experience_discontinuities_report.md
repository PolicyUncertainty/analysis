# Experience-grid discontinuities and how they travel backwards

**Script:** `src/benchmarks/inspect_experience_discontinuities.py`
**Outputs:** `src/benchmarks/output_experience_discontinuities/` (4 plots + `summary.txt`)
**Type inspected:** low-education men, SRA fixed at 67, no subjective uncertainty.

## Question

The very-long-insured (VLI) pension pays a **deduction-free** early retirement to
workers who reach the ~42 model-year credited threshold *and* retire within 2 years
of the SRA. That makes the retirement payoff jump at the threshold. Does the solved
**value function** inherit that jump along the experience dimension, and — since a
still-working man can *reach* the threshold by working more full-time years — do
extra jumps appear at lower experience and **travel backwards** as we move to
younger ages?

## Method / design decisions

- **Model:** single type (men, low edu). Men already can only work full time (the
  model forbids part-time for `sex==0`), so experience moves in whole years — the
  clean case the task asked for. No custom choice set needed.
- **Fixed, known SRA (67), `subj_unc=False`.** Removing SRA-belief uncertainty
  gives the *sharpest possible* jump; the real estimated model, which averages over
  many possible SRAs, can only be smoother. This also shrinks the state space
  (~173k states) so the solve runs locally in a few minutes.
- **"From age 50 onwards":** we solve the full backward induction (start_age
  unchanged, so all age-dependent economics stay exact) and only *inspect* ages
  ≥ 50. Backward induction makes the age-50+ solution identical to the full model's,
  so this needs no fragile period truncation.
- **Grid:** dense at integer model-years 25–45 (the task's 30–45 band, extended a
  little lower to follow the travelling jump), plus coarse anchors so the normalized
  grid still spans [0,1].
- **What we read:** the choice-specific value of *continuing full time* (the object
  that carries the travelling jump) and, in the window, the value of *retiring*.
  A true discontinuity shows up as a large value change across one grid cell
  (linear interpolation cannot represent a jump inside a cell).
- **Frame:** we plot against **real experience-years** (`construct_experience_years`
  = normalized × period scale), the frame in which the jump travels linearly. Note
  the experience state is stored *normalized* by a period-dependent scale
  (`max_exps_period_working`) that **jumps from 48 to 59 exactly at age 63**
  (`min_period_very_long_insured`); this matters for the grid (see below).

## Findings

### 1. Yes — the value function is discontinuous along experience, near the threshold

- **Retirement value** (`retirement_value_window.png`): a clear upward kink at the
  threshold (~41–42 yr) for **in-window** retirement ages (65, 66). Retiring at 64
  (3 years before the SRA, outside the 2-year window) and at 67 (the SRA itself,
  full pension regardless) show **no** VLI kink — exactly as the rule predicts.
- **Full-time (continuation) value** (`fulltime_value_by_age.png`,
  `fulltime_slope_by_age.png`): a sharp kink appears **one to two years below the
  threshold**, at ~40–41 real-years. The slope plot makes it unmistakable: a spike
  in `dV/d(experience)` to ~0.9 (age 64) and ~0.64 (age 65) right at 40–41, against
  a smooth background slope of ~0.3–0.4. This is the reachability jump: from just
  below the threshold, one more full-time year reaches VLI eligibility at the window.

### 2. The jumps do **not** travel far backwards — they wash out within ~2 years of the window

This is the central result (`jump_location_by_age.png`, `summary.txt`):

| ages | largest in-band experience jump | size | resolved by grid? |
|---|---|---|---|
| 63–65 | ~39.5–40.5 real-years (just below threshold) | 0.6–0.9 | **yes** (sharp, interior) |
| 66–67 | weaker / at the SRA (no early-VLI benefit for full time) | ~0.3 | marginal |
| 50–62 | none — detector only finds the band edge | ~0.25–0.33 | **no interior jump** |

So the sharp VLI discontinuity in the *continuation* value is confined to roughly
**ages 63–65** — within about two years of the window. By age 62 it is already
gone: the full-time value is smooth in experience (flat slope in
`fulltime_slope_by_age.png`), and the "largest jump" the detector reports for ages
50–62 is just the generic low-experience curvature at the band edge, not a
discontinuity.

**Why they wash out** (in the experience dimension, going backwards):

- **Taste shocks** — the value is a `logsumexp` over choices, so the "switch to
  work-full-time-to-reach-VLI" kink becomes a soft logit transition.
- **Discounting + horizon** — the further from the window, the smaller and more
  smeared the option value of just barely reaching the threshold.
- *Not* the income-shock quadrature: that smooths the **assets** dimension;
  experience evolves deterministically given the choice, so the shock never blurs
  the experience location of a jump.

(With the real model's SRA-belief uncertainty, this washing-out would be even
faster, because the value additionally averages over many candidate windows.)

### 3. The period-dependent normalization degrades the grid's reach below age 63

Because the normalization scale jumps 48→59 at age 63, a **fixed normalized grid**
images onto a *shrinking* real-year window as age falls: the dense normalized band
covers real-years ≈ `[25,45] × scale(age)/59`, i.e. [25,45] at age ≥ 63 but only
~[15,28] at age 50. Meanwhile the (would-be) jump travels down linearly in real
years. The two do not track each other, so below the constant-scale region the grid
cannot cleanly resolve the travelling jump — but since the jump has already washed
out there, this costs little in practice.

## Implication for the production experience grid

The sharp, grid-relevant experience discontinuities are **local**: they sit in the
band **[threshold − 2, threshold]** (retirement jump at the threshold ≈ 42;
full-time reachability jumps at ≈ 40–41) and only for ages within ~2 years of the
VLI window. This is concrete evidence for how wide the per-sex bracket should be:

- The current **two-node bracket** `{threshold, threshold − 0.5}` (= men-years
  {42, 41.5}) captures the **retirement** jump and the age-65 full-time jump (~41),
  but the strongest full-time jump — age 64 at ~40–40.5 — falls **just below** the
  bracket, into the wide coarse cell, and is not resolved by the production grid.
- A bracket spanning **[threshold − 2, threshold]** (≈ nodes 42, 41.5, 41, 40)
  would resolve all of the sharp jumps observed here.
- Going **wider** than ~2 years is unnecessary: the jumps demonstrably wash out
  beyond the immediate neighbourhood of the window, so a dense lattice reaching
  many years below the threshold would spend nodes where the value is already
  smooth. Going **narrower** (two nodes) misses the age-64 full-time jump.

Net: the evidence supports a **modest widening of the per-sex bracket to about
`threshold − 2`** (roughly four nodes), rather than either the tight two-node
bracket or the far-reaching lattice considered earlier.

## Caveats

- Fixed SRA and no belief uncertainty — a deliberately sharp, best-case setting; the
  estimated model is smoother, so the wash-out is if anything faster.
- Single asset slice (begin-of-period assets = 25) and one fixed discrete state
  (good health, working partner, job offer, informed). Spot checks at other assets
  behave the same qualitatively; the experience jump is an experience-dimension
  feature largely independent of the asset level.
- Jump magnitudes are in value (utility) units; ~0.6–0.9 against value levels of
  ~40, i.e. a couple of percent — modest but not negligible.
```
How to reproduce:
  cd src && python3 benchmarks/inspect_experience_discontinuities.py
  # delete output_experience_discontinuities/sol_cache.pkl to force a re-solve
```
