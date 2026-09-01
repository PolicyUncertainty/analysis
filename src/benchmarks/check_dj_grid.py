# %%
"""Pin down a good druedahl_jorgensen (dj) assets_begin_of_period grid.

Ground truth is `fues` solved on production's real (non-uniform) assets_end_of_period
and experience grids -- fues needs no assets_begin_of_period grid at all, so it's
untainted by the thing being tested here. That one fues solve produces two things
this script uses:

  - `endog_grid_sample`: fues's own solved endogenous wealth grid, pooled across all
    periods and discrete states. This is dcegm's own answer to "where does resolving
    the policy function actually require density" -- dense near the credit
    constraint, sparser where consumption is smooth -- so it's used as the *shape*
    source for the dj assets_begin_of_period candidates in dj_candidates.py.
    Deliberately NOT derived from assets_end_of_period: begin-of-period wealth and
    end-of-period savings are different quantities, so borrowing endog_grid's
    placement is a property of the solved problem itself, not an assumption about a
    different grid.
  - `fues_sim_df`: fues simulated forward from real starting states/seed. This is
    the realistic distribution of (period, discrete state, wealth) combinations
    agents actually visit -- used to *weight* how much a candidate grid's error
    matters. A wealth region no simulated agent ever visits isn't worth resolving;
    one visited every period by a third of the population is.

For each candidate (see dj_candidates.CANDIDATE_SPECS), dj is solved on the same
assets_end_of_period/experience grids as the fues reference, then simulated with the
*same seed and initial states*. Comparing per-(period, agent) consumption and choice
against the fues reference directly measures what switching to that dj grid would
change in reported simulated moments -- errors are implicitly weighted by
visitation frequency, since more-visited states simply contribute more (period,
agent) observations to the comparison, and are additionally broken out by wealth
decile (of the fues reference's own simulated distribution) to surface whether
errors concentrate near the credit constraint specifically.

Caveat: choices/wealth feed into next period's state, so a candidate's simulated
path can diverge from the fues reference after any early mismatch (compounding, not
just local grid error). That's intentional here -- it's a direct closed-loop check
of whether reported simulated moments would change. `choice_mismatch_rate` flags
when a consumption-error number is confounded by a different discrete choice being
taken, since consumption under different choices isn't the same object.

No dcegm-branch bookkeeping here; this only cares about grid accuracy on whatever
dcegm version happens to be checked out.

Results: src/benchmarks/output_dj_grid_check/error_by_candidate.csv and
error_by_wealth_decile.csv. Run run_dj_grid_timing.py separately for the solve-cost
side of the same candidate sweep.
"""
import os
import pickle as pkl
from pathlib import Path

import jax
import pandas as pd

jax.config.update("jax_enable_x64", True)

from benchmarks.dj_candidates import CANDIDATE_SPECS, build_candidate_grid
from benchmarks.grids import pool_positive_finite
from model_code.specify_model import specify_model
from model_code.state_space.experience import define_experience_grid
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from set_paths import create_path_dict
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import generate_derived_and_data_derived_specs

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR = BENCHMARK_DIR / "output_dj_grid_check"

SPECS = {"sex_type": "all", "edu_type": "all", "seed": 123}

# %%
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

# Redirect the model-spec cache dcegm writes during setup to a benchmark-local
# folder, so this never collides with (or overwrites) the production model cache
# under output/intermediate_data/.
model_cache_dir = str(RESULTS_DIR / "model_cache") + "/"
os.makedirs(model_cache_dir, exist_ok=True)
bench_path_dict = dict(path_dict)
bench_path_dict["intermediate_data"] = model_cache_dir

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

production_assets_end_of_period_grid = create_end_of_period_assets()
production_experience_grid = define_experience_grid(specs)


def build_model(upper_envelope_method, assets_begin_of_period_grid=None):
    return specify_model(
        path_dict=bench_path_dict,
        specs=specs,
        subj_unc=True,
        custom_resolution_age=None,
        sim_specs={
            "announcement_age": None,
            "SRA_at_start": 67,
            "SRA_at_retirement": 67,
        },
        simulate_expectations=False,
        load_model=False,
        sex_type=SPECS["sex_type"],
        edu_type=SPECS["edu_type"],
        util_type=specs["util_type"],
        upper_envelope_method=upper_envelope_method,
        assets_end_of_period_grid=production_assets_end_of_period_grid,
        assets_begin_of_period_grid=assets_begin_of_period_grid,
        experience_grid=production_experience_grid,
    )


def simulate_model(model_solved, initial_states):
    sim_df = model_solved.simulate(states_initial=initial_states, seed=SPECS["seed"])
    sim_df = sim_df[sim_df["health"] != 3].copy()
    sim_df.reset_index(inplace=True)  # "period"/"agent" start out as a MultiIndex
    return sim_df


# %% Ground truth: fues
print("=== Solving fues reference (production grids) ===", flush=True)
fues_solved = build_model(upper_envelope_method="fues").solve(params)
jax.block_until_ready((fues_solved.value, fues_solved.policy))

endog_grid_sample = pool_positive_finite(fues_solved.endog_grid)
print(
    f"Pooled {endog_grid_sample.size} positive, finite endogenous grid points from "
    "the fues solve.",
    flush=True,
)

initial_states = generate_start_states_from_obs(
    path_dict=path_dict,
    params=params,
    inital_SRA=67,
    seed=SPECS["seed"],
    model_class=fues_solved,
    only_informed=False,
)

print("Simulating fues reference ...", flush=True)
fues_sim_df = simulate_model(fues_solved, initial_states)
# Pooled across periods/agents -- decile 0 is the bottom of the simulated wealth
# distribution, i.e. closest to the credit constraint on average.
fues_sim_df["wealth_decile"] = pd.qcut(
    fues_sim_df["assets_begin_of_period"], 10, labels=False, duplicates="drop"
)

# %% Sweep dj candidates
# Results are appended to disk after each candidate (not batched to the end) so a
# crash partway through the sweep doesn't lose the candidates that already
# finished. A failing candidate is logged and skipped rather than aborting the rest
# of the sweep.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
candidate_csv_path = OUTPUT_DIR / "error_by_candidate.csv"
decile_csv_path = OUTPUT_DIR / "error_by_wealth_decile.csv"
# Start each run from a clean file -- the header-on-first-write logic below assumes
# these don't already exist from a previous run.
candidate_csv_path.unlink(missing_ok=True)
decile_csv_path.unlink(missing_ok=True)

candidate_rows = []
decile_rows = []
failed_candidates = []
for cand in CANDIDATE_SPECS:
    print(f"\n=== Candidate: {cand['name']} ===", flush=True)
    grid = build_candidate_grid(cand, endog_grid_sample)

    try:
        dj_solved = build_model(
            upper_envelope_method="druedahl_jorgensen",
            assets_begin_of_period_grid=grid,
        ).solve(params)
        jax.block_until_ready((dj_solved.value, dj_solved.policy))

        print(f"Simulating '{cand['name']}' ...", flush=True)
        dj_sim_df = simulate_model(dj_solved, initial_states)
        del dj_solved
    except Exception as exc:  # noqa: BLE001 -- keep sweeping past a bad candidate
        print(f"FAILED '{cand['name']}' (n_points={len(grid)}): {exc}", flush=True)
        failed_candidates.append(cand["name"])
        continue

    paired = fues_sim_df[
        ["period", "agent", "consumption", "choice", "wealth_decile"]
    ].merge(
        dj_sim_df[["period", "agent", "consumption", "choice"]],
        on=["period", "agent"],
        suffixes=("_fues", "_dj"),
    )
    paired["abs_error"] = (paired["consumption_dj"] - paired["consumption_fues"]).abs()
    paired["choice_mismatch"] = paired["choice_dj"] != paired["choice_fues"]

    candidate_row = {
        "candidate": cand["name"],
        "n_assets_begin_of_period": len(grid),
        "mean_abs_consumption_error": paired["abs_error"].mean(),
        "max_abs_consumption_error": paired["abs_error"].max(),
        "choice_mismatch_rate": paired["choice_mismatch"].mean(),
    }
    candidate_rows.append(candidate_row)
    pd.DataFrame([candidate_row]).to_csv(
        candidate_csv_path,
        mode="a",
        header=not candidate_csv_path.exists(),
        index=False,
    )

    decile_means = paired.groupby("wealth_decile")["abs_error"].mean()
    new_decile_rows = [
        {
            "candidate": cand["name"],
            "wealth_decile": decile,
            "mean_abs_consumption_error": mean_abs_error,
        }
        for decile, mean_abs_error in decile_means.items()
    ]
    decile_rows.extend(new_decile_rows)
    pd.DataFrame(new_decile_rows).to_csv(
        decile_csv_path, mode="a", header=not decile_csv_path.exists(), index=False
    )
    print(f"Saved '{cand['name']}' to {OUTPUT_DIR}", flush=True)

# %%
if failed_candidates:
    print(f"\nFailed candidates (skipped): {failed_candidates}")

print("\n=== Error vs. fues reference, by candidate ===")
print(pd.DataFrame(candidate_rows).to_string(index=False))

decile_df = pd.DataFrame(decile_rows).pivot(
    index="wealth_decile", columns="candidate", values="mean_abs_consumption_error"
)
print("\n=== Mean abs consumption error by wealth decile (0 = poorest) ===")
print(decile_df.to_string())

print(f"\nSaved incrementally to: {OUTPUT_DIR}")
