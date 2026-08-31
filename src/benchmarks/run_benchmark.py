# %%
"""dcegm engine benchmark: solve + simulate our retirement model once, for one case.

Usage
-----
Each run only ever executes ONE case (dcegm branch x upper-envelope method), because
switching the dcegm branch is a `git checkout` in `submodules/dcegm` -- it can't
happen mid-process, only between runs of this script. Two ways to pick the case:

  * Interactive/manual: in CASES below, comment out all entries except the one
    matching whatever branch you've checked out, then `python run_benchmark.py`.
  * Non-interactive/batch: pass `--case <name>` (or set env var BENCHMARK_CASE),
    which overrides CASES. See hu_server_benchmark.sh, which loops over all four,
    checking out the right dcegm branch before each.

Results are pickled to `src/benchmarks/results/<case_name>.pkl`. Once all four cases
have been run, `python make_table.py` builds the comparison table.
"""
import argparse
import os
import pickle as pkl
import time
from pathlib import Path

import jax
import pandas as pd

jax.config.update("jax_enable_x64", True)

from benchmarks.grids import (
    build_assets_begin_of_period_grid,
    build_assets_end_of_period_grid,
    build_experience_grid,
)
from benchmarks.provenance import get_dcegm_git_info
from model_code.specify_model import specify_model
from set_paths import create_path_dict
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import generate_derived_and_data_derived_specs

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"

# =====================================================================================
# Cases
# =====================================================================================
ALL_CASES = {
    "main_fues": {
        "name": "main_fues",
        "expected_dcegm_branch": "main",
        "upper_envelope_method": "fues",
    },
    "main_dj": {
        "name": "main_dj",
        "expected_dcegm_branch": "main",
        "upper_envelope_method": "druedahl_jorgensen",
    },
    "new_dj": {
        "name": "new_dj",
        "expected_dcegm_branch": "remove_endog",
        "upper_envelope_method": "druedahl_jorgensen",
    },
    "new_fues": {
        "name": "new_fues",
        "expected_dcegm_branch": "remove_endog",
        "upper_envelope_method": "fues",
    },
}

# Interactive/manual selection -- comment out all but one before running. Ignored
# if --case/BENCHMARK_CASE is set (see module docstring).
CASES = [
    ALL_CASES["main_fues"],
    # ALL_CASES["main_dj"],
    # ALL_CASES["new_dj"],
    # ALL_CASES["new_fues"],
]

# =====================================================================================
# Specification -- shared across all four cases, so they're comparable
#
# Memory warning: dcegm allocates the value/policy/endog_grid solution containers as
# three (n_state_choices, n_continuous_state_combinations, n_total_wealth_grid)
# float64 arrays *before* backward induction even starts. n_total_wealth_grid scales
# ~linearly with n_assets_end_of_period (fues: n_assets_end_of_period * 1.2; dj:
# len(assets_begin_of_period) + 1), so doubling either wealth grid roughly doubles
# these containers. At sex_type/edu_type = "all"/"all" (the real production state
# space) with n_assets_end_of_period=100, this OOM'd on an 80GB H100 -- just the
# three containers needed ~47GB, before any working memory for the actual solve.
# The production model itself uses a 28-point assets_end_of_period grid (see
# model_code.wealth_and_budget.assets_grid.create_end_of_period_assets), which is
# what these defaults are set close to. Increase deliberately, and expect roughly
# proportional GPU memory growth.
# =====================================================================================
SPECS = {
    "sex_type": "all",
    "edu_type": "all",
    "seed": 123,
    "n_assets_end_of_period": 30,
    "n_assets_begin_of_period": 30,
    "n_experience_points": 11,
    "max_wealth_raw": 10_000_000,
    "max_wealth_model_units": 1_000,
}

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    choices=sorted(ALL_CASES),
    default=None,
    help="Run a specific case non-interactively; overrides CASES above.",
)
args, _ = parser.parse_known_args()
case_name_override = args.case or os.environ.get("BENCHMARK_CASE")

if case_name_override is not None:
    case = ALL_CASES[case_name_override]
else:
    assert len(CASES) == 1, (
        "Comment out all but one entry in CASES before running -- each run executes "
        f"exactly one case. Currently {len(CASES)} entries are active. "
        "(Or pass --case/BENCHMARK_CASE to bypass CASES entirely.)"
    )
    case = CASES[0]

dcegm_git_info = get_dcegm_git_info()
print(f"dcegm submodule: {dcegm_git_info}", flush=True)
if dcegm_git_info["branch"] != case["expected_dcegm_branch"]:
    print(
        f"WARNING: case '{case['name']}' expects dcegm branch "
        f"'{case['expected_dcegm_branch']}', but 'submodules/dcegm' is currently on "
        f"'{dcegm_git_info['branch']}'. Did you forget to check out the right branch?",
        flush=True,
    )
if dcegm_git_info["dirty"]:
    print(
        "WARNING: 'submodules/dcegm' has uncommitted changes -- results may not be "
        "reproducible from the recorded commit hash alone.",
        flush=True,
    )

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

# Redirect the model-spec cache dcegm writes during setup to a benchmark-local
# folder, so this never collides with (or overwrites) the production model cache
# under output/intermediate_data/ -- everything else still reads real project paths.
model_cache_dir = str(RESULTS_DIR / "model_cache") + "/"
os.makedirs(model_cache_dir, exist_ok=True)
bench_path_dict = dict(path_dict)
bench_path_dict["intermediate_data"] = model_cache_dir

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

assets_end_of_period_grid = build_assets_end_of_period_grid(
    SPECS["n_assets_end_of_period"], max_wealth_raw=SPECS["max_wealth_raw"]
)
experience_grid = build_experience_grid(SPECS["n_experience_points"])

assets_begin_of_period_grid = None
if case["upper_envelope_method"] == "druedahl_jorgensen":
    assets_begin_of_period_grid = build_assets_begin_of_period_grid(
        SPECS["n_assets_begin_of_period"],
        max_wealth_model_units=SPECS["max_wealth_model_units"],
    )

# %%
# Build + solve
model = specify_model(
    path_dict=bench_path_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    sim_specs={"announcement_age": None, "SRA_at_start": 67, "SRA_at_retirement": 67},
    simulate_expectations=False,
    load_model=False,
    sex_type=SPECS["sex_type"],
    edu_type=SPECS["edu_type"],
    util_type=specs["util_type"],
    upper_envelope_method=case["upper_envelope_method"],
    assets_end_of_period_grid=assets_end_of_period_grid,
    assets_begin_of_period_grid=assets_begin_of_period_grid,
    experience_grid=experience_grid,
)

# dcegm builds fresh jax.jit(...) closures inside backward_induction/simulate on
# every call, so the *first* call to .solve()/.simulate() always pays XLA
# compilation cost on top of actual runtime. JAX's compilation cache keys on the
# traced computation, not the wrapper object's identity, so a second call with the
# same shapes/case still hits the cache and is fast -- do one untimed warm-up call
# before each timed one, and only record the second.
print(
    f"Warm-up solve for case '{case['name']}' (includes JIT compilation)...", flush=True
)
warmup_solved = model.solve(params)
jax.block_until_ready((warmup_solved.value, warmup_solved.policy))
del warmup_solved

print(f"Solving case '{case['name']}' (timed) ...", flush=True)
solve_start = time.perf_counter()
model_solved = model.solve(params)
jax.block_until_ready((model_solved.value, model_solved.policy))
solve_seconds = time.perf_counter() - solve_start
print(f"Solved in {solve_seconds:.1f}s", flush=True)

# %%
# Simulate
initial_states = generate_start_states_from_obs(
    path_dict=path_dict,
    params=params,
    inital_SRA=67,
    seed=SPECS["seed"],
    model_class=model_solved,
    only_informed=False,
)


def _simulate():
    sim_df = model_solved.simulate(states_initial=initial_states, seed=SPECS["seed"])
    sim_df = sim_df[sim_df["health"] != 3].copy()
    sim_df.reset_index(inplace=True)  # "period"/"agent" start out as a MultiIndex
    return sim_df


print("Warm-up simulate (includes JIT compilation)...", flush=True)
_simulate()

print("Simulating (timed) ...", flush=True)
simulate_start = time.perf_counter()
df = _simulate()
simulate_seconds = time.perf_counter() - simulate_start
print(f"Simulated in {simulate_seconds:.1f}s", flush=True)

# Peak memory across the whole case (warm-up + timed calls) -- a high-water mark
# that JAX doesn't reset between calls, so one reading at the end covers both.
mem_stats = jax.local_devices()[0].memory_stats()
peak_bytes_in_use = None if mem_stats is None else mem_stats.get("peak_bytes_in_use")

# %%
# Per-period outcomes
per_period_consumption = df.groupby("period")["consumption"].mean()
per_period_choice_shares = pd.crosstab(df["period"], df["choice"], normalize="index")

result = {
    "case_name": case["name"],
    "upper_envelope_method": case["upper_envelope_method"],
    "dcegm_branch": dcegm_git_info["branch"],
    "dcegm_commit": dcegm_git_info["commit"],
    "dcegm_dirty": dcegm_git_info["dirty"],
    "specs": SPECS,
    "solve_seconds": solve_seconds,
    "simulate_seconds": simulate_seconds,
    "total_seconds": solve_seconds + simulate_seconds,
    "jax_platform": jax.local_devices()[0].platform,
    "jax_device_kind": jax.local_devices()[0].device_kind,
    "peak_bytes_in_use": peak_bytes_in_use,
    "n_agents_simulated": df["agent"].nunique(),
    "per_period_consumption": per_period_consumption,
    "per_period_choice_shares": per_period_choice_shares,
    "choice_labels": specs["choice_labels"],
}

os.makedirs(RESULTS_DIR, exist_ok=True)
result_path = RESULTS_DIR / f"{case['name']}.pkl"
with open(result_path, "wb") as f:
    pkl.dump(result, f)

print(f"Saved: {result_path}", flush=True)
