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
# =====================================================================================
SPECS = {
    "sex_type": "all",
    "edu_type": "all",
    "seed": 123,
    "n_assets_end_of_period": 100,
    "n_assets_begin_of_period": 100,
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

print(f"Solving case '{case['name']}' ...", flush=True)
solve_start = time.perf_counter()
model_solved = model.solve(params)
jax.block_until_ready((model_solved.value, model_solved.policy))
solve_seconds = time.perf_counter() - solve_start
print(f"Solved in {solve_seconds:.1f}s", flush=True)

mem_after_solve = jax.local_devices()[0].memory_stats()

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

print("Simulating ...", flush=True)
simulate_start = time.perf_counter()
df = model_solved.simulate(states_initial=initial_states, seed=SPECS["seed"])
df = df[df["health"] != 3].copy()
df.reset_index(inplace=True)  # "period"/"agent" start out as a MultiIndex
simulate_seconds = time.perf_counter() - simulate_start
print(f"Simulated in {simulate_seconds:.1f}s", flush=True)

mem_after_simulate = jax.local_devices()[0].memory_stats()


def _peak_bytes(mem_stats):
    return None if mem_stats is None else mem_stats.get("peak_bytes_in_use")


peak_bytes_in_use = max(
    (
        b
        for b in (_peak_bytes(mem_after_solve), _peak_bytes(mem_after_simulate))
        if b is not None
    ),
    default=None,
)

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
