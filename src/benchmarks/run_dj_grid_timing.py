# %%
"""Solve-time / GPU-memory cost of each candidate druedahl_jorgensen (dj)
assets_begin_of_period grid (see dj_candidates.py), against a fues baseline on
production's real assets_end_of_period/experience grids.

This is the cost counterpart to check_dj_grid.py -- that script measures whether a
candidate grid is *accurate enough* against the fues reference; this one measures
what it costs to solve, over the identical candidate set, so the two results can be
read together as one accuracy-vs-cost picture.

Only solve is timed (no simulate): for judging a grid's runtime/memory footprint,
solving is what matters, and simulating roughly doubles wall time for no extra cost
information.

No dcegm-branch bookkeeping -- settled in an earlier benchmark round, not revisited.

Note on GPU memory: `peak_bytes_in_use` is a high-water mark that JAX never resets
within a process, so later candidates' readings are contaminated by whatever the
largest earlier candidate already allocated -- only useful as a rough per-run
ceiling, not a clean per-candidate isolate. Solve time is unaffected by this.

Results: src/benchmarks/output_dj_grid_check/timing_by_candidate.csv
"""
import os
import pickle as pkl
import time
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
from specs.derive_specs import generate_derived_and_data_derived_specs

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR = BENCHMARK_DIR / "output_dj_grid_check"

SPECS = {"sex_type": "all", "edu_type": "all"}

# %%
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

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


def timed_solve(name, model):
    print(f"\n=== {name}: warm-up solve (JIT compilation) ===", flush=True)
    warmup_solved = model.solve(params)
    jax.block_until_ready((warmup_solved.value, warmup_solved.policy))
    del warmup_solved

    print(f"=== {name}: timed solve ===", flush=True)
    start = time.perf_counter()
    model_solved = model.solve(params)
    jax.block_until_ready((model_solved.value, model_solved.policy))
    solve_seconds = time.perf_counter() - start
    print(f"{name}: solved in {solve_seconds:.1f}s", flush=True)

    mem_stats = jax.local_devices()[0].memory_stats()
    peak_bytes = None if mem_stats is None else mem_stats.get("peak_bytes_in_use")
    return model_solved, solve_seconds, peak_bytes


# %%
rows = []

fues_solved, solve_seconds, peak_bytes = timed_solve(
    "fues_production", build_model("fues")
)
rows.append(
    {
        "candidate": "fues_production",
        "n_assets_begin_of_period": None,
        "solve_seconds": solve_seconds,
        "peak_gpu_mem_gb": None if peak_bytes is None else peak_bytes / 1e9,
    }
)

# fues's own solved endogenous grid gives the candidates their shape -- see
# dj_candidates.py / check_dj_grid.py for why this and not assets_end_of_period.
endog_grid_sample = pool_positive_finite(fues_solved.endog_grid)
del fues_solved

for cand in CANDIDATE_SPECS:
    grid = build_candidate_grid(cand, endog_grid_sample)
    dj_solved, solve_seconds, peak_bytes = timed_solve(
        cand["name"], build_model("druedahl_jorgensen", grid)
    )
    rows.append(
        {
            "candidate": cand["name"],
            "n_assets_begin_of_period": len(grid),
            "solve_seconds": solve_seconds,
            "peak_gpu_mem_gb": None if peak_bytes is None else peak_bytes / 1e9,
        }
    )
    del dj_solved

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timing_df = pd.DataFrame(rows)
timing_df.to_csv(OUTPUT_DIR / "timing_by_candidate.csv", index=False)
print("\n=== Solve time / peak GPU memory, by candidate ===")
print(timing_df.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'timing_by_candidate.csv'}")
