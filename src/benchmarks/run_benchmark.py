# %%
"""dcegm engine benchmark: solve our retirement model for three specs.

Round 1 (see output_v1_uniform_grid_branch_check/) confirmed the on-demand
law-of-motion refactor on this dcegm branch reproduces `main` exactly, so branch
comparison is settled. Round 2 confirmed the grid-override plumbing in
specify_model faithfully reproduces production when given production's grids.

This is the grid-refinement round: only the *solve* is compared (no simulation) --
for judging whether a grid is good enough, the solved value/policy functions are
what matters; simulated behavior is a derived, noisier proxy for the same thing and
roughly doubles runtime for no extra information here. All three specs are fully
explicit (no None anywhere) so nothing depends on a hidden default:

  - "production": fues, production's real (non-uniform) assets_end_of_period and
    experience grids -- the fixed, never-touched baseline we compare grid
    experiments against going forward.
  - "new_fues": fues, same grids as "production" right now -- this is the one to
    start shrinking/repositioning in later rounds (keeping "production" pinned as
    the reference).
  - "new_dj": druedahl_jorgensen, same assets_end_of_period/experience grids as
    "production", plus an explicit assets_begin_of_period grid (see note below).

Both asset AND experience grids are production's hand-tuned ones on purpose: both
are manually optimized, and experience in particular has large discontinuities
around retirement timing that a uniform grid misses (see the very_long_insured
threshold discussion -- production pins exact grid nodes there deliberately).

Note on assets_begin_of_period: dj evaluates directly on this grid, so it needs to
be in the same units as everything else the model computes wealth in (i.e. divided
by specs["wealth_unit"], exactly like assets_end_of_period is inside
create_model_config_wo_informed) -- that division is a pure unit conversion, nothing
more. It is a separate question whether assets_end_of_period's *values* are a good
stand-in for assets_begin_of_period: they aren't derived from one another (end-of-
period savings and begin-of-period wealth, after returns and income, are genuinely
different quantities), we're just borrowing assets_end_of_period's non-uniform
*point placement* (dense near the credit constraint) as the best available prior,
since production never needed a begin-of-period grid before (fues doesn't use one).
Flagging this as a simplification worth revisiting, not a derivation.

Results (solved value/policy/endog_grid arrays + the grids used + timing/memory)
are pickled to src/benchmarks/results/<spec_name>.pkl. Run make_table.py afterwards
for the runtime/memory summary.
"""
import os
import pickle as pkl
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from benchmarks.grids import refine_grid
from benchmarks.provenance import get_dcegm_git_info
from model_code.specify_model import specify_model
from model_code.state_space.experience import define_experience_grid
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"

# Round 1 confirmed this branch reproduces main exactly -- this is just a sanity
# check that you're benchmarking the branch you think you are.
EXPECTED_DCEGM_BRANCH = "remove_endog"

SPECS = {
    "sex_type": "all",
    "edu_type": "all",
    # dj evaluates directly on assets_begin_of_period (no endogenous refinement),
    # so it needs more resolution than fues's endogenous grid to hit comparable
    # accuracy. 1 = production's assets_end_of_period point placement as-is (27
    # points); 2 = one extra point per interval (~53); 3 = two extra (~79); etc.
    # See refine_grid() -- density stays concentrated wherever the production grid
    # already is (near the credit constraint), just more so.
    "assets_begin_of_period_subdivisions": 2,
}

# %%
dcegm_git_info = get_dcegm_git_info()
print(f"dcegm submodule: {dcegm_git_info}", flush=True)
if dcegm_git_info["branch"] != EXPECTED_DCEGM_BRANCH:
    print(
        f"WARNING: expected dcegm branch '{EXPECTED_DCEGM_BRANCH}', but "
        f"'submodules/dcegm' is on '{dcegm_git_info['branch']}'.",
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

# Production's real, hand-tuned grids.
production_assets_end_of_period_grid = create_end_of_period_assets()
production_experience_grid = define_experience_grid(specs)

# See module docstring "Note on assets_begin_of_period" -- unit conversion is
# necessary and unrelated to the (separate, unresolved) question of whether these
# point locations are the right ones for begin-of-period wealth.
assets_begin_of_period_grid = (
    production_assets_end_of_period_grid / specs["wealth_unit"]
)
assets_begin_of_period_grid = assets_begin_of_period_grid[
    assets_begin_of_period_grid > 0
]
# dj evaluates directly on this grid (no endogenous refinement), so densify it --
# see SPECS["assets_begin_of_period_subdivisions"].
assets_begin_of_period_grid = refine_grid(
    assets_begin_of_period_grid, SPECS["assets_begin_of_period_subdivisions"]
)

SPEC_LIST = [
    {
        "name": "production",
        "upper_envelope_method": "fues",
        "assets_end_of_period_grid": production_assets_end_of_period_grid,
        "assets_begin_of_period_grid": None,  # not used by fues
        "experience_grid": production_experience_grid,
    },
    {
        "name": "new_fues",
        "upper_envelope_method": "fues",
        "assets_end_of_period_grid": production_assets_end_of_period_grid,
        "assets_begin_of_period_grid": None,  # not used by fues
        "experience_grid": production_experience_grid,
    },
    {
        "name": "new_dj",
        "upper_envelope_method": "druedahl_jorgensen",
        "assets_end_of_period_grid": production_assets_end_of_period_grid,
        "assets_begin_of_period_grid": assets_begin_of_period_grid,
        "experience_grid": production_experience_grid,
    },
]


# %%
def run_spec(spec):
    print(f"\n=== Spec: {spec['name']} ===", flush=True)
    model = specify_model(
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
        upper_envelope_method=spec["upper_envelope_method"],
        assets_end_of_period_grid=spec["assets_end_of_period_grid"],
        assets_begin_of_period_grid=spec["assets_begin_of_period_grid"],
        experience_grid=spec["experience_grid"],
    )

    # dcegm builds fresh jax.jit(...) closures inside backward_induction on every
    # call, so the *first* call always pays XLA compilation cost on top of actual
    # runtime. JAX's compilation cache keys on the traced computation, not the
    # wrapper object's identity, so a second call with the same shapes is fast --
    # do one untimed warm-up call before the timed one.
    print(
        f"Warm-up solve for '{spec['name']}' (includes JIT compilation)...", flush=True
    )
    warmup_solved = model.solve(params)
    jax.block_until_ready((warmup_solved.value, warmup_solved.policy))
    del warmup_solved

    print(f"Solving '{spec['name']}' (timed) ...", flush=True)
    solve_start = time.perf_counter()
    model_solved = model.solve(params)
    jax.block_until_ready((model_solved.value, model_solved.policy))
    solve_seconds = time.perf_counter() - solve_start
    print(f"Solved in {solve_seconds:.1f}s", flush=True)

    # Peak memory across both calls (warm-up + timed) -- a high-water mark that
    # JAX doesn't reset between calls, so one reading at the end covers both.
    mem_stats = jax.local_devices()[0].memory_stats()
    peak_bytes_in_use = (
        None if mem_stats is None else mem_stats.get("peak_bytes_in_use")
    )

    result = {
        "case_name": spec["name"],
        "upper_envelope_method": spec["upper_envelope_method"],
        "dcegm_branch": dcegm_git_info["branch"],
        "dcegm_commit": dcegm_git_info["commit"],
        "dcegm_dirty": dcegm_git_info["dirty"],
        "specs": SPECS,
        "assets_end_of_period_grid": np.asarray(spec["assets_end_of_period_grid"]),
        "assets_begin_of_period_grid": (
            None
            if spec["assets_begin_of_period_grid"] is None
            else np.asarray(spec["assets_begin_of_period_grid"])
        ),
        "experience_grid": np.asarray(spec["experience_grid"]),
        "solve_seconds": solve_seconds,
        "jax_platform": jax.local_devices()[0].platform,
        "jax_device_kind": jax.local_devices()[0].device_kind,
        "peak_bytes_in_use": peak_bytes_in_use,
        # The solution itself -- what this round is actually about.
        "value": np.asarray(model_solved.value),
        "policy": np.asarray(model_solved.policy),
        "endog_grid": (
            None
            if model_solved.endog_grid is None
            else np.asarray(model_solved.endog_grid)
        ),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = RESULTS_DIR / f"{spec['name']}.pkl"
    with open(result_path, "wb") as f:
        pkl.dump(result, f)
    print(f"Saved: {result_path}", flush=True)


for spec in SPEC_LIST:
    run_spec(spec)
