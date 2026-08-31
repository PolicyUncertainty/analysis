# %%
"""dcegm engine benchmark: solve + simulate our retirement model for three specs.

Round 1 (see output_v1_uniform_grid_branch_check/) already confirmed that the
on-demand law-of-motion refactor on this dcegm branch reproduces `main` exactly, so
branch comparison is settled -- this round always runs all three specs below in one
process against whatever dcegm branch is currently checked out:

  - "production": fues, production's real (non-uniform) assets_end_of_period and
    experience grids, completely unmodified -- i.e. exactly what run_cf_debias.py
    etc. use.
  - "new_fues": fues, the same production grids passed explicitly through the
    benchmark's override machinery -- a control that should match "production"
    exactly, proving the override plumbing is faithful.
  - "new_dj": druedahl_jorgensen, the same production assets_end_of_period/
    experience grids, plus assets_begin_of_period built from the same (non-uniform)
    points as assets_end_of_period, in model units -- so dj gets a comparably
    well-resolved grid instead of round 1's uniform one (round 1 found fues/dj
    diverging substantially, suspected to be partly because dj's uniform
    assets_begin_of_period had zero resolution below ~323,000 raw currency, right
    where the credit constraint bites).

Both asset AND experience grids are production's hand-tuned ones on purpose: both
are manually optimized, and experience in particular has large discontinuities
around retirement timing that a uniform grid misses.

Results are pickled to src/benchmarks/results/<spec_name>.pkl. Run make_table.py
afterwards to build the comparison table.
"""
import os
import pickle as pkl
import time
from pathlib import Path

import jax
import pandas as pd

jax.config.update("jax_enable_x64", True)

from benchmarks.provenance import get_dcegm_git_info
from model_code.specify_model import specify_model
from model_code.state_space.experience import define_experience_grid
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from set_paths import create_path_dict
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import generate_derived_and_data_derived_specs

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"

# Round 1 confirmed this branch reproduces main exactly -- this is just a sanity
# check that you're benchmarking the branch you think you are.
EXPECTED_DCEGM_BRANCH = "remove_endog"

SPECS = {
    "sex_type": "all",
    "edu_type": "all",
    "seed": 123,
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

# druedahl_jorgensen needs assets_begin_of_period explicitly and evaluates directly
# on it (no endogenous refinement), so give it the same non-uniform points as
# assets_end_of_period, converted to model units. dcegm prepends its own 0, so drop
# ours to avoid a duplicate/degenerate first grid point.
assets_begin_of_period_grid = (
    production_assets_end_of_period_grid / specs["wealth_unit"]
)
assets_begin_of_period_grid = assets_begin_of_period_grid[
    assets_begin_of_period_grid > 0
]

SPEC_LIST = [
    {
        "name": "production",
        "upper_envelope_method": None,
        "assets_end_of_period_grid": None,
        "assets_begin_of_period_grid": None,
        "experience_grid": None,
    },
    {
        "name": "new_fues",
        "upper_envelope_method": "fues",
        "assets_end_of_period_grid": production_assets_end_of_period_grid,
        "assets_begin_of_period_grid": None,
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

    # dcegm builds fresh jax.jit(...) closures inside backward_induction/simulate on
    # every call, so the *first* call always pays XLA compilation cost on top of
    # actual runtime. JAX's compilation cache keys on the traced computation, not
    # the wrapper object's identity, so a second call with the same shapes is fast
    # -- do one untimed warm-up call before each timed one.
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

    initial_states = generate_start_states_from_obs(
        path_dict=path_dict,
        params=params,
        inital_SRA=67,
        seed=SPECS["seed"],
        model_class=model_solved,
        only_informed=False,
    )

    def _simulate():
        sim_df = model_solved.simulate(
            states_initial=initial_states, seed=SPECS["seed"]
        )
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

    # Peak memory across the whole spec (warm-up + timed calls) -- a high-water
    # mark that JAX doesn't reset between calls, so one reading at the end covers
    # both.
    mem_stats = jax.local_devices()[0].memory_stats()
    peak_bytes_in_use = (
        None if mem_stats is None else mem_stats.get("peak_bytes_in_use")
    )

    per_period_consumption = df.groupby("period")["consumption"].mean()
    per_period_choice_shares = pd.crosstab(
        df["period"], df["choice"], normalize="index"
    )

    result = {
        "case_name": spec["name"],
        "upper_envelope_method": spec["upper_envelope_method"] or "fues",
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
    result_path = RESULTS_DIR / f"{spec['name']}.pkl"
    with open(result_path, "wb") as f:
        pkl.dump(result, f)
    print(f"Saved: {result_path}", flush=True)


for spec in SPEC_LIST:
    run_spec(spec)
