"""Aggregate per-spec results from run_benchmark.py into a comparison table.

Run after all three specs (see SPEC_LIST in run_benchmark.py) have been
benchmarked -- i.e. src/benchmarks/results/*.pkl contains one file per spec.

This round only compares the *solve* (see run_benchmark.py docstring), so this
script reports runtime/memory plus grid sizes, and -- for any pair of specs whose
solved arrays happen to share the same shape (same grids and upper-envelope
method, e.g. "production" vs "new_fues") -- a direct max-abs-difference check.
Specs with different grids/methods (e.g. fues vs dj) have differently-shaped
solution containers and can't be diffed elementwise; comparing those properly
needs evaluating both solutions at a common set of query points via
model_solved.policy_and_value_for_states_and_choices(), which has to happen inside
run_benchmark.py while the solved objects are still alive (not here) -- not done
yet, flagged as a follow-up rather than guessed at.
"""

import pickle as pkl
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR = BENCHMARK_DIR / "output_v2_production_grid_method_check"

# Canonical case order (falls back to discovery order for any case not listed here).
CASE_ORDER = ["production", "new_fues", "new_dj"]


def load_results():
    result_files = sorted(RESULTS_DIR.glob("*.pkl"))
    if not result_files:
        raise FileNotFoundError(
            f"No result files found in {RESULTS_DIR}. Run run_benchmark.py for each "
            "spec first."
        )
    results = {}
    for path in result_files:
        with open(path, "rb") as f:
            result = pkl.load(f)
        results[result["case_name"]] = result

    ordered_names = [name for name in CASE_ORDER if name in results] + [
        name for name in results if name not in CASE_ORDER
    ]
    missing = [name for name in CASE_ORDER if name not in results]
    if missing:
        print(f"Note: no result yet for spec(s) {missing}; table will omit them.")
    return {name: results[name] for name in ordered_names}


def build_summary_table(results):
    rows = []
    for name, res in results.items():
        peak_mem = res["peak_bytes_in_use"]
        rows.append(
            {
                "Spec": name,
                "Upper Envelope": res["upper_envelope_method"],
                "n assets_end_of_period": len(res["assets_end_of_period_grid"]),
                "n assets_begin_of_period": (
                    "n/a"
                    if res["assets_begin_of_period_grid"] is None
                    else len(res["assets_begin_of_period_grid"])
                ),
                "n experience": len(res["experience_grid"]),
                "Solve (s)": round(res["solve_seconds"], 1),
                "Peak GPU Mem (GB)": (
                    round(peak_mem / 1e9, 2) if peak_mem is not None else "N/A"
                ),
                "Device": f"{res['jax_platform']} ({res['jax_device_kind']})",
                "dcegm Branch": res["dcegm_branch"],
                "Commit": res["dcegm_commit"],
            }
        )
    return pd.DataFrame(rows).set_index("Spec")


def compare_solutions(results):
    """Elementwise max-abs-diff for every pair of specs whose solved arrays share a
    shape (same grids + method); notes which pairs can't be compared this way."""
    print("\n=== Solution comparison (value/policy) ===")
    for name_a, name_b in combinations(results, 2):
        res_a, res_b = results[name_a], results[name_b]
        if res_a["value"].shape != res_b["value"].shape:
            print(
                f"{name_a} vs {name_b}: different solution shapes "
                f"({res_a['value'].shape} vs {res_b['value'].shape}) -- not directly "
                "comparable elementwise, skipping."
            )
            continue
        value_diff = np.nanmax(np.abs(res_a["value"] - res_b["value"]))
        policy_diff = np.nanmax(np.abs(res_a["policy"] - res_b["policy"]))
        print(
            f"{name_a} vs {name_b}: max|value diff| = {value_diff:.6g}, "
            f"max|policy diff| = {policy_diff:.6g}"
        )


def save_table(df, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv")
    with open(OUTPUT_DIR / f"{name}.tex", "w") as f:
        f.write(df.to_latex() + "\n")
    print(f"Saved: {OUTPUT_DIR / f'{name}.tex'}")


def main():
    results = load_results()

    summary = build_summary_table(results)
    print("\n=== Runtime / GPU / grid-size summary ===")
    print(summary.to_string())
    save_table(summary, "benchmark_summary")

    compare_solutions(results)


if __name__ == "__main__":
    main()
