"""Aggregate per-spec results from run_benchmark.py into comparison tables.

Run after all three specs (see SPEC_LIST in run_benchmark.py) have been
benchmarked -- i.e. src/benchmarks/results/*.pkl contains one file per spec.
"""

import pickle as pkl
from pathlib import Path

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
            "case first."
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
        print(f"Note: no result yet for case(s) {missing}; table will omit them.")
    return {name: results[name] for name in ordered_names}


def build_summary_table(results):
    rows = []
    for name, res in results.items():
        peak_mem = res["peak_bytes_in_use"]
        rows.append(
            {
                "Case": name,
                "dcegm Branch": res["dcegm_branch"],
                "Commit": res["dcegm_commit"],
                "Upper Envelope": res["upper_envelope_method"],
                "Solve (s)": round(res["solve_seconds"], 1),
                "Simulate (s)": round(res["simulate_seconds"], 1),
                "Total (s)": round(res["total_seconds"], 1),
                "Peak GPU Mem (GB)": (
                    round(peak_mem / 1e9, 2) if peak_mem is not None else "N/A"
                ),
                "Device": f"{res['jax_platform']} ({res['jax_device_kind']})",
            }
        )
    return pd.DataFrame(rows).set_index("Case")


def build_per_period_consumption_table(results):
    series = {name: res["per_period_consumption"] for name, res in results.items()}
    return pd.DataFrame(series).round(3)


def build_choice_share_tables(results):
    """One table per choice: rows=period, columns=case, values=share choosing it."""
    any_result = next(iter(results.values()))
    choice_labels = any_result["choice_labels"]

    tables = {}
    for choice_idx, choice_label in enumerate(choice_labels):
        series = {}
        for name, res in results.items():
            shares = res["per_period_choice_shares"]
            if choice_idx in shares.columns:
                series[name] = shares[choice_idx]
        if series:
            tables[choice_label] = pd.DataFrame(series).round(3)
    return tables


def save_table(df, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{name}.csv")
    with open(OUTPUT_DIR / f"{name}.tex", "w") as f:
        f.write(df.to_latex() + "\n")
    print(f"Saved: {OUTPUT_DIR / f'{name}.tex'}")


def main():
    results = load_results()

    summary = build_summary_table(results)
    print("\n=== Runtime / GPU summary ===")
    print(summary.to_string())
    save_table(summary, "benchmark_summary")

    consumption = build_per_period_consumption_table(results)
    print("\n=== Per-period mean consumption ===")
    print(consumption.to_string())
    save_table(consumption, "benchmark_per_period_consumption")

    for choice_label, table in build_choice_share_tables(results).items():
        safe_label = choice_label.lower().replace(" ", "_")
        print(f"\n=== Per-period choice share: {choice_label} ===")
        print(table.to_string())
        save_table(table, f"benchmark_choice_share_{safe_label}")


if __name__ == "__main__":
    main()
