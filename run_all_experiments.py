from __future__ import annotations

import csv
import os
from statistics import mean
from typing import Dict, List, Sequence

from experiments import cost_margin, path_multiplicity, temperature_analysis


RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _summarize_numeric(rows: List[Dict[str, str]], columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for col in columns:
        values: List[float] = []
        for row in rows:
            if col in row and row[col] not in ("", None):
                try:
                    values.append(float(row[col]))
                except ValueError:
                    continue
        if values:
            summary[col] = {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
            }
    return summary


def _print_summary(label: str, csv_file: str, columns: Sequence[str]) -> None:
    path = os.path.join(RESULTS_DIR, csv_file)
    rows = _read_csv_rows(path)
    summary = _summarize_numeric(rows, columns)
    print(f"\n[{label}] {csv_file}")
    print(f"Rows: {len(rows)}")
    for col, stats in summary.items():
        print(
            f"  {col}: min={stats['min']:.6g}, max={stats['max']:.6g}, mean={stats['mean']:.6g}"
        )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    temperature_analysis.main()
    cost_margin.main()
    path_multiplicity.main()

    _print_summary(
        "Temperature analysis",
        "temperature_gap.csv",
        ["gap_d_star_minus_dT", "bound_theorem_iii_1", "T"],
    )
    _print_summary(
        "Exponential convergence",
        "exponential_convergence.csv",
        ["gap_d_star_minus_dT", "T", "inv_T"],
    )
    _print_summary(
        "Classical vs soft",
        "classical_vs_soft.csv",
        ["d_star", "d_T"],
    )
    _print_summary(
        "Cost margin",
        "cost_margin.csv",
        ["gap_d_star_minus_dT", "bound_theorem_iii_1", "Delta", "T"],
    )
    _print_summary(
        "Path multiplicity",
        "path_multiplicity.csv",
        ["gap_d_star_minus_dT", "bound_theorem_iii_1", "N_tot", "Delta", "T"],
    )


if __name__ == "__main__":
    main()
