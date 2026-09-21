"""Run the Monte Carlo cost model from the command line.

Examples
--------
    python run_model.py                                  # sample data, all three models
    python run_model.py my_wbs.xlsx --dist pert          # your file, Beta-PERT only
    python run_model.py my_wbs.xlsx --correlation 0.3    # tasks share some common drift
    python run_model.py my_wbs.xlsx --risk event         # risks happen or not, by likelihood
    python run_model.py my_wbs.xlsx --risk none          # estimate uncertainty only
    python run_model.py --pick                           # choose the file in a dialog
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from montecarlo import (  # noqa: E402
    DISTRIBUTIONS, Settings, check_subtotals, export_excel, load_wbs,
    milestone_summary, project_summary, run, sensitivity,
)
from montecarlo import charts  # noqa: E402

SAMPLE = Path(__file__).parent / "data" / "sample_wbs.xlsx"


def pick_file() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askopenfilename(
        title="Select your WBS file",
        filetypes=[("Excel or CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
    root.destroy()
    if not chosen:
        sys.exit("No file selected.")
    return Path(chosen)


def main(argv=None):
    p = argparse.ArgumentParser(description="Monte Carlo cost-risk model for a WBS.")
    p.add_argument("file", nargs="?", help="Excel/CSV WBS file (default: data/sample_wbs.xlsx)")
    p.add_argument("--pick", action="store_true", help="choose the input file in a dialog")
    p.add_argument("--dist", default="all", choices=["all", *DISTRIBUTIONS])
    p.add_argument("--sims", type=int, default=10_000, help="number of simulations")
    p.add_argument("--low", type=float, default=0.90, help="optimistic factor when none given")
    p.add_argument("--high", type=float, default=1.20, help="pessimistic factor when none given")
    p.add_argument("--correlation", type=float, default=0.0, help="0 to 0.95, shared drift")
    p.add_argument("--risk", default="loaded", choices=["loaded", "event", "none"],
                   help="how risk multipliers are used (see README)")
    p.add_argument("--seed", type=int, default=42, help="random seed (use -1 for none)")
    p.add_argument("--currency", default="£")
    p.add_argument("--out", default="outputs", help="folder for charts and the Excel report")
    a = p.parse_args(argv)

    path = pick_file() if a.pick else Path(a.file) if a.file else SAMPLE
    tasks, summaries = load_wbs(path)
    print(f"Risk mode: {a.risk}   Correlation: {a.correlation}   Simulations: {a.sims:,}")
    print(f"Loaded {path.name}: {len(tasks)} tasks "
          f"({len(summaries)} milestone/sub-total rows excluded so nothing is counted twice)")
    for w in check_subtotals(tasks, summaries):
        print("  Note:", w)

    settings = Settings(n_sims=a.sims, optimistic_factor=a.low, pessimistic_factor=a.high,
                        risk_mode=a.risk, correlation=a.correlation,
                        seed=None if a.seed == -1 else a.seed)
    dists = DISTRIBUTIONS if a.dist == "all" else (a.dist,)
    estimates, results = run(tasks, dists, settings)

    summary = project_summary(results, estimates)
    cur = a.currency
    print(f"\nBase estimate (sum of task costs): {cur}{summary['Base estimate'].iat[0]:,.0f}")
    if settings.risk_mode == "loaded":
        print(f"Risk-loaded estimate:              {cur}{summary['Risk-loaded estimate'].iat[0]:,.0f}")
    print()
    cols = ["Model", "Mean", "P10", "P50", "P80", "P90", "Std Dev", "Contingency at P80 (%)"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    out = Path(a.out) / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    for key, res in results.items():
        charts.distribution(res, out / f"{key}_distribution.png", cur)
        charts.tornado(sensitivity(res), res, out / f"{key}_tornado.png")
        charts.milestones(milestone_summary(res, estimates), res, out / f"{key}_milestones.png", cur)
    charts.s_curve(results, out / "s_curve.png", cur)
    xlsx = export_excel(results, estimates, out / "results.xlsx")
    print(f"\nCharts and report saved in {out}/ (Excel: {xlsx.name})")


if __name__ == "__main__":
    main()
