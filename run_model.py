"""Run the Monte Carlo cost model from the command line.

Examples
--------
    python run_model.py                                  # sample data, all three models
    python run_model.py my_wbs.xlsx --dist pert          # your file, Beta-PERT only
    python run_model.py my_wbs.xlsx --correlation 0.3    # tasks share some common drift
    python run_model.py my_wbs.xlsx --risk event         # risks happen or not, by likelihood
    python run_model.py my_wbs.xlsx --risk none          # estimate uncertainty only
    python run_model.py my_wbs.xlsx --risk register      # risks from the 'Risk Register' sheet
    python run_model.py --start 2026-03-02 --jcl 0.7     # schedule dates and joint confidence target
    python run_model.py --pick                           # choose the file in a dialog
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from montecarlo import (  # noqa: E402
    DISTRIBUTIONS, Settings, allocate_contingency, as_result, budget_for_jcl, check_subtotals,
    contingency_split, export_excel, has_schedule, jcl, load_risk_register, load_wbs, milestone_summary,
    project_summary, risk_ranking, run, run_integrated, schedule_sensitivity, sensitivity,
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
    p.add_argument("--risk", default="auto", choices=["auto", "loaded", "event", "none", "register"],
                   help="how risk is modelled; auto = the risk register if the file has one, else loaded")
    p.add_argument("--start", default="2026-01-05", help="project start date, for finish dates")
    p.add_argument("--jcl", type=float, default=0.70, help="target joint confidence level (cost and schedule)")
    p.add_argument("--seed", type=int, default=42, help="random seed (use -1 for none)")
    p.add_argument("--currency", default="£")
    p.add_argument("--out", default="outputs", help="folder for charts and the Excel report")
    a = p.parse_args(argv)

    path = pick_file() if a.pick else Path(a.file) if a.file else SAMPLE
    tasks, summaries = load_wbs(path)
    register = load_risk_register(path)
    risk = a.risk if a.risk != "auto" else ("register" if register is not None else "loaded")
    if risk == "register" and register is None:
        sys.exit("--risk register needs a 'Risk Register' sheet in the workbook (see data/sample_wbs.xlsx).")
    schedule = has_schedule(tasks)
    print(f"Risk mode: {risk}   Correlation: {a.correlation}   Simulations: {a.sims:,}   "
          f"Schedule: {'yes' if schedule else 'no (add a Duration column to simulate dates)'}")
    print(f"Loaded {path.name}: {len(tasks)} tasks "
          f"({len(summaries)} milestone/sub-total rows excluded so nothing is counted twice)")
    for w in check_subtotals(tasks, summaries):
        print("  Note:", w)

    settings = Settings(n_sims=a.sims, optimistic_factor=a.low, pessimistic_factor=a.high,
                        risk_mode=risk, correlation=a.correlation, start_date=a.start,
                        seed=None if a.seed == -1 else a.seed)
    dists = DISTRIBUTIONS if a.dist == "all" else (a.dist,)
    integrated = {}
    if risk == "register" or schedule:
        # Integrated model: risk events, time-dependent cost and (if durations exist) the schedule.
        for d in dists:
            integrated[d] = run_integrated(tasks, register if risk == "register" else None, d, settings,
                                           use_register=risk == "register")
        estimates = next(iter(integrated.values())).estimates
        results = {d: as_result(r) for d, r in integrated.items()}
    else:
        estimates, results = run(tasks, dists, settings)

    summary = project_summary(results, estimates)
    cur = a.currency
    print(f"\nBase estimate (sum of task costs): {cur}{summary['Base estimate'].iat[0]:,.0f}")
    if risk == "loaded":
        print(f"Risk-loaded estimate:              {cur}{summary['Risk-loaded estimate'].iat[0]:,.0f}")
    print()
    cols = ["Model", "Mean", "P10", "P50", "P80", "P90", "Std Dev", "Contingency at P80 (%)"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    extras = {}
    main_key = "pert" if "pert" in integrated else next(iter(integrated), None)
    if main_key:
        res = integrated[main_key]
        if res.schedule is not None:
            sr = res.schedule
            p50, p80 = np.percentile(sr.project_days, [50, 80])
            on_plan = (sr.project_days <= sr.deterministic_days).mean()
            print(f"\nSchedule ({res.name}): plan {sr.deterministic_days:.0f} working days "
                  f"(finish {sr.finish_date(sr.deterministic_days):%d %b %Y}, {on_plan:.0%} chance of making it)")
            print(f"  P50 {p50:.0f} days -> {sr.finish_date(p50):%d %b %Y}    "
                  f"P80 {p80:.0f} days -> {sr.finish_date(p80):%d %b %Y}")
            c80 = float(np.percentile(res.total_cost, 80))
            print(f"  Joint confidence of P80 cost and P80 date together: {jcl(res, c80, p80):.0%}")
            need = budget_for_jcl(res, p80, a.jcl)
            if need:
                print(f"  Budget for a {a.jcl:.0%} joint confidence with the P80 date: {cur}{need:,.0f}")
            extras["Schedule"] = schedule_sensitivity(sr)
        split = contingency_split(tasks, register if risk == "register" else None, main_key, settings)
        print(f"\nP80 contingency {cur}{split['total_budget'] - split['base_cost']:,.0f}: "
              f"estimate uncertainty {cur}{split['estimate_uncertainty']:,.0f}, "
              f"named risks {cur}{split['risk_events']:,.0f}")
        extras["Contingency allocation"] = allocate_contingency(res)
        if risk == "register":
            ranking = risk_ranking(tasks, register, main_key, settings)
            extras["Risk ranking"] = ranking

    out = Path(a.out) / f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    for key, res in results.items():
        charts.distribution(res, out / f"{key}_distribution.png", cur)
        charts.tornado(sensitivity(res), res, out / f"{key}_tornado.png")
        charts.milestones(milestone_summary(res, estimates), res, out / f"{key}_milestones.png", cur)
    charts.s_curve(results, out / "s_curve.png", cur)
    if main_key and integrated[main_key].schedule is not None:
        res = integrated[main_key]
        charts.finish_s_curve(res, out / "finish_dates.png")
        charts.criticality(extras["Schedule"], out / "criticality.png")
        charts.jcl_scatter(res, float(np.percentile(res.total_cost, 80)),
                           float(np.percentile(res.project_days, 80)), out / "joint_confidence.png", cur)
    if "Risk ranking" in extras:
        charts.risk_bars(extras["Risk ranking"], out / "risk_ranking.png", cur)
    xlsx = export_excel(results, estimates, out / "results.xlsx", extras)
    print(f"\nCharts and report saved in {out}/ (Excel: {xlsx.name})")


if __name__ == "__main__":
    main()
