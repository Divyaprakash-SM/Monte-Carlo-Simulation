"""Integrated cost-and-schedule risk analysis.

This is the full model used by the web app. One simulation run does all of this:

1. **Estimate uncertainty**: each task's cost and duration is drawn from its
   three-point range.
2. **Risk events** from the risk register either happen or don't (by their
   probability). If one happens it adds cost, and adds days to the task it affects.
3. **Time-dependent cost**: part of each task's cost is people and equipment paid
   by the day, so a task that runs long also costs more. This is what ties cost to
   schedule.
4. **Critical Path Method** turns the task durations into a project finish.

From those runs come the outputs project boards ask for:

* **Joint confidence level (JCL)**: the chance of finishing on budget *and* on
  time together. NASA budgets major projects at a 70% JCL; P80 cost and P80
  date taken separately give a joint chance well below 80%.
* **Contingency split**: how much of the contingency covers estimating
  uncertainty and how much covers specific named risks.
* **Contingency allocation** of the P80 contingency to milestones, so each
  milestone owner holds the share their work creates.
* **Risk ranking**: what each risk adds to the P80 cost and P80 finish if it is
  removed (the payoff from mitigating it).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .loader import has_schedule
from .model import DISPLAY_NAMES, Settings, _pert_ppf, simulate, three_point_estimates
from .schedule import (ScheduleResult, cpm, duration_estimates, resolve_predecessors, sample_durations,
                       topological_order)


@dataclass
class IntegratedResult:
    distribution: str
    settings: Settings
    estimates: pd.DataFrame
    risks: pd.DataFrame | None
    task_costs: np.ndarray           # (n, k) after time-dependent adjustment
    risk_costs: np.ndarray           # (n, r) cost of each risk event (0 when it didn't happen)
    risk_days: np.ndarray            # (n, r) days each risk added to its task
    total_cost: np.ndarray           # (n,)
    schedule: ScheduleResult | None

    @property
    def name(self) -> str:
        return DISPLAY_NAMES[self.distribution]

    @property
    def base_cost(self) -> float:
        return float(self.estimates["Cost"].sum())

    @property
    def project_days(self) -> np.ndarray | None:
        return None if self.schedule is None else self.schedule.project_days


def _risk_sample(u, lo, ml, hi, lam=4.0):
    lo, ml, hi = (np.full_like(u, v, dtype=float) for v in (lo, ml, hi))
    return _pert_ppf(u, lo, ml, hi, lam)


def run_integrated(tasks: pd.DataFrame, risks: pd.DataFrame | None = None, distribution: str = "pert",
                   settings: Settings | None = None, use_register: bool = True,
                   exclude_risks: tuple = ()) -> IntegratedResult:
    """Run cost (and schedule, when durations exist) with optional risk-register events.

    When the register is used, per-task risk multipliers are switched off so risk is not counted
    twice. ``exclude_risks`` drops named risks (used to measure what each risk contributes)."""
    s = settings or Settings()
    reg = risks if (use_register and risks is not None and len(risks)) else None
    if reg is not None:
        reg = reg[~reg["Risk ID"].isin(exclude_risks)].reset_index(drop=True)
        s = replace(s, risk_mode="none")
    est = three_point_estimates(tasks, s)
    cost = simulate(est, distribution, s).task_samples
    n, k = cost.shape
    rng = np.random.default_rng(None if s.seed is None else s.seed + 1)

    sched = has_schedule(tasks)
    dur = None
    if sched:
        dest = duration_estimates(est, s)
        dur = sample_durations(dest, distribution, s, rng)
        dur_ml = dest["Dur ML"].to_numpy(float)

    r = 0 if reg is None else len(reg)
    risk_costs = np.zeros((n, r))
    risk_days = np.zeros((n, r))
    extra_project_days = np.zeros(n)
    codes = est["WBS Code"].tolist()
    # Every risk gets its own random streams so excluding one never reshuffles the others.
    for j in range(r):
        row = reg.iloc[j]
        seed = None if s.seed is None else s.seed + 100 + zlib.crc32(str(row["Risk ID"]).encode()) % 10_000
        rr = np.random.default_rng(seed)
        happens = rr.random(n) < float(row["Probability"])
        uc, ud = rr.random(n), rr.random(n)
        risk_costs[:, j] = happens * _risk_sample(uc, row["Cost Min"], row["Cost ML"], row["Cost Max"])
        if sched:
            days = happens * _risk_sample(ud, row["Days Min"], row["Days ML"], row["Days Max"])
            risk_days[:, j] = days
            target = str(row.get("Affects", "") or "")
            if target in codes:
                dur[:, codes.index(target)] += days
            else:
                extra_project_days += days  # project-level delay

    schedule = None
    if sched:
        td = est["Time-dependent %"].astype(float)
        td = td.where(td.notna(), s.time_dependent_share)
        td = np.where(td > 1, td / 100, td).clip(0, 1)
        base_dur = np.maximum(dur_ml, 1e-9)
        task_only = dur.copy()
        if r:
            for j in range(r):
                target = str(reg.iloc[j].get("Affects", "") or "")
                if target in codes:
                    task_only[:, codes.index(target)] -= risk_days[:, j]
        # Estimate-driven duration changes move time-dependent cost; risk delays are priced in the risk's
        # own cost impact, so they are not charged twice.
        cost = cost * ((1 - td) + td * task_only / base_dur)
        preds = resolve_predecessors(est)
        order = topological_order(preds)
        ef, tf, project = cpm(dur, preds, order)
        project = project + extra_project_days
        _, _, det = cpm(dur_ml[None, :], preds, order)
        schedule = ScheduleResult(est["Label"].tolist(), codes, dur, ef, tf, project, float(det[0]),
                                  pd.Timestamp(s.start_date))
    total = cost.sum(axis=1) + risk_costs.sum(axis=1)
    return IntegratedResult(distribution, s, est, reg, cost, risk_costs, risk_days, total, schedule)


# ---------------------------------------------------------------------------
# Joint confidence
# ---------------------------------------------------------------------------
def jcl(res: IntegratedResult, budget: float, days: float) -> float:
    """Probability of finishing within budget AND within the given working days."""
    if res.project_days is None:
        return float((res.total_cost <= budget).mean())
    return float(((res.total_cost <= budget) & (res.project_days <= days)).mean())


def budget_for_jcl(res: IntegratedResult, days: float, target: float = 0.7) -> float | None:
    """Smallest budget that reaches the target joint confidence for a given duration (None if impossible)."""
    if res.project_days is None:
        return float(np.quantile(res.total_cost, target))
    ok = np.sort(res.total_cost[res.project_days <= days])
    need = int(np.ceil(target * len(res.total_cost)))
    return None if len(ok) < need else float(ok[need - 1])


def jcl_grid(res: IntegratedResult, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JCL over a grid of budgets x durations, for the contour chart."""
    b = np.linspace(*np.quantile(res.total_cost, [0.01, 0.995]), n)
    d = np.linspace(*np.quantile(res.project_days, [0.01, 0.995]), n)
    c, t = res.total_cost, res.project_days
    z = np.array([[((c <= bb) & (t <= dd)).mean() for bb in b] for dd in d])
    return b, d, z


# ---------------------------------------------------------------------------
# Contingency
# ---------------------------------------------------------------------------
def contingency_split(tasks, risks, distribution="pert", settings: Settings | None = None, level: float = 0.8) -> dict:
    """Split the contingency at a confidence level into estimate uncertainty and named risks."""
    s = settings or Settings()
    with_r = run_integrated(tasks, risks, distribution, s, use_register=True)
    without = run_integrated(tasks, None, distribution, replace(s, risk_mode="none"))
    base = with_r.base_cost
    q = lambda x: float(np.quantile(x, level))
    out = {"level": level, "base_cost": base,
           "estimate_uncertainty": q(without.total_cost) - base,
           "risk_events": q(with_r.total_cost) - q(without.total_cost),
           "total_budget": q(with_r.total_cost)}
    if with_r.schedule is not None:
        out.update({"base_days": with_r.schedule.deterministic_days,
                    "schedule_estimate_days": q(without.project_days) - with_r.schedule.deterministic_days,
                    "schedule_risk_days": q(with_r.project_days) - q(without.project_days),
                    "total_days": q(with_r.project_days)})
    return out


def allocate_contingency(res: IntegratedResult, level: float = 0.8) -> pd.DataFrame:
    """Share the contingency at `level` across milestones.

    Each milestone gets (its expected overrun) + (its share of the variance) x (P-level minus mean).
    Risk costs are assigned to the milestone of the task they affect; others are 'Project-level risks'.
    The allocations add up exactly to the total contingency."""
    est = res.estimates
    groups = {ms: [res.estimates.index.get_loc(i) for i in g.index] for ms, g in est.groupby("Milestone", sort=False)}
    cols, base = {}, {}
    for ms, idx in groups.items():
        cols[ms] = res.task_costs[:, idx].sum(axis=1)
        base[ms] = float(est.iloc[idx]["Cost"].sum())
    if res.risks is not None:
        code_to_ms = dict(zip(est["WBS Code"], est["Milestone"]))
        for j, row in res.risks.reset_index(drop=True).iterrows():
            ms = code_to_ms.get(str(row.get("Affects", "") or ""), "Project-level risks")
            cols.setdefault(ms, np.zeros_like(res.total_cost))
            base.setdefault(ms, 0.0)
            cols[ms] = cols[ms] + res.risk_costs[:, j]
    total = res.total_cost
    spread = float(np.quantile(total, level)) - total.mean()
    tc = total - total.mean()
    var = float((tc ** 2).mean())
    rows = []
    for ms, x in cols.items():
        share = float(((x - x.mean()) * tc).mean() / var) if var else 0.0
        alloc = (x.mean() - base[ms]) + share * spread
        rows.append({"Milestone": ms, "Base estimate": base[ms], "Expected cost": float(x.mean()),
                     "Share of variance": share, "Contingency": alloc,
                     f"P{int(level * 100)} budget": base[ms] + alloc})
    return pd.DataFrame(rows)


def risk_ranking(tasks, risks, distribution="pert", settings: Settings | None = None, level: float = 0.8) -> pd.DataFrame:
    """For each risk: expected cost, and how much P80 cost / P80 finish fall if it is removed."""
    if risks is None or not len(risks):
        return pd.DataFrame()
    s = settings or Settings()
    full = run_integrated(tasks, risks, distribution, s)
    q = lambda x: float(np.quantile(x, level))
    rows = []
    for j, row in full.risks.iterrows():
        without = run_integrated(tasks, risks, distribution, s, exclude_risks=(row["Risk ID"],))
        rec = {"Risk ID": row["Risk ID"], "Risk": row["Risk"], "Probability": row["Probability"],
               "Owner": row.get("Owner", ""), "Affects": row.get("Affects", ""),
               "Expected cost": float(full.risk_costs[:, j].mean()),
               f"P{int(level * 100)} cost saved if removed": q(full.total_cost) - q(without.total_cost)}
        if full.schedule is not None:
            rec["Expected delay (days)"] = float(full.risk_days[:, j].mean())
            rec[f"P{int(level * 100)} days saved if removed"] = q(full.project_days) - q(without.project_days)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(f"P{int(level * 100)} cost saved if removed", ascending=False).reset_index(drop=True)


def as_result(res: IntegratedResult):
    """Wrap an integrated run as a classic cost Result. Risk events appear as extra 'tasks',
    so the tornado chart ranks named risks alongside tasks."""
    from .model import Result
    labels = res.estimates["Label"].tolist()
    samples = res.task_costs
    if res.risks is not None and len(res.risks):
        labels += [f"Risk {r['Risk ID']} - {r['Risk']}" for _, r in res.risks.iterrows()]
        samples = np.hstack([samples, res.risk_costs])
    return Result(res.distribution, labels, samples, res.total_cost, res.settings)
