"""Schedule risk analysis: simulate task durations through the dependency network.

Plain-English summary
---------------------
A cost estimate can be added up task by task. A schedule cannot: the project
finishes when its *longest chain of dependent tasks* (the critical path)
finishes, and which chain is longest changes from one simulation to the next.
So for every simulation we run the Critical Path Method (CPM):

* forward pass  - earliest start / finish of each task, given its predecessors
* backward pass - latest finish each task can have without delaying the project
* float         - the slack between the two. Zero float = on the critical path

Across 10,000 runs this gives:

* the finish-date range (P50, P80 ...) instead of one deterministic date
* the **criticality index**: the share of runs in which each task was on the
  critical path. A task that is critical 95% of the time is where to manage;
  one at 5% can slip without consequence
* **schedule sensitivity**: how strongly each task's duration moves the end date

Everything is vectorised: each pass handles all simulations at once, task by
task in dependency order, so 10,000 runs of a 20-task network take milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Settings, _lognormal_ppf, _pert_ppf, _triangular_ppf, _uniforms


@dataclass
class ScheduleResult:
    labels: list[str]
    codes: list[str]
    durations: np.ndarray      # (n_sims, n_tasks) working days
    early_finish: np.ndarray   # (n_sims, n_tasks)
    total_float: np.ndarray    # (n_sims, n_tasks)
    project_days: np.ndarray   # (n_sims,)
    deterministic_days: float  # most-likely durations, no uncertainty
    start_date: pd.Timestamp

    @property
    def criticality(self) -> np.ndarray:
        return (self.total_float <= 1e-9).mean(axis=0)

    def finish_date(self, days) -> pd.Timestamp:
        """Working days from the start date -> calendar date (Mon-Fri)."""
        d = np.datetime64(self.start_date.date(), "D")
        return pd.Timestamp(np.busday_offset(d, int(np.ceil(days)), roll="forward"))


def duration_estimates(tasks: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Optimistic / most-likely / pessimistic durations, defaulting to the schedule factors."""
    out = tasks.copy()
    ml = out["Duration"].astype(float)
    out["Dur ML"] = ml
    out["Dur Opt"] = out["Optimistic Duration"].where(out["Optimistic Duration"].notna(),
                                                     ml * s.duration_optimistic_factor)
    out["Dur Pes"] = out["Pessimistic Duration"].where(out["Pessimistic Duration"].notna(),
                                                      ml * s.duration_pessimistic_factor)
    bad = out[(out["Dur Opt"] > out["Dur ML"]) | (out["Dur ML"] > out["Dur Pes"])]
    if not bad.empty:
        raise ValueError("Each task needs Optimistic <= Most Likely <= Pessimistic duration. Check: "
                         + ", ".join(bad["Label"]))
    return out


def resolve_predecessors(tasks: pd.DataFrame) -> list[list[int]]:
    """Map each task's predecessor codes to task indices.

    A predecessor can be a task ('1.2') or a milestone/summary code ('1'), which means
    'after every task under 1'. Unknown codes raise an error rather than being ignored."""
    codes = tasks["WBS Code"].tolist()
    index = {c: i for i, c in enumerate(codes)}
    preds = []
    for i, raw in enumerate(tasks["Predecessors"].fillna("")):
        out = []
        for code in [c for c in str(raw).split(",") if c]:
            if code in index:
                out.append(index[code])
            else:
                below = [j for j, c in enumerate(codes) if c.startswith(code + ".")]
                if not below:
                    raise ValueError(f"Task {codes[i]} lists predecessor '{code}', which is not in the WBS.")
                out.extend(below)
        preds.append(sorted(set(j for j in out if j != i)))
    return preds


def topological_order(preds: list[list[int]]) -> list[int]:
    n = len(preds)
    succs = [[] for _ in range(n)]
    indeg = [len(p) for p in preds]
    for i, p in enumerate(preds):
        for j in p:
            succs[j].append(i)
    ready = [i for i in range(n) if indeg[i] == 0]
    order = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for k in succs[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.append(k)
    if len(order) != n:
        raise ValueError("The dependency network has a loop: a task cannot (indirectly) depend on itself.")
    return order


def cpm(durations: np.ndarray, preds: list[list[int]], order: list[int]):
    """Vectorised Critical Path Method. durations: (n_sims, n_tasks). Returns (EF, total float, project)."""
    n_sims, n = durations.shape
    es = np.zeros_like(durations)
    ef = np.zeros_like(durations)
    for i in order:
        if preds[i]:
            es[:, i] = ef[:, preds[i]].max(axis=1)
        ef[:, i] = es[:, i] + durations[:, i]
    project = ef.max(axis=1)
    succs = [[] for _ in range(n)]
    for i, p in enumerate(preds):
        for j in p:
            succs[j].append(i)
    lf = np.zeros_like(durations)
    ls = np.zeros_like(durations)
    for i in reversed(order):
        lf[:, i] = ls[:, succs[i]].min(axis=1) if succs[i] else project
        ls[:, i] = lf[:, i] - durations[:, i]
    return ef, ls - es, project


def sample_durations(est: pd.DataFrame, distribution: str, s: Settings, rng) -> np.ndarray:
    a, m, b = (est[c].to_numpy(float) for c in ("Dur Opt", "Dur ML", "Dur Pes"))
    u = np.clip(_uniforms(s.n_sims, len(est), s.correlation, rng), 1e-12, 1 - 1e-12)
    if distribution == "triangular":
        return _triangular_ppf(u, a, m, b)
    if distribution == "pert":
        return _pert_ppf(u, a, m, b, s.pert_lambda)
    return _lognormal_ppf(u, np.maximum(a, 1e-9), np.maximum(m, 1e-9), np.maximum(b, 1e-9))


def schedule_sensitivity(sr: ScheduleResult) -> pd.DataFrame:
    """Criticality index plus 'cruciality': correlation of task duration with project duration."""
    d, total = sr.durations, sr.project_days
    dc, tc = d - d.mean(axis=0), total - total.mean()
    corr = (dc * tc[:, None]).mean(axis=0) / (d.std(axis=0) * total.std() + 1e-12)
    return (pd.DataFrame({"Task": sr.labels, "Mean duration (days)": d.mean(axis=0),
                          "Criticality index": sr.criticality, "Correlation with finish": corr,
                          "Cruciality": sr.criticality * np.clip(corr, 0, None)})
            .sort_values("Cruciality", ascending=False).reset_index(drop=True))
