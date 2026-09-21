"""Turn simulated costs into numbers a project team can act on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import Result

PERCENTILES = (10, 50, 80, 90)


def summarize(samples: np.ndarray) -> dict:
    out = {"Mean": float(np.mean(samples)), "Std Dev": float(np.std(samples))}
    for p in PERCENTILES:
        out[f"P{p}"] = float(np.percentile(samples, p))
    out["Min"] = float(np.min(samples))
    out["Max"] = float(np.max(samples))
    return out


def project_summary(results: dict[str, Result], estimates: pd.DataFrame) -> pd.DataFrame:
    """One row per model, plus the contingency each confidence level implies."""
    base = float(estimates["Cost"].sum())               # the plain sum of the estimates
    risked = float(estimates["Most Likely"].sum())      # the same, after risk loading
    rows = []
    for res in results.values():
        s = summarize(res.total)
        s = {"Model": res.name, **s}
        s["Base estimate"] = base
        s["Risk-loaded estimate"] = risked
        for p in (50, 80, 90):
            s[f"Contingency at P{p}"] = s[f"P{p}"] - base
            s[f"Contingency at P{p} (%)"] = (s[f"P{p}"] - base) / base * 100 if base else np.nan
        rows.append(s)
    return pd.DataFrame(rows)


def milestone_summary(result: Result, estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ms, group in estimates.groupby("Milestone", sort=False):
        idx = [result.labels.index(lbl) for lbl in group["Label"]]
        samples = result.task_samples[:, idx].sum(axis=1)
        rows.append({"Model": result.name, "Milestone": ms,
                     "Base estimate": float(group["Cost"].sum()), **summarize(samples)})
    return pd.DataFrame(rows)


def sensitivity(result: Result) -> pd.DataFrame:
    """Which tasks drive the uncertainty in the total?

    * Share of variance: each task's covariance with the total divided by the
      total's variance. The shares add up to 100%, so they read as "how much of
      the spread in the final cost comes from this task".
    * Correlation: Pearson correlation between the task cost and the total.
    * Mean cost: the task's average simulated cost (its size, not its risk).
    """
    x = result.task_samples
    total = result.total
    var_total = np.var(total)
    xc = x - x.mean(axis=0)
    tc = total - total.mean()
    cov = (xc * tc[:, None]).mean(axis=0)
    corr = cov / (x.std(axis=0) * total.std() + 1e-12)
    df = pd.DataFrame({
        "Task": result.labels,
        "Mean cost": x.mean(axis=0),
        "Share of variance (%)": cov / var_total * 100 if var_total else 0.0,
        "Correlation with total": corr,
    })
    return df.sort_values("Share of variance (%)", ascending=False).reset_index(drop=True)
