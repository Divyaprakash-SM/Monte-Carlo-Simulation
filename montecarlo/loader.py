"""Load a Work Breakdown Structure (WBS) cost sheet from Excel or CSV.

The loader is deliberately forgiving about layout:

* the header row is found automatically (it does not have to be row 1),
* columns are matched by name, not by position,
* summary rows (milestones, sub-totals) are detected from the WBS hierarchy
  and excluded, so costs are never counted twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Accepted spellings for each column (compared in lower case, trimmed).
COLUMN_ALIASES = {
    "WBS Code": ["wbs code", "wbs", "wbs id", "code"],
    "Task": ["task", "task name", "activity", "description", "work package"],
    "Cost": ["cost", "most likely", "most likely cost", "base cost", "estimate"],
    "Optimistic": ["optimistic cost", "optimistic", "min cost", "minimum"],
    "Pessimistic": ["pessimistic cost", "pessimistic", "max cost", "maximum"],
    "Risk Impact": ["risk - impact", "risk impact", "impact"],
    "Risk Likelihood": ["risk - likelihood", "risk likelihood", "likelihood"],
    "Risk Score": ["risk - score", "risk score", "score"],
    "Risk Multiplier": ["risk % multiplier", "risk multiplier", "risk %", "multiplier"],
}


def _clean_code(value) -> str:
    """Turn a WBS cell into a clean text code ('1', '1.1', '1.2.1')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else repr(float(value))
    return str(value).strip()


def _read_raw(path: Path, sheet) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None, dtype=object)
    return pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(len(raw), 30)):
        cells = {str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)}
        if cells & set(COLUMN_ALIASES["Task"]) and cells & set(COLUMN_ALIASES["Cost"]):
            return i
    raise ValueError(
        "Could not find a header row. The sheet needs at least a 'Task' column "
        "and a 'Cost' (or 'Most Likely') column."
    )


def _map_columns(header: list) -> dict:
    """Return {standard name: column index} for every column we recognise."""
    lowered = [str(h).strip().lower() if pd.notna(h) else "" for h in header]
    mapping = {}
    for std, aliases in COLUMN_ALIASES.items():
        for idx, name in enumerate(lowered):
            if name in aliases and idx not in mapping.values():
                mapping[std] = idx
                break
    return mapping


def _detect_summaries(df: pd.DataFrame, tol: float = 0.01) -> list[bool]:
    """A row is a summary (sub-total) when its cost equals the sum of the tasks
    beneath it. A parent row whose cost does not add up is a real task with
    child tasks of its own (common in WBS sheets) and is kept.

    Works bottom-up so nested sub-totals are handled correctly.
    """
    codes = df["WBS Code"].tolist()
    costs = df["Cost"].tolist()
    depth = [c.count(".") if c else -1 for c in codes]
    summary = [False] * len(codes)
    for i in sorted(range(len(codes)), key=lambda k: -depth[k]):
        code = codes[i]
        if not code:
            continue
        below = [j for j, c in enumerate(codes)
                 if j != i and c.startswith(code + ".") and not summary[j]]
        if below:
            total = sum(costs[j] for j in below)
            summary[i] = abs(total - costs[i]) <= tol * max(abs(costs[i]), 1)
    return summary


def load_wbs(path, sheet=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a WBS sheet.

    Returns
    -------
    tasks : DataFrame
        One row per leaf task (the rows that are actually simulated).
    summaries : DataFrame
        The summary rows (milestones / sub-totals) that were excluded, kept for
        reference and for the consistency check.
    """
    path = Path(path)
    raw = _read_raw(path, sheet)
    header_row = _find_header_row(raw)
    cols = _map_columns(raw.iloc[header_row].tolist())
    body = raw.iloc[header_row + 1:].reset_index(drop=True)

    df = pd.DataFrame({std: body.iloc[:, idx] for std, idx in cols.items()})
    if "WBS Code" not in df:
        df["WBS Code"] = [str(i + 1) for i in range(len(df))]
    df["WBS Code"] = df["WBS Code"].map(_clean_code)
    df["Task"] = df["Task"].astype(object).where(df["Task"].notna(), "").astype(str).str.strip()
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df = df[(df["Task"] != "") & df["Cost"].notna()].reset_index(drop=True)

    for col in ("Optimistic", "Pessimistic", "Risk Impact", "Risk Likelihood",
                "Risk Score", "Risk Multiplier"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df["Risk Multiplier"] = df["Risk Multiplier"].fillna(0.0)

    df["Is Summary"] = _detect_summaries(df)

    tasks = df[~df["Is Summary"]].copy().reset_index(drop=True)
    summaries = df[df["Is Summary"]].copy().reset_index(drop=True)

    # Group every task under its top-level WBS element (the milestone).
    top_names = {
        row["WBS Code"]: row["Task"]
        for _, row in summaries.iterrows() if "." not in row["WBS Code"]
    }
    tasks["Milestone"] = tasks["WBS Code"].str.split(".").str[0].map(
        lambda c: top_names.get(c, f"WBS {c}")
    )
    tasks["Label"] = tasks["WBS Code"] + " - " + tasks["Task"]
    # Keep labels unique even if a code/name pair repeats in the sheet.
    dup = tasks.groupby("Label").cumcount()
    tasks.loc[dup > 0, "Label"] = tasks["Label"] + " (" + (dup + 1).astype(str) + ")"
    return tasks.drop(columns="Is Summary"), summaries.drop(columns="Is Summary")


def check_subtotals(tasks: pd.DataFrame, summaries: pd.DataFrame) -> list[str]:
    """Explain how parent rows were treated, so the user can confirm it.

    Returns one note per parent row that was kept as a task because its cost
    does not equal the sum of the tasks beneath it.
    """
    notes = []
    for _, row in tasks.iterrows():
        code = row["WBS Code"]
        below = tasks[tasks["WBS Code"].str.startswith(code + ".")]
        if not below.empty:
            notes.append(
                f"WBS {code} ({row['Task']}) has sub-tasks but its cost "
                f"({row['Cost']:,.0f}) is not their total ({below['Cost'].sum():,.0f}), "
                "so it is treated as a task in its own right."
            )
    return notes
