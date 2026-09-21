"""Write every result to one Excel workbook."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .analysis import milestone_summary, project_summary, sensitivity
from .model import Result


def export_excel(results: dict[str, Result], estimates: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = next(iter(results.values())).settings

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        project_summary(results, estimates).round(2).to_excel(
            xl, sheet_name="Project Summary", index=False)

        pd.concat([milestone_summary(r, estimates) for r in results.values()]).round(2).to_excel(
            xl, sheet_name="Milestones", index=False)

        sens = None
        for r in results.values():
            s = sensitivity(r)[["Task", "Share of variance (%)", "Correlation with total"]]
            s = s.rename(columns={c: f"{c} - {r.name}" for c in s.columns if c != "Task"})
            sens = s if sens is None else sens.merge(s, on="Task", how="outer")
        sens.round(4).to_excel(xl, sheet_name="Sensitivity", index=False)

        cols = ["WBS Code", "Task", "Milestone", "Cost", "Risk Multiplier",
                "Optimistic", "Most Likely", "Pessimistic"]
        estimates[cols].round(2).to_excel(xl, sheet_name="Inputs (3-point)", index=False)

        pd.DataFrame(list(asdict(settings).items()), columns=["Setting", "Value"]).to_excel(
            xl, sheet_name="Assumptions", index=False)

        for sheet in xl.book.worksheets:
            for col in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                sheet.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    return path
