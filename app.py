"""
Monte Carlo cost and schedule risk: web app.

Run locally:  streamlit run app.py
Upload your own WBS (Excel/CSV) or explore the sample project.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from montecarlo import (DISTRIBUTIONS, Settings, allocate_contingency, as_result, budget_for_jcl, contingency_split,
                        export_excel, has_schedule, jcl, jcl_grid, load_risk_register, load_wbs, milestone_summary,
                        project_summary, risk_ranking, run, run_integrated, schedule_sensitivity, sensitivity)
from montecarlo.model import DISPLAY_NAMES

st.set_page_config(page_title="Monte Carlo Risk Model", page_icon="🎲", layout="wide")
SAMPLE = Path(__file__).parent / "data" / "sample_wbs.xlsx"
COL = {"triangular": "#2a78d6", "lognormal": "#eb6834", "pert": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

st.markdown("""<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 650;}
  [data-testid="stMetricLabel"] p {font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; color: #52514e;}
  .note {font-size:.92rem; color:#52514e;}
  .rec {border-left: 4px solid #1c5cab; background:#f3f2ee; padding: 14px 18px; border-radius: 6px; margin: 4px 0 18px;}
  h1 {font-weight: 700; letter-spacing: -.02em;}
</style>""", unsafe_allow_html=True)


def style(fig, height=380, **kw):
    top = 40
    if "title" in kw:
        kw["title"] = {**kw["title"], "x": 0.01, "xanchor": "left", "y": 0.985, "yanchor": "top"}
        top = 78
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=top, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_size=12), **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def money(x, cur):
    return f"{'-' if x < 0 else ''}{cur}{abs(x):,.0f}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Your project")
    up = st.file_uploader("Upload a WBS (Excel or CSV)", type=["xlsx", "xls", "csv"])
    st.download_button("Download the template (sample project)", SAMPLE.read_bytes(), "wbs_template.xlsx",
                       use_container_width=True)
    st.header("2 · Assumptions")
    dist = st.selectbox("Distribution for the detailed views", list(DISTRIBUTIONS), index=2,
                        format_func=lambda d: DISPLAY_NAMES[d],
                        help="Beta-PERT is the usual choice in project management. All three are compared on the Cost tab.")
    risk_choice = st.selectbox("Risk method", ["auto", "register", "loaded", "event", "none"],
                               format_func={"auto": "Auto (register if the file has one)", "register": "Risk register events",
                                            "loaded": "Load task costs by risk %", "event": "Task risks as events",
                                            "none": "Estimate uncertainty only"}.get)
    n_sims = st.select_slider("Simulations", [2_000, 5_000, 10_000, 20_000], 10_000)
    corr = st.slider("Correlation between tasks", 0.0, 0.9, 0.0, 0.05,
                     help="0 = tasks vary independently. 0.3 = moderate shared drift (one delay knocks on to others).")
    with st.expander("Default ranges and schedule"):
        lo = st.number_input("Cost: optimistic factor", 0.5, 1.0, 0.90, 0.05)
        hi = st.number_input("Cost: pessimistic factor", 1.0, 3.0, 1.20, 0.05)
        dlo = st.number_input("Duration: optimistic factor", 0.5, 1.0, 0.90, 0.05)
        dhi = st.number_input("Duration: pessimistic factor", 1.0, 3.0, 1.35, 0.05)
        td = st.slider("Time-dependent share of cost (when not in file)", 0.0, 1.0, 0.5, 0.05)
        start = st.date_input("Project start", pd.Timestamp("2026-01-05"))
    cur = st.text_input("Currency symbol", "£")


@st.cache_data(show_spinner=False)
def load(data: bytes | None, name: str):
    path = SAMPLE
    if data is not None:
        tmp = Path(tempfile.mkdtemp()) / name
        tmp.write_bytes(data)
        path = tmp
    tasks, summaries = load_wbs(path)
    return tasks, summaries, load_risk_register(path)


try:
    tasks, summaries, register = load(up.getvalue() if up else None, up.name if up else "sample.xlsx")
except Exception as e:  # show the loader's plain-English message instead of a traceback
    st.error(f"Could not read that file: {e}")
    st.stop()

risk = risk_choice if risk_choice != "auto" else ("register" if register is not None else "loaded")
if risk == "register" and register is None:
    st.warning("This file has no 'Risk Register' sheet, so risks are loaded onto task costs instead.")
    risk = "loaded"
S = Settings(n_sims=n_sims, optimistic_factor=lo, pessimistic_factor=hi, risk_mode=risk, correlation=corr,
             duration_optimistic_factor=dlo, duration_pessimistic_factor=dhi, time_dependent_share=td,
             start_date=str(start))
schedule = has_schedule(tasks)
use_reg = risk == "register"
integrated_mode = use_reg or schedule
reg = register if use_reg else None


@st.cache_data(show_spinner="Running the simulations…")
def simulate_all(key, _tasks, _reg, _s):
    if integrated_mode:
        ires = {d: run_integrated(_tasks, _reg, d, _s, use_register=use_reg) for d in DISTRIBUTIONS}
        return next(iter(ires.values())).estimates, {d: as_result(r) for d, r in ires.items()}, ires
    est, res = run(_tasks, DISTRIBUTIONS, _s)
    return est, res, {}


@st.cache_data(show_spinner="Measuring each risk and the contingency split…")
def extras(key, _tasks, _reg, _s, d):
    split = contingency_split(_tasks, _reg, d, _s)
    rank = risk_ranking(_tasks, _reg, d, _s) if _reg is not None else pd.DataFrame()
    return split, rank


key = (up.file_id if up else "sample", repr(S), risk)
estimates, results, ires = simulate_all(key, tasks, reg, S)
R = results[dist]
I = ires.get(dist)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Monte Carlo Cost & Schedule Risk")
st.markdown(f"<p class='note'>{'Your file' if up else 'Sample project'}: <b>{len(tasks)} tasks</b> "
            f"({len(summaries)} milestone/sub-total rows excluded so nothing is counted twice)"
            f"{f', <b>{len(register)} risks</b> in the register' if register is not None else ''}"
            f"{', <b>dependencies and durations</b> found, so the schedule is simulated too' if schedule else ''}. "
            f"{n_sims:,} simulations · risk method: <b>{risk}</b>.</p>", unsafe_allow_html=True)

base = float(estimates["Cost"].sum())
p50, p80 = np.percentile(R.total, [50, 80])
c = st.columns(5 if schedule else 4)
c[0].metric("Base estimate", money(base, cur), help="The plain sum of the task estimates: no uncertainty, no risk.")
c[1].metric("P50 cost", money(p50, cur), f"{(p50 - base) / base:+.0%} vs base", delta_color="off")
c[2].metric("P80 cost (budget)", money(p80, cur), f"contingency {money(p80 - base, cur)}", delta_color="off")
c[3].metric("Chance base estimate is enough", f"{(R.total <= base).mean():.0%}")
if schedule and I is not None:
    sr = I.schedule
    d80 = np.percentile(sr.project_days, 80)
    c[4].metric("P80 finish", f"{sr.finish_date(d80):%d %b %Y}",
                f"plan {sr.finish_date(sr.deterministic_days):%d %b} · {(sr.project_days <= sr.deterministic_days).mean():.0%} chance",
                delta_color="off")

tabs = st.tabs(["Cost", "Schedule", "Joint confidence", "Risks & contingency", "Inputs & report"])

# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
with tabs[0]:
    l, r = st.columns([1.3, 1])
    with l:
        fig = go.Figure()
        for d, res in results.items():
            xs = np.sort(res.total)[:: max(1, n_sims // 600)]
            fig.add_trace(go.Scatter(x=xs, y=np.linspace(0, 1, len(xs)), name=DISPLAY_NAMES[d], mode="lines",
                                     line=dict(color=COL[d], width=2.5 if d == dist else 1.5),
                                     hovertemplate=f"{DISPLAY_NAMES[d]}: %{{y:.0%}} chance ≤ {cur}%{{x:,.0f}}<extra></extra>"))
        fig.add_vline(x=base, line=dict(color=INK, width=1, dash="dot"))
        fig.add_annotation(x=base, y=0.98, text="Base estimate", showarrow=False, xanchor="right", font=dict(size=11, color=MUTED))
        fig.add_hline(y=0.8, line=dict(color=MUTED, width=1, dash="dash"))
        style(fig, 400, title=dict(text="Chance the project comes in at or under each cost", font=dict(size=15)))
        fig.update_xaxes(tickprefix=cur, tickformat=",.0f"); fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(fig, use_container_width=True)
    with r:
        summ = project_summary(results, estimates)
        st.markdown("##### The three distributions compared")
        st.dataframe(summ[["Model", "P10", "P50", "P80", "P90", "Contingency at P80 (%)"]], hide_index=True,
                     use_container_width=True,
                     column_config={k: st.column_config.NumberColumn(format=f"{cur}%,.0f") for k in ("P10", "P50", "P80", "P90")}
                     | {"Contingency at P80 (%)": st.column_config.NumberColumn("P80 contingency", format="%.1f%%")})
        st.caption("All three share the same random numbers, so differences come from the distribution alone. "
                   "Lognormal has the longest overrun tail; Triangular puts the most weight on the extremes.")
    l, r = st.columns(2)
    with l:
        sens = sensitivity(R).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=sens["Share of variance (%)"], y=sens["Task"], orientation="h",
                               marker=dict(color=[("#eb6834" if t.startswith("Risk ") else COL[dist]) for t in sens["Task"]], cornerradius=4),
                               text=[f"{v:.1f}%" for v in sens["Share of variance (%)"]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:.1f}% of the variance<extra></extra>"))
        style(fig, 440, title=dict(text="What drives the uncertainty (orange = named risk)", font=dict(size=15)))
        fig.update_xaxes(ticksuffix="%", range=[0, sens["Share of variance (%)"].max() * 1.2])
        st.plotly_chart(fig, use_container_width=True)
    with r:
        ms = milestone_summary(R, estimates).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ms.P90, y=ms.Milestone, mode="markers", marker=dict(size=0.1), showlegend=False, hoverinfo="skip"))
        for row in ms.itertuples():
            fig.add_shape(type="line", x0=row.P10, x1=row.P90, y0=row.Milestone, y1=row.Milestone,
                          line=dict(color=COL[dist], width=3))
        fig.add_trace(go.Scatter(x=ms.P50, y=ms.Milestone, mode="markers", name="P50 (line = P10 to P90)",
                                 marker=dict(size=12, color=COL[dist], line=dict(color="white", width=2)),
                                 hovertemplate=f"%{{y}}<br>P50 {cur}%{{x:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ms["Base estimate"], y=ms.Milestone, mode="markers", name="Base estimate",
                                 marker=dict(size=10, symbol="diamond-open", color=INK),
                                 hovertemplate=f"%{{y}}<br>Base {cur}%{{x:,.0f}}<extra></extra>"))
        style(fig, 440, title=dict(text="Cost range by milestone (tasks only)", font=dict(size=15)))
        fig.update_xaxes(tickprefix=cur, tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
with tabs[1]:
    if not schedule or I is None:
        st.info("Add **Duration (days)** and **Predecessors** columns to your WBS to simulate the schedule. "
                "The template shows the format.")
    else:
        sr = I.schedule
        det = sr.deterministic_days
        l, r = st.columns([1.3, 1])
        with l:
            days = np.sort(sr.project_days)[:: max(1, n_sims // 600)]
            dates = [sr.finish_date(d) for d in days]
            fig = go.Figure(go.Scatter(x=dates, y=np.linspace(0, 1, len(dates)), mode="lines",
                                       line=dict(color=COL[dist], width=2.5),
                                       hovertemplate="%{y:.0%} chance of finishing by %{x|%d %b %Y}<extra></extra>"))
            plan = sr.finish_date(det)
            fig.add_vline(x=plan, line=dict(color=INK, width=1, dash="dot"))
            fig.add_annotation(x=plan, y=0.98, text=f"Plan {plan:%d %b}", showarrow=False, xanchor="right",
                               font=dict(size=11, color=MUTED))
            fig.add_hline(y=0.8, line=dict(color=MUTED, width=1, dash="dash"))
            style(fig, 400, title=dict(text="Chance of finishing by each date", font=dict(size=15)))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
            st.plotly_chart(fig, use_container_width=True)
        with r:
            on_plan = (sr.project_days <= det).mean()
            st.markdown(f"""<div class="rec">The plan says <b>{plan:%d %b %Y}</b> ({det:.0f} working days), but only
            <b>{on_plan:.0%}</b> of simulations finish by then. P50 is <b>{sr.finish_date(np.percentile(sr.project_days, 50)):%d %b}</b>,
            P80 <b>{sr.finish_date(np.percentile(sr.project_days, 80)):%d %b %Y}</b>.<br><br>
            Why is the plan so optimistic? Wherever parallel paths merge, the project waits for the <i>slowest</i> one,
            so uncertainty only ever pushes the date later. This is called <b>merge bias</b>, and a single-date plan can't see it.</div>""",
                        unsafe_allow_html=True)
        ss = schedule_sensitivity(sr)
        l, r = st.columns([1, 1.2])
        with l:
            d = ss.sort_values(["Criticality index", "Cruciality"], ascending=False).head(12).iloc[::-1]
            fig = go.Figure(go.Bar(x=d["Criticality index"], y=d["Task"], orientation="h",
                                   marker=dict(color=COL[dist], cornerradius=4), text=[f"{v:.0%}" for v in d["Criticality index"]],
                                   textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>On the critical path in %{x:.0%} of runs<extra></extra>"))
            style(fig, 440, title=dict(text="Criticality index: how often each task is critical", font=dict(size=15)))
            fig.update_xaxes(tickformat=".0%", range=[0, 1.22])
            st.plotly_chart(fig, use_container_width=True)
        with r:
            # Deterministic Gantt, coloured by criticality
            from montecarlo.schedule import cpm, resolve_predecessors, topological_order
            from montecarlo.schedule import duration_estimates
            dest = duration_estimates(I.estimates, S)
            preds = resolve_predecessors(dest)
            order = topological_order(preds)
            ml = dest["Dur ML"].to_numpy(float)[None, :]
            ef, _, _ = cpm(ml, preds, order)
            g = pd.DataFrame({"Task": dest["Label"], "start": (ef[0] - ml[0]), "dur": ml[0], "crit": sr.criticality})
            g = g.iloc[::-1]
            fig = go.Figure(go.Bar(x=g.dur, y=g.Task, base=g.start, orientation="h",
                                   marker=dict(color=g.crit, colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], cmin=0, cmax=1,
                                               colorbar=dict(title="Critical", tickformat=".0%"), cornerradius=3),
                                   customdata=g[["crit", "dur"]],
                                   hovertemplate="%{y}<br>%{customdata[1]:.0f} days · critical in %{customdata[0]:.0%} of runs<extra></extra>"))
            style(fig, 440, title=dict(text="Plan (most-likely durations), shaded by criticality", font=dict(size=15)))
            fig.update_xaxes(title="Working days from start")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ss, hide_index=True, use_container_width=True,
                     column_config={"Mean duration (days)": st.column_config.NumberColumn(format="%.1f"),
                                    "Criticality index": st.column_config.NumberColumn(format="percent"),
                                    "Correlation with finish": st.column_config.NumberColumn(format="%.2f"),
                                    "Cruciality": st.column_config.NumberColumn(format="%.2f",
                                                                                help="Criticality × correlation: where to focus schedule management")})

# ---------------------------------------------------------------------------
# Joint confidence
# ---------------------------------------------------------------------------
with tabs[2]:
    if not schedule or I is None:
        st.info("Joint confidence needs a schedule: add Duration and Predecessors columns.")
    else:
        sr = I.schedule
        st.markdown("<p class='note'>A budget at P80 and a date at P80 do <b>not</b> give an 80% chance of hitting both. "
                    "Joint confidence (JCL) counts the simulations that meet the budget <i>and</i> the date. NASA requires "
                    "major projects to be budgeted at a 70% JCL.</p>", unsafe_allow_html=True)
        c = st.columns(3)
        b_in = c[0].number_input(f"Budget ({cur})", value=float(round(np.percentile(I.total_cost, 80), -2)), step=500.0)
        d_in = c[1].number_input("Working days", value=float(round(np.percentile(sr.project_days, 80))), step=1.0)
        target = c[2].slider("Target joint confidence", 0.5, 0.95, 0.70, 0.05)
        jc = jcl(I, b_in, d_in)
        need = budget_for_jcl(I, d_in, target)
        st.markdown(f"""<div class="rec">Chance of finishing within <b>{money(b_in, cur)}</b> and by
        <b>{sr.finish_date(d_in):%d %b %Y}</b> ({d_in:.0f} working days): <b>{jc:.0%}</b>.
        {f"To reach {target:.0%} joint confidence with that date, budget <b>{money(need, cur)}</b>." if need else
        f"No budget reaches {target:.0%} with that date: fewer than {target:.0%} of runs finish in time. Move the date first."}</div>""",
                    unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            idx = np.random.default_rng(0).choice(len(I.total_cost), min(3000, len(I.total_cost)), replace=False)
            cc, dd = I.total_cost[idx], sr.project_days[idx]
            ok = (cc <= b_in) & (dd <= d_in)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=dd[~ok], y=cc[~ok], mode="markers", name="Over budget or late",
                                       marker=dict(size=4, color="#c9c8c2", opacity=0.6)))
            fig.add_trace(go.Scattergl(x=dd[ok], y=cc[ok], mode="markers", name="On budget and on time",
                                       marker=dict(size=4, color=COL[dist], opacity=0.6)))
            fig.add_hline(y=b_in, line=dict(color=INK, width=1, dash="dash"))
            fig.add_vline(x=d_in, line=dict(color=INK, width=1, dash="dash"))
            style(fig, 440, title=dict(text="Each dot is one simulated project", font=dict(size=15)))
            fig.update_xaxes(title="Working days"); fig.update_yaxes(tickprefix=cur, tickformat=",.0f", title="Total cost")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            bx, dy, z = jcl_grid(I)
            fig = go.Figure(go.Contour(x=bx, y=dy, z=z, contours=dict(start=0.1, end=0.9, size=0.1, showlabels=True,
                                                                     labelfont=dict(size=11, color="white")),
                                       colorscale=[[0, "#f0efec"], [0.5, "#6da7ec"], [1, "#0d366b"]], showscale=False,
                                       hovertemplate=f"Budget {cur}%{{x:,.0f}} · %{{y:.0f}} days<br>JCL %{{z:.0%}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[b_in], y=[d_in], mode="markers", marker=dict(size=12, color="#eb6834", line=dict(color="white", width=2)),
                                     name="Your budget and date"))
            style(fig, 440, title=dict(text="Joint confidence for every budget and duration", font=dict(size=15)))
            fig.update_xaxes(tickprefix=cur, tickformat=",.0f", title="Budget"); fig.update_yaxes(title="Working days")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Read the contour like a map: every point on the 0.7 line is a budget-and-date pair with a 70% chance "
                   "of meeting both. Moving along the line trades money for time.")

# ---------------------------------------------------------------------------
# Risks & contingency
# ---------------------------------------------------------------------------
with tabs[3]:
    if I is None:
        st.info("Add a 'Risk Register' sheet (see the template) or durations to unlock contingency analysis.")
    else:
        split, rank = extras(key + (dist,), tasks, reg, S, dist)
        l, r = st.columns([1, 1.2])
        with l:
            steps = ["Base estimate", "Estimate uncertainty", "Named risks", "P80 budget"]
            fig = go.Figure(go.Waterfall(x=steps, measure=["absolute", "relative", "relative", "total"],
                                         y=[split["base_cost"], split["estimate_uncertainty"], split["risk_events"], 0],
                                         text=[money(v, cur) for v in (split["base_cost"], split["estimate_uncertainty"],
                                                                       split["risk_events"], split["total_budget"])],
                                         textposition="outside", cliponaxis=False,
                                         increasing=dict(marker=dict(color="#eb6834")), totals=dict(marker=dict(color=INK)),
                                         decreasing=dict(marker=dict(color=COL[dist])),
                                         connector=dict(line=dict(color=MUTED, dash="dot", width=1))))
            style(fig, 400, showlegend=False, title=dict(text="Where the P80 contingency comes from", font=dict(size=15)))
            fig.update_yaxes(tickprefix=cur, tickformat=",.0f", range=[split["base_cost"] * 0.8, split["total_budget"] * 1.06])
            st.plotly_chart(fig, use_container_width=True)
            if "schedule_risk_days" in split:
                st.caption(f"Schedule: plan {split['base_days']:.0f} days + {split['schedule_estimate_days']:.0f} for estimate "
                           f"uncertainty + {split['schedule_risk_days']:.0f} for named risks = P80 {split['total_days']:.0f} days.")
        with r:
            alloc = allocate_contingency(I)
            st.markdown("##### Contingency allocated to milestones")
            st.dataframe(alloc, hide_index=True, use_container_width=True,
                         column_config={"Base estimate": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
                                        "Expected cost": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
                                        "Share of variance": st.column_config.NumberColumn(format="percent"),
                                        "Contingency": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
                                        "P80 budget": st.column_config.NumberColumn(format=f"{cur}%,.0f")})
            st.caption("Each milestone holds its expected overrun plus its share of the spread. The allocations add up "
                       "exactly to the total P80 contingency, so every pound has an owner.")
        if len(rank):
            st.subheader("Which risks are worth mitigating?")
            cost_col = [c for c in rank.columns if c.endswith("cost saved if removed")][0]
            l, r = st.columns([1.2, 1])
            with l:
                d = rank.iloc[::-1]
                fig = go.Figure(go.Bar(x=d[cost_col], y=d["Risk ID"] + " · " + d["Risk"].str.slice(0, 45), orientation="h",
                                       marker=dict(color="#eb6834", cornerradius=4), text=[money(v, cur) for v in d[cost_col]],
                                       textposition="outside", cliponaxis=False,
                                       hovertemplate=f"%{{y}}<br>P80 falls by {cur}%{{x:,.0f}} if removed<extra></extra>"))
                style(fig, 360, title=dict(text="Reduction in the P80 budget if each risk were removed", font=dict(size=15)))
                fig.update_xaxes(tickprefix=cur, tickformat=",.0f", range=[0, max(d[cost_col].max(), 1) * 1.25])
                st.plotly_chart(fig, use_container_width=True)
            with r:
                st.markdown("**How to read it:** expected cost (probability × impact) is what most registers show. "
                            "What matters for the budget is how much each risk moves the P80. If a risk hits a task "
                            "with float, it can cost money without delaying the project.")
            st.dataframe(rank, hide_index=True, use_container_width=True,
                         column_config={"Probability": st.column_config.NumberColumn(format="percent"),
                                        "Expected cost": st.column_config.NumberColumn(format=f"{cur}%,.0f"),
                                        cost_col: st.column_config.NumberColumn(format=f"{cur}%,.0f"),
                                        "Expected delay (days)": st.column_config.NumberColumn(format="%.1f"),
                                        "P80 days saved if removed": st.column_config.NumberColumn(format="%.1f")})

# ---------------------------------------------------------------------------
# Inputs & report
# ---------------------------------------------------------------------------
with tabs[4]:
    cols = [c for c in ["WBS Code", "Task", "Milestone", "Cost", "Optimistic", "Most Likely", "Pessimistic",
                        "Risk Multiplier", "Duration", "Predecessors", "Time-dependent %"] if c in estimates]
    st.markdown("##### Three-point estimates used")
    st.dataframe(estimates[cols], hide_index=True, use_container_width=True)
    if register is not None:
        st.markdown("##### Risk register")
        st.dataframe(register, hide_index=True, use_container_width=True)
    if len(summaries):
        st.caption("Excluded as milestone/sub-total rows: " + ", ".join(summaries["WBS Code"] + " " + summaries["Task"]))
    extra = {}
    if I is not None:
        if I.schedule is not None:
            extra["Schedule"] = schedule_sensitivity(I.schedule)
        extra["Contingency allocation"] = allocate_contingency(I)
        split, rank = extras(key + (dist,), tasks, reg, S, dist)
        extra["Contingency split"] = pd.DataFrame(list(split.items()), columns=["Measure", "Value"])
        if len(rank):
            extra["Risk ranking"] = rank
    buf = io.BytesIO()
    tmp = Path(tempfile.mkdtemp()) / "results.xlsx"
    export_excel(results, estimates, tmp, extra)
    st.download_button("Download the full Excel report", tmp.read_bytes(), "monte_carlo_results.xlsx", type="primary")
