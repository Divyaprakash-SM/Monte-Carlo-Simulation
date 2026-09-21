"""Run with:  python -m pytest -q"""

from pathlib import Path

import numpy as np
import pytest

from montecarlo import Settings, load_wbs, project_summary, run, sensitivity, three_point_estimates

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_wbs.xlsx"


@pytest.fixture(scope="module")
def tasks():
    t, _ = load_wbs(SAMPLE)
    return t


def test_milestone_rows_are_not_double_counted():
    tasks, summaries = load_wbs(SAMPLE)
    assert len(summaries) == 4                      # the four milestone rows
    assert set(summaries["WBS Code"]) == {"1", "2", "3", "4"}
    assert tasks["Cost"].sum() == pytest.approx(summaries["Cost"].sum())


def test_parent_tasks_with_their_own_cost_are_kept(tasks):
    # 1.2 has a child (1.2.1) but its cost is its own, so both are tasks.
    assert {"1.2", "1.2.1", "3.3", "3.3.1"} <= set(tasks["WBS Code"])


def test_triangular_and_pert_means_match_theory(tasks):
    s = Settings(n_sims=200_000, risk_mode="none", seed=1)
    est, res = run(tasks, ("triangular", "pert"), s)
    a, m, b = (est[c].sum() for c in ("Optimistic", "Most Likely", "Pessimistic"))
    assert res["triangular"].total.mean() == pytest.approx((a + m + b) / 3, rel=2e-3)
    assert res["pert"].total.mean() == pytest.approx((a + 4 * m + b) / 6, rel=2e-3)


def test_samples_stay_inside_their_bounds(tasks):
    est, res = run(tasks, ("triangular", "pert"), Settings(n_sims=5_000, risk_mode="none"))
    for r in res.values():
        assert (r.task_samples >= est["Optimistic"].to_numpy() - 1e-9).all()
        assert (r.task_samples <= est["Pessimistic"].to_numpy() + 1e-9).all()


def test_same_seed_gives_same_answer(tasks):
    _, r1 = run(tasks, ("pert",), Settings(seed=7))
    _, r2 = run(tasks, ("pert",), Settings(seed=7))
    np.testing.assert_array_equal(r1["pert"].total, r2["pert"].total)


def test_correlation_widens_the_spread(tasks):
    _, indep = run(tasks, ("pert",), Settings(correlation=0.0))
    _, corr = run(tasks, ("pert",), Settings(correlation=0.5))
    assert corr["pert"].total.std() > 1.5 * indep["pert"].total.std()


def test_risk_loading_raises_the_estimate(tasks):
    est_loaded = three_point_estimates(tasks, Settings(risk_mode="loaded"))
    expected = (tasks["Cost"] * (1 + tasks["Risk Multiplier"])).sum()
    assert est_loaded["Most Likely"].sum() == pytest.approx(expected)


def test_event_mode_sits_between_none_and_loaded(tasks):
    means = {m: run(tasks, ("pert",), Settings(risk_mode=m))[1]["pert"].total.mean()
             for m in ("none", "event", "loaded")}
    assert means["none"] < means["event"] < means["loaded"]


def test_variance_shares_add_up_to_100(tasks):
    _, res = run(tasks, ("lognormal",), Settings())
    assert sensitivity(res["lognormal"])["Share of variance (%)"].sum() == pytest.approx(100, abs=1e-6)


def test_summary_percentiles_are_ordered(tasks):
    est, res = run(tasks)
    s = project_summary(res, est)
    assert (s["P10"] < s["P50"]).all() and (s["P50"] < s["P80"]).all() and (s["P80"] < s["P90"]).all()


def test_bad_three_point_estimate_is_reported(tasks):
    broken = tasks.copy()
    broken.loc[0, "Optimistic"] = broken.loc[0, "Cost"] * 2
    with pytest.raises(ValueError, match="Optimistic <= Most Likely"):
        three_point_estimates(broken, Settings(risk_mode="none"))


def test_csv_input_works(tmp_path, tasks):
    csv = tmp_path / "wbs.csv"
    tasks[["WBS Code", "Task", "Cost"]].to_csv(csv, index=False)
    t2, _ = load_wbs(csv)
    assert len(t2) == len(tasks)
    assert t2["Risk Multiplier"].eq(0).all()        # missing column defaults to no risk
