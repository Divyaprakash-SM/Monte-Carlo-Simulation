"""Schedule, risk-register and joint-confidence tests.  Run with:  python -m pytest -q"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from montecarlo import (Settings, allocate_contingency, budget_for_jcl, contingency_split, jcl, load_risk_register,
                        load_wbs, risk_ranking, run_integrated)
from montecarlo.schedule import cpm, resolve_predecessors, topological_order

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_wbs.xlsx"


@pytest.fixture(scope="module")
def data():
    t, _ = load_wbs(SAMPLE)
    return t, load_risk_register(SAMPLE)


def toy(preds):
    return pd.DataFrame({"WBS Code": list(preds), "Predecessors": list(preds.values())})


# --- Critical path, checked by hand -------------------------------------------
def test_cpm_on_a_known_network():
    # A(3) -> B(4) -> D(2);  A -> C(1) -> D.  Critical path A-B-D = 9 days; C has 3 days float.
    t = toy({"A": "", "B": "A", "C": "A", "D": "B,C"})
    preds = resolve_predecessors(t)
    order = topological_order(preds)
    ef, tf, project = cpm(np.array([[3.0, 4.0, 1.0, 2.0]]), preds, order)
    assert project[0] == 9
    assert ef[0].tolist() == [3, 7, 4, 9]
    assert tf[0].tolist() == [0, 0, 3, 0]


def test_milestone_code_as_predecessor_expands_to_its_tasks():
    t = toy({"1.1": "", "1.2": "", "2.1": "1"})
    assert resolve_predecessors(t)[2] == [0, 1]


def test_loops_and_unknown_predecessors_are_rejected():
    with pytest.raises(ValueError, match="loop"):
        topological_order(resolve_predecessors(toy({"A": "B", "B": "A"})))
    with pytest.raises(ValueError, match="not in the WBS"):
        resolve_predecessors(toy({"A": "Z"}))


# --- Integrated model ----------------------------------------------------------
def test_no_uncertainty_and_no_risks_reproduces_the_plan(data):
    tasks, _ = data
    s = Settings(n_sims=200, optimistic_factor=1, pessimistic_factor=1, duration_optimistic_factor=1,
                 duration_pessimistic_factor=1, risk_mode="none")
    flat = tasks.assign(**{"Optimistic": np.nan, "Pessimistic": np.nan, "Optimistic Duration": np.nan,
                           "Pessimistic Duration": np.nan})
    res = run_integrated(flat, None, "pert", s, use_register=False)
    assert np.allclose(res.total_cost, tasks["Cost"].sum())
    assert np.allclose(res.project_days, res.schedule.deterministic_days)


def test_merge_bias_pushes_the_finish_later(data):
    tasks, _ = data
    res = run_integrated(tasks, None, "pert", Settings(risk_mode="none"), use_register=False)
    assert np.median(res.project_days) > res.schedule.deterministic_days


def test_risk_register_adds_cost_and_time(data):
    tasks, reg = data
    s = Settings()
    with_r = run_integrated(tasks, reg, "pert", s)
    without = run_integrated(tasks, None, "pert", Settings(risk_mode="none"), use_register=False)
    assert with_r.total_cost.mean() > without.total_cost.mean()
    assert with_r.project_days.mean() > without.project_days.mean()
    # Each risk's average cost is close to probability x mean impact
    for j, r in reg.iterrows():
        expected = r["Probability"] * (r["Cost Min"] + 4 * r["Cost ML"] + r["Cost Max"]) / 6
        assert with_r.risk_costs[:, j].mean() == pytest.approx(expected, rel=0.08)


def test_results_are_reproducible(data):
    tasks, reg = data
    a = run_integrated(tasks, reg, "pert", Settings(seed=3))
    b = run_integrated(tasks, reg, "pert", Settings(seed=3))
    np.testing.assert_array_equal(a.total_cost, b.total_cost)
    np.testing.assert_array_equal(a.project_days, b.project_days)


def test_joint_confidence_is_below_each_single_confidence(data):
    tasks, reg = data
    res = run_integrated(tasks, reg, "pert", Settings())
    c80, d80 = np.quantile(res.total_cost, 0.8), np.quantile(res.project_days, 0.8)
    j = jcl(res, c80, d80)
    assert 0.6 <= j < 0.8
    need = budget_for_jcl(res, d80, 0.7)
    assert jcl(res, need, d80) >= 0.7 - 1e-9
    assert budget_for_jcl(res, np.quantile(res.project_days, 0.5), 0.7) is None   # impossible: date too tight


def test_contingency_split_adds_up(data):
    tasks, reg = data
    s = contingency_split(tasks, reg, "pert", Settings())
    assert s["base_cost"] + s["estimate_uncertainty"] + s["risk_events"] == pytest.approx(s["total_budget"])
    assert s["risk_events"] > 0 and s["estimate_uncertainty"] > 0


def test_contingency_allocation_sums_to_total(data):
    tasks, reg = data
    res = run_integrated(tasks, reg, "pert", Settings())
    alloc = allocate_contingency(res, 0.8)
    assert alloc["Contingency"].sum() == pytest.approx(np.quantile(res.total_cost, 0.8) - res.base_cost)


def test_risk_on_a_task_with_float_does_not_delay_the_project(data):
    tasks, reg = data
    rank = risk_ranking(tasks, reg, "pert", Settings(n_sims=4000)).set_index("Risk ID")
    assert rank.loc["R2", "P80 days saved if removed"] == pytest.approx(0, abs=0.5)   # 2.2 has float
    assert rank.loc["R1", "P80 days saved if removed"] > 2                          # 2.3 is critical
