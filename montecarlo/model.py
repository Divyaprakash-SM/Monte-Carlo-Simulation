"""Monte Carlo engine: three-point estimates, three distributions, optional correlation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

DISTRIBUTIONS = ("triangular", "lognormal", "pert")
RISK_MODES = ("loaded", "event", "none")
DISPLAY_NAMES = {"triangular": "Triangular", "lognormal": "Lognormal", "pert": "Beta-PERT"}


@dataclass
class Settings:
    """Every assumption the model makes, in one place."""

    n_sims: int = 10_000
    optimistic_factor: float = 0.90   # used when the sheet has no Optimistic value
    pessimistic_factor: float = 1.20  # used when the sheet has no Pessimistic value
    risk_mode: str = "loaded"         # "loaded", "event" or "none" (see README)
    likelihood_scale: float = 6.0     # top of the likelihood scale, for "event" mode
    correlation: float = 0.0          # 0 = tasks independent; 0.3 = moderate shared drift
    pert_lambda: float = 4.0          # standard Beta-PERT shape
    seed: int | None = 42


@dataclass
class Result:
    distribution: str
    labels: list[str]
    task_samples: np.ndarray          # shape (n_sims, n_tasks)
    total: np.ndarray                 # shape (n_sims,)
    settings: Settings = field(repr=False)

    @property
    def name(self) -> str:
        return DISPLAY_NAMES[self.distribution]

    def task(self, label: str) -> np.ndarray:
        return self.task_samples[:, self.labels.index(label)]


def three_point_estimates(tasks: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Add Optimistic / Most Likely / Pessimistic columns (risk-adjusted if enabled)."""
    out = tasks.copy()
    if s.risk_mode not in RISK_MODES:
        raise ValueError(f"risk_mode must be one of {RISK_MODES}")
    risk = 1.0 + (out["Risk Multiplier"] if s.risk_mode == "loaded" else 0.0)
    out["Most Likely"] = out["Cost"] * risk
    opt = out["Optimistic"].where(out["Optimistic"].notna(), out["Cost"] * s.optimistic_factor)
    pes = out["Pessimistic"].where(out["Pessimistic"].notna(), out["Cost"] * s.pessimistic_factor)
    out["Optimistic"] = opt * risk
    out["Pessimistic"] = pes * risk
    bad = out[(out["Optimistic"] > out["Most Likely"]) | (out["Most Likely"] > out["Pessimistic"])]
    if not bad.empty:
        raise ValueError(
            "Each task needs Optimistic <= Most Likely <= Pessimistic. Check: "
            + ", ".join(bad["Label"].tolist())
        )
    return out


# ---------------------------------------------------------------------------
# Inverse CDFs: turn a uniform number u in (0, 1) into a cost.
# Using inverse CDFs for every distribution is what lets the same uniform
# numbers be correlated across tasks.
# ---------------------------------------------------------------------------

def _triangular_ppf(u, a, m, b):
    width = np.maximum(b - a, 1e-12)
    fc = (m - a) / width
    left = a + np.sqrt(u * width * (m - a))
    right = b - np.sqrt((1 - u) * width * (b - m))
    return np.where(b - a <= 0, m, np.where(u < fc, left, right))


def _pert_ppf(u, a, m, b, lam):
    width = np.maximum(b - a, 1e-12)
    alpha = 1 + lam * (m - a) / width
    beta_ = 1 + lam * (b - m) / width
    return np.where(b - a <= 0, m, a + width * beta_dist.ppf(u, alpha, beta_))


def _lognormal_ppf(u, a, m, b):
    # Median set at the most-likely cost; the optimistic-pessimistic range is
    # treated as +/- 2 standard deviations in log space (about a 95% band).
    mu = np.log(m)
    sigma = np.maximum((np.log(b) - np.log(a)) / 4, 1e-12)
    return np.exp(mu + sigma * norm.ppf(u))


def _uniforms(n, k, rho, rng):
    if rho <= 0:
        return rng.random((n, k))
    common = rng.standard_normal((n, 1))
    own = rng.standard_normal((n, k))
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * own   # equicorrelated normals
    return norm.cdf(z)


def simulate(estimates: pd.DataFrame, distribution: str, s: Settings) -> Result:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")
    if not 0 <= s.correlation < 1:
        raise ValueError("correlation must be between 0 and 1")
    rng = np.random.default_rng(s.seed)
    a = estimates["Optimistic"].to_numpy(float)
    m = estimates["Most Likely"].to_numpy(float)
    b = estimates["Pessimistic"].to_numpy(float)
    u = np.clip(_uniforms(s.n_sims, len(estimates), s.correlation, rng), 1e-12, 1 - 1e-12)

    if distribution == "triangular":
        samples = _triangular_ppf(u, a, m, b)
    elif distribution == "pert":
        samples = _pert_ppf(u, a, m, b, s.pert_lambda)
    else:
        samples = _lognormal_ppf(u, a, m, b)

    if s.risk_mode == "event":
        # Each risk either happens or it does not. Chance = likelihood / scale;
        # if it happens, that task's cost rises by its risk multiplier.
        if estimates["Risk Likelihood"].isna().all():
            raise ValueError("'event' risk mode needs a Risk Likelihood column")
        chance = (estimates["Risk Likelihood"].fillna(0) / s.likelihood_scale).clip(0, 1).to_numpy()
        happens = rng.random(samples.shape) < chance
        samples = samples * (1 + happens * estimates["Risk Multiplier"].to_numpy(float))

    return Result(distribution, estimates["Label"].tolist(), samples, samples.sum(axis=1), s)


def run(tasks: pd.DataFrame, distributions=DISTRIBUTIONS, settings: Settings | None = None):
    """Convenience wrapper: estimates + one Result per distribution."""
    s = settings or Settings()
    est = three_point_estimates(tasks, s)
    return est, {d: simulate(est, d, s) for d in distributions}
