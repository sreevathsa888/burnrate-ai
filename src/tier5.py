"""
Tier 5 -- prescriptive application layer for BurnRate AI.

Two capabilities, both specified in the IDF:
  * intervention ranking  (objective vi)  -- perturb each controllable cost
    parameter, RE-EXECUTE the projection, and measure the actual displacement
    of the exhaustion date.
  * scenario simulation   (objective vii) -- run the same path on a user-supplied
    modified parameter vector and return the differential.

An honest scope note, stated here because it governs how the output should be
read. The Tier 3 survival estimator is fitted on funding history and sector.
None of those is a controllable cost parameter -- a founder cannot change how
many rounds they have already raised. Perturbing a cost line therefore moves
the cash-flow arm of the pipeline and the fused index, but leaves the hazard
estimate unchanged. Displacement reported here is displacement of the projected
exhaustion date, not of the hazard. That is the honest claim and the one the
numbers support.
"""

from dataclasses import dataclass, replace
import numpy as np

# Cost-line split as a share of total monthly burn.
# STATED ASSUMPTION, not measurement: the Crunchbase snapshot carries no
# itemised expense data, so these shares stand in for a real chart of accounts
# and are drawn from commonly reported early-stage cost structures. Replace
# them with founder-supplied figures once the intake form exists; every figure
# downstream inherits this assumption and should be labelled accordingly.
COST_LINES = {
    "Payroll":        0.55,
    "Cloud & infra":  0.22,
    "Marketing":      0.15,
    "Tooling & SaaS": 0.08,
}

SAFETY_CAP_MONTHS = 600   # 50 years; stops a near-zero burn running forever


@dataclass(frozen=True)
class CompanyState:
    """Everything the projection needs for one company."""
    object_id: str
    spend: np.ndarray          # observed/synthesised monthly spend series
    starting_cash: float
    monthly_burn: float        # most recent monthly burn
    trend_window: int = 6


def project_zero_crossing(spend, starting_cash, trend_window=6):
    """Fit a linear trend to the trailing window and step the balance forward
    until it crosses zero. Identical logic to notebook 07 -- the ranker must
    re-execute the real projection, not an approximation of it."""
    spend = np.asarray(spend, dtype=float)
    recent = spend[-trend_window:] if len(spend) >= trend_window else spend
    if len(recent) >= 2:
        slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)
    else:
        slope = 0.0
        intercept = float(recent.mean()) if len(recent) else 0.0

    balance = float(starting_cash)
    months = 0
    projected = intercept + slope * (len(recent) - 1)
    while balance > 0 and months < SAFETY_CAP_MONTHS:
        projected = max(projected + slope, 1.0)
        if balance - projected <= 0:
            # Interpolate within the final month rather than rounding up to a whole
            # one. Integer resolution is too coarse here: a 20% cut to an 8% cost
            # line moves total burn by 1.6%, which whole months cannot register,
            # and the ranker would report a spurious zero for every small line.
            return months + balance / projected
        balance -= projected
        months += 1
    return float(months)


def runway_months(state: CompanyState) -> float:
    return project_zero_crossing(state.spend, state.starting_cash, state.trend_window)


def apply_cut(state: CompanyState, line: str, pct: float) -> CompanyState:
    """Return a new state with `line` reduced by `pct` (0-1). The cut scales the
    whole spend series, because the cost share is assumed constant over it."""
    share = COST_LINES[line]
    factor = 1.0 - share * pct
    return replace(state, spend=state.spend * factor,
                   monthly_burn=state.monthly_burn * factor)


def rank_interventions(state: CompanyState, pct: float = 0.20):
    """Perturb each cost line by `pct` and re-execute the projection.
    Returns rows ordered by measured runway gain."""
    base = runway_months(state)
    rows = []
    for line in COST_LINES:
        moved = runway_months(apply_cut(state, line, pct))
        gain_months = moved - base
        rows.append({
            "intervention": f"Reduce {line.lower()} by {int(pct*100)}%",
            "cost_line": line,
            "baseline_months": round(base, 2),
            "projected_months": round(moved, 2),
            "gain_months": round(gain_months, 3),
            "gain_weeks": round(gain_months * 4.345, 1),
        })
    return sorted(rows, key=lambda r: -r["gain_months"])


def analytic_gain_months(state: CompanyState, line: str, pct: float = 0.20) -> float:
    """First-order (linearised) estimate of the same quantity, using the flat-burn
    identity runway = cash / burn. This is the approximation the IDF argues against;
    Experiment 6 measures how far it diverges from the re-executed figure."""
    share = COST_LINES[line]
    burn = float(state.monthly_burn)
    if burn <= 0:
        return 0.0
    new_burn = burn * (1.0 - share * pct)
    if new_burn <= 0:
        return float("inf")
    return state.starting_cash / new_burn - state.starting_cash / burn


def simulate_scenario(state: CompanyState, burn_multiplier=1.0, new_raise=0.0,
                      revenue_offset=0.0):
    """Objective (vii). Re-executes the identical projection path on a modified
    parameter vector so the simulated figure is directly comparable with the live
    one rather than an approximation of it."""
    modified = replace(
        state,
        spend=np.maximum(state.spend * burn_multiplier - revenue_offset, 1.0),
        starting_cash=state.starting_cash + new_raise,
        monthly_burn=max(state.monthly_burn * burn_multiplier - revenue_offset, 1.0),
    )
    base = runway_months(state)
    moved = runway_months(modified)
    return {
        "baseline_months": round(base, 2),
        "scenario_months": round(moved, 2),
        "delta_months": round(moved - base, 2),
        "state": modified,
    }


def state_from_trajectory(object_id, spend_series, cash_multiple=3.0):
    """Build a CompanyState from a synthesised spend series.

    `cash_multiple` is the placeholder for cash on hand -- the snapshot carries
    no balance-sheet figure, so starting cash is taken as a multiple of the most
    recent monthly spend. Same placeholder as notebook 07; replace with the
    founder-supplied balance when the intake form exists."""
    spend = np.asarray(spend_series, dtype=float)
    if len(spend) == 0:
        raise ValueError(f"no spend series for {object_id}")
    last = float(spend[-1])
    return CompanyState(object_id=object_id, spend=spend,
                        starting_cash=last * cash_multiple, monthly_burn=last)
