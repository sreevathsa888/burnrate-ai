"""
Add conditional exhaustion probabilities to survival_scored.csv.

WHY THIS EXISTS
---------------
Notebook 03 reports S(6) and S(12): survival measured FROM FOUNDING, because that
is where each venture's duration clock starts. Almost nothing fails that early, so
those columns sit near zero for the entire population -- on the real data they span
roughly 1.3% to 1.7% across 98,280 companies -- and every curve drawn from them is
a flat line at 100%. That slice simply carries no discrimination.

The useful question is conditional: given this venture has already survived to its
current age t, what is the probability it exhausts its cash in the next 3/6/12
months?

    P(fail within d | survived to t)  =  1 - S(t + d) / S(t)

HOW IT IS COMPUTED
------------------
Not by calling predict_survival_function over every distinct time point -- that
builds an (n_times x n_companies) matrix, which for this population is about
30,000 x 98,000 and will exhaust memory. Proportional hazards gives a much cheaper
identity:

    S(t | x) = S0(t) ** exp(beta' x)

so one baseline curve plus one partial hazard per company is enough, and the whole
job is O(n).

Run once from the PROJECT ROOT (not from src/, and not through the VS Code
debugger, which may change the working directory):

    python src/conditional_survival.py
"""
import os
import joblib
import numpy as np
import pandas as pd

PROCESSED = "data/processed"
MODELS = "models"
HORIZONS = (3, 6, 12)

scored_path = os.path.join(PROCESSED, "survival_scored.csv")
model_path = os.path.join(MODELS, "cox_model.pkl")

for p in (scored_path, model_path):
    if not os.path.exists(p):
        raise SystemExit(
            f"missing {p}\n"
            "Run this from the project root -- the folder containing data/ and models/."
        )

feat = pd.read_csv(scored_path)
cph = joblib.load(model_path)
print(f"[load] {len(feat):,} companies, model from {model_path}")

# baseline survival curve S0(t), evaluated on the model's own timeline
base = cph.baseline_survival_
timeline = base.index.values.astype(float)
s0 = base.iloc[:, 0].values.astype(float)
print(f"[baseline] {len(timeline):,} time points, "
      f"t in [{timeline.min():.1f}, {timeline.max():.1f}] months")

# one partial hazard per company: exp(beta' x)
covariates = [c for c in cph.params_.index if c in feat.columns]
ph = cph.predict_partial_hazard(feat[covariates]).values.astype(float)
print(f"[hazard] partial hazards computed over {len(covariates)} covariates")


def s0_at(t):
    """Baseline survival at arbitrary times, clamped to the fitted timeline."""
    return np.interp(np.clip(t, timeline.min(), timeline.max()), timeline, s0)


ages = feat["duration_months"].clip(lower=0.5).values.astype(float)
s_now = np.clip(s0_at(ages) ** ph, 1e-12, 1.0)

for d in HORIZONS:
    s_future = s0_at(ages + d) ** ph
    feat[f"p_exhaust_{d}m_cond"] = np.clip(1.0 - s_future / s_now, 0.0, 1.0)

feat["current_age_months"] = ages
feat.to_csv(scored_path, index=False)

cols = [f"p_exhaust_{d}m_cond" for d in HORIZONS]
print("\n[done] appended conditional columns to survival_scored.csv")
print(feat[cols].describe().round(4).to_string())

hi = feat["p_exhaust_12m_cond"].max()
print(f"\n[check] highest 12-month conditional risk in the population: {hi:.1%}")
if hi < 0.10:
    print("[check] that is low -- if every company still looks flat, the covariates")
    print("[check] may carry little late-life discrimination, which is itself a finding.")
else:
    print("[check] good spread -- the dashboard curve will now be informative.")
    print("[check] Ventures near the top of that range are the ones with visible curves.")