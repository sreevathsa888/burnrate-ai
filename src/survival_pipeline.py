"""
BurnRate AI — Experiment 1 pipeline: clean funding data, resolve outcomes
under censoring, fit a survival model, and benchmark against a naive
fixed-horizon baseline. This is the one defensible, real-data result to
walk into review with.

Run:
    pip install pandas numpy lifelines scikit-learn matplotlib --break-system-packages
    python survival_pipeline.py

Expects these files in the same folder (Crunchbase/Kaggle 2013 snapshot):
    funding_rounds.csv
    objects_slim.csv   <- see note below if you only have the big objects.csv
    acquisitions.csv
    ipos.csv

If you still only have objects.csv (the ~462k-row one), slim it first:

    import pandas as pd
    o = pd.read_csv("objects.csv", encoding="ISO-8859-1", low_memory=False)
    cols = ["id","entity_type","name","category_code","status","founded_at","closed_at",
            "country_code","first_funding_at","last_funding_at","funding_rounds","funding_total_usd"]
    o[o.entity_type == "Company"][cols].to_csv("objects_slim.csv", index=False)

Column names assume the standard 2013 Crunchbase Kaggle dump. If your
CSVs use different headers, the error will tell you which column is
missing — fix the name at the top of the relevant function, not the logic.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAPSHOT_DATE = pd.Timestamp("2013-12-12")  # dataset snapshot date — cite this in your paper/IDF
STALE_MONTHS = 36  # presumed-dead threshold. Rerun at 24 and 48 too and report how much the
                    # censoring % and C-index shift — a panel/examiner will ask for this sensitivity check.


# ---------- Stage 1: load & clean ----------
def load_and_clean():
    fr = pd.read_csv("funding_rounds.csv", encoding="ISO-8859-1", low_memory=False)
    obj = pd.read_csv("objects_slim.csv", encoding="ISO-8859-1", low_memory=False)
    acq = pd.read_csv("acquisitions.csv", encoding="ISO-8859-1", low_memory=False)
    ipo = pd.read_csv("ipos.csv", encoding="ISO-8859-1", low_memory=False)

    fr["funded_at"] = pd.to_datetime(fr["funded_at"], errors="coerce")
    n0 = len(fr)
    fr = fr.dropna(subset=["funded_at"])
    fr = fr[fr["raised_amount_usd"].fillna(0) > 0]  # drop undisclosed rows (=0/NaN) — NOT zero capital
    print(f"[clean] funding_rounds: {n0} -> {len(fr)} after dropping undisclosed/undated rounds")

    for col in ["founded_at", "closed_at", "last_funding_at", "first_funding_at"]:
        obj[col] = pd.to_datetime(obj[col], errors="coerce")
    acq["acquired_at"] = pd.to_datetime(acq["acquired_at"], errors="coerce")
    ipo["public_at"] = pd.to_datetime(ipo["public_at"], errors="coerce")

    return fr, obj, acq, ipo


# ---------- Stage 2: outcome resolution (this is the part prior art skips) ----------
def resolve_outcomes(obj, acq, ipo):
    df = obj.copy()

    # LEFT joins only — an inner join here silently intersects the cohort down to
    # "companies that were both acquired AND went public", which is a real bug that
    # produced a 21-company cohort the first time this pipeline was written.
    df = df.merge(
        acq[["acquired_object_id", "acquired_at"]].rename(columns={"acquired_object_id": "id"}),
        on="id", how="left",
    )
    df = df.merge(
        ipo[["object_id", "public_at"]].rename(columns={"object_id": "id"}),
        on="id", how="left",
    )

    df["event"] = 0
    df["event_date"] = pd.NaT

    # 1. closed -> event observed
    closed = df["status"] == "closed"
    df.loc[closed, "event"] = 1
    df.loc[closed, "event_date"] = df.loc[closed, "closed_at"]

    # 2. acquired / ipo -> censored at exit (an acquired company can no longer "fail")
    exited = (~closed) & (df["acquired_at"].notna() | df["public_at"].notna())
    exit_date = df["acquired_at"].combine_first(df["public_at"])
    df.loc[exited, "event"] = 0
    df.loc[exited, "event_date"] = exit_date[exited]

    # 3. operating but stale -> presumed dead (Crunchbase under-reports shutdowns)
    still_open = ~closed & ~exited
    last_seen = df["last_funding_at"].combine_first(df["first_funding_at"])
    months_stale = (SNAPSHOT_DATE - last_seen).dt.days / 30.44
    stale = still_open & (months_stale >= STALE_MONTHS)
    df.loc[stale, "event"] = 1
    df.loc[stale, "event_date"] = last_seen[stale] + pd.DateOffset(months=STALE_MONTHS)

    # 4. everything else -> genuinely right-censored at the snapshot date
    remaining = still_open & ~stale
    df.loc[remaining, "event"] = 0
    df.loc[remaining, "event_date"] = SNAPSHOT_DATE

    start = df["founded_at"].combine_first(df["first_funding_at"])
    df["duration_months"] = (df["event_date"] - start).dt.days / 30.44
    n_before = len(df)
    df = df[df["duration_months"] > 0]  # drop bad rows: missing/inverted dates
    print(f"[outcomes] dropped {n_before - len(df)} rows with missing/inverted dates")

    censor_pct = 100 * (1 - df["event"].mean())
    print(f"[outcomes] N={len(df)}  events={int(df['event'].sum())}  censored={censor_pct:.1f}%")
    print(f"[outcomes] stale-presumed-dead at STALE_MONTHS={STALE_MONTHS}: {int(stale.sum())} companies")
    print("[outcomes] -> the censored %% above is your headline number: this is the fraction of "
          "data a fixed-horizon classifier would have deleted, and that this pipeline retains.")
    return df


# ---------- Stage 3: features ----------
def build_features(df):
    feat = df[["id", "duration_months", "event", "funding_total_usd", "funding_rounds",
               "category_code", "country_code"]].copy()
    feat["funding_total_usd"] = feat["funding_total_usd"].fillna(0)
    feat["funding_rounds"] = feat["funding_rounds"].fillna(0)
    feat["log_funding"] = np.log1p(feat["funding_total_usd"])

    top_sectors = feat["category_code"].value_counts().head(8).index
    feat["sector"] = feat["category_code"].where(feat["category_code"].isin(top_sectors), "other")
    feat = pd.get_dummies(feat, columns=["sector"], drop_first=True)
    feat = feat.drop(columns=["category_code", "country_code", "funding_total_usd"])
    feat = feat.dropna()
    return feat


# ---------- Stage 4a: survival model ----------
def fit_survival(feat):
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(feat.drop(columns=["id"]), duration_col="duration_months", event_col="event")
    cph.print_summary()

    c_index = concordance_index(feat["duration_months"], -cph.predict_partial_hazard(feat), feat["event"])
    print(f"\n[survival] concordance index = {c_index:.3f}   <- headline number, cite this in Section 8 / your eval slide")

    surv_func = cph.predict_survival_function(feat, times=[6, 12])
    feat["p_exhaust_6m"] = 1 - surv_func.loc[6].values
    feat["p_exhaust_12m"] = 1 - surv_func.loc[12].values
    return cph, feat, c_index


# ---------- Stage 4b: naive fixed-horizon baseline, for the comparison your evaluation plan promised ----------
def fit_baseline(feat):
    labeled = feat[(feat["event"] == 1) | (feat["duration_months"] >= 12)].copy()
    labeled["failed_12m"] = ((labeled["event"] == 1) & (labeled["duration_months"] <= 12)).astype(int)

    drop_cols = ["id", "duration_months", "event", "p_exhaust_6m", "p_exhaust_12m", "failed_12m"]
    X = labeled.drop(columns=[c for c in drop_cols if c in labeled.columns])
    y = labeled["failed_12m"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    discarded = len(feat) - len(labeled)
    print(f"[baseline] fixed-horizon logistic regression AUC @ 12mo = {auc:.3f}")
    print(f"[baseline] this approach discards {discarded} still-operating, <12mo-observed companies "
          f"({100 * discarded / len(feat):.1f}%) that the survival model above retains as censored — "
          f"this is Experiment 2 from your IDF in one line.")
    return auc


# ---------- Stage 5: one chart worth having on screen at review ----------
def plot_km_by_sector(feat):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    sector_cols = [c for c in feat.columns if c.startswith("sector_")]
    for col in sector_cols:
        g = feat[feat[col] == 1]
        if len(g) < 15:
            continue
        kmf.fit(g["duration_months"], g["event"], label=col.replace("sector_", ""))
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Months since founding")
    ax.set_ylabel("Probability still operating")
    ax.set_title("Kaplan-Meier survival by sector")
    plt.tight_layout()
    plt.savefig("km_by_sector.png", dpi=150)
    print("[plot] saved km_by_sector.png")


if __name__ == "__main__":
    fr, obj, acq, ipo = load_and_clean()
    outcomes = resolve_outcomes(obj, acq, ipo)
    feat = build_features(outcomes)
    cph, feat, c_index = fit_survival(feat)
    auc = fit_baseline(feat)
    plot_km_by_sector(feat)
    feat.to_csv("survival_scored.csv", index=False)
    print("\n[done] wrote survival_scored.csv + km_by_sector.png")
    print("[done] your Experiment 1 table: Cox C-index vs logistic-baseline AUC vs naive cash/burn (add that one by hand).")
