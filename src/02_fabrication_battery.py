"""
02_fabrication_battery.py   (EXPERIMENT 2)
==========================================
Why this experiment exists
--------------------------
Before we can claim anything about personalization we have to know which of the
downloaded datasets are real. Our previously published screen tests one thing:
whether a modelling target is an exact function of a single other column. During
acquisition we found six fabricated datasets, and that screen returned a benign
verdict on all but one of them.

So the screen is not wrong, it is incomplete. A dataset can be fake in at least
four structurally different ways, and each needs its own detector:

  D1  LOOKUP TABLE            the label is an exact function of one column
                              -> exact functional dependency + row compression
  D2  INDEPENDENT NOISE       columns drawn independently, nothing predicts
                              anything. Invisible to D1 because there are no
                              dependencies at all.
                              -> predictive signal under a person-grouped split
  D3  TEMPORAL FABRICATION    rows independent across time, so no person
                              persists. Invisible to any cross-sectional test.
                              -> per-person lag-1 autocorrelation of state
                                 variables, and whether a person's own past
                                 predicts their future
  D4  INTERNAL INCONSISTENCY  derived quantities contradict their inputs
                              -> recompute the derived column and compare

The experiment applies all four to every dataset, real and fabricated, and
reports how many each detector catches. The headline is the marginal value of
D2-D4 over D1 alone.

Outputs
-------
outputs/texts/e2_battery_results.csv     one row per dataset x detector
outputs/texts/e2_battery_summary.txt     the readable verdict table
outputs/texts/e2_detector_coverage.json  counts for the paper
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/macbook/Documents/MyProjects/Greg_Research/multidataset")
from audit_harness import audit_frame, verdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42
MAX_ROWS = 120_000


# ------------------------------------------------------------------ registry
# ground_truth is our prior belief from the acquisition reports; the detectors
# are what actually decide. Keeping the label here lets us score the battery.
REGISTRY = [
    # name, path, person_col, date_col, target, ground_truth
    ("fitness_wellness_plan", "fitness_wellness_plan/GYM.csv",
     None, None, "Exercise Schedule", "FABRICATED"),
    ("diet_rec_medical", "diet_rec_medical/Personalized_Diet_Recommendations.csv",
     "Patient_ID", None, None, "FABRICATED"),
    ("fitness365", "fitness365/health_fitness_tracking_365days.csv",
     "user_id", "date", None, "FABRICATED"),
    ("fitlife", "fitlife/health_fitness_dataset.csv",
     "participant_id", "date", None, "FABRICATED"),
    ("whoop100k", "whoop100k/whoop_fitness_dataset_100k.csv",
     "user_id", "date", None, "FABRICATED"),
    ("lifesnaps", "lifesnaps/rais_anonymized/csv_rais_anonymized/"
                  "daily_fitbit_sema_df_unprocessed.csv",
     "id", "date", None, "REAL"),
    ("fitbit_mobius", None, "Id", "ActivityDate", None, "REAL"),
    ("smoking_body", "smoking_body/smoking.csv", None, None, "smoking", "REAL"),
    ("cardio_train", "cardio_train/cardio_train.csv", None, None, "cardio", "REAL"),
]

# State variables to test for temporal persistence. Physiology first: these
# genuinely cannot jump independently from day to day in a real person.
# D3 applies ONLY to physiological state variables. Behaviour (steps, active
# minutes) is genuinely noisy from day to day even in real people, so testing it
# produces false alarms. Static attributes (weight held constant per person) must
# also be excluded: a constant series has autocorrelation 1.0 and would mask the
# very signal we are looking for. Both exclusions were added after the first run
# produced a false alarm on fitbit_mobius and a masked result on whoop100k.
STATE_CANDIDATES = [
    "resting_hr", "bpm", "heart_rate_avg", "avg_heart_rate", "recovery_score",
    "rmssd", "hrv", "spo2", "sleep_efficiency", "day_strain",
    "nightly_temperature", "stress_score", "full_sleep_breathing_rate",
]
MIN_DISTINCT_WITHIN_PERSON = 10


def load(name, rel):
    """Load a dataset, with the special cases the archives require."""
    if name == "fitbit_mobius":
        import glob
        hits = glob.glob(os.path.join(DATA, "fitbit_mobius", "**",
                                      "dailyActivity_merged.csv"), recursive=True)
        return pd.read_csv(hits[0])
    path = os.path.join(DATA, rel)
    # cardio_train ships semicolon-separated
    if name == "cardio_train":
        return pd.read_csv(path, sep=";")
    df = pd.read_csv(path, low_memory=False)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return df


# ------------------------------------------------------------------ D1
def d1_lookup_table(df, target):
    """Existing screen: exact single-column determination and row compression."""
    rep = audit_frame(df, name="x", source="local")
    v = verdict(rep)
    return {
        "d1_verdict": v,
        "d1_n_exact_deps": rep.get("n_exact_dependencies"),
        "d1_compression": rep.get("compression"),
        "d1_flag": v in ("RULE TABLE", "TARGET LEAK"),
    }


# ------------------------------------------------------------------ D2
def _auc_for_target(d, target, person_col):
    """Grouped-CV AUC for one candidate target. Returns nan if not evaluable."""
    dd = d[d[target].notna()]
    y_raw = dd[target]
    if y_raw.nunique() < 2 or len(dd) < 300:
        return np.nan
    if y_raw.nunique() > 2:
        top = y_raw.value_counts().index[0]
        y = (y_raw == top).astype(int).values
    else:
        y = pd.factorize(y_raw)[0]
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 30:
        return np.nan

    X = dd[[c for c in dd.columns if c not in {target, person_col}]]
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, X.notna().mean() > 0.5]
    if X.shape[1] == 0:
        return np.nan
    X = X.fillna(X.median(numeric_only=True))

    n = min(len(X), 25_000)
    idx = np.random.RandomState(SEED).choice(len(X), n, replace=False)
    X, y = X.iloc[idx], y[idx]
    groups = dd[person_col].values[idx] if person_col in dd.columns else None

    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=100)
    try:
        if groups is not None and pd.Series(groups).nunique() >= 5:
            cv = GroupKFold(n_splits=5)
            p = cross_val_predict(clf, X, y, cv=cv.split(X, y, groups),
                                  method="predict_proba")[:, 1]
        else:
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def d2_signal(df, person_col, target):
    """
    Is there ANY predictable structure in this dataset?

    Independent-noise fabrication produces chance-level prediction for every
    target, because the columns were drawn independently. A real dataset almost
    always contains some learnable relationship. The first version of this
    detector auto-selected a single target and produced a false alarm on
    LifeSnaps, where the chosen column happened to be genuinely unpredictable.
    So we now try up to EIGHT plausible targets and keep the BEST result: the
    claim "nothing predicts anything here" is only credible if no candidate
    works.

    Splits are grouped by person, so a model cannot score by recognising
    individuals it has already seen. That distinction is not cosmetic: on
    fitlife the same column yields AUC 0.9999 ungrouped and 0.4885 grouped.
    """
    d = df.copy()
    if target is not None:
        cands = [target]
    else:
        cands = [c for c in d.columns
                 if 2 <= d[c].nunique(dropna=True) <= 6
                 and c != person_col and d[c].notna().mean() > 0.7]
        def balance(c):
            p = d[c].value_counts(normalize=True).values
            return -np.sum((p - 1 / len(p)) ** 2)
        cands = sorted(cands, key=balance, reverse=True)[:8]

    if not cands:
        return {"d2_auc": np.nan, "d2_target": None, "d2_flag": False,
                "d2_n_targets": 0}

    scored = [(c, _auc_for_target(d, c, person_col)) for c in cands]
    scored = [(c, a) for c, a in scored if pd.notna(a)]
    if not scored:
        return {"d2_auc": np.nan, "d2_target": None, "d2_flag": False,
                "d2_n_targets": 0}

    best_c, best_a = max(scored, key=lambda x: x[1])
    return {"d2_auc": round(best_a, 4), "d2_target": best_c,
            "d2_flag": bool(best_a < 0.55), "d2_n_targets": len(scored)}


# ------------------------------------------------------------------ D3
def d3_temporal(df, person_col, date_col):
    """
    Do individuals persist through time?

    For each candidate state variable we compute the within-person lag-1
    autocorrelation, then take the median across people and across variables.
    Real physiology is strongly autocorrelated. Rows sampled independently are
    not, regardless of how plausible each row looks on its own.
    """
    if person_col is None or date_col is None:
        return {"d3_lag1_ac": np.nan, "d3_var": None, "d3_flag": False,
                "d3_applicable": False}
    if person_col not in df.columns or date_col not in df.columns:
        return {"d3_lag1_ac": np.nan, "d3_var": None, "d3_flag": False,
                "d3_applicable": False}

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", format="mixed")
    d = d.dropna(subset=[date_col]).sort_values([person_col, date_col])

    best = (None, np.nan)
    for v in STATE_CANDIDATES:
        if v not in d.columns or not pd.api.types.is_numeric_dtype(d[v]):
            continue
        acs = []
        for _, g in d.groupby(person_col):
            s = g[v].astype(float)
            s = s[s.notna()]
            # require real within-person variation, else this is a static
            # attribute repeated down the rows, not a state that evolves
            if len(s) < 15 or s.std() == 0 or s.nunique() < MIN_DISTINCT_WITHIN_PERSON:
                continue
            acs.append(s.autocorr(lag=1))
        acs = [a for a in acs if pd.notna(a)]
        if len(acs) >= 5:
            m = float(np.median(acs))
            if pd.isna(best[1]) or m > best[1]:
                best = (v, m)

    if best[0] is None:
        return {"d3_lag1_ac": np.nan, "d3_var": None, "d3_flag": False,
                "d3_applicable": False}
    # Even the MOST persistent state variable being near zero is decisive.
    return {"d3_lag1_ac": round(best[1], 4), "d3_var": best[0],
            "d3_flag": bool(best[1] < 0.30), "d3_applicable": True}


# ------------------------------------------------------------------ D4
def d4_consistency(df):
    """
    Recompute BMI from weight and height and compare with the stored column.
    A real dataset agrees to rounding. A fabricated one drew them independently.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    bmi = next((cols[k] for k in cols if k == "bmi"), None)
    wt = next((cols[k] for k in cols if k in
               ("weight_kg", "weight(kg)", "weight", "weightkg")), None)
    ht = next((cols[k] for k in cols if k in
               ("height_cm", "height(cm)", "height", "heightcm")), None)
    if not (bmi and wt and ht):
        return {"d4_disagree_pct": np.nan, "d4_flag": False, "d4_applicable": False}

    d = df[[bmi, wt, ht]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d[ht] > 100) & (d[ht] < 230) & (d[wt] > 25) & (d[wt] < 300)]
    if len(d) < 50:
        return {"d4_disagree_pct": np.nan, "d4_flag": False, "d4_applicable": False}
    recomputed = d[wt] / (d[ht] / 100.0) ** 2
    # generous tolerance: 1 BMI unit absorbs any rounding convention
    disagree = float((np.abs(recomputed - d[bmi]) > 1.0).mean() * 100)
    return {"d4_disagree_pct": round(disagree, 2),
            "d4_flag": bool(disagree > 20.0), "d4_applicable": True}


# ------------------------------------------------------------------ main
def main():
    rows = []
    for name, rel, person, date, target, truth in REGISTRY:
        print(f"\n{'=' * 70}\n{name}  (expected: {truth})\n{'=' * 70}")
        try:
            df = load(name, rel)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            continue
        print(f"  shape {df.shape}")

        r = {"dataset": name, "ground_truth": truth,
             "n_rows": len(df), "n_cols": df.shape[1]}
        r.update(d1_lookup_table(df, target))
        print(f"  D1 lookup      : {r['d1_verdict']}  "
              f"deps={r['d1_n_exact_deps']} compression={r['d1_compression']} "
              f"-> flag={r['d1_flag']}")
        r.update(d2_signal(df, person, target))
        print(f"  D2 signal      : AUC={r['d2_auc']} on '{r['d2_target']}' "
              f"-> flag={r['d2_flag']}")
        r.update(d3_temporal(df, person, date))
        _d3 = (f"lag1 ac={r['d3_lag1_ac']} ({r['d3_var']})"
               if r.get("d3_applicable") else "not applicable (no physiological state variable)")
        print(f"  D3 temporal    : {_d3} -> flag={r['d3_flag']}")
        r.update(d4_consistency(df))
        print(f"  D4 consistency : BMI disagreement={r['d4_disagree_pct']}% "
              f"-> flag={r['d4_flag']}")

        r["any_flag"] = bool(r["d1_flag"] or r["d2_flag"] or
                             r["d3_flag"] or r["d4_flag"])
        r["caught_by"] = ",".join([d for d, f in
                                   [("D1", r["d1_flag"]), ("D2", r["d2_flag"]),
                                    ("D3", r["d3_flag"]), ("D4", r["d4_flag"])] if f]) or "none"
        print(f"  VERDICT        : {'FLAGGED' if r['any_flag'] else 'clean'} "
              f"[{r['caught_by']}]")
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TEXTS, "e2_battery_results.csv"), index=False)

    fab = out[out.ground_truth == "FABRICATED"]
    real = out[out.ground_truth == "REAL"]
    cov = {
        "n_fabricated": int(len(fab)),
        "n_real": int(len(real)),
        "caught_by_D1_alone": int(fab.d1_flag.sum()),
        "caught_by_D1_to_D4": int(fab.any_flag.sum()),
        "false_alarms_on_real": int(real.any_flag.sum()),
        "per_detector_on_fabricated": {
            "D1_lookup": int(fab.d1_flag.sum()),
            "D2_signal": int(fab.d2_flag.sum()),
            "D3_temporal": int(fab.d3_flag.sum()),
            "D4_consistency": int(fab.d4_flag.sum()),
        },
    }
    with open(os.path.join(TEXTS, "e2_detector_coverage.json"), "w") as f:
        json.dump(cov, f, indent=1)

    lines = [
        "EXPERIMENT 2: FABRICATION DETECTION BATTERY",
        "=" * 78,
        "",
        f"{'dataset':<24}{'truth':<12}{'D1':>5}{'D2':>5}{'D3':>5}{'D4':>5}  caught by",
        "-" * 78,
    ]
    for _, x in out.iterrows():
        lines.append(f"{x.dataset:<24}{x.ground_truth:<12}"
                     f"{'X' if x.d1_flag else '.':>5}"
                     f"{'X' if x.d2_flag else '.':>5}"
                     f"{'X' if x.d3_flag else '.':>5}"
                     f"{'X' if x.d4_flag else '.':>5}  {x.caught_by}")
    lines += [
        "-" * 78, "",
        "DETECTOR EVIDENCE",
        f"{'dataset':<24}{'D1 verdict':<26}{'D2 AUC':>9}{'D3 lag1':>9}{'D4 BMI%':>9}",
        "-" * 78,
    ]
    for _, x in out.iterrows():
        lines.append(f"{x.dataset:<24}{str(x.d1_verdict):<26}"
                     f"{x.d2_auc if pd.notna(x.d2_auc) else 'n/a':>9}"
                     f"{x.d3_lag1_ac if pd.notna(x.d3_lag1_ac) else 'n/a':>9}"
                     f"{x.d4_disagree_pct if pd.notna(x.d4_disagree_pct) else 'n/a':>9}")
    lines += [
        "", "=" * 78,
        "COVERAGE",
        f"  fabricated datasets                : {cov['n_fabricated']}",
        f"  caught by D1 alone (published screen): {cov['caught_by_D1_alone']}",
        f"  caught by the full battery          : {cov['caught_by_D1_to_D4']}",
        f"  false alarms on real datasets       : {cov['false_alarms_on_real']}"
        f" of {cov['n_real']}",
        "",
        "  per detector, on fabricated data:",
    ]
    for k, v in cov["per_detector_on_fabricated"].items():
        lines.append(f"    {k:<18}{v}")
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e2_battery_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
