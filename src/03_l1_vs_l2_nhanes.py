"""
03_l1_vs_l2_nhanes.py   (EXPERIMENT 3)
======================================
Question
--------
The book's engine keys recommendations on age, sex and a self-reported activity
level. That is rung L1 of the personalization ladder. The obvious next step a
developer would take is to add body measurements: BMI, then waist and
waist-to-height ratio. That is L2.

So: how much does each rung actually explain, and explain OF WHAT?

The distinction that matters is between explaining a person's BODY and
explaining their BEHAVIOUR. A recommender that tells you how to act needs the
second. This experiment measures both on the same people.

Data
----
NHANES 2013-2014 (datasets/nhanes_cdc). Chosen because it is the only dataset in
the study with documented provenance, a MEASURED waist circumference, age in
single years, behaviour and body on the same participants, and complete survey
design variables.

This is also an independent replication: the earlier project measured the same
relationship on NHANES 2021-2023. A different cycle, collected years apart, is a
genuine out-of-sample test of that finding rather than a re-run of it.

Survey design
-------------
Point estimates are weighted with WTMEC2YR (examination weights, appropriate
because waist and BMI come from the physical examination). Confidence intervals
use a delete-one-PSU jackknife over the 15 strata x 2 PSU design, which is the
standard approach for NHANES and correctly widens intervals for clustering.
Unweighted estimates are reported alongside so the effect of weighting is visible
rather than hidden.

Outputs
-------
outputs/texts/e3_variance_explained.csv    R2 per predictor set per target
outputs/texts/e3_prediction_auc.csv        grouped-CV AUC for guideline adherence
outputs/texts/e3_summary.txt               readable summary
outputs/texts/e3_nhanes_sample.json        sample sizes and derivation audit
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets", "nhanes_cdc")
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42

# NHANES codes missing/refused/don't know as 7, 9, 77, 99, 7777, 9999 depending
# on the field width. Treating these as real values is a classic and severe bug.
DK_REFUSED = {7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999}


def clean_codes(s, valid_max=None):
    """Blank out don't-know / refused codes and impossible values."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(~s.isin(DK_REFUSED))
    if valid_max is not None:
        s = s.where(s <= valid_max)
    return s


def load_nhanes():
    dem = pd.read_csv(os.path.join(DATA, "demographic.csv"), low_memory=False)
    exa = pd.read_csv(os.path.join(DATA, "examination.csv"), low_memory=False)
    que = pd.read_csv(os.path.join(DATA, "questionnaire.csv"), low_memory=False)

    keep_dem = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2",
                "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
    keep_exa = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BPXSY1", "BPXDI1"]
    paq_days = ["PAQ610", "PAQ625", "PAQ640", "PAQ655", "PAQ670"]
    paq_mins = ["PAD615", "PAD630", "PAD645", "PAD660", "PAD675"]
    paq_yn = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
    keep_que = (["SEQN", "PAD680", "SLD010H", "SMQ020", "ALQ101",
                 "MCQ080", "DIQ010", "BPQ020"] + paq_days + paq_mins + paq_yn)

    dem = dem[[c for c in keep_dem if c in dem.columns]]
    exa = exa[[c for c in keep_exa if c in exa.columns]]
    que = que[[c for c in keep_que if c in que.columns]]

    df = dem.merge(exa, on="SEQN", how="left").merge(que, on="SEQN", how="left")
    return df, paq_days, paq_mins, paq_yn


def derive(df, paq_days, paq_mins, paq_yn):
    """Build the analytic variables, and record how many rows each step costs."""
    audit = {"n_raw": int(len(df))}

    df = df[df.RIDAGEYR >= 18].copy()
    audit["n_adults"] = int(len(df))

    df["sex"] = df.RIAGENDR.map({1: "male", 2: "female"})
    df["age"] = df.RIDAGEYR
    df["ethnicity"] = df.RIDRETH3.map({
        1: "mexican_american", 2: "other_hispanic", 3: "white_nh",
        4: "black_nh", 6: "asian_nh", 7: "other_multi"})
    df["education"] = clean_codes(df.get("DMDEDUC2"), valid_max=5)
    df["income_ratio"] = pd.to_numeric(df.get("INDFMPIR"), errors="coerce")

    # ---- body
    df["bmi"] = pd.to_numeric(df.BMXBMI, errors="coerce")
    df["waist"] = pd.to_numeric(df.BMXWAIST, errors="coerce")
    df["height"] = pd.to_numeric(df.BMXHT, errors="coerce")
    df["weight"] = pd.to_numeric(df.BMXWT, errors="coerce")
    df["whtr"] = df.waist / df.height
    df["sbp"] = clean_codes(df.get("BPXSY1"), valid_max=300)
    df["bmi_cat"] = pd.cut(df.bmi, [0, 18.5, 25, 30, 100],
                           labels=["under", "normal", "over", "obese"])

    # ---- behaviour: weekly moderate-equivalent minutes
    # Vigorous activity is weighted x2, the standard convention used by WHO and
    # by the US guidelines when expressing everything in moderate equivalents.
    # Domains: 0 vigorous work, 1 moderate work, 2 walk/bicycle,
    #          3 vigorous recreation, 4 moderate recreation
    weights = [2, 1, 1, 2, 1]
    total = pd.Series(0.0, index=df.index)
    any_valid = pd.Series(False, index=df.index)
    for w, dcol, mcol, ycol in zip(weights, paq_days, paq_mins, paq_yn):
        if dcol not in df.columns or mcol not in df.columns:
            continue
        days = clean_codes(df[dcol], valid_max=7)
        mins = clean_codes(df[mcol], valid_max=1440)
        yes = pd.to_numeric(df[ycol], errors="coerce") == 1
        # a "no" to the gate question is a genuine zero, not missing
        d = days.where(yes, 0.0)
        m = mins.where(yes, 0.0)
        contrib = (d.fillna(0) * m.fillna(0)) * w
        total = total + contrib
        any_valid = any_valid | yes.notna()

    df["mvpa_week"] = total.where(any_valid)
    df["meets_guideline"] = (df.mvpa_week >= 150).astype(float)
    df.loc[df.mvpa_week.isna(), "meets_guideline"] = np.nan

    df["sedentary_min"] = clean_codes(df.get("PAD680"), valid_max=1440)
    df["sleep_hours"] = clean_codes(df.get("SLD010H"), valid_max=14)

    df["weight_kg"] = df.weight
    df["wt"] = pd.to_numeric(df.WTMEC2YR, errors="coerce")
    df["strata"] = df.SDMVSTRA
    df["psu"] = df.SDMVPSU

    audit["n_with_bmi"] = int(df.bmi.notna().sum())
    audit["n_with_waist"] = int(df.waist.notna().sum())
    audit["n_with_mvpa"] = int(df.mvpa_week.notna().sum())
    audit["n_with_weight_var"] = int(df.wt.notna().sum())
    audit["mvpa_median"] = float(df.mvpa_week.median())
    audit["mvpa_pct_zero"] = float((df.mvpa_week == 0).mean() * 100)
    audit["pct_meets_guideline_unweighted"] = float(df.meets_guideline.mean() * 100)
    return df, audit


# ------------------------------------------------------------------ weighted R2
def weighted_r2(y, X, w):
    """Weighted least squares R2 with an intercept."""
    m = np.isfinite(y) & np.isfinite(w) & np.all(np.isfinite(X), axis=1)
    y, X, w = y[m], X[m], w[m]
    if len(y) < 50:
        return np.nan, 0
    X1 = np.column_stack([np.ones(len(X)), X])
    W = w / w.sum()
    # solve weighted normal equations
    XtW = X1.T * W
    try:
        beta = np.linalg.solve(XtW @ X1, XtW @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X1 * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
    pred = X1 @ beta
    ybar = np.sum(W * y)
    ss_res = np.sum(W * (y - pred) ** 2)
    ss_tot = np.sum(W * (y - ybar) ** 2)
    return float(1 - ss_res / ss_tot), int(len(y))


def jackknife_r2(y, X, w, strata, psu, n_rep=None):
    """
    Delete-one-PSU jackknife over the NHANES design. Each replicate drops one
    PSU within one stratum and reweights the remaining PSU in that stratum.
    """
    combos = pd.DataFrame({"s": strata, "p": psu}).drop_duplicates().values
    ests = []
    for s, p in combos:
        keep = ~((strata == s) & (psu == p))
        wr = w.copy().astype(float)
        # double the weight of the surviving PSU in that stratum
        same_stratum_other = (strata == s) & (psu != p)
        wr[same_stratum_other] = wr[same_stratum_other] * 2.0
        wr[~keep] = 0.0
        r, _ = weighted_r2(y, X, wr)
        if np.isfinite(r):
            ests.append(r)
    if len(ests) < 4:
        return (np.nan, np.nan)
    ests = np.array(ests)
    full, _ = weighted_r2(y, X, w)
    # jackknife variance for a delete-one design
    var = ((len(ests) - 1) / len(ests)) * np.sum((ests - ests.mean()) ** 2)
    se = np.sqrt(var)
    return (float(full - 1.96 * se), float(full + 1.96 * se))


def design_matrix(df, spec):
    """Build a numeric design matrix from a list of column specs."""
    parts = []
    for c in spec:
        if c == "sex":
            parts.append((df.sex == "male").astype(float).values[:, None])
        elif c == "ethnicity":
            d = pd.get_dummies(df.ethnicity, drop_first=True).astype(float)
            parts.append(d.values)
        elif c == "age2":
            parts.append((df.age.astype(float) ** 2).values[:, None])
        else:
            parts.append(pd.to_numeric(df[c], errors="coerce").values[:, None])
    return np.hstack(parts)


LADDER = {
    "L1  age + sex": ["age", "sex"],
    "L1+ age + sex + ethnicity + SES": ["age", "sex", "ethnicity",
                                        "education", "income_ratio"],
    "L2  + BMI": ["age", "sex", "bmi"],
    "L2+ + waist": ["age", "sex", "bmi", "waist"],
    "L2++ + waist-to-height + BP": ["age", "sex", "bmi", "waist", "whtr", "sbp"],
}

TARGETS = {
    "BODY: waist-to-height ratio": "whtr",
    "BODY: waist circumference": "waist",
    "BEHAVIOUR: weekly MVPA minutes": "mvpa_week",
    "BEHAVIOUR: sedentary minutes/day": "sedentary_min",
    "BEHAVIOUR: sleep hours": "sleep_hours",
}


def main():
    raw, pdays, pmins, pyn = load_nhanes()
    df, audit = derive(raw, pdays, pmins, pyn)
    print(json.dumps(audit, indent=1))

    rows = []
    for tname, tcol in TARGETS.items():
        for lname, spec in LADDER.items():
            # A target must not be predicted from itself OR from its own
            # components. waist-to-height is waist/height, so any spec
            # containing waist or height is circular and is excluded. This was
            # caught on the first run, where "L2+ + waist" reported R2 = 0.946
            # for waist-to-height, which measures arithmetic, not explanation.
            COMPONENTS = {"whtr": {"waist", "height", "whtr"},
                          "waist": {"waist", "whtr"},
                          "bmi": {"bmi", "weight", "height"}}
            banned = COMPONENTS.get(tcol, {tcol})
            if banned & set(spec):
                continue
            sub = df.dropna(subset=[tcol, "wt"]).copy()
            X = design_matrix(sub, spec)
            y = pd.to_numeric(sub[tcol], errors="coerce").values.astype(float)
            w = sub.wt.values.astype(float)
            m = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w)
            if m.sum() < 100:
                continue
            r2w, n = weighted_r2(y[m], X[m], w[m])
            r2u, _ = weighted_r2(y[m], X[m], np.ones(m.sum()))
            lo, hi = jackknife_r2(y[m], X[m], w[m],
                                  sub.strata.values[m], sub.psu.values[m])
            rows.append({"target": tname, "predictors": lname, "n": n,
                         "R2_weighted": round(r2w, 4),
                         "R2_ci_low": round(lo, 4) if np.isfinite(lo) else np.nan,
                         "R2_ci_high": round(hi, 4) if np.isfinite(hi) else np.nan,
                         "R2_unweighted": round(r2u, 4)})
            print(f"{tname:<34}{lname:<32}n={n:<6}R2w={r2w:.4f} "
                  f"[{lo:.4f},{hi:.4f}]  R2u={r2u:.4f}")

    var = pd.DataFrame(rows)
    var.to_csv(os.path.join(TEXTS, "e3_variance_explained.csv"), index=False)

    # -------------------------------------------------- prediction of adherence
    arows = []
    sub = df.dropna(subset=["meets_guideline"]).copy()
    for lname, spec in LADDER.items():
        X = design_matrix(sub, spec)
        y = sub.meets_guideline.values.astype(int)
        m = np.all(np.isfinite(X), axis=1)
        X, yy = X[m], y[m]
        if len(np.unique(yy)) < 2 or len(yy) < 200:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=SEED))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        p = cross_val_predict(clf, X, yy, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(yy, p)
        arows.append({"predictors": lname, "n": int(len(yy)),
                      "AUC": round(float(auc), 4)})
        print(f"ADHERENCE  {lname:<32}n={len(yy):<6}AUC={auc:.4f}")
    # single-variable references
    for label, col in [("waist-to-height ratio alone", "whtr"),
                       ("BMI alone", "bmi"),
                       ("age alone", "age")]:
        s2 = sub.dropna(subset=[col])
        y = s2.meets_guideline.values.astype(int)
        X = s2[[col]].values.astype(float)
        if len(np.unique(y)) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=SEED))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        arows.append({"predictors": label, "n": int(len(y)),
                      "AUC": round(float(roc_auc_score(y, p)), 4)})
        print(f"ADHERENCE  {label:<32}n={len(y):<6}AUC={roc_auc_score(y,p):.4f}")

    pred = pd.DataFrame(arows)
    pred.to_csv(os.path.join(TEXTS, "e3_prediction_auc.csv"), index=False)

    with open(os.path.join(TEXTS, "e3_nhanes_sample.json"), "w") as f:
        json.dump(audit, f, indent=1)

    # -------------------------------------------------------------- summary
    body = var[var.target.str.startswith("BODY")]
    beh = var[var.target.str.startswith("BEHAVIOUR")]
    lines = [
        "EXPERIMENT 3: WHAT DO THE LOWER RUNGS OF THE LADDER EXPLAIN?",
        "NHANES 2013-2014, design-weighted (WTMEC2YR), jackknife CIs over "
        "15 strata x 2 PSU",
        "=" * 78, "",
        f"adults 18+                     : {audit['n_adults']:,}",
        f"with measured BMI              : {audit['n_with_bmi']:,}",
        f"with measured waist            : {audit['n_with_waist']:,}",
        f"with derivable weekly MVPA     : {audit['n_with_mvpa']:,}",
        f"median weekly MVPA (min)       : {audit['mvpa_median']:.0f}",
        f"reporting zero MVPA            : {audit['mvpa_pct_zero']:.1f}%",
        "",
        "VARIANCE EXPLAINED (weighted R2)",
        f"{'target':<34}{'predictors':<32}{'R2':>8}{'95% CI':>20}",
        "-" * 94,
    ]
    for _, r in var.iterrows():
        ci = (f"[{r.R2_ci_low:.3f}, {r.R2_ci_high:.3f}]"
              if np.isfinite(r.R2_ci_low) else "n/a")
        lines.append(f"{r.target:<34}{r.predictors:<32}{r.R2_weighted:>8.4f}{ci:>20}")
    lines += ["", "PREDICTING WHO MEETS THE ACTIVITY GUIDELINE (grouped CV AUC)",
              f"{'predictors':<34}{'n':>8}{'AUC':>8}", "-" * 50]
    for _, r in pred.iterrows():
        lines.append(f"{r.predictors:<34}{r.n:>8}{r.AUC:>8.4f}")

    if len(body) and len(beh):
        lines += ["", "=" * 78, "HEADLINE",
                  f"  best R2 on a BODY target      : {body.R2_weighted.max():.4f}",
                  f"  best R2 on a BEHAVIOUR target : {beh.R2_weighted.max():.4f}",
                  f"  ratio                         : "
                  f"{body.R2_weighted.max() / max(beh.R2_weighted.max(), 1e-9):.1f}x"]
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e3_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
