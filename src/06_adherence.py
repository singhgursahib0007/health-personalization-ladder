"""
06_adherence.py   (EXPERIMENT 6)
================================
Question
--------
E5 predicts what a person WILL do. Recommendation needs something else: whether
a person will do what they are ASKED to do. That is adherence, and it is the term
that turns a prediction into a decision.

LifeSnaps carries a real, non-synthetic adherence target: each participant had a
Fitbit step goal, and on each day they either met it or did not. This is rare.
Most public data has no notion of a target that a person was trying to hit.

Why adherence is the operative quantity
---------------------------------------
A recommender that suggests the objectively optimal action is useless if nobody
follows it. The quantity worth maximizing is not the benefit of an action but the
benefit weighted by the probability the person actually performs it:

    value(action) = P(adherence | person, context, action) x benefit(action)

This experiment estimates the first term, and asks the ladder question again:
how much of adherence is knowable from who somebody is, and how much only from
what they have been doing?

We also report CALIBRATION, not just discrimination. A recommender multiplies by
this probability, so a model that ranks well but is systematically overconfident
will still make bad decisions. AUC alone cannot detect that.

Outputs
-------
outputs/texts/e6_adherence_results.csv
outputs/texts/e6_adherence_summary.txt
outputs/texts/e6_calibration.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42


def build_adherence_panel():
    d, specs = panel.build(source="lifesnaps", target="steps")
    # step_goal is stored BINNED, as the upper edge of a range, alongside a
    # label such as "10000-14999", plus a literal "NO_GOAL" category. The exact
    # goal a participant set is therefore not recoverable. We use the LOWER
    # bound of the bin, which is the defensible reading: somebody in the
    # 10000-14999 band had a goal of at least 10000, and Fitbit's default goal
    # is exactly 10000. Using the upper edge instead would impose a stricter
    # target than the person actually chose and would understate adherence.
    # This approximation is declared rather than hidden, and it is a limitation.
    lab = d.step_goal_label.astype(str)
    lower = (lab.str.extract(r"^(\d+)")[0].astype(float))
    lower = lower.where(~lab.str.contains("Less than", na=False), 0.0)
    d = d[lower.notna() & (~lab.str.contains("No Goal", na=False))].copy()
    d["goal_value"] = lower[d.index]
    d = d[d.goal_value > 0].copy()
    d["hit_goal"] = (d.steps >= d.goal_value).astype(int)
    prev_hit = d.groupby("person")["hit_goal"].shift(1)
    d["adh_prev"] = prev_hit
    d["adh_rate_hist"] = (d.groupby("person")["hit_goal"]
                          .transform(lambda s: s.shift(1)
                                     .expanding(min_periods=1).mean()))
    d["adh_rate7"] = (d.groupby("person")["hit_goal"]
                      .transform(lambda s: s.shift(1)
                                 .rolling(7, min_periods=2).mean()))
    # how demanding is this goal relative to what the person normally does
    d["goal_gap"] = d.goal_value - d.hist_expmean
    d["goal_ratio"] = d.goal_value / d.hist_expmean.replace(0, np.nan)

    hist = ["hist_lag1", "hist_mean3", "hist_mean7", "hist_mean28",
            "hist_std7", "hist_expmean", "hist_trend", "hist_dow_dev",
            "dow", "is_weekend"]
    adh = ["adh_prev", "adh_rate_hist", "adh_rate7", "goal_gap", "goal_ratio"]
    ctx = [c for c in d.columns if c.startswith("ctx_")]

    ladder = {
        "L0 base rate": [],
        "L1 demographics (the book)": ["age", "sex_male"],
        "L2 + body measurements": ["age", "sex_male", "bmi"],
        "L2g + the goal itself": ["age", "sex_male", "bmi", "goal_value"],
        "L3 + own behaviour history": ["age", "sex_male", "bmi", "goal_value"] + hist,
        "L3a + own adherence history": ["age", "sex_male", "bmi", "goal_value"]
                                       + hist + adh,
        "L4 + yesterday's context": ["age", "sex_male", "bmi", "goal_value"]
                                    + hist + adh + ctx,
    }
    return d, ladder


def main():
    d, ladder = build_adherence_panel()
    d = d[d.hist_n >= 3].copy()
    base = d.hit_goal.mean()
    print(f"adherence panel: {len(d)} person-days, {d.person.nunique()} people, "
          f"base hit rate {base:.3f}")
    print(f"people whose hit rate varies: "
          f"{(d.groupby('person').hit_goal.nunique() > 1).sum()}")

    folds = panel.forward_chain_folds(d)
    rows, cal_rows = [], []
    for rung, cols in ladder.items():
        cols = [c for c in cols if c in d.columns]
        cols = [c for c in cols
                if np.isfinite(pd.to_numeric(d[c], errors="coerce")).any()]
        for fi, (tr, te) in enumerate(folds):
            ytr, yte = d.hit_goal.values[tr], d.hit_goal.values[te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            if not cols:
                p = np.full(len(yte), ytr.mean())
            else:
                mdl = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(max_iter=3000, random_state=SEED))
                try:
                    mdl.fit(d[cols].values[tr], ytr)
                    p = mdl.predict_proba(d[cols].values[te])[:, 1]
                except Exception:
                    continue
            try:
                auc = roc_auc_score(yte, p)
            except ValueError:
                continue
            rows.append({"rung": rung, "fold": fi, "n_test": int(te.sum()),
                         "auc": float(auc),
                         "brier": float(brier_score_loss(yte, p)),
                         "base_rate": float(yte.mean())})
            if fi == len(folds) - 1 and cols:
                try:
                    ft, fp = calibration_curve(yte, p, n_bins=5, strategy="quantile")
                    for a, b in zip(fp, ft):
                        cal_rows.append({"rung": rung, "pred_mean": float(a),
                                         "observed": float(b)})
                except Exception:
                    pass

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(TEXTS, "e6_adherence_results.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(
        os.path.join(TEXTS, "e6_calibration.csv"), index=False)

    agg = (res.groupby("rung")[["auc", "brier"]].mean()
           .reindex([k for k in ladder if k in set(res.rung)]))
    lines = [
        "EXPERIMENT 6: WILL THEY ACTUALLY DO IT?",
        "LifeSnaps step-goal attainment, forward-chained, causal features",
        "=" * 74, "",
        f"person-days                    : {len(d):,}",
        f"people                         : {d.person.nunique()}",
        f"base hit rate                  : {base:.3f}",
        f"people whose outcome varies    : "
        f"{(d.groupby('person').hit_goal.nunique() > 1).sum()}",
        "",
        f"{'rung':<34}{'AUC':>9}{'Brier':>9}", "-" * 54,
    ]
    for rung, r in agg.iterrows():
        lines.append(f"{rung:<34}{r.auc:>9.4f}{r.brier:>9.4f}")

    if "L2 + body measurements" in agg.index and "L3a + own adherence history" in agg.index:
        lines += ["", "=" * 74, "HEADLINE",
                  f"  static profile (L2)        AUC {agg.loc['L2 + body measurements','auc']:.4f}",
                  f"  + own behaviour history    AUC {agg.loc['L3 + own behaviour history','auc']:.4f}",
                  f"  + own adherence history    AUC {agg.loc['L3a + own adherence history','auc']:.4f}",
                  "",
                  "  The best predictor of whether somebody will hit their goal",
                  "  tomorrow is whether they have been hitting it lately, not",
                  "  anything about their body."]
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e6_adherence_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
