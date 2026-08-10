"""
10_bootstrap_ci.py   (uncertainty for E5, E6, E7)
=================================================
Why
---
The paper's argument turns on differences of +0.0004, +0.0032 and -0.0057 in
out-of-sample R2, measured on 71 people. Reporting the sign of a third-decimal
quantity without an interval is indefensible, and an independent audit said so.

This script attaches a cluster bootstrap over PEOPLE, which is the correct
resampling unit for a panel: the sampling variability that matters is which
participants happened to be recruited, not which of their days we look at.

Three quantities get intervals:
  1. E5, ladder R2 per rung on next-day steps (full refit per replicate)
  2. E6, adherence AUC per rung (full refit per replicate)
  3. E7, new-user R2 per rung (bootstrap over the leave-one-person-out
     predictions; refitting 71 folds per replicate is not affordable, and the
     dominant uncertainty is which people are in the sample)

MARGINAL gains are bootstrapped PAIRED, inside the same replicate, because the
claims are about differences between rungs on the same participants. A paired
interval is much tighter than differencing two independent intervals, and it is
the correct one for the question.

Outputs
-------
outputs/texts/e5_ladder_ci.csv
outputs/texts/e6_adherence_ci.csv
outputs/texts/e7_privacy_ci.csv
outputs/texts/bootstrap_summary.txt
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
SEED = 42
B_REFIT = 300      # replicates for the refit bootstraps
B_PRED = 4000      # replicates for the prediction-level bootstrap

RUNGS = ["L0 population constant", "L1 demographics (the book)",
         "L2 + body measurements", "L3 + own behaviour history",
         "L4 + yesterday's context"]


def resample_people(d, rng):
    """Cluster bootstrap: draw people with replacement, keep them distinct."""
    people = d.person.unique()
    draw = rng.choice(people, len(people), replace=True)
    parts = []
    for j, p in enumerate(draw):
        g = d[d.person == p].copy()
        g["person"] = f"{p}__{j}"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def ladder_scores(d, specs, target, task):
    """One evaluation pass over all rungs. Returns {rung: metric}."""
    folds = panel.forward_chain_folds(d)
    out = {}
    for rung in specs:
        cols = [c for c in specs[rung] if c in d.columns]
        cols = [c for c in cols
                if np.isfinite(pd.to_numeric(d[c], errors="coerce")).any()]
        vals = []
        for tr, te in folds:
            y = d[target].values
            ytr, yte = y[tr], y[te]
            if len(yte) < 30:
                continue
            if task == "auc" and (len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2):
                continue
            try:
                if not cols:
                    pred = (np.full(len(yte), np.nanmean(ytr)) if task == "r2"
                            else np.full(len(yte), ytr.mean()))
                elif task == "r2":
                    m = make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler(),
                                      RidgeCV(alphas=np.logspace(-3, 3, 15)))
                    m.fit(d[cols].values[tr], ytr)
                    pred = m.predict(d[cols].values[te])
                else:
                    m = make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler(),
                                      LogisticRegression(max_iter=2000,
                                                         random_state=SEED))
                    m.fit(d[cols].values[tr], ytr)
                    pred = m.predict_proba(d[cols].values[te])[:, 1]
                vals.append(r2_score(yte, pred) if task == "r2"
                            else roc_auc_score(yte, pred))
            except Exception:
                continue
        if vals:
            out[rung] = float(np.mean(vals))
    return out


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray([x for x in a if np.isfinite(x)])
    if len(a) < 30:
        return (np.nan, np.nan)
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi)))


def run_refit_bootstrap(build_fn, target, task, rungs, label):
    d, specs = build_fn()
    specs = {k: v for k, v in specs.items() if k in rungs}
    point = ladder_scores(d, specs, target, task)

    rng = np.random.RandomState(SEED)
    reps = {r: [] for r in specs}
    for b in range(B_REFIT):
        db = resample_people(d, rng)
        s = ladder_scores(db, specs, target, task)
        for r in specs:
            reps[r].append(s.get(r, np.nan))
        if (b + 1) % 50 == 0:
            print(f"  {label}: {b + 1}/{B_REFIT}")

    rows = []
    order = [r for r in rungs if r in specs]
    for i, r in enumerate(order):
        lo, hi = ci(reps[r])
        row = {"rung": r, "point": round(point.get(r, np.nan), 4),
               "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        if i > 0:
            prev = order[i - 1]
            # PAIRED difference within each replicate
            diffs = [a - b for a, b in zip(reps[r], reps[prev])
                     if np.isfinite(a) and np.isfinite(b)]
            dlo, dhi = ci(diffs)
            row["marginal"] = round(point.get(r, np.nan) - point.get(prev, np.nan), 4)
            row["marg_ci_low"] = round(dlo, 4)
            row["marg_ci_high"] = round(dhi, 4)
            row["excludes_zero"] = bool(dlo > 0 or dhi < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("E5: ladder R2, next-day steps")
    e5 = run_refit_bootstrap(
        lambda: (lambda dd: (dd[0][dd[0].hist_n >= 3].copy(), dd[1]))(
            panel.build("lifesnaps", "steps")),
        "steps", "r2", RUNGS, "E5")
    e5.to_csv(os.path.join(TEXTS, "e5_ladder_ci.csv"), index=False)
    print(e5.to_string(index=False))

    print("\nE6: adherence AUC")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "adh", os.path.join(HERE, "06_adherence.py"))
    adh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adh)

    def build_adh():
        d, ladder = adh.build_adherence_panel()
        return d[d.hist_n >= 3].copy(), ladder

    arungs = ["L0 base rate", "L1 demographics (the book)",
              "L2 + body measurements", "L2g + the goal itself",
              "L3 + own behaviour history", "L3a + own adherence history",
              "L4 + yesterday's context"]
    e6 = run_refit_bootstrap(build_adh, "hit_goal", "auc", arungs, "E6")
    e6.to_csv(os.path.join(TEXTS, "e6_adherence_ci.csv"), index=False)
    print(e6.to_string(index=False))

    # ---------------------------------------------------- E7 prediction-level
    print("\nE7: new-user R2 (bootstrap over leave-one-person-out predictions)")
    d, specs = panel.build("lifesnaps", "steps")
    d = d.reset_index(drop=True)
    from sklearn.model_selection import LeaveOneGroupOut
    logo, groups = LeaveOneGroupOut(), d.person.values
    preds = {}
    for rung in RUNGS:
        cols = [c for c in specs.get(rung, []) if c in d.columns
                and np.isfinite(pd.to_numeric(d[c], errors="coerce")).any()]
        p = np.full(len(d), np.nan)
        for tr_i, te_i in logo.split(d, groups=groups):
            try:
                if not cols:
                    p[te_i] = np.nanmean(d.steps.values[tr_i])
                else:
                    m = make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler(),
                                      RidgeCV(alphas=np.logspace(-3, 3, 15)))
                    m.fit(d[cols].values[tr_i], d.steps.values[tr_i])
                    p[te_i] = m.predict(d[cols].values[te_i])
            except Exception:
                pass
        preds[rung] = p

    y = d.steps.values
    people = d.person.values
    uniq = np.unique(people)
    idx_by_person = {p: np.where(people == p)[0] for p in uniq}
    rng = np.random.RandomState(SEED)
    reps = {r: [] for r in RUNGS}
    for b in range(B_PRED):
        draw = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_by_person[p] for p in draw])
        for r in RUNGS:
            pr, yy = preds[r][sel], y[sel]
            m = np.isfinite(pr) & np.isfinite(yy)
            reps[r].append(r2_score(yy[m], pr[m]) if m.sum() > 100 else np.nan)

    rows = []
    for i, r in enumerate(RUNGS):
        mm = np.isfinite(preds[r]) & np.isfinite(y)
        pt = r2_score(y[mm], preds[r][mm])
        lo, hi = ci(reps[r])
        row = {"rung": r, "point": round(pt, 4),
               "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        if i > 0:
            prev = RUNGS[i - 1]
            diffs = [a - b for a, b in zip(reps[r], reps[prev])
                     if np.isfinite(a) and np.isfinite(b)]
            dlo, dhi = ci(diffs)
            mp = np.isfinite(preds[prev]) & np.isfinite(y)
            row["marginal"] = round(pt - r2_score(y[mp], preds[prev][mp]), 4)
            row["marg_ci_low"], row["marg_ci_high"] = round(dlo, 4), round(dhi, 4)
            row["excludes_zero"] = bool(dlo > 0 or dhi < 0)
        rows.append(row)
    e7 = pd.DataFrame(rows)
    e7.to_csv(os.path.join(TEXTS, "e7_privacy_ci.csv"), index=False)
    print(e7.to_string(index=False))

    # ------------------------------------------------------------- summary
    L = ["CLUSTER BOOTSTRAP OVER PEOPLE", "=" * 78,
         f"E5 and E6: {B_REFIT} replicates, full refit per replicate.",
         f"E7: {B_PRED} replicates over leave-one-person-out predictions.",
         "Marginal gains are bootstrapped PAIRED within each replicate.", ""]
    for name, df in [("E5 ladder, next-day steps R2", e5),
                     ("E6 adherence AUC", e6),
                     ("E7 new-user R2", e7)]:
        L += [name, f"  {'rung':<32}{'point':>9}{'95% CI':>20}"
                    f"{'marginal':>10}{'marginal 95% CI':>22}{'sig':>5}",
              "  " + "-" * 98]
        for _, r in df.iterrows():
            mg = (f"{r.marginal:+.4f}" if "marginal" in r and pd.notna(r.get("marginal"))
                  else "")
            mci = (f"[{r.marg_ci_low:+.4f}, {r.marg_ci_high:+.4f}]"
                   if "marg_ci_low" in r and pd.notna(r.get("marg_ci_low")) else "")
            sig = ("yes" if r.get("excludes_zero") is True
                   else ("no" if "excludes_zero" in r and pd.notna(r.get("excludes_zero")) else ""))
            L.append(f"  {r.rung:<32}{r.point:>+9.4f}"
                     f"{f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]':>20}"
                     f"{mg:>10}{mci:>22}{sig:>5}")
        L.append("")
    txt = "\n".join(L)
    with open(os.path.join(TEXTS, "bootstrap_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
