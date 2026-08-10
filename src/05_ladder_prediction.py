"""
05_ladder_prediction.py   (EXPERIMENT 5)
========================================
Question
--------
E4 showed that behaviour is mostly a property of the day, not the person, and
that any static profile is capped near a quarter of the variance. E5 tests the
whole ladder out of sample to find out what each rung is actually worth.

    L0  population constant          what the book prescribes for activity
    L1  age + sex                    what the book's engine uses
    L2  + BMI                        the obvious next step a developer takes
    L3  + the person's own history   requires logging
    L4  + yesterday's context        requires continuous passive sensing

Protocol
--------
Forward chaining over the pooled calendar: every fold trains on the past and
tests on the future. All history features are shifted so that day t is predicted
only from days before t. Two datasets, LifeSnaps as primary and fitbit_mobius as
an independent replication with no demographics at all.

Targets: next-day steps, and next-day active minutes, plus a weekly aggregate.

Why two model families
----------------------
Ridge is reported as the headline because it is interpretable and cannot exploit
spurious interactions. Gradient boosting is reported alongside so that a reader
can see whether a stronger learner changes the ordering of the rungs. If the
ordering is the same under both, the conclusion is about the information, not
about the model.

Outputs
-------
outputs/texts/e5_ladder_results.csv    R2 and MAE per rung per fold
outputs/texts/e5_ladder_summary.txt    readable summary
outputs/texts/e5_feature_importance.csv which features carry the L3 gain
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42


def evaluate(d, specs, target, model_name):
    """Forward-chained evaluation of every rung."""
    folds = panel.forward_chain_folds(d)
    rows = []
    for rung, cols in specs.items():
        cols = [c for c in cols if c in d.columns]
        for fi, (tr, te) in enumerate(folds):
            ytr, yte = d[target].values[tr], d[target].values[te]
            if len(yte) < 30:
                continue

            if not cols:
                # L0: the population constant, estimated on the training past
                pred = np.full(len(yte), np.nanmean(ytr))
            else:
                Xtr, Xte = d[cols].values[tr], d[cols].values[te]
                # a rung whose columns are entirely missing cannot be fitted
                if np.all(~np.isfinite(Xtr)):
                    continue
                if model_name == "ridge":
                    mdl = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        RidgeCV(alphas=np.logspace(-3, 3, 25)))
                else:
                    mdl = make_pipeline(
                        SimpleImputer(strategy="median"),
                        HistGradientBoostingRegressor(
                            random_state=SEED, max_iter=250,
                            learning_rate=0.06, max_depth=4))
                try:
                    mdl.fit(Xtr, ytr)
                    pred = mdl.predict(Xte)
                except Exception:
                    continue

            rows.append({
                "target": target, "model": model_name, "rung": rung, "fold": fi,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "r2": float(r2_score(yte, pred)),
                "mae": float(mean_absolute_error(yte, pred)),
            })
    return rows


def feature_importance(d, specs, target):
    """
    Which features carry the L3 gain? Fit ridge on the full panel and report
    standardized coefficients, which are comparable across features.
    """
    cols = [c for c in specs.get("L3 + own behaviour history", []) if c in d.columns]
    # SimpleImputer silently drops columns that are entirely missing, which
    # would misalign the coefficient vector against the feature names.
    cols = [c for c in cols if np.isfinite(pd.to_numeric(d[c], errors="coerce")).any()]
    if not cols:
        return pd.DataFrame()
    X, y = d[cols].values, d[target].values
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        RidgeCV(alphas=np.logspace(-3, 3, 25)))
    mdl.fit(X, y)
    coef = mdl[-1].coef_
    return (pd.DataFrame({"feature": cols, "std_coef": coef,
                          "abs_coef": np.abs(coef)})
            .sort_values("abs_coef", ascending=False))


def main():
    all_rows, notes = [], {}

    for source in ["lifesnaps", "mobius"]:
        for target in ["steps", "very_active_minutes"]:
            try:
                d, specs = panel.build(source=source, target=target)
            except Exception as e:
                print(f"skip {source}/{target}: {e}")
                continue
            if target not in d.columns or d[target].notna().sum() < 300:
                continue
            # a person needs some history before any L3 feature exists
            d = d[d.hist_n >= 3].copy()
            notes[f"{source}_{target}"] = {
                "n_person_days": int(len(d)),
                "n_people": int(d.person.nunique()),
                "median_days_per_person": float(
                    d.groupby("person").size().median()),
            }
            print(f"\n=== {source} / {target} : {len(d)} person-days, "
                  f"{d.person.nunique()} people ===")
            for model_name in ["ridge", "gbm"]:
                rows = evaluate(d, specs, target, model_name)
                for r in rows:
                    r["source"] = source
                all_rows.extend(rows)
                agg = (pd.DataFrame(rows).groupby("rung")
                       .r2.mean().reindex(list(specs.keys())).dropna())
                for rung, v in agg.items():
                    print(f"  {model_name:<6}{rung:<32}R2={v:+.4f}")

            if source == "lifesnaps" and target == "steps":
                fi = feature_importance(d, specs, target)
                fi.to_csv(os.path.join(TEXTS, "e5_feature_importance.csv"),
                          index=False)

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(TEXTS, "e5_ladder_results.csv"), index=False)

    # ------------------------------------------------------------- summary
    lines = ["EXPERIMENT 5: WHAT IS EACH RUNG OF THE LADDER WORTH?",
             "Forward-chained, features strictly causal, two model families",
             "=" * 82, ""]
    for (src, tgt), g in res.groupby(["source", "target"]):
        key = f"{src}_{tgt}"
        n = notes.get(key, {})
        lines += [f"{src.upper()} / {tgt}   "
                  f"({n.get('n_person_days', '?')} person-days, "
                  f"{n.get('n_people', '?')} people)",
                  f"  {'rung':<32}{'ridge R2':>10}{'gbm R2':>10}"
                  f"{'ridge MAE':>12}", "  " + "-" * 64]
        piv = g.pivot_table(index="rung", columns="model",
                            values=["r2", "mae"], aggfunc="mean")
        order = [r for r in ["L0 population constant", "L1 demographics (the book)",
                             "L2 + body measurements", "L3 + own behaviour history",
                             "L4 + yesterday's context"] if r in piv.index]
        for rung in order:
            rr = piv.loc[rung, ("r2", "ridge")] if ("r2", "ridge") in piv.columns else np.nan
            rg = piv.loc[rung, ("r2", "gbm")] if ("r2", "gbm") in piv.columns else np.nan
            mr = piv.loc[rung, ("mae", "ridge")] if ("mae", "ridge") in piv.columns else np.nan
            lines.append(f"  {rung:<32}{rr:>+10.4f}{rg:>+10.4f}{mr:>12.1f}")
        lines.append("")

    ls = res[(res.source == "lifesnaps") & (res.target == "steps") &
             (res.model == "ridge")]
    if len(ls):
        m = ls.groupby("rung").r2.mean()
        def get(k):
            return m.get(k, np.nan)
        L0K = "L0 population constant"
        L1K = "L1 demographics (the book)"
        L2K = "L2 + body measurements"
        L3K = "L3 + own behaviour history"
        L4K = "L4 + yesterday's context"
        lines += ["=" * 82, "HEADLINE (LifeSnaps, next-day steps, ridge)",
                  f"  L0 population constant     : {get(L0K):+.4f}",
                  f"  L1 demographics            : {get(L1K):+.4f}",
                  f"  L2 + body measurements     : {get(L2K):+.4f}",
                  f"  L3 + own history           : {get(L3K):+.4f}",
                  f"  L4 + context               : {get(L4K):+.4f}",
                  "",
                  f"  gain from L1 to L2 (body)   : {get(L2K) - get(L1K):+.4f}",
                  f"  gain from L2 to L3 (logs)   : {get(L3K) - get(L2K):+.4f}",
                  f"  gain from L3 to L4 (sensing): {get(L4K) - get(L3K):+.4f}"]
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e5_ladder_summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(TEXTS, "e5_panel_notes.json"), "w") as f:
        json.dump(notes, f, indent=1)
    print("\n" + txt)


if __name__ == "__main__":
    main()
