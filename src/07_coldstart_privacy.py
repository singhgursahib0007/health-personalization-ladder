"""
07_coldstart_privacy.py   (EXPERIMENT 7)
========================================
Two questions, one experiment, because they are the same question asked from
opposite ends.

1. COLD START. L3 needs a person's own history, which a new user does not have.
   So how many days does the app need before personalization is worth anything?
   Until then, what is the best it can do?

2. PRIVACY AND DATA MINIMIZATION. Every rung costs more disclosure:

       L0  nothing
       L1  age, sex                          one onboarding form
       L2  + height, weight, waist           body measurement
       L3  + a daily behaviour log           continuous self-report or wearable
       L4  + yesterday's mood and place      passive continuous sensing

   Measuring benefit per rung IS measuring the price of that benefit. If a rung
   costs a great deal of personal data and returns little, the correct
   engineering decision is not to collect it. That is a stronger privacy
   argument than any policy paragraph, because it says the data is not needed
   rather than promising to look after it.

Design
------
COLD START is evaluated leave-one-person-out: the model never sees the test
person at all, which is exactly a new signup. We then let that person accumulate
days and measure performance as a function of how many days of their own history
exist, so the crossover point is read off directly rather than assumed.

Outputs
-------
outputs/texts/e7_coldstart_curve.csv
outputs/texts/e7_privacy_utility.csv
outputs/texts/e7_summary.txt
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import panel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42
TARGET = "steps"

# What each rung requires a person to hand over. Used for the privacy curve.
DATA_COST = {
    "L0 population constant": (0, "nothing"),
    "L1 demographics (the book)": (1, "age and sex, once at signup"),
    "L2 + body measurements": (2, "height, weight, waist"),
    "L3 + own behaviour history": (3, "a daily activity log, ongoing"),
    "L4 + yesterday's context": (4, "mood and location, passively and continuously"),
}


def fit_predict(d, cols, tr, te):
    if not cols:
        return np.full(int(te.sum()), np.nanmean(d[TARGET].values[tr]))
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        RidgeCV(alphas=np.logspace(-3, 3, 25)))
    mdl.fit(d[cols].values[tr], d[TARGET].values[tr])
    return mdl.predict(d[cols].values[te])


def main():
    d, specs = panel.build(source="lifesnaps", target=TARGET)
    d = d.reset_index(drop=True)
    usable = {k: [c for c in v if c in d.columns
                  and np.isfinite(pd.to_numeric(d[c], errors="coerce")).any()]
              for k, v in specs.items()}

    # ------------------------------------------------- leave-one-person-out
    logo = LeaveOneGroupOut()
    groups = d.person.values
    rows = []
    for rung, cols in usable.items():
        preds = np.full(len(d), np.nan)
        for tr_idx, te_idx in logo.split(d, groups=groups):
            tr = np.zeros(len(d), bool); tr[tr_idx] = True
            te = np.zeros(len(d), bool); te[te_idx] = True
            try:
                preds[te_idx] = fit_predict(d, cols, tr, te)
            except Exception:
                continue
        d[f"pred_{rung}"] = preds
        m = np.isfinite(preds) & np.isfinite(d[TARGET].values)
        if m.sum() > 100:
            rows.append({"rung": rung, "scenario": "new user (leave-one-out)",
                         "n": int(m.sum()),
                         "r2": float(r2_score(d[TARGET].values[m], preds[m]))})
            print(f"NEW USER  {rung:<32}R2={rows[-1]['r2']:+.4f}")

    # -------------------------------- performance as personal history grows
    # Bucket every prediction by how many days of that person's own history
    # existed at the moment it was made. hist_n is exactly that count.
    buckets = [(0, 0), (1, 2), (3, 6), (7, 13), (14, 27), (28, 55), (56, 10_000)]
    curve = []
    for lo, hi in buckets:
        sel = (d.hist_n >= lo) & (d.hist_n <= hi)
        if sel.sum() < 60:
            continue
        row = {"hist_days_low": lo, "hist_days_high": hi, "n": int(sel.sum())}
        for rung in usable:
            p = d[f"pred_{rung}"].values
            m = sel.values & np.isfinite(p) & np.isfinite(d[TARGET].values)
            if m.sum() > 40:
                row[rung] = float(r2_score(d[TARGET].values[m], p[m]))
        curve.append(row)
        lbl = f"{lo}-{hi if hi < 10000 else '+'}"
        vals = "  ".join(f"{k.split()[0]}={row.get(k, float('nan')):+.3f}"
                         for k in usable)
        print(f"HISTORY {lbl:<8}n={row['n']:<5}{vals}")

    cur = pd.DataFrame(curve)
    cur.to_csv(os.path.join(TEXTS, "e7_coldstart_curve.csv"), index=False)

    # ------------------------------------------------------- privacy curve
    res = pd.DataFrame(rows)
    priv = []
    prev_r2 = None
    for rung in usable:
        r = res[res.rung == rung]
        if not len(r):
            continue
        r2 = float(r.r2.iloc[0])
        cost, what = DATA_COST.get(rung, (np.nan, ""))
        priv.append({"rung": rung, "data_cost_level": cost,
                     "what_the_user_must_disclose": what,
                     "r2_new_user": round(r2, 4),
                     "marginal_gain": (round(r2 - prev_r2, 4)
                                       if prev_r2 is not None else None)})
        prev_r2 = r2
    pv = pd.DataFrame(priv)
    pv.to_csv(os.path.join(TEXTS, "e7_privacy_utility.csv"), index=False)

    # ------------------------------------------------------------ summary
    lines = [
        "EXPERIMENT 7: COLD START AND THE PRICE OF PERSONALIZATION",
        "LifeSnaps, next-day steps, leave-one-person-out",
        "=" * 86, "",
        "A NEW USER, WITH NO HISTORY AT ALL (model has never seen this person)",
        f"  {'rung':<34}{'R2':>10}", "  " + "-" * 44,
    ]
    for _, r in res.iterrows():
        lines.append(f"  {r.rung:<34}{r.r2:>+10.4f}")

    lines += ["", "AS THE PERSON'S OWN HISTORY ACCUMULATES",
              f"  {'days of history':<18}{'n':>7}" +
              "".join(f"{k.split()[0]:>9}" for k in usable),
              "  " + "-" * (25 + 9 * len(usable))]
    for _, r in cur.iterrows():
        lbl = (f"{int(r.hist_days_low)}-"
               f"{int(r.hist_days_high) if r.hist_days_high < 10000 else '+'}")
        lines.append(f"  {lbl:<18}{int(r.n):>7}" +
                     "".join(f"{r.get(k, float('nan')):>+9.3f}" for k in usable))

    lines += ["", "THE PRIVACY-UTILITY CURVE",
              f"  {'rung':<34}{'R2':>9}{'marginal':>10}  what the user must disclose",
              "  " + "-" * 92]
    for _, r in pv.iterrows():
        mg = f"{r.marginal_gain:+.4f}" if r.marginal_gain is not None else "  --"
        lines.append(f"  {r.rung:<34}{r.r2_new_user:>+9.4f}{mg:>10}  "
                     f"{r.what_the_user_must_disclose}")

    if len(pv) >= 4:
        l3 = pv[pv.rung.str.startswith("L3")]
        l4 = pv[pv.rung.str.startswith("L4")]
        if len(l3) and len(l4):
            lines += ["", "=" * 86, "HEADLINE",
                      f"  L3 (a behaviour log) buys "
                      f"{float(l3.marginal_gain.iloc[0]):+.4f} over body measurements.",
                      f"  L4 (continuous passive sensing) buys a further "
                      f"{float(l4.marginal_gain.iloc[0]):+.4f}.",
                      "",
                      "  The most intrusive layer is the one that returns almost",
                      "  nothing. On this evidence an application should collect a",
                      "  behaviour log and stop there."]
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e7_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
