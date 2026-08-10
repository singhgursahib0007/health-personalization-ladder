"""
04_variance_decomposition.py   (EXPERIMENT 4)
=============================================
Question
--------
Every rung of the ladder below L3 assigns a person a fixed recommendation. That
design has an implicit assumption: that a person is a stable thing, so knowing
WHO they are tells you WHAT they will do.

This experiment tests that assumption directly. For each behaviour and each
physiological signal we split the total variance into two parts:

    between-person variance   how much people differ from each other
    within-person variance    how much one person differs from themselves,
                              day to day

The ratio is the intraclass correlation (ICC). It sets a hard ceiling on any
static recommender: if ICC is 0.3, then even a PERFECT static profile, one that
knew everything permanent about you, could account for at most 30 percent of the
variation in your behaviour. The other 70 percent is which day it happens to be,
and no amount of onboarding data can reach it.

We then ask what does reach it:
    day of week
    the person's own recent history (trailing mean, lag-1)
which is exactly the information that only becomes available at L3.

Data
----
LifeSnaps (71 people, median 88 days each). Non-wear days are removed first: in
this dataset non-wear appears as a MISSING step count accompanied by exactly 1440
sedentary minutes, so treating those as real zeros would manufacture sedentary
behaviour that never happened.

Outputs
-------
outputs/texts/e4_icc.csv                 ICC per variable, with CIs
outputs/texts/e4_variance_sources.csv    incremental R2 from each source
outputs/texts/e4_summary.txt             readable summary
outputs/texts/e4_nonwear_audit.json      how many days the filter removed and why
"""

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)
SEED = 42
MIN_DAYS = 20          # a person needs enough days for within-person variance
N_BOOT = 400


def load_lifesnaps():
    f = glob.glob(os.path.join(ROOT, "datasets", "lifesnaps", "**",
                               "daily_fitbit_sema_df_unprocessed.csv"),
                  recursive=True)[0]
    df = pd.read_csv(f, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "id"]).sort_values(["id", "date"])


def filter_nonwear(df):
    """
    Remove days the device was not worn.

    In LifeSnaps non-wear surfaces as a missing step count together with exactly
    1440 sedentary minutes (the whole day counted as sitting). Both conditions
    are recorded separately so the audit is transparent.
    """
    audit = {"n_rows_raw": int(len(df)),
             "n_people_raw": int(df.id.nunique())}
    miss_steps = df.steps.isna()
    full_sed = df.sedentary_minutes.fillna(0) >= 1440
    zero_steps = df.steps.fillna(-1) == 0

    audit["n_missing_steps"] = int(miss_steps.sum())
    audit["n_sedentary_1440"] = int(full_sed.sum())
    audit["n_missing_steps_and_1440"] = int((miss_steps & full_sed).sum())
    audit["n_literal_zero_steps"] = int(zero_steps.sum())

    drop = miss_steps | full_sed
    out = df[~drop].copy()
    audit["n_rows_dropped"] = int(drop.sum())
    audit["n_rows_kept"] = int(len(out))
    audit["pct_dropped"] = round(float(drop.mean() * 100), 2)
    audit["n_people_kept"] = int(out.id.nunique())
    return out, audit


# ------------------------------------------------------------------------ ICC
def icc_oneway(df, var, person="id", min_days=MIN_DAYS):
    """
    One-way random-effects ICC(1) via variance components.

    ICC = s2_between / (s2_between + s2_within)

    Computed from the ANOVA mean squares, which handles the unbalanced case
    (people contribute different numbers of days) through the standard k0
    correction rather than pretending the design is balanced.
    """
    d = df[[person, var]].dropna()
    counts = d.groupby(person)[var].size()
    keep = counts[counts >= min_days].index
    d = d[d[person].isin(keep)]
    if d[person].nunique() < 5:
        return None

    grand = d[var].mean()
    groups = d.groupby(person)[var]
    n_i = groups.size().values
    m_i = groups.mean().values
    k = len(n_i)
    N = n_i.sum()

    ss_between = np.sum(n_i * (m_i - grand) ** 2)
    ss_within = np.sum([((g - g.mean()) ** 2).sum() for _, g in groups])
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    # k0: effective group size for an unbalanced design
    k0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    s2_between = max((ms_between - ms_within) / k0, 0.0)
    s2_within = ms_within
    total = s2_between + s2_within
    icc = s2_between / total if total > 0 else np.nan

    return {"variable": var, "n_people": int(k), "n_obs": int(N),
            "median_days": float(np.median(n_i)),
            "icc": float(icc),
            "sd_between": float(np.sqrt(s2_between)),
            "sd_within": float(np.sqrt(s2_within)),
            "mean": float(grand)}


def icc_bootstrap_ci(df, var, person="id", n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap over people, which is the correct resampling unit."""
    rng = np.random.RandomState(seed)
    people = df[person].unique()
    ests = []
    for _ in range(n_boot):
        samp = rng.choice(people, len(people), replace=True)
        parts = []
        for j, p in enumerate(samp):
            g = df[df[person] == p].copy()
            g[person] = f"{p}_{j}"          # keep resampled people distinct
            parts.append(g)
        r = icc_oneway(pd.concat(parts, ignore_index=True), var, person)
        if r:
            ests.append(r["icc"])
    if len(ests) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5)))


# -------------------------------------------------------- incremental sources
def variance_sources(df, var, person="id"):
    """
    How much of the variance in `var` is recoverable from each information
    source? Each model is fit on all data and scored in sample, because the
    question here is a variance-decomposition question, not a prediction
    question. E5 does the honest out-of-sample version.

      person mean       what a perfect static profile could achieve
      + day of week     calendar structure, free to any app
      + own lag-1       yesterday's value, needs one day of history
      + own trailing 7  the person's recent baseline, needs a week
    """
    d = df[[person, "date", var]].dropna().sort_values([person, "date"]).copy()
    counts = d.groupby(person)[var].size()
    d = d[d[person].isin(counts[counts >= MIN_DAYS].index)]
    if d[person].nunique() < 5:
        return None

    y = d[var].values.astype(float)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def r2(pred):
        pred = np.asarray(pred, dtype=float)
        m = np.isfinite(pred)
        if m.sum() < 50:
            return np.nan
        return float(1 - np.sum((y[m] - pred[m]) ** 2) /
                     np.sum((y[m] - y[m].mean()) ** 2))

    out = {"variable": var, "n_obs": int(len(d)),
           "n_people": int(d[person].nunique())}

    out["r2_population_mean"] = 0.0
    out["r2_person_mean"] = r2(d.groupby(person)[var].transform("mean").values)

    dow = d.date.dt.dayofweek
    pm = d.groupby(person)[var].transform("mean")
    dow_dev = d.assign(_dev=d[var] - pm).groupby(dow)["_dev"].transform("mean")
    out["r2_person_plus_dow"] = r2((pm + dow_dev).values)

    lag1 = d.groupby(person)[var].shift(1)
    out["r2_person_plus_lag1"] = r2(np.where(lag1.notna(),
                                             0.5 * pm + 0.5 * lag1.fillna(pm),
                                             pm).astype(float))

    trail = (d.groupby(person)[var]
               .transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean()))
    out["r2_person_plus_trailing7"] = r2(np.where(trail.notna(), trail, pm))

    return out


VARIABLES = [
    ("steps", "behaviour"),
    ("very_active_minutes", "behaviour"),
    ("moderately_active_minutes", "behaviour"),
    ("sedentary_minutes", "behaviour"),
    ("calories", "behaviour"),
    ("minutesAsleep", "behaviour"),
    ("sleep_efficiency", "physiology"),
    ("resting_hr", "physiology"),
    ("rmssd", "physiology"),
    ("spo2", "physiology"),
    ("stress_score", "physiology"),
    ("nightly_temperature", "physiology"),
]


def main():
    raw = load_lifesnaps()
    df, audit = filter_nonwear(raw)
    print("NON-WEAR FILTER")
    print(json.dumps(audit, indent=1))
    with open(os.path.join(TEXTS, "e4_nonwear_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)

    icc_rows, src_rows = [], []
    for var, kind in VARIABLES:
        if var not in df.columns:
            print(f"  (skip {var}: not present)")
            continue
        r = icc_oneway(df, var)
        if r is None:
            print(f"  (skip {var}: too few people with >= {MIN_DAYS} days)")
            continue
        lo, hi = icc_bootstrap_ci(df[[ "id", var]].dropna(), var)
        r["kind"] = kind
        r["icc_ci_low"], r["icc_ci_high"] = lo, hi
        icc_rows.append(r)
        print(f"  ICC {var:<26}{r['icc']:.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"people={r['n_people']:<4}obs={r['n_obs']}")

        s = variance_sources(df, var)
        if s:
            s["kind"] = kind
            src_rows.append(s)

    icc = pd.DataFrame(icc_rows).sort_values("icc")
    icc.to_csv(os.path.join(TEXTS, "e4_icc.csv"), index=False)
    src = pd.DataFrame(src_rows)
    src.to_csv(os.path.join(TEXTS, "e4_variance_sources.csv"), index=False)

    beh = icc[icc.kind == "behaviour"]
    phy = icc[icc.kind == "physiology"]

    lines = [
        "EXPERIMENT 4: IS A PERSON A STABLE THING?",
        "LifeSnaps, one-way random-effects ICC, cluster bootstrap over people",
        "=" * 84, "",
        "NON-WEAR FILTER",
        f"  rows before                       : {audit['n_rows_raw']:,}",
        f"  missing step count                : {audit['n_missing_steps']:,}",
        f"  exactly 1440 sedentary minutes    : {audit['n_sedentary_1440']:,}",
        f"  both together                     : {audit['n_missing_steps_and_1440']:,}",
        f"  literal zero-step days            : {audit['n_literal_zero_steps']:,}",
        f"  rows removed                      : {audit['n_rows_dropped']:,} "
        f"({audit['pct_dropped']}%)",
        f"  rows kept                         : {audit['n_rows_kept']:,} "
        f"across {audit['n_people_kept']} people",
        "",
        "INTRACLASS CORRELATION  (share of variance that is BETWEEN people)",
        f"{'variable':<28}{'kind':<12}{'ICC':>7}{'95% CI':>18}"
        f"{'sd_betw':>10}{'sd_within':>11}",
        "-" * 86,
    ]
    for _, r in icc.iterrows():
        ci = (f"[{r.icc_ci_low:.3f}, {r.icc_ci_high:.3f}]"
              if np.isfinite(r.icc_ci_low) else "n/a")
        lines.append(f"{r.variable:<28}{r.kind:<12}{r.icc:>7.3f}{ci:>18}"
                     f"{r.sd_between:>10.1f}{r.sd_within:>11.1f}")

    lines += ["", "WHERE THE REST OF THE VARIANCE LIVES  (in-sample R2)",
              f"{'variable':<28}{'person':>9}{'+dow':>9}{'+lag1':>9}{'+trail7':>9}",
              "-" * 66]
    for _, r in src.iterrows():
        lines.append(f"{r.variable:<28}{r.r2_person_mean:>9.3f}"
                     f"{r.r2_person_plus_dow:>9.3f}{r.r2_person_plus_lag1:>9.3f}"
                     f"{r.r2_person_plus_trailing7:>9.3f}")

    if len(beh) and len(phy):
        lines += [
            "", "=" * 84, "HEADLINE",
            f"  median ICC, BEHAVIOUR   : {beh.icc.median():.3f}",
            f"  median ICC, PHYSIOLOGY  : {phy.icc.median():.3f}",
            "",
            "  A static profile, however rich, is bounded by the between-person",
            f"  share. For behaviour that ceiling is about "
            f"{beh.icc.median() * 100:.0f} percent of the variance;",
            f"  the remaining {100 - beh.icc.median() * 100:.0f} percent is "
            "within-person, which is to say it is",
            "  which day it is, and it is unreachable from any onboarding form.",
        ]
    txt = "\n".join(lines)
    with open(os.path.join(TEXTS, "e4_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
