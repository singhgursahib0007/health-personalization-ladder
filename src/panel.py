"""
panel.py
========
Shared construction of the per-person-per-day panel and the ladder feature sets.
Imported by experiments 5, 6 and 7 so that all three use identical features and
identical causality rules. Keeping this in one place is deliberate: the easiest
way to accidentally leak information in a longitudinal study is to build the
features twice and get it right only once.

CAUSALITY RULE, enforced everywhere in this file
------------------------------------------------
Every feature used to predict day t is computed from days strictly BEFORE t.
Rolling statistics are shifted by one day before aggregation. Person-level
summaries (the person's mean, their standard deviation, their day-of-week
profile) are computed on an expanding window of that person's own past, never on
the whole series, because at deployment time the future does not exist.

There is one exception, declared: age, sex and BMI are treated as fixed and known
at onboarding, which is how an application would actually hold them.
"""

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
SEED = 42


# --------------------------------------------------------------------- loading
def load_lifesnaps():
    f = glob.glob(os.path.join(DATA, "lifesnaps", "**",
                               "daily_fitbit_sema_df_unprocessed.csv"),
                  recursive=True)[0]
    df = pd.read_csv(f, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "id"])
    df = df.rename(columns={"id": "person"})
    return df.sort_values(["person", "date"]).reset_index(drop=True)


def load_mobius():
    f = glob.glob(os.path.join(DATA, "fitbit_mobius", "**",
                               "dailyActivity_merged.csv"), recursive=True)[0]
    df = pd.read_csv(f)
    df["date"] = pd.to_datetime(df["ActivityDate"], errors="coerce", format="mixed")
    df = df.rename(columns={
        "Id": "person", "TotalSteps": "steps", "Calories": "calories",
        "VeryActiveMinutes": "very_active_minutes",
        "FairlyActiveMinutes": "moderately_active_minutes",
        "LightlyActiveMinutes": "lightly_active_minutes",
        "SedentaryMinutes": "sedentary_minutes"})
    return df.dropna(subset=["date", "person"]).sort_values(
        ["person", "date"]).reset_index(drop=True)


def filter_nonwear(df, source):
    """
    Remove non-wear days. The convention differs by dataset, which is why this
    takes a `source` argument rather than applying one rule everywhere.

      LifeSnaps : non-wear appears as a MISSING step count, usually alongside
                  exactly 1440 sedentary minutes.
      Fitbit    : non-wear appears as a LITERAL ZERO step count, again with
                  1440 sedentary minutes.

    Treating either convention as a real zero manufactures sedentary behaviour
    that never happened.
    """
    sed = df.get("sedentary_minutes")
    full_sed = sed.fillna(0) >= 1440 if sed is not None else pd.Series(False, index=df.index)
    if source == "lifesnaps":
        drop = df.steps.isna() | full_sed
    else:
        drop = df.steps.isna() | (df.steps == 0) | full_sed
    return df[~drop].copy()


# -------------------------------------------------------------------- features
def add_causal_features(df, target="steps"):
    """
    Build history features for each person using only their own past.

    Every rolling window is shifted by one day first, so the value on day t
    never enters a feature used to predict day t.
    """
    d = df.sort_values(["person", "date"]).copy()
    g = d.groupby("person")[target]
    prev = g.shift(1)

    d["hist_lag1"] = prev
    d["hist_lag2"] = g.shift(2)
    d["hist_lag7"] = g.shift(7)
    d["hist_mean3"] = prev.groupby(d.person).transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    d["hist_mean7"] = prev.groupby(d.person).transform(
        lambda s: s.rolling(7, min_periods=2).mean())
    d["hist_mean28"] = prev.groupby(d.person).transform(
        lambda s: s.rolling(28, min_periods=3).mean())
    d["hist_std7"] = prev.groupby(d.person).transform(
        lambda s: s.rolling(7, min_periods=3).std())
    # expanding mean: everything this person has done so far
    d["hist_expmean"] = prev.groupby(d.person).transform(
        lambda s: s.expanding(min_periods=1).mean())
    d["hist_expstd"] = prev.groupby(d.person).transform(
        lambda s: s.expanding(min_periods=3).std())
    # short-term trend: is this person rising or falling relative to baseline
    d["hist_trend"] = d["hist_mean3"] - d["hist_mean28"]
    # how many days of history the app actually has for this person
    d["hist_n"] = g.transform(lambda s: np.arange(len(s)))

    # day-of-week deviation, learned from this person's own past only
    d["dow"] = d.date.dt.dayofweek
    d["is_weekend"] = (d.dow >= 5).astype(float)
    dev = (prev - d["hist_expmean"])
    d["hist_dow_dev"] = (
        dev.groupby([d.person, d.dow])
           .transform(lambda s: s.expanding(min_periods=1).mean().shift(1)))
    return d


def add_context_features(d):
    """
    L4 context: yesterday's physiology and yesterday's reported mood/place.
    Shifted by one day, because at the moment a recommendation is issued for
    today, today's outcome has not happened yet.
    """
    phys = ["resting_hr", "rmssd", "sleep_efficiency", "minutesAsleep",
            "spo2", "nightly_temperature", "stress_score"]
    mood = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD",
            "TENSE/ANXIOUS", "TIRED"]
    place = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE", "OTHER",
             "OUTDOORS", "TRANSIT", "WORK/SCHOOL"]
    made = []
    for c in phys + mood + place:
        if c in d.columns:
            name = "ctx_" + c.replace("/", "_")
            d[name] = d.groupby("person")[c].shift(1)
            made.append(name)
    return d, made


# ------------------------------------------------------------------ the ladder
def ladder_specs(context_cols, has_demo=True):
    """
    The five rungs, as feature lists.

    L0 is deliberately empty: it is the population constant, which is what the
    book prescribes for activity, sleep and water.
    """
    L1 = ["age", "sex_male"] if has_demo else []
    L2 = L1 + (["bmi"] if has_demo else [])
    L3 = L2 + ["hist_lag1", "hist_lag2", "hist_lag7", "hist_mean3", "hist_mean7",
               "hist_mean28", "hist_std7", "hist_expmean", "hist_expstd",
               "hist_trend", "hist_dow_dev", "dow", "is_weekend"]
    L4 = L3 + context_cols
    return {
        "L0 population constant": [],
        "L1 demographics (the book)": L1,
        "L2 + body measurements": L2,
        "L3 + own behaviour history": L3,
        "L4 + yesterday's context": L4,
    }


def build(source="lifesnaps", target="steps"):
    """Return (panel, ladder_specs) ready for modelling."""
    if source == "lifesnaps":
        raw = load_lifesnaps()
        d = filter_nonwear(raw, "lifesnaps")
        d["sex_male"] = (d.get("gender").astype(str).str.upper().str[0] == "M").astype(float)
        d["age"] = pd.to_numeric(d.get("age"), errors="coerce")
        d["bmi"] = pd.to_numeric(d.get("bmi"), errors="coerce")
        has_demo = True
    else:
        raw = load_mobius()
        d = filter_nonwear(raw, "mobius")
        # this dataset carries no demographics at all, which is itself
        # informative: it is the situation an app is in on day one
        d["sex_male"] = np.nan
        d["age"] = np.nan
        d["bmi"] = np.nan
        has_demo = False

    d = d[d[target].notna()].copy()
    d = add_causal_features(d, target=target)
    ctx = []
    if source == "lifesnaps":
        d, ctx = add_context_features(d)

    specs = ladder_specs(ctx, has_demo=has_demo)
    if not has_demo:
        specs.pop("L1 demographics (the book)", None)
        specs.pop("L2 + body measurements", None)
    return d, specs


def forward_chain_folds(d, n_folds=5, min_train_frac=0.4):
    """
    Time-ordered folds over the pooled calendar.

    Every fold trains on the past and tests on the future, which is the only
    protocol that matches deployment. A random split would let the model see a
    person's later days while predicting their earlier ones.
    """
    dates = np.sort(d.date.unique())
    start = int(len(dates) * min_train_frac)
    cuts = np.linspace(start, len(dates) - 1, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        tr_end, te_end = dates[cuts[i]], dates[cuts[i + 1]]
        tr = d.date <= tr_end
        te = (d.date > tr_end) & (d.date <= te_end)
        if tr.sum() > 200 and te.sum() > 50:
            folds.append((tr.values, te.values))
    return folds
