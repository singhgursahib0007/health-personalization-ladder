"""
acquire_long_04_finalize.py
===========================
Finish the longitudinal recon the earlier pass started but did not complete
(recon/inspect_long/ contained only aw_fb.txt).

For every dataset in the longitudinal list this script builds the best available
PER-PERSON-PER-DAY panel, then MEASURES (never assumes):

  * n distinct persons
  * rows per person: min / median / max
  * distinct DAYS per person and calendar span per person, plus overall
    first/last date
  * the exact column list of the source frame, grouped by hand-written regex
    into identifiers / behaviour / physiology / self-report / outcome / other
  * missing-data pattern.  Specifically, for each person we expand the full
    calendar between their first and last observed day and count
        - days ABSENT entirely (no row at all)
        - days PRESENT but with the primary behaviour metric == 0
    because a zero-step day that is really a non-wear day is the single most
    common trap in wearable data.
  * duplicate rows, distinct rows, and duplicate (person, day) keys
  * fabrication screen via multidataset/audit_harness.audit_frame + verdict,
    plus hand-rolled synthetic tells (near-uniform categoricals, zero
    missingness, decimal-place regularity, digit-level roundness, and the
    inter-day autocorrelation of the primary behaviour series -- a real human
    behaviour series has positive lag-1 autocorrelation and week-of-day
    structure; an i.i.d. random generator has neither).

Outputs:
  recon/long_final.json         machine-readable, everything measured
  recon/long_final.txt          human-readable dump
"""
import glob
import io
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/macbook/Documents/MyProjects/Greg_Research/multidataset")
from audit_harness import audit_frame, verdict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
RECON = os.path.join(ROOT, "recon")

# ----------------------------------------------------------------- grouping --
GROUPS = [
    ("identifier", r"^(id|uid|user|user_id|number|subject|participant|patient|"
                   r"athlete|__subject|person|record|logid|unnamed|x1|index)$|"
                   r"(_id$|^id_)"),
    ("time",       r"(date|day|timestamp|time_of_day|week|month|creationdate|"
                   r"startdate|enddate)"),
    ("behaviour",  r"(step|distance|activ|workout|exercis|minute|sedentar|"
                   r"floor|cadence|pace|speed|elev|session|screen|read|"
                   r"journal|mindful|sleep_hours|sleep_duration|minutesasleep|"
                   r"minutestofallasleep|minutesawake|bedtime|strain|"
                   r"logged|intensit|met|conversation|dark|lock|charge|"
                   r"expense|calorie|energy|lap|split|type|sport|gear|"
                   r"equip|title|name|zone)"),
    ("physiology", r"(heart|hr\b|hrv|bpm|rmssd|spo2|breath|respirat|temperat|"
                   r"vo2|weight|bmi|height|fat|scl|efficiency|recovery|"
                   r"resting|nremhr|rem|deep|light_sleep|wake|oxygen)"),
    ("selfreport", r"(mood|stress|panas|stai|breq|ttm|personality|survey|sema|"
                   r"phq|score|rating|responsiveness|perceived|bprs|feel|"
                   r"badge|note|description|comment)"),
    ("outcome",    r"(class|label|target|condition|group|diagnos|adher|"
                   r"completed|schtype|migraine)"),
    ("demographic", r"^(age|gender|sex|bmi|fitness_level)$"),
]


def group_columns(cols):
    out = {}
    for c in cols:
        s = str(c).strip().lower()
        g = "other"
        for name, pat in GROUPS:
            if re.search(pat, s):
                g = name
                break
        out.setdefault(g, []).append(str(c))
    return out


# ------------------------------------------------------------ measurements --
def day_structure(df, pcol, dcol, behcol):
    """Everything about the per-person-per-day grid, measured."""
    t = pd.DataFrame({
        "p": df[pcol].astype(str),
        "d": pd.to_datetime(df[dcol], errors="coerce").dt.normalize(),
    })
    if behcol is not None and behcol in df.columns:
        t["b"] = pd.to_numeric(df[behcol], errors="coerce")
    else:
        t["b"] = np.nan
    n_unparseable = int(t["d"].isna().sum())
    t = t.dropna(subset=["d"])
    if not len(t):
        return {"error": "no parseable dates"}

    g = t.groupby("p")
    rows = g.size()
    days = g["d"].nunique()
    dmin, dmax = g["d"].min(), g["d"].max()
    span = (dmax - dmin).dt.days + 1          # inclusive calendar length

    # absent days: expand each person's own calendar
    absent = (span - days).clip(lower=0)

    # present-but-zero days (behaviour metric), on the DAY level
    daily = t.groupby(["p", "d"])["b"].sum(min_count=1)
    present_days = int(daily.shape[0])
    zero_days = int((daily == 0).sum())
    nan_days = int(daily.isna().sum())
    total_absent = int(absent.sum())
    total_expected = int(span.sum())

    # dup (person, day) keys
    dup_keys = int(len(t) - t.drop_duplicates(subset=["p", "d"]).shape[0])

    # lag-1 autocorrelation and day-of-week F-ratio of the behaviour series,
    # computed per person on the reindexed daily series, then averaged.
    acs, dows = [], []
    for p, s in daily.groupby(level=0):
        s = s.droplevel(0).sort_index()
        if s.notna().sum() < 14:
            continue
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        v = full.astype(float)
        try:
            a = v.autocorr(lag=1)
            if np.isfinite(a):
                acs.append(float(a))
        except Exception:
            pass
        try:
            dd = pd.DataFrame({"v": v.values, "w": full.index.dayofweek}).dropna()
            if dd["w"].nunique() >= 7 and dd["v"].std() > 0:
                gm = dd["v"].mean()
                between = dd.groupby("w")["v"].apply(
                    lambda x: len(x) * (x.mean() - gm) ** 2).sum() / 6
                within = dd.groupby("w")["v"].apply(
                    lambda x: ((x - x.mean()) ** 2).sum()).sum() / max(
                        len(dd) - 7, 1)
                if within > 0:
                    dows.append(float(between / within))
        except Exception:
            pass

    return {
        "n_persons": int(t["p"].nunique()),
        "rows_per_person": {"min": int(rows.min()), "median": float(rows.median()),
                            "mean": round(float(rows.mean()), 1),
                            "max": int(rows.max())},
        "days_per_person": {"min": int(days.min()), "median": float(days.median()),
                            "mean": round(float(days.mean()), 1),
                            "max": int(days.max())},
        "span_days_per_person": {"min": int(span.min()),
                                 "median": float(span.median()),
                                 "max": int(span.max())},
        "overall_first_date": str(t["d"].min().date()),
        "overall_last_date": str(t["d"].max().date()),
        "overall_span_days": int((t["d"].max() - t["d"].min()).days) + 1,
        "unparseable_dates": n_unparseable,
        "grid_expected_person_days": total_expected,
        "grid_present_person_days": present_days,
        "grid_absent_person_days": total_absent,
        "grid_absent_pct": round(100 * total_absent / max(total_expected, 1), 2),
        "present_days_behaviour_zero": zero_days,
        "present_days_behaviour_zero_pct": round(
            100 * zero_days / max(present_days, 1), 2),
        "present_days_behaviour_nan": nan_days,
        "dup_person_day_keys": dup_keys,
        "behaviour_col_used": behcol,
        "mean_lag1_autocorr": round(float(np.mean(acs)), 4) if acs else None,
        "n_persons_autocorr": len(acs),
        "mean_dayofweek_F": round(float(np.mean(dows)), 3) if dows else None,
    }


def tells(df):
    n = len(df)
    num = df.select_dtypes(include=[np.number])
    obj = df.select_dtypes(include=["object", "category", "bool"])
    t = {
        "n_rows": int(n), "n_cols": int(df.shape[1]),
        "null_cells_pct": round(float(df.isna().sum().sum()) /
                                max(n * df.shape[1], 1) * 100, 4),
        "cols_with_zero_nulls": int((df.isna().sum() == 0).sum()),
        "exact_dup_rows": int(n - df.drop_duplicates().shape[0]),
        "distinct_rows": int(df.drop_duplicates().shape[0]),
    }
    t["compression"] = round(t["distinct_rows"] / max(n, 1), 6)
    unif = []
    for c in obj.columns:
        vc = df[c].value_counts(normalize=True)
        k = len(vc)
        if 2 <= k <= 40:
            dev = float(np.abs(vc.values - 1.0 / k).sum())
            unif.append({"col": str(c), "k": k,
                         "max_share": round(float(vc.max()), 4),
                         "L1_dev_from_uniform": round(dev, 4)})
    unif.sort(key=lambda x: x["L1_dev_from_uniform"])
    t["categorical_uniformity"] = unif[:12]
    t["n_near_uniform_cats"] = sum(1 for u in unif
                                   if u["L1_dev_from_uniform"] < 0.08)
    rnd = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 50 or s.nunique() < 3:
            continue
        dp = s.astype(str).str.split(".").str[-1].str.len()
        rnd.append({"col": str(c),
                    "pct_integer": round(float((s % 1 == 0).mean()), 4),
                    "n_unique": int(s.nunique()),
                    "modal_decimals": int(dp.mode().iloc[0]) if len(dp) else -1,
                    "pct_modal_decimals": round(float((dp == dp.mode().iloc[0]).mean()), 4)
                    if len(dp) else None,
                    "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                    "mean": round(float(s.mean()), 4),
                    "cv": round(float(s.std() / s.mean()), 4) if s.mean() else None})
    t["numeric_shape"] = rnd[:30]
    # fraction of numeric cols whose decimals are ALL exactly k places -> generator
    fixed = [r for r in rnd if r["pct_modal_decimals"] and
             r["pct_modal_decimals"] > 0.95 and r["pct_integer"] < 0.5]
    t["n_fixed_decimal_cols"] = len(fixed)
    t["fixed_decimal_cols"] = [r["col"] for r in fixed][:20]
    return t


def screen(df, name):
    try:
        rep = audit_frame(df, name=name, source="kaggle")
        return {"verdict": verdict(rep),
                "status": rep.get("status"),
                "n_rows": rep.get("n_rows"), "n_analyzed": rep.get("n_analyzed"),
                "compression": rep.get("compression"),
                "distinct_rows": rep.get("distinct_rows"),
                "n_exact_dependencies": rep.get("n_exact_dependencies"),
                "dependent_columns": rep.get("dependent_columns"),
                "max_cramers_v": rep.get("max_cramers_v"),
                "ablation": rep.get("ablation")}
    except Exception as e:
        return {"verdict": "error", "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------- per-dataset --
def p(*a):
    print(*a, file=BUF)


def load_lifesnaps():
    f = os.path.join(DATA, "lifesnaps/rais_anonymized/csv_rais_anonymized/"
                           "daily_fitbit_sema_df_unprocessed.csv")
    df = pd.read_csv(f, low_memory=False)
    return df, "id", "date", "steps", f


def load_fitbit_mobius():
    f = os.path.join(DATA, "fitbit_mobius/dailyActivity_merged.csv")
    df = pd.read_csv(f)
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="mixed",
                                        errors="coerce")
    return df, "Id", "ActivityDate", "TotalSteps", f


def load_whoop():
    f = os.path.join(DATA, "whoop100k/whoop_fitness_dataset_100k.csv")
    df = pd.read_csv(f, low_memory=False)
    return df, "user_id", "date", "activity_duration_min", f


def load_my_applewatch():
    f = os.path.join(DATA, "my_applewatch/AppleWatch - HeartRate StepCount etc "
                           "92406 rows - export20200620105726.csv")
    df = pd.read_csv(f, low_memory=False)
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df["__person"] = "single_owner"
    # daily panel: pivot the HK long format
    df["__day"] = df["startDate"].dt.tz_localize(None).dt.normalize()
    return df, "__person", "__day", None, f


def load_aw_fb():
    f = os.path.join(DATA, "aw_fb/aw_fb_data.csv")
    return pd.read_csv(f), None, None, None, f


def load_strava():
    f = os.path.join(DATA, "strava_personal/my_activites.csv")
    df = pd.read_csv(f, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]
    mon = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
           "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}
    def parse(s):
        m = re.match(r"(\d+) de (\w+) de (\d+)", str(s))
        if not m:
            return pd.NaT
        return pd.Timestamp(int(m.group(3)), mon.get(m.group(2).lower(), 1),
                            int(m.group(1)))
    df["__day"] = df["Data da atividade"].map(parse)
    df["__person"] = "single_athlete"
    dist = [c for c in df.columns if c.startswith("Dist")][0]
    df["__dist"] = pd.to_numeric(df[dist].astype(str).str.replace(",", "."),
                                 errors="coerce")
    return df, "__person", "__day", "__dist", f


def load_running_log():
    f = os.path.join(DATA, "running_log/activity_log.csv")
    df = pd.read_csv(f)
    df["__day"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["__person"] = "single_runner"
    df["__dist"] = pd.to_numeric(df["Distance"], errors="coerce")
    return df, "__person", "__day", "__dist", f


def load_habit90():
    f = os.path.join(DATA, "habit90/90_day_habit_tracker.csv")
    df = pd.read_csv(f)
    df["__day"] = pd.to_datetime(df["Date"], errors="coerce")
    df["__person"] = "single_user"
    return df, "__person", "__day", "Workout_Duration_Min", f


def load_chargehr():
    f = os.path.join(DATA, "chargehr_1yr/One_Year_of_FitBitChargeHR_Data.csv")
    df = pd.read_csv(f, dtype=str)
    df["__day"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df["__person"] = "single_user"
    for c in df.columns:
        if c in ("Date", "__day", "__person"):
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False), errors="coerce")
    return df, "__person", "__day", "Steps", f


def _concat_subjects(folder, pattern="*.csv"):
    parts = []
    for c in sorted(glob.glob(os.path.join(folder, pattern))):
        try:
            d = pd.read_csv(c)
            d["__person"] = os.path.splitext(os.path.basename(c))[0]
            d["__group"] = os.path.basename(folder)
            parts.append(d)
        except Exception:
            pass
    return parts


def load_depresjon():
    parts = []
    for sub in ("adhd", "clinical", "control", "depression", "schizophrenia"):
        parts += _concat_subjects(os.path.join(DATA, "depresjon", sub))
    df = pd.concat(parts, ignore_index=True)
    df["__day"] = pd.to_datetime(df["date"], errors="coerce")
    return df, "__person", "__day", "activity", os.path.join(DATA, "depresjon")


def load_psykose():
    parts = _concat_subjects(os.path.join(DATA, "psykose", "patient"))
    parts += _concat_subjects(os.path.join(DATA, "psykose", "control"))
    df = pd.concat(parts, ignore_index=True)
    df["__day"] = pd.to_datetime(df["date"], errors="coerce")
    return df, "__person", "__day", "activity", os.path.join(DATA, "psykose")


def load_studentlife():
    """sensing/activity/*.csv: unix timestamp + activity inference code
    (0=stationary,1=walking,2=running,3=unknown). Collapse to per-user-per-day
    counts of non-stationary samples."""
    parts = []
    for c in sorted(glob.glob(os.path.join(
            DATA, "studentlife/dataset/sensing/activity/*.csv"))):
        try:
            d = pd.read_csv(c)
            d.columns = [str(x).strip() for x in d.columns]
            d["__person"] = re.sub(r"^activity_", "",
                                   os.path.splitext(os.path.basename(c))[0])
            parts.append(d[["timestamp", "activity inference", "__person"]])
        except Exception as e:
            print("  studentlife read fail", c, e)
    df = pd.concat(parts, ignore_index=True)
    df["__day"] = pd.to_datetime(df["timestamp"], unit="s",
                                 errors="coerce").dt.normalize()
    df["__moving"] = (pd.to_numeric(df["activity inference"],
                                    errors="coerce").isin([1, 2])).astype(int)
    return df, "__person", "__day", "__moving", os.path.join(
        DATA, "studentlife/dataset/sensing/activity")


LOADERS = {
    "lifesnaps": load_lifesnaps,
    "fitbit_mobius": load_fitbit_mobius,
    "whoop100k": load_whoop,
    "my_applewatch": load_my_applewatch,
    "aw_fb": load_aw_fb,
    "strava_personal": load_strava,
    "running_log": load_running_log,
    "habit90": load_habit90,
    "chargehr_1yr": load_chargehr,
    "depresjon": load_depresjon,
    "psykose": load_psykose,
    "studentlife": load_studentlife,
}


def main():
    global BUF
    only = sys.argv[1:]
    outpath = os.path.join(RECON, "long_final.json")
    res = json.load(open(outpath)) if os.path.exists(outpath) else {}
    BUF = io.StringIO()
    names = only or list(LOADERS)
    for name in names:
        print(f"\n===== {name} =====", flush=True)
        p(f"\n{'='*100}\n{name}\n{'='*100}")
        try:
            df, pcol, dcol, bcol, src = LOADERS[name]()
        except Exception as e:
            print("  LOAD FAILED", type(e).__name__, e)
            res[name] = {"error": f"load: {type(e).__name__}: {e}"}
            continue
        r = {"source": src, "shape": list(df.shape),
             "columns": [str(c) for c in df.columns],
             "column_groups": group_columns(df.columns),
             "person_col": pcol, "date_col": str(dcol),
             "behaviour_col": bcol,
             "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
             "null_pct": {str(c): round(float(v) * 100, 3)
                          for c, v in df.isna().mean().items()},
             "n_unique": {str(c): int(df[c].nunique(dropna=True))
                          for c in df.columns}}
        p(f"source={src}\nshape={df.shape}")
        p(f"columns grouped: {json.dumps(r['column_groups'], indent=1)}")

        r["tells"] = tells(df)
        p(f"tells: {json.dumps(r['tells'], indent=1, default=str)[:4000]}")

        if pcol and dcol is not None:
            r["day"] = day_structure(df, pcol, dcol, bcol)
            p(f"day structure: {json.dumps(r['day'], indent=1, default=str)}")
        else:
            r["day"] = {"error": "no person and/or date column"}

        # screen a bounded frame
        sdf = df if len(df) <= 200_000 else df.sample(200_000, random_state=0)
        r["screen"] = screen(sdf, name)
        p(f"screen: {json.dumps(r['screen'], indent=1, default=str)[:3000]}")

        res[name] = r
        json.dump(res, open(outpath, "w"), indent=1, default=str)
        d = r.get("day", {})
        print(f"  shape={df.shape} persons={d.get('n_persons')} "
              f"days/person med={(d.get('days_per_person') or {}).get('median')} "
              f"absent%={d.get('grid_absent_pct')} "
              f"zero%={d.get('present_days_behaviour_zero_pct')} "
              f"verdict={r['screen'].get('verdict')}", flush=True)

    with open(os.path.join(RECON, "long_final.txt"), "a") as f:
        f.write(BUF.getvalue())
    print("\nWROTE", outpath)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- deepscreen --
def autocorr_profile(df, pcol, dcol, cols, minlen=30):
    """Per-person lag-1/lag-7 autocorrelation of each named daily series.
    Real behavioural series are strongly self-correlated day to day; an i.i.d.
    row generator produces ~0. Run identically on every dataset so the numbers
    are comparable."""
    out = {}
    t = df[[pcol, dcol] + [c for c in cols if c in df.columns]].copy()
    t[dcol] = pd.to_datetime(t[dcol], errors="coerce")
    t = t.dropna(subset=[dcol])
    for c in cols:
        if c not in t.columns:
            continue
        a1, a7, wsd = [], [], []
        for pid, sub in t.groupby(pcol):
            s = sub.groupby(sub[dcol].dt.normalize())[c].mean()
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() < minlen:
                continue
            full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
            try:
                v1, v7 = full.autocorr(1), full.autocorr(7)
                if np.isfinite(v1):
                    a1.append(float(v1))
                if np.isfinite(v7):
                    a7.append(float(v7))
                wsd.append(float(full.std()))
            except Exception:
                pass
        if a1:
            out[c] = {"n_persons": len(a1),
                      "mean_lag1": round(float(np.mean(a1)), 4),
                      "median_lag1": round(float(np.median(a1)), 4),
                      "mean_lag7": round(float(np.mean(a7)), 4) if a7 else None,
                      "mean_within_person_sd": round(float(np.mean(wsd)), 4)}
    return out


def deepscreen():
    """Side-by-side behavioural-realism comparison across the datasets that have
    a real per-person-per-day grid."""
    out = {}
    specs = {
        "whoop100k": (load_whoop, ["day_strain", "sleep_hours", "recovery_score",
                                   "hrv", "resting_heart_rate", "calories_burned",
                                   "activity_duration_min"]),
        "lifesnaps": (load_lifesnaps, ["steps", "calories", "resting_hr",
                                       "minutesAsleep", "very_active_minutes",
                                       "sedentary_minutes"]),
        "fitbit_mobius": (load_fitbit_mobius, ["TotalSteps", "Calories",
                                               "VeryActiveMinutes",
                                               "SedentaryMinutes"]),
        "psykose": (load_psykose, ["activity"]),
        "depresjon": (load_depresjon, ["activity"]),
        "studentlife": (load_studentlife, ["__moving"]),
    }
    for name, (loader, cols) in specs.items():
        df, pcol, dcol, _, _ = loader()
        out[name] = autocorr_profile(df, pcol, dcol, cols)
        print(name, json.dumps(out[name]), flush=True)
    # whoop-specific: are per-person demographics constant, and are the
    # "baseline" columns literally derivable from the person?
    df, _, _, _, _ = load_whoop()
    g = df.groupby("user_id")
    const = {c: int((g[c].nunique() == 1).sum()) for c in df.columns
             if c != "user_id"}
    out["whoop_constant_per_person"] = {
        "n_persons": int(df["user_id"].nunique()),
        "cols_constant_within_every_person":
            sorted([c for c, v in const.items() if v == df["user_id"].nunique()])}
    # exact grid check: is every (user, day) present with no gaps?
    d = pd.to_datetime(df["date"])
    per = df.assign(d=d).groupby("user_id")["d"].agg(["min", "max", "nunique"])
    per["expected"] = (per["max"] - per["min"]).dt.days + 1
    out["whoop_grid"] = {
        "persons_with_perfect_contiguous_calendar":
            int((per["expected"] == per["nunique"]).sum()),
        "n_persons": int(len(per)),
        "all_start_same_day": bool(per["min"].nunique() == 1),
        "distinct_start_dates": int(per["min"].nunique()),
        "distinct_end_dates": int(per["max"].nunique())}
    # digit-level: last-decimal distribution of a "measured" physiological col
    for c in ("hrv", "recovery_score", "sleep_hours", "weight_kg"):
        s = df[c].dropna()
        last = (np.round(s * 10).astype(np.int64) % 10).value_counts(normalize=True)
        out.setdefault("whoop_last_digit", {})[c] = {
            str(k): round(float(v), 4) for k, v in last.sort_index().items()}
    json.dump(out, open(os.path.join(RECON, "long_deepscreen.json"), "w"),
              indent=1, default=str)
    print("\nWROTE recon/long_deepscreen.json")
    return out


# ----------------------------------------------------------------- non-wear --
def nonwear():
    """Quantify the zero-step / non-wear trap explicitly, per dataset.

    A Fitbit day that reports 0 steps AND 1440 sedentary minutes is the device
    reporting a full day of nothing, i.e. almost certainly non-wear, not a
    genuinely sedentary human. A day with total wear minutes far below 1440 is
    partial wear. Both must be excluded or imputed before any next-day model,
    otherwise the model learns the charging schedule."""
    out = {}

    # ---- fitbit_mobius
    df, _, _, _, _ = load_fitbit_mobius()
    act = ["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes",
           "SedentaryMinutes"]
    df["wear_min"] = df[act].sum(axis=1)
    z = df["TotalSteps"] == 0
    out["fitbit_mobius"] = {
        "n_person_days": int(len(df)),
        "zero_step_days": int(z.sum()),
        "zero_step_AND_sedentary_1440": int((z & (df["SedentaryMinutes"] >= 1440)).sum()),
        "zero_step_but_sedentary_lt_1440": int((z & (df["SedentaryMinutes"] < 1440)).sum()),
        "days_wear_min_ge_1440": int((df["wear_min"] >= 1440).sum()),
        "days_wear_min_lt_600": int((df["wear_min"] < 600).sum()),
        "median_wear_min": float(df["wear_min"].median()),
        "days_steps_lt_1000_nonzero": int(((df["TotalSteps"] > 0) &
                                           (df["TotalSteps"] < 1000)).sum()),
        "calories_on_zero_step_days_median":
            float(df.loc[z, "Calories"].median()) if z.any() else None,
        "calories_on_normal_days_median": float(df.loc[~z, "Calories"].median()),
        "days_calories_zero": int((df["Calories"] == 0).sum()),
        "usable_days_after_nonwear_filter":
            int(((df["TotalSteps"] > 0) & (df["wear_min"] >= 600)).sum()),
    }

    # ---- lifesnaps: column-level coverage on the daily frame
    ls, pcol, dcol, _, _ = load_lifesnaps()
    cov = (1 - ls.isna().mean()).sort_values(ascending=False)
    out["lifesnaps"] = {
        "n_person_days": int(len(ls)),
        "steps_missing_days": int(ls["steps"].isna().sum()),
        "steps_zero_days": int((ls["steps"] == 0).sum()),
        "steps_present_days": int(ls["steps"].notna().sum()),
        "days_steps_lt_1000_nonzero": int(((ls["steps"] > 0) &
                                           (ls["steps"] < 1000)).sum()),
        "sedentary_1440_days": int((ls["sedentary_minutes"] >= 1440).sum()),
        "column_coverage_pct": {str(k): round(float(v) * 100, 1)
                                for k, v in cov.items()},
        "n_cols_ge_80pct_coverage": int((cov >= 0.8).sum()),
        "n_cols_ge_50pct_coverage": int((cov >= 0.5).sum()),
        "n_cols_lt_20pct_coverage": int((cov < 0.2).sum()),
        # complete-case daily panel on the core behaviour block
        "complete_core_days": int(ls[["steps", "calories", "sedentary_minutes",
                                      "very_active_minutes"]].notna().all(axis=1).sum()),
        "complete_core_plus_sleep_days":
            int(ls[["steps", "calories", "minutesAsleep",
                    "sleep_efficiency"]].notna().all(axis=1).sum()),
        "persons_with_ge_60_complete_core_days": int(
            ls[ls[["steps", "calories", "sedentary_minutes"]].notna().all(axis=1)]
            .groupby("id").size().ge(60).sum()),
    }

    # ---- my_applewatch: pivot the HK long export to a daily panel
    aw, _, _, _, _ = load_my_applewatch()
    aw["value_n"] = pd.to_numeric(aw["value"], errors="coerce")
    piv = aw.pivot_table(index="__day", columns="type", values="value_n",
                         aggfunc="sum")
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
    step_col = [c for c in piv.columns if "StepCount" in str(c)]
    out["my_applewatch"] = {
        "record_types": {str(k): int(v) for k, v in
                         aw["type"].value_counts().items()},
        "daily_panel_days": int(len(piv)),
        "calendar_span_days": int((piv.index.max() - piv.index.min()).days) + 1,
        "days_with_any_record": int(aw["__day"].nunique()),
        "days_absent_entirely": int(len(piv) - aw["__day"].nunique()),
        "step_col": step_col,
    }
    if step_col:
        s = piv[step_col[0]]
        out["my_applewatch"].update({
            "step_days_present": int(s.notna().sum()),
            "step_days_absent": int(s.isna().sum()),
            "step_days_zero": int((s == 0).sum()),
            "step_median": float(s.median()),
            "step_lag1_autocorr": round(float(s.autocorr(1)), 4)})

    # ---- chargehr
    ch, _, _, _, _ = load_chargehr()
    out["chargehr_1yr"] = {
        "n_days": int(len(ch)),
        "zero_step_days": int((ch["Steps"] == 0).sum()),
        "days_sitting_1440": int((ch["Minutes_sitting"] >= 1440).sum()),
        "median_steps": float(ch["Steps"].median()),
        "duplicate_date_rows": int(ch["__day"].duplicated().sum()),
        "steps_lag1_autocorr": round(float(
            ch.groupby("__day")["Steps"].mean().sort_index()
              .asfreq("D").autocorr(1)), 4),
    }

    # ---- psykose / depresjon daily actigraphy
    for nm, loader in (("psykose", load_psykose), ("depresjon", load_depresjon)):
        d, p_, d_, b_, _ = loader()
        daily = d.groupby([p_, d_])["activity"].agg(["sum", "count", "mean"])
        daily["nonwear_like"] = (daily["sum"] == 0) | (daily["count"] < 1000)
        first_last = daily.reset_index().groupby(p_)[d_].agg(["min", "max"])
        out[nm] = {
            "n_person_days": int(len(daily)),
            "days_with_full_1440_minutes": int((daily["count"] == 1440).sum()),
            "days_with_partial_minutes": int((daily["count"] != 1440).sum()),
            "median_minutes_per_day": float(daily["count"].median()),
            "zero_activity_days": int((daily["sum"] == 0).sum()),
            "days_flagged_nonwear_like": int(daily["nonwear_like"].sum()),
            "first_date": str(first_last["min"].min().date()),
            "last_date": str(first_last["max"].max().date()),
        }

    json.dump(out, open(os.path.join(RECON, "long_nonwear.json"), "w"),
              indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk not in ("column_coverage_pct", "record_types")}
                      for k, v in out.items()}, indent=1, default=str))
    print("\nWROTE recon/long_nonwear.json")
    return out


# ------------------------------------------------- lifesnaps / studentlife --
def targets():
    """Measure the availability of explicit ADHERENCE targets, i.e. a per-person
    goal or a self-reported behaviour that a recommendation could be scored
    against."""
    out = {}
    ls, _, _, _, _ = load_lifesnaps()
    ls["sg"] = pd.to_numeric(ls["step_goal"], errors="coerce")
    sub = ls[ls["sg"].notna() & ls["steps"].notna()]
    g = sub.groupby("id").size()
    out["lifesnaps_step_goal"] = {
        "person_days_with_goal_and_steps": int(len(sub)),
        "persons": int(sub["id"].nunique()),
        "days_per_person": {"min": int(g.min()), "median": float(g.median()),
                            "max": int(g.max())},
        "persons_ge_30_days": int((g >= 30).sum()),
        "persons_ge_60_days": int((g >= 60).sum()),
        "goal_hit_rate": round(float((sub["steps"] >= sub["sg"]).mean()), 4),
        "persons_whose_goal_changes": int(
            (ls.groupby("id")["step_goal"].nunique() > 1).sum()),
        "step_goal_label_counts": ls["step_goal_label"].value_counts().to_dict(),
    }
    sema = ["ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED", "SAD",
            "TENSE/ANXIOUS", "TIRED"]
    ctx = ["ENTERTAINMENT", "GYM", "HOME", "HOME_OFFICE", "OTHER", "OUTDOORS",
           "TRANSIT", "WORK/SCHOOL"]
    s2 = ls[ls[sema].notna().any(axis=1)]
    g2 = s2.groupby("id").size()
    out["lifesnaps_sema"] = {
        "person_days": int(len(s2)), "persons": int(s2["id"].nunique()),
        "median_days_per_person": float(g2.median()),
        "persons_ge_30_days": int((g2 >= 30).sum()),
        "mood_cols": sema, "context_cols": ctx,
        "gym_days": int((ls["GYM"] == 1).sum()),
    }

    # studentlife Exercise EMA
    rows = []
    for f in sorted(glob.glob(os.path.join(
            DATA, "studentlife/dataset/EMA/response/Exercise/*.json"))):
        uid = re.sub(r"^Exercise_", "", os.path.splitext(os.path.basename(f))[0])
        try:
            for rec in json.load(open(f)):
                rows.append({"uid": uid,
                             "t": pd.to_datetime(rec.get("resp_time"), unit="s",
                                                 errors="coerce"),
                             "exercise": rec.get("exercise"),
                             "walk": rec.get("walk"),
                             "have": rec.get("have")})
        except Exception:
            pass
    e = pd.DataFrame(rows).dropna(subset=["t"])
    e["day"] = e["t"].dt.normalize()
    gg = e.groupby("uid")["day"].nunique()
    out["studentlife_exercise_ema"] = {
        "n_responses": int(len(e)), "n_users": int(e["uid"].nunique()),
        "distinct_user_days": int(e.drop_duplicates(["uid", "day"]).shape[0]),
        "days_per_person": {"min": int(gg.min()), "median": float(gg.median()),
                            "max": int(gg.max())},
        "persons_ge_20_days": int((gg >= 20).sum()),
        "exercise_value_counts": e["exercise"].value_counts().to_dict(),
    }
    json.dump(out, open(os.path.join(RECON, "long_targets.json"), "w"),
              indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return out


# ------------------------------------------------------- L0..L3 feasibility --
def ladder_probe():
    """Does the data actually support the personalization ladder?
    Forward-chaining split per person (first 70% of a person's days train, last
    30% test). Compare, on next-day steps:
      L0 population mean (constant)
      L1/L2 demographic/anthropometric group mean  (where available)
      L3a person mean (own history, level only)
      L3b person mean + lag1 + lag7 + day-of-week ridge (own history, dynamic)
    Reported as out-of-sample R^2 and MAE. This is the experiment the paper
    needs; if it cannot be run, the dataset is not a backbone."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score

    def probe(daily, name, group_key=None):
        # daily: DataFrame with columns person, day, y  (already non-wear filtered)
        daily = daily.sort_values(["person", "day"])
        feats, ytr_all, yte_all = [], [], []
        rows_tr, rows_te = [], []
        for pid, s in daily.groupby("person"):
            s = s.set_index("day")["y"].asfreq("D")
            if s.notna().sum() < 30:
                continue
            df = pd.DataFrame({"y": s})
            df["lag1"] = s.shift(1)
            df["lag7"] = s.shift(7)
            df["roll7"] = s.shift(1).rolling(7, min_periods=3).mean()
            df["dow"] = df.index.dayofweek
            df["person"] = pid
            df = df.dropna()
            if len(df) < 20:
                continue
            k = int(len(df) * 0.7)
            rows_tr.append(df.iloc[:k])
            rows_te.append(df.iloc[k:])
        if not rows_tr:
            return {"error": "insufficient per-person history"}
        tr = pd.concat(rows_tr)
        te = pd.concat(rows_te)
        res = {"name": name, "n_persons": int(tr["person"].nunique()),
               "n_train_days": int(len(tr)), "n_test_days": int(len(te))}

        y = te["y"].values
        # L0
        p0 = np.full(len(te), tr["y"].mean())
        # L3a person mean from train only
        pm = tr.groupby("person")["y"].mean()
        p3a = te["person"].map(pm).fillna(tr["y"].mean()).values
        # L3b ridge with own-history features, fitted pooled with person mean
        X = lambda d: np.column_stack([
            d["lag1"], d["lag7"], d["roll7"],
            d["person"].map(pm).fillna(tr["y"].mean()),
            *[(d["dow"] == i).astype(float) for i in range(7)]])
        m = Ridge(alpha=1.0).fit(X(tr), tr["y"])
        p3b = m.predict(X(te))
        # persistence
        pP = te["lag1"].values
        for lbl, pred in (("L0_population_mean", p0), ("L3a_person_mean", p3a),
                          ("L3b_person_history_ridge", p3b),
                          ("persistence_lag1", pP)):
            res[lbl] = {"r2": round(float(r2_score(y, pred)), 4),
                        "mae": round(float(mean_absolute_error(y, pred)), 1)}
        if group_key is not None and group_key in daily.columns:
            gm = tr.join(daily.set_index(["person"])[group_key].drop_duplicates(),
                         on="person")
        return res

    out = {}
    # lifesnaps steps, non-wear filtered (steps present, not a 1440-sedentary day)
    ls, _, _, _, _ = load_lifesnaps()
    ls["date"] = pd.to_datetime(ls["date"])
    keep = ls["steps"].notna() & ~(ls["sedentary_minutes"] >= 1440)
    d = ls.loc[keep, ["id", "date", "steps"]].rename(
        columns={"id": "person", "date": "day", "steps": "y"})
    out["lifesnaps_steps"] = probe(d, "lifesnaps steps (non-wear filtered)")

    # fitbit_mobius steps, non-wear filtered
    fb, _, _, _, _ = load_fitbit_mobius()
    fb["wear"] = fb[["VeryActiveMinutes", "FairlyActiveMinutes",
                     "LightlyActiveMinutes", "SedentaryMinutes"]].sum(axis=1)
    keep = (fb["TotalSteps"] > 0) & (fb["wear"] >= 600)
    d = fb.loc[keep, ["Id", "ActivityDate", "TotalSteps"]].rename(
        columns={"Id": "person", "ActivityDate": "day", "TotalSteps": "y"})
    out["fitbit_mobius_steps"] = probe(d, "fitbit_mobius steps (non-wear filtered)")

    # whoop synthetic control
    wh, _, _, _, _ = load_whoop()
    wh["date"] = pd.to_datetime(wh["date"])
    d = wh[["user_id", "date", "day_strain"]].rename(
        columns={"user_id": "person", "date": "day", "day_strain": "y"})
    out["whoop100k_day_strain"] = probe(d, "whoop100k day_strain (synthetic control)")

    # psykose / depresjon daily activity
    for nm, loader in (("psykose", load_psykose), ("depresjon", load_depresjon)):
        dd, p_, d_, _, _ = loader()
        daily = dd.groupby([p_, d_])["activity"].sum().reset_index()
        daily.columns = ["person", "day", "y"]
        daily = daily[daily["y"] > 0]
        out[nm + "_daily_activity"] = probe(daily, nm + " daily activity counts")

    # studentlife daily moving samples
    sl, p_, d_, b_, _ = load_studentlife()
    daily = sl.groupby([p_, d_])[b_].sum().reset_index()
    daily.columns = ["person", "day", "y"]
    daily = daily[daily["y"] > 0]
    out["studentlife_moving"] = probe(daily, "studentlife daily moving samples")

    json.dump(out, open(os.path.join(RECON, "long_ladder_probe.json"), "w"),
              indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    return out
