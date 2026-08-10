"""
acquire_long_03_inspect.py
==========================
Full structural inspection + fabrication screen for every downloaded
longitudinal candidate.

For EVERY csv in each dataset folder it prints and saves:
  - first 20 rows
  - shape, columns, dtypes, memory
  - null counts and null % per column
  - describe() for numerics, value_counts() top-10 for objects
  - n unique persons, rows/person (min/median/max), per-person date range,
    total span
  - duplicate row count and distinct-row count

Then it runs the pre-existing fabrication screen from
multidataset/audit_harness.py (audit_frame + verdict) on the richest frame of
each dataset, and adds hand-rolled synthetic tells (uniformity of categorical
distributions, roundness of numerics, digit-distribution, exact-duplicate
compression, zero-missingness).

Outputs:
  recon/inspect_long/<short_name>.txt     full printed dump
  recon/inspect_long_summary.json         machine-readable summary
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
OUT = os.path.join(RECON, "inspect_long")
os.makedirs(OUT, exist_ok=True)

MAX_ROWS_READ = 3_000_000

PERSON_PAT = re.compile(
    r"^(id|user|users?[_ ]?id|userid|participant|participant[_ ]?id|subject|"
    r"subject[_ ]?id|person|person[_ ]?id|patient|patient[_ ]?id|member|"
    r"member[_ ]?id|athlete|athlete[_ ]?id|pid|uid|customer[_ ]?id|"
    r"individual|individual[_ ]?id|number|record[_ ]?id)$",
    re.I)
DATE_PAT = re.compile(
    r"(date|day|timestamp|time|datetime|activityday|activityhour|"
    r"activityminute|sleepday|start|logid|week|month)", re.I)


def find_person_col(df):
    """Prefer an explicit *_id name; fall back to any low-ish-cardinality
    non-date column that repeats a lot (a plausible grouping key)."""
    cands = [c for c in df.columns if PERSON_PAT.match(str(c).strip())]
    if not cands:
        cands = [c for c in df.columns
                 if re.search(r"(^|[_ ])(id|user|subject|participant|patient)",
                              str(c), re.I)
                 and not DATE_PAT.search(str(c))]
    best, best_key = None, None
    for c in cands:
        nu = df[c].nunique(dropna=True)
        if nu < 2 or nu > len(df) * 0.9:
            continue
        key = (len(df) / max(nu, 1), -nu)
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def find_date_col(df):
    cands = [c for c in df.columns if DATE_PAT.search(str(c))]
    for c in cands:
        s = df[c]
        if np.issubdtype(s.dtype, np.datetime64):
            return c
        try:
            parsed = pd.to_datetime(s.head(3000), errors="coerce",
                                    format="mixed")
            if parsed.notna().mean() > 0.8:
                return c
        except Exception:
            continue
    return None


def synth_tells(df, p):
    """Hand-rolled fabrication tells, independent of audit_harness."""
    t = {}
    n = len(df)
    num = df.select_dtypes(include=[np.number])
    obj = df.select_dtypes(include=["object", "category", "bool"])

    t["null_cells_pct"] = round(float(df.isna().sum().sum()) / (n * df.shape[1]) * 100, 4)
    t["cols_with_zero_nulls"] = int((df.isna().sum() == 0).sum())
    t["n_cols"] = int(df.shape[1])
    t["exact_dup_rows"] = int(n - df.drop_duplicates().shape[0])
    t["compression_distinct_over_n"] = round(df.drop_duplicates().shape[0] / n, 6)

    # near-uniform categoricals: max class share close to 1/k
    unif = []
    for c in obj.columns:
        vc = df[c].value_counts(normalize=True)
        k = len(vc)
        if 2 <= k <= 30:
            # chi-square-ish deviation from perfect uniformity
            dev = float(np.abs(vc.values - 1.0 / k).sum())
            unif.append((c, k, round(float(vc.max()), 4), round(dev, 4)))
    t["categorical_uniformity"] = [
        {"col": c, "k": k, "max_share": m, "L1_dev_from_uniform": d}
        for c, k, m, d in sorted(unif, key=lambda x: x[3])[:12]]
    t["n_near_uniform_cats"] = int(sum(1 for _, k, _, d in unif if d < 0.06))

    # roundness + digit behaviour of numerics
    rnd = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 50 or s.nunique() < 3:
            continue
        frac = s - np.floor(s)
        pct_int = float((frac == 0).mean())
        # decimals: synthetic generators often emit exactly 1-2 dp
        dp = s.astype(str).str.split(".").str[-1].str.len()
        rnd.append({"col": c, "pct_integer": round(pct_int, 4),
                    "n_unique": int(s.nunique()),
                    "modal_decimals": int(dp.mode().iloc[0]) if len(dp) else -1,
                    "min": float(s.min()), "max": float(s.max()),
                    "mean": round(float(s.mean()), 4),
                    "cv": round(float(s.std() / s.mean()), 4) if s.mean() else None})
    t["numeric_shape"] = rnd[:25]

    # per-person constancy of a behavioural series is a strong synthetic tell
    if p and p in df.columns:
        stds = {}
        for c in num.columns[:15]:
            try:
                g = df.groupby(p)[c].std()
                stds[c] = round(float(g.mean()), 4)
            except Exception:
                pass
        t["mean_within_person_std"] = stds
    return t


def inspect_csv(path, buf, sample_cap=MAX_ROWS_READ):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1e6
    p = lambda *a: print(*a, file=buf)
    p("\n" + "=" * 100)
    p(f"FILE: {path}   ({size_mb:.2f} MB)")
    p("=" * 100)
    try:
        df = pd.read_csv(path, low_memory=False, nrows=sample_cap)
    except Exception as e:
        p(f"  READ FAILED: {type(e).__name__}: {e}")
        return None
    if len(df) == sample_cap:
        p(f"  NOTE: truncated read at {sample_cap} rows")

    p(f"\n--- FIRST 20 ROWS ---")
    with pd.option_context("display.width", 250, "display.max_columns", 80,
                           "display.max_colwidth", 28):
        p(df.head(20).to_string())

    p(f"\n--- SHAPE / DTYPES / MEMORY ---")
    p(f"shape = {df.shape}")
    p(f"memory = {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_null": df.isna().sum(),
        "pct_null": (df.isna().mean() * 100).round(3),
        "n_unique": df.nunique(dropna=True),
    })
    with pd.option_context("display.max_rows", 300, "display.width", 200):
        p(info.to_string())

    p(f"\n--- DUPLICATES ---")
    dd = df.drop_duplicates().shape[0]
    p(f"total rows      = {len(df)}")
    p(f"distinct rows   = {dd}")
    p(f"duplicate rows  = {len(df) - dd}  ({(len(df)-dd)/len(df)*100:.3f}%)")

    num = df.select_dtypes(include=[np.number])
    if num.shape[1]:
        p(f"\n--- DESCRIBE (numeric) ---")
        with pd.option_context("display.width", 250, "display.max_columns", 80):
            p(num.describe().T.to_string())
    obj = df.select_dtypes(include=["object", "bool", "category"])
    if obj.shape[1]:
        p(f"\n--- VALUE COUNTS (object, top 10) ---")
        for c in obj.columns[:40]:
            p(f"\n[{c}]  n_unique={df[c].nunique()}")
            p(df[c].value_counts(dropna=False).head(10).to_string())

    # person / date structure
    pc = find_person_col(df)
    dc = find_date_col(df)
    p(f"\n--- LONGITUDINAL STRUCTURE ---")
    p(f"person col detected = {pc!r}")
    p(f"date   col detected = {dc!r}")
    res = {"file": name, "path": path, "size_mb": round(size_mb, 3),
           "shape": list(df.shape), "person_col": pc, "date_col": dc,
           "dup_rows": int(len(df) - dd), "distinct_rows": int(dd),
           "columns": [str(c) for c in df.columns]}
    if pc:
        g = df.groupby(pc).size()
        res["n_persons"] = int(df[pc].nunique())
        res["rows_per_person"] = {"min": int(g.min()),
                                  "median": float(g.median()),
                                  "mean": round(float(g.mean()), 2),
                                  "max": int(g.max())}
        p(f"n unique persons = {res['n_persons']}")
        p(f"rows per person  : min={g.min()} median={g.median()} "
          f"mean={g.mean():.1f} max={g.max()}")
        p(f"rows-per-person distribution:\n{g.describe().to_string()}")
    if dc:
        d = pd.to_datetime(df[dc], errors="coerce", format="mixed")
        ok = d.dropna()
        if len(ok):
            res["date_min"] = str(ok.min())
            res["date_max"] = str(ok.max())
            res["span_days"] = int((ok.max() - ok.min()).days)
            p(f"date range = {ok.min()}  ->  {ok.max()}   "
              f"span={res['span_days']} days")
            p(f"unparseable dates = {d.isna().sum()}")
            if pc:
                tmp = pd.DataFrame({"p": df[pc], "d": d}).dropna()
                per = tmp.groupby("p")["d"].agg(["min", "max", "nunique"])
                per["span_days"] = (per["max"] - per["min"]).dt.days
                res["days_per_person"] = {
                    "min": int(per["nunique"].min()),
                    "median": float(per["nunique"].median()),
                    "max": int(per["nunique"].max())}
                res["span_days_per_person"] = {
                    "min": int(per["span_days"].min()),
                    "median": float(per["span_days"].median()),
                    "max": int(per["span_days"].max())}
                p(f"\nDISTINCT DAYS per person : min={per['nunique'].min()} "
                  f"median={per['nunique'].median()} max={per['nunique'].max()}")
                p(f"CALENDAR SPAN per person : min={per['span_days'].min()} "
                  f"median={per['span_days'].median()} max={per['span_days'].max()}")
                # missing-day detection
                per["expected"] = per["span_days"] + 1
                per["coverage"] = per["nunique"] / per["expected"].clip(lower=1)
                res["day_coverage"] = {
                    "median": round(float(per["coverage"].median()), 4),
                    "min": round(float(per["coverage"].min()), 4)}
                p(f"DAY COVERAGE (distinct days / calendar span): "
                  f"median={per['coverage'].median():.3f} "
                  f"min={per['coverage'].min():.3f}")
                p("\nper-person head (20):")
                p(per.head(20).to_string())
    return res, df


def audit_and_screen(df, name, buf):
    p = lambda *a: print(*a, file=buf)
    p(f"\n--- FABRICATION SCREEN: audit_harness.audit_frame ---")
    out = {}
    try:
        rep = audit_frame(df, name=name, source="kaggle")
        v = verdict(rep)
        p(f"VERDICT              = {v}")
        p(f"n_exact_dependencies = {rep.get('n_exact_dependencies')}")
        p(f"compression          = {rep.get('compression')}")
        p(f"distinct_rows        = {rep.get('distinct_rows')} / {rep.get('n_analyzed')}")
        p(f"max_cramers_v        = {rep.get('max_cramers_v')}")
        p(f"dependent_columns    = {rep.get('dependent_columns')}")
        p(f"ablation             = {json.dumps(rep.get('ablation'), default=str)[:1200]}")
        p(f"exact_dependencies   = {json.dumps(rep.get('exact_dependencies'), default=str)[:1500]}")
        out = {"verdict": v,
               "n_exact_dependencies": rep.get("n_exact_dependencies"),
               "compression": rep.get("compression"),
               "max_cramers_v": rep.get("max_cramers_v"),
               "dependent_columns": rep.get("dependent_columns"),
               "status": rep.get("status")}
    except Exception as e:
        p(f"  audit_frame failed: {type(e).__name__}: {e}")
        out = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
    return out


def main():
    only = sys.argv[1:]
    summary = {}
    sumpath = os.path.join(RECON, "inspect_long_summary.json")
    if os.path.exists(sumpath):
        summary = json.load(open(sumpath))

    folders = sorted(d for d in os.listdir(DATA)
                     if os.path.isdir(os.path.join(DATA, d)))
    if only:
        folders = [f for f in folders if f in only]

    for short in folders:
        folder = os.path.join(DATA, short)
        csvs = sorted(glob.glob(os.path.join(folder, "**", "*.csv"),
                                recursive=True))
        if not csvs:
            continue
        buf = io.StringIO()
        print(f"\n########## {short}  ({len(csvs)} csv files) ##########", file=buf)

        files, frames = [], {}
        # inspect every csv, but for per-subject directories (dozens of tiny
        # identically-shaped files) inspect a sample and aggregate the rest
        groups = {}
        for c in csvs:
            key = os.path.dirname(c)
            groups.setdefault(key, []).append(c)
        for key, lst in groups.items():
            take = lst if len(lst) <= 6 else lst[:3]
            if len(lst) > 6:
                print(f"\n[NOTE] {key}: {len(lst)} csv files with repeated "
                      f"per-subject schema; inspecting {len(take)} in full and "
                      f"aggregating all {len(lst)} below.", file=buf)
            for c in take:
                r = inspect_csv(c, buf)
                if r:
                    files.append(r[0])
                    frames[c] = r[1]
            if len(lst) > 6:
                # aggregate the whole directory into one long frame
                parts = []
                for c in lst:
                    try:
                        d = pd.read_csv(c, low_memory=False)
                        d["__subject"] = os.path.splitext(os.path.basename(c))[0]
                        parts.append(d)
                    except Exception:
                        pass
                if parts:
                    agg = pd.concat(parts, ignore_index=True)
                    aggpath = os.path.join(folder,
                                           f"_AGG_{os.path.basename(key)}.csv")
                    print(f"\n[AGGREGATED {len(parts)} per-subject files -> "
                          f"{agg.shape}]", file=buf)
                    agg.to_csv(aggpath, index=False)
                    r = inspect_csv(aggpath, buf)
                    if r:
                        files.append(r[0])
                        frames[aggpath] = r[1]

        # screen the richest frame (most columns, then most rows)
        best = max(frames, key=lambda k: (frames[k].shape[1], frames[k].shape[0]))
        print(f"\n>>> screening richest frame: {best}", file=buf)
        scr = audit_and_screen(frames[best], short, buf)
        pc = find_person_col(frames[best])
        print(f"\n--- HAND-ROLLED SYNTHETIC TELLS ({os.path.basename(best)}) ---",
              file=buf)
        tells = synth_tells(frames[best], pc)
        print(json.dumps(tells, indent=1, default=str), file=buf)

        txt = os.path.join(OUT, f"{short}.txt")
        with open(txt, "w") as f:
            f.write(buf.getvalue())
        summary[short] = {"files": files, "screen": scr,
                          "screened_file": os.path.basename(best),
                          "tells": tells}
        json.dump(summary, open(sumpath, "w"), indent=1, default=str)
        best_f = max(files, key=lambda r: r.get("n_persons", 0)) if files else {}
        print(f"{short:18s} csvs={len(csvs):4d}  verdict={scr.get('verdict'):26s} "
              f"maxpersons={best_f.get('n_persons')}  -> {txt}", flush=True)

    print("\nWROTE", sumpath)


if __name__ == "__main__":
    main()
