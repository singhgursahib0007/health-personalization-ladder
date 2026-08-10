"""
acquire_03_inspect.py
Full structural inspection + fabrication screen for every downloaded dataset.

Per dataset prints and saves:
  head(20), shape, columns, dtypes, null counts/pct, describe(), value_counts top10,
  duplicate rows, distinct rows, numeric correlation matrix, Cramer's V matrix
  for categorical pairs, and the audit_harness verdict.

Outputs -> recon/inspect/<short>.txt  and  recon/inspect_summary.json
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/macbook/Documents/MyProjects/Greg_Research/multidataset")
from audit_harness import audit_frame, verdict, cramers_v, classify_columns

ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "recon", "inspect")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)

# short -> primary file to analyse (relative to datasets/<short>/)
PRIMARY = {
    "uci_obesity": "ObesityDataSet_raw_and_data_sinthetic.csv",
    "brfss2021_cvd": "CVD_cleaned.csv",
    "brfss2015_diab": "diabetes_012_health_indicators_BRFSS2015.csv",
    "cardio_train": "cardio_train.csv",
    "body_measure": "dataset-310405444.csv",
    "smoking_body": "smoking.csv",
    "gym_members": "gym_members_exercise_tracking.csv",
    "sleep_lifestyle": "Sleep_health_and_lifestyle_dataset.csv",
    "diet_rec_medical": "Personalized_Diet_Recommendations.csv",
    "fitness_wellness_plan": "GYM.csv",
    "exercise_metrics": "exercise_dataset.csv",
    "cab_survey_india": "CAB_05_UT.csv",          # one state; smallest full-schema file
    "nhanes_cdc": "__NHANES_MERGE__",             # special: merged demo+exam+diet+quest
}

MAX_ROWS_FULL = 200_000


def load(short):
    d = os.path.join(DATA, short)
    p = PRIMARY[short]
    if p == "__NHANES_MERGE__":
        demo = pd.read_csv(os.path.join(d, "demographic.csv"), low_memory=False)
        exam = pd.read_csv(os.path.join(d, "examination.csv"), low_memory=False)
        diet = pd.read_csv(os.path.join(d, "diet.csv"), low_memory=False)
        q = pd.read_csv(os.path.join(d, "questionnaire.csv"), low_memory=False)
        keep_demo = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "DMDMARTL"]
        keep_exam = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BMXARMC", "BMXLEG",
                     "BPXSY1", "BPXDI1", "BPXPLS"]
        keep_diet = ["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TTFAT",
                     "DR1TFIBE", "DR1TSODI", "DR1TALCO"]
        keep_q = ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ665", "PAD680", "SLD010H",
                  "SMQ020", "ALQ101", "ALQ130", "DIQ010", "MCQ160C", "BPQ020",
                  "WHD010", "WHD020", "HSD010"]
        def sel(df, cols):
            return df[[c for c in cols if c in df.columns]]
        m = sel(demo, keep_demo)
        for other, cols in ((exam, keep_exam), (diet, keep_diet), (q, keep_q)):
            m = m.merge(sel(other, cols), on="SEQN", how="left")
        return m, "demographic+examination+diet+questionnaire merged on SEQN"
    path = os.path.join(d, p)
    df = pd.read_csv(path, low_memory=False)
    if df.shape[1] == 1:                      # semicolon-delimited (cardio_train)
        df = pd.read_csv(path, sep=";", low_memory=False)
    return df, p


def cramers_matrix(df, cat_cols, cap=14):
    cols = cat_cols[:cap]
    M = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        M.loc[a, a] = 1.0
        for b in cols[i + 1:]:
            try:
                v = cramers_v(df[a].astype(str), df[b].astype(str))
            except Exception:
                v = np.nan
            M.loc[a, b] = M.loc[b, a] = round(v, 3)
    return M


def inspect(short, fh):
    def w(*a):
        s = " ".join(str(x) for x in a)
        print(s); fh.write(s + "\n")

    df, srcfile = load(short)
    w("=" * 100)
    w(f"DATASET: {short}   file: {srcfile}")
    w("=" * 100)
    w("\n--- SHAPE ---"); w(df.shape)
    w("\n--- FIRST 20 ROWS ---"); w(df.head(20).to_string())
    w("\n--- DTYPES ---"); w(df.dtypes.to_string())
    nulls = df.isna().sum()
    w("\n--- NULLS (count / pct) ---")
    w(pd.DataFrame({"n_null": nulls, "pct": (nulls / len(df) * 100).round(3)}).to_string())

    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])
    w("\n--- DESCRIBE (numeric) ---")
    w(num.describe().T.to_string() if num.shape[1] else "(none)")
    w("\n--- VALUE COUNTS (categoricals, top 10) ---")
    for c in cat.columns:
        w(f"\n[{c}] nunique={df[c].nunique()}")
        w(df[c].value_counts(dropna=False).head(10).to_string())
    # low-cardinality numerics behave as categoricals too
    lowcard = [c for c in num.columns if df[c].nunique(dropna=True) <= 12]
    if lowcard:
        w("\n--- VALUE COUNTS (low-cardinality numerics) ---")
        for c in lowcard:
            w(f"\n[{c}] nunique={df[c].nunique()}")
            w(df[c].value_counts(dropna=False).head(10).to_string())

    dup = int(df.duplicated().sum())
    dis = int(df.drop_duplicates().shape[0])
    w(f"\n--- DUPLICATES --- duplicated_rows={dup}  distinct_rows={dis}  "
      f"compression={dis/len(df):.6f}")

    w("\n--- NUMERIC CORRELATION MATRIX ---")
    if num.shape[1] >= 2:
        w(num.corr(numeric_only=True).round(3).to_string())
    else:
        w("(insufficient numerics)")

    usable, cat_cols, dropped = classify_columns(df)
    w(f"\n--- audit_harness classify: {len(usable)} usable, {len(cat_cols)} categorical")
    w(f"    dropped: {dropped}")
    sub = df if len(df) <= 50_000 else df.sample(50_000, random_state=42)
    w("\n--- CRAMER'S V (categorical pairs, capped at 14 cols) ---")
    if len(cat_cols) >= 2:
        w(cramers_matrix(sub, cat_cols).to_string())
    else:
        w("(insufficient categoricals)")

    w("\n--- FABRICATION SCREEN (audit_harness) ---")
    rep = audit_frame(df, name=short, source="kaggle")
    v = verdict(rep)
    w(f"VERDICT={v}  n_exact_dependencies={rep.get('n_exact_dependencies')}  "
      f"compression={rep.get('compression')}")
    w(json.dumps({k: rep[k] for k in rep if k not in ("dropped_cols",)},
                 indent=1, default=str))

    # hand check: is any low-card column an exact function of ONE other column?
    w("\n--- HAND CHECK: single-column determination of each low-card column ---")
    lc = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 30]
    hits = []
    for t in lc:
        for f in df.columns:
            if f == t:
                continue
            if df[f].nunique(dropna=True) > len(df) / 5:
                continue
            g = df[[f, t]].dropna().groupby(f)[t].nunique()
            if len(g) and g.max() == 1:
                hits.append((f, t, int(df[f].nunique())))
    for f, t, k in hits[:40]:
        w(f"   EXACT: {f} (card {k}) -> {t}")
    if not hits:
        w("   none")

    return {
        "short": short, "file": srcfile, "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "columns": list(df.columns), "dup_rows": dup, "distinct_rows": dis,
        "compression": round(dis / len(df), 6),
        "null_pct_max": float((nulls / len(df) * 100).max()),
        "verdict": v, "n_exact_deps": rep.get("n_exact_dependencies"),
        "ablation": rep.get("ablation"), "max_cramers_v": rep.get("max_cramers_v"),
        "exact_single_col_determinations": [{"det": f, "dep": t, "card": k} for f, t, k in hits],
    }


def main():
    only = sys.argv[1:] or list(PRIMARY)
    summ = {}
    sp = os.path.join(ROOT, "recon", "inspect_summary.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
    for short in only:
        try:
            with open(os.path.join(OUT, short + ".txt"), "w") as fh:
                summ[short] = inspect(short, fh)
        except Exception as e:
            print(f"!! {short} FAILED: {type(e).__name__}: {e}")
            summ[short] = {"short": short, "error": f"{type(e).__name__}: {e}"}
    json.dump(summ, open(sp, "w"), indent=1, default=str)
    print("\nsummary ->", sp)


if __name__ == "__main__":
    main()
