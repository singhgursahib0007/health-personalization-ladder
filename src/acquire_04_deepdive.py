"""
acquire_04_deepdive.py
Targeted forensic checks that the generic screen cannot express:

 A. UCI obesity: exact SMOTE-synthesised proportion, and whether the label
    NObeyesdad is a deterministic function of BMI = Weight / Height^2.
 B. The "recommendation / plan" datasets: is the recommendation column a
    lookup on a handful of inputs?
 C. Class-balance and precision fingerprints across every dataset.

Writes recon/deepdive.txt
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
OUTP = os.path.join(ROOT, "recon", "deepdive.txt")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
FH = open(OUTP, "w")


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s); FH.write(s + "\n")


# ---------------------------------------------------------------- A. UCI obesity
def uci_smote():
    w("\n" + "=" * 90 + "\nA. UCI ObesityDataSet - SMOTE synthesis proportion\n" + "=" * 90)
    p = os.path.join(DATA, "uci_obesity", "ObesityDataSet_raw_and_data_sinthetic.csv")
    df = pd.read_csv(p)
    w(f"filename on Kaggle/UCI: {os.path.basename(p)}  (literally '_raw_and_data_sinthetic')")
    w(f"rows={len(df)}")

    # The survey instrument produced INTEGER answers for these items.
    # SMOTE interpolates between neighbours and yields non-integers.
    likert = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for c in likert:
        frac_int = float((df[c] % 1 == 0).mean())
        w(f"  {c:8s} integer-valued in {frac_int*100:6.2f}% of rows; "
          f"range [{df[c].min()}, {df[c].max()}], nunique={df[c].nunique()}")

    all_int = np.ones(len(df), dtype=bool)
    for c in likert:
        all_int &= (df[c] % 1 == 0).values
    # Height/Weight were self-reported: real rows have <=2dp on height, <=1dp weight
    ht_clean = (df["Height"].round(2) == df["Height"]).values
    wt_clean = (df["Weight"].round(1) == df["Weight"]).values
    real_like = all_int & ht_clean & wt_clean
    w(f"\n  rows with ALL Likert/Age items integer-valued : {all_int.sum()} "
      f"({all_int.mean()*100:.2f}%)")
    w(f"  ... AND Height at <=2dp AND Weight at <=1dp    : {real_like.sum()} "
      f"({real_like.mean()*100:.2f}%)")
    w(f"  => estimated SYNTHETIC rows                    : {(~real_like).sum()} "
      f"({(~real_like).mean()*100:.2f}%)")
    w("  Published figure (Palechor & de la Hoz Manotas 2019, Data in Brief 25:104344):")
    w("    77% of records generated synthetically with SMOTE in Weka, 23% collected")
    w("    directly from users via a web platform -> 485 real of 2111.")
    w(f"  Our integer-fingerprint estimate of real rows: {real_like.sum()} "
      f"({real_like.mean()*100:.2f}%)  [target 485 / 22.97%]")

    w("\n  Decimal-place fingerprint of Height (top 8 counts of dp):")
    dp = df["Height"].astype(str).str.split(".").str[1].fillna("").str.len()
    w(dp.value_counts().head(8).to_string())

    w("\n  Class balance of NObeyesdad (SMOTE balances classes by construction):")
    vc = df["NObeyesdad"].value_counts()
    w(vc.to_string())
    w(f"  max/min class ratio = {vc.max()/vc.min():.3f}  (1.00 would be perfect balance)")

    # BMI determinism
    w("\n  Is the label a deterministic function of BMI = Weight/Height^2 ?")
    bmi = df["Weight"] / df["Height"] ** 2
    cuts = [-np.inf, 18.5, 25, 30, 35, 40, np.inf]
    lab = ["Insufficient_Weight", "Normal_Weight", "OVERWEIGHT(I/II)",
           "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    who = pd.cut(bmi, cuts, labels=lab, right=False)
    coll = df["NObeyesdad"].replace({"Overweight_Level_I": "OVERWEIGHT(I/II)",
                                     "Overweight_Level_II": "OVERWEIGHT(I/II)"})
    agree = float((who.astype(str) == coll).mean())
    w(f"  WHO BMI cut-points reproduce the label in {agree*100:.2f}% of rows")
    w("  cross-tab (rows = WHO BMI band, cols = NObeyesdad):")
    w(pd.crosstab(who, df["NObeyesdad"]).to_string())
    w("\n  Single-feature decision stump on BMI alone:")
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    X = bmi.values.reshape(-1, 1)
    y = df["NObeyesdad"]
    for d in (3, 6, None):
        s = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42),
                            X, y, cv=5).mean()
        w(f"    depth={str(d):5s} 5-fold accuracy on BMI only = {s:.4f}")
    s2 = cross_val_score(DecisionTreeClassifier(random_state=42), df[["Height", "Weight"]],
                         y, cv=5).mean()
    w(f"    Height+Weight only, unpruned tree, 5-fold = {s2:.4f}")
    beh = [c for c in df.columns if c not in ("Height", "Weight", "NObeyesdad")]
    Xb = pd.get_dummies(df[beh])
    s3 = cross_val_score(DecisionTreeClassifier(random_state=42), Xb, y, cv=5).mean()
    w(f"    ALL behaviour cols but NO Height/Weight, 5-fold = {s3:.4f}")


# --------------------------------------------- B. the recommendation/plan datasets
def rec_datasets():
    w("\n" + "=" * 90 + "\nB. Datasets carrying an explicit recommendation / plan column\n" + "=" * 90)

    w("\n--- B1. ayeshaseherr/juymmm  'Personalized Fitness Goals and Wellness Plans' ---")
    df = pd.read_csv(os.path.join(DATA, "fitness_wellness_plan", "GYM.csv"))
    w(f"shape={df.shape}  columns={list(df.columns)}")
    w(f"distinct rows = {df.drop_duplicates().shape[0]}  of {len(df)}")
    w("THE ENTIRE FILE IS 16 DISTINCT ROWS REPEATED 5000x EACH:")
    w(df.drop_duplicates().to_string())
    for c in df.columns:
        w(f"  [{c}] nunique={df[c].nunique()} -> {list(df[c].unique())[:8]}")

    w("\n--- B2. ziya07/personalized-medical-diet-recommendations-dataset ---")
    d2 = pd.read_csv(os.path.join(DATA, "diet_rec_medical", "Personalized_Diet_Recommendations.csv"))
    w(f"shape={d2.shape}")
    reccols = [c for c in d2.columns if any(k in c.lower() for k in
               ("recommend", "plan", "suggest", "advice"))]
    w(f"recommendation-like columns: {reccols}")
    for c in reccols:
        w(f"\n [{c}] value_counts:")
        w(d2[c].value_counts().head(12).to_string())
        # dependence of the recommendation on every other column
        best = []
        for f in d2.columns:
            if f == c or d2[f].nunique() > 60:
                continue
            t = pd.crosstab(d2[f], d2[c], normalize="index")
            best.append((t.max(axis=1).mean(), f))
        best.sort(reverse=True)
        w("  strongest single-column predictors (mean modal purity, 1.0 = deterministic):")
        for v, f in best[:8]:
            w(f"    {f:34s} {v:.4f}")
        w(f"  marginal modal share of the recommendation itself: "
          f"{d2[c].value_counts(normalize=True).max():.4f}")

    w("\n--- B3. aakashjoshi123/exercise-and-fitness-metrics-dataset ---")
    d3 = pd.read_csv(os.path.join(DATA, "exercise_metrics", "exercise_dataset.csv"))
    w(f"shape={d3.shape}  columns={list(d3.columns)}")
    for c in d3.columns:
        if d3[c].nunique() <= 20:
            w(f"  [{c}] nunique={d3[c].nunique()}: {dict(d3[c].value_counts().head(8))}")
    w("\n  Is 'Dream Weight'/'Actual Weight' -> 'Exercise' style lookup present?")
    lc = [c for c in d3.columns if 2 <= d3[c].nunique() <= 30]
    for t in lc:
        for f in d3.columns:
            if f == t or d3[f].nunique() > len(d3) / 5:
                continue
            g = d3[[f, t]].dropna().groupby(f)[t].nunique()
            if len(g) and g.max() == 1:
                w(f"    EXACT DEPENDENCY {f} -> {t}")
    w("  (none printed above => no exact single-column dependency)")
    w("\n  numeric precision fingerprint (decimal places per numeric column):")
    for c in d3.select_dtypes(include=[np.number]).columns:
        dp = d3[c].astype(str).str.split(".").str[1].fillna("").str.len()
        w(f"    {c:22s} mean dp={dp.mean():.2f} max dp={dp.max()}  nunique={d3[c].nunique()}")


# ------------------------------------------------- C. balance / precision fingerprints
def fingerprints():
    w("\n" + "=" * 90 + "\nC. Generated-data fingerprints across all datasets\n" + "=" * 90)
    files = {
        "uci_obesity": ("uci_obesity", "ObesityDataSet_raw_and_data_sinthetic.csv", None),
        "brfss2021_cvd": ("brfss2021_cvd", "CVD_cleaned.csv", None),
        "brfss2015_diab": ("brfss2015_diab", "diabetes_012_health_indicators_BRFSS2015.csv", None),
        "cardio_train": ("cardio_train", "cardio_train.csv", ";"),
        "body_measure": ("body_measure", "dataset-310405444.csv", None),
        "smoking_body": ("smoking_body", "smoking.csv", None),
        "gym_members": ("gym_members", "gym_members_exercise_tracking.csv", None),
        "sleep_lifestyle": ("sleep_lifestyle", "Sleep_health_and_lifestyle_dataset.csv", None),
        "diet_rec_medical": ("diet_rec_medical", "Personalized_Diet_Recommendations.csv", None),
        "fitness_wellness_plan": ("fitness_wellness_plan", "GYM.csv", None),
        "exercise_metrics": ("exercise_metrics", "exercise_dataset.csv", None),
        "cab_survey_india": ("cab_survey_india", "CAB_05_UT.csv", None),
    }
    rows = []
    for short, (d, f, sep) in files.items():
        df = pd.read_csv(os.path.join(DATA, d, f), sep=sep or ",", low_memory=False)
        num = df.select_dtypes(include=[np.number])
        cat = [c for c in df.columns if 2 <= df[c].nunique() <= 12]
        # uniformity of categorical marginals: mean |p - 1/k| normalised
        unif = []
        for c in cat:
            p = df[c].value_counts(normalize=True).values
            k = len(p)
            unif.append(float(np.abs(p - 1 / k).sum() / (2 * (1 - 1 / k))))  # 0=uniform,1=degenerate
        # dp fingerprint
        dps = []
        for c in num.columns:
            s = num[c].dropna().astype(str).str.split(".").str[1].fillna("").str.len()
            if len(s):
                dps.append(s.mean())
        rows.append({
            "dataset": short, "n": len(df), "cols": df.shape[1],
            "null_pct": round(df.isna().mean().mean() * 100, 3),
            "distinct_frac": round(df.drop_duplicates().shape[0] / len(df), 5),
            "mean_cat_nonuniformity": round(float(np.mean(unif)), 3) if unif else None,
            "mean_decimal_places": round(float(np.mean(dps)), 2) if dps else None,
        })
    t = pd.DataFrame(rows)
    w(t.to_string(index=False))
    w("\nmean_cat_nonuniformity near 0 = suspiciously uniform categories.")
    w("null_pct exactly 0 across a health survey is itself a warning sign.")


if __name__ == "__main__":
    uci_smote(); rec_datasets(); fingerprints()
    FH.close()
    print("\n->", OUTP)
