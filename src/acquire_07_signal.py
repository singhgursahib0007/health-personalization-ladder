"""
acquire_07_signal.py
Two-sided fabrication test, complementary to audit_harness.

audit_harness catches tables where a target is TOO predictable (lookup tables).
The opposite failure is equally diagnostic: a table where the "outcome" carries
NO signal at all above the majority baseline, i.e. the column was drawn from an
RNG independently of every covariate. Real epidemiology sits in between.

For each dataset/target we report honest 5-fold CV accuracy (or AUC) of a
gradient-boosted tree on all other columns, against the majority baseline.

Writes recon/signal.txt
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
FH = open(os.path.join(ROOT, "recon", "signal.txt"), "w")


def w(*a):
    s = " ".join(str(x) for x in a); print(s); FH.write(s + "\n")


def test(df, target, name, drop=()):
    d = df.drop(columns=[c for c in drop if c in df.columns]).dropna(subset=[target])
    if len(d) > 60000:
        d = d.sample(60000, random_state=42)
    y = pd.factorize(d[target])[0]
    X = d.drop(columns=[target])
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
        X[c] = pd.to_numeric(X[c], errors="coerce")
    base = float(pd.Series(y).value_counts(normalize=True).max())
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    m = HistGradientBoostingClassifier(random_state=42)
    acc = cross_val_score(m, X, y, cv=cv, scoring="accuracy").mean()
    # ACCURACY IS USELESS ON IMBALANCED TARGETS: a 92%-negative outcome cannot
    # beat its own baseline even with strong genuine discrimination. AUC is the
    # decisive statistic; accuracy is kept only to expose lookup-table behaviour.
    nclass = len(np.unique(y))
    scoring = "roc_auc" if nclass == 2 else "roc_auc_ovr"
    try:
        auc = cross_val_score(m, X, y, cv=cv, scoring=scoring).mean()
    except Exception:
        auc = float("nan")
    tag = ("DETERMINISTIC (lookup / leakage)" if acc > 0.99 or auc > 0.999 else
           "NO SIGNAL: outcome independent of covariates -> RNG-generated"
           if auc < 0.55 else
           "plausible epidemiological signal")
    w(f"  {name:46s} target={target:24s} n={len(d):>7,} k={nclass} "
      f"base={base:.4f} acc={acc:.4f} AUC={auc:.4f}  {tag}")
    return acc, base


w("=" * 110)
w("PREDICTIVE-SIGNAL TEST: is the outcome learnable from the covariates at all?")
w("  lift < 0.01  => the column is independent noise (fabricated by RNG)")
w("  acc  > 0.99  => the column is a deterministic function of the inputs (lookup table)")
w("=" * 110)

w("\n[real-provenance candidates]")
d = pd.read_csv(os.path.join(DATA, "brfss2021_cvd", "CVD_cleaned.csv"))
test(d, "Heart_Disease", "brfss2021_cvd")
test(d, "Diabetes", "brfss2021_cvd")

d = pd.read_csv(os.path.join(DATA, "brfss2015_diab", "diabetes_012_health_indicators_BRFSS2015.csv"))
test(d, "Diabetes_012", "brfss2015_diab")

d = pd.read_csv(os.path.join(DATA, "cardio_train", "cardio_train.csv"), sep=";")
test(d, "cardio", "cardio_train", drop=("id",))

d = pd.read_csv(os.path.join(DATA, "smoking_body", "smoking.csv"))
test(d, "smoking", "smoking_body", drop=("ID",))

d = pd.read_csv(os.path.join(DATA, "uci_obesity", "ObesityDataSet_raw_and_data_sinthetic.csv"))
test(d, "NObeyesdad", "uci_obesity (all cols)")
test(d, "NObeyesdad", "uci_obesity (no Height/Weight)", drop=("Height", "Weight"))

d = pd.read_csv(os.path.join(DATA, "body_measure", "dataset-310405444.csv"))
test(d, "sex", "body_measure")

ca = pd.read_csv(os.path.join(DATA, "cab_survey_india", "CAB_05_UT.csv"),
                 low_memory=False).replace(-1, np.nan)
ca = ca[ca.age >= 18]
ca["hypertension"] = ((ca.bp_systolic >= 140) | (ca.bp_diastolic >= 90)).astype(int)
test(ca[["age", "sex", "weight_in_kg", "length_height_cm", "haemoglobin_level",
         "pulse_rate", "fasting_blood_glucose_mg_dl", "rural_urban", "marital_status",
         "hypertension"]], "hypertension", "cab_survey_india")

w("\n[suspect candidates]")
d = pd.read_csv(os.path.join(DATA, "gym_members", "gym_members_exercise_tracking.csv"))
test(d, "Experience_Level", "gym_members")
test(d, "Workout_Type", "gym_members")
test(d, "Gender", "gym_members")

d = pd.read_csv(os.path.join(DATA, "sleep_lifestyle", "Sleep_health_and_lifestyle_dataset.csv"))
d["Sleep Disorder"] = d["Sleep Disorder"].fillna("None")
test(d, "Sleep Disorder", "sleep_lifestyle (with Occupation)", drop=("Person ID",))
test(d, "Sleep Disorder", "sleep_lifestyle (no Occupation)", drop=("Person ID", "Occupation"))

d = pd.read_csv(os.path.join(DATA, "diet_rec_medical", "Personalized_Diet_Recommendations.csv"))
test(d, "Recommended_Meal_Plan", "diet_rec_medical",
     drop=("Patient_ID", "Recommended_Calories", "Recommended_Protein",
           "Recommended_Carbs", "Recommended_Fats"))

d = pd.read_csv(os.path.join(DATA, "exercise_metrics", "exercise_dataset.csv"))
test(d, "Exercise", "exercise_metrics", drop=("ID",))
test(d, "Exercise Intensity", "exercise_metrics", drop=("ID",))

d = pd.read_csv(os.path.join(DATA, "fitness_wellness_plan", "GYM.csv"))
test(d, "Exercise Schedule", "fitness_wellness_plan")

FH.close()
print("\n-> recon/signal.txt")
