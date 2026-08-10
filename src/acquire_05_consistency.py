"""
acquire_05_consistency.py
Physical-plausibility tests the generic screen cannot express.

A real anthropometric measurement obeys arithmetic identities (BMI = kg/m^2),
physiological bounds, and non-trivial demographic correlations (age <-> resting
HR, sex <-> body fat, weight <-> waist). Generated tables usually get the
identity right OR the correlation structure right, rarely both.

Writes recon/consistency.txt
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
FH = open(os.path.join(ROOT, "recon", "consistency.txt"), "w")
pd.set_option("display.width", 200)


def w(*a):
    s = " ".join(str(x) for x in a); print(s); FH.write(s + "\n")


def bmi_identity(df, wt, ht_m, bmi_col, name):
    calc = df[wt] / df[ht_m] ** 2
    err = (calc - df[bmi_col]).abs()
    w(f"  [{name}] BMI identity |W/H^2 - BMI|: mean={err.mean():.4f} "
      f"median={err.median():.4f} p99={err.quantile(.99):.4f} max={err.max():.4f}")
    w(f"           agrees within 0.05 in {(err < 0.05).mean()*100:.2f}% of rows")


w("=" * 90 + "\nPHYSICAL-PLAUSIBILITY / INTERNAL-CONSISTENCY TESTS\n" + "=" * 90)

# ------------------------------------------------------------- gym_members
w("\n--- gym_members (valakhorasani/gym-members-exercise-dataset) ---")
g = pd.read_csv(os.path.join(DATA, "gym_members", "gym_members_exercise_tracking.csv"))
bmi_identity(g, "Weight (kg)", "Height (m)", "BMI", "gym_members")
w(f"  Age vs Resting_BPM r = {g['Age'].corr(g['Resting_BPM']):.4f}  "
  f"(real cohorts: resting HR falls slightly with age/fitness)")
w(f"  Age vs Max_BPM     r = {g['Age'].corr(g['Max_BPM']):.4f}  "
  f"(physiology: HRmax ~ 220-age, expect r about -0.7)")
w(f"  Max_BPM range = [{g['Max_BPM'].min()}, {g['Max_BPM'].max()}]  "
  f"Age range = [{g['Age'].min()}, {g['Age'].max()}]")
w(f"  Fat_Percentage by Gender:\n{g.groupby('Gender')['Fat_Percentage'].describe()[['mean','std','min','max']].to_string()}")
w(f"  BMI vs Fat_Percentage r = {g['BMI'].corr(g['Fat_Percentage']):.4f} "
  f"(real: about +0.7)")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
X = g[["Session_Duration (hours)", "Avg_BPM", "Weight (kg)", "Age", "Height (m)"]]
r2 = cross_val_score(GradientBoostingRegressor(random_state=42), X, g["Calories_Burned"],
                     cv=5, scoring="r2").mean()
w(f"  Calories_Burned from (duration, avgBPM, weight, age, height): 5-fold R2 = {r2:.4f}")
w(f"    -> a closed-form generator would give R2 ~ 1.0")
w(f"  Experience_Level x Workout_Frequency crosstab:")
w(pd.crosstab(g["Experience_Level"], g["Workout_Frequency (days/week)"]).to_string())
w("  Uniformity of Max_BPM / Resting_BPM (real HR is bell-shaped, generators use uniform):")
for c in ["Max_BPM", "Resting_BPM", "Avg_BPM", "Age"]:
    s = g[c]
    w(f"    {c:14s} min={s.min()} max={s.max()} mean={s.mean():.2f} "
      f"std={s.std():.2f}  uniform-std would be {(s.max()-s.min())/np.sqrt(12):.2f}")

# ------------------------------------------------------------- exercise_metrics
w("\n--- exercise_metrics (aakashjoshi123) ---")
e = pd.read_csv(os.path.join(DATA, "exercise_metrics", "exercise_dataset.csv"))
w(f"  BMI vs Actual Weight r = {e['BMI'].corr(e['Actual Weight']):.4f} "
  f"(no height column exists; BMI must be independent noise)")
w(f"  Dream Weight vs Actual Weight r = {e['Dream Weight'].corr(e['Actual Weight']):.4f}")
w(f"  Calories Burn vs Duration r = {e['Calories Burn'].corr(e['Duration']):.4f}")
w(f"  Calories Burn vs Heart Rate r = {e['Calories Burn'].corr(e['Heart Rate']):.4f}")
w(f"  Age vs Heart Rate r = {e['Age'].corr(e['Heart Rate']):.4f}")
w("  full numeric correlation matrix:")
w(e.select_dtypes(include=[np.number]).corr().round(3).to_string())
w("  ranges:")
w(e.select_dtypes(include=[np.number]).agg(["min", "max"]).round(3).to_string())

# ------------------------------------------------------------- sleep_lifestyle
w("\n--- sleep_lifestyle (uom190346a) ---")
s = pd.read_csv(os.path.join(DATA, "sleep_lifestyle", "Sleep_health_and_lifestyle_dataset.csv"))
w(f"  shape={s.shape}  (below the 500-row floor)")
w(f"  columns={list(s.columns)}")
bp = s["Blood Pressure"].str.split("/", expand=True).astype(float)
w(f"  distinct Blood Pressure strings = {s['Blood Pressure'].nunique()} for {len(s)} people")
w(s["Blood Pressure"].value_counts().head(10).to_string())
w(f"  BMI Category values: {dict(s['BMI Category'].value_counts())}   "
  f"(note both 'Normal' and 'Normal Weight' present = coding error)")
w("  Occupation x Sleep Disorder crosstab (a person's JOB should not determine a diagnosis):")
w(pd.crosstab(s["Occupation"], s["Sleep Disorder"], dropna=False).to_string())
w("  Determinism check: does (Occupation, Gender, Age) fix every other column?")
grp = s.groupby(["Occupation", "Gender", "Age"])
for c in s.columns:
    if c in ("Person ID", "Occupation", "Gender", "Age"):
        continue
    mx = grp[c].nunique(dropna=False).max()
    w(f"    {c:26s} max distinct within (Occupation,Gender,Age) group = {mx}")
w(f"  distinct (Occupation,Gender,Age) groups = {grp.ngroups} for {len(s)} rows")
w("  Sleep Disorder vs BMI Category crosstab:")
w(pd.crosstab(s["BMI Category"], s["Sleep Disorder"], dropna=False).to_string())

# ------------------------------------------------------------- body_measure
w("\n--- body_measure (utkarshx27/body-measurements) ---")
b = pd.read_csv(os.path.join(DATA, "body_measure", "dataset-310405444.csv"))
w(f"  shape={b.shape} columns={list(b.columns)}")
num = b.select_dtypes(include=[np.number])
w("  describe:")
w(num.describe().T[["mean", "std", "min", "max"]].round(2).to_string())
if {"wgt", "hgt"} <= set(b.columns):
    bmi = b["wgt"] / (b["hgt"] / 100) ** 2
    w(f"  derived BMI mean={bmi.mean():.2f} sd={bmi.std():.2f} range=[{bmi.min():.1f},{bmi.max():.1f}]")
    w(f"  wgt vs hgt r = {b['wgt'].corr(b['hgt']):.4f}")
    if "sex" in b.columns:
        w(b.groupby("sex")[["wgt", "hgt", "age"]].mean().round(2).to_string())
w("  correlations among girth measures (should be strongly positive, real anthropometry):")
gcols = [c for c in b.columns if "gi" in c.lower() or "di" in c.lower()][:10]
if gcols:
    w(b[gcols].corr().round(3).to_string())

# ------------------------------------------------------------- cardio_train
w("\n--- cardio_train (pirogovskiy) ---")
c = pd.read_csv(os.path.join(DATA, "cardio_train", "cardio_train.csv"), sep=";")
w(f"  shape={c.shape}")
w(f"  age is in DAYS: min={c['age'].min()} max={c['age'].max()} "
  f"-> years [{c['age'].min()/365.25:.1f}, {c['age'].max()/365.25:.1f}]")
w(f"  height cm range [{c['height'].min()}, {c['height'].max()}]  "
  f"weight kg range [{c['weight'].min()}, {c['weight'].max()}]")
w(f"  ap_hi (systolic) range [{c['ap_hi'].min()}, {c['ap_hi'].max()}]  "
  f"ap_lo range [{c['ap_lo'].min()}, {c['ap_lo'].max()}]")
w(f"  IMPLAUSIBLE ap_hi (<60 or >250): {((c.ap_hi<60)|(c.ap_hi>250)).sum()} rows "
  f"({((c.ap_hi<60)|(c.ap_hi>250)).mean()*100:.2f}%)")
w(f"  IMPLAUSIBLE height (<130 or >210): {((c.height<130)|(c.height>210)).sum()} rows")
w(f"  ap_lo > ap_hi in {(c.ap_lo>c.ap_hi).sum()} rows")
w("  -> messy, out-of-range, human-entry errors: a fingerprint of REAL data entry")
w(f"  cardio target balance: {dict(c['cardio'].value_counts())}")
w("  BMI vs cardio (should rise monotonically):")
c["bmi"] = c.weight / (c.height / 100) ** 2
w(c.groupby(pd.cut(c.bmi, [0, 18.5, 25, 30, 35, 100]))["cardio"].mean().round(4).to_string())
w("  age(yrs) vs cardio:")
w(c.groupby(pd.cut(c.age / 365.25, [29, 40, 50, 55, 60, 70]))["cardio"].mean().round(4).to_string())

# ------------------------------------------------------------- smoking_body
w("\n--- smoking_body (kukuroo3, Korean NHIS screening) ---")
k = pd.read_csv(os.path.join(DATA, "smoking_body", "smoking.csv"))
w(f"  shape={k.shape}")
w(f"  columns={list(k.columns)}")
if "height(cm)" in k.columns:
    w(f"  height granularity (real screening rounds to 5cm): "
      f"{dict(k['height(cm)'].value_counts().head(8))}")
w("  correlation of anthropometrics with labs:")
sel = [c for c in ["age", "height(cm)", "weight(kg)", "waist(cm)", "systolic",
                   "relaxation", "fasting blood sugar", "Cholesterol", "triglyceride",
                   "HDL", "LDL", "hemoglobin", "serum creatinine", "ALT", "Gtp"]
       if c in k.columns]
w(k[sel].corr().round(3).to_string())
if "smoking" in k.columns:
    w(f"  smoking balance: {dict(k['smoking'].value_counts())}")
    w("  mean profile by smoking status:")
    w(k.groupby("smoking")[sel].mean().round(2).T.to_string())

# ------------------------------------------------------------- cab_survey_india
w("\n--- cab_survey_india (rajanand/cab-survey, Govt of India AHS CAB) ---")
ca = pd.read_csv(os.path.join(DATA, "cab_survey_india", "CAB_05_UT.csv"), low_memory=False)
w(f"  shape={ca.shape}")
w(f"  columns={list(ca.columns)}")
w("  the 34 'exact dependencies' flagged by the screen are geographic hierarchy keys:")
for a, b_ in [("District_code", "State_code"), ("Rural_Urban", "State_code"),
              ("PSU_ID", "District_code")]:
    if a in ca.columns and b_ in ca.columns:
        mx = ca.groupby(a)[b_].nunique().max()
        w(f"    {a} -> {b_}: max distinct = {mx} (1 means exact, i.e. an administrative nesting)")
key = [c for c in ["Weight_in_kg", "Length_height_cm", "Haemoglobin_level",
                   "BP_systolic", "BP_Diastolic", "Fasting", "Age"] if c in ca.columns]
if key:
    w("  key clinical columns present:")
    w(ca[key].describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())
    w(f"  missingness on those columns:\n"
      f"{(ca[key].isna().mean()*100).round(2).to_string()}")

FH.close()
print("\n-> recon/consistency.txt")
