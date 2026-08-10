"""
acquire_06_supplement.py
Follow-ups the first consistency pass missed:
  - CAB (India AHS) clinical columns, missingness, administrative nesting
  - NHANES merged frame: coverage of each block, plausibility
  - BRFSS derivatives: what the columns actually are, and how they were derived
Appends to recon/consistency.txt
"""
import os, warnings, glob
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
FH = open(os.path.join(ROOT, "recon", "consistency.txt"), "a")
pd.set_option("display.width", 200)


def w(*a):
    s = " ".join(str(x) for x in a); print(s); FH.write(s + "\n")


w("\n" + "=" * 90 + "\nSUPPLEMENT\n" + "=" * 90)

# ---------------------------------------------------------------- CAB
w("\n--- cab_survey_india detail ---")
files = sorted(glob.glob(os.path.join(DATA, "cab_survey_india", "*.csv")))
w(f"  {len(files)} state files: {[os.path.basename(f) for f in files]}")
tot = 0
for f in files:
    n = sum(1 for _ in open(f, errors="ignore")) - 1
    tot += n
    w(f"    {os.path.basename(f):16s} {n:>9,} rows")
w(f"  TOTAL individuals across all state files: {tot:,}")

ca = pd.read_csv(files[0], low_memory=False)
for a, b in [("district_code", "state_code"), ("psu_id", "district_code"),
             ("stratum", "rural_urban")]:
    w(f"    {a} -> {b}: max distinct = {ca.groupby(a)[b].nunique().max()} "
      f"(1 = exact administrative nesting, not a fabricated rule)")
key = ["age", "sex", "weight_in_kg", "length_height_cm", "haemoglobin",
       "bp_systolic", "bp_diastolic", "pulse_rate", "fasting_blood_glucose_mg_dl",
       "marital_status", "illness_type", "test_salt_iodine"]
key = [c for c in key if c in ca.columns]
w("\n  clinical columns (one state file, Uttarakhand):")
w(ca[key].describe(include="all").T.to_string())
w("\n  missingness %:")
w((ca[key].isna().mean() * 100).round(2).to_string())
ad = ca[(ca.age >= 18) & ca.weight_in_kg.notna() & ca.length_height_cm.notna()].copy()
ad["bmi"] = ad.weight_in_kg / (ad.length_height_cm / 100) ** 2
ad = ad[(ad.bmi > 10) & (ad.bmi < 60)]
w(f"\n  adults (18+) with height AND weight: {len(ad):,} of {len(ca):,}")
w(f"  BMI mean={ad.bmi.mean():.2f} sd={ad.bmi.std():.2f}  "
  f"(India adult mean is ~21-22: plausible)")
w(f"  BMI vs bp_systolic r = {ad.bmi.corr(ad.bp_systolic):.4f}")
w(f"  age vs bp_systolic  r = {ad.age.corr(ad.bp_systolic):.4f}")
w(f"  haemoglobin by sex:\n{ad.groupby('sex')['haemoglobin'].mean().round(2).to_string()} "
  f"(1=male 2=female; males should be ~2 g/dL higher)")

# ---------------------------------------------------------------- NHANES
w("\n--- nhanes_cdc merged frame ---")
d = os.path.join(DATA, "nhanes_cdc")
demo = pd.read_csv(os.path.join(d, "demographic.csv"), low_memory=False)
exam = pd.read_csv(os.path.join(d, "examination.csv"), low_memory=False)
diet = pd.read_csv(os.path.join(d, "diet.csv"), low_memory=False)
q = pd.read_csv(os.path.join(d, "questionnaire.csv"), low_memory=False)
labs = pd.read_csv(os.path.join(d, "labs.csv"), low_memory=False)
for nm, df in [("demographic", demo), ("examination", exam), ("diet", diet),
               ("questionnaire", q), ("labs", labs)]:
    w(f"  {nm:14s} shape={df.shape}")
w(f"  This is NHANES 2013-2014 (SEQN range {demo.SEQN.min():.0f}-{demo.SEQN.max():.0f}).")
m = demo[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]]
m = m.merge(exam[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BPXSY1", "BPXDI1"]], on="SEQN")
m = m.merge(diet[["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TFIBE", "DR1TSODI"]],
            on="SEQN", how="left")
qq = [c for c in ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ665", "PAD680", "SLD010H",
                  "SMQ020", "ALQ101", "DIQ010", "BPQ020"] if c in q.columns]
m = m.merge(q[qq], on="SEQN", how="left")
w(f"  merged adult-capable frame shape = {m.shape}")
ad = m[m.RIDAGEYR >= 18]
w(f"  adults 18+ = {len(ad):,}")
comp = ad.dropna(subset=["BMXBMI", "BMXWAIST", "DR1TKCAL", "PAQ605", "SMQ020"])
w(f"  adults with BMI+waist+kcal+activity+smoking all present = {len(comp):,}")
w("  missingness % on the merged frame:")
w((m.isna().mean() * 100).round(2).to_string())
w("  BMI identity check BMXWT/BMXHT^2 vs BMXBMI:")
e = (ad.BMXWT / (ad.BMXHT / 100) ** 2 - ad.BMXBMI).abs()
w(f"    mean abs err = {e.mean():.4f}, within 0.05 in {(e < 0.05).mean()*100:.2f}%")
w(f"  BMI vs waist r = {ad.BMXBMI.corr(ad.BMXWAIST):.4f}")
w(f"  age vs systolic BP r = {ad.RIDAGEYR.corr(ad.BPXSY1):.4f}")
w(f"  kcal by sex: {ad.groupby('RIAGENDR')['DR1TKCAL'].mean().round(0).to_dict()} "
  f"(1=M 2=F; men should eat ~700 kcal more)")

# ---------------------------------------------------------------- BRFSS derivatives
w("\n--- brfss2021_cvd (alphiree/cardiovascular-diseases-risk-prediction-dataset) ---")
b1 = pd.read_csv(os.path.join(DATA, "brfss2021_cvd", "CVD_cleaned.csv"))
w(f"  shape={b1.shape} columns={list(b1.columns)}")
w(b1.head(6).to_string())
e = (b1["Weight_(kg)"] / (b1["Height_(cm)"] / 100) ** 2 - b1["BMI"]).abs()
w(f"  BMI identity: mean abs err={e.mean():.4f}, within 0.05 in {(e<0.05).mean()*100:.2f}%")
w(f"  Height rounded to inches? distinct heights = {b1['Height_(cm)'].nunique()} "
  f"-> {sorted(b1['Height_(cm)'].unique())[:12]}")
w("  Heart_Disease rate by BMI band:")
w(b1.groupby(pd.cut(b1.BMI, [0, 18.5, 25, 30, 35, 100]))["Heart_Disease"]
  .apply(lambda s: (s == "Yes").mean()).round(4).to_string())
w("  Heart_Disease rate by Exercise:")
w(b1.groupby("Exercise")["Heart_Disease"].apply(lambda s: (s == "Yes").mean()).round(4).to_string())
w("  Heart_Disease rate by Smoking_History:")
w(b1.groupby("Smoking_History")["Heart_Disease"].apply(lambda s: (s == "Yes").mean()).round(4).to_string())
w("  Heart_Disease rate by Age_Category:")
w(b1.groupby("Age_Category")["Heart_Disease"].apply(lambda s: (s == "Yes").mean()).round(4).to_string())
w(f"  class balance Heart_Disease: {dict(b1.Heart_Disease.value_counts())} "
  f"(8% positives = real epidemiological prevalence, not a balanced synthetic target)")

w("\n--- brfss2015_diab (alexteboul/diabetes-health-indicators-dataset) ---")
b2 = pd.read_csv(os.path.join(DATA, "brfss2015_diab", "diabetes_012_health_indicators_BRFSS2015.csv"))
w(f"  shape={b2.shape} columns={list(b2.columns)}")
w(f"  Diabetes_012 balance: {dict(b2.Diabetes_012.value_counts())}")
w("  Diabetes rate by BMI band:")
w(b2.groupby(pd.cut(b2.BMI, [0, 18.5, 25, 30, 35, 100]))["Diabetes_012"]
  .apply(lambda s: (s > 0).mean()).round(4).to_string())
w("  Diabetes rate by PhysActivity:")
w(b2.groupby("PhysActivity")["Diabetes_012"].apply(lambda s: (s > 0).mean()).round(4).to_string())
w("  Diabetes rate by Age band (BRFSS 14-level code):")
w(b2.groupby("Age")["Diabetes_012"].apply(lambda s: (s > 0).mean()).round(4).to_string())
w(f"  duplicate rows: {b2.duplicated().sum():,} of {len(b2):,} "
  f"-> after 21 binarised columns, distinct profiles collide by construction")
w(f"  distinct rows on the 21 features only: "
  f"{b2.drop(columns=['Diabetes_012']).drop_duplicates().shape[0]:,}")

FH.close()
print("\n-> appended to recon/consistency.txt")
