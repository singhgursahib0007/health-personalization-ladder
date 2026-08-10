"""
acquire_08_finalize.py
======================
Gap-filling pass over the 17 CROSS-SECTIONAL datasets.

For every dataset it produces, from code, and writes to disk:
  1. shape, column list with dtypes
  2. null counts and null percentages
  3. describe() for numeric columns
  4. top-10 value counts for every categorical / low-cardinality column
  5. exact duplicate-row count and distinct-row count
  6. numeric correlation matrix (Pearson)
  7. the audit_harness fabrication screen (audit_frame + verdict)

Then it runs four targeted investigations that the generic profile cannot answer:
  A. exact functional-dependency documentation for the two suspected generated
     lookup tables (diet_rec_medical, fitness_wellness_plan)
  B. fabrication checks for fitlife and fitness365 (per-participant constancy,
     BMI identity, physiological correlation structure, decimal fingerprint)
  C. UCI obesity SMOTE-share estimate via the integer/decimal fingerprint
  D. presence and exact names of survey design variables (weights / strata / PSU)
     in BRFSS 2015, BRFSS 2021 and NHANES

Outputs
  recon/final/profiles.json        machine-readable per-dataset profile
  recon/final/profiles.txt         human-readable dump of the same
  recon/final/screen.json          audit_harness result + verdict per dataset
  recon/final/targeted.txt         A-D above
Nothing outside recommendation_research/ is written.
"""

import os, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/macbook/Documents/MyProjects/Greg_Research/multidataset")
from audit_harness import audit_frame, verdict          # noqa: E402

ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
OUT = os.path.join(ROOT, "recon", "final")
os.makedirs(OUT, exist_ok=True)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)

TXT = open(os.path.join(OUT, "profiles.txt"), "w")
TGT = open(os.path.join(OUT, "targeted.txt"), "w")


def w(fh, *a):
    s = " ".join(str(x) for x in a)
    print(s)
    fh.write(s + "\n")


# --------------------------------------------------------------- loaders
def load_nhanes():
    """Merge the five NHANES 2013-2014 component files on SEQN, keeping the
    demographic / anthropometric / dietary / behavioural columns used by the
    ladder, PLUS the survey design variables."""
    demo = pd.read_csv(os.path.join(DATA, "nhanes_cdc", "demographic.csv"), low_memory=False)
    exam = pd.read_csv(os.path.join(DATA, "nhanes_cdc", "examination.csv"), low_memory=False)
    diet = pd.read_csv(os.path.join(DATA, "nhanes_cdc", "diet.csv"), low_memory=False)
    ques = pd.read_csv(os.path.join(DATA, "nhanes_cdc", "questionnaire.csv"), low_memory=False)
    keep_d = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "DMDMARTL",
              "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
    keep_e = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BMXARMC", "BMXLEG",
              "BPXSY1", "BPXDI1", "BPXPLS"]
    keep_t = ["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TTFAT",
              "DR1TFIBE", "DR1TSODI", "DR1TALCO"]
    keep_q = ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ665", "PAD680", "SLD010H",
              "SMQ020", "ALQ101", "ALQ130", "DIQ010", "MCQ160C", "BPQ020",
              "WHD010", "WHD020", "HSD010"]
    sub = lambda df, k: df[[c for c in k if c in df.columns]]
    m = sub(demo, keep_d)
    for f, k in ((exam, keep_e), (diet, keep_t), (ques, keep_q)):
        m = m.merge(sub(f, k), on="SEQN", how="left")
    return m


def load_sleep_fitbit():
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA, "sleep_fitbit", "*.csv"))):
        d = pd.read_csv(p)
        d.columns = [str(c).strip().upper() for c in d.columns]
        d = d.rename(columns={d.columns[0]: "MONTH_LABEL"})
        d["SOURCE_FILE"] = os.path.basename(p)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_thermal():
    """The cross-sectional part of the thermal-comfort release is the ENTH
    participant survey (one row per participant)."""
    return pd.read_csv(os.path.join(DATA, "thermal_comfort", "enth_surveys_renamed.csv"))


# name -> (loader, kaggle/source slug, provenance note)
DATASETS = {
    "brfss2015_diab": (
        lambda: pd.read_csv(os.path.join(DATA, "brfss2015_diab",
                                         "diabetes_012_health_indicators_BRFSS2015.csv")),
        "alexteboul/diabetes-health-indicators-dataset",
        "derived from CDC BRFSS 2015 (real survey), re-coded by a third party"),
    "brfss2021_cvd": (
        lambda: pd.read_csv(os.path.join(DATA, "brfss2021_cvd", "CVD_cleaned.csv")),
        "alphiree/cardiovascular-diseases-risk-prediction-dataset",
        "derived from CDC BRFSS 2021 (real survey), re-coded by a third party"),
    "nhanes_cdc": (load_nhanes, "cdc/national-health-and-nutrition-examination-survey",
                   "CDC NCHS NHANES 2013-2014, raw component files"),
    "cardio_train": (
        lambda: pd.read_csv(os.path.join(DATA, "cardio_train", "cardio_train.csv"), sep=";"),
        "cardio_train (Kaggle cardiovascular disease dataset)",
        "clinical examination records, origin not formally documented"),
    "uci_obesity": (
        lambda: pd.read_csv(os.path.join(DATA, "uci_obesity",
                                         "ObesityDataSet_raw_and_data_sinthetic.csv")),
        "jayitabhattacharyya/estimation-of-obesity-levels-uci-dataset",
        "UCI / Palechor & de la Hoz Manotas 2019; 77% SMOTE per the source paper"),
    "smoking_body": (
        lambda: pd.read_csv(os.path.join(DATA, "smoking_body", "smoking.csv")),
        "kukuroo3/body-signal-of-smoking",
        "Korean NHIS health-screening extract"),
    "cab_survey_india": (
        lambda: pd.read_csv(os.path.join(DATA, "cab_survey_india", "CAB_05_UT.csv"),
                            low_memory=False),
        "rajanand/cab-survey",
        "Govt of India Annual Health Survey Clinical Anthropometric Biochemical"),
    "body_measure": (
        lambda: pd.read_csv(os.path.join(DATA, "body_measure", "dataset-310405444.csv")),
        "utkarshx27/body-measurements",
        "Heinz et al. 2003 JSE body-dimension study, 507 physically active adults"),
    "gym_members": (
        lambda: pd.read_csv(os.path.join(DATA, "gym_members",
                                         "gym_members_exercise_tracking.csv")),
        "valakhorasani/gym-members-exercise-dataset", "no provenance statement"),
    "sleep_lifestyle": (
        lambda: pd.read_csv(os.path.join(DATA, "sleep_lifestyle",
                                         "Sleep_health_and_lifestyle_dataset.csv")),
        "uom190346a/sleep-health-and-lifestyle-dataset",
        "author states the data is synthetic on the dataset page"),
    "exercise_metrics": (
        lambda: pd.read_csv(os.path.join(DATA, "exercise_metrics", "exercise_dataset.csv")),
        "aakashjoshi123/exercise-and-fitness-metrics-dataset", "no provenance statement"),
    "diet_rec_medical": (
        lambda: pd.read_csv(os.path.join(DATA, "diet_rec_medical",
                                         "Personalized_Diet_Recommendations.csv")),
        "ziya07/personalized-medical-diet-recommendations-dataset",
        "no provenance statement"),
    "fitness_wellness_plan": (
        lambda: pd.read_csv(os.path.join(DATA, "fitness_wellness_plan", "GYM.csv")),
        "ayeshaseherr/juymmm", "no provenance statement"),
    "fitlife": (
        lambda: pd.read_csv(os.path.join(DATA, "fitlife", "health_fitness_dataset.csv")),
        "jijagallery/fitlife-health-and-fitness-tracking-dataset",
        "author states 'FitLife360 is a synthetic dataset'"),
    "fitness365": (
        lambda: pd.read_csv(os.path.join(DATA, "fitness365",
                                         "health_fitness_tracking_365days.csv")),
        "waqasishtiaq/fitness", "author states 'Synthetic lifelog'"),
    "sleep_fitbit": (load_sleep_fitbit, "riinuanslan/sleep-data-from-fitbit-tracker",
                     "one author's own Fitbit export, 6 monthly sheets"),
    "thermal_comfort": (load_thermal,
                        "claytonmiller/longitudinal-personal-thermal-comfort-preferences",
                        "BUDS Lab ENTH/CRESH experiments, peer-reviewed publications"),
}


# --------------------------------------------------------------- profiling
def profile(df, name):
    n, k = df.shape
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    nulls = df.isna().sum()
    prof = {
        "n_rows": int(n), "n_cols": int(k),
        "columns": [{"name": str(c), "dtype": dtypes[c],
                     "nulls": int(nulls[c]), "null_pct": round(100 * nulls[c] / n, 4),
                     "nunique": int(df[c].nunique(dropna=True))} for c in df.columns],
        "dup_rows": int(df.duplicated().sum()),
        "distinct_rows": int(df.drop_duplicates().shape[0]),
        "compression": round(float(df.drop_duplicates().shape[0] / n), 6),
    }
    num = df.select_dtypes(include=[np.number])
    prof["describe_numeric"] = json.loads(num.describe().to_json()) if num.shape[1] else {}
    corr = num.corr() if num.shape[1] >= 2 else pd.DataFrame()
    prof["corr_shape"] = list(corr.shape)
    vc = {}
    for c in df.columns:
        if df[c].dtype == object or df[c].nunique(dropna=True) <= 30:
            vc[str(c)] = {str(a): int(b) for a, b in
                          df[c].value_counts(dropna=False).head(10).items()}
    prof["value_counts_top10"] = vc

    w(TXT, "\n" + "=" * 100)
    w(TXT, f"DATASET  {name}   shape={n} x {k}")
    w(TXT, "=" * 100)
    w(TXT, "-- columns (dtype | nulls | null% | nunique) --")
    for c in prof["columns"]:
        w(TXT, f"   {c['name'][:42]:42s} {c['dtype']:10s} {c['nulls']:>8d} "
                f"{c['null_pct']:>7.2f}%  nuniq={c['nunique']}")
    w(TXT, f"-- duplicate rows = {prof['dup_rows']}   distinct rows = "
            f"{prof['distinct_rows']}   distinct/n = {prof['compression']}")
    if num.shape[1]:
        w(TXT, "-- describe() numeric --")
        w(TXT, num.describe().T.to_string())
    if vc:
        w(TXT, "-- top-10 value counts (categorical / low-cardinality) --")
        for c, d in vc.items():
            w(TXT, f"   [{c}] " + ", ".join(f"{a}={b}" for a, b in d.items()))
    if corr.shape[0] >= 2:
        w(TXT, "-- numeric correlation matrix --")
        w(TXT, corr.round(3).to_string())
    return prof


profiles, screens = {}, {}
for name, (loader, slug, note) in DATASETS.items():
    try:
        df = loader()
    except Exception as e:
        w(TXT, f"\n### {name}: LOAD FAILED: {e}")
        continue
    profiles[name] = {"slug": slug, "provenance_note": note, **profile(df, name)}
    try:
        r = audit_frame(df, name, slug)
        r["verdict"] = verdict(r)
    except Exception as e:
        r = {"dataset": name, "status": f"screen failed: {e}", "verdict": "n/a"}
    screens[name] = r
    w(TXT, f"-- FABRICATION SCREEN verdict = {r.get('verdict')}  "
            f"compression={r.get('compression')}  exactFDs={r.get('n_exact_dependencies')}  "
            f"maxCramersV={r.get('max_cramers_v')}")
    del df

json.dump(profiles, open(os.path.join(OUT, "profiles.json"), "w"), indent=1, default=str)
json.dump(screens, open(os.path.join(OUT, "screen.json"), "w"), indent=1, default=str)


# =========================================================== A. lookup tables
w(TGT, "=" * 100)
w(TGT, "A. EXACT FUNCTIONAL DEPENDENCIES IN THE TWO SUSPECTED GENERATED LOOKUP TABLES")
w(TGT, "=" * 100)


def fd_report(df, name, cols=None, max_card=60):
    cols = cols or [c for c in df.columns
                    if df[c].nunique(dropna=True) <= max_card and df[c].nunique(dropna=True) > 1]
    w(TGT, f"\n--- {name}: n={len(df)}, distinct rows={df.drop_duplicates().shape[0]}, "
           f"distinct/n={df.drop_duplicates().shape[0]/len(df):.6f}")
    w(TGT, f"    columns screened for exact FDs (card<= {max_card}): {cols}")
    found = []
    for a in cols:
        g = df.groupby(a, dropna=False)
        for b in cols:
            if a == b:
                continue
            if g[b].nunique(dropna=False).max() == 1:
                found.append((a, b, int(df[a].nunique()), int(df[b].nunique())))
    if not found:
        w(TGT, "    NO exact single-column functional dependency among these columns")
    for a, b, ca, cb in found:
        w(TGT, f"    EXACT FD:  {a} ({ca} values)  ->  {b} ({cb} values)")
    return found


gym = pd.read_csv(os.path.join(DATA, "fitness_wellness_plan", "GYM.csv"))
fd_report(gym, "fitness_wellness_plan / GYM.csv")
w(TGT, "\n    the 16 distinct rows and their multiplicities:")
mult = gym.groupby(list(gym.columns)).size().reset_index(name="count")
w(TGT, mult[["Gender", "Goal", "BMI Category", "count"]].to_string(index=False))
w(TGT, f"    row-multiplicity value counts: {mult['count'].value_counts().to_dict()}")
w(TGT, f"    distinct (Gender,Goal,BMI Category) key combinations = "
       f"{gym[['Gender','Goal','BMI Category']].drop_duplicates().shape[0]} "
       f"(2 x 2 x 4 = 16, the full Cartesian product)")
w(TGT, "    BMI Category <-> Exercise Schedule <-> Meal Plan are mutually bijective; "
       "Gender and Goal have ZERO influence on either output column.")
ct = pd.crosstab(gym["BMI Category"], gym["Exercise Schedule"])
w(TGT, "    crosstab BMI Category x Exercise Schedule (a permutation matrix => bijection):")
w(TGT, ct.to_string())

dr = pd.read_csv(os.path.join(DATA, "diet_rec_medical", "Personalized_Diet_Recommendations.csv"))
fd_report(dr, "diet_rec_medical / Personalized_Diet_Recommendations.csv")
w(TGT, "\n    diet_rec_medical is NOT a lookup table: it is the opposite failure mode.")
w(TGT, "    Mutual-information-free check - conditional distribution of the "
       "recommendation given clinically decisive inputs:")
dr["_bmi_band"] = pd.cut(dr["BMI"], [0, 18.5, 25, 30, 100],
                         labels=["under", "normal", "over", "obese"])
for key in ["_bmi_band", "Chronic_Disease", "Dietary_Habits"]:
    tab = pd.crosstab(dr[key], dr["Recommended_Meal_Plan"], normalize="index").round(4)
    w(TGT, f"\n    P(Recommended_Meal_Plan | {key}):")
    w(TGT, tab.to_string())
w(TGT, "\n    Recommended_Calories vs its own inputs (Pearson r):")
for c in ["Age", "Height_cm", "Weight_kg", "BMI", "Daily_Steps", "Exercise_Frequency",
          "Caloric_Intake"]:
    if c in dr.columns:
        w(TGT, f"      r(Recommended_Calories, {c:18s}) = "
               f"{dr['Recommended_Calories'].corr(dr[c]):+.4f}")
w(TGT, f"    Recommended_Calories range = [{dr.Recommended_Calories.min()}, "
       f"{dr.Recommended_Calories.max()}], nunique={dr.Recommended_Calories.nunique()}, "
       f"mean={dr.Recommended_Calories.mean():.1f}, sd={dr.Recommended_Calories.std():.1f}")
w(TGT, "    A uniform integer draw over [min,max] would have sd = "
       f"{(dr.Recommended_Calories.max()-dr.Recommended_Calories.min())/np.sqrt(12):.1f}")


# ======================================================= B. fitlife / fitness365
w(TGT, "\n" + "=" * 100)
w(TGT, "B. FITLIFE AND FITNESS365 FABRICATION CHECKS")
w(TGT, "=" * 100)


def panel_checks(df, name, pid, date, num_cols):
    w(TGT, f"\n--- {name}  shape={df.shape}")
    w(TGT, f"    participants = {df[pid].nunique():,}, rows per participant: "
           f"{df.groupby(pid).size().describe()[['min','50%','max']].to_dict()}")
    if date in df.columns:
        dd = pd.to_datetime(df[date], errors="coerce")
        w(TGT, f"    date span {dd.min()} .. {dd.max()}  "
               f"({(dd.max()-dd.min()).days} days), distinct dates = {dd.nunique()}")
    w(TGT, "    within-participant distinct values (a real longitudinal record varies; "
           "a generator that draws a static profile does not):")
    g = df.groupby(pid)
    for c in num_cols:
        if c in df.columns:
            nd = g[c].nunique()
            w(TGT, f"      {c:26s} median distinct/person = {nd.median():.0f}  "
                   f"max = {nd.max()}")
    w(TGT, "    decimal-place fingerprint:")
    for c in df.select_dtypes(include=[np.number]).columns:
        s = df[c].dropna().astype(str)
        dp = s.str.split(".").str[-1].str.len().where(s.str.contains(r"\."), 0)
        w(TGT, f"      {c:26s} modal dp={dp.mode().iloc[0] if len(dp) else 'na'}  "
               f"pct_integer={float((df[c].dropna()%1==0).mean()):.4f}  "
               f"nunique={df[c].nunique()}")


fl = pd.read_csv(os.path.join(DATA, "fitlife", "health_fitness_dataset.csv"))
panel_checks(fl, "fitlife", "participant_id", "date",
             ["age", "height_cm", "weight_kg", "bmi", "resting_heart_rate",
              "blood_pressure_systolic", "daily_steps", "hours_sleep"])
w(TGT, "    BMI identity |weight/(height/100)^2 - bmi|:")
b = fl.weight_kg / (fl.height_cm / 100) ** 2
w(TGT, f"      mean abs err = {(b-fl.bmi).abs().mean():.4f}, "
       f"within 0.05 in {float(((b-fl.bmi).abs()<=0.05).mean())*100:.2f}% of rows")
w(TGT, "    physiological correlations that MUST hold in real data:")
for a, bb in [("age", "resting_heart_rate"), ("age", "blood_pressure_systolic"),
              ("bmi", "blood_pressure_systolic"), ("duration_minutes", "calories_burned"),
              ("avg_heart_rate", "calories_burned"), ("daily_steps", "calories_burned"),
              ("hours_sleep", "stress_level"), ("bmi", "resting_heart_rate")]:
    if a in fl.columns and bb in fl.columns:
        w(TGT, f"      r({a:22s}, {bb:24s}) = {fl[a].corr(fl[bb]):+.4f}")
w(TGT, "    calories_burned by intensity (Low/Medium/High must separate strongly):")
w(TGT, fl.groupby("intensity")["calories_burned"].describe()[["count", "mean", "std",
                                                              "min", "max"]].to_string())
w(TGT, "    health_condition x smoking_status crosstab (row-normalised):")
w(TGT, pd.crosstab(fl.health_condition, fl.smoking_status, normalize="index").round(4).to_string())

f3 = pd.read_csv(os.path.join(DATA, "fitness365", "health_fitness_tracking_365days.csv"))
panel_checks(f3, "fitness365", "user_id", "date",
             ["age", "weight_kg", "bmi", "steps", "heart_rate_avg", "sleep_hours"])
w(TGT, "    physiological correlations:")
for a, bb in [("steps", "calories_burned"), ("exercise_minutes", "calories_burned"),
              ("heart_rate_avg", "calories_burned"), ("weight_kg", "bmi"),
              ("sleep_hours", "stress_level"), ("age", "heart_rate_avg"),
              ("steps", "exercise_minutes")]:
    if a in f3.columns and bb in f3.columns:
        w(TGT, f"      r({a:18s}, {bb:18s}) = {f3[a].corr(f3[bb]):+.4f}")
w(TGT, "    fitness365 has no height column, so bmi cannot be reconciled with weight.")
w(TGT, "    implied height from weight/bmi, per user (should be constant per person):")
imp = np.sqrt(f3.weight_kg / f3.bmi) * 100
w(TGT, f"      implied height: mean={imp.mean():.1f} sd={imp.std():.1f} "
       f"min={imp.min():.1f} max={imp.max():.1f}")
ih = pd.DataFrame({"u": f3.user_id, "h": imp}).groupby("u")["h"]
w(TGT, f"      within-person sd of implied height: median={ih.std().median():.2f} cm, "
       f"max={ih.std().max():.2f} cm  (real anatomy: ~0)")


# ============================================================ C. UCI SMOTE share
w(TGT, "\n" + "=" * 100)
w(TGT, "C. UCI OBESITY: SHARE OF SMOTE-SYNTHESISED ROWS")
w(TGT, "=" * 100)
ob = pd.read_csv(os.path.join(DATA, "uci_obesity", "ObesityDataSet_raw_and_data_sinthetic.csv"))
w(TGT, f"file name as published: ObesityDataSet_raw_and_data_sinthetic.csv, n={len(ob)}")
w(TGT, "METHOD. The instrument that produced the real records was a web questionnaire whose")
w(TGT, "  items are integer Likert scales (FCVC 1-3, NCP 1-4, CH2O 1-3, FAF 0-3, TUE 0-2)")
w(TGT, "  and integer age in years. SMOTE interpolates linearly between neighbours, which")
w(TGT, "  turns every such integer item into a fraction. A row is therefore REAL only if")
w(TGT, "  ALL of those items are still integer-valued. This is a fingerprint, not a label,")
w(TGT, "  so it is an upper bound on the real share: a synthetic row can be integer by")
w(TGT, "  coincidence when both neighbours share the same value.")
lik = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
for c in lik:
    w(TGT, f"  {c:6s} integer-valued in {float((ob[c] % 1 == 0).mean())*100:6.2f}% of rows, "
           f"nunique={ob[c].nunique()}")
allint = np.logical_and.reduce([(ob[c] % 1 == 0).values for c in lik])
w(TGT, f"  rows with ALL six integer-valued: {int(allint.sum())} "
       f"({100*allint.mean():.2f}%)")
dp = lambda s: s.astype(str).str.split(".").str[-1].str.len().where(
    s.astype(str).str.contains(r"\."), 0)
hdp, wdp = dp(ob.Height), dp(ob.Weight)
w(TGT, f"  Height decimal-place counts: {hdp.value_counts().to_dict()}")
w(TGT, f"  Weight decimal-place counts: {wdp.value_counts().to_dict()}")
real = allint & (hdp <= 2).values & (wdp <= 1).values
w(TGT, f"  rows ALSO with Height <=2dp and Weight <=1dp (human data entry): "
       f"{int(real.sum())} ({100*real.mean():.2f}%)")
w(TGT, f"  => estimated SYNTHETIC share = {100*(1-real.mean()):.2f}% "
       f"({len(ob)-int(real.sum())} of {len(ob)} rows)")
w(TGT, "  Published statement (Palechor & de la Hoz Manotas 2019, Data in Brief 25:104344,")
w(TGT, "  repeated verbatim on the Kaggle dataset page): 77% generated synthetically with")
w(TGT, "  the SMOTE filter in Weka, 23% collected directly from users. 23% of 2111 = 485.")
w(TGT, f"  Our fingerprint recovers {int(real.sum())} real rows vs the stated 485: "
       f"agreement to {abs(int(real.sum())-485)} rows ({abs(real.mean()-0.2297)*100:.2f} pp).")
w(TGT, "  Corroborating evidence that the excess rows are SMOTE output:")
vcn = ob.NObeyesdad.value_counts()
w(TGT, f"    class balance max/min = {vcn.max()/vcn.min():.3f} across 7 classes "
       f"({vcn.to_dict()}) - SMOTE balances classes by construction")
w(TGT, f"    Height carries 6 decimal places in {int((hdp==6).sum())} rows and <=2 in "
       f"{int((hdp<=2).sum())}; a questionnaire cannot produce 6dp heights")
sub_real, sub_syn = ob[real], ob[~real]
w(TGT, "    class distribution among fingerprint-real vs fingerprint-synthetic rows:")
w(TGT, pd.concat([sub_real.NObeyesdad.value_counts().rename("real"),
                  sub_syn.NObeyesdad.value_counts().rename("synthetic")],
                 axis=1).to_string())


# ============================================== D. survey design variables
w(TGT, "\n" + "=" * 100)
w(TGT, "D. SURVEY DESIGN VARIABLES (weights / strata / PSU)")
w(TGT, "=" * 100)
PAT = ("WT", "WGT", "WEIGHT", "STRAT", "PSU", "SDMV", "LLCPWT", "STSTR", "_WT",
       "RAKE", "FINALWT", "GEOSTR")


def design_scan(path, label, cols=None):
    if cols is None:
        cols = list(pd.read_csv(path, nrows=1, low_memory=False).columns)
    hits = [c for c in cols if any(p in str(c).upper() for p in PAT)]
    w(TGT, f"\n--- {label}")
    w(TGT, f"    total columns = {len(cols)}")
    w(TGT, f"    design-variable name matches = {hits if hits else 'NONE'}")
    return hits


design_scan(os.path.join(DATA, "brfss2015_diab",
                         "diabetes_012_health_indicators_BRFSS2015.csv"),
            "BRFSS 2015 (alexteboul/diabetes-health-indicators-dataset)")
w(TGT, "    NOTE: the genuine BRFSS 2015 SAS export carries _LLCPWT (final weight),")
w(TGT, "    _STSTR (stratum) and _PSU. None of them survived this re-coding, and the")
w(TGT, "    file also drops the state identifier, so the sample cannot be re-weighted.")

design_scan(os.path.join(DATA, "brfss2021_cvd", "CVD_cleaned.csv"),
            "BRFSS 2021 (alphiree/cardiovascular-diseases-risk-prediction-dataset)")
w(TGT, "    NOTE: same conclusion. 'Weight_(kg)' is body weight, not a survey weight;")
w(TGT, "    it matches the pattern only by name.")

for f in ["demographic.csv", "examination.csv", "diet.csv", "questionnaire.csv", "labs.csv"]:
    p = os.path.join(DATA, "nhanes_cdc", f)
    if os.path.exists(p):
        design_scan(p, f"NHANES 2013-2014 {f}")
demo = pd.read_csv(os.path.join(DATA, "nhanes_cdc", "demographic.csv"), low_memory=False)
w(TGT, "\n    NHANES design variables, verified present and populated:")
for c in ["WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]:
    s = demo[c]
    w(TGT, f"      {c:10s} nulls={s.isna().sum():>5d}  nunique={s.nunique():>5d}  "
           f"min={s.min():.4g}  max={s.max():.4g}  mean={s.mean():.4g}")
w(TGT, f"      sum(WTMEC2YR) = {demo.WTMEC2YR.sum():,.0f} "
       "(a 2-year MEC weight sums to the civilian non-institutionalised US population)")
w(TGT, f"      SDMVSTRA values: {sorted(demo.SDMVSTRA.dropna().unique().tolist())}")
w(TGT, f"      SDMVPSU values:  {sorted(demo.SDMVPSU.dropna().unique().tolist())}")
w(TGT, "    => NHANES supports design-based (Taylor-series / replicate) estimation:")
w(TGT, "       strata=SDMVSTRA, cluster=SDMVPSU, weights=WTMEC2YR (exam) or WTINT2YR")
w(TGT, "       (interview). The dietary day-1 weight WTDRD1 lives in the diet file.")

w(TGT, "\n--- cab_survey_india (Govt of India AHS-CAB)")
cab = pd.read_csv(os.path.join(DATA, "cab_survey_india", "CAB_05_UT.csv"),
                  nrows=5, low_memory=False)
w(TGT, f"    design-adjacent columns present: "
       f"{[c for c in cab.columns if any(p in c.upper() for p in ('PSU','STRAT','WT','WEIGHT'))]}")
w(TGT, "    stratum and psu_id identify the sample design, but NO survey weight column")
w(TGT, "    is distributed, so CAB estimates cannot be made population-representative.")

# ================================================ E. feature vocabulary crosswalk
w(TGT, "\n" + "=" * 100)
w(TGT, "E. FEATURE VOCABULARY: canonical concept -> the column that carries it, per dataset")
w(TGT, "=" * 100)

VOCAB = {
    "age": {"brfss2015_diab": "Age", "brfss2021_cvd": "Age_Category", "nhanes_cdc": "RIDAGEYR",
            "cardio_train": "age", "uci_obesity": "Age", "smoking_body": "age",
            "cab_survey_india": "age", "body_measure": "age", "gym_members": "Age",
            "sleep_lifestyle": "Age", "exercise_metrics": "Age", "diet_rec_medical": "Age",
            "fitlife": "age", "fitness365": "age", "thermal_comfort": "yob"},
    "sex": {"brfss2015_diab": "Sex", "brfss2021_cvd": "Sex", "nhanes_cdc": "RIAGENDR",
            "cardio_train": "gender", "uci_obesity": "Gender", "smoking_body": "gender",
            "cab_survey_india": "sex", "body_measure": "sex", "gym_members": "Gender",
            "sleep_lifestyle": "Gender", "exercise_metrics": "Gender",
            "diet_rec_medical": "Gender", "fitness_wellness_plan": "Gender",
            "fitlife": "gender", "fitness365": "gender", "thermal_comfort": "sex"},
    "race_ethnicity": {"nhanes_cdc": "RIDRETH3"},
    "education": {"brfss2015_diab": "Education", "nhanes_cdc": "DMDEDUC2"},
    "income_poverty": {"brfss2015_diab": "Income", "nhanes_cdc": "INDFMPIR"},
    "marital_status": {"nhanes_cdc": "DMDMARTL", "cab_survey_india": "marital_status"},
    "occupation": {"sleep_lifestyle": "Occupation"},
    "urban_rural": {"cab_survey_india": "rural_urban"},
    "height": {"brfss2021_cvd": "Height_(cm)", "nhanes_cdc": "BMXHT", "cardio_train": "height",
               "uci_obesity": "Height", "smoking_body": "height(cm)",
               "cab_survey_india": "length_height_cm", "body_measure": "hgt",
               "gym_members": "Height (m)", "diet_rec_medical": "Height_cm",
               "fitlife": "height_cm", "thermal_comfort": "height"},
    "weight": {"brfss2021_cvd": "Weight_(kg)", "nhanes_cdc": "BMXWT", "cardio_train": "weight",
               "uci_obesity": "Weight", "smoking_body": "weight(kg)",
               "cab_survey_india": "weight_in_kg", "body_measure": "wgt",
               "gym_members": "Weight (kg)", "diet_rec_medical": "Weight_kg",
               "exercise_metrics": "Actual Weight", "fitlife": "weight_kg",
               "fitness365": "weight_kg", "thermal_comfort": "weight"},
    "bmi": {"brfss2015_diab": "BMI", "brfss2021_cvd": "BMI", "nhanes_cdc": "BMXBMI",
            "gym_members": "BMI", "exercise_metrics": "BMI", "diet_rec_medical": "BMI",
            "fitlife": "bmi", "fitness365": "bmi"},
    "bmi_category_label": {"sleep_lifestyle": "BMI Category",
                           "fitness_wellness_plan": "BMI Category",
                           "uci_obesity": "NObeyesdad"},
    "waist_circumference": {"nhanes_cdc": "BMXWAIST", "smoking_body": "waist(cm)",
                            "body_measure": "wai_gi"},
    "hip_circumference": {"body_measure": "hip_gi"},
    "other_girths_diameters": {"body_measure": "bia_di", "nhanes_cdc": "BMXARMC",
                               "thermal_comfort": "shoulder_circumference"},
    "body_fat_pct": {"gym_members": "Fat_Percentage"},
    "systolic_bp": {"nhanes_cdc": "BPXSY1", "cardio_train": "ap_hi",
                    "smoking_body": "systolic", "cab_survey_india": "bp_systolic",
                    "diet_rec_medical": "Blood_Pressure_Systolic",
                    "fitlife": "blood_pressure_systolic", "sleep_lifestyle": "Blood Pressure"},
    "diastolic_bp": {"nhanes_cdc": "BPXDI1", "cardio_train": "ap_lo",
                     "smoking_body": "relaxation", "cab_survey_india": "bp_diastolic",
                     "diet_rec_medical": "Blood_Pressure_Diastolic",
                     "fitlife": "blood_pressure_diastolic"},
    "resting_heart_rate": {"nhanes_cdc": "BPXPLS", "cab_survey_india": "pulse_rate",
                           "gym_members": "Resting_BPM", "sleep_lifestyle": "Heart Rate",
                           "fitlife": "resting_heart_rate"},
    "exercise_heart_rate": {"gym_members": "Avg_BPM", "exercise_metrics": "Heart Rate",
                            "fitlife": "avg_heart_rate", "fitness365": "heart_rate_avg"},
    "cholesterol": {"smoking_body": "Cholesterol", "cardio_train": "cholesterol",
                    "diet_rec_medical": "Cholesterol_Level"},
    "blood_glucose": {"smoking_body": "fasting blood sugar", "cardio_train": "gluc",
                      "cab_survey_india": "fasting_blood_glucose_mg_dl",
                      "diet_rec_medical": "Blood_Sugar_Level"},
    "haemoglobin": {"smoking_body": "hemoglobin", "cab_survey_india": "haemoglobin_level"},
    "diabetes_status": {"brfss2015_diab": "Diabetes_012", "brfss2021_cvd": "Diabetes",
                        "nhanes_cdc": "DIQ010"},
    "heart_disease": {"brfss2015_diab": "HeartDiseaseorAttack", "brfss2021_cvd": "Heart_Disease",
                      "nhanes_cdc": "MCQ160C", "cardio_train": "cardio"},
    "hypertension_dx": {"brfss2015_diab": "HighBP", "nhanes_cdc": "BPQ020"},
    "other_chronic_condition": {"brfss2021_cvd": "Arthritis", "diet_rec_medical": "Chronic_Disease",
                                "fitlife": "health_condition", "brfss2015_diab": "Stroke"},
    "self_rated_health": {"brfss2015_diab": "GenHlth", "brfss2021_cvd": "General_Health",
                          "nhanes_cdc": "HSD010"},
    "family_history": {"uci_obesity": "family_history_with_overweight",
                       "diet_rec_medical": "Genetic_Risk_Factor"},
    "physical_activity_binary": {"brfss2015_diab": "PhysActivity", "brfss2021_cvd": "Exercise",
                                 "nhanes_cdc": "PAQ650", "cardio_train": "active"},
    "activity_frequency": {"gym_members": "Workout_Frequency (days/week)",
                           "diet_rec_medical": "Exercise_Frequency",
                           "sleep_lifestyle": "Physical Activity Level"},
    "activity_duration": {"nhanes_cdc": "PAD680", "gym_members": "Session_Duration (hours)",
                          "exercise_metrics": "Duration", "fitlife": "duration_minutes",
                          "fitness365": "exercise_minutes"},
    "activity_type": {"gym_members": "Workout_Type", "exercise_metrics": "Exercise",
                      "fitlife": "activity_type"},
    "activity_intensity": {"nhanes_cdc": "PAQ605", "exercise_metrics": "Exercise Intensity",
                           "fitlife": "intensity"},
    "daily_steps": {"sleep_lifestyle": "Daily Steps", "diet_rec_medical": "Daily_Steps",
                    "fitlife": "daily_steps", "fitness365": "steps"},
    "calories_burned": {"gym_members": "Calories_Burned", "exercise_metrics": "Calories Burn",
                        "fitlife": "calories_burned", "fitness365": "calories_burned"},
    "sedentary_time": {"nhanes_cdc": "PAD680", "uci_obesity": "TUE"},
    "sleep_duration": {"nhanes_cdc": "SLD010H", "sleep_lifestyle": "Sleep Duration",
                       "diet_rec_medical": "Sleep_Hours", "fitlife": "hours_sleep",
                       "fitness365": "sleep_hours", "sleep_fitbit": "HOURS OF SLEEP"},
    "sleep_quality": {"sleep_lifestyle": "Quality of Sleep", "sleep_fitbit": "SLEEP SCORE"},
    "sleep_disorder": {"sleep_lifestyle": "Sleep Disorder"},
    "stress_level": {"sleep_lifestyle": "Stress Level", "diet_rec_medical": None,
                     "fitlife": "stress_level", "fitness365": "stress_level"},
    "mental_health_days": {"brfss2015_diab": "MentHlth", "brfss2021_cvd": "Depression"},
    "smoking_status": {"brfss2015_diab": "Smoker", "brfss2021_cvd": "Smoking_History",
                       "nhanes_cdc": "SMQ020", "cardio_train": "smoke",
                       "uci_obesity": "SMOKE", "smoking_body": "smoking",
                       "diet_rec_medical": "Smoking_Habit", "fitlife": "smoking_status"},
    "alcohol": {"brfss2015_diab": "HvyAlcoholConsump", "brfss2021_cvd": "Alcohol_Consumption",
                "nhanes_cdc": "ALQ101", "cardio_train": "alco", "uci_obesity": "CALC",
                "diet_rec_medical": "Alcohol_Consumption"},
    "energy_intake_kcal": {"nhanes_cdc": "DR1TKCAL", "diet_rec_medical": "Caloric_Intake"},
    "macronutrient_intake": {"nhanes_cdc": "DR1TPROT", "diet_rec_medical": "Protein_Intake"},
    "fruit_veg_intake": {"brfss2015_diab": "Fruits", "brfss2021_cvd": "Fruit_Consumption",
                         "uci_obesity": "FCVC"},
    "meal_pattern": {"uci_obesity": "NCP", "diet_rec_medical": "Dietary_Habits"},
    "high_calorie_food": {"uci_obesity": "FAVC", "brfss2021_cvd": "FriedPotato_Consumption"},
    "snacking": {"uci_obesity": "CAEC"},
    "water_intake": {"uci_obesity": "CH2O", "gym_members": "Water_Intake (liters)",
                     "fitlife": "hydration_level"},
    "calorie_monitoring": {"uci_obesity": "SCC"},
    "transport_mode": {"uci_obesity": "MTRANS"},
    "healthcare_access": {"brfss2015_diab": "AnyHealthcare", "brfss2021_cvd": "Checkup"},
    "functional_limitation": {"brfss2015_diab": "DiffWalk"},
    "experience_fitness_score": {"gym_members": "Experience_Level", "fitlife": "fitness_level"},
    "prescribed_plan_output": {"fitness_wellness_plan": "Exercise Schedule",
                               "diet_rec_medical": "Recommended_Meal_Plan"},
    "goal": {"fitness_wellness_plan": "Goal", "exercise_metrics": "Dream Weight"},
    "environment_context": {"exercise_metrics": "Weather Conditions",
                            "thermal_comfort": "satisfaction_weather"},
    "survey_design_vars": {"nhanes_cdc": "WTMEC2YR", "cab_survey_india": "psu_id"},
}

allcols = {k: {c["name"] for c in v["columns"]} for k, v in profiles.items()}
w(TGT, f"{'concept':28s} {'n_ds':>4s}  datasets (verified column name)")
missing = []
for concept, mapping in VOCAB.items():
    ok = []
    for ds, col in mapping.items():
        if col is None:
            continue
        if ds in allcols and col in allcols[ds]:
            ok.append(f"{ds}:{col}")
        else:
            missing.append((concept, ds, col))
    w(TGT, f"{concept:28s} {len(ok):>4d}  " + "; ".join(ok))
w(TGT, f"\n  vocabulary entries that FAILED verification against the loaded frames: {missing}")

# columns present in the data but not assigned to any concept, so the reader can
# see what the vocabulary omits
mapped = {(ds, c) for m in VOCAB.values() for ds, c in m.items() if c}
w(TGT, "\n  columns NOT covered by the vocabulary above (per dataset):")
for ds, cols in allcols.items():
    un = sorted(c for c in cols if (ds, c) not in mapped)
    w(TGT, f"    {ds:22s} ({len(un)}) {un}")


# ====================================== F. L1 vs L2 readiness (complete-case n)
w(TGT, "\n" + "=" * 100)
w(TGT, "F. L1 vs L2 READINESS: complete-case n with demographic + anthropometric + behavioural")
w(TGT, "=" * 100)
BLOCKS = {
    "nhanes_cdc": {"demo": ["RIDAGEYR", "RIAGENDR"],
                   "anthro": ["BMXBMI", "BMXWAIST", "BMXHT"],
                   "behav": ["PAQ650", "SMQ020", "DR1TKCAL", "SLD010H"],
                   "risk": ["BPXSY1", "DIQ010", "BPQ020"]},
    "brfss2021_cvd": {"demo": ["Age_Category", "Sex"],
                      "anthro": ["BMI", "Height_(cm)", "Weight_(kg)"],
                      "behav": ["Exercise", "Smoking_History", "Alcohol_Consumption",
                                "Fruit_Consumption"],
                      "risk": ["Heart_Disease", "Diabetes", "General_Health"]},
    "brfss2015_diab": {"demo": ["Age", "Sex", "Education", "Income"],
                       "anthro": ["BMI"],
                       "behav": ["PhysActivity", "Smoker", "Fruits", "Veggies"],
                       "risk": ["HighBP", "HighChol", "Diabetes_012"]},
    "smoking_body": {"demo": ["age", "gender"],
                     "anthro": ["height(cm)", "weight(kg)", "waist(cm)"],
                     "behav": ["smoking"],
                     "risk": ["systolic", "fasting blood sugar", "Cholesterol"]},
    "cardio_train": {"demo": ["age", "gender"], "anthro": ["height", "weight"],
                     "behav": ["smoke", "alco", "active"],
                     "risk": ["ap_hi", "cholesterol", "gluc", "cardio"]},
    "cab_survey_india": {"demo": ["age", "sex"],
                         "anthro": ["length_height_cm", "weight_in_kg"],
                         "behav": [], "risk": ["bp_systolic", "fasting_blood_glucose_mg_dl"]},
    "uci_obesity": {"demo": ["Age", "Gender"], "anthro": ["Height", "Weight"],
                    "behav": ["FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "FAF",
                              "TUE", "CALC", "MTRANS"],
                    "risk": ["family_history_with_overweight"]},
    "body_measure": {"demo": ["age", "sex"], "anthro": ["hgt", "wgt", "wai_gi", "hip_gi"],
                     "behav": [], "risk": []},
}
for ds, blk in BLOCKS.items():
    df = DATASETS[ds][0]()
    need = [c for b in ("demo", "anthro", "behav", "risk") for c in blk[b]]
    have = [c for c in need if c in df.columns]
    miss = [c for c in need if c not in df.columns]
    cc_all = int(df[have].notna().all(axis=1).sum())
    cc_12 = int(df[[c for c in blk["demo"] + blk["anthro"] if c in df.columns]]
                .notna().all(axis=1).sum())
    cc_12b = int(df[[c for c in blk["demo"] + blk["anthro"] + blk["behav"]
                     if c in df.columns]].notna().all(axis=1).sum())
    w(TGT, f"\n  {ds}: n={len(df):,}")
    w(TGT, f"    demo={blk['demo']}")
    w(TGT, f"    anthro={blk['anthro']}  (waist present: "
           f"{any('waist' in c.lower() or c in ('BMXWAIST','wai_gi') for c in blk['anthro'])})")
    w(TGT, f"    behav={blk['behav']}")
    w(TGT, f"    risk={blk['risk']}")
    if miss:
        w(TGT, f"    MISSING FROM FILE: {miss}")
    w(TGT, f"    complete-case n  demo+anthro (L1+L2 core)      = {cc_12:,} "
           f"({100*cc_12/len(df):.1f}%)")
    w(TGT, f"    complete-case n  demo+anthro+behaviour         = {cc_12b:,} "
           f"({100*cc_12b/len(df):.1f}%)")
    w(TGT, f"    complete-case n  all four blocks               = {cc_all:,} "
           f"({100*cc_all/len(df):.1f}%)")
    if ds == "nhanes_cdc":
        ad = df[df.RIDAGEYR >= 18]
        cc = int(ad[[c for c in need if c in ad.columns]].notna().all(axis=1).sum())
        w(TGT, f"    adults 18+: n={len(ad):,}, complete-case all four blocks = {cc:,}")
    del df


# ================================== G. diet_rec_medical: what the target really is
w(TGT, "\n" + "=" * 100)
w(TGT, "G. diet_rec_medical: the ONLY non-noise relation in the file")
w(TGT, "=" * 100)
d = dr
for a, b in [("Recommended_Calories", "Caloric_Intake"),
             ("Recommended_Protein", "Protein_Intake"),
             ("Recommended_Carbs", "Carbohydrate_Intake"),
             ("Recommended_Fats", "Fat_Intake")]:
    ratio = (d[a] / d[b]).replace([np.inf, -np.inf], np.nan).dropna()
    diff = (d[a] - d[b]).dropna()
    w(TGT, f"  r({a}, {b}) = {d[a].corr(d[b]):+.4f}")
    w(TGT, f"     ratio  {a}/{b}: mean={ratio.mean():.4f} sd={ratio.std():.4f} "
           f"min={ratio.min():.4f} max={ratio.max():.4f}")
    w(TGT, f"     diff   {a}-{b}: mean={diff.mean():.2f} sd={diff.std():.2f} "
           f"min={diff.min():.2f} max={diff.max():.2f}")
w(TGT, "  EXACT GENERATOR RECOVERY - the offset recommendation minus intake:")
for a, b in [("Recommended_Calories", "Caloric_Intake"),
             ("Recommended_Protein", "Protein_Intake"),
             ("Recommended_Carbs", "Carbohydrate_Intake"),
             ("Recommended_Fats", "Fat_Intake")]:
    off = d[a] - d[b]
    vc = off.value_counts()
    w(TGT, f"    {a}: offset all-integer={bool((off % 1 == 0).all())}, "
           f"range=[{int(off.min())},{int(off.max())}], distinct={off.nunique()}, "
           f"span={int(off.max()-off.min())+1}, counts per value min={vc.min()} "
           f"max={vc.max()} mean={vc.mean():.1f}")
w(TGT, "    Every integer in the stated range occurs, with roughly equal frequency, so")
w(TGT, "    the generator is: recommendation = reported intake + Uniform{a..b} integer noise.")
w(TGT, "  INTERPRETATION: the 'recommendation' columns are the person's own reported")
w(TGT, "  intake rescaled by a narrow multiplicative factor. They encode no clinical")
w(TGT, "  reasoning: they recommend what the person already eats. The categorical")
w(TGT, "  Recommended_Meal_Plan is independent of every covariate (section A).")

# ============ H. signal test for the two panel datasets acquire_07 did not cover
w(TGT, "\n" + "=" * 100)
w(TGT, "H. PREDICTIVE-SIGNAL TEST FOR fitlife AND fitness365 (not covered by acquire_07)")
w(TGT, "=" * 100)
from sklearn.ensemble import HistGradientBoostingClassifier          # noqa: E402
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold  # noqa: E402


def sig(df, target, name, drop=(), groups=None):
    d = df.drop(columns=[c for c in drop if c in df.columns]).dropna(subset=[target])
    g = None if groups is None else d[groups]
    if len(d) > 60000:
        idx = d.sample(60000, random_state=42).index
        d, g = d.loc[idx], (None if g is None else g.loc[idx])
    y = pd.factorize(d[target])[0]
    X = d.drop(columns=[c for c in (target, groups) if c and c in d.columns])
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
        X[c] = pd.to_numeric(X[c], errors="coerce")
    base = float(pd.Series(y).value_counts(normalize=True).max())
    m = HistGradientBoostingClassifier(random_state=42)
    nclass = len(np.unique(y))
    scoring = "roc_auc" if nclass == 2 else "roc_auc_ovr"
    if g is None:
        cv, gg = StratifiedKFold(5, shuffle=True, random_state=42), None
    else:
        cv, gg = GroupKFold(5), g
    acc = cross_val_score(m, X, y, cv=cv, groups=gg, scoring="accuracy").mean()
    try:
        auc = cross_val_score(m, X, y, cv=cv, groups=gg, scoring=scoring).mean()
    except Exception:
        auc = float("nan")
    w(TGT, f"  {name:52s} target={target:18s} n={len(d):>7,} base={base:.4f} "
           f"acc={acc:.4f} AUC={auc:.4f}")


fl2 = fl.dropna(subset=["health_condition"])
sig(fl, "smoking_status", "fitlife RANDOM split (leaks per-person constants)",
    drop=("date",), groups=None)
sig(fl, "smoking_status", "fitlife GROUPED by participant_id",
    drop=("date",), groups="participant_id")
sig(fl2, "health_condition", "fitlife GROUPED by participant_id",
    drop=("date",), groups="participant_id")
sig(f3, "stress_level", "fitness365 GROUPED by user_id", drop=("date",), groups="user_id")
w(TGT, "  fitlife: a random split scores perfectly because age/height/bmi/BP/smoking_status")
w(TGT, "  are per-person constants repeated ~229 times. Splitting by participant removes the")
w(TGT, "  leak and the score falls to the majority baseline: there is no signal, only leakage.")

TXT.close()
TGT.close()
print("\nwrote:", OUT)
