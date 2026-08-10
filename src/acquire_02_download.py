"""
acquire_02_download.py
Download the cross-sectional shortlist into recommendation_research/datasets/<short>/
and capture full Kaggle metadata per dataset into recon/kaggle_metadata.json.
Download implementation adapted from scaleup/run_scaled_audit.py::download.
"""
import io, json, os, sys, time, urllib.error, urllib.request, urllib.parse, zipfile

ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
DATA = os.path.join(ROOT, "datasets")
RECON = os.path.join(ROOT, "recon")
TOKEN = os.environ.get("KAGGLE_TOKEN", "")
os.makedirs(DATA, exist_ok=True); os.makedirs(RECON, exist_ok=True)

# short_name -> (slug, why)
TARGETS = {
    # --- required / real provenance ---
    "uci_obesity":      ("jayitabhattacharyya/estimation-of-obesity-levels-uci-dataset",
                         "UCI ObesityDataSet, Palechor & de la Hoz Manotas 2019 (SMOTE check)"),
    "brfss2021_cvd":    ("alphiree/cardiovascular-diseases-risk-prediction-dataset",
                         "BRFSS 2021 derived per-person: anthro + behaviour + CVD outcome"),
    "brfss2015_diab":   ("alexteboul/diabetes-health-indicators-dataset",
                         "BRFSS 2015 derived per-person: 253k rows, 22 cols"),
    "nhanes_cdc":       ("cdc/national-health-and-nutrition-examination-survey",
                         "Official CDC NHANES mirror"),
    "cab_survey_india": ("rajanand/cab-survey",
                         "Clinical Anthropometric & Bio-Chemical survey, Govt of India AHS"),
    "cardio_train":     ("pirogovskiy/cardio-train",
                         "70k cardiovascular exam records, classic"),
    "body_measure":     ("utkarshx27/body-measurements",
                         "Heinz et al. 2003 body dimensions of physically active adults"),
    "smoking_body":     ("kukuroo3/body-signal-of-smoking",
                         "Korean NHIS health screening: anthro + labs + smoking"),
    "gym_members":      ("valakhorasani/gym-members-exercise-dataset",
                         "Gym Members Exercise Dataset (named in brief)"),
    "sleep_lifestyle":  ("uom190346a/sleep-health-and-lifestyle-dataset",
                         "Sleep Health and Lifestyle Dataset (named in brief)"),
    # --- explicit recommendation/plan columns: screen sceptically ---
    "diet_rec_medical": ("ziya07/personalized-medical-diet-recommendations-dataset",
                         "explicit diet recommendation column"),
    "fitness_wellness_plan": ("ayeshaseherr/juymmm",
                         "Personalized Fitness Goals and Wellness Plans"),
    "exercise_metrics": ("aakashjoshi123/exercise-and-fitness-metrics-dataset",
                         "Exercise & Fitness Metrics, claimed recommendation column"),
}


def api(path):
    req = urllib.request.Request("https://www.kaggle.com/api/v1/" + path,
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    delay = 5.0
    for _ in range(5):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay); delay = min(delay * 2, 120); continue
            return {"_error": f"http {e.code}"}
        except Exception as e:
            return {"_error": type(e).__name__}
    return {"_error": "rate limited"}


def download(slug, dest):
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    delay = 5.0
    for _ in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            blob = urllib.request.urlopen(req, timeout=600).read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                names = [n for n in z.namelist()
                         if n.lower().endswith((".csv", ".xpt", ".txt")) and not n.startswith("__")]
                if not names:
                    return None, f"no csv in archive: {z.namelist()[:6]}"
                for n in names:
                    z.extract(n, dest)
            return dest, None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay); delay = min(delay * 2, 180); continue
            return None, f"http {e.code}"
        except zipfile.BadZipFile:
            return None, "bad zip"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, "rate limited"


def main():
    meta_out = {}
    only = sys.argv[1:] or list(TARGETS)
    for short in only:
        slug, why = TARGETS[short]
        dest = os.path.join(DATA, short)
        # metadata
        m = api(f"datasets/view/{slug}")
        meta_out[short] = {"slug": slug, "why": why, "view": m}
        files = api(f"datasets/list/files/{slug}")
        meta_out[short]["files"] = files
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"{short:24s} already present")
        else:
            os.makedirs(dest, exist_ok=True)
            p, err = download(slug, dest)
            print(f"{short:24s} {slug:62s} {'OK' if p else 'FAIL ' + str(err)}")
            time.sleep(2)
        got = []
        for r, _, fs in os.walk(dest):
            for f in fs:
                got.append((os.path.relpath(os.path.join(r, f), dest),
                            os.path.getsize(os.path.join(r, f))))
        meta_out[short]["local_files"] = got
        print("   ", [(a, round(b / 1e6, 2)) for a, b in got][:12])

    prev = {}
    mp = os.path.join(RECON, "kaggle_metadata.json")
    if os.path.exists(mp):
        prev = json.load(open(mp))
    prev.update(meta_out)
    json.dump(prev, open(mp, "w"), indent=1, default=str)
    print("\nmetadata ->", mp)


if __name__ == "__main__":
    main()
