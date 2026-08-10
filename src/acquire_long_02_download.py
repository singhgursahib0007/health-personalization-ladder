"""
acquire_long_02_download.py
===========================
Download the shortlisted PER-PERSON LONGITUDINAL Kaggle datasets into
recommendation_research/datasets/<short_name>/ and capture their Kaggle
metadata (title, subtitle, owner, description, licence, usability, downloads,
votes, last-updated, member list + sizes) into recon/kaggle_long_metadata.json.

Retry/backoff logic follows scaleup/run_scaled_audit.py::download.

Usage:  python3 acquire_long_02_download.py [short_name ...]
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets")
RECON = os.path.join(ROOT, "recon")
TOKEN = os.environ.get("KAGGLE_TOKEN", "")
META_PATH = os.path.join(RECON, "kaggle_long_metadata.json")

# short_name -> kaggle slug. Ordered by expected value for per-person
# longitudinal modelling.
TARGETS = {
    # --- tier 1: expected real, published, multi-person, many days ---
    "fitbit_mobius":    "josesantibanez/crowd-sourced-fitbit-datasets-4-12-16-5-12-16",
    "psykose":          "stevenhicks/psykose",
    "depresjon":        "nikitamanaenkov/psychiatric-motor-activity-dataset",
    "aw_fb":            "aleespinosa/apple-watch-and-fitbit-data",
    "lifesnaps":        "skywescar/lifesnaps-fitbit-dataset",
    "studentlife":      "dartweichen/student-life",
    # --- tier 2: single-person real logs ---
    "chargehr_1yr":     "alketcecaj/one-year-of-fitbit-chargehr-data",
    "running_log":      "jeffreybraun/running-log-insight",
    "my_applewatch":    "daiearth22/applewatch",
    "strava_personal":  "sc0v1n0/my-strava-activities",
    "sleep_fitbit":     "riinuanslan/sleep-data-from-fitbit-tracker",
    # --- tier 3: large-N but provenance unclear -> fabrication screen ---
    "fitness365":       "waqasishtiaq/fitness",
    "fitlife":          "jijagallery/fitlife-health-and-fitness-tracking-dataset",
    "whoop100k":        "likithagedipudi/whoop-fitness-dataset",
    "habit90":          "uthaya1995/90-day-habit-tracker-for-personal-growth",
    "thermal_comfort":  "claytonmiller/longitudinal-personal-thermal-comfort-preferences",
}

MAX_EXTRACT_BYTES = 900 * 1024 * 1024


def api_json(path, retries=4):
    url = f"https://www.kaggle.com/api/v1/{path}"
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    delay = 3.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(delay); delay = min(delay * 2, 60); continue
            return {"_error": f"http {e.code}"}
        except Exception:
            time.sleep(delay); delay = min(delay * 2, 60)
    return {"_error": "retries exhausted"}


def download(slug, dest):
    """Fetch the dataset zip and extract tabular members. Returns (info, err).

    The full member manifest is always recorded, even when nothing is
    extracted, so the report can state what the archive actually contains.
    """
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    delay = 3.0
    for _ in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            blob = urllib.request.urlopen(req, timeout=1800).read()
            zip_mb = len(blob) / 1e6
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                members = [i for i in z.infolist()
                           if not i.filename.startswith("__") and not i.is_dir()]
                manifest = [{"name": i.filename, "bytes": i.file_size}
                            for i in members]
                tab = [i for i in members
                       if i.filename.lower().endswith((".csv", ".tsv", ".json"))]
                base = {"zip_mb": zip_mb, "n_members": len(members),
                        "manifest": manifest[:400],
                        "manifest_truncated": len(manifest) > 400,
                        "total_uncompressed_mb":
                            sum(m["bytes"] for m in manifest) / 1e6}
                if not tab:
                    return base, "no csv/tsv/json members"
                # csv first, then smallest first, until the extraction cap
                tab.sort(key=lambda i: (not i.filename.lower().endswith(".csv"),
                                        i.file_size))
                total, extracted = 0, []
                for i in tab:
                    if total + i.file_size > MAX_EXTRACT_BYTES:
                        continue
                    z.extract(i, dest)
                    total += i.file_size
                    extracted.append(i.filename)
                base["extracted"] = extracted[:400]
                base["n_extracted"] = len(extracted)
                base["extracted_mb"] = total / 1e6
                return base, None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay); delay = min(delay * 2, 120); continue
            return None, f"http {e.code}"
        except zipfile.BadZipFile:
            return None, "bad zip"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, "rate limited"


def main():
    wanted = sys.argv[1:] or list(TARGETS)
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}

    for short in wanted:
        slug = TARGETS[short]
        dest = os.path.join(DATA, short)
        print(f"\n=== {short}  ({slug}) ===", flush=True)

        view = api_json(f"datasets/view/{slug}")
        rec = {
            "short_name": short, "slug": slug,
            "title": view.get("titleNullable") or view.get("title"),
            "subtitle": view.get("subtitleNullable"),
            "owner": view.get("ownerNameNullable"),
            "creator": view.get("creatorNameNullable"),
            "license": view.get("licenseNameNullable"),
            "usabilityRating": view.get("usabilityRatingNullable"),
            "downloadCount": view.get("downloadCount"),
            "voteCount": view.get("voteCount"),
            "viewCount": view.get("viewCount"),
            "lastUpdated": view.get("lastUpdated"),
            "totalBytes": view.get("totalBytesNullable"),
            "url": view.get("urlNullable"),
            "description": (view.get("descriptionNullable") or "")[:4000],
        }
        print(f"  title      : {rec['title']}")
        print(f"  license    : {rec['license']}   usability={rec['usabilityRating']}")
        print(f"  downloads  : {rec['downloadCount']}  votes={rec['voteCount']}")
        print(f"  totalBytes : {(rec['totalBytes'] or 0)/1e6:.1f} MB")

        if os.path.isdir(dest) and os.listdir(dest):
            print("  already present on disk, skipping fetch")
        else:
            os.makedirs(dest, exist_ok=True)
            t0 = time.time()
            info, err = download(slug, dest)
            if err:
                print(f"  DOWNLOAD ISSUE: {err}")
                rec["download_error"] = err
            if info:
                rec["archive"] = info
                print(f"  zip={info['zip_mb']:.1f}MB  members={info['n_members']}"
                      f"  uncompressed={info['total_uncompressed_mb']:.1f}MB"
                      f"  extracted={info.get('n_extracted', 0)} files"
                      f" ({info.get('extracted_mb', 0):.1f}MB)"
                      f"  in {time.time()-t0:.0f}s")
        meta[short] = rec
        json.dump(meta, open(META_PATH, "w"), indent=1)

    print("\nWROTE", META_PATH)


if __name__ == "__main__":
    main()
