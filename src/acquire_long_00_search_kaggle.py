"""
acquire_00_search_kaggle.py
===========================
Broad search of the Kaggle dataset index for PER-PERSON LONGITUDINAL behavioural
data (wearables, activity trackers, habit/exercise logs).

Writes: recommendation_research/recon/kaggle_long_search_raw.json
        recommendation_research/recon/kaggle_long_search_ranked.csv
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECON = os.path.join(ROOT, "recon")
os.makedirs(RECON, exist_ok=True)
TOKEN = os.environ.get("KAGGLE_TOKEN", "")

TERMS = [
    "fitbit", "smartwatch", "wearable", "activity tracker", "daily steps",
    "sleep tracking", "heart rate", "physical activity longitudinal",
    "exercise log", "workout log", "gym attendance", "habit tracker",
    "calorie log", "food diary", "weight loss journey", "running log",
    "strava", "apple watch", "garmin", "mhealth", "PMData", "LifeSnaps",
    "step count", "accelerometer", "actigraphy", "sleep quality daily",
    "personal health data", "quantified self", "fitness tracker data",
    "daily activity", "wearable sensor", "heart rate variability",
    "sleep diary", "activity diary", "self tracking", "mobile health",
    "digital biomarkers", "physical activity intervention",
    "smart band", "health app usage", "daily habits", "marathon training",
    "cycling log", "swimming log", "steps sleep calories",
    "user activity time series", "wellness tracking",
]


def api_get(path, params, retries=5):
    qs = urllib.parse.urlencode(params)
    url = f"https://www.kaggle.com/api/v1/{path}?{qs}"
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    delay = 3.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            print(f"  http {e.code} for {path} {params}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"  {type(e).__name__} for {path} {params}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return []


def main():
    seen = {}
    # NOTE: the endpoint honours pageSize but caps the returned page at 20 rows,
    # so we page until a short/empty page comes back rather than trusting count.
    for term in TERMS:
        added = 0
        for page in range(1, 6):
            res = api_get("datasets/list",
                          {"search": term, "fileType": "csv",
                           "pageSize": 50, "page": page})
            if not isinstance(res, list) or not res:
                break
            for d in res:
                ref = d.get("ref")
                if not ref:
                    continue
                if ref not in seen:
                    d["_terms"] = []
                    seen[ref] = d
                    added += 1
                seen[ref]["_terms"].append(term)
            if len(res) < 20:
                break
        print(f"{term:42s}  +{added:3d}  total={len(seen)}")
        time.sleep(0.3)

    out = os.path.join(RECON, "kaggle_long_search_raw.json")
    with open(out, "w") as f:
        json.dump(list(seen.values()), f, indent=1)
    print(f"\nWROTE {out}  n={len(seen)}")


if __name__ == "__main__":
    main()
