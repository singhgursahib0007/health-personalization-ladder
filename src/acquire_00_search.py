"""
acquire_00_search.py
Broad Kaggle search across cross-sectional health/lifestyle terms.
Writes candidates to recon/kaggle_search_raw.json and prints a ranked table.
"""
import json, os, time, urllib.request, urllib.error, urllib.parse

ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
RECON = os.path.join(ROOT, "recon")
os.makedirs(RECON, exist_ok=True)
TOKEN = os.environ.get("KAGGLE_TOKEN", "")

TERMS = [
    "physical activity survey", "lifestyle health", "obesity levels",
    "fitness assessment", "gym members exercise tracking", "body measurements",
    "body fat", "cardiorespiratory fitness", "VO2 max", "exercise habits",
    "diet and lifestyle", "sleep health lifestyle", "health and fitness survey",
    "metabolic syndrome", "diabetes lifestyle", "cardiovascular risk factors",
    "nutrition intake", "BRFSS", "NHANES", "obesity eating habits",
    "health survey", "fitness tracker dataset", "exercise recommendation",
    "diet recommendation", "workout plan dataset", "personal health data",
    "anthropometric", "waist circumference", "smoking alcohol health",
    "heart disease risk", "wellness dataset", "FRIEND fitness registry",
    "cooper center longitudinal", "treadmill exercise test", "fitness club",
    "calorie intake nutrition", "student health lifestyle", "body composition",
]


def fetch(term, page=1):
    q = urllib.parse.urlencode({"search": term, "fileType": "csv",
                                "pageSize": 50, "page": page})
    url = f"https://www.kaggle.com/api/v1/datasets/list?{q}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    delay = 3.0
    for _ in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            return {"_error": f"http {e.code}"}
        except Exception as e:
            time.sleep(2); return {"_error": type(e).__name__}
    return {"_error": "rate limited"}


def main():
    seen = {}
    for t in TERMS:
        res = fetch(t)
        if isinstance(res, dict):
            print(f"{t:38s} ERROR {res.get('_error')}")
            continue
        for d in res:
            ref = d.get("ref")
            if not ref:
                continue
            rec = seen.setdefault(ref, dict(d))
            rec.setdefault("_terms", [])
            rec["_terms"].append(t)
        print(f"{t:38s} {len(res):3d} results, cumulative {len(seen)}")
        time.sleep(0.4)

    out = os.path.join(RECON, "kaggle_search_raw.json")
    json.dump(list(seen.values()), open(out, "w"), indent=1)
    print(f"\n{len(seen)} unique datasets -> {out}")


if __name__ == "__main__":
    main()
