"""
acquire_01_triage.py
Rank the 420 search hits for cross-sectional health/lifestyle relevance.
Prints a ranked table; writes recon/kaggle_triage.csv
"""
import json, os, re
import pandas as pd

ROOT = "/Users/macbook/Documents/MyProjects/Greg_Research/recommendation_research"
RECON = os.path.join(ROOT, "recon")
raw = json.load(open(os.path.join(RECON, "kaggle_search_raw.json")))

GOOD = ["obesity", "nhanes", "brfss", "lifestyle", "physical activity", "fitness",
        "body fat", "body measurement", "anthropom", "sleep health", "exercise",
        "diet", "nutrition", "gym", "cardiovascular", "metabolic", "cardio",
        "vo2", "smoking", "alcohol", "waist", "bmi", "health survey",
        "behavioral risk", "wellness", "habit", "calorie", "workout"]
REC = ["recommend", "prescription", "plan", "advice", "suggest"]
BAD = ["image", "png", "jpeg", "audio", "tweet", "sentiment", "stock", "price",
       "covid vaccin", "chest x", "mri", "song", "movie", "amazon", "car "]

rows = []
for d in raw:
    title = (d.get("title") or "")
    sub = (d.get("subtitle") or "")
    ref = d.get("ref")
    hay = (title + " " + sub + " " + ref).lower()
    score = sum(3 for k in GOOD if k in hay)
    score += sum(2 for k in REC if k in hay)
    score -= sum(6 for k in BAD if k in hay)
    score += min(len(d.get("_terms", [])), 5)
    rows.append({
        "ref": ref, "title": title[:70], "subtitle": sub[:70],
        "owner": d.get("ownerName"), "size_mb": round((d.get("totalBytes") or 0) / 1e6, 2),
        "usability": d.get("usabilityRating"), "votes": d.get("voteCount"),
        "downloads": d.get("downloadCount"), "license": d.get("licenseName"),
        "updated": (d.get("lastUpdated") or "")[:10],
        "terms": len(d.get("_terms", [])), "score": score,
        "isrec": int(any(k in hay for k in REC)),
    })

df = pd.DataFrame(rows).sort_values(["score", "votes"], ascending=False)
df.to_csv(os.path.join(RECON, "kaggle_triage.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 62)
print(df.head(70)[["ref", "title", "size_mb", "votes", "downloads", "usability", "score"]].to_string(index=False))
print("\n--- datasets mentioning recommendation/plan/prescription ---")
print(df[df.isrec == 1].head(30)[["ref", "title", "size_mb", "votes", "score"]].to_string(index=False))
