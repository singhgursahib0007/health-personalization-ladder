"""
acquire_01_rank_candidates.py
=============================
Score the 1518 raw Kaggle hits for likely PER-PERSON LONGITUDINAL structure,
using only the metadata returned by the list endpoint (title/subtitle/ref/size/
usability/votes/downloads). This is a cheap prefilter; nothing is trusted until
the file is actually downloaded and inspected.

Writes recon/kaggle_long_search_ranked.csv
"""
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECON = os.path.join(ROOT, "recon")

# words that imply repeated observation of the same unit over time
LONGI = ["daily", "log", "logs", "diary", "journal", "history", "time series",
         "timeseries", "longitudinal", "tracking", "tracker", "minute",
         "hourly", "per day", "day", "days", "week", "month", "session",
         "record", "activity", "over time", "journey", "streak"]
# words implying a wearable / real device capture
DEVICE = ["fitbit", "garmin", "apple watch", "smartwatch", "wearable",
          "accelerometer", "actigraph", "polar", "whoop", "oura", "strava",
          "mi band", "smart band", "sensor", "pmdata", "lifesnaps", "empatica",
          "huawei", "samsung health", "google fit", "withings"]
# words implying the row is a PERSON not an aggregate
PERSON = ["user", "users", "participant", "participants", "subject", "subjects",
          "individual", "patient", "patients", "member", "athlete", "person",
          "people", "employee", "customer", "id"]
# strong negative: single-shot survey / cross-sectional / obviously generated
NEG = ["synthetic", "simulated", "generated", "dummy", "fake", "mock",
       "artificial", "sample dataset", "for practice", "randomly generated"]
# cross-sectional health-risk profile tables (our prior work: mostly fabricated)
NEG_SOFT = ["prediction dataset", "classification dataset", "risk prediction",
            "disease prediction", "recommendation system dataset",
            "obesity", "diabetes prediction", "stroke", "heart disease",
            "cardiovascular disease", "churn"]


def score(row):
    txt = " ".join(str(row.get(k) or "") for k in
                   ("title", "subtitle", "ref")).lower()
    s = 0
    s += 3 * sum(w in txt for w in DEVICE)
    s += 2 * sum(w in txt for w in LONGI)
    s += 2 * sum(w in txt for w in PERSON)
    s -= 12 * sum(w in txt for w in NEG)
    s -= 4 * sum(w in txt for w in NEG_SOFT)
    # a dataset with many CSVs / decent size is more likely a real export
    try:
        mb = float(row.get("totalBytes") or 0) / 1e6
    except Exception:
        mb = 0
    if 0.05 < mb < 800:
        s += 2
    if mb > 2000:
        s -= 3
    s += min(float(row.get("usabilityRating") or 0) * 4, 4)
    return s


def main():
    raw = json.load(open(os.path.join(RECON, "kaggle_long_search_raw.json")))
    df = pd.DataFrame(raw)
    print("raw cols:", sorted(df.columns.tolist()))
    df["score"] = df.apply(score, axis=1)
    df["mb"] = pd.to_numeric(df.get("totalBytes"), errors="coerce") / 1e6
    keep = ["ref", "title", "subtitle", "score", "mb", "usabilityRating",
            "voteCount", "downloadCount", "lastUpdated", "licenseName",
            "ownerName", "_terms"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].sort_values("score", ascending=False)
    p = os.path.join(RECON, "kaggle_long_search_ranked.csv")
    out.to_csv(p, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(out.head(70).to_string())
    print("\nWROTE", p)


if __name__ == "__main__":
    main()
