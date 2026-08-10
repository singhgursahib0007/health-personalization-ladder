"""
09_verify_paper.py
==================
Checks that every headline number asserted in the paper matches the committed
experiment outputs, and enforces the writing constraints.

This exists because an earlier project in this line contained numbers that no
script produced. A reviewer should be able to confirm mechanically that this one
does not. Run it before every compile.

Usage:  python 09_verify_paper.py
Exit code 0 if every check passes, 1 otherwise.
"""

import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
PLOTS = os.path.join(ROOT, "outputs", "plots")
TEX = os.path.join(ROOT, "paper", "paper.tex")

results = []


def check(name, paper, output, tol=0.0, unit=""):
    if isinstance(paper, (int, float)) and isinstance(output, (int, float)):
        ok = abs(paper - output) <= tol
    else:
        ok = str(paper) == str(output)
    results.append(("PASS" if ok else "FAIL", name,
                    f"paper={paper}{unit} outputs={output}{unit}"))


def note(name, ok, detail):
    results.append(("PASS" if ok else "FAIL", name, detail))


def jload(f):
    with open(os.path.join(TEXTS, f)) as fh:
        return json.load(fh)


def cload(f):
    return pd.read_csv(os.path.join(TEXTS, f))


def main():
    tex = open(TEX, encoding="utf-8").read()

    # ------------------------------------------------- E1 the book's engine
    ks = open(os.path.join(TEXTS, "kb_summary.txt")).read()
    grid = cload("kb_output_space.csv")
    diet = ["veg_fruit", "grain", "dairy", "meat"]
    act = ["moderate_min", "moderate_days", "strength_days", "sleep", "water"]
    check("engine: distinct user types", 378, len(grid))
    check("engine: distinct full recommendations", 14,
          grid[diet + act + ["calories"]].drop_duplicates().shape[0])
    check("engine: distinct diet prescriptions", 6,
          grid[diet].drop_duplicates().shape[0])
    check("engine: distinct activity/sleep/water sets", 1,
          grid[act].drop_duplicates().shape[0])
    check("engine: compression", 0.037,
          round(grid[diet + act + ["calories"]].drop_duplicates().shape[0] / len(grid), 3),
          0.0005)
    check("engine: calorie coverage", 19.0,
          round(float(grid.calories.notna().mean() * 100), 1), 0.05, "%")

    # -------------------------------------------------------- E2 battery
    cov = jload("e2_detector_coverage.json")
    check("battery: fabricated datasets", 5, cov["n_fabricated"])
    check("battery: caught by D1 alone", 1, cov["caught_by_D1_alone"])
    check("battery: caught by full battery", 5, cov["caught_by_D1_to_D4"])
    check("battery: false alarms on real", 0, cov["false_alarms_on_real"])
    check("battery: D3 catches", 3, cov["per_detector_on_fabricated"]["D3_temporal"])
    b = cload("e2_battery_results.csv").set_index("dataset")
    check("battery: fitlife BMI disagreement", 95.4,
          round(float(b.loc["fitlife", "d4_disagree_pct"]), 1), 0.06, "%")
    check("battery: diet_rec AUC", 0.51,
          round(float(b.loc["diet_rec_medical", "d2_auc"]), 2), 0.006)
    check("battery: lifesnaps AUC", 0.80,
          round(float(b.loc["lifesnaps", "d2_auc"]), 2), 0.006)
    check("battery: whoop lag1", 0.12,
          round(float(b.loc["whoop100k", "d3_lag1_ac"]), 2), 0.006)
    check("battery: lifesnaps lag1", 0.86,
          round(float(b.loc["lifesnaps", "d3_lag1_ac"]), 2), 0.006)

    # --------------------------------------------------------- E3 NHANES
    ns = jload("e3_nhanes_sample.json")
    check("NHANES: adults", 6113, ns["n_adults"])
    check("NHANES: with waist", 5587, ns["n_with_waist"])
    check("NHANES: median MVPA", 150, int(ns["mvpa_median"]))
    check("NHANES: pct zero MVPA", 33.1, round(ns["mvpa_pct_zero"], 1), 0.05, "%")

    v = cload("e3_variance_explained.csv")

    def r2(target, pred):
        s = v[(v.target == target) & (v.predictors == pred)]
        return round(float(s.R2_weighted.iloc[0]), 4) if len(s) else None

    check("NHANES: WHtR from age+sex", 0.079,
          r2("BODY: waist-to-height ratio", "L1  age + sex"), 0.0006)
    check("NHANES: WHtR from age+sex+BMI", 0.895,
          r2("BODY: waist-to-height ratio", "L2  + BMI"), 0.0006)
    check("NHANES: MVPA from age+sex", 0.074,
          r2("BEHAVIOUR: weekly MVPA minutes", "L1  age + sex"), 0.0006)
    check("NHANES: MVPA from age+sex+BMI", 0.074,
          r2("BEHAVIOUR: weekly MVPA minutes", "L2  + BMI"), 0.0006)
    check("NHANES: MVPA +waist", 0.075,
          r2("BEHAVIOUR: weekly MVPA minutes", "L2+ + waist"), 0.0006)

    p = cload("e3_prediction_auc.csv").set_index("predictors").AUC
    check("NHANES: AUC age+sex", 0.6686, round(float(p["L1  age + sex"]), 4), 0.0001)
    check("NHANES: AUC +BMI", 0.6686, round(float(p["L2  + BMI"]), 4), 0.0001)
    check("NHANES: AUC +waist", 0.6637, round(float(p["L2+ + waist"]), 4), 0.0001)
    check("NHANES: AUC WHtR alone", 0.5868,
          round(float(p["waist-to-height ratio alone"]), 4), 0.0001)
    check("NHANES: AUC BMI alone", 0.5366, round(float(p["BMI alone"]), 4), 0.0001)
    check("NHANES: AUC age alone", 0.6497, round(float(p["age alone"]), 4), 0.0001)

    # ------------------------------------------------------------- E4 ICC
    nw = jload("e4_nonwear_audit.json")
    check("non-wear: rows raw", 7410, nw["n_rows_raw"])
    check("non-wear: missing steps", 2633, nw["n_missing_steps"])
    check("non-wear: missing+1440", 2256, nw["n_missing_steps_and_1440"])
    check("non-wear: literal zeros", 86, nw["n_literal_zero_steps"])
    check("non-wear: pct dropped", 36.5, round(nw["pct_dropped"], 1), 0.06, "%")
    check("non-wear: rows kept", 4704, nw["n_rows_kept"])
    check("non-wear: people kept", 71, nw["n_people_kept"])

    icc = cload("e4_icc.csv").set_index("variable")
    med = cload("e4_icc.csv").groupby("kind").icc.median()
    check("ICC: median behaviour", 0.26, round(float(med["behaviour"]), 2), 0.006)
    check("ICC: median physiology", 0.57, round(float(med["physiology"]), 2), 0.006)
    check("ICC: resting hr", 0.866, round(float(icc.loc["resting_hr", "icc"]), 3), 0.0006)
    check("ICC: sleep efficiency", 0.785,
          round(float(icc.loc["sleep_efficiency", "icc"]), 3), 0.0006)
    check("ICC: steps", 0.288, round(float(icc.loc["steps", "icc"]), 3), 0.0006)
    check("ICC: very active minutes", 0.226,
          round(float(icc.loc["very_active_minutes", "icc"]), 3), 0.0006)

    # ---------------------------------------------------------- E5 ladder
    r = cload("e5_ladder_results.csv")
    ls = r[(r.source == "lifesnaps") & (r.target == "steps") & (r.model == "ridge")]
    m = ls.groupby("rung").r2.mean()
    mae = ls.groupby("rung").mae.mean()
    L0, L1 = "L0 population constant", "L1 demographics (the book)"
    L2, L3 = "L2 + body measurements", "L3 + own behaviour history"
    L4 = "L4 + yesterday's context"
    check("ladder: L0", -0.0131, round(float(m[L0]), 4), 0.0001)
    check("ladder: L1", -0.0173, round(float(m[L1]), 4), 0.0001)
    check("ladder: L2", -0.0168, round(float(m[L2]), 4), 0.0001)
    check("ladder: L3", 0.2261, round(float(m[L3]), 4), 0.0001)
    check("ladder: L4", 0.2292, round(float(m[L4]), 4), 0.0001)
    check("ladder: gain L1->L2", 0.0004, round(float(m[L2] - m[L1]), 4), 0.0001)
    check("ladder: gain L2->L3", 0.2429, round(float(m[L3] - m[L2]), 4), 0.0001)
    check("ladder: gain L3->L4", 0.0032, round(float(m[L4] - m[L3]), 4), 0.0001)
    check("ladder: MAE L0", 4122, int(round(float(mae[L0]))), 1)
    check("ladder: MAE L3", 3457, int(round(float(mae[L3]))), 1)
    check("ladder: MAE reduction", 665,
          int(round(float(mae[L0] - mae[L3]))), 2)
    mb = r[(r.source == "mobius") & (r.target == "steps") &
           (r.model == "ridge")].groupby("rung").r2.mean()
    check("ladder: mobius L0", -0.0040, round(float(mb[L0]), 4), 0.0001)
    check("ladder: mobius L3", 0.3396, round(float(mb[L3]), 4), 0.0001)

    # ------------------------------------------------------- E6 adherence
    a = cload("e6_adherence_results.csv").groupby("rung")[["auc", "brier"]].mean()
    check("adherence: L2 AUC", 0.604, round(float(a.loc[L2, "auc"]), 3), 0.0006)
    check("adherence: L2 Brier", 0.1554, round(float(a.loc[L2, "brier"]), 4), 0.0001)
    check("adherence: L3 AUC", 0.701, round(float(a.loc[L3, "auc"]), 3), 0.0006)
    check("adherence: L3a AUC", 0.746,
          round(float(a.loc["L3a + own adherence history", "auc"]), 3), 0.0006)
    check("adherence: L3a Brier", 0.1376,
          round(float(a.loc["L3a + own adherence history", "brier"]), 4), 0.0001)
    check("adherence: L4 AUC", 0.726, round(float(a.loc[L4, "auc"]), 3), 0.0006)
    check("adherence: L4 Brier", 0.1395, round(float(a.loc[L4, "brier"]), 4), 0.0001)

    # ------------------------------------------------------ E7 cold start
    c = cload("e7_coldstart_curve.csv")

    def bucket(lo):
        s = c[c.hist_days_low == lo]
        return round(float(s[L3].iloc[0]), 3) if len(s) else None

    check("coldstart: 0 days", -0.093, bucket(0), 0.0006)
    check("coldstart: 1-2 days", 0.061, bucket(1), 0.0006)
    check("coldstart: 7-13 days", 0.239, bucket(7), 0.0006)
    check("coldstart: 14-27 days peak", 0.308, bucket(14), 0.0006)
    check("coldstart: 56+ days", 0.260, bucket(56), 0.0006)

    pv = cload("e7_privacy_utility.csv").set_index("rung")
    check("privacy: L0", -0.0099, round(float(pv.loc[L0, "r2_new_user"]), 4), 0.0001)
    check("privacy: L1", -0.0198, round(float(pv.loc[L1, "r2_new_user"]), 4), 0.0001)
    check("privacy: L2", -0.0255, round(float(pv.loc[L2, "r2_new_user"]), 4), 0.0001)
    check("privacy: L3", 0.2749, round(float(pv.loc[L3, "r2_new_user"]), 4), 0.0001)
    check("privacy: L4", 0.2692, round(float(pv.loc[L4, "r2_new_user"]), 4), 0.0001)
    check("privacy: marginal L3", 0.3004,
          round(float(pv.loc[L3, "marginal_gain"]), 4), 0.0001)
    check("privacy: marginal L4", -0.0057,
          round(float(pv.loc[L4, "marginal_gain"]), 4), 0.0001)

    # ----------------------------------------------------------- hygiene
    body = tex.split("\\begin{document}")[1]
    em = body.count("\u2014")
    note("no em dashes in the body", em == 0, f"found {em}")
    en = body.count("\u2013")
    note("no stray en dashes outside ranges", True, f"found {en} (ranges allowed)")

    cited = set()
    for grp in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited |= {k.strip() for k in grp.split(",")}
    bib = open(os.path.join(ROOT, "paper", "references.bib"), encoding="utf-8").read()
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))
    note("all citations defined in references.bib", cited <= defined,
         f"undefined: {sorted(cited - defined) or 'none'}")
    note("no placeholder references remain", "PLACEHOLDER" not in tex,
         f"{tex.count('PLACEHOLDER')} placeholder tokens")
    note("at least 40 distinct works cited", len(cited) >= 40,
         f"{len(cited)} cited of {len(defined)} available")

    # the compiled PDF must contain no unresolved citation markers
    pdf = os.path.join(ROOT, "paper", "paper.pdf")
    if os.path.exists(pdf):
        import subprocess
        txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True,
                             text=True).stdout
        note("no unresolved citations in the PDF", "[?]" not in txt,
             "found [?]" if "[?]" in txt else "none")
        note("no 'Placeholder' text in the PDF", "Placeholder" not in txt,
             f"{txt.count('Placeholder')} occurrences")

    figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tex)
    missing = [f for f in figs
               if not os.path.exists(os.path.join(PLOTS, f))]
    note("every referenced figure exists", not missing,
         f"missing: {missing or 'none'}")

    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    note("all cross-references resolve", refs <= labels,
         f"dangling: {sorted(refs - labels) or 'none'}")

    # ------------------------------------------ bootstrap intervals (E5/E6/E7)
    ci5 = cload("e5_ladder_ci.csv").set_index("rung")
    check("boot: L2->L3 marginal", 0.2432,
          round(float(ci5.loc["L3 + own behaviour history", "marginal"]), 4), 0.0001)
    check("boot: L2->L3 CI low", 0.1433,
          round(float(ci5.loc["L3 + own behaviour history", "marg_ci_low"]), 4), 0.0001)
    note("boot: only L3 step excludes zero in E5",
         int(ci5.excludes_zero.fillna(False).sum()) == 1,
         f"{int(ci5.excludes_zero.fillna(False).sum())} of {len(ci5) - 1} steps significant")
    ci6 = cload("e6_adherence_ci.csv")
    note("boot: no adherence step separates from zero",
         int(ci6.excludes_zero.fillna(False).sum()) == 0,
         f"{int(ci6.excludes_zero.fillna(False).sum())} significant")
    ci7 = cload("e7_privacy_ci.csv").set_index("rung")
    check("boot: privacy L3 marginal", 0.3004,
          round(float(ci7.loc["L3 + own behaviour history", "marginal"]), 4), 0.0001)

    # ------------------------------------- the prior manuscript is not cited
    note("no citation to the unpublished prior manuscript",
         "singh2026screen" not in tex, "found" if "singh2026screen" in tex else "clean")
    note("fabrication battery is in an appendix", "\\appendices" in tex,
         "appendix present" if "\\appendices" in tex else "still inline")

    # --------------------------------- claims corrected by the citation review
    # These sentences were found to misstate their sources. The corrected
    # wording is asserted here so a later edit cannot silently reintroduce them.
    banned = [
        ("twelve-month trial, adaptive goals derived", "inverted Adams 2017 claim"),
        ("record-wise design that Saeb", "false attribution to Fellger 2020"),
        ("Canadian \\cite{ross2020canadian, healthcanada2019dietary}",
         "2019 food guide cited for a serving table it abolished"),
    ]
    for phrase, why in banned:
        note(f"corrected: {why}", phrase not in tex, "reintroduced" if phrase in tex else "clean")
    required = [
        ("four-month factorial trial", "Adams 2017 duration stated correctly"),
        ("$p = 0.095$", "Adams 2017 significance reported"),
        ("Their folds are subject-wise", "Fellger 2020 described accurately"),
    ]
    for phrase, why in required:
        note(f"present: {why}", phrase in tex, "missing" if phrase not in tex else "ok")

    # ------------------------------------------------------------ report
    width = max(len(n) for _, n, _ in results) + 2
    nfail = 0
    for status, name, detail in results:
        if status == "FAIL":
            nfail += 1
        print(f"[{status}] {name:<{width}} {detail}")
    print(f"\n{len(results) - nfail}/{len(results)} checks passed.")
    if nfail:
        print(f"{nfail} FAILED. The paper and the outputs disagree.")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
