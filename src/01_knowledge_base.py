"""
01_knowledge_base.py
====================
Encodes the expert knowledge layer of the recommender.

Everything in this file traces to a source: either Dr. Anderson's book
(WELLNESSHUB4/Book.md, cited by section and line) or a published guideline.
Nothing here is learned from data, and nothing here is invented by the authors.
That separation is deliberate: the paper needs to be able to say exactly which
part of a recommendation came from expertise and which part came from evidence.

Outputs
-------
outputs/texts/kb_book_rules.json        machine-readable rule set
outputs/texts/kb_summary.txt            human-readable audit of the rule set
outputs/texts/kb_output_space.csv       every distinct recommendation the book's
                                        engine can emit, enumerated

The last output is the point of this file. If the book's engine can only produce
a small number of distinct recommendations, then no amount of user data changes
what a person is told, and that is a measurable ceiling on personalization.
"""

import itertools
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
os.makedirs(TEXTS, exist_ok=True)

BOOK = "WELLNESSHUB4/Book.md"


# ----------------------------------------------------------------- prevention
# Book section 2, lines 64-87. An expert-authored utility over behaviours,
# scored 0-10. We use it to rank competing recommendations; without it a
# recommender has to invent its own priorities.
PREVENTION_INDEX = {
    "do_not_smoke":                 {"score": 9.8, "tier": "critical"},
    "wear_seat_belt":               {"score": 9.2, "tier": "critical"},
    "avoid_drinking_and_driving":   {"score": 9.0, "tier": "critical"},
    "socialize_regularly":          {"score": 8.3, "tier": "high"},
    "regular_strenuous_activity":   {"score": 8.2, "tier": "high"},
    "alcohol_in_moderation":        {"score": 8.2, "tier": "high"},
    "limit_dietary_fat":            {"score": 7.8, "tier": "high"},
    "maintain_healthy_weight":      {"score": 7.7, "tier": "high"},
    "do_not_speed":                 {"score": 7.7, "tier": "high"},
    "monitor_bp_control_stress":    {"score": 7.6, "tier": "medium"},
    "fibre_up_cholesterol_down":    {"score": 7.3, "tier": "medium"},
    "adequate_nutrition":           {"score": 7.1, "tier": "medium"},
    "limit_salt_and_sugar":         {"score": 7.0, "tier": "medium"},
    "sleep_7_to_8_hours":           {"score": 6.7, "tier": "medium"},
}

# ------------------------------------------------------------------ wellness
# Book section 3, lines 87-167. Five domains, ten items each, fixed rubric.
WELLNESS_SCORING = {
    "response_points": {"always": 10, "often": 7, "sometimes": 5,
                        "rarely": 3, "never": 0},
    "bands": [(86, 100, "excellent"), (70, 85, "good"), (50, 69, "average"),
              (30, 49, "below_average"), (0, 29, "poor")],
    "domains": ["physical", "nutritional", "social_emotional",
                "psychological", "intellectual_occupational"],
}

# --------------------------------------------------------------- food groups
# Book section 56, lines 1628-1665. Servings per day as f(age band, sex).
# Ranges in the book are stored as (low, high); a single value repeats.
FOOD_SERVINGS = {
    ("14-18", "female"): {"veg_fruit": (7, 7), "grain": (6, 6), "dairy": (3, 4), "meat": (2, 2)},
    ("14-18", "male"):   {"veg_fruit": (8, 8), "grain": (7, 7), "dairy": (3, 4), "meat": (3, 3)},
    ("19-50", "female"): {"veg_fruit": (7, 8), "grain": (6, 7), "dairy": (2, 2), "meat": (2, 2)},
    ("19-50", "male"):   {"veg_fruit": (8, 10), "grain": (8, 8), "dairy": (2, 2), "meat": (3, 3)},
    ("51+", "female"):   {"veg_fruit": (7, 7), "grain": (6, 6), "dairy": (3, 3), "meat": (2, 2)},
    ("51+", "male"):     {"veg_fruit": (7, 7), "grain": (7, 7), "dairy": (3, 3), "meat": (3, 3)},
}

# ------------------------------------------------------------------ calories
# Book section 56, lines 1667-1681. The book gives the 18-35 band explicitly and
# writes "[Continue pattern for other age ranges...]" for the rest. We encode
# ONLY what the book states. The gap is itself a finding: the deployed engine is
# undefined for most of its own input space.
CALORIES = {
    ("18-35", "male",   "inactive"):    2500,
    ("18-35", "male",   "very_active"): 3500,
    ("18-35", "female", "inactive"):    2000,
    ("18-35", "female", "very_active"): 2500,
}
CALORIES_UNSPECIFIED = "book states the pattern continues but does not give values"

# ------------------------------------------------------------------ activity
# Book section 56, lines 1683-1705, consistent with Canada's Physical Activity
# Guide and WHO 2020. Note these are population constants: no user attribute
# enters them.
ACTIVITY = {
    "moderate_min_per_session": 30,
    "moderate_days_per_week": 5,
    "vigorous_min_per_session": 20,
    "vigorous_days_per_week": 3,
    "endurance_days": (4, 7),
    "flexibility_days": (4, 7),
    "strength_days": (2, 4),
}

# ---------------------------------------------------------------- heart rate
# Book section 56, lines 1707-1717. Age-predicted maximum, the Fox formula.
def hr_zones(age):
    """Return the book's heart-rate zones for a given age."""
    hr_max = 220 - age
    return {
        "hr_max": hr_max,
        "moderate": (0.60 * hr_max, 0.75 * hr_max),
        "vigorous": (0.70 * hr_max, 0.85 * hr_max),
    }


# ------------------------------------------------------- constant lifestyle
# Book section 56, lines 1719-1731. Identical for every user.
SLEEP_HOURS = (7, 8)
WATER_CUPS = (6, 8)

# --------------------------------------------------------------- weight loss
# Book section 12, lines 380-402.
SAFE_WEIGHT_LOSS_KG_PER_WEEK = (0.45, 0.9)   # 1-2 lb

# ---------------------------------------------------------------- adherence
# Book section 27, lines 787-815. The PRECEDE structure. These name the feature
# families an adherence model should contain.
ADHERENCE_FACTORS = {
    "predisposing": ["knowledge_of_benefits", "value_placed_on_outcomes",
                     "medical_professional_suggestion"],
    "enabling": ["activity_proficiency", "facility_access", "fitness_level",
                 "repertoire_of_comfortable_activities"],
    "reinforcing": ["training_partner", "supportive_partner", "peer_comments",
                    "medical_support", "progress_acknowledgment"],
}

# ------------------------------------------------------------------- refusal
# Book section 59, lines 1813-1834. Conditions under which the system must stop
# recommending and defer to a clinician. A recommender that cannot refuse is
# unsafe, so these are hard gates rather than ranked suggestions.
REFERRAL_TRIGGERS = [
    "chest_pain_during_exercise",
    "persistent_high_resting_hr",
    "unexplained_weight_change",
    "chronic_insomnia",
    "chronic_stress_symptoms",
    "suspected_depression",
    "chronic_lower_back_pain",
]
EXERCISE_CAUTIONS = [
    "start_slowly_progress_gradually",
    "do_not_increase_too_fast",
    "rest_when_unwell",
    "stop_on_sharp_pain",
]


# =============================================================== the engine
def age_band_food(age):
    if 14 <= age <= 18:
        return "14-18"
    if 19 <= age <= 50:
        return "19-50"
    if age >= 51:
        return "51+"
    return None


def book_recommendation(age, sex, activity_level):
    """
    The book's recommendation engine, implemented exactly as specified in
    section 56. This is the L1 baseline in the paper's personalization ladder.

    Note what is absent from the signature: weight, height, waist, goal, health
    conditions, and every logged behaviour. Nothing the user records after
    onboarding can change this output.
    """
    band = age_band_food(age)
    rec = {
        "food_servings": FOOD_SERVINGS.get((band, sex)),
        "calories": CALORIES.get(("18-35" if 18 <= age <= 35 else None,
                                  sex, activity_level)),
        "activity": dict(ACTIVITY),
        "hr_zones": hr_zones(age),
        "sleep_hours": SLEEP_HOURS,
        "water_cups": WATER_CUPS,
    }
    return rec


def enumerate_output_space():
    """
    Enumerate every distinct recommendation the book's engine can produce over a
    realistic adult input grid. The size of this set is a hard ceiling on how
    personalized the engine can possibly be, regardless of how much data the app
    collects about a user.
    """
    ages = list(range(18, 81))
    sexes = ["female", "male"]
    levels = ["inactive", "moderately_active", "very_active"]

    rows = []
    for age, sex, lvl in itertools.product(ages, sexes, levels):
        r = book_recommendation(age, sex, lvl)
        fs = r["food_servings"] or {}
        rows.append({
            "age": age, "sex": sex, "activity_level": lvl,
            "veg_fruit": str(fs.get("veg_fruit")),
            "grain": str(fs.get("grain")),
            "dairy": str(fs.get("dairy")),
            "meat": str(fs.get("meat")),
            "calories": r["calories"],
            "moderate_min": r["activity"]["moderate_min_per_session"],
            "moderate_days": r["activity"]["moderate_days_per_week"],
            "strength_days": str(r["activity"]["strength_days"]),
            "sleep": str(r["sleep_hours"]),
            "water": str(r["water_cups"]),
            "hr_moderate_low": round(r["hr_zones"]["moderate"][0], 1),
            "hr_moderate_high": round(r["hr_zones"]["moderate"][1], 1),
        })
    return pd.DataFrame(rows)


def main():
    kb = {
        "source": BOOK,
        "prevention_index": PREVENTION_INDEX,
        "wellness_scoring": WELLNESS_SCORING,
        "food_servings": {f"{k[0]}|{k[1]}": v for k, v in FOOD_SERVINGS.items()},
        "calories": {"|".join(map(str, k)): v for k, v in CALORIES.items()},
        "calories_gap": CALORIES_UNSPECIFIED,
        "activity": ACTIVITY,
        "sleep_hours": SLEEP_HOURS,
        "water_cups": WATER_CUPS,
        "safe_weight_loss_kg_per_week": SAFE_WEIGHT_LOSS_KG_PER_WEEK,
        "adherence_factors": ADHERENCE_FACTORS,
        "referral_triggers": REFERRAL_TRIGGERS,
        "exercise_cautions": EXERCISE_CAUTIONS,
    }
    with open(os.path.join(TEXTS, "kb_book_rules.json"), "w") as f:
        json.dump(kb, f, indent=1)

    df = enumerate_output_space()
    df.to_csv(os.path.join(TEXTS, "kb_output_space.csv"), index=False)

    # How many genuinely distinct recommendations can the engine emit?
    diet_cols = ["veg_fruit", "grain", "dairy", "meat"]
    act_cols = ["moderate_min", "moderate_days", "strength_days", "sleep", "water"]
    all_cols = diet_cols + act_cols + ["calories"]

    n_inputs = len(df)
    n_diet = df[diet_cols].drop_duplicates().shape[0]
    n_act = df[act_cols].drop_duplicates().shape[0]
    n_all = df[all_cols].drop_duplicates().shape[0]
    n_with_hr = df[all_cols + ["hr_moderate_low"]].drop_duplicates().shape[0]
    cal_defined = df.calories.notna().mean() * 100

    lines = [
        "BOOK KNOWLEDGE BASE: AUDIT OF THE DEPLOYED RECOMMENDATION ENGINE",
        "=" * 68,
        f"source: {BOOK}, section 56 (HARDCODED RECOMMENDATION LOGIC)",
        "",
        f"input grid enumerated                     : {n_inputs} "
        f"(ages 18-80 x 2 sexes x 3 activity levels)",
        "",
        "DISTINCT OUTPUTS THE ENGINE CAN PRODUCE",
        f"  distinct diet prescriptions             : {n_diet}",
        f"  distinct activity/sleep/water sets      : {n_act}",
        f"  distinct full recommendations           : {n_all}",
        f"  distinct once age-based HR zones added  : {n_with_hr}",
        "",
        f"  compression (outputs / inputs)          : {n_all / n_inputs:.4f}",
        "",
        "COVERAGE",
        f"  inputs with a defined calorie target    : {cal_defined:.1f}%",
        f"  note: {CALORIES_UNSPECIFIED}",
        "",
        "INTERPRETATION",
        "  The diet and activity components of the engine are a lookup table.",
        "  Excluding the continuous heart-rate zones, which vary with age by",
        f"  arithmetic rather than by clinical reasoning, {n_inputs} distinct users",
        f"  receive one of only {n_all} distinct recommendations. Activity, sleep and",
        f"  water targets are population constants: {n_act} distinct value(s) across",
        "  the whole grid. No logged behaviour enters the engine at any point.",
        "",
        "  This is the L1 rung of the personalization ladder and the baseline the",
        "  paper measures against. It is guideline-faithful and safe; the open",
        "  question the paper answers is whether it is sufficient.",
    ]
    out = "\n".join(lines)
    with open(os.path.join(TEXTS, "kb_summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
