# Bodies Are Stable, Behaviour Is Not

Replication package for *"Bodies Are Stable, Behaviour Is Not: A Personalization Ladder for Health and Fitness Recommendation"* (G. Singh, G. S. Anderson, Thompson Rivers University).

Every number, figure and table in the paper is produced by the code here and checked against the committed outputs by `src/09_verify_paper.py`. If a claim in the paper is not reproducible from these scripts, that is a bug and we want to hear about it.

---

## The finding, in one minute

Health apps personalize on what you tell them at signup: age, sex, height, weight. We measure how far that can go.

1. **Bodies are stable, behaviour is not.** Across 71 people wearing trackers for a median of 88 days, the median intraclass correlation is **0.57 for physiology** and **0.26 for behaviour**. Resting heart rate is 87% a property of the person; step count is 29% a property of the person and 71% a property of the day.
2. **So static profiles cannot predict behaviour.** In NHANES 2013-2014 (n = 6,113 adults, design-weighted), age, sex and BMI explain **89.5%** of the variance in waist-to-height ratio and **7.4%** of the variance in physical activity. Adding BMI to age and sex changes behaviour prediction by nothing.
3. **The ladder, measured.** Out of sample with forward chaining, next-day step count:

   | Rung | R² | Marginal gain [95% CI] |
   |---|---|---|
   | L0 population constant | −0.013 | |
   | L1 demographics | −0.017 | −0.004 [−0.071, +0.032] |
   | L2 + body measurements | −0.017 | +0.000 [−0.055, +0.022] |
   | **L3 + own behaviour history** | **+0.226** | **+0.243 [+0.143, +0.369]** |
   | L4 + yesterday's context | +0.229 | +0.003 [−0.013, +0.009] |

   **Only one step on the ladder is distinguishable from zero.** Intervals are a paired cluster bootstrap over participants.
4. **Cold start is days, not months.** One to two days of logging beats every static rung; the curve peaks at two to four weeks.
5. **A privacy-utility curve.** Each rung costs more disclosure, so measuring benefit per rung prices that disclosure. The only rung that pays for itself is a behaviour log.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/09_verify_paper.py     # checks the paper against the committed outputs
```

To re-run the analysis you need the datasets, which are **not redistributed here** (see `docs/DATA.md`). With them in place:

```bash
python src/01_knowledge_base.py        # the expert engine's output space
python src/02_fabrication_battery.py   # dataset screening (paper appendix)
python src/03_l1_vs_l2_nhanes.py       # NHANES, design-weighted
python src/04_variance_decomposition.py# ICC: between vs within person
python src/05_ladder_prediction.py     # the ladder, forward-chained
python src/06_adherence.py             # step-goal attainment
python src/07_coldstart_privacy.py     # cold start + privacy curve
python src/10_bootstrap_ci.py          # cluster bootstrap intervals
python src/08_make_figures.py          # all 12 figures at 600 dpi
```

`src/panel.py` builds the per-person-per-day panel and is shared by experiments 5, 6, 7 and 10, so all four use identical, causally-safe features.

---

## Layout

```
src/         experiments 1-10 plus the shared panel builder
outputs/
  texts/     every CSV and JSON the paper cites
  plots/     every figure at 600 dpi
paper/       LaTeX source, bibliography, compiled PDF
docs/        data provenance
DECISION_LOG.txt   a running record of what was decided and why, written as work happened
```

---

## Two things we would rather state than have found

**Evaluation protocol.** Experiments 5 and 6 use forward-chained folds over the calendar. The same participants appear in training and test at different times. That is deliberate: it is the deployment case where an app already knows a user and must forecast their future. It is **not** a test of generalization to a new person. Experiment 7 runs the leave-one-person-out design for that, and gets a slightly higher L3 value (+0.275), so the ordering is not an artefact of participant overlap.

**What the L4 null does and does not show.** Our L4 is one operationalization: day-lagged momentary self-report and physiology, present on a median of 45% of days, mean-imputed. Its null result bounds that design at this sample size. It is not evidence that context-aware intervention is worthless in general.

---

## Non-wear

Non-wear days are the dominant hazard in this data and the convention differs by dataset. In LifeSnaps non-wear appears as a *missing* step count, usually with exactly 1440 sedentary minutes; in the second Fitbit cohort the same condition appears as a literal *zero*. Calories cannot disambiguate it, because the device emits basal metabolic rate on non-wear days. Filtering removes 36.5% of person-days. Treating either convention as real would manufacture sedentary behaviour that never happened.

---

## Licence

Code is MIT (see `LICENSE`). Third-party datasets keep their own licences and are not redistributed; see `docs/DATA.md`.

## Contact

Open an issue. Corrections and failed replications are especially welcome.
