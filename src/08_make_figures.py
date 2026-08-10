"""
08_make_figures.py
==================
Publication figures. Authored at true print size (IEEE column = 3.5 in, double
column = 7.16 in) at 600 dpi, so nothing is scaled down in LaTeX and every label
is readable at the size it will actually be printed.

Design rules followed here
--------------------------
- Colour carries the FINDING, not arbitrary series identity. The ladder is an
  ordered thing, so position on the axis carries identity and colour separates
  the two regimes the paper is about: static rungs versus behaviour-based rungs.
- Palette is Okabe-Ito, validated for colour-vision deficiency: worst adjacent
  pair dE 11.0 (deutan), 24.2 (normal vision).
- Every categorical mark is directly labelled, so identity is never colour alone.
- No dual axes anywhere. Recessive grid, no top or right spine.
- Zero lines are drawn explicitly wherever R2 can go negative, because a
  negative R2 is the point of several of these figures.
"""

import json
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEXTS = os.path.join(ROOT, "outputs", "texts")
PLOTS = os.path.join(ROOT, "outputs", "plots")
os.makedirs(PLOTS, exist_ok=True)

COL, DBL = 3.5, 7.16
DPI = 600

# Okabe-Ito, validated
BLUE = "#0072B2"     # behaviour-based rungs: the finding
VERM = "#D55E00"     # fabricated / harmful
GREEN = "#009E73"    # real / beneficial
ORANGE = "#E69F00"
GREY = "#7A7A7A"     # static rungs: the null result
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#D9D9D9"

mpl.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 8.0, "axes.labelsize": 7.4,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.edgecolor": "#4A4A4A", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
    "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

RUNG_SHORT = {
    "L0 population constant": "L0\nconstant",
    "L1 demographics (the book)": "L1\ndemographics",
    "L2 + body measurements": "L2\n+ body",
    "L3 + own behaviour history": "L3\n+ own history",
    "L4 + yesterday's context": "L4\n+ context",
    "L0 base rate": "L0\nbase rate",
    "L2g + the goal itself": "L2g\n+ the goal",
    "L3a + own adherence history": "L3a\n+ adherence hx",
}
STATIC = {"L0", "L1", "L2", "L2g"}

# Typeset negative numbers with a real minus sign (U+2212) so that data labels
# match the minus sign matplotlib already uses on the tick labels.
MINUS = "−"


def sig(s):
    """Replace the ASCII hyphen in a formatted number with a true minus."""
    return s.replace("-", MINUS)


def rung_colour(rung):
    tag = rung.split()[0]
    return GREY if tag in STATIC else BLUE


def grid_y(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


# ============================================================ F1 output space
def f1_output_space():
    df = pd.read_csv(os.path.join(TEXTS, "kb_output_space.csv"))
    diet = ["veg_fruit", "grain", "dairy", "meat"]
    act = ["moderate_min", "moderate_days", "strength_days", "sleep", "water"]
    n_in = len(df)
    counts = [n_in, df[diet + act + ["calories"]].drop_duplicates().shape[0],
              df[diet].drop_duplicates().shape[0],
              df[act].drop_duplicates().shape[0]]
    labels = ["Distinct adult\nuser types", "Distinct full\nrecommendations",
              "Distinct diet\nprescriptions", "Distinct activity,\nsleep, water sets"]
    cols = [GREY, VERM, VERM, VERM]

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    bars = ax.bar(range(4), counts, color=cols, width=0.62, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 900)
    ax.set_ylabel("Count (log scale)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c * 1.18, f"{c}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold",
                color=INK)
    grid_y(ax)
    ax.set_title("The book's engine: 378 user types, 14 answers", pad=5)
    save(fig, "f01_output_space.png")


# ======================================================= F2 fabrication matrix
def f2_battery():
    r = pd.read_csv(os.path.join(TEXTS, "e2_battery_results.csv"))
    dets = ["d1_flag", "d2_flag", "d3_flag", "d4_flag"]
    names = ["D1\nlookup", "D2\nno signal", "D3\ntemporal", "D4\ninconsistent"]
    r = r.sort_values(["ground_truth", "dataset"], ascending=[True, True])
    M = r[dets].astype(bool).values

    fig, ax = plt.subplots(figsize=(DBL * 0.62, 2.5))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            fired = M[i, j]
            fab = r.ground_truth.iloc[i] == "FABRICATED"
            ax.add_patch(plt.Rectangle((j - .30, i - .30), .60, .60,
                                       facecolor=(VERM if fab else GREEN) if fired else "#F2F2F2",
                                       edgecolor="white", linewidth=1.4, zorder=3))
            if fired:
                # drawn, not a text glyph: the multiplication sign is missing
                # from the default sans font and rendered as a hollow box
                ax.plot([j - .10, j + .10], [i - .10, i + .10],
                        color="white", lw=1.6, zorder=5, solid_capstyle="round")
                ax.plot([j - .10, j + .10], [i + .10, i - .10],
                        color="white", lw=1.6, zorder=5, solid_capstyle="round")
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"{d}" for d in r.dataset], fontsize=6.6)
    for i, t in enumerate(r.ground_truth):
        ax.get_yticklabels()[i].set_color(VERM if t == "FABRICATED" else GREEN)
    ax.set_xlim(-.55, 3.55); ax.set_ylim(-.55, len(r) - .45)
    ax.invert_yaxis()
    ax.set_aspect(0.42)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Each detector catches a different fabrication\n"
                 "(red = fabricated dataset, green = real; no real dataset fires)",
                 pad=6, fontsize=7.6)
    save(fig, "f02_fabrication_matrix.png")


# ================================================== F3 temporal fingerprint
def f3_temporal():
    r = pd.read_csv(os.path.join(TEXTS, "e2_battery_results.csv"))
    r = r[r.d3_lag1_ac.notna()].sort_values("d3_lag1_ac")
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    cols = [VERM if t == "FABRICATED" else GREEN for t in r.ground_truth]
    y = np.arange(len(r))
    ax.barh(y, r.d3_lag1_ac, color=cols, height=0.6, zorder=3)
    ax.axvline(0, color="#9A9A9A", linewidth=0.7, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(r.dataset, fontsize=6.8)
    for yi, v in zip(y, r.d3_lag1_ac):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, sig(f"{v:.2f}"),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=6.8, color=INK)
    ax.set_xlim(-0.18, 1.03)
    ax.set_xlabel("Within-person lag-1 autocorrelation")
    grid_x(ax)
    ax.set_title("Real people persist from day to day.\nFabricated rows do not.",
                 pad=5)
    save(fig, "f03_temporal_fingerprint.png")


# ====================================================== F4 NHANES body vs beh
def f4_nhanes_variance():
    v = pd.read_csv(os.path.join(TEXTS, "e3_variance_explained.csv"))
    body = v[v.target.str.startswith("BODY") &
             (v.target.str.contains("waist-to-height"))]
    beh = v[v.target == "BEHAVIOUR: weekly MVPA minutes"]
    order = ["L1  age + sex", "L1+ age + sex + ethnicity + SES", "L2  + BMI",
             "L2+ + waist", "L2++ + waist-to-height + BP"]
    short = ["L1\nage+sex", "L1+\n+ethnicity\n+SES", "L2\n+BMI",
             "L2+\n+waist", "L2++\n+WHtR+BP"]

    fig, ax = plt.subplots(figsize=(DBL * 0.52, 2.35))
    x = np.arange(len(order)); w = 0.38
    bv = [float(body[body.predictors == o].R2_weighted.iloc[0])
          if len(body[body.predictors == o]) else np.nan for o in order]
    ev = [float(beh[beh.predictors == o].R2_weighted.iloc[0])
          if len(beh[beh.predictors == o]) else np.nan for o in order]
    ax.bar(x - w / 2, bv, w, color=ORANGE, label="Body (waist-to-height ratio)",
           zorder=3)
    ax.bar(x + w / 2, ev, w, color=BLUE, label="Behaviour (weekly activity)",
           zorder=3)
    for xi, (b, e) in enumerate(zip(bv, ev)):
        if np.isfinite(b):
            ax.text(xi - w / 2, b + .02, f"{b:.2f}", ha="center", fontsize=6.6)
        if np.isfinite(e):
            ax.text(xi + w / 2, e + .02, f"{e:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("Variance explained (weighted $R^2$)")
    # Headroom above 1.0 so the legend clears the 0.89 bar and its data label;
    # the ticks still stop at 1.0 because the quantity cannot exceed it.
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", ncol=1)
    grid_y(ax)
    ax.set_title("Adding body measurements explains the body,\n"
                 "and adds nothing to behaviour", pad=5)
    save(fig, "f04_nhanes_body_vs_behaviour.png")


# ================================================================ F5 ICC
def f5_icc():
    icc = pd.read_csv(os.path.join(TEXTS, "e4_icc.csv")).sort_values("icc")
    # The long category names sit outside the axes, so a tight bounding box
    # widens the saved image past the column. Narrow the figure by that much,
    # otherwise LaTeX scales the whole plot down and the tick labels land
    # under 6 pt at print size.
    fig, ax = plt.subplots(figsize=(COL - 0.58, 2.6))
    y = np.arange(len(icc))
    cols = [BLUE if k == "behaviour" else ORANGE for k in icc.kind]
    ax.barh(y, icc.icc, color=cols, height=0.62, zorder=3)
    err_lo = icc.icc - icc.icc_ci_low
    err_hi = icc.icc_ci_high - icc.icc
    ax.errorbar(icc.icc, y, xerr=[err_lo, err_hi], fmt="none",
                ecolor="#4A4A4A", elinewidth=0.7, capsize=1.6, zorder=4)
    ax.set_yticks(y)
    pretty = {"resting_hr": "resting heart rate", "rmssd": "heart-rate variability",
              "spo2": "blood oxygen", "minutesAsleep": "minutes asleep",
              "sleep_efficiency": "sleep efficiency", "stress_score": "stress score",
              "nightly_temperature": "nightly temperature", "steps": "steps",
              "calories": "calories burned", "sedentary_minutes": "sedentary minutes",
              "very_active_minutes": "very active minutes",
              "moderately_active_minutes": "moderately active minutes"}
    ax.set_yticklabels([pretty.get(v, v.replace("_", " ")) for v in icc.variable],
                       fontsize=6.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of variance that is between people (ICC)")
    grid_x(ax)
    ax.axvline(0.5, color="#9A9A9A", linewidth=0.6, linestyle=(0, (3, 3)), zorder=2)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
         plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(h, ["Behaviour", "Physiology"], loc="lower right")
    ax.set_title("Bodies are stable. Behaviour is not.", pad=5)
    save(fig, "f05_icc.png")


# ============================================================ F6 THE LADDER
def f6_ladder():
    r = pd.read_csv(os.path.join(TEXTS, "e5_ladder_results.csv"))
    ls = r[(r.source == "lifesnaps") & (r.target == "steps")]
    order = ["L0 population constant", "L1 demographics (the book)",
             "L2 + body measurements", "L3 + own behaviour history",
             "L4 + yesterday's context"]
    fig, ax = plt.subplots(figsize=(DBL * 0.54, 2.5))
    x = np.arange(len(order)); w = 0.38
    for k, (mdl, hatch, lbl) in enumerate(
            [("ridge", None, "Ridge"), ("gbm", "////", "Gradient boosting")]):
        vals = [ls[(ls.rung == o) & (ls.model == mdl)].r2.mean() for o in order]
        ax.bar(x + (k - 0.5) * w, vals, w,
               color=[rung_colour(o) for o in order],
               hatch=hatch, edgecolor="white", linewidth=0.5, zorder=3,
               label=lbl)
        for xi, v in zip(x, vals):
            # Nudge each label outward from the group centre (still inside its
            # own bar) so the ridge and boosting labels of a pair cannot touch.
            ax.text(xi + (k - 0.5) * w * 1.34,
                    v + (0.012 if v >= 0 else -0.028),
                    sig(f"{v:+.2f}"), ha="center", fontsize=6.2,
                    va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="#4A4A4A", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([RUNG_SHORT[o] for o in order])
    ax.set_ylabel("Out-of-sample $R^2$, next-day steps")
    ax.set_ylim(-0.11, 0.30)
    grid_y(ax)
    hs = [plt.Rectangle((0, 0), 1, 1, facecolor=GREY, edgecolor="white"),
          plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="white"),
          plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#4A4A4A",
                        hatch="////")]
    ax.legend(hs, ["Static rungs", "Behaviour-based rungs", "Gradient boosting"],
              loc="upper left", ncol=1)
    ax.set_title("Everything below L3 is worse than useless", pad=5)
    save(fig, "f06_ladder.png")


# ================================================ F7 ladder replication
def f7_ladder_replication():
    r = pd.read_csv(os.path.join(TEXTS, "e5_ladder_results.csv"))
    r = r[r.model == "ridge"]
    combos = [("lifesnaps", "steps", "LifeSnaps\nnext-day steps"),
              ("lifesnaps", "very_active_minutes", "LifeSnaps\nactive minutes"),
              ("mobius", "steps", "Fitbit cohort\nnext-day steps"),
              ("mobius", "very_active_minutes", "Fitbit cohort\nactive minutes")]
    full = ["L0 population constant", "L1 demographics (the book)",
            "L2 + body measurements", "L3 + own behaviour history",
            "L4 + yesterday's context"]
    fig, axes = plt.subplots(1, 4, figsize=(DBL, 2.15), sharey=True)
    for ax, (src, tgt, title) in zip(axes, combos):
        g = r[(r.source == src) & (r.target == tgt)]
        # Keep every panel on the same five-rung x axis so bar widths and
        # positions are comparable; rungs absent from a cohort are simply blank.
        pos = [k for k, o in enumerate(full) if o in set(g.rung)]
        order = [full[k] for k in pos]
        vals = [g[g.rung == o].r2.mean() for o in order]
        ax.bar(pos, vals,
               color=[rung_colour(o) for o in order], width=0.66, zorder=3)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + (0.012 if v >= 0 else -0.03), sig(f"{v:+.2f}"),
                    ha="center", fontsize=6.0,
                    va="bottom" if v >= 0 else "top")
        ax.axhline(0, color="#4A4A4A", linewidth=0.8, zorder=4)
        ax.set_xlim(-0.72, 4.72)
        ax.set_xticks(pos)
        ax.set_xticklabels([o.split()[0] for o in order], fontsize=6.6)
        ax.set_title(title, fontsize=7.0, pad=4)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel("Out-of-sample $R^2$")
    axes[0].set_ylim(-0.12, 0.47)
    fig.suptitle("The same ordering in two cohorts and two outcomes",
                 fontsize=8.0, y=1.04)
    save(fig, "f07_ladder_replication.png")


# ============================================================ F8 adherence
def f8_adherence():
    r = pd.read_csv(os.path.join(TEXTS, "e6_adherence_results.csv"))
    agg = r.groupby("rung")[["auc", "brier"]].mean()
    order = [o for o in ["L0 base rate", "L1 demographics (the book)",
                         "L2 + body measurements", "L2g + the goal itself",
                         "L3 + own behaviour history",
                         "L3a + own adherence history",
                         "L4 + yesterday's context"] if o in agg.index]
    fig, ax = plt.subplots(figsize=(DBL * 0.54, 2.4))
    vals = [agg.loc[o, "auc"] for o in order]
    ax.bar(range(len(order)), vals,
           color=[rung_colour(o) for o in order], width=0.64, zorder=3)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.4)
    ax.axhline(0.5, color=VERM, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    ax.text(len(order) - 0.62, 0.508, "chance", color=VERM, fontsize=6.2,
            ha="right")
    # Seven rungs in one column: the descriptors must be short enough that
    # neighbouring tick labels cannot touch.
    tight = {"L0 base rate": "L0\nbase rate",
             "L1 demographics (the book)": "L1\ndemog.",
             "L2 + body measurements": "L2\nbody",
             "L2g + the goal itself": "L2g\ngoal",
             "L3 + own behaviour history": "L3\nhistory",
             "L3a + own adherence history": "L3a\nadherence",
             "L4 + yesterday's context": "L4\ncontext"}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([tight.get(o, o.split()[0]) for o in order],
                       fontsize=6.4)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("AUC, will they hit their goal tomorrow?")
    grid_y(ax)
    ax.set_title("Whether you kept your goal lately beats\nanything about your body",
                 pad=5)
    save(fig, "f08_adherence.png")


# =========================================================== F9 cold start
def f9_coldstart():
    c = pd.read_csv(os.path.join(TEXTS, "e7_coldstart_curve.csv"))
    cols = ["L0 population constant", "L1 demographics (the book)",
            "L2 + body measurements", "L3 + own behaviour history"]
    labels = ["L0 constant", "L1 demographics", "L2 + body", "L3 + own history"]
    colours = [GREY, GREY, GREY, BLUE]
    styles = [(0, (1, 1.6)), (0, (4, 2)), (0, (6, 2, 1, 2)), "-"]
    xlab = [("0" if (a == 0 and b == 0) else
             (f"{int(a)}-{int(b)}" if b < 10000 else f"{int(a)}+"))
            for a, b in zip(c.hist_days_low, c.hist_days_high)]
    x = np.arange(len(c))

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for col, lab, cc, st in zip(cols, labels, colours, styles):
        if col not in c.columns:
            continue
        ax.plot(x, c[col], color=cc, linestyle=st, linewidth=1.6 if cc == BLUE else 1.0,
                marker="o" if cc == BLUE else None, markersize=3.4, zorder=4,
                label=lab)
    ax.axhline(0, color="#4A4A4A", linewidth=0.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=6.6)
    ax.set_xlabel("Days of the person's own history available")
    ax.set_ylabel("Out-of-sample $R^2$")
    grid_y(ax)
    # Opaque background: the three static curves run through the legend corner.
    leg = ax.legend(loc="lower right", ncol=1, frameon=True, facecolor="white",
                    edgecolor="none", framealpha=1.0, borderpad=0.35)
    leg.set_zorder(6)
    # annotate crossover
    i_cross = next((i for i in range(len(c))
                    if c["L3 + own behaviour history"].iloc[i] > 0), None)
    if i_cross is not None:
        ax.annotate("personalization overtakes\neverything static after\n1-2 days of logging",
                    xy=(i_cross, c["L3 + own behaviour history"].iloc[i_cross]),
                    xytext=(i_cross - 0.72, 0.175), fontsize=6.2, color=INK,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="#4A4A4A", lw=0.7,
                                    connectionstyle="arc3,rad=-0.2"))
    ax.set_title("Two days of logging beats every static profile", pad=5)
    save(fig, "f09_coldstart.png")


# ========================================================= F10 privacy curve
def f10_privacy():
    p = pd.read_csv(os.path.join(TEXTS, "e7_privacy_utility.csv"))
    fig, ax = plt.subplots(figsize=(DBL * 0.55, 2.5))
    x = p.data_cost_level.values
    y = p.r2_new_user.values
    ax.plot(x, y, color="#B0B0B0", linewidth=1.0, zorder=2)
    ax.scatter(x, y, s=[46] * len(x),
               color=[BLUE if v > 0 else VERM for v in y],
               zorder=4, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#4A4A4A", linewidth=0.8, zorder=3)
    notes = ["nothing", "age, sex", "+ height, weight,\nwaist",
             "+ a daily\nactivity log", "+ mood and place,\ncontinuously"]
    for xi, yi, tag, note in zip(x, y, p.rung, notes):
        ax.annotate(sig(f"{tag.split()[0]}  {yi:+.3f}"), (xi, yi),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=6.6, fontweight="bold")
        # The steep L2 to L3 segment runs straight through the L3 note, so the
        # notes are masked with an opaque background rather than overprinted.
        ax.annotate(note, (xi, yi), textcoords="offset points",
                    xytext=(0, -22), ha="center", fontsize=6.0, color=MUTED,
                    zorder=6,
                    bbox=dict(boxstyle="square,pad=0.18", facecolor="white",
                              edgecolor="none"))
    ax.set_xticks(x)
    ax.set_xticklabels(["none", "low", "moderate", "high", "very high"],
                       fontsize=6.8)
    ax.set_xlabel("How much the user must disclose")
    ax.set_ylabel("Out-of-sample $R^2$, new user")
    ax.set_ylim(-0.14, 0.40)
    # Room for the "+ mood and place, continuously" note under the last point,
    # which otherwise runs past the right edge of the axes.
    ax.set_xlim(-0.62, 4.62)
    grid_y(ax)
    ax.set_title("The only rung that pays is the behaviour log.\n"
                 "The most intrusive one returns less than nothing.", pad=5)
    save(fig, "f10_privacy_utility.png")


# ==================================================== F11 feature importance
def f11_features():
    p = os.path.join(TEXTS, "e5_feature_importance.csv")
    if not os.path.exists(p):
        return
    f = pd.read_csv(p).head(10).iloc[::-1]
    pretty = {"hist_expmean": "their long-run average",
              "hist_mean28": "average, last 28 days",
              "hist_mean7": "average, last 7 days",
              "hist_mean3": "average, last 3 days",
              "hist_lag1": "yesterday", "hist_lag2": "two days ago",
              "hist_lag7": "same day last week",
              "hist_std7": "how variable they are",
              "hist_trend": "recent trend vs baseline",
              "hist_dow_dev": "their own day-of-week pattern",
              "dow": "day of week", "is_weekend": "weekend",
              "age": "age", "sex_male": "sex", "bmi": "BMI"}
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    y = np.arange(len(f))
    cols = [GREY if v in ("age", "sex_male", "bmi") else BLUE for v in f.feature]
    ax.barh(y, f.std_coef, color=cols, height=0.62, zorder=3)
    ax.axvline(0, color="#4A4A4A", linewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(v, v) for v in f.feature], fontsize=6.6)
    ax.set_xlabel("Standardized ridge coefficient")
    grid_x(ax)
    ax.set_title("What actually carries the signal", pad=5)
    save(fig, "f11_feature_importance.png")


# ================================================= F12 architecture diagram
def f12_architecture():
    fig, ax = plt.subplots(figsize=(DBL, 2.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.45); ax.axis("off")

    layers = [
        (0.15, "Knowledge layer", "Guidelines and the book, encoded as rules",
         "PAR-Q+ screening  |  contraindications  |  WHO and CSEP volume floors\n"
         "age-based heart-rate zones  |  food-group targets  |  referral triggers",
         ORANGE, "Defines what is PERMITTED.\nNever learned. Auditable.\nCan refuse."),
        (1.5, "Personalization layer", "Learned, on quantities that are observed",
         "P(adherence | person, history, context)  x  benefit(action)\n"
         "ranks the permitted actions using the person's own logged behaviour",
         BLUE, "Decides what is LIKELY\nto be done. L3 is where\nthe signal lives."),
        (2.85, "Adaptation layer", "Updates as evidence arrives",
         "each suggestion is a decision point  |  log action and proximal outcome\n"
         "off-policy evaluation  |  seeded from the knowledge layer, so day one works",
         GREEN, "Improves with use.\nCold start covered by\nthe layer above."),
    ]
    for y, title, sub, body, colour, note in layers:
        ax.add_patch(plt.Rectangle((0.15, y), 6.75, 1.12, facecolor=colour,
                                   alpha=0.09, edgecolor=colour, linewidth=1.0))
        ax.add_patch(plt.Rectangle((0.15, y), 0.055, 1.12, facecolor=colour,
                                   edgecolor="none"))
        ax.text(0.36, y + 0.90, title, fontsize=8.0, fontweight="bold", color=INK)
        ax.text(0.36, y + 0.68, sub, fontsize=6.6, color=MUTED, style="italic")
        ax.text(0.36, y + 0.36, body, fontsize=6.3, color=INK,
                linespacing=1.45, va="center")
        ax.text(7.10, y + 0.56, note, fontsize=6.4, color=INK, va="center",
                linespacing=1.5)

    for y in (1.32, 2.67):
        ax.annotate("", xy=(3.5, y + 0.17), xytext=(3.5, y),
                    arrowprops=dict(arrowstyle="-|>", color="#8A8A8A", lw=1.0))
    ax.text(3.55, 4.22, "Feasible set flows up; observed outcomes flow back down",
            fontsize=6.4, color=MUTED, ha="center", style="italic")
    save(fig, "f12_architecture.png")


if __name__ == "__main__":
    print("building figures")
    for fn in [f1_output_space, f2_battery, f3_temporal, f4_nhanes_variance,
               f5_icc, f6_ladder, f7_ladder_replication, f8_adherence,
               f9_coldstart, f10_privacy, f11_features, f12_architecture]:
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
    print("done")
