"""
charts.py  —  Visualization Layer
==================================
Responsible for:
  - Gauge chart      (score dial from 0–10)
  - Comparison chart (your wine vs average vs premium)

This file has ZERO ML code and ZERO Streamlit code.
It only takes numbers as input and returns matplotlib Figure objects.
The frontend (app.py) decides how and where to display them.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Shared theme colours ──────────────────────────────────────────────────────
BG_DARK   = "#0d0d0d"
BG_PANEL  = "#110505"
TEXT_MAIN = "#f0e6d3"
TEXT_DIM  = "#a08080"
TEXT_AXIS = "#d4b8b8"
ACCENT    = "#c0392b"
SPINE     = "#3a1010"
LEG_BG    = "#1a0505"


# ── Gauge Chart ───────────────────────────────────────────────────────────────

def make_gauge(score: float) -> plt.Figure:
    """
    Draw a semicircular gauge showing the wine quality score.

    Parameters
    ----------
    score : float — predicted quality score (0–10)

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    # Colour bands: low → normal → good → premium
    bands = [
        (3.0, 4.5, "#7f1d1d"),   # red    — low
        (4.5, 6.0, "#1d3557"),   # navy   — below average
        (6.0, 7.5, "#1a4a6e"),   # blue   — normal
        (7.5, 9.0, "#145a32"),   # green  — premium
    ]

    r_outer, r_inner = 1.0, 0.6

    for lo, hi, color in bands:
        t_start = np.pi - (lo - 3) / 6 * np.pi
        t_end   = np.pi - (hi - 3) / 6 * np.pi
        t = np.linspace(t_start, t_end, 60)
        ax.fill_between(
            np.cos(t) * r_outer,
            np.sin(t) * r_outer,
            np.cos(t) * r_inner,
            alpha=0.85, color=color
        )

    # Needle
    clamped = min(max(score, 3), 9)
    angle   = np.pi - (clamped - 3) / 6 * np.pi
    ax.annotate(
        "",
        xy=(np.cos(angle) * 0.75, np.sin(angle) * 0.75),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=TEXT_MAIN, lw=2, mutation_scale=15)
    )
    ax.plot(0, 0, "o", color=ACCENT, markersize=8, zorder=5)

    # Score text
    ax.text(0, 0.15, f"{score:.1f}",
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=TEXT_MAIN, fontfamily="serif")
    ax.text(0, -0.05, "Quality Score",
            ha="center", va="center",
            fontsize=9, color=TEXT_DIM)

    # Scale labels
    for val, lbl in [(3, "3\nLow"), (5.25, "5\nNormal"), (7.5, "7\nGood"), (9, "9\nPremium")]:
        a = np.pi - (val - 3) / 6 * np.pi
        ax.text(np.cos(a) * 1.12, np.sin(a) * 1.12, lbl,
                ha="center", va="center", fontsize=7, color=TEXT_DIM)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Comparison Bar Chart ──────────────────────────────────────────────────────

def make_comparison(
    inputs: dict,
    dataset_averages: dict,
    premium_averages: dict
) -> plt.Figure:
    """
    Draw a grouped bar chart comparing the user's wine against
    the dataset average and the premium wine average.

    Parameters
    ----------
    inputs           : dict — raw wine feature values from the user
    dataset_averages : dict — average values across the full dataset
    premium_averages : dict — average values for premium wines (quality >= 7)

    Returns
    -------
    matplotlib Figure
    """
    # Features to show in the chart (most impactful ones)
    keys   = ["alcohol", "volatile acidity", "residual sugar", "pH", "sulphates", "chlorides"]
    labels = ["Alcohol", "Volatile\nAcidity", "Residual\nSugar", "pH", "Sulphates", "Chlorides"]

    user_vals    = [inputs[k]           for k in keys]
    avg_vals     = [dataset_averages[k] for k in keys]
    premium_vals = [premium_averages[k] for k in keys]

    # Normalise each feature to 0–1 so different scales are comparable
    max_vals = [max(u, a, p) for u, a, p in zip(user_vals, avg_vals, premium_vals)]
    u_norm = [v / m if m else 0 for v, m in zip(user_vals,    max_vals)]
    a_norm = [v / m if m else 0 for v, m in zip(avg_vals,     max_vals)]
    p_norm = [v / m if m else 0 for v, m in zip(premium_vals, max_vals)]

    x, w = np.arange(len(labels)), 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)

    ax.bar(x - w, u_norm, w, label="Your Wine",    color="#c0392b", alpha=0.9)
    ax.bar(x,     a_norm, w, label="Dataset Avg",  color="#2471a3", alpha=0.7)
    ax.bar(x + w, p_norm, w, label="Premium Avg",  color="#1e8449", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_AXIS, fontsize=9)
    ax.set_yticks([])
    ax.set_title("Your Wine vs Average vs Premium",
                 color=TEXT_MAIN, fontfamily="serif", fontsize=12, pad=10)
    ax.legend(facecolor=LEG_BG, labelcolor=TEXT_AXIS, fontsize=8, framealpha=0.8)
    ax.spines[:].set_color(SPINE)
    ax.tick_params(colors=TEXT_DIM)

    plt.tight_layout()
    return fig
