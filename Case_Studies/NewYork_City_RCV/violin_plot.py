"""
Generate a violin / boxplot of 2017 vs. 2021 victory‑margin distributions,
with background color‑bands for competitiveness categories.

Save this file (e.g. make_violin_bands.py) in the same directory as
`comparable_margins_2017_vs_2021_sorted.xlsx`, then run:

    python make_violin_bands.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# 1. Read & clean data
# ------------------------------------------------------------------
FILE = "/Users/saeesbox/Desktop/Social_Choice_Work/empirical RCV/comparable_margins_2017_vs_2021_sorted.xlsx"   # <— adjust path if needed
df = pd.read_excel(FILE)

# Ensure numeric
df["Margin2017"] = pd.to_numeric(df["Margin2017"], errors="coerce")
df["Margin2021"] = pd.to_numeric(df["Margin2021"], errors="coerce")

# Keep rows with both margins present and Margin2017 ≠ 100
data = df.loc[
    df["Margin2017"].notna()
    & df["Margin2021"].notna()
    & (df["Margin2017"] != 100)
]

# ------------------------------------------------------------------
# 2. Category bands & colours (RGB values scale 0‑1)
# ------------------------------------------------------------------
winner_color      = (189/255, 223/255, 167/255)  # Winner (0 %)
categories = [
    ("Near Winner",    0,   5,  (223/255, 240/255, 216/255)),
    ("Contender",      5,  20,  (253/255, 245/255, 206/255)),
    ("Competitive",   20,  30,  (253/255, 231/255, 208/255)),
    ("Distant",       30,  45,  (248/255, 218/255, 205/255)),
    ("Far Behind",    45, 100,  (242/255, 201/255, 198/255)),
]

# ------------------------------------------------------------------
# 3. Build the plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Background bands
for label, low, high, color in categories:
    ax.axhspan(low, high, color=color, alpha=0.5, zorder=0)

# Thin line for exact winners (margin = 0)
ax.axhline(0, color=winner_color, linewidth=6, alpha=0.5, zorder=0)

# Violin + boxplot
data_to_plot = [data["Margin2017"], data["Margin2021"]]
vp = ax.violinplot(
    data_to_plot,
    positions=[1, 2],
    widths=0.6,
    showmeans=False,
    showmedians=False,
)
# for body in vp["bodies"]:
#     body.set_facecolor("#F5CFA5")

ax.boxplot(
    data_to_plot,
    positions=[1, 2],
    widths=0.2,
    showfliers=False,
    boxprops=dict(color="black"),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    medianprops=dict(color="black", linewidth=2),
)

# Labels & grid
ax.set_xticks([1, 2])
ax.set_xticklabels(["2017", "2021"])
ax.set_ylabel("Margin of Victory (%)")
ax.set_title("Victory Margin Distribution: NYC")
ax.set_ylim(-1, 100)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=winner_color, alpha=0.5, label="Winner (0 %)")]
legend_handles += [
    Patch(facecolor=clr, alpha=0.5, label=lbl) for lbl, _, _, clr in categories
]
ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1))

plt.tight_layout()

# ------------------------------------------------------------------
# 4. Export
# ------------------------------------------------------------------
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/violin_competitive_bands.pdf")
fig.savefig("outputs/violin_competitive_bands.png", dpi=300)
print("Figure saved to outputs/violin_competitive_bands.(pdf|png)")


"""
Create violin + boxplot for 2020 vs 2024 victory margins
with competitiveness color bands.

Requires: pandas, matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 1  Load and clean ----------
FILE = "/Users/saeesbox/Desktop/Social_Choice_Work/empirical RCV/alaska_margins.xlsx"          # adjust if CSV or other name
df = pd.read_excel(FILE)

df["Margin2020"] = pd.to_numeric(df["Margin2020"], errors="coerce")
df["Margin2024"] = pd.to_numeric(df["Margin2024"], errors="coerce")

# Exclude 100 % values *within* each year
dist_2020 = df.loc[df["Margin2020"] != 100, "Margin2020"]
dist_2024 = df.loc[df["Margin2024"] != 100, "Margin2024"]

# ---------- 2  Color categories ----------
winner_color = (189/255, 223/255, 167/255)
cats = [
    ("Near Winner", 0, 5,  (223/255, 240/255, 216/255)),
    ("Contender",   5, 20, (253/255, 245/255, 206/255)),
    ("Competitive", 20, 30,(253/255, 231/255, 208/255)),
    ("Distant",     30, 45,(248/255, 218/255, 205/255)),
    ("Far Behind",  45,100,(242/255, 201/255, 198/255)),
]

# ---------- 3  Plot ----------
fig, ax = plt.subplots(figsize=(6,6))

# Background bands
ax.axhline(0, color=winner_color, linewidth=6, alpha=0.5, zorder=0)
for label, lo, hi, clr in cats:
    ax.axhspan(lo, hi, color=clr, alpha=0.5, zorder=0)

# Violin + box
vp = ax.violinplot([dist_2020, dist_2024], positions=[1,2], widths=.6)
# for body in vp["bodies"]:
#     body.set_facecolor("#F5CFA5")
ax.boxplot(  [dist_2020, dist_2024], positions=[1,2], widths=.18,
            showfliers=False, medianprops=dict(color="black", linewidth=2))

ax.set_xticks([1,2]); ax.set_xticklabels(["2020","2024"])
ax.set_ylabel("Margin of Victory (%)")
ax.set_ylim(-1,100)
ax.set_title("Victory Margin Distribution: Alaska")
ax.grid(axis="y", linestyle=":", alpha=.6)

from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=winner_color, alpha=.5, label="Winner (0 %)")]
legend_handles += [Patch(facecolor=c[3], alpha=.5, label=c[0]) for c in cats]
ax.legend(handles=legend_handles, frameon=False, bbox_to_anchor=(1.05,1), loc="upper left")

plt.tight_layout()

# ---------- 4  Export ----------
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/violin_bands_alaska.pdf")
fig.savefig("outputs/violin_bands_alaska.png", dpi=300)
print("Figure saved to outputs/violin_bands_alaska.(pdf|png)")
