from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

# ==================================================================================================
# STEP 7F-FINAL v2 — Clean publication figure without text overlap
# ==================================================================================================
analysis_dir = str(ANALYSIS_DIR)
step7c = os.path.join(analysis_dir, "STEP7C_compartment_matched_unadjusted_association.csv")
step7d = os.path.join(analysis_dir, "STEP7D_compartment_association_age_sex_adjusted.csv")
step7e = os.path.join(analysis_dir, "STEP7E_compartment_interaction_GEE_summary.csv")

out_png = os.path.join(analysis_dir, "STEP7F_Figure_Compartment_Meniscus_Cartilage_FINAL.png")
out_pdf = os.path.join(analysis_dir, "STEP7F_Figure_Compartment_Meniscus_Cartilage_FINAL.pdf")
out_tif = os.path.join(analysis_dir, "STEP7F_Figure_Compartment_Meniscus_Cartilage_FINAL.tif")

u, a, g = pd.read_csv(step7c), pd.read_csv(step7d), pd.read_csv(step7e)
assert len(u) == 2 and set(u["Compartment"]) == {"Medial", "Lateral"}
u = u.set_index("Compartment")
u["BH_q"] = multipletests(u["Fisher_P"].values, method="fdr_bh")[1]

# ==================================================================================================
# Helpers
# ==================================================================================================
def p3(x):
    d = Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return "<.001" if d < Decimal("0.001") else f"{d:.3f}".lstrip("0")

def ci_text(est, lo, hi): return f"{est:.2f} ({lo:.2f}–{hi:.2f})"

# ==================================================================================================
# Forest data
# ==================================================================================================
forest = []
for comp in ["Medial", "Lateral"]:
    ru = u.loc[comp]; ra = a[(a["Compartment"] == comp) & (a["Predictor"] == "Meniscal lesion")]
    assert len(ra) == 1
    ra = ra.iloc[0]
    forest += [
        {"Compartment":comp, "Model":"Unadjusted", "OR":ru["Conditional_OR"], "Low":ru["OR95_low"], "High":ru["OR95_high"], "P":ru["Fisher_P"]},
        {"Compartment":comp, "Model":"Age/sex adjusted", "OR":ra["OR"], "Low":ra["CI95_low"], "High":ra["CI95_high"], "P":ra["P"]}
    ]
forest = pd.DataFrame(forest)

interaction = g[g["Effect"].astype(str).str.contains("interaction", case=False, na=False)]
assert len(interaction) == 1
interaction = interaction.iloc[0]

# ==================================================================================================
# Panel A prevalence + Wilson CI
# ==================================================================================================
pa = {}
for comp in ["Medial", "Lateral"]:
    r = u.loc[comp]; ep, en = int(r["a_exp_pos"]), int(r["Exposure_N"]); up, un = int(r["c_unexp_pos"]), int(r["Unexposed_N"])
    elo, ehi = proportion_confint(ep, en, method="wilson"); ulo, uhi = proportion_confint(up, un, method="wilson")
    pa[comp] = dict(pos=100*ep/en, neg=100*up/un, pos_n=ep, pos_N=en, neg_n=up, neg_N=un,
                    pos_lo=100*elo, pos_hi=100*ehi, neg_lo=100*ulo, neg_hi=100*uhi,
                    PR=float(r["PR"]), PRlo=float(r["PR95_low"]), PRhi=float(r["PR95_high"]), q=float(r["BH_q"]))

# ==================================================================================================
# Style
# ==================================================================================================
plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":7.4, "axes.titlesize":9.0, "axes.labelsize":8.0, "xtick.labelsize":7.2,
    "ytick.labelsize":7.2, "legend.fontsize":7.0, "axes.linewidth":0.7,
    "pdf.fonttype":42, "ps.fonttype":42
})

c_pos, c_neg, c_unadj, c_adj = "#24577A", "#B8BEC4", "#666666", "#17365D"
c_text, c_grid, c_sep = "#222222", "#DDDDDD", "#D0D0D0"

fig = plt.figure(figsize=(7.6, 4.35), dpi=180, facecolor="white")
outer = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.60], left=0.075, right=0.985, top=0.79, bottom=0.22, wspace=0.31)

# ==================================================================================================
# PANEL A
# ==================================================================================================
ax1 = fig.add_subplot(outer[0, 0])
x, width, comps = np.arange(2), 0.29, ["Medial", "Lateral"]

pos = np.array([pa[c]["pos"] for c in comps]); neg = np.array([pa[c]["neg"] for c in comps])
pos_lo = np.array([pa[c]["pos_lo"] for c in comps]); pos_hi = np.array([pa[c]["pos_hi"] for c in comps])
neg_lo = np.array([pa[c]["neg_lo"] for c in comps]); neg_hi = np.array([pa[c]["neg_hi"] for c in comps])

xp, xn = x-width/2-0.02, x+width/2+0.02
ax1.bar(xp, pos, width, color=c_pos, edgecolor="none", label="Meniscal lesion +", zorder=3)
ax1.bar(xn, neg, width, color=c_neg, edgecolor="none", label="Meniscal lesion −", zorder=3)

ax1.errorbar(xp, pos, yerr=np.vstack([pos-pos_lo, pos_hi-pos]), fmt="none", ecolor="#333333", elinewidth=0.85, capsize=2.3, zorder=4)
ax1.errorbar(xn, neg, yerr=np.vstack([neg-neg_lo, neg_hi-neg]), fmt="none", ecolor="#666666", elinewidth=0.85, capsize=2.3, zorder=4)

ymax = max(47, float(max(pos_hi.max(), neg_hi.max()) + 10))
stat_y = ymax - 3.2

for j, comp in enumerate(comps):
    r = pa[comp]
    ax1.text(xp[j], r["pos_hi"]+0.7, f'{r["pos"]:.1f}%\n({r["pos_n"]}/{r["pos_N"]})', ha="center", va="bottom", fontsize=6.7, linespacing=1.0)
    ax1.text(xn[j], r["neg_hi"]+0.7, f'{r["neg"]:.1f}%\n({r["neg_n"]}/{r["neg_N"]})', ha="center", va="bottom", fontsize=6.7, linespacing=1.0)
    ax1.text(j, stat_y, f'PR {r["PR"]:.2f} ({r["PRlo"]:.2f}–{r["PRhi"]:.2f})\nFDR q={p3(r["q"])}',
             ha="center", va="center", fontsize=6.4, color="#333333", linespacing=1.15)

ax1.set_ylim(0, ymax); ax1.set_xticks(x); ax1.set_xticklabels(["Medial", "Lateral"])
ax1.set_ylabel("High-grade femoral cartilage lesion (%)")
ax1.set_title("A  High-grade cartilage prevalence", loc="left", fontweight="bold", y=1.15, pad=0)
ax1.legend(loc="lower center", bbox_to_anchor=(0.52, 1.025), frameon=False, ncol=2, columnspacing=1.1, handletextpad=0.4)

ax1.yaxis.grid(True, color=c_grid, linewidth=0.5); ax1.xaxis.grid(False); ax1.set_axisbelow(True); ax1.tick_params(axis="x", length=0)
ax1.spines[["top","right"]].set_visible(False)

# ==================================================================================================
# PANEL B — four independent columns eliminate overlap
# ==================================================================================================
inner = outer[0, 1].subgridspec(1, 4, width_ratios=[0.70, 1.22, 2.30, 1.45], wspace=0.035)
ax_sub, ax_model, ax_forest, ax_num = [fig.add_subplot(inner[0, k]) for k in range(4)]

yrows = [3.10, 2.35, 1.15, 0.40]
rows = [("Medial","Unadjusted"), ("Medial","Age/sex adjusted"), ("Lateral","Unadjusted"), ("Lateral","Age/sex adjusted")]
ymin, ymax2 = 0.05, 4.05

for ax in [ax_sub, ax_model, ax_forest, ax_num]: ax.set_ylim(ymin, ymax2)

# Panel title — safely above all columns
ax_sub.text(0.00, 1.145, "B  Compartment-specific association", transform=ax_sub.transAxes,
            ha="left", va="bottom", fontsize=9.0, fontweight="bold", clip_on=False)

# Interaction — separate line below title, centered over forest area
int_txt = f'Meniscus × compartment interaction: OR ratio {interaction["OR"]:.2f} ({interaction["CI95_low"]:.2f}–{interaction["CI95_high"]:.2f}); P={p3(interaction["P"])}'
ax_forest.text(0.50, 1.065, int_txt, transform=ax_forest.transAxes, ha="center", va="bottom", fontsize=6.4, color="#333333", clip_on=False)

# --------------------------------------------------------------------------------------------------
# Subgroup column
# --------------------------------------------------------------------------------------------------
ax_sub.axis("off")
ax_sub.text(0.00, 3.78, "Subgroup", fontsize=6.6, fontweight="bold", ha="left", va="center")
ax_sub.text(0.00, 2.72, "Medial", fontsize=7.3, fontweight="bold", ha="left", va="center")
ax_sub.text(0.00, 0.78, "Lateral", fontsize=7.3, fontweight="bold", ha="left", va="center")

# --------------------------------------------------------------------------------------------------
# Model column
# --------------------------------------------------------------------------------------------------
ax_model.axis("off")
ax_model.text(0.00, 3.78, "Model", fontsize=6.6, fontweight="bold", ha="left", va="center")
for (_, model), y in zip(rows, yrows): ax_model.text(0.00, y, model, fontsize=6.7, ha="left", va="center", color="#333333")

# --------------------------------------------------------------------------------------------------
# Forest
# --------------------------------------------------------------------------------------------------
ax_forest.set_xscale("log"); ax_forest.set_xlim(0.25, 30); ax_forest.set_ylim(ymin, ymax2)

for (comp, model), y in zip(rows, yrows):
    r = forest[(forest["Compartment"] == comp) & (forest["Model"] == model)].iloc[0]
    color, marker = (c_unadj, "o") if model == "Unadjusted" else (c_adj, "s")
    ax_forest.errorbar(r["OR"], y, xerr=[[r["OR"]-r["Low"]], [r["High"]-r["OR"]]], fmt=marker,
                       color=color, ecolor=color, elinewidth=1.15, capsize=2.3, markersize=4.4, markeredgewidth=0, zorder=4)

ax_forest.axvline(1, color="#444444", linestyle="--", linewidth=0.75)
ax_forest.set_xticks([0.25,0.5,1,2,5,10,20]); ax_forest.set_xticklabels(["0.25","0.5","1","2","5","10","20"])
ax_forest.set_xlabel("Odds ratio"); ax_forest.set_yticks([])
ax_forest.xaxis.grid(True, color=c_grid, linewidth=0.45); ax_forest.set_axisbelow(True)
ax_forest.spines[["top","right","left"]].set_visible(False)

# --------------------------------------------------------------------------------------------------
# Numerical OR column
# --------------------------------------------------------------------------------------------------
ax_num.axis("off")
ax_num.text(0.02, 3.78, "OR (95% CI)", fontsize=6.6, fontweight="bold", ha="left", va="center")
for (comp, model), y in zip(rows, yrows):
    r = forest[(forest["Compartment"] == comp) & (forest["Model"] == model)].iloc[0]
    ax_num.text(0.02, y, ci_text(r["OR"], r["Low"], r["High"]), fontsize=6.7, ha="left", va="center")

# --------------------------------------------------------------------------------------------------
# Horizontal separators across all four columns
# --------------------------------------------------------------------------------------------------
for ax in [ax_sub, ax_model, ax_forest, ax_num]:
    for yy in [3.57, 1.75]: ax.axhline(yy, color=c_sep, linewidth=0.55)

# ==================================================================================================
# Footnote — only two compact lines
# ==================================================================================================
fig.text(0.075, 0.092, "High-grade femoral cartilage lesion = Grade 2B/3. Bars show prevalence with Wilson 95% CIs; values above bars are % (n/N).",
         ha="left", va="bottom", fontsize=6.1, color="#555555")

fig.text(0.075, 0.052, "PR = prevalence ratio; q = BH-FDR-adjusted Fisher P value. Unadjusted ORs are conditional exact estimates; adjusted ORs include age and sex. Interaction P tests medial–lateral heterogeneity.",
         ha="left", va="bottom", fontsize=6.1, color="#555555")

# ==================================================================================================
# Save
# ==================================================================================================
fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
fig.savefig(out_tif, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression":"tiff_lzw"})

plt.show(); plt.close(fig)

print("="*105)
print("✅ STEP 7F-FINAL v2 COMPLETE")
print("PNG :", out_png)
print("PDF :", out_pdf)
print("TIFF:", out_tif)
print("="*105)
