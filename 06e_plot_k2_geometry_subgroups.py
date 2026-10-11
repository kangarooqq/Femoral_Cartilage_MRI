from project_paths import DATA_ROOT, ANALYSIS_DIR
# Optimize plotting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================================================================================================
# Data and calculation sections unchanged
# ==================================================================================================
analysis_dir = str(ANALYSIS_DIR)
cluster_path = os.path.join(analysis_dir, "STEP6B_K2_cluster_master_112.csv")
gee_path = os.path.join(analysis_dir, "STEP6C_K2_grade_ordinal_GEE.csv")
out_png = os.path.join(analysis_dir, "STEP6D_Figure_K2_Geometric_Subgroups_PERFECT.png")

df = pd.read_csv(cluster_path)
gee = pd.read_csv(gee_path)

cluster_col = "cluster_K2"
grade_col = "grade"
features = ["ln_bbox_volume_mm3", "bbox_anisotropy", "AP_dominance_ratio", "center_SI_norm", "center_AP_norm", "medial_score"]
feature_labels = ["Log BBox Vol", "Anisotropy", "AP Dominance", "SI Location", "AP Location", "Medial Score"]

df[cluster_col] = df[cluster_col].astype(int)
df[grade_col] = df[grade_col].astype(str)

phenotype_map = {0: "Anisotropic / Elongated", 1: "Isotropic / Localized"}
df["phenotype"] = df[cluster_col].map(phenotype_map)

grade_map = {"1":"Grade 1", "2A":"Grade 2A", "2B":"Grade 2B", "3":"Grade 3"}
grade_order = ["Grade 1", "Grade 2A", "Grade 2B", "Grade 3"]
df["grade_display"] = pd.Categorical(df[grade_col].map(grade_map), categories=grade_order, ordered=True)

X = df[features].astype(float).copy()
Xz = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
PC = pca.fit_transform(Xz)
df["PC1"], df["PC2"] = PC[:, 0], PC[:, 1]

pc1_pct = pca.explained_variance_ratio_[0] * 100
pc2_pct = pca.explained_variance_ratio_[1] * 100

zdf = pd.DataFrame(Xz, columns=feature_labels, index=df.index)
zdf["phenotype"] = df["phenotype"].values
centroids = zdf.groupby("phenotype")[feature_labels].mean().reindex([phenotype_map[0], phenotype_map[1]])

grade_counts = pd.crosstab(df["phenotype"], df["grade_display"]).reindex(index=[phenotype_map[0], phenotype_map[1]], columns=grade_order, fill_value=0)
grade_pct = grade_counts.div(grade_counts.sum(axis=1), axis=0) * 100

adjusted = gee[gee["Model"].astype(str).str.strip().str.lower().eq("age/sex adjusted")].iloc[0]
adj_or, adj_lo, adj_hi, adj_p = float(adjusted["OR"]), float(adjusted["CI95_low"]), float(adjusted["CI95_high"]), float(adjusted["P"])
p_text = "P < 0.001" if adj_p < 0.001 else f"P = {adj_p:.3f}"
gee_text = f"Age/sex-adjusted Ordinal GEE: OR = {adj_or:.2f} (95% CI: {adj_lo:.2f}–{adj_hi:.2f}), {p_text}"

# ==================================================================================================
# Optimize plotting settings
# ==================================================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 350,
    "savefig.dpi": 350
})

pheno_colors = {phenotype_map[0]: "#2B5C8F", phenotype_map[1]: "#D95F02"}
grade_colors = {"Grade 1": "#E0F3F8", "Grade 2A": "#91BFDB", "Grade 2B": "#FC8D59", "Grade 3": "#D73027"}

fig = plt.figure(figsize=(15.5, 5.2)) # Slightly increase height to provide space at the top
# Adjust top=0.81 and bottom=0.22 to resolve vertical crowding
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], left=0.06, right=0.98, bottom=0.22, top=0.81, wspace=0.34)

def draw_confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# PANEL A — PCA Projection with 95% Ellipses
# ==================================================================================================
ax1 = fig.add_subplot(gs[0, 0])

for pheno_name in [phenotype_map[0], phenotype_map[1]]:
    sub = df[df["phenotype"] == pheno_name]
    color = pheno_colors[pheno_name]
    
    # Draw 95% confidence ellipse background
    draw_confidence_ellipse(sub["PC1"], sub["PC2"], ax1, n_std=1.96, 
                            facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.2, linestyle="--")
    
    # Draw scatter points
    ax1.scatter(sub["PC1"], sub["PC2"], s=38, alpha=0.85, 
                color=color, edgecolors="white", linewidth=0.5, 
                label=f"{pheno_name} (n={len(sub)})", zorder=3)
    
    # Mark centroids
    cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
    ax1.scatter(cx, cy, marker="D", s=70, color=color, edgecolors="black", linewidth=1.2, zorder=5)

ax1.axhline(0, color="#CCCCCC", linestyle=":", linewidth=0.7, zorder=1)
ax1.axvline(0, color="#CCCCCC", linestyle=":", linewidth=0.7, zorder=1)

ax1.set_xlabel(f"Principal Component 1 ({pc1_pct:.1f}%)", fontweight="bold")
ax1.set_ylabel(f"Principal Component 2 ({pc2_pct:.1f}%)", fontweight="bold")
ax1.set_title("A  PCA projection of phenotypes", loc="left", fontweight="bold")
ax1.legend(frameon=False, loc="upper right", handletextpad=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# PANEL B
ax2 = fig.add_subplot(gs[0, 1])
heat_data = centroids.values
vmax = max(abs(np.nanmin(heat_data)), abs(np.nanmax(heat_data)))
im = ax2.imshow(heat_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

ax2.set_xticks(np.arange(len(feature_labels)))
ax2.set_xticklabels(feature_labels, rotation=35, ha="right", fontsize=8.5)
ax2.set_yticks([0, 1])
ax2.set_yticklabels([phenotype_map[0], phenotype_map[1]], fontweight="bold")
ax2.set_title("B  Standardized feature centroids", loc="left", fontweight="bold", pad=12)

for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        val = heat_data[i, j]
        txt_color = "white" if abs(val) > 0.45 else "#111111"
        ax2.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=txt_color)

cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Z-score", rotation=90, labelpad=6, fontsize=8.5)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.6)
ax2.tick_params(length=0)
for spine in ax2.spines.values(): spine.set_visible(False)

# PANEL C
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(2)
bottom = np.zeros(2)

for grade in grade_order:
    vals = grade_pct[grade].values
    ax3.bar(x, vals, bottom=bottom, width=0.52, color=grade_colors[grade], edgecolor="white", linewidth=0.8, label=grade)
    for i, v in enumerate(vals):
        if v >= 5:
            text_c = "#FFFFFF" if grade in ["Grade 2B", "Grade 3"] else "#222222"
            ax3.text(x[i], bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold", color=text_c)
    bottom += vals

cluster_n = grade_counts.sum(axis=1)
ax3.set_xticks(x)
ax3.set_xticklabels([f"{phenotype_map[0]}\n(n={cluster_n.iloc[0]})", f"{phenotype_map[1]}\n(n={cluster_n.iloc[1]})"])
ax3.set_ylim(0, 100)
ax3.set_ylabel("Proportion (%)", fontweight="bold")
ax3.set_title("C  Cartilage-grade composition", loc="left", fontweight="bold", pad=12)

# Modification: place the GEE annotation below Panel C (below the x-axis) to avoid conflict with the title.
ax3.text(0.5, -0.22, gee_text, transform=ax3.transAxes, ha="center", va="top",
         fontsize=8, color="#222222", bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=0.7))

ax3.legend(title="Grade", frameon=False, bbox_to_anchor=(1.02, 1.00), loc="upper left", handletextpad=0.4)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Place the main title at the top
fig.suptitle("Geometric Phenotypes of Femoral Cartilage Lesions", fontsize=13, fontweight="bold", y=0.96)

fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"✅ 彻底修复重叠 Bug！完美图像已生成：{out_png}")
