from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

# ============================================================
# STEP 4C. Geometry redundancy audit
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
df = pd.read_csv(path)

# Raw geometry + derived geometry + spatial location
features = [
    "extent_SI_mm", "extent_AP_mm", "extent_ML_mm",
    "ln_bbox_volume_mm3", "max_extent_mm", "min_extent_mm",
    "bbox_anisotropy", "AP_dominance_ratio",
    "center_SI_norm", "center_AP_norm", "medial_score"
]

labels = {
    "extent_SI_mm":"SI extent", "extent_AP_mm":"AP extent", "extent_ML_mm":"ML extent",
    "ln_bbox_volume_mm3":"Log bbox volume", "max_extent_mm":"Maximum extent", "min_extent_mm":"Minimum extent",
    "bbox_anisotropy":"BBox anisotropy", "AP_dominance_ratio":"AP dominance",
    "center_SI_norm":"SI location", "center_AP_norm":"AP location", "medial_score":"Medial score"
}

# ============================================================
# 1. Spearman correlation matrix
# ============================================================
rho = df[features].corr(method="spearman")
rho_named = rho.rename(index=labels, columns=labels)

print("=" * 110); print("STEP 4C-A — Spearman geometry correlation matrix"); print("=" * 110)
print(rho_named.round(2).to_string())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        a, b = features[i], features[j]
        r, p = spearmanr(df[a], df[b], nan_policy="omit")
        pairs.append({"Feature_A":a, "Feature_B":b, "Label_A":labels[a], "Label_B":labels[b],
                      "Spearman_rho":r, "Abs_rho":abs(r), "P":p})

df_pairs = pd.DataFrame(pairs).sort_values("Abs_rho", ascending=False).reset_index(drop=True)

print("\n" + "=" * 110); print("STEP 4C-B — Strong correlations |rho| >= 0.70"); print("=" * 110)
strong = df_pairs[df_pairs["Abs_rho"] >= 0.70]
if len(strong):
    print(strong[["Label_A","Label_B","Spearman_rho","P"]].round({"Spearman_rho":3,"P":5}).to_string(index=False))
else:
    print("No feature pairs with |rho| >= 0.70")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
core_candidates = [
    "ln_bbox_volume_mm3",
    "bbox_anisotropy",
    "AP_dominance_ratio",
    "center_SI_norm",
    "center_AP_norm",
    "medial_score"
]

core_rho = df[core_candidates].corr(method="spearman").rename(
    index=labels, columns=labels
)

print("\n" + "=" * 110); print("STEP 4C-C — Candidate low-redundancy feature set"); print("=" * 110)
print(core_rho.round(2).to_string())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\nMaximum absolute correlation within candidate set:")
for feat in core_candidates:
    others = [x for x in core_candidates if x != feat]
    max_r = df[others].corrwith(df[feat], method="spearman").abs().max()
    max_with = df[others].corrwith(df[feat], method="spearman").abs().idxmax()
    print(f"{labels[feat]:<22}: max |rho| = {max_r:.3f} with {labels[max_with]}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\n" + "=" * 110); print("STEP 4C-D — Core candidate distributions"); print("=" * 110)
summary = df[core_candidates].agg(["mean","std","median","min","max"]).T
summary.columns = ["Mean","SD","Median","Min","Max"]
print(summary.round(3).to_string())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
rho_path = os.path.join(analysis_dir, "STEP4C_geometry_spearman_matrix.csv")
pair_path = os.path.join(analysis_dir, "STEP4C_geometry_pairwise_correlations.csv")
core_path = os.path.join(analysis_dir, "STEP4C_core_candidate_spearman_matrix.csv")

rho.to_csv(rho_path)
df_pairs.to_csv(pair_path, index=False)
df[core_candidates].corr(method="spearman").to_csv(core_path)

print("\n" + "=" * 110)
print("✅ STEP 4C COMPLETE")
print("Full correlation matrix :", rho_path)
print("Pairwise correlations   :", pair_path)
print("Core candidate matrix   :", core_path)
print("=" * 110)
