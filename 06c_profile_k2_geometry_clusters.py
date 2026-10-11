from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 6B. Descriptive profiling of stable K=2 geometry clusters
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
geometry_path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
cluster_path = os.path.join(analysis_dir, "STEP6A_geometry_cluster_assignments_REBUILT.csv")

geo = pd.read_csv(geometry_path)
clu = pd.read_csv(cluster_path)

keys = ["scan_id","split","annotation_id"]
features = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
labels = {"ln_bbox_volume_mm3":"Log bbox volume", "bbox_anisotropy":"BBox anisotropy", "AP_dominance_ratio":"AP dominance",
          "center_SI_norm":"SI location", "center_AP_norm":"AP location", "medial_score":"Medial score"}

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
df = geo.merge(clu[keys + ["cluster_K2"]], on=keys, how="left", validate="one_to_one")
assert len(df) == 112 and df["cluster_K2"].notna().all()

print("=" * 110); print("STEP 6B — K=2 geometry cluster profiling"); print("=" * 110)
print("Cluster sizes:")
print(df["cluster_K2"].value_counts().sort_index())

# ============================================================
# 2. Raw-scale descriptive profile
# ============================================================
rows = []

for cluster in sorted(df["cluster_K2"].unique()):
    d = df[df["cluster_K2"] == cluster]

    for feat in features:
        x = d[feat]
        rows.append({
            "Cluster":int(cluster), "Feature":feat, "Label":labels[feat], "N":len(x),
            "Mean":x.mean(), "SD":x.std(ddof=1), "Median":x.median(),
            "Q1":x.quantile(.25), "Q3":x.quantile(.75)
        })

profile = pd.DataFrame(rows)

print("\n" + "=" * 110); print("STEP 6B-A — Raw geometry profiles"); print("=" * 110)

for cluster in sorted(df["cluster_K2"].unique()):
    print(f"\n[Cluster {cluster}]")
    x = profile[profile["Cluster"] == cluster]
    for _, r in x.iterrows():
        print(f"{r['Label']:<20}: mean={r['Mean']:.3f}, median={r['Median']:.3f} [{r['Q1']:.3f}, {r['Q3']:.3f}]")

# ============================================================
# 3. Standardized centroids
# Facilitate identification of variables driving differences between the two clusters
# ============================================================
scaler = StandardScaler()
Z = scaler.fit_transform(df[features])

zdf = pd.DataFrame(Z, columns=features)
zdf["cluster_K2"] = df["cluster_K2"].values

centroids = zdf.groupby("cluster_K2")[features].mean().T
centroids.index = [labels[x] for x in centroids.index]

print("\n" + "=" * 110); print("STEP 6B-B — Standardized cluster centroids"); print("=" * 110)
print(centroids.round(3).to_string())

# Absolute standardized differences between the two clusters
if centroids.shape[1] == 2:
    c0, c1 = centroids.columns
    contrast = pd.DataFrame({
        "Cluster_0_z":centroids[c0],
        "Cluster_1_z":centroids[c1],
        "Absolute_centroid_difference":(centroids[c1] - centroids[c0]).abs()
    }).sort_values("Absolute_centroid_difference", ascending=False)

    print("\n" + "=" * 110); print("STEP 6B-C — Variables driving the K=2 separation"); print("=" * 110)
    print(contrast.round(3).to_string())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\n" + "=" * 110); print("STEP 6B-D — Additional descriptive geometry"); print("=" * 110)

extra = ["extent_SI_mm","extent_AP_mm","extent_ML_mm","bbox_volume_mm3","dominant_axis","compartment","AP_dominant"]

for cluster in sorted(df["cluster_K2"].unique()):
    d = df[df["cluster_K2"] == cluster]

    print(f"\n[Cluster {cluster}] N={len(d)}")
    print(f"SI extent : {d['extent_SI_mm'].median():.2f} mm [{d['extent_SI_mm'].quantile(.25):.2f}, {d['extent_SI_mm'].quantile(.75):.2f}]")
    print(f"AP extent : {d['extent_AP_mm'].median():.2f} mm [{d['extent_AP_mm'].quantile(.25):.2f}, {d['extent_AP_mm'].quantile(.75):.2f}]")
    print(f"ML extent : {d['extent_ML_mm'].median():.2f} mm [{d['extent_ML_mm'].quantile(.25):.2f}, {d['extent_ML_mm'].quantile(.75):.2f}]")
    print(f"BBox volume: {d['bbox_volume_mm3'].median():.1f} mm³ [{d['bbox_volume_mm3'].quantile(.25):.1f}, {d['bbox_volume_mm3'].quantile(.75):.1f}]")
    print("Dominant axis:", d["dominant_axis"].value_counts().to_dict())
    print("Compartment :", d["compartment"].value_counts().to_dict())
    print("AP dominant :", d["AP_dominant"].value_counts().to_dict())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
profile_path = os.path.join(analysis_dir, "STEP6B_K2_geometry_profile.csv")
centroid_path = os.path.join(analysis_dir, "STEP6B_K2_standardized_centroids.csv")
master_path = os.path.join(analysis_dir, "STEP6B_K2_cluster_master_112.csv")

profile.to_csv(profile_path, index=False)
centroids.to_csv(centroid_path)
df.to_csv(master_path, index=False)

print("\n" + "=" * 110)
print("✅ STEP 6B COMPLETE")
print("Geometry profile :", profile_path)
print("Centroids        :", centroid_path)
print("Cluster master   :", master_path)
print("=" * 110)
