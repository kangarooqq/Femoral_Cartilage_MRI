from project_paths import DATA_ROOT, ANALYSIS_DIR
# Comment translated from the original Chinese annotation.
import os, numpy as np, pandas as pd

# ============================================================
# STEP 4B. Annotation-derived focal femoral lesion geometry
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
bbox_path = os.path.join(analysis_dir, "STEP4A_femoral_cartilage_lesion_bbox_112_REBUILT.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")

df = pd.read_csv(bbox_path); master = pd.read_csv(master_path)
master["scan_id"] = master["scan_id"].astype(str).str.strip(); df["scan_id"] = df["scan_id"].astype(str).str.strip()

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
assert master["scan_id"].nunique() == 155
assert set(master["MedialDirection"].dropna().astype(str).str.upper().unique()).issubset({"L","R"})

df = df.merge(master[["scan_id","MedialDirection"]], on="scan_id", how="left", validate="many_to_one")
df["MedialDirection"] = df["MedialDirection"].astype(str).str.strip().str.upper()
assert df["MedialDirection"].isin(["L","R"]).all()

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
df["signed_delta_corrected"] = (df["bbox_dx"] < 0) | (df["bbox_dy"] < 0) | (df["bbox_dz"] < 0)

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
ext_cols = ["extent_SI_mm","extent_AP_mm","extent_ML_mm"]
df["max_extent_mm"] = df[ext_cols].max(axis=1); df["min_extent_mm"] = df[ext_cols].min(axis=1); df["mid_extent_mm"] = df[ext_cols].apply(lambda r: np.sort(r.values)[1], axis=1)

# Comment translated from the original Chinese annotation.
df["bbox_anisotropy"] = 1 - df["min_extent_mm"] / df["max_extent_mm"]

# Comment translated from the original Chinese annotation.
df["AP_dominance_ratio"] = df["extent_AP_mm"] / df[["extent_SI_mm","extent_ML_mm"]].max(axis=1)
df["AP_dominant"] = df["extent_AP_mm"].ge(df[ext_cols].max(axis=1))

# Direction of the major axis
df["dominant_axis"] = df[ext_cols].idxmax(axis=1).map({"extent_SI_mm":"SI","extent_AP_mm":"AP","extent_ML_mm":"ML"})

# Comment translated from the original Chinese annotation.
assert (df["bbox_volume_mm3"] > 0).all()
df["ln_bbox_volume_mm3"] = np.log(df["bbox_volume_mm3"])

# Annotation grade ordering
grade_order = {"1":1,"2A":2,"2B":3,"3":4}
df["grade"] = df["grade"].astype(str)
df["grade_order"] = df["grade"].map(grade_order)
assert df["grade_order"].notna().all()

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
# Raw coordinate follows the third axis; LR is retained and RL is reversed
raw_ml = df["center_ML_vox"] / df["matrix_ML"]
df["center_LR_norm"] = np.where(df["orientation_2"].eq("LR"), raw_ml, 1 - raw_ml)

# Convert coordinates according to MedialDirection for a unified 0=lateral, 1=medial representation
df["medial_score"] = np.where(df["MedialDirection"].eq("R"), df["center_LR_norm"], 1 - df["center_LR_norm"])
df["compartment"] = np.where(df["medial_score"] >= 0.5, "Medial", "Lateral")

# SI/AP locations are standardized only; no finer anatomical subregion labels are assigned
df["center_SI_norm"] = df["center_SI_vox"] / df["matrix_SI"]
df["center_AP_norm"] = df["center_AP_vox"] / df["matrix_AP"]

# ============================================================
# 5. Hard QC
# ============================================================
assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df["bbox_anisotropy"].between(0,1).all()
assert df["center_LR_norm"].between(0,1).all()
assert df["medial_score"].between(0,1).all()
assert df["center_SI_norm"].between(0,1).all()
assert df["center_AP_norm"].between(0,1).all()
assert df["signed_delta_corrected"].sum() == 3

print("=" * 105); print("STEP 4B — Focal femoral lesion geometry construction"); print("=" * 105)
print(f"Annotations              : {len(df)}")
print(f"Positive scans           : {df['scan_id'].nunique()}")
print(f"Signed-delta corrected   : {df['signed_delta_corrected'].sum()}")

print("\nGrade counts:")
print(df["grade"].value_counts().reindex(["1","2A","2B","3"], fill_value=0))

print("\nCompartment counts — annotation level:")
print(df["compartment"].value_counts())

print("\nOrientation × MedialDirection:")
print(pd.crosstab(df["orientation_2"], df["MedialDirection"]))

print("\nDominant axis:")
print(df["dominant_axis"].value_counts())

print("\nAP-dominant:")
print(df["AP_dominant"].value_counts())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
geometry = ["extent_SI_mm","extent_AP_mm","extent_ML_mm","bbox_volume_mm3","ln_bbox_volume_mm3",
            "max_extent_mm","min_extent_mm","bbox_anisotropy","AP_dominance_ratio",
            "center_SI_norm","center_AP_norm","medial_score"]

print("\n" + "=" * 105); print("Geometry summary"); print("=" * 105)
summary = df[geometry].agg(["mean","std","median","min","max"]).T
summary.columns = ["Mean","SD","Median","Min","Max"]
print(summary.round(3).to_string())

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\n" + "=" * 105); print("Grade × compartment"); print("=" * 105)
print(pd.crosstab(df["grade"], df["compartment"]).reindex(["1","2A","2B","3"], fill_value=0))

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
lesions_per_scan = df.groupby("scan_id").size()

print("\nLesions per positive scan:")
print(lesions_per_scan.value_counts().sort_index())
print(f"Mean lesions/positive scan   : {lesions_per_scan.mean():.2f}")
print(f"Median lesions/positive scan : {lesions_per_scan.median():.0f}")
print(f"Maximum lesions/scan         : {lesions_per_scan.max()}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
out = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
df.to_csv(out, index=False)

print("\n" + "=" * 105)
print("✅ STEP 4B COMPLETE")
print(f"Saved: {out}")
print("=" * 105)
