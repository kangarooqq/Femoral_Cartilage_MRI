from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, json, numpy as np, pandas as pd

# ==================================================================================================
# STEP 7B — Official femoral cartilage lesion compartment reconstruction
# ==================================================================================================
folder = str(DATA_ROOT); analysis_dir = str(ANALYSIS_DIR)
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")
meniscus_path = os.path.join(analysis_dir, "STEP7A_meniscal_scan_level_compartments_155_FINAL.csv")
json_paths = {"train":os.path.join(folder,"json","train.json"), "val":os.path.join(folder,"json","val.json"), "test":os.path.join(folder,"json","test.json")}

master = pd.read_csv(master_path); meniscus = pd.read_csv(meniscus_path)
master["scan_id"] = master["scan_id"].astype(str).str.strip(); meniscus["scan_id"] = meniscus["scan_id"].astype(str).str.strip()

assert len(master) == 155 and master["scan_id"].nunique() == 155
assert len(meniscus) == 155 and meniscus["scan_id"].nunique() == 155

grade_map = {12:"1", 13:"2A", 14:"2B", 15:"3"}
grade_order = {"0":0, "1":1, "2A":2, "2B":3, "3":4}
high_grade_categories = {14, 15}  # Grade 2B / Grade 3

# ==================================================================================================
# 1. Strict extraction from official JSON
#    tissue_id = 4 = Femoral Cartilage
#    category_id = 12–15 = Cartilage Lesion grades
# ==================================================================================================
rows = []

for split, path in json_paths.items():
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)

    image_map = {img["id"]:img for img in data["images"]}
    category_map = {x["id"]:x["name"] for x in data["categories"]}

    for ann in data["annotations"]:
        if ann.get("tissue_id") != 4 or ann.get("category_id") not in [12,13,14,15]: continue

        img = image_map[ann["image_id"]]
        bbox, shape, orient = ann.get("bbox"), img.get("matrix_shape"), img.get("orientation")

        if bbox is None or len(bbox) != 6: raise ValueError(f"{img['scan_id']}: invalid bbox={bbox}")
        if shape is None or len(shape) != 3: raise ValueError(f"{img['scan_id']}: invalid matrix_shape={shape}")
        if orient is None or len(orient) != 3: raise ValueError(f"{img['scan_id']}: invalid orientation={orient}")

        x, y, z, dx, dy, dz = map(float, bbox); nx, ny, nz = map(float, shape)

        # Endpoint normalization — preserves the actual annotated region when delta is negative
        si_lo, si_hi = min(x, x+dx), max(x, x+dx)
        ap_lo, ap_hi = min(y, y+dy), max(y, y+dy)
        ml_lo, ml_hi = min(z, z+dz), max(z, z+dz)

        center_si, center_ap, center_ml = (si_lo+si_hi)/2, (ap_lo+ap_hi)/2, (ml_lo+ml_hi)/2
        cid = ann["category_id"]

        rows.append({
            "split":split, "annotation_id":ann.get("id"), "image_id":ann["image_id"], "scan_id":img["scan_id"],
            "tissue_id":ann["tissue_id"], "category_id":cid, "category_name":category_map.get(cid), "grade":grade_map[cid],
            "grade_order":grade_order[grade_map[cid]], "high_grade":cid in high_grade_categories,
            "confidence":ann.get("confidence"), "labeler":ann.get("labeler"),
            "orientation_0":orient[0], "orientation_1":orient[1], "orientation_2":orient[2],
            "matrix_SI":int(nx), "matrix_AP":int(ny), "matrix_ML":int(nz),
            "bbox_x":x, "bbox_y":y, "bbox_z":z, "bbox_dx":dx, "bbox_dy":dy, "bbox_dz":dz,
            "SI_lo":si_lo, "SI_hi":si_hi, "AP_lo":ap_lo, "AP_hi":ap_hi, "ML_lo":ml_lo, "ML_hi":ml_hi,
            "center_SI_vox":center_si, "center_AP_vox":center_ap, "center_ML_vox":center_ml,
            "center_SI_norm":center_si/nx, "center_AP_norm":center_ap/ny, "center_ML_norm_raw":center_ml/nz,
            "signed_delta_corrected":(dx < 0) or (dy < 0) or (dz < 0),
            "bbox_positive_fixed":(si_hi > si_lo) and (ap_hi > ap_lo) and (ml_hi > ml_lo),
            "bbox_inside_image_fixed":si_lo >= 0 and si_hi <= nx and ap_lo >= 0 and ap_hi <= ny and ml_lo >= 0 and ml_hi <= nz
        })

df_fem = pd.DataFrame(rows).sort_values(["scan_id","annotation_id"]).reset_index(drop=True)

# ==================================================================================================
# 2. Add MedialDirection
# ==================================================================================================
df_fem = df_fem.merge(master[["scan_id","MedialDirection"]], on="scan_id", how="left", validate="many_to_one")
df_fem["MedialDirection"] = df_fem["MedialDirection"].astype(str).str.strip().str.upper()

assert df_fem["MedialDirection"].isin(["L","R"]).all()

# ==================================================================================================
# 3. Canonicalize ML coordinate
#    orientation LR → raw coordinate already Left→Right
#    orientation RL → reverse coordinate to canonical Left→Right
# ==================================================================================================
raw_ml = df_fem["center_ML_norm_raw"]
df_fem["center_LR_norm"] = np.where(df_fem["orientation_2"].eq("LR"), raw_ml, 1 - raw_ml)

# ==================================================================================================
# 4. Unified medial score
#    0 = lateral side, 1 = medial side
# ==================================================================================================
df_fem["medial_score"] = np.where(
    df_fem["MedialDirection"].eq("R"),
    df_fem["center_LR_norm"],
    1 - df_fem["center_LR_norm"]
)

df_fem["compartment"] = np.where(df_fem["medial_score"] >= 0.5, "Medial", "Lateral")

# ==================================================================================================
# 5. Annotation-level QC
# ==================================================================================================
print("=" * 115); print("STEP 7B-A — Official femoral cartilage lesion compartment reconstruction"); print("=" * 115)
print(f"Femoral lesion annotations : {len(df_fem)}")
print(f"Femoral-positive scans     : {df_fem['scan_id'].nunique()}")
print(f"Unique (split,ann_id)      : {df_fem[['split','annotation_id']].drop_duplicates().shape[0]}")
print(f"Signed-delta corrected     : {int(df_fem['signed_delta_corrected'].sum())}")
print(f"Invalid bbox dimensions    : {int((~df_fem['bbox_positive_fixed']).sum())}")
print(f"Outside-image bbox         : {int((~df_fem['bbox_inside_image_fixed']).sum())}")

print("\nOrientation combinations:")
print(df_fem.groupby(["orientation_0","orientation_1","orientation_2"]).size())

print("\nAnnotation-level grade counts:")
print(df_fem["grade"].value_counts().reindex(["1","2A","2B","3"], fill_value=0))

print("\nAnnotation-level compartment counts:")
print(df_fem["compartment"].value_counts())

print("\nGrade × compartment — annotation level:")
print(pd.crosstab(df_fem["grade"], df_fem["compartment"]).reindex(["1","2A","2B","3"], fill_value=0))

print("\nHigh-grade (2B/3) annotations:")
print(pd.crosstab(df_fem["compartment"], df_fem["high_grade"]))

# ==================================================================================================
# 6. Hard annotation QC
# ==================================================================================================
assert len(df_fem) == 112
assert df_fem["scan_id"].nunique() == 71
assert df_fem["bbox_positive_fixed"].all()
assert df_fem["bbox_inside_image_fixed"].all()
assert set(df_fem["orientation_0"]) == {"SI"}
assert set(df_fem["orientation_1"]) == {"AP"}
assert set(df_fem["orientation_2"]).issubset({"LR","RL"})
assert df_fem["center_LR_norm"].between(0,1).all()
assert df_fem["medial_score"].between(0,1).all()

# ==================================================================================================
# 7. Build scan-level compartment outcomes for all 155 subjects
# ==================================================================================================
scan_rows = []

for scan_id in master["scan_id"]:
    d = df_fem[df_fem["scan_id"].eq(scan_id)]
    med = d[d["compartment"].eq("Medial")]
    lat = d[d["compartment"].eq("Lateral")]

    def highest_grade(x):
        if len(x) == 0: return "0"
        return max(x["grade"].astype(str).tolist(), key=lambda g: grade_order[g])

    scan_rows.append({
        "scan_id":scan_id,
        "femoral_cartilage_lesion":len(d) > 0,
        "medial_femoral_cartilage_lesion":len(med) > 0,
        "lateral_femoral_cartilage_lesion":len(lat) > 0,
        "both_femoral_compartments":len(med) > 0 and len(lat) > 0,
        "medial_high_grade_femoral_cartilage":med["high_grade"].any() if len(med) else False,
        "lateral_high_grade_femoral_cartilage":lat["high_grade"].any() if len(lat) else False,
        "any_high_grade_femoral_cartilage":d["high_grade"].any() if len(d) else False,
        "highest_medial_femoral_grade":highest_grade(med),
        "highest_lateral_femoral_grade":highest_grade(lat),
        "highest_femoral_grade_rebuilt":highest_grade(d),
        "femoral_annotation_count":len(d),
        "medial_femoral_annotation_count":len(med),
        "lateral_femoral_annotation_count":len(lat),
        "medial_high_grade_annotation_count":int(med["high_grade"].sum()) if len(med) else 0,
        "lateral_high_grade_annotation_count":int(lat["high_grade"].sum()) if len(lat) else 0
    })

df_scan = pd.DataFrame(scan_rows)

assert len(df_scan) == 155 and df_scan["scan_id"].nunique() == 155
assert (df_scan["femoral_cartilage_lesion"] == (df_scan["medial_femoral_cartilage_lesion"] | df_scan["lateral_femoral_cartilage_lesion"])).all()

# ==================================================================================================
# 8. Cross-check against Step 2 femoral-specific pathology
# ==================================================================================================
check = df_scan.merge(
    master[["scan_id","femoral_cartilage_lesion","highest_femoral_cartilage_grade"]],
    on="scan_id", how="left", validate="one_to_one", suffixes=("_rebuilt","_step2")
)

def as_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

status_agreement = check["femoral_cartilage_lesion_rebuilt"].astype(bool) == as_bool(check["femoral_cartilage_lesion_step2"])
grade_agreement = check["highest_femoral_grade_rebuilt"].astype(str) == check["highest_femoral_cartilage_grade"].astype(str)

print("\n" + "=" * 115); print("STEP 7B-B — Scan-level femoral compartment summary"); print("=" * 115)
print(f"Any femoral lesion               : {int(df_scan['femoral_cartilage_lesion'].sum())} / 155")
print(f"Medial femoral lesion            : {int(df_scan['medial_femoral_cartilage_lesion'].sum())} / 155")
print(f"Lateral femoral lesion           : {int(df_scan['lateral_femoral_cartilage_lesion'].sum())} / 155")
print(f"Both femoral compartments        : {int(df_scan['both_femoral_compartments'].sum())} / 155")
print(f"Medial high-grade (2B/3)         : {int(df_scan['medial_high_grade_femoral_cartilage'].sum())} / 155")
print(f"Lateral high-grade (2B/3)        : {int(df_scan['lateral_high_grade_femoral_cartilage'].sum())} / 155")
print(f"Any high-grade femoral cartilage : {int(df_scan['any_high_grade_femoral_cartilage'].sum())} / 155")

print("\nHighest MEDIAL femoral grade:")
print(df_scan["highest_medial_femoral_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0))

print("\nHighest LATERAL femoral grade:")
print(df_scan["highest_lateral_femoral_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0))

print("\nCross-check with Step 2:")
print(f"Femoral lesion-status agreement : {status_agreement.sum()} / 155 ({100*status_agreement.mean():.1f}%)")
print(f"Highest-grade agreement         : {grade_agreement.sum()} / 155 ({100*grade_agreement.mean():.1f}%)")

if not status_agreement.all() or not grade_agreement.all():
    print("\nDiscordant scans:")
    bad = check.loc[~(status_agreement & grade_agreement),
                    ["scan_id","femoral_cartilage_lesion_rebuilt","femoral_cartilage_lesion_step2",
                     "highest_femoral_grade_rebuilt","highest_femoral_cartilage_grade"]]
    print(bad.to_string(index=False))

assert status_agreement.all()
assert grade_agreement.all()

# ==================================================================================================
# 9. Merge Step 7A meniscus + Step 7B femoral cartilage → crosstalk-ready 155-subject master
# ==================================================================================================
men_cols = [
    "scan_id","meniscal_lesion","medial_meniscal_lesion","lateral_meniscal_lesion",
    "meniscal_annotation_count","medial_meniscal_annotation_count","lateral_meniscal_annotation_count"
]

cart_cols = [
    "scan_id","femoral_cartilage_lesion","medial_femoral_cartilage_lesion","lateral_femoral_cartilage_lesion",
    "medial_high_grade_femoral_cartilage","lateral_high_grade_femoral_cartilage","any_high_grade_femoral_cartilage",
    "highest_medial_femoral_grade","highest_lateral_femoral_grade","highest_femoral_grade_rebuilt"
]

df_cross = master[["scan_id","subject_id","PatientAge","PatientSex","MedialDirection"]].merge(
    meniscus[men_cols], on="scan_id", how="left", validate="one_to_one"
).merge(
    df_scan[cart_cols], on="scan_id", how="left", validate="one_to_one"
)

assert len(df_cross) == 155 and df_cross["scan_id"].nunique() == 155
assert df_cross.isna().sum().sum() == 0

# ==================================================================================================
# 10. Simple pairing counts — QC only, NO association testing yet
# ==================================================================================================
print("\n" + "=" * 115); print("STEP 7B-C — Crosstalk-ready compartment counts (QC only)"); print("=" * 115)

print("\nMedial compartment:")
print(f"Medial meniscal lesion                  : {int(df_cross['medial_meniscal_lesion'].sum())}")
print(f"Medial femoral cartilage lesion         : {int(df_cross['medial_femoral_cartilage_lesion'].sum())}")
print(f"Medial high-grade femoral cartilage     : {int(df_cross['medial_high_grade_femoral_cartilage'].sum())}")
print(f"Both medial meniscus + medial high-grade: {int((df_cross['medial_meniscal_lesion'] & df_cross['medial_high_grade_femoral_cartilage']).sum())}")

print("\nLateral compartment:")
print(f"Lateral meniscal lesion                   : {int(df_cross['lateral_meniscal_lesion'].sum())}")
print(f"Lateral femoral cartilage lesion          : {int(df_cross['lateral_femoral_cartilage_lesion'].sum())}")
print(f"Lateral high-grade femoral cartilage      : {int(df_cross['lateral_high_grade_femoral_cartilage'].sum())}")
print(f"Both lateral meniscus + lateral high-grade: {int((df_cross['lateral_meniscal_lesion'] & df_cross['lateral_high_grade_femoral_cartilage']).sum())}")

# ==================================================================================================
# 11. Save
# ==================================================================================================
ann_out = os.path.join(analysis_dir, "STEP7B_femoral_cartilage_annotations_with_compartment_FINAL.csv")
scan_out = os.path.join(analysis_dir, "STEP7B_femoral_cartilage_scan_level_compartments_155_FINAL.csv")
cross_out = os.path.join(analysis_dir, "STEP7B_crosstalk_ready_master_155.csv")

df_fem.to_csv(ann_out, index=False)
df_scan.to_csv(scan_out, index=False)
df_cross.to_csv(cross_out, index=False)

print("\n" + "=" * 115)
print("✅ STEP 7B COMPLETE")
print(f"Annotation-level : {ann_out}")
print(f"Scan-level       : {scan_out}")
print(f"Crosstalk master : {cross_out}")
print("=" * 115)
