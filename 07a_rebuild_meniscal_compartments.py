from project_paths import DATA_ROOT, ANALYSIS_DIR
# Comment translated from the original Chinese annotation.
import os, json, numpy as np, pandas as pd

# ==================================================================================================
# STEP 7A — Rebuild meniscal lesion compartment mapping from official JSON
# ==================================================================================================
folder = str(DATA_ROOT); analysis_dir = str(ANALYSIS_DIR)
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")
json_paths = {"train":os.path.join(folder,"json","train.json"), "val":os.path.join(folder,"json","val.json"), "test":os.path.join(folder,"json","test.json")}

master = pd.read_csv(master_path); master["scan_id"] = master["scan_id"].astype(str).str.strip()
assert len(master) == 155 and master["scan_id"].nunique() == 155
assert set(master["MedialDirection"].astype(str).str.strip().str.upper().unique()).issubset({"L","R"})

# ==================================================================================================
# Comment translated from the original Chinese annotation.
#    tissue_id = 1 (Meniscus)
#    category_id = 1–8 (Meniscal Tear subtypes)
# ==================================================================================================
rows, excluded_mismatch = [], []

for split, path in json_paths.items():
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)

    image_map = {img["id"]:img for img in data["images"]}
    category_map = {x["id"]:x["name"] for x in data["categories"]}
    tissue_map = {x["id"]:x["name"] for x in data["tissues"]}

    for ann in data["annotations"]:
        tid, cid = ann.get("tissue_id"), ann.get("category_id")

        # Record annotations with meniscus tissue but categories outside 1–8
        if tid == 1 and cid not in range(1, 9):
            img = image_map[ann["image_id"]]
            excluded_mismatch.append({
                "split":split, "scan_id":img["scan_id"], "annotation_id":ann.get("id"),
                "tissue_id":tid, "tissue_name":tissue_map.get(tid),
                "category_id":cid, "category_name":category_map.get(cid)
            })

        # Formal meniscal lesions must satisfy both tissue and category criteria
        if tid != 1 or cid not in range(1, 9): continue

        img = image_map[ann["image_id"]]
        bbox, spacing, shape, orient = ann.get("bbox"), img.get("voxel_spacing"), img.get("matrix_shape"), img.get("orientation")

        if bbox is None or len(bbox) != 6: raise ValueError(f"{img['scan_id']}: invalid bbox={bbox}")
        if spacing is None or len(spacing) != 3: raise ValueError(f"{img['scan_id']}: invalid spacing={spacing}")
        if shape is None or len(shape) != 3: raise ValueError(f"{img['scan_id']}: invalid matrix_shape={shape}")
        if orient is None or len(orient) != 3: raise ValueError(f"{img['scan_id']}: invalid orientation={orient}")

        x, y, z, dx, dy, dz = map(float, bbox)
        nx, ny, nz = map(float, shape)

        # Endpoint normalization: preserve annotated region even if a delta is negative
        si_lo, si_hi = min(x, x+dx), max(x, x+dx)
        ap_lo, ap_hi = min(y, y+dy), max(y, y+dy)
        ml_lo, ml_hi = min(z, z+dz), max(z, z+dz)

        center_si, center_ap, center_ml = (si_lo+si_hi)/2, (ap_lo+ap_hi)/2, (ml_lo+ml_hi)/2

        rows.append({
            "split":split, "annotation_id":ann.get("id"), "image_id":ann["image_id"], "scan_id":img["scan_id"],
            "tissue_id":tid, "tissue_name":tissue_map.get(tid), "category_id":cid, "category_name":category_map.get(cid),
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

df_men = pd.DataFrame(rows).sort_values(["scan_id","annotation_id"]).reset_index(drop=True)
df_excluded = pd.DataFrame(excluded_mismatch)

# ==================================================================================================
# 2. Merge MedialDirection
# ==================================================================================================
df_men = df_men.merge(master[["scan_id","MedialDirection"]], on="scan_id", how="left", validate="many_to_one")
df_men["MedialDirection"] = df_men["MedialDirection"].astype(str).str.strip().str.upper()

assert df_men["MedialDirection"].isin(["L","R"]).all()

# ==================================================================================================
# 3. Canonicalize third-axis coordinate to Left → Right
#
# orientation LR : raw 0→1 already Left→Right
# orientation RL : reverse it, so canonical coordinate is again Left→Right
# ==================================================================================================
raw_ml = df_men["center_ML_norm_raw"]

df_men["center_LR_norm"] = np.where(
    df_men["orientation_2"].eq("LR"),
    raw_ml,
    1 - raw_ml
)

# ==================================================================================================
# 4. Convert canonical L→R coordinate into a unified medial score
#
# MedialDirection = R : higher L→R coordinate is more medial
# MedialDirection = L : lower L→R coordinate is more medial
#
# medial_score: 0 = maximally lateral side, 1 = maximally medial side
# ==================================================================================================
df_men["medial_score"] = np.where(
    df_men["MedialDirection"].eq("R"),
    df_men["center_LR_norm"],
    1 - df_men["center_LR_norm"]
)

df_men["compartment"] = np.where(df_men["medial_score"] >= 0.5, "Medial", "Lateral")

# ==================================================================================================
# 5. Hard QC
# ==================================================================================================
print("=" * 110); print("STEP 7A — Official meniscal lesion extraction"); print("=" * 110)
print(f"Meniscal annotations       : {len(df_men)}")
print(f"Meniscal-positive scans    : {df_men['scan_id'].nunique()}")
print(f"Unique (split,ann_id)      : {df_men[['split','annotation_id']].drop_duplicates().shape[0]}")
print(f"Signed-delta corrected     : {df_men['signed_delta_corrected'].sum()}")
print(f"Invalid dimensions         : {(~df_men['bbox_positive_fixed']).sum()}")
print(f"Outside image              : {(~df_men['bbox_inside_image_fixed']).sum()}")
print(f"Missing MedialDirection    : {df_men['MedialDirection'].isna().sum()}")

print("\nOrientation combinations:")
print(df_men.groupby(["orientation_0","orientation_1","orientation_2"]).size())

print("\nMeniscal category counts:")
print(df_men.groupby(["category_id","category_name"]).size().to_string())

print("\nCompartment counts — annotation level:")
print(df_men["compartment"].value_counts())

print("\nOrientation × compartment:")
print(pd.crosstab(df_men["orientation_2"], df_men["compartment"]))

print("\nMedialDirection × compartment:")
print(pd.crosstab(df_men["MedialDirection"], df_men["compartment"]))

print("\nMedial score summary:")
print(df_men["medial_score"].describe().round(3).to_string())

# ==================================================================================================
# 6. Explicitly audit excluded Meniscus-tissue mismatches
# ==================================================================================================
print("\n" + "=" * 110); print("STEP 7A — Excluded Meniscus-tissue annotations with non-meniscal categories"); print("=" * 110)

if len(df_excluded):
    print(df_excluded.to_string(index=False))
else:
    print("None")

# ==================================================================================================
# 7. Scan-level compartment summary
#    One scan may contain both medial and lateral meniscal lesions
# ==================================================================================================
scan_rows = []

for scan_id in master["scan_id"]:
    d = df_men[df_men["scan_id"].eq(scan_id)]

    scan_rows.append({
        "scan_id":scan_id,
        "meniscal_lesion":len(d) > 0,
        "medial_meniscal_lesion":d["compartment"].eq("Medial").any(),
        "lateral_meniscal_lesion":d["compartment"].eq("Lateral").any(),
        "meniscal_annotation_count":len(d),
        "medial_meniscal_annotation_count":int(d["compartment"].eq("Medial").sum()),
        "lateral_meniscal_annotation_count":int(d["compartment"].eq("Lateral").sum())
    })

df_scan = pd.DataFrame(scan_rows)

assert len(df_scan) == 155 and df_scan["scan_id"].nunique() == 155
assert (df_scan["meniscal_lesion"] == (df_scan["medial_meniscal_lesion"] | df_scan["lateral_meniscal_lesion"])).all()

print("\n" + "=" * 110); print("STEP 7A — Scan-level compartment summary"); print("=" * 110)
print(f"Any meniscal lesion    : {int(df_scan['meniscal_lesion'].sum())} / 155")
print(f"Medial meniscal lesion : {int(df_scan['medial_meniscal_lesion'].sum())} / 155")
print(f"Lateral meniscal lesion: {int(df_scan['lateral_meniscal_lesion'].sum())} / 155")
print(f"Both compartments      : {int((df_scan['medial_meniscal_lesion'] & df_scan['lateral_meniscal_lesion']).sum())} / 155")

print("\nAnnotations per positive scan:")
print(df_scan.loc[df_scan["meniscal_lesion"], "meniscal_annotation_count"].value_counts().sort_index())

# ==================================================================================================
# 8. Cross-check against official scan-level meniscal flag reconstructed in Step 2
# ==================================================================================================
check = df_scan.merge(master[["scan_id","meniscal_tear"]], on="scan_id", how="left", validate="one_to_one")
official_flag = check["meniscal_tear"].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False})

agreement = (check["meniscal_lesion"] == official_flag)

print("\n" + "=" * 110); print("STEP 7A — Cross-check with Step 2 pathology flag"); print("=" * 110)
print(f"Agreement : {agreement.sum()} / {len(agreement)} ({100*agreement.mean():.1f}%)")

if not agreement.all():
    print("\nDiscordant scans:")
    print(check.loc[~agreement, ["scan_id","meniscal_lesion","meniscal_tear"]].to_string(index=False))

# ==================================================================================================
# 9. Final hard assertions
# ==================================================================================================
assert df_men["bbox_positive_fixed"].all(), "仍存在无效 meniscal bbox"
assert int((~df_men["bbox_inside_image_fixed"]).sum()) == 1
assert set(df_men["orientation_0"]) == {"SI"}
assert set(df_men["orientation_1"]) == {"AP"}
assert set(df_men["orientation_2"]).issubset({"LR","RL"})
assert df_men["center_LR_norm"].between(0,1).all()
assert df_men["medial_score"].between(0,1).all()
assert agreement.all(), "Step 7A meniscal status 与 Step 2 官方重建结果不一致"

# ==================================================================================================
# 10. Save
# ==================================================================================================
ann_out = os.path.join(analysis_dir, "STEP7A_meniscal_annotations_with_compartment.csv")
scan_out = os.path.join(analysis_dir, "STEP7A_meniscal_scan_level_compartments_155.csv")
excluded_out = os.path.join(analysis_dir, "STEP7A_excluded_meniscus_tissue_mismatches.csv")

df_men.to_csv(ann_out, index=False)
df_scan.to_csv(scan_out, index=False)
df_excluded.to_csv(excluded_out, index=False)

print("\n" + "=" * 110)
print("✅ STEP 7A COMPLETE")
print(f"Annotation-level : {ann_out}")
print(f"Scan-level       : {scan_out}")
print(f"Excluded audit   : {excluded_out}")
print("=" * 110)

# ===== FINAL QC LOCK =====

# Step 7A final: lock the QC status
# ==================================================================================================
# STEP 7A-final — Lock meniscal compartment dataset with explicit bbox QC flags
# ==================================================================================================
import os, numpy as np, pandas as pd

analysis_dir = str(ANALYSIS_DIR)

# Comment translated from the original Chinese annotation.
df_men["bbox_geometry_valid"] = df_men["bbox_positive_fixed"] & df_men["bbox_inside_image_fixed"]

# Comment translated from the original Chinese annotation.
df_men["ML_center_valid"] = (
    (df_men["center_ML_vox"] >= 0) &
    (df_men["center_ML_vox"] <= df_men["matrix_ML"])
)

df_men["compartment_valid"] = (
    df_men["ML_center_valid"] &
    df_men["orientation_2"].isin(["LR","RL"]) &
    df_men["MedialDirection"].isin(["L","R"]) &
    df_men["medial_score"].between(0,1)
)

# Comment translated from the original Chinese annotation.
df_men["bbox_qc_note"] = np.where(
    df_men["bbox_geometry_valid"],
    "",
    "Official annotation bbox extends outside image; retained for compartment mapping only"
)

print("=" * 110)
print("STEP 7A-final — Meniscal annotation QC lock")
print("=" * 110)
print(f"Total meniscal annotations       : {len(df_men)}")
print(f"Valid full 3D bbox geometry      : {int(df_men['bbox_geometry_valid'].sum())} / {len(df_men)}")
print(f"Invalid full 3D bbox geometry    : {int((~df_men['bbox_geometry_valid']).sum())}")
print(f"Valid compartment assignments   : {int(df_men['compartment_valid'].sum())} / {len(df_men)}")

print("\nInvalid bbox annotation(s):")
print(df_men.loc[~df_men["bbox_geometry_valid"],
    ["scan_id","annotation_id","category_name","confidence","orientation_2",
     "MedialDirection","center_ML_vox","matrix_ML","medial_score","compartment","bbox_qc_note"]
].to_string(index=False))

# Comment translated from the original Chinese annotation.
assert len(df_men) == 107
assert df_men["scan_id"].nunique() == 68
assert df_men["compartment_valid"].all()
assert int((~df_men["bbox_geometry_valid"]).sum()) == 1

# Comment translated from the original Chinese annotation.
assert int(df_scan["meniscal_lesion"].sum()) == 68
assert int(df_scan["medial_meniscal_lesion"].sum()) == 48
assert int(df_scan["lateral_meniscal_lesion"].sum()) == 33
assert int((df_scan["medial_meniscal_lesion"] & df_scan["lateral_meniscal_lesion"]).sum()) == 13

# Comment translated from the original Chinese annotation.
ann_out = os.path.join(analysis_dir, "STEP7A_meniscal_annotations_with_compartment_FINAL.csv")
scan_out = os.path.join(analysis_dir, "STEP7A_meniscal_scan_level_compartments_155_FINAL.csv")

df_men.to_csv(ann_out, index=False)
df_scan.to_csv(scan_out, index=False)

print("\n" + "=" * 110)
print("✅ STEP 7A FINAL COMPLETE")
print("107/107 annotations have valid compartment assignments.")
print("106/107 have valid complete 3D bbox geometry.")
print("MTR_047 retained for compartment analysis but excluded from any full-bbox geometry analysis.")
print(f"Annotation file : {ann_out}")
print(f"Scan-level file : {scan_out}")
print("=" * 110)
