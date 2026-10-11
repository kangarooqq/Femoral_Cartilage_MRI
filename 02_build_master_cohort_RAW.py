from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, json
import numpy as np
import pandas as pd
# Comment translated from the original Chinese annotation.
# ============================================================
folder = str(DATA_ROOT); analysis_dir = str(ANALYSIS_DIR); os.makedirs(analysis_dir, exist_ok=True)
path_meta = os.path.join(folder, "all_metadata.csv")
path_cartilage = os.path.join(analysis_dir, "SKM_TEA_femoral_cartilage_results_100K_20260925.csv")
json_paths = {"train": os.path.join(folder, "json", "train.json"), "val": os.path.join(folder, "json", "val.json"), "test": os.path.join(folder, "json", "test.json")}

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
json_data = {}
for split, path in json_paths.items():
    with open(path, "r", encoding="utf-8") as f: json_data[split] = json.load(f)

df_meta = pd.read_csv(path_meta); df_meta.columns = df_meta.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
df_cart_qc = pd.read_csv(path_cartilage); df_cart_qc.columns = df_cart_qc.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

# Official dictionaries across the three splits must be identical
ref_tissues = {(x["id"], x["name"]) for x in json_data["train"]["tissues"]}
ref_categories = {(x["id"], x["name"]) for x in json_data["train"]["categories"]}
for split in ["val", "test"]:
    assert {(x["id"], x["name"]) for x in json_data[split]["tissues"]} == ref_tissues
    assert {(x["id"], x["name"]) for x in json_data[split]["categories"]} == ref_categories

tissue_map = {x["id"]: x["name"] for x in json_data["train"]["tissues"]}
category_map = {x["id"]: x["name"] for x in json_data["train"]["categories"]}

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
image_rows = []
for split, data in json_data.items():
    for img in data["images"]:
        image_rows.append({"split": split, "image_id": img["id"], "scan_id": img["scan_id"], "subject_id": img.get("subject_id"),
                           "timepoint": img.get("timepoint"), "file_name": img.get("file_name"), "msp_id": img.get("msp_id"),
                           "voxel_spacing": json.dumps(img.get("voxel_spacing")), "matrix_shape": json.dumps(img.get("matrix_shape")),
                           "orientation": json.dumps(img.get("orientation")), "inspected": img.get("inspected")})

df_images = pd.DataFrame(image_rows).sort_values("scan_id").reset_index(drop=True)

assert len(df_images) == 155
assert df_images["scan_id"].nunique() == 155
assert df_images[["split", "image_id"]].duplicated().sum() == 0

print("=" * 95); print("STEP 2A — Official image-level cohort"); print("=" * 95)
print(f"Rows                    : {len(df_images)}")
print(f"Unique scan_id          : {df_images['scan_id'].nunique()}")
print(f"Duplicate scan_id       : {df_images['scan_id'].duplicated().sum()}")
print(f"Unique (split,image_id) : {df_images[['split','image_id']].drop_duplicates().shape[0]}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
annotation_rows = []
for split, data in json_data.items():
    image_map = {img["id"]: img["scan_id"] for img in data["images"]}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_map: raise ValueError(f"{split}: annotation image_id={ann['image_id']} 无对应 image")
        tid, cid = ann.get("tissue_id"), ann.get("category_id")
        annotation_rows.append({"split": split, "annotation_id": ann.get("id"), "image_id": ann["image_id"], "scan_id": image_map[ann["image_id"]],
                                "tissue_id": tid, "tissue_name": tissue_map.get(tid, "Unknown"), "category_id": cid,
                                "category_name": category_map.get(cid, "Unknown"), "bbox": json.dumps(ann.get("bbox")),
                                "confidence": ann.get("confidence"), "labeler": ann.get("labeler")})

df_ann = pd.DataFrame(annotation_rows)

print("\n" + "=" * 95); print("STEP 2B — Official annotation table"); print("=" * 95)
print(f"Total annotations : {len(df_ann)}")
print(f"Annotated scans   : {df_ann['scan_id'].nunique()}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
men_mask = (df_ann["tissue_id"] == 1) & df_ann["category_id"].isin(range(1, 9))
lig_mask = df_ann["tissue_id"].isin([2, 3]) & df_ann["category_id"].isin([9, 10, 11])
cart_mask = df_ann["tissue_id"].isin([4, 5, 6]) & df_ann["category_id"].isin([12, 13, 14, 15])
eff_mask = df_ann["category_id"].eq(16)  # Effusion does not depend on specific cartilage/ligament tissue

df_ann["pathology_group"] = np.select([men_mask, lig_mask, cart_mask, eff_mask], ["meniscal_tear", "ligament_tear", "cartilage_lesion", "effusion"], default="other")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
category_family = np.select([df_ann["category_id"].isin(range(1, 9)), df_ann["category_id"].isin([9,10,11]), df_ann["category_id"].isin([12,13,14,15]), df_ann["category_id"].eq(16)],
                            ["Meniscal Tear", "Ligament Tear", "Cartilage Lesion", "Effusion"], default="Other")
compatible = np.select([df_ann["category_id"].isin(range(1,9)), df_ann["category_id"].isin([9,10,11]), df_ann["category_id"].isin([12,13,14,15]), df_ann["category_id"].eq(16)],
                       [df_ann["tissue_id"].eq(1), df_ann["tissue_id"].isin([2,3]), df_ann["tissue_id"].isin([4,5,6]), True], default=True).astype(bool)

df_ann["category_family"] = category_family; df_ann["tissue_category_compatible"] = compatible
df_mismatch = df_ann[(df_ann["category_family"] != "Other") & (~df_ann["tissue_category_compatible"])].copy()

print("\n" + "=" * 95); print("STEP 2C — Tissue/category compatibility audit"); print("=" * 95)
print(f"Incompatible annotations: {len(df_mismatch)}")
if len(df_mismatch):
    print(df_mismatch[["scan_id","split","annotation_id","tissue_id","tissue_name","category_id","category_name"]].to_string(index=False))
else:
    print("No tissue/category mismatches detected.")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
grade_map = {12: "1", 13: "2A", 14: "2B", 15: "3"}; grade_order = {"0": 0, "1": 1, "2A": 2, "2B": 3, "3": 4}
scan_rows = []

for scan_id in df_images["scan_id"]:
    a = df_ann[df_ann["scan_id"].eq(scan_id)]
    men = a[(a["tissue_id"].eq(1)) & a["category_id"].isin(range(1,9))]
    lig = a[a["tissue_id"].isin([2,3]) & a["category_id"].isin([9,10,11])]
    cart = a[a["tissue_id"].isin([4,5,6]) & a["category_id"].isin([12,13,14,15])]
    eff = a[a["category_id"].eq(16)]

    grades = [grade_map[c] for c in cart["category_id"].tolist()]
    highest = max(grades, key=lambda x: grade_order[x]) if grades else "0"
    valid_pathology = pd.concat([men, lig, cart, eff], axis=0)

    scan_rows.append({
        "scan_id": scan_id,
        "meniscal_tear": len(men) > 0, "ligament_tear": len(lig) > 0, "cartilage_lesion": len(cart) > 0, "effusion": len(eff) > 0,
        "highest_cartilage_grade": highest, "highest_cartilage_grade_order": grade_order[highest],
        "meniscal_annotation_count": len(men), "ligament_annotation_count": len(lig), "cartilage_annotation_count": len(cart), "effusion_annotation_count": len(eff),
        "femoral_cartilage_lesion": ((cart["tissue_id"] == 4).any()), "patellar_cartilage_lesion": ((cart["tissue_id"] == 5).any()), "tibial_cartilage_lesion": ((cart["tissue_id"] == 6).any()),
        "cartilage_grade_labels": "; ".join(sorted(set(grades), key=lambda x: grade_order[x])) if grades else "",
        "lesion_labels": "; ".join(sorted(valid_pathology["category_name"].dropna().unique().tolist())),
        "annotation_warning": scan_id in set(df_mismatch["scan_id"])
    })

df_pathology = pd.DataFrame(scan_rows)

assert len(df_pathology) == 155
assert df_pathology["scan_id"].nunique() == 155
assert (df_pathology["cartilage_lesion"] == df_pathology["highest_cartilage_grade"].ne("0")).all()

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
df_json_master = df_images.merge(df_pathology, on="scan_id", how="left", validate="one_to_one")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
assert "MTR_ID" in df_meta.columns
df_meta2 = df_meta.copy(); df_meta2["MTR_ID"] = df_meta2["MTR_ID"].astype(str).str.strip()
assert df_meta2["MTR_ID"].nunique() == 155 and df_meta2["MTR_ID"].duplicated().sum() == 0

json_ids, meta_ids, cart_ids = set(df_json_master["scan_id"]), set(df_meta2["MTR_ID"]), set(df_cart_qc["scan_id"].astype(str).str.strip())
print("\n" + "=" * 95); print("STEP 2D — Cross-source ID-set audit"); print("=" * 95)
print(f"JSON vs Metadata identical     : {json_ids == meta_ids}")
print(f"JSON vs Morphometry identical : {json_ids == cart_ids}")
print(f"JSON only / Metadata only      : {len(json_ids-meta_ids)} / {len(meta_ids-json_ids)}")
print(f"JSON only / Morphometry only   : {len(json_ids-cart_ids)} / {len(cart_ids-json_ids)}")

assert json_ids == meta_ids == cart_ids

df_master = df_json_master.merge(df_meta2, left_on="scan_id", right_on="MTR_ID", how="left", validate="one_to_one", suffixes=("", "_meta"))
assert len(df_master) == 155 and df_master["scan_id"].nunique() == 155

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\n" + "=" * 95); print("STEP 2E — Reconstructed pathology prevalence"); print("=" * 95)
for col in ["cartilage_lesion","meniscal_tear","ligament_tear","effusion"]:
    n = int(df_master[col].sum()); print(f"{col:<22}: {n:3d} / 155 ({100*n/155:5.1f}%)")

print("\nHighest cartilage grade:")
grade_counts = df_master["highest_cartilage_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0)
for grade, n in grade_counts.items(): print(f"Grade {grade:<2}: {n:3d} / 155 ({100*n/155:5.1f}%)")

print("\nCartilage lesion by tissue:")
for col in ["femoral_cartilage_lesion","patellar_cartilage_lesion","tibial_cartilage_lesion"]:
    print(f"{col:<28}: {int(df_master[col].sum())}")

print("\nAnnotation warnings:", int(df_master["annotation_warning"].sum()))

# Key demographic fields are checked for missingness only
for col in ["PatientAge","PatientSex","MedialDirection"]:
    if col in df_master.columns: print(f"Missing {col:<17}: {df_master[col].isna().sum()}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
ann_path = os.path.join(analysis_dir, "STEP2_official_annotations_long.csv")
pathology_path = os.path.join(analysis_dir, "STEP2_pathology_scan_level_155.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155.csv")
mismatch_path = os.path.join(analysis_dir, "STEP2_annotation_tissue_category_mismatches.csv")

df_ann.to_csv(ann_path, index=False); df_pathology.to_csv(pathology_path, index=False); df_master.to_csv(master_path, index=False); df_mismatch.to_csv(mismatch_path, index=False)

print("\n" + "=" * 95)
print("✅ STEP 2 COMPLETE")
print(f"Annotation long table : {ann_path}")
print(f"Pathology scan table  : {pathology_path}")
print(f"Master cohort         : {master_path}")
print(f"Mismatch audit        : {mismatch_path}")
print("=" * 95)


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 012
# ====================================================================================================
# Perform a strict Step 2F
import os, pandas as pd, numpy as np

analysis_dir = str(ANALYSIS_DIR)
ann_path = os.path.join(analysis_dir, "STEP2_official_annotations_long.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155.csv")

df_ann = pd.read_csv(ann_path)
df_master = pd.read_csv(master_path)

grade_map = {12:"1", 13:"2A", 14:"2B", 15:"3"}; grade_order = {"0":0, "1":1, "2A":2, "2B":3, "3":4}

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
fem = df_ann[(df_ann["tissue_id"] == 4) & (df_ann["category_id"].isin([12,13,14,15]))].copy()
fem["femoral_grade"] = fem["category_id"].map(grade_map)

fem_scan = []
for scan_id in df_master["scan_id"]:
    g = fem.loc[fem["scan_id"] == scan_id, "femoral_grade"].tolist()
    highest = max(g, key=lambda x: grade_order[x]) if g else "0"
    fem_scan.append({"scan_id":scan_id, "highest_femoral_cartilage_grade":highest, "highest_femoral_cartilage_grade_order":grade_order[highest],
                     "femoral_cartilage_annotation_count":len(g)})

df_fem_grade = pd.DataFrame(fem_scan)

# Remove possible old versions and merge again
drop_cols = ["highest_femoral_cartilage_grade","highest_femoral_cartilage_grade_order","femoral_cartilage_annotation_count"]
df_master = df_master.drop(columns=[c for c in drop_cols if c in df_master.columns])
df_master = df_master.merge(df_fem_grade, on="scan_id", how="left", validate="one_to_one")

# Strict consistency check
assert (df_master["femoral_cartilage_lesion"].astype(bool) == df_master["highest_femoral_cartilage_grade"].ne("0")).all()
assert int(df_master["femoral_cartilage_lesion"].sum()) == fem["scan_id"].nunique()

print("=" * 90); print("STEP 2F — Femoral-cartilage-specific pathology audit"); print("=" * 90)
print(f"Femoral cartilage annotations : {len(fem)}")
print(f"Femoral lesion-positive scans : {fem['scan_id'].nunique()}")

print("\nHighest FEMORAL cartilage grade:")
counts = df_master["highest_femoral_cartilage_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0)
for grade, n in counts.items(): print(f"Grade {grade:<2}: {n:3d} / 155 ({100*n/155:5.1f}%)")

print("\nOverall cartilage lesion vs femoral cartilage lesion:")
print(f"Any cartilage tissue lesion : {int(df_master['cartilage_lesion'].sum())}")
print(f"Femoral cartilage lesion    : {int(df_master['femoral_cartilage_lesion'].sum())}")

out = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")
df_master.to_csv(out, index=False)
print(f"\n✅ Saved: {out}")


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 015
# ====================================================================================================
import os, pandas as pd, numpy as np

# ============================================================
