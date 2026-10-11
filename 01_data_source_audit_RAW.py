from project_paths import DATA_ROOT, ANALYSIS_DIR


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 004
# ====================================================================================================
import os, json
import pandas as pd
import numpy as np

# ============================================================
# STEP 1. Define data sources
# ============================================================
folder = str(DATA_ROOT)
analysis_dir = str(ANALYSIS_DIR)
os.makedirs(analysis_dir, exist_ok=True)

path_meta = os.path.join(folder, "all_metadata.csv")
path_cartilage = os.path.join(analysis_dir, "SKM_TEA_femoral_cartilage_results_100K_20260925.csv")
path_lesion_summary = os.path.join(folder, "SKM_TEA_v1.0_patient_lesion_summary.xlsx")  # Used for QC only
json_paths = {
    "train": os.path.join(folder, "json", "train.json"),
    "val": os.path.join(folder, "json", "val.json"),
    "test": os.path.join(folder, "json", "test.json")
}

# ============================================================
# STEP 2. Check whether files exist
# ============================================================
all_paths = {"Metadata": path_meta, "Femoral cartilage morphometry": path_cartilage, "Lesion summary (QC only)": path_lesion_summary, **{f"JSON {k}": v for k, v in json_paths.items()}}
print("=" * 90); print("STEP 1A — File existence audit"); print("=" * 90)
for name, path in all_paths.items():
    print(f"{name:<35} {'OK' if os.path.exists(path) else 'MISSING':<8} {path}")

missing_files = [p for p in all_paths.values() if not os.path.exists(p)]
if missing_files: raise FileNotFoundError(f"存在缺失文件，请先处理：{missing_files}")

# ============================================================
# STEP 3. Read tables and clean column names
# ============================================================
df_meta = pd.read_csv(path_meta)
df_cartilage = pd.read_csv(path_cartilage)
df_lesion_qc = pd.read_excel(path_lesion_summary)

for df in [df_meta, df_cartilage, df_lesion_qc]:
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)

print("\n" + "=" * 90); print("STEP 1B — Tabular source audit"); print("=" * 90)
print(f"Metadata                     : {df_meta.shape}")
print(f"Femoral cartilage morphometry: {df_cartilage.shape}")
print(f"Lesion summary (QC only)     : {df_lesion_qc.shape}")

# ============================================================
# STEP 4. Define primary keys; do not automatically guess the first column
# ============================================================
def find_id_col(df, dataset_name, candidates=("MTR_ID", "scan_id", "subject_id")):
    found = [c for c in candidates if c in df.columns]
    if len(found) == 0: raise ValueError(f"{dataset_name}: 未找到合法主键候选 {candidates}")
    if len(found) > 1: print(f"⚠ {dataset_name}: 存在多个ID字段 {found}，暂使用 {found[0]}")
    return found[0]

id_meta = find_id_col(df_meta, "Metadata")
id_cart = find_id_col(df_cartilage, "Femoral cartilage morphometry")
id_lesion_qc = find_id_col(df_lesion_qc, "Lesion summary")

for df, col in [(df_meta, id_meta), (df_cartilage, id_cart), (df_lesion_qc, id_lesion_qc)]:
    # df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].astype("string").str.strip()

def audit_id(df, id_col, name):
    print(f"\n【{name}】")
    print(f"ID column      : {id_col}")
    print(f"Rows           : {len(df)}")
    print(f"Unique IDs     : {df[id_col].nunique()}")
    print(f"Missing IDs    : {df[id_col].isna().sum()}")
    print(f"Duplicated IDs : {df[id_col].duplicated().sum()}")
    print(f"Examples       : {df[id_col].head().tolist()}")

audit_id(df_meta, id_meta, "Metadata")
audit_id(df_cartilage, id_cart, "Femoral cartilage morphometry")
audit_id(df_lesion_qc, id_lesion_qc, "Lesion summary — QC only")

# ============================================================
# STEP 5. Read official JSON and lock official tissue/category dictionaries
# ============================================================
json_data = {}
for split, path in json_paths.items():
    with open(path, "r", encoding="utf-8") as f:
        json_data[split] = json.load(f)

reference_json = json_data["train"]
df_tissues = pd.DataFrame(reference_json["tissues"])
df_categories = pd.DataFrame(reference_json["categories"])

print("\n" + "=" * 90); print("STEP 1C — Official JSON label dictionaries"); print("=" * 90)
print("\nOfficial tissues:")
print(df_tissues.to_string(index=False))
print("\nOfficial pathology categories:")
print(df_categories[["id", "supercategory", "name"]].to_string(index=False))

# Required official pathology definitions
tissue_map = dict(zip(df_tissues["id"], df_tissues["name"]))
category_map = dict(zip(df_categories["id"], df_categories["name"]))

assert tissue_map[4] == "Femoral Cartilage", f"JSON tissue_id=4 unexpected: {tissue_map.get(4)}"
assert all("Cartilage Lesion" in category_map[i] for i in [12, 13, 14, 15])

print("\n✅ JSON pathology mapping confirmed:")
print("   tissue_id 4  = Femoral Cartilage")
print("   category 12  = Cartilage Lesion (1)")
print("   category 13  = Cartilage Lesion (2A)")
print("   category 14  = Cartilage Lesion (2B)")
print("   category 15  = Cartilage Lesion (3)")

# ============================================================
# STEP 6. Basic counts across the three JSON splits
# ============================================================
print("\n" + "=" * 90); print("STEP 1D — Official JSON split audit"); print("=" * 90)

json_scan_ids = []
for split, data in json_data.items():
    image_ids = [img["id"] for img in data["images"]]
    annotations = data["annotations"]
    json_scan_ids.extend(image_ids)
    print(f"{split:<6}: images={len(data['images']):3d}, annotations={len(annotations):4d}, unique image IDs={len(set(image_ids)):3d}")

print(f"\nCombined JSON image records : {len(json_scan_ids)}")
print(f"Combined unique image IDs   : {len(set(json_scan_ids))}")
print(f"Duplicate image IDs         : {len(json_scan_ids) - len(set(json_scan_ids))}")

# ============================================================
# STEP 7. Lock femoral cartilage morphometry provenance
# ============================================================
required_cart_cols = ["scan_id", "mean_thickness_mm", "median_thickness_mm", "cartilage_volume_mm3", "surface_area_mm2", "P10_thickness_mm", "P90_thickness_mm", "thin_lt_1mm_pct", "label2_voxels"]
missing_cart_cols = [c for c in required_cart_cols if c not in df_cartilage.columns]
if missing_cart_cols: raise ValueError(f"Femoral cartilage morphometry 缺少字段: {missing_cart_cols}")

print("\n" + "=" * 90); print("STEP 1E — Femoral cartilage morphometry audit"); print("=" * 90)
print(f"N subjects          : {df_cartilage['scan_id'].nunique()}")
print(f"Duplicate scan_id   : {df_cartilage['scan_id'].duplicated().sum()}")
print(f"Missing morphometry : {df_cartilage[required_cart_cols[1:]].isna().sum().sum()}")
print("\nReminder: NIfTI segmentation Label 2 = Femoral Cartilage")
print("Previous provenance check: MTR_001 NIfTI Label-2 voxels = CSV label2_voxels = 89,027")

# ============================================================
# STEP 8. Save source manifest for this reconstruction
# ============================================================
source_manifest = pd.DataFrame([
    ["Metadata", path_meta, "Primary", id_meta],
    ["Femoral cartilage morphometry", path_cartilage, "Primary derived imaging measurement", id_cart],
    ["Official pathology JSON — train", json_paths["train"], "Primary pathology annotation", "image.id"],
    ["Official pathology JSON — val", json_paths["val"], "Primary pathology annotation", "image.id"],
    ["Official pathology JSON — test", json_paths["test"], "Primary pathology annotation", "image.id"],
    ["Lesion summary Excel", path_lesion_summary, "QC only", id_lesion_qc]
], columns=["Source", "Path", "Role", "ID_Field"])

manifest_path = os.path.join(analysis_dir, "STEP1_source_manifest.csv")
source_manifest.to_csv(manifest_path, index=False)

print("\n" + "=" * 90)
print("✅ STEP 1 COMPLETE — No merging or statistical analysis performed yet.")
print(f"Source manifest saved: {manifest_path}")
print("=" * 90)


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 009
# ====================================================================================================
import os, json
import numpy as np
import pandas as pd

# ============================================================
