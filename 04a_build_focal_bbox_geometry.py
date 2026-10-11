from project_paths import DATA_ROOT, ANALYSIS_DIR
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = DATA_ROOT

JSON_PATHS = {
    "train": ROOT / "json" / "train.json",
    "val": ROOT / "json" / "val.json",
    "test": ROOT / "json" / "test.json",
}

OUTPUT = ANALYSIS_DIR / "STEP4A_femoral_cartilage_lesion_bbox_112_REBUILT.csv"

GRADE_MAP = {
    12: "1",
    13: "2A",
    14: "2B",
    15: "3",
}


rows = []

for split, path in JSON_PATHS.items():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_map = {img["id"]: img for img in data["images"]}

    for ann in data["annotations"]:
        if ann.get("tissue_id") != 4:
            continue
        if ann.get("category_id") not in GRADE_MAP:
            continue

        img = image_map[ann["image_id"]]

        bbox = ann["bbox"]
        spacing = img["voxel_spacing"]
        shape = img["matrix_shape"]
        orient = img["orientation"]

        x, y, z, dx, dy, dz = map(float, bbox)
        sx, sy, sz = map(float, spacing)
        nx, ny, nz = map(float, shape)

        rows.append({
            "split": split,
            "annotation_id": ann["id"],
            "scan_id": img["scan_id"],
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "grade": GRADE_MAP[ann["category_id"]],
            "tissue_id": ann["tissue_id"],
            "confidence": ann.get("confidence"),
            "labeler": ann.get("labeler"),
            "orientation_0": orient[0],
            "orientation_1": orient[1],
            "orientation_2": orient[2],
            "spacing_SI_mm": sx,
            "spacing_AP_mm": sy,
            "spacing_ML_mm": sz,
            "matrix_SI": int(nx),
            "matrix_AP": int(ny),
            "matrix_ML": int(nz),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_z": z,
            "bbox_dx": dx,
            "bbox_dy": dy,
            "bbox_dz": dz,
        })


df = (
    pd.DataFrame(rows)
    .sort_values(["scan_id", "annotation_id"])
    .reset_index(drop=True)
)

assert len(df) == 112
assert df["scan_id"].nunique() == 71


# Original signed-delta quantities
df["extent_SI_mm"] = df["bbox_dx"] * df["spacing_SI_mm"]
df["extent_AP_mm"] = df["bbox_dy"] * df["spacing_AP_mm"]
df["extent_ML_mm"] = df["bbox_dz"] * df["spacing_ML_mm"]

df["bbox_volume_mm3"] = (
    df["extent_SI_mm"]
    * df["extent_AP_mm"]
    * df["extent_ML_mm"]
)

df["center_SI_vox"] = df["bbox_x"] + df["bbox_dx"] / 2
df["center_AP_vox"] = df["bbox_y"] + df["bbox_dy"] / 2
df["center_ML_vox"] = df["bbox_z"] + df["bbox_dz"] / 2

df["center_SI_norm"] = df["center_SI_vox"] / df["matrix_SI"]
df["center_AP_norm"] = df["center_AP_vox"] / df["matrix_AP"]
df["center_ML_norm"] = df["center_ML_vox"] / df["matrix_ML"]

df["bbox_positive"] = (
    (df["bbox_dx"] > 0)
    & (df["bbox_dy"] > 0)
    & (df["bbox_dz"] > 0)
)

df["bbox_inside_image"] = (
    (df["bbox_x"] >= 0)
    & (df["bbox_y"] >= 0)
    & (df["bbox_z"] >= 0)
    & (df["bbox_x"] + df["bbox_dx"] <= df["matrix_SI"])
    & (df["bbox_y"] + df["bbox_dy"] <= df["matrix_AP"])
    & (df["bbox_z"] + df["bbox_dz"] <= df["matrix_ML"])
)


# Endpoint normalization for signed bbox deltas
df["SI_lo"] = np.minimum(df["bbox_x"], df["bbox_x"] + df["bbox_dx"])
df["SI_hi"] = np.maximum(df["bbox_x"], df["bbox_x"] + df["bbox_dx"])

df["AP_lo"] = np.minimum(df["bbox_y"], df["bbox_y"] + df["bbox_dy"])
df["AP_hi"] = np.maximum(df["bbox_y"], df["bbox_y"] + df["bbox_dy"])

df["ML_lo"] = np.minimum(df["bbox_z"], df["bbox_z"] + df["bbox_dz"])
df["ML_hi"] = np.maximum(df["bbox_z"], df["bbox_z"] + df["bbox_dz"])

df["extent_SI_vox"] = df["SI_hi"] - df["SI_lo"]
df["extent_AP_vox"] = df["AP_hi"] - df["AP_lo"]
df["extent_ML_vox"] = df["ML_hi"] - df["ML_lo"]

df["extent_SI_mm"] = df["extent_SI_vox"] * df["spacing_SI_mm"]
df["extent_AP_mm"] = df["extent_AP_vox"] * df["spacing_AP_mm"]
df["extent_ML_mm"] = df["extent_ML_vox"] * df["spacing_ML_mm"]

df["bbox_volume_mm3"] = (
    df["extent_SI_mm"]
    * df["extent_AP_mm"]
    * df["extent_ML_mm"]
)

df["center_SI_vox"] = (df["SI_lo"] + df["SI_hi"]) / 2
df["center_AP_vox"] = (df["AP_lo"] + df["AP_hi"]) / 2
df["center_ML_vox"] = (df["ML_lo"] + df["ML_hi"]) / 2

df["center_SI_norm"] = df["center_SI_vox"] / df["matrix_SI"]
df["center_AP_norm"] = df["center_AP_vox"] / df["matrix_AP"]
df["center_ML_norm"] = df["center_ML_vox"] / df["matrix_ML"]
df["center_ML_norm_raw"] = df["center_ML_norm"]

df["bbox_inside_image_fixed"] = (
    (df["SI_lo"] >= 0)
    & (df["SI_hi"] <= df["matrix_SI"])
    & (df["AP_lo"] >= 0)
    & (df["AP_hi"] <= df["matrix_AP"])
    & (df["ML_lo"] >= 0)
    & (df["ML_hi"] <= df["matrix_ML"])
)

df["bbox_positive_fixed"] = (
    (df["extent_SI_vox"] > 0)
    & (df["extent_AP_vox"] > 0)
    & (df["extent_ML_vox"] > 0)
)

df["signed_delta_corrected"] = (
    (df["bbox_dx"] < 0)
    | (df["bbox_dy"] < 0)
    | (df["bbox_dz"] < 0)
)


assert df["bbox_positive_fixed"].all()
assert df["bbox_inside_image_fixed"].all()

df.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")
print(f"Annotations: {len(df)}")
print(f"Scans: {df['scan_id'].nunique()}")
print(f"Signed-delta corrected: {int(df['signed_delta_corrected'].sum())}")
