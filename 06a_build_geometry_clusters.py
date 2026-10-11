from project_paths import DATA_ROOT, ANALYSIS_DIR
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(
    analysis_dir,
    "STEP4B_femoral_lesion_geometry_112.csv",
)

df = pd.read_csv(path)

features = [
    "ln_bbox_volume_mm3",
    "bbox_anisotropy",
    "AP_dominance_ratio",
    "center_SI_norm",
    "center_AP_norm",
    "medial_score",
]

assert len(df) == 112
assert df["scan_id"].nunique() == 71
assert df[features].notna().all().all()

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

rows = []
models = {}

for k in [2, 3, 4]:
    model = KMeans(
        n_clusters=k,
        n_init=100,
        random_state=2026,
    )
    cluster = model.fit_predict(X)
    models[k] = cluster

    counts = pd.Series(cluster).value_counts()

    rows.append({
        "K": k,
        "Silhouette": silhouette_score(X, cluster),
        "Calinski_Harabasz": calinski_harabasz_score(X, cluster),
        "Davies_Bouldin": davies_bouldin_score(X, cluster),
        "Min_cluster_n": counts.min(),
        "Max_cluster_n": counts.max(),
    })

evaluation = pd.DataFrame(rows)

out_df = df[
    ["scan_id", "split", "annotation_id"] + features
].copy()

for k in [2, 3, 4]:
    out_df[f"cluster_K{k}"] = models[k]

eval_path = os.path.join(
    analysis_dir,
    "STEP6A_clustering_internal_metrics_REBUILT.csv",
)

label_path = os.path.join(
    analysis_dir,
    "STEP6A_geometry_cluster_assignments_REBUILT.csv",
)

evaluation.to_csv(eval_path, index=False)
out_df.to_csv(label_path, index=False)

print(evaluation.round(3).to_string(index=False))

for k in [2, 3, 4]:
    print(f"\nK={k}:")
    print(
        pd.Series(models[k])
        .value_counts()
        .sort_index()
        .to_string()
    )

print(f"\nSaved metrics     : {eval_path}")
print(f"Saved assignments : {label_path}")
