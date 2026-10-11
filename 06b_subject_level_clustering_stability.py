from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ============================================================
# STEP 6A-fix. Subject-level subsampling stability
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
df = pd.read_csv(path)

features = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
Xraw = df[features].values
subjects = df["scan_id"].astype(str).values
unique_subjects = np.unique(subjects)

# ------------------------------------------------------------
# 1. Full-data reference clustering
# ------------------------------------------------------------
full_scaler = StandardScaler()
Xfull = full_scaler.fit_transform(Xraw)
references = {}

for k in [2,3,4]:
    km = KMeans(n_clusters=k, n_init=100, random_state=2026)
    references[k] = km.fit_predict(Xfull)

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
rng = np.random.default_rng(2026)
rows = []

for k in [2,3,4]:
    aris = []

    for rep in range(200):
        selected_subjects = rng.choice(unique_subjects, size=int(round(0.80*len(unique_subjects))), replace=False)
        idx = np.isin(subjects, selected_subjects)

        # Fit the scaler only on sampled subjects
        scaler = StandardScaler()
        X_sub = scaler.fit_transform(Xraw[idx])

        km = KMeans(n_clusters=k, n_init=50, random_state=2026+rep)
        km.fit(X_sub)

        # Transform all lesions using the same subsample scaler and predict clusters in the full sample
        X_all_using_sub_scaler = scaler.transform(Xraw)
        predicted_full = km.predict(X_all_using_sub_scaler)

        aris.append(adjusted_rand_score(references[k], predicted_full))

    rows.append({
        "K":k, "ARI_mean":np.mean(aris), "ARI_SD":np.std(aris,ddof=1),
        "ARI_median":np.median(aris), "ARI_P10":np.percentile(aris,10),
        "ARI_P90":np.percentile(aris,90), "N_subjects_total":len(unique_subjects),
        "N_subjects_per_rep":int(round(0.80*len(unique_subjects)))
    })

res = pd.DataFrame(rows)

print("=" * 105)
print("STEP 6A-fix — Subject-level 80% subsampling stability")
print("=" * 105)
print(f"Subjects = {len(unique_subjects)} | Lesions = {len(df)}")
print(res.round(3).to_string(index=False))

out = os.path.join(analysis_dir, "STEP6A_subject_level_clustering_stability.csv")
res.to_csv(out, index=False)

print("\n" + "=" * 105)
print("✅ STEP 6A-fix COMPLETE")
print(f"Saved: {out}")
print("=" * 105)
