from project_paths import DATA_ROOT, ANALYSIS_DIR
# Step 5A：Ordinal GEE
import os, warnings, numpy as np, pandas as pd
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.genmod.cov_struct import OrdinalIndependence
from statsmodels.stats.multitest import multipletests

# ============================================================
# STEP 5A. Focal lesion geometry vs lesion grade — ordinal GEE
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
df = pd.read_csv(path)

features = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
labels = {"ln_bbox_volume_mm3":"Log bbox volume", "bbox_anisotropy":"BBox anisotropy", "AP_dominance_ratio":"AP dominance",
          "center_SI_norm":"SI location", "center_AP_norm":"AP location", "medial_score":"Medial score"}
grade_order = {"1":1,"2A":2,"2B":3,"3":4}

df["grade"] = df["grade"].astype(str); df["grade_ord"] = df["grade"].map(grade_order)
assert len(df) == 112 and df["scan_id"].nunique() == 71 and df["grade_ord"].notna().all()
assert df[features].notna().all().all()

print("=" * 110); print("STEP 5A — Cluster-aware ordinal GEE: lesion geometry vs lesion grade"); print("=" * 110)
print(f"Lesions = {len(df)} | Independent subjects/clusters = {df['scan_id'].nunique()}")
print("Grade counts:", df["grade"].value_counts().reindex(["1","2A","2B","3"], fill_value=0).to_dict())

# ============================================================
# 1. Grade-specific descriptive statistics
# ============================================================
desc_rows = []
for feat in features:
    row = {"Feature":feat, "Label":labels[feat]}
    for g in ["1","2A","2B","3"]:
        x = df.loc[df["grade"].eq(g), feat]
        row[f"G{g}_N"] = len(x); row[f"G{g}_median_IQR"] = f"{x.median():.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]"
    desc_rows.append(row)

df_desc = pd.DataFrame(desc_rows)

print("\n" + "=" * 110); print("STEP 5A-A — Grade-specific descriptive statistics"); print("=" * 110)
print(df_desc[["Label","G1_median_IQR","G2A_median_IQR","G2B_median_IQR","G3_median_IQR"]].to_string(index=False))

# ============================================================
# Comment translated from the original Chinese annotation.
# Note: OrdinalGEE generates threshold intercepts internally; do not add_constant() here
# ============================================================
results = []

for feat in features:
    d = df[["scan_id","grade_ord",feat]].dropna().copy()
    mu, sd = d[feat].mean(), d[feat].std(ddof=1)
    d["x_z"] = (d[feat] - mu) / sd

    model = OrdinalGEE(endog=d["grade_ord"], exog=d[["x_z"]], groups=d["scan_id"], cov_struct=OrdinalIndependence())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(maxiter=100)

    beta, se, p = fit.params["x_z"], fit.bse["x_z"], fit.pvalues["x_z"]
    lo, hi = fit.conf_int().loc["x_z"]
    results.append({"Feature":feat, "Label":labels[feat], "N_lesions":len(d), "N_clusters":d["scan_id"].nunique(),
                    "Raw_mean":mu, "Raw_SD":sd, "Beta_per_SD":beta, "SE_robust":se,
                    "OR_per_SD":np.exp(beta), "OR95_low":np.exp(lo), "OR95_high":np.exp(hi), "P_raw":p})

df_gee = pd.DataFrame(results)
df_gee["P_FDR"] = multipletests(df_gee["P_raw"], method="fdr_bh")[1]

print("\n" + "=" * 110); print("STEP 5A-B — Ordinal GEE results"); print("=" * 110)
print(df_gee[["Label","Beta_per_SD","OR_per_SD","OR95_low","OR95_high","P_raw","P_FDR"]]
      .round({"Beta_per_SD":3,"OR_per_SD":3,"OR95_low":3,"OR95_high":3,"P_raw":4,"P_FDR":4}).to_string(index=False))

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
print("\nInterpretation direction:")
for _, r in df_gee.iterrows():
    direction = "higher grade" if r["Beta_per_SD"] > 0 else "lower grade"
    print(f"{r['Label']:<20}: higher value → {direction} (OR/SD={r['OR_per_SD']:.3f}, q={r['P_FDR']:.4f})")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
desc_path = os.path.join(analysis_dir, "STEP5A_geometry_by_grade_descriptive.csv")
gee_path = os.path.join(analysis_dir, "STEP5A_geometry_grade_ordinal_GEE.csv")
df_desc.to_csv(desc_path, index=False); df_gee.to_csv(gee_path, index=False)

print("\n" + "=" * 110)
print("✅ STEP 5A COMPLETE")
print("Descriptive :", desc_path)
print("Ordinal GEE :", gee_path)
print("=" * 110)
