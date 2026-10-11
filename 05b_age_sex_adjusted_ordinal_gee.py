from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, warnings, numpy as np, pandas as pd
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.genmod.cov_struct import OrdinalIndependence
from statsmodels.stats.multitest import multipletests

# ============================================================
# STEP 5B. Age/sex-adjusted ordinal GEE
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
geometry_path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")

df = pd.read_csv(geometry_path); master = pd.read_csv(master_path)

features = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
labels = {"ln_bbox_volume_mm3":"Log bbox volume", "bbox_anisotropy":"BBox anisotropy", "AP_dominance_ratio":"AP dominance",
          "center_SI_norm":"SI location", "center_AP_norm":"AP location", "medial_score":"Medial score"}
grade_order = {"1":1,"2A":2,"2B":3,"3":4}

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
covars = master[["scan_id","PatientAge","PatientSex"]].copy()
df = df.merge(covars, on="scan_id", how="left", validate="many_to_one")

df["grade"] = df["grade"].astype(str); df["grade_ord"] = df["grade"].map(grade_order)
df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10

sex = df["PatientSex"].astype(str).str.strip().str.lower()
df["Male"] = sex.map({"m":1,"male":1,"f":0,"female":0})

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df[features + ["grade_ord","Age_per10y","Male"]].notna().all().all()

print("=" * 110); print("STEP 5B — Age/sex-adjusted ordinal GEE"); print("=" * 110)
print(f"Lesions = {len(df)} | clusters = {df['scan_id'].nunique()}")
print(f"Age = {df.groupby('scan_id')['PatientAge'].first().astype(float).mean():.1f} years (subject-level mean)")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
results = []

for feat in features:
    d = df[["scan_id","grade_ord",feat,"Age_per10y","Male"]].dropna().copy()
    mu, sd = d[feat].mean(), d[feat].std(ddof=1)
    d["x_z"] = (d[feat] - mu) / sd

    X = d[["x_z","Age_per10y","Male"]]
    model = OrdinalGEE(endog=d["grade_ord"], exog=X, groups=d["scan_id"], cov_struct=OrdinalIndependence())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(maxiter=100)

    beta, se, p = fit.params["x_z"], fit.bse["x_z"], fit.pvalues["x_z"]
    lo, hi = fit.conf_int().loc["x_z"]

    results.append({
        "Feature":feat, "Label":labels[feat], "N_lesions":len(d), "N_clusters":d["scan_id"].nunique(),
        "Beta_per_SD_adjusted":beta, "SE_robust":se,
        "OR_per_SD_adjusted":np.exp(beta), "OR95_low":np.exp(lo), "OR95_high":np.exp(hi), "P_raw":p,
        "Age_beta":fit.params["Age_per10y"], "Age_P":fit.pvalues["Age_per10y"],
        "Male_beta":fit.params["Male"], "Male_P":fit.pvalues["Male"]
    })

df_adj = pd.DataFrame(results)
df_adj["P_FDR"] = multipletests(df_adj["P_raw"], method="fdr_bh")[1]

print("\n" + "=" * 110); print("STEP 5B-A — Adjusted geometry coefficients"); print("=" * 110)
print(df_adj[["Label","Beta_per_SD_adjusted","OR_per_SD_adjusted","OR95_low","OR95_high","P_raw","P_FDR"]]
      .round({"Beta_per_SD_adjusted":3,"OR_per_SD_adjusted":3,"OR95_low":3,"OR95_high":3,"P_raw":4,"P_FDR":4}).to_string(index=False))

print("\n" + "=" * 110); print("STEP 5B-B — Adjustment terms"); print("=" * 110)
print(df_adj[["Label","Age_beta","Age_P","Male_beta","Male_P"]]
      .round({"Age_beta":3,"Age_P":4,"Male_beta":3,"Male_P":4}).to_string(index=False))

out = os.path.join(analysis_dir, "STEP5B_geometry_grade_ordinal_GEE_age_sex_adjusted.csv")
df_adj.to_csv(out, index=False)

print("\n" + "=" * 110)
print("✅ STEP 5B COMPLETE")
print(f"Saved: {out}")
print("=" * 110)
