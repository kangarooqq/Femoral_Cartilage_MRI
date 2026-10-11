from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, warnings, numpy as np, pandas as pd
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.genmod.cov_struct import OrdinalIndependence
from statsmodels.stats.multitest import multipletests

# ============================================================
# STEP 5C. Annotation-quality sensitivity analysis
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
geometry_path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")

df = pd.read_csv(geometry_path); master = pd.read_csv(master_path)

features = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
labels = {"ln_bbox_volume_mm3":"Log bbox volume", "bbox_anisotropy":"BBox anisotropy", "AP_dominance_ratio":"AP dominance",
          "center_SI_norm":"SI location", "center_AP_norm":"AP location", "medial_score":"Medial score"}
grade_order = {"1":1,"2A":2,"2B":3,"3":4}

# ------------------------------------------------------------
# Covariates
# ------------------------------------------------------------
df = df.merge(master[["scan_id","PatientAge","PatientSex"]], on="scan_id", how="left", validate="many_to_one")
df["grade"] = df["grade"].astype(str); df["grade_ord"] = df["grade"].map(grade_order)
df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10
df["Male"] = df["PatientSex"].astype(str).str.strip().str.lower().map({"m":1,"male":1,"f":0,"female":0})

assert df[features + ["grade_ord","Age_per10y","Male","confidence"]].notna().all().all()

# ------------------------------------------------------------
# Three prespecified datasets
# ------------------------------------------------------------
datasets = {
    "Primary_all": df.copy(),
    "Confidence_ge3": df[df["confidence"] >= 3].copy(),
    "Strict_conf_ge3_no_signed_delta": df[(df["confidence"] >= 3) & (~df["signed_delta_corrected"].astype(bool))].copy()
}

print("=" * 115); print("STEP 5C — Annotation-quality sensitivity analysis"); print("=" * 115)

for name, d in datasets.items():
    print(f"{name:<34}: lesions={len(d):3d}, clusters={d['scan_id'].nunique():2d}, grades={d['grade'].value_counts().reindex(['1','2A','2B','3'], fill_value=0).to_dict()}")

# ------------------------------------------------------------
# Adjusted ordinal GEE function
# ------------------------------------------------------------
rows = []

for dataset_name, dat in datasets.items():
    for feat in features:
        d = dat[["scan_id","grade_ord",feat,"Age_per10y","Male"]].dropna().copy()
        mu, sd = d[feat].mean(), d[feat].std(ddof=1); d["x_z"] = (d[feat] - mu) / sd

        model = OrdinalGEE(d["grade_ord"], d[["x_z","Age_per10y","Male"]], groups=d["scan_id"], cov_struct=OrdinalIndependence())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(maxiter=100)

        beta, p = fit.params["x_z"], fit.pvalues["x_z"]; lo, hi = fit.conf_int().loc["x_z"]

        rows.append({"Dataset":dataset_name, "Feature":feat, "Label":labels[feat],
                     "N_lesions":len(d), "N_clusters":d["scan_id"].nunique(),
                     "OR_per_SD":np.exp(beta), "OR95_low":np.exp(lo), "OR95_high":np.exp(hi), "P_raw":p})

res = pd.DataFrame(rows)
res["P_FDR"] = np.nan

# FDR separately within each prespecified sensitivity dataset
for name in datasets:
    idx = res["Dataset"].eq(name)
    res.loc[idx, "P_FDR"] = multipletests(res.loc[idx, "P_raw"], method="fdr_bh")[1]

print("\n" + "=" * 115); print("Adjusted ordinal GEE sensitivity results"); print("=" * 115)

for name in datasets:
    x = res[res["Dataset"].eq(name)]
    print(f"\n[{name}]")
    print(x[["Label","OR_per_SD","OR95_low","OR95_high","P_raw","P_FDR"]]
          .round({"OR_per_SD":3,"OR95_low":3,"OR95_high":3,"P_raw":4,"P_FDR":4}).to_string(index=False))

# ------------------------------------------------------------
# Compact comparison of the three main signals
# ------------------------------------------------------------
main = ["ln_bbox_volume_mm3","center_SI_norm","center_AP_norm"]

print("\n" + "=" * 115); print("Primary findings across sensitivity datasets"); print("=" * 115)
compact = res[res["Feature"].isin(main)].pivot(index="Label", columns="Dataset", values=["OR_per_SD","P_FDR"])
print(compact.round(4).to_string())

out = os.path.join(analysis_dir, "STEP5C_annotation_quality_sensitivity.csv")
res.to_csv(out, index=False)

print("\n" + "=" * 115)
print("✅ STEP 5C COMPLETE")
print(f"Saved: {out}")
print("=" * 115)
