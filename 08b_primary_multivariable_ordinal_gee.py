from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==================================================================================================
# STEP 8B — Mutually adjusted core focal-geometry Ordinal GEE
# ==================================================================================================
D = str(ANALYSIS_DIR)
path = os.path.join(D, "STEP8A_integrated_lesion_level_master_112_FINAL.csv")

df = pd.read_csv(path).sort_values(["scan_id","split","annotation_id"]).reset_index(drop=True)

core = ["ln_bbox_volume_mm3","center_SI_norm","center_AP_norm"]
covars = ["age_per10y","male"]

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df["grade_order_locked"].isin([1,2,3,4]).all()
assert df[core + covars].notna().all().all()

# ==================================================================================================
# 1. Standardize the 3 geometry predictors only
#    ORs will therefore be reported per 1-SD increase.
# ==================================================================================================
scale_rows = []

for c in core:
    mu, sd = df[c].mean(), df[c].std(ddof=1)
    assert np.isfinite(mu) and np.isfinite(sd) and sd > 0
    df[c + "_z"] = (df[c] - mu) / sd
    scale_rows.append({"Variable":c, "Mean":mu, "SD":sd})

core_z = [c + "_z" for c in core]

print("=" * 120)
print("STEP 8B — DATA / MODEL AUDIT")
print("=" * 120)
print(f"Lesions / scans     : {len(df)} / {df['scan_id'].nunique()}")
print(f"Grade counts        : {df['grade'].value_counts().reindex(['1','2A','2B','3'], fill_value=0).to_dict()}")
print(f"Cluster sizes       : {df.groupby('scan_id').size().value_counts().sort_index().to_dict()}")

print("\nStandardization:")
print(pd.DataFrame(scale_rows).round(4).to_string(index=False))

# ==================================================================================================
# 2. Collinearity audit
# ==================================================================================================
X_vif = sm.add_constant(df[core_z + covars], has_constant="add")
vif = pd.DataFrame({
    "Predictor":core_z + covars,
    "VIF":[variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
})

print("\nVIF:")
print(vif.round(3).to_string(index=False))

rho = df[core].corr(method="spearman")
print("\nCore geometry Spearman correlations:")
print(rho.round(3).to_string())

# ==================================================================================================
# 3. Ordinal GEE
#
# Outcome coding:
# 1 = Grade 1
# 2 = Grade 2A
# 3 = Grade 2B
# 4 = Grade 3
#
# Positive beta / OR>1 = greater odds of being in a higher grade category.
#
# IMPORTANT:
# Do NOT add an intercept. OrdinalGEE creates its own threshold intercepts.
# ==================================================================================================
X = df[core_z + covars].astype(float)
y = df["grade_order_locked"].astype(int)
groups = df["scan_id"]

model = OrdinalGEE(
    endog=y,
    exog=X,
    groups=groups,
    cov_struct=sm.cov_struct.OrdinalIndependence()
)

fit = model.fit(maxiter=100, ctol=1e-8, cov_type="robust")

assert fit.converged
assert np.isfinite(fit.params).all()
assert np.isfinite(fit.bse).all()

# ==================================================================================================
# 4. Extract coefficients
# ==================================================================================================
labels = {
    "ln_bbox_volume_mm3_z":"Log bbox volume, per 1 SD",
    "center_SI_norm_z":"SI location, per 1 SD increase",
    "center_AP_norm_z":"AP location, per 1 SD increase",
    "age_per10y":"Age, per 10 years",
    "male":"Male vs female"
}

rows = []

for var in core_z + covars:
    beta, se, p = fit.params[var], fit.bse[var], fit.pvalues[var]
    lo, hi = fit.conf_int().loc[var]

    rows.append({
        "Predictor":labels[var],
        "Variable":var,
        "Beta":beta,
        "SE":se,
        "OR":np.exp(beta),
        "CI95_low":np.exp(lo),
        "CI95_high":np.exp(hi),
        "P":p
    })

res = pd.DataFrame(rows)

# BH-FDR across the 3 prespecified geometry coefficients only
geom_mask = res["Variable"].isin(core_z)
res["Geometry_FDR_q"] = np.nan
res.loc[geom_mask, "Geometry_FDR_q"] = multipletests(res.loc[geom_mask, "P"], method="fdr_bh")[1]

# ==================================================================================================
# 5. Joint 3-df Wald test for the geometry block
# ==================================================================================================
b = fit.params.loc[core_z].to_numpy()
V = fit.cov_params().loc[core_z, core_z].to_numpy()

rank = np.linalg.matrix_rank(V)
assert rank == 3

wald_chi2 = float(b.T @ np.linalg.pinv(V) @ b)
wald_df = 3
wald_p = float(chi2.sf(wald_chi2, wald_df))

joint = pd.DataFrame([{
    "Test":"Joint Wald test: three core focal geometry predictors",
    "Chi2":wald_chi2,
    "df":wald_df,
    "P":wald_p
}])

# ==================================================================================================
# 6. Threshold parameters — QC only
# ==================================================================================================
threshold_mask = fit.params.index.str.startswith("I(y>")
thresholds = pd.DataFrame({
    "Threshold":fit.params.index[threshold_mask],
    "Beta":fit.params[threshold_mask].values,
    "SE":fit.bse[threshold_mask].values
})

# ==================================================================================================
# 7. Output
# ==================================================================================================
print("\n" + "=" * 120)
print("STEP 8B-A — MUTUALLY ADJUSTED ORDINAL GEE")
print("=" * 120)

display_cols = ["Predictor","OR","CI95_low","CI95_high","P","Geometry_FDR_q"]
print(res[display_cols].round({
    "OR":3,"CI95_low":3,"CI95_high":3,"P":4,"Geometry_FDR_q":4
}).to_string(index=False))

print("\n" + "=" * 120)
print("STEP 8B-B — JOINT GEOMETRY TEST")
print("=" * 120)
print(f"Wald χ²({wald_df}) = {wald_chi2:.3f}, P = {wald_p:.4g}")

print("\nModel convergence:", fit.converged)
print("Working correlation: Ordinal independence")
print("Covariance estimator: robust sandwich")
print("Clusters:", df["scan_id"].nunique())

# ==================================================================================================
# 8. Direction reminder
# ==================================================================================================
print("\nDirection reminder:")
print("SI coordinate: lower value = more superior; higher value = more inferior.")
print("AP coordinate: lower value = more anterior; higher value = more posterior.")
print("Therefore OR<1 for SI/AP means higher grade is associated with a more superior/anterior location.")

# ==================================================================================================
# 9. Save
# ==================================================================================================
out_res = os.path.join(D, "STEP8B_core_geometry_multivariable_ordinal_GEE.csv")
out_joint = os.path.join(D, "STEP8B_core_geometry_joint_Wald.csv")
out_scale = os.path.join(D, "STEP8B_geometry_standardization_parameters.csv")
out_vif = os.path.join(D, "STEP8B_predictor_VIF.csv")
out_threshold = os.path.join(D, "STEP8B_ordinal_thresholds_QC.csv")

res.to_csv(out_res, index=False)
joint.to_csv(out_joint, index=False)
pd.DataFrame(scale_rows).to_csv(out_scale, index=False)
vif.to_csv(out_vif, index=False)
thresholds.to_csv(out_threshold, index=False)

print("\n" + "=" * 120)
print("✅ STEP 8B COMPLETE")
print(f"Main results       : {out_res}")
print(f"Joint Wald test    : {out_joint}")
print(f"Standardization    : {out_scale}")
print(f"VIF                : {out_vif}")
print(f"Threshold QC       : {out_threshold}")
print("=" * 120)
