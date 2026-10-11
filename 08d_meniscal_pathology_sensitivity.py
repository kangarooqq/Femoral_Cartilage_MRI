from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==================================================================================================
# STEP 8D — Compartment-matched meniscal pathology sensitivity analysis
# ==================================================================================================
D = str(ANALYSIS_DIR)
path = os.path.join(D, "STEP8A_integrated_lesion_level_master_112_FINAL.csv")

df = pd.read_csv(path).sort_values(["scan_id","split","annotation_id"]).reset_index(drop=True)

core = ["ln_bbox_volume_mm3","center_SI_norm","center_AP_norm"]
core_z = [c + "_z" for c in core]
covars = ["age_per10y","male"]
men_var = "ipsilateral_meniscal_lesion"

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df["grade_order_locked"].isin([1,2,3,4]).all()
assert set(core + covars + [men_var]).issubset(df.columns)

# ==================================================================================================
# 1. Clean / standardize exactly as Step 8B
# ==================================================================================================
def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

df[men_var] = to_bool(df[men_var])
assert df[men_var].notna().all()

df["compartment_matched_meniscal_lesion"] = df[men_var].astype(int)

scale_rows = []
for c in core:
    mu, sd = df[c].mean(), df[c].std(ddof=1)
    df[c + "_z"] = (df[c] - mu) / sd
    scale_rows.append({"Variable":c,"Mean":mu,"SD":sd})

assert df[core_z + covars + ["compartment_matched_meniscal_lesion"]].notna().all().all()

y = df["grade_order_locked"].astype(int)
groups = df["scan_id"]

# ==================================================================================================
# 2. Ordinal GEE helper
# ==================================================================================================
def fit_gee(predictors):
    fit = OrdinalGEE(
        endog=y,
        exog=df[predictors].astype(float),
        groups=groups,
        cov_struct=sm.cov_struct.OrdinalIndependence()
    ).fit(maxiter=100, ctol=1e-8, cov_type="robust")

    assert fit.converged
    assert np.isfinite(fit.params).all()
    assert np.isfinite(fit.bse).all()
    return fit

# Baseline = exact Step 8B structure
base_fit = fit_gee(core_z + covars)

# Step 8D sensitivity model
men_fit = fit_gee(core_z + covars + ["compartment_matched_meniscal_lesion"])

# ==================================================================================================
# 3. Data audit
# ==================================================================================================
print("=" * 125)
print("STEP 8D — DATA / EXPOSURE AUDIT")
print("=" * 125)
print(f"Lesions / scans                         : {len(df)} / {df['scan_id'].nunique()}")
print(f"Compartment-matched meniscal lesion +  : {int(df['compartment_matched_meniscal_lesion'].sum())} lesion-rows")
print(f"Compartment-matched meniscal lesion -  : {int((1-df['compartment_matched_meniscal_lesion']).sum())} lesion-rows")

print("\nGrade × compartment-matched meniscal lesion:")
print(pd.crosstab(df["grade"], df["compartment_matched_meniscal_lesion"]).reindex(["1","2A","2B","3"], fill_value=0))

# ==================================================================================================
# 4. Geometry stability relative to Step 8B baseline
# ==================================================================================================
labels = {
    "ln_bbox_volume_mm3_z":"Log bbox volume",
    "center_SI_norm_z":"SI location",
    "center_AP_norm_z":"AP location"
}

rows = []

for v in core_z:
    b0 = base_fit.params[v]
    b1 = men_fit.params[v]
    lo0, hi0 = base_fit.conf_int().loc[v]
    lo1, hi1 = men_fit.conf_int().loc[v]

    rows.append({
        "Geometry_variable":v,
        "Baseline_OR":np.exp(b0),
        "Baseline_CI_low":np.exp(lo0),
        "Baseline_CI_high":np.exp(hi0),
        "Baseline_P":base_fit.pvalues[v],
        "Meniscus_adjusted_OR":np.exp(b1),
        "Meniscus_adjusted_CI_low":np.exp(lo1),
        "Meniscus_adjusted_CI_high":np.exp(hi1),
        "Meniscus_adjusted_P":men_fit.pvalues[v],
        "Beta_change_pct":100*(b1-b0)/abs(b0),
        "Direction_preserved":np.sign(b1) == np.sign(b0)
    })

stability = pd.DataFrame(rows)

print("\n" + "=" * 125)
print("STEP 8D-A — GEOMETRY STABILITY AFTER MENISCAL-PATHOLOGY ADJUSTMENT")
print("=" * 125)

for _, r in stability.iterrows():
    print(
        f"{labels[r['Geometry_variable']]:<17} "
        f"Baseline OR {r['Baseline_OR']:.3f} -> Adjusted OR {r['Meniscus_adjusted_OR']:.3f} "
        f"({r['Meniscus_adjusted_CI_low']:.3f}–{r['Meniscus_adjusted_CI_high']:.3f}), "
        f"P={r['Meniscus_adjusted_P']:.4f}, β change={r['Beta_change_pct']:+.1f}%"
    )

# ==================================================================================================
# 5. Joint 3-df focal-geometry Wald test
# ==================================================================================================
b = men_fit.params.loc[core_z].to_numpy()
V = men_fit.cov_params().loc[core_z, core_z].to_numpy()

assert np.linalg.matrix_rank(V) == 3

wald_chi2 = float(b.T @ np.linalg.pinv(V) @ b)
wald_p = float(chi2.sf(wald_chi2, 3))

joint = pd.DataFrame([{
    "Model":"Core geometry + age + sex + compartment-matched meniscal lesion",
    "Geometry_joint_Wald_chi2":wald_chi2,
    "df":3,
    "P":wald_p
}])

print("\n" + "=" * 125)
print("STEP 8D-B — JOINT GEOMETRY TEST")
print("=" * 125)
print(f"Wald χ²(3) = {wald_chi2:.3f}, P = {wald_p:.4g}")

# ==================================================================================================
# 6. Meniscal-pathology coefficient — secondary sensitivity covariate
# ==================================================================================================
m = "compartment_matched_meniscal_lesion"
beta, p = men_fit.params[m], men_fit.pvalues[m]
lo, hi = men_fit.conf_int().loc[m]

men_result = pd.DataFrame([{
    "Predictor":"Compartment-matched meniscal lesion",
    "Beta":beta,
    "OR":np.exp(beta),
    "CI95_low":np.exp(lo),
    "CI95_high":np.exp(hi),
    "P":p
}])

print("\n" + "=" * 125)
print("STEP 8D-C — MENISCAL COVARIATE (SECONDARY)")
print("=" * 125)
print(men_result.round({"Beta":3,"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

# ==================================================================================================
# 7. VIF / collinearity QC
# ==================================================================================================
predictors = core_z + covars + [m]
X_vif = sm.add_constant(df[predictors].astype(float), has_constant="add")

vif = pd.DataFrame({
    "Predictor":predictors,
    "VIF":[variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
})

print("\n" + "=" * 125)
print("STEP 8D-D — VIF")
print("=" * 125)
print(vif.round(3).to_string(index=False))

# ==================================================================================================
# 8. Full model coefficients for archival purposes
# ==================================================================================================
full_rows = []

for v in predictors:
    beta, p = men_fit.params[v], men_fit.pvalues[v]
    lo, hi = men_fit.conf_int().loc[v]

    full_rows.append({
        "Variable":v,
        "Beta":beta,
        "OR":np.exp(beta),
        "CI95_low":np.exp(lo),
        "CI95_high":np.exp(hi),
        "P":p
    })

full = pd.DataFrame(full_rows)

print("\n" + "=" * 125)
print("STEP 8D-E — FULL SENSITIVITY MODEL")
print("=" * 125)
print(full.round({"Beta":3,"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

print("\nDirection reminder:")
print("SI: lower coordinate = more superior.")
print("AP: lower coordinate = more anterior.")
print("Thus OR<1 for SI/AP indicates association of higher grade with more superior/anterior location.")

# ==================================================================================================
# 9. Hard QC
# ==================================================================================================
assert base_fit.converged and men_fit.converged
assert stability["Direction_preserved"].all()
assert len(stability) == 3
assert np.isfinite(wald_chi2) and np.isfinite(wald_p)

# ==================================================================================================
# 10. Save
# ==================================================================================================
out_stability = os.path.join(D, "STEP8D_meniscal_adjustment_geometry_stability.csv")
out_joint = os.path.join(D, "STEP8D_meniscal_adjustment_joint_Wald.csv")
out_men = os.path.join(D, "STEP8D_compartment_matched_meniscal_coefficient.csv")
out_full = os.path.join(D, "STEP8D_full_ordinal_GEE.csv")
out_vif = os.path.join(D, "STEP8D_predictor_VIF.csv")

stability.to_csv(out_stability, index=False)
joint.to_csv(out_joint, index=False)
men_result.to_csv(out_men, index=False)
full.to_csv(out_full, index=False)
vif.to_csv(out_vif, index=False)

print("\n" + "=" * 125)
print("✅ STEP 8D COMPLETE")
print(f"Geometry stability : {out_stability}")
print(f"Joint Wald         : {out_joint}")
print(f"Meniscal covariate : {out_men}")
print(f"Full model         : {out_full}")
print(f"VIF                : {out_vif}")
print("=" * 125)
