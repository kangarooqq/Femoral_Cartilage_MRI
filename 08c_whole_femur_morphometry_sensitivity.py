from project_paths import DATA_ROOT, ANALYSIS_DIR
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE

import os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE

# ==================================================================================================
# STEP 8C — Whole-femur morphometry sensitivity analysis
# Each model = 3 core focal geometry variables + age + sex + ONE global morphometry metric
# ==================================================================================================
D = str(ANALYSIS_DIR)
path = os.path.join(D, "STEP8A_integrated_lesion_level_master_112_FINAL.csv")

df = pd.read_csv(path).sort_values(["scan_id","split","annotation_id"]).reset_index(drop=True)

core = ["ln_bbox_volume_mm3","center_SI_norm","center_AP_norm"]
covars = ["age_per10y","male"]
morph = [
    "global_mean_thickness_mm",
    "global_median_thickness_mm",
    "global_cartilage_volume_mm3",
    "global_surface_area_mm2",
    "global_p10_thickness_mm",
    "global_p90_thickness_mm",
    "global_thin_lt1mm_pct"
]

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df["grade_order_locked"].isin([1,2,3,4]).all()
assert df[core + covars + morph].notna().all().all()

# ==================================================================================================
# 1. Standardize geometry exactly as in Step 8B; standardize each global metric per 1 SD
# ==================================================================================================
for c in core + morph:
    mu, sd = df[c].mean(), df[c].std(ddof=1)
    assert np.isfinite(mu) and np.isfinite(sd) and sd > 0
    df[c + "_z"] = (df[c] - mu) / sd

core_z = [c + "_z" for c in core]
morph_z = [c + "_z" for c in morph]

y = df["grade_order_locked"].astype(int)
groups = df["scan_id"]

# ==================================================================================================
# 2. Helper: fit Ordinal GEE
# ==================================================================================================
def fit_ordinal_gee(predictors):
    X = df[predictors].astype(float)

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

    return fit

# ==================================================================================================
# 3. Refit Step 8B baseline inside this script
# ==================================================================================================
baseline_predictors = core_z + covars
base_fit = fit_ordinal_gee(baseline_predictors)

base_beta = base_fit.params.loc[core_z].copy()

print("=" * 125)
print("STEP 8C — BASELINE REPRODUCTION")
print("=" * 125)

for v in core_z:
    b = base_fit.params[v]
    lo, hi = base_fit.conf_int().loc[v]
    print(f"{v:<28} OR={np.exp(b):.3f}  95% CI {np.exp(lo):.3f}–{np.exp(hi):.3f}  P={base_fit.pvalues[v]:.4f}")

# ==================================================================================================
# 4. Fit seven one-at-a-time whole-femur morphometry sensitivity models
# ==================================================================================================
detail_rows, joint_rows, morph_rows = [], [], []

for raw_morph, mz in zip(morph, morph_z):

    fit = fit_ordinal_gee(core_z + covars + [mz])

    # ----------------------------------------------------------------------------------------------
    # Geometry coefficients
    # ----------------------------------------------------------------------------------------------
    for v in core_z:
        beta, se, p = fit.params[v], fit.bse[v], fit.pvalues[v]
        lo, hi = fit.conf_int().loc[v]

        beta_change_pct = 100 * (beta - base_beta[v]) / abs(base_beta[v])

        detail_rows.append({
            "Added_global_metric":raw_morph,
            "Geometry_variable":v,
            "Baseline_beta":base_beta[v],
            "Sensitivity_beta":beta,
            "Beta_change_pct":beta_change_pct,
            "OR":np.exp(beta),
            "CI95_low":np.exp(lo),
            "CI95_high":np.exp(hi),
            "P":p,
            "Direction_same_as_baseline":np.sign(beta) == np.sign(base_beta[v]),
            "Converged":bool(fit.converged)
        })

    # ----------------------------------------------------------------------------------------------
    # Joint 3-df geometry Wald test
    # ----------------------------------------------------------------------------------------------
    b = fit.params.loc[core_z].to_numpy()
    V = fit.cov_params().loc[core_z, core_z].to_numpy()

    assert np.linalg.matrix_rank(V) == 3

    wald = float(b.T @ np.linalg.pinv(V) @ b)
    joint_rows.append({
        "Added_global_metric":raw_morph,
        "Geometry_joint_Wald_chi2":wald,
        "df":3,
        "P":float(chi2.sf(wald, 3))
    })

    # ----------------------------------------------------------------------------------------------
    # Added global morphometry coefficient — secondary/descriptive only
    # ----------------------------------------------------------------------------------------------
    beta, p = fit.params[mz], fit.pvalues[mz]
    lo, hi = fit.conf_int().loc[mz]

    morph_rows.append({
        "Global_metric":raw_morph,
        "Beta_per_1SD":beta,
        "OR_per_1SD":np.exp(beta),
        "CI95_low":np.exp(lo),
        "CI95_high":np.exp(hi),
        "P":p
    })

detail = pd.DataFrame(detail_rows)
joint = pd.DataFrame(joint_rows)
morph_results = pd.DataFrame(morph_rows)

# ==================================================================================================
# 5. Stability summary for each geometry predictor
# ==================================================================================================
summary_rows = []

for v in core_z:
    d = detail[detail["Geometry_variable"].eq(v)].copy()

    summary_rows.append({
        "Geometry_variable":v,
        "Baseline_OR":np.exp(base_beta[v]),
        "OR_min_across_7_models":d["OR"].min(),
        "OR_max_across_7_models":d["OR"].max(),
        "Max_abs_beta_change_pct":d["Beta_change_pct"].abs().max(),
        "Direction_preserved_models":int(d["Direction_same_as_baseline"].sum()),
        "Models_P_lt_0.05":int((d["P"] < 0.05).sum()),
        "Models_CI_excludes_1":int(((d["CI95_low"] > 1) | (d["CI95_high"] < 1)).sum())
    })

stability = pd.DataFrame(summary_rows)

# ==================================================================================================
# 6. Compact wide table for direct reading
# ==================================================================================================
name_map = {
    "ln_bbox_volume_mm3_z":"Log bbox volume",
    "center_SI_norm_z":"SI location",
    "center_AP_norm_z":"AP location"
}

wide_rows = []

for m in morph:
    row = {"Added_global_metric":m}

    for v in core_z:
        r = detail[(detail["Added_global_metric"].eq(m)) & (detail["Geometry_variable"].eq(v))].iloc[0]
        short = name_map[v]

        row[f"{short}_OR"] = r["OR"]
        row[f"{short}_CI_low"] = r["CI95_low"]
        row[f"{short}_CI_high"] = r["CI95_high"]
        row[f"{short}_P"] = r["P"]
        row[f"{short}_beta_change_pct"] = r["Beta_change_pct"]

    j = joint[joint["Added_global_metric"].eq(m)].iloc[0]
    row["Geometry_joint_Wald_P"] = j["P"]

    wide_rows.append(row)

wide = pd.DataFrame(wide_rows)

# ==================================================================================================
# 7. Output
# ==================================================================================================
print("\n" + "=" * 125)
print("STEP 8C-A — GEOMETRY STABILITY ACROSS 7 WHOLE-FEMUR MORPHOMETRY MODELS")
print("=" * 125)

for m in morph:
    print(f"\n[{m}]")

    d = detail[detail["Added_global_metric"].eq(m)]

    for v in core_z:
        r = d[d["Geometry_variable"].eq(v)].iloc[0]
        print(
            f"{name_map[v]:<17} "
            f"OR {r['OR']:.3f} ({r['CI95_low']:.3f}–{r['CI95_high']:.3f}), "
            f"P={r['P']:.4f}, β change={r['Beta_change_pct']:+.1f}%"
        )

    jp = joint.loc[joint["Added_global_metric"].eq(m), "P"].iloc[0]
    print(f"Joint geometry Wald P = {jp:.4g}")

print("\n" + "=" * 125)
print("STEP 8C-B — STABILITY SUMMARY")
print("=" * 125)
print(stability.round({
    "Baseline_OR":3,
    "OR_min_across_7_models":3,
    "OR_max_across_7_models":3,
    "Max_abs_beta_change_pct":1
}).to_string(index=False))

print("\n" + "=" * 125)
print("STEP 8C-C — ADDED WHOLE-FEMUR MORPHOMETRY COEFFICIENTS (SECONDARY)")
print("=" * 125)
print(morph_results.round({
    "Beta_per_1SD":3,
    "OR_per_1SD":3,
    "CI95_low":3,
    "CI95_high":3,
    "P":4
}).to_string(index=False))

# ==================================================================================================
# 8. Hard QC
# ==================================================================================================
assert len(detail) == 21
assert len(joint) == 7
assert len(morph_results) == 7
assert detail["Converged"].all()
assert detail["Direction_same_as_baseline"].groupby(detail["Geometry_variable"]).sum().min() >= 0

print("\nAll seven sensitivity models converged.")

# ==================================================================================================
# 9. Save
# ==================================================================================================
out_detail = os.path.join(D, "STEP8C_whole_femur_morphometry_sensitivity_geometry_detail.csv")
out_stability = os.path.join(D, "STEP8C_whole_femur_morphometry_sensitivity_stability_summary.csv")
out_joint = os.path.join(D, "STEP8C_whole_femur_morphometry_sensitivity_joint_Wald.csv")
out_morph = os.path.join(D, "STEP8C_added_global_morphometry_coefficients.csv")
out_wide = os.path.join(D, "STEP8C_whole_femur_morphometry_sensitivity_wide.csv")

detail.to_csv(out_detail, index=False)
stability.to_csv(out_stability, index=False)
joint.to_csv(out_joint, index=False)
morph_results.to_csv(out_morph, index=False)
wide.to_csv(out_wide, index=False)

print("\n" + "=" * 125)
print("✅ STEP 8C COMPLETE")
print(f"Geometry detail   : {out_detail}")
print(f"Stability summary : {out_stability}")
print(f"Joint Wald tests  : {out_joint}")
print(f"Global metrics    : {out_morph}")
print(f"Wide summary      : {out_wide}")
print("=" * 125)
