from project_paths import DATA_ROOT, ANALYSIS_DIR
# Sensitivity analysis for cumulative grade thresholds (Step 5D). This analysis does not search for new significance; it verifies that the ordinal GEE direction is not driven by a single grade level.
import os, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial

# ============================================================
# STEP 5D. Cumulative-grade threshold sensitivity
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
geometry_path = os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")

df = pd.read_csv(geometry_path); master = pd.read_csv(master_path)

features = ["ln_bbox_volume_mm3","center_SI_norm","center_AP_norm"]
labels = {"ln_bbox_volume_mm3":"Log bbox volume", "center_SI_norm":"SI location", "center_AP_norm":"AP location"}
grade_order = {"1":1,"2A":2,"2B":3,"3":4}

# ============================================================
# 1. Covariates
# ============================================================
df = df.merge(master[["scan_id","PatientAge","PatientSex"]], on="scan_id", how="left", validate="many_to_one")
df["grade"] = df["grade"].astype(str); df["grade_ord"] = df["grade"].map(grade_order)
df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10
df["Male"] = df["PatientSex"].astype(str).str.strip().str.lower().map({"m":1,"male":1,"f":0,"female":0})

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df[features + ["grade_ord","Age_per10y","Male"]].notna().all().all()

# Standardize once so OR per SD can be directly compared across the three thresholds
for feat in features:
    df[f"{feat}_z"] = (df[feat] - df[feat].mean()) / df[feat].std(ddof=1)

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
thresholds = {
    "Grade_ge_2A": df["grade_ord"] >= 2,
    "Grade_ge_2B": df["grade_ord"] >= 3,
    "Grade_3":     df["grade_ord"] >= 4
}

print("=" * 115); print("STEP 5D — Cumulative-grade threshold sensitivity"); print("=" * 115)

for name, y in thresholds.items():
    print(f"{name:<15}: positive={int(y.sum()):3d}, negative={int((~y).sum()):3d}")

# ============================================================
# 3. Age/sex-adjusted binary GEE for each feature × threshold
# ============================================================
rows = []

for threshold_name, y in thresholds.items():
    df["binary_outcome"] = y.astype(int)

    for feat in features:
        z = f"{feat}_z"
        d = df[["scan_id","binary_outcome",z,"Age_per10y","Male"]].copy()
        X = sm.add_constant(d[[z,"Age_per10y","Male"]], has_constant="add")

        model = GEE(endog=d["binary_outcome"], exog=X, groups=d["scan_id"], family=Binomial(), cov_struct=Independence())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(maxiter=100)

        beta, p = fit.params[z], fit.pvalues[z]
        lo, hi = fit.conf_int().loc[z]

        rows.append({
            "Threshold":threshold_name, "Feature":feat, "Label":labels[feat],
            "N":len(d), "N_clusters":d["scan_id"].nunique(),
            "Events":int(d["binary_outcome"].sum()),
            "Beta_per_SD":beta, "OR_per_SD":np.exp(beta),
            "OR95_low":np.exp(lo), "OR95_high":np.exp(hi), "P":p
        })

res = pd.DataFrame(rows)

print("\n" + "=" * 115); print("Adjusted cumulative-threshold GEE results"); print("=" * 115)

for threshold in thresholds:
    x = res[res["Threshold"] == threshold]
    print(f"\n[{threshold}]")
    print(x[["Label","Events","OR_per_SD","OR95_low","OR95_high","P"]]
          .round({"OR_per_SD":3,"OR95_low":3,"OR95_high":3,"P":4}).to_string(index=False))

# ============================================================
# 4. Direction consistency table
# ============================================================
print("\n" + "=" * 115); print("Direction consistency"); print("=" * 115)

direction = res.pivot(index="Label", columns="Threshold", values="OR_per_SD")
print(direction.round(3).to_string())

for label in direction.index:
    vals = direction.loc[label].values
    same_side = np.all(vals > 1) or np.all(vals < 1)
    print(f"{label:<20}: {'CONSISTENT' if same_side else 'MIXED'}")

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
out = os.path.join(analysis_dir, "STEP5D_cumulative_grade_threshold_sensitivity.csv")
res.to_csv(out, index=False)

print("\n" + "=" * 115)
print("✅ STEP 5D COMPLETE")
print(f"Saved: {out}")
print("=" * 115)
