from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, warnings, numpy as np, pandas as pd
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE
from statsmodels.genmod.cov_struct import OrdinalIndependence

# ============================================================
# STEP 6C. External grade association of K=2 geometry clusters
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
cluster_path = os.path.join(analysis_dir, "STEP6B_K2_cluster_master_112.csv")
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")

df = pd.read_csv(cluster_path); master = pd.read_csv(master_path)

grade_order = {"1":1,"2A":2,"2B":3,"3":4}
df["grade"] = df["grade"].astype(str); df["grade_ord"] = df["grade"].map(grade_order)
df["cluster_K2"] = df["cluster_K2"].astype(int)

# covariates
cov = master[["scan_id","PatientAge","PatientSex"]].copy()
df = df.merge(cov, on="scan_id", how="left", validate="many_to_one")
df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10
df["Male"] = df["PatientSex"].astype(str).str.strip().str.lower().map({"m":1,"male":1,"f":0,"female":0})

assert len(df) == 112 and df["scan_id"].nunique() == 71
assert df[["grade_ord","cluster_K2","Age_per10y","Male"]].notna().all().all()

print("=" * 110); print("STEP 6C — External grade association of K=2 geometry clusters"); print("=" * 110)
print("Cluster sizes:", df["cluster_K2"].value_counts().sort_index().to_dict())

# ============================================================
# 1. Grade distribution
# ============================================================
grade_tab = pd.crosstab(df["cluster_K2"], df["grade"]).reindex(columns=["1","2A","2B","3"], fill_value=0)

print("\n" + "=" * 110); print("STEP 6C-A — Grade distribution by cluster"); print("=" * 110)
print(grade_tab)

print("\nRow percentages:")
print((grade_tab.div(grade_tab.sum(axis=1), axis=0) * 100).round(1))

# supplementary descriptive severity summaries
for c in [0,1]:
    d = df[df["cluster_K2"] == c]
    high = d["grade_ord"].ge(3).sum()
    g3 = d["grade_ord"].eq(4).sum()
    print(f"\nCluster {c}: Grade ≥2B = {high}/{len(d)} ({100*high/len(d):.1f}%), Grade 3 = {g3}/{len(d)} ({100*g3/len(d):.1f}%)")

# ============================================================
# 2. Unadjusted ordinal GEE
# Cluster 1 vs Cluster 0
# ============================================================
model1 = OrdinalGEE(endog=df["grade_ord"], exog=df[["cluster_K2"]], groups=df["scan_id"], cov_struct=OrdinalIndependence())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit1 = model1.fit(maxiter=100)

b1, p1 = fit1.params["cluster_K2"], fit1.pvalues["cluster_K2"]
lo1, hi1 = fit1.conf_int().loc["cluster_K2"]

# ============================================================
# 3. Age/sex-adjusted ordinal GEE
# ============================================================
X2 = df[["cluster_K2","Age_per10y","Male"]]

model2 = OrdinalGEE(endog=df["grade_ord"], exog=X2, groups=df["scan_id"], cov_struct=OrdinalIndependence())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit2 = model2.fit(maxiter=100)

b2, p2 = fit2.params["cluster_K2"], fit2.pvalues["cluster_K2"]
lo2, hi2 = fit2.conf_int().loc["cluster_K2"]

results = pd.DataFrame([
    {"Model":"Unadjusted", "Comparison":"Cluster 1 vs Cluster 0", "Beta":b1, "OR":np.exp(b1), "CI95_low":np.exp(lo1), "CI95_high":np.exp(hi1), "P":p1},
    {"Model":"Age/sex adjusted", "Comparison":"Cluster 1 vs Cluster 0", "Beta":b2, "OR":np.exp(b2), "CI95_low":np.exp(lo2), "CI95_high":np.exp(hi2), "P":p2}
])

print("\n" + "=" * 110); print("STEP 6C-B — Ordinal GEE"); print("=" * 110)
print(results.round({"Beta":3,"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

# ============================================================
# 4. Adjustment terms
# ============================================================
print("\nAdjusted covariates:")
print(f"Age per 10 y : beta={fit2.params['Age_per10y']:.3f}, OR={np.exp(fit2.params['Age_per10y']):.3f}, P={fit2.pvalues['Age_per10y']:.4f}")
print(f"Male         : beta={fit2.params['Male']:.3f}, OR={np.exp(fit2.params['Male']):.3f}, P={fit2.pvalues['Male']:.4f}")

# ============================================================
# 5. Save
# ============================================================
grade_path = os.path.join(analysis_dir, "STEP6C_K2_grade_distribution.csv")
result_path = os.path.join(analysis_dir, "STEP6C_K2_grade_ordinal_GEE.csv")

grade_tab.to_csv(grade_path)
results.to_csv(result_path, index=False)

print("\n" + "=" * 110)
print("✅ STEP 6C COMPLETE")
print("Grade distribution :", grade_path)
print("Ordinal GEE        :", result_path)
print("=" * 110)
