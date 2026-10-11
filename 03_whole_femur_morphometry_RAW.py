from project_paths import DATA_ROOT, ANALYSIS_DIR
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
# STEP 3A. Femoral morphometry × femoral-specific pathology
# ============================================================
folder = str(DATA_ROOT); analysis_dir = str(ANALYSIS_DIR)
master_path = os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv")
morph_path = os.path.join(analysis_dir, "SKM_TEA_femoral_cartilage_results_100K_20260925.csv")

df_master = pd.read_csv(master_path); df_morph = pd.read_csv(morph_path)
df_master["scan_id"] = df_master["scan_id"].astype(str).str.strip(); df_morph["scan_id"] = df_morph["scan_id"].astype(str).str.strip()

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
print("=" * 95); print("STEP 3A — ID and merge audit"); print("=" * 95)
print(f"Master      : rows={len(df_master)}, unique scan_id={df_master['scan_id'].nunique()}, duplicates={df_master['scan_id'].duplicated().sum()}")
print(f"Morphometry : rows={len(df_morph)}, unique scan_id={df_morph['scan_id'].nunique()}, duplicates={df_morph['scan_id'].duplicated().sum()}")

master_ids, morph_ids = set(df_master["scan_id"]), set(df_morph["scan_id"])
print(f"ID sets identical : {master_ids == morph_ids}")
print(f"Master only       : {len(master_ids - morph_ids)}")
print(f"Morphometry only  : {len(morph_ids - master_ids)}")

assert len(df_master) == 155 and len(df_morph) == 155
assert df_master["scan_id"].nunique() == df_morph["scan_id"].nunique() == 155
assert master_ids == morph_ids

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
morph_cols = ["scan_id","mean_thickness_mm","median_thickness_mm","cartilage_volume_mm3","surface_area_mm2",
              "P10_thickness_mm","P90_thickness_mm","thin_lt_1mm_pct","label2_voxels","hit_rate_pct","qc_retention_pct","status"]

missing = [c for c in morph_cols if c not in df_morph.columns]
if missing: raise ValueError(f"Morphometry 缺失字段: {missing}")

df_analysis = df_master.merge(df_morph[morph_cols], on="scan_id", how="left", validate="one_to_one", suffixes=("", "_morph"))

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
analysis_vars = ["mean_thickness_mm","median_thickness_mm","cartilage_volume_mm3","surface_area_mm2","P10_thickness_mm","P90_thickness_mm","thin_lt_1mm_pct"]
print("\n" + "=" * 95); print("STEP 3B — Analysis-variable completeness"); print("=" * 95)
print(df_analysis[analysis_vars].isna().sum().to_string())

print("\nFemoral lesion status:")
print(df_analysis["femoral_cartilage_lesion"].value_counts().sort_index())

print("\nHighest femoral cartilage grade:")
print(df_analysis["highest_femoral_cartilage_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0))

print("\nMorphometry QC status:")
print(df_analysis["status"].value_counts(dropna=False))

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
desc = df_analysis[analysis_vars].agg(["mean","std","median","min","max"]).T
desc.columns = ["Mean","SD","Median","Min","Max"]

print("\n" + "=" * 95); print("STEP 3C — Overall femoral morphometry"); print("=" * 95)
print(desc.round(3).to_string())

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
out = os.path.join(analysis_dir, "STEP3_femoral_morphometry_master_155.csv")
df_analysis.to_csv(out, index=False)

print("\n" + "=" * 95)
print(f"✅ STEP 3A COMPLETE — Saved: {out}")
print("=" * 95)


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 017
# ====================================================================================================
# Step 3B：Global morphometry vs focal femoral cartilage pathology

import os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

# ============================================================
# STEP 3B. Global femoral morphometry vs focal femoral pathology
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP3_femoral_morphometry_master_155.csv")
df = pd.read_csv(path)

features = ["mean_thickness_mm","median_thickness_mm","cartilage_volume_mm3","surface_area_mm2","P10_thickness_mm","P90_thickness_mm","thin_lt_1mm_pct"]
labels = {"mean_thickness_mm":"Mean thickness (mm)", "median_thickness_mm":"Median thickness (mm)", "cartilage_volume_mm3":"Cartilage volume (mm³)",
          "surface_area_mm2":"Surface area (mm²)", "P10_thickness_mm":"P10 thickness (mm)", "P90_thickness_mm":"P90 thickness (mm)", "thin_lt_1mm_pct":"Thickness <1 mm (%)"}

# Robust boolean conversion
if df["femoral_cartilage_lesion"].dtype != bool:
    df["femoral_cartilage_lesion"] = df["femoral_cartilage_lesion"].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False})
assert df["femoral_cartilage_lesion"].notna().all()

grade_order = {"0":0,"1":1,"2A":2,"2B":3,"3":4}
df["femoral_grade_order"] = df["highest_femoral_cartilage_grade"].astype(str).map(grade_order)
assert df["femoral_grade_order"].notna().all()

print("=" * 105); print("STEP 3B — Global femoral morphometry vs focal femoral cartilage pathology"); print("=" * 105)
print("Lesion-negative / positive:", (~df["femoral_cartilage_lesion"]).sum(), "/", df["femoral_cartilage_lesion"].sum())
print("Grades:", df["highest_femoral_cartilage_grade"].value_counts().reindex(["0","1","2A","2B","3"], fill_value=0).to_dict())

# ============================================================
# 1. Lesion-negative vs lesion-positive
# ============================================================
binary_results = []
for feat in features:
    g0 = df.loc[~df["femoral_cartilage_lesion"], feat].dropna().values
    g1 = df.loc[df["femoral_cartilage_lesion"], feat].dropna().values
    U, p = mannwhitneyu(g0, g1, alternative="two-sided")
    rbc = 1 - 2*U/(len(g0)*len(g1))  # positive => lesion-positive tends to have larger values
    binary_results.append({"Feature":feat, "Label":labels[feat], "No_lesion_N":len(g0), "Lesion_N":len(g1),
                           "No_lesion_mean_SD":f"{np.mean(g0):.3f} ± {np.std(g0,ddof=1):.3f}", "Lesion_mean_SD":f"{np.mean(g1):.3f} ± {np.std(g1,ddof=1):.3f}",
                           "No_lesion_median_IQR":f"{np.median(g0):.3f} [{np.percentile(g0,25):.3f}, {np.percentile(g0,75):.3f}]",
                           "Lesion_median_IQR":f"{np.median(g1):.3f} [{np.percentile(g1,25):.3f}, {np.percentile(g1,75):.3f}]",
                           "U":U, "P_raw":p, "Rank_biserial":rbc})

df_binary = pd.DataFrame(binary_results)
df_binary["P_FDR"] = multipletests(df_binary["P_raw"], method="fdr_bh")[1]

print("\n" + "=" * 105); print("A. Femoral lesion-negative vs lesion-positive"); print("=" * 105)
print(df_binary[["Label","No_lesion_mean_SD","Lesion_mean_SD","Rank_biserial","P_raw","P_FDR"]].round({"Rank_biserial":3,"P_raw":4,"P_FDR":4}).to_string(index=False))

# ============================================================
# 2. Across femoral cartilage grades 0 / 1 / 2A / 2B / 3
# ============================================================
grade_results = []; grade_levels = ["0","1","2A","2B","3"]
for feat in features:
    groups = [df.loc[df["highest_femoral_cartilage_grade"].astype(str).eq(g), feat].dropna().values for g in grade_levels]
    H, p = kruskal(*groups)
    n, k = sum(len(x) for x in groups), len(groups)
    eps2 = max(0, (H-k+1)/(n-k))
    row = {"Feature":feat, "Label":labels[feat], "H":H, "P_raw":p, "Epsilon_squared":eps2}
    for g, vals in zip(grade_levels, groups):
        row[f"Grade_{g}_N"] = len(vals); row[f"Grade_{g}_mean_SD"] = f"{np.mean(vals):.3f} ± {np.std(vals,ddof=1):.3f}"
        row[f"Grade_{g}_median_IQR"] = f"{np.median(vals):.3f} [{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]"
    grade_results.append(row)

df_grade = pd.DataFrame(grade_results)
df_grade["P_FDR"] = multipletests(df_grade["P_raw"], method="fdr_bh")[1]

display_cols = ["Label","H","Epsilon_squared","P_raw","P_FDR"] + [f"Grade_{g}_mean_SD" for g in grade_levels]
print("\n" + "=" * 105); print("B. Across femoral cartilage grades"); print("=" * 105)
print(df_grade[display_cols].round({"H":3,"Epsilon_squared":3,"P_raw":4,"P_FDR":4}).to_string(index=False))

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
binary_path = os.path.join(analysis_dir, "STEP3B_global_morphometry_lesion_binary.csv")
grade_path = os.path.join(analysis_dir, "STEP3B_global_morphometry_femoral_grade.csv")
df_binary.to_csv(binary_path, index=False); df_grade.to_csv(grade_path, index=False)

print("\n" + "=" * 105)
print("✅ STEP 3B COMPLETE")
print("Binary comparison :", binary_path)
print("Grade comparison  :", grade_path)
print("=" * 105)


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 020
# ====================================================================================================
import os, numpy as np, pandas as pd
import statsmodels.api as sm

# ============================================================
# STEP 3C. Age/sex-adjusted global femoral morphometry
# ============================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP3_femoral_morphometry_master_155.csv")
df = pd.read_csv(path)

features = ["mean_thickness_mm","median_thickness_mm","cartilage_volume_mm3","surface_area_mm2","P10_thickness_mm","P90_thickness_mm","thin_lt_1mm_pct"]
labels = {"mean_thickness_mm":"Mean thickness (mm)", "median_thickness_mm":"Median thickness (mm)", "cartilage_volume_mm3":"Cartilage volume (mm³)",
          "surface_area_mm2":"Surface area (mm²)", "P10_thickness_mm":"P10 thickness (mm)", "P90_thickness_mm":"P90 thickness (mm)", "thin_lt_1mm_pct":"Thickness <1 mm (%)"}

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
df["PatientAge_num"] = pd.to_numeric(df["PatientAge"], errors="coerce")

sex = df["PatientSex"].astype(str).str.strip().str.lower()
sex_map = {"m":1, "male":1, "f":0, "female":0}
df["Male"] = sex.map(sex_map)

if df["femoral_cartilage_lesion"].dtype != bool:
    df["Femoral_Lesion"] = df["femoral_cartilage_lesion"].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0})
else:
    df["Femoral_Lesion"] = df["femoral_cartilage_lesion"].astype(int)

assert df[["PatientAge_num","Male","Femoral_Lesion"]].notna().all().all()
assert int(df["Femoral_Lesion"].sum()) == 71

print("=" * 105); print("STEP 3C — Age/sex-adjusted global femoral morphometry"); print("=" * 105)
print(f"N = {len(df)} | Femoral lesion positive = {df['Femoral_Lesion'].sum()} | negative = {(df['Femoral_Lesion']==0).sum()}")
print("Sex counts:", df["Male"].value_counts().sort_index().to_dict())
print(f"Age: {df['PatientAge_num'].mean():.1f} ± {df['PatientAge_num'].std():.1f} years")

# ------------------------------------------------------------
# 2. HC3 robust OLS
# outcome ~ femoral lesion + age (per 10 y) + sex
# ------------------------------------------------------------
df["Age_per10y"] = df["PatientAge_num"] / 10
results = []

for feat in features:
    d = df[[feat,"Femoral_Lesion","Age_per10y","Male"]].dropna().copy()
    X = sm.add_constant(d[["Femoral_Lesion","Age_per10y","Male"]])
    model = sm.OLS(d[feat], X).fit(cov_type="HC3")

    for var in ["Femoral_Lesion","Age_per10y","Male"]:
        beta = model.params[var]; se = model.bse[var]; p = model.pvalues[var]
        ci_low, ci_high = model.conf_int().loc[var]
        results.append({"Outcome":feat, "Label":labels[feat], "Predictor":var, "Beta":beta, "SE_HC3":se,
                        "CI95_low":ci_low, "CI95_high":ci_high, "P":p, "N":len(d), "R_squared":model.rsquared})

df_adj = pd.DataFrame(results)

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
lesion_results = df_adj[df_adj["Predictor"] == "Femoral_Lesion"].copy()

print("\n" + "=" * 105); print("Adjusted association of focal femoral cartilage lesion with GLOBAL morphometry"); print("=" * 105)

for _, r in lesion_results.iterrows():
    print(f"{r['Label']:<28} β = {r['Beta']:>9.3f}  (95% CI {r['CI95_low']:>9.3f} to {r['CI95_high']:>9.3f}), P = {r['P']:.4f}")

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
print("\n" + "=" * 105); print("Full adjusted model coefficients"); print("=" * 105)
print(df_adj[["Label","Predictor","Beta","CI95_low","CI95_high","P","R_squared"]].round(4).to_string(index=False))

# ------------------------------------------------------------
# Comment translated from the original Chinese annotation.
# ------------------------------------------------------------
out = os.path.join(analysis_dir, "STEP3C_global_morphometry_adjusted_age_sex.csv")
df_adj.to_csv(out, index=False)

print("\n" + "=" * 105)
print("✅ STEP 3C COMPLETE")
print(f"Saved: {out}")
print("=" * 105)


# ====================================================================================================
# ORIGINAL NOTEBOOK CELL 022
# ====================================================================================================
# Comment translated from the original Chinese annotation.
import os, pandas as pd, numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

analysis_dir = str(ANALYSIS_DIR)
master_path = os.path.join(analysis_dir, "STEP3_femoral_morphometry_master_155.csv")
adj_path = os.path.join(analysis_dir, "STEP3C_global_morphometry_adjusted_age_sex.csv")

df = pd.read_csv(master_path)
adj = pd.read_csv(adj_path)

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
les = adj[adj["Predictor"] == "Femoral_Lesion"].copy()
les["P_FDR"] = multipletests(les["P"], method="fdr_bh")[1]

print("=" * 95); print("STEP 3D-A — Multiple-testing correction for lesion coefficients"); print("=" * 95)
print(les[["Label","Beta","CI95_low","CI95_high","P","P_FDR"]].round(4).to_string(index=False))

# ============================================================
# Comment translated from the original Chinese annotation.
# ============================================================
if df["femoral_cartilage_lesion"].dtype != bool:
    df["Femoral_Lesion"] = df["femoral_cartilage_lesion"].astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0})
else:
    df["Femoral_Lesion"] = df["femoral_cartilage_lesion"].astype(int)

df["PatientAge_num"] = pd.to_numeric(df["PatientAge"], errors="coerce")
sex = df["PatientSex"].astype(str).str.strip().str.lower()
df["Male"] = sex.map({"m":1,"male":1,"f":0,"female":0})

g0 = df.loc[df["Femoral_Lesion"] == 0, "PatientAge_num"]
g1 = df.loc[df["Femoral_Lesion"] == 1, "PatientAge_num"]
U, p_age = mannwhitneyu(g0, g1, alternative="two-sided")

tab = pd.crosstab(df["Femoral_Lesion"], df["Male"])
chi2, p_sex, _, _ = chi2_contingency(tab)

print("\n" + "=" * 95); print("STEP 3D-B — Covariate balance"); print("=" * 95)
print(f"Age — no lesion : {g0.mean():.1f} ± {g0.std():.1f} years")
print(f"Age — lesion    : {g1.mean():.1f} ± {g1.std():.1f} years")
print(f"Mann–Whitney P  : {p_age:.4g}")

print("\nSex table:")
print(tab)
print(f"Chi-square P     : {p_sex:.4g}")

out = os.path.join(analysis_dir, "STEP3D_global_morphometry_FDR.csv")
les.to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
