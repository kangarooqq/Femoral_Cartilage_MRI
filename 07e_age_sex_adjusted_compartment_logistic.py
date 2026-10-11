from project_paths import DATA_ROOT, ANALYSIS_DIR
# Comment translated from the original Chinese annotation.
import os, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# ==================================================================================================
# STEP 7D — Age/sex-adjusted compartment-specific logistic regression
# ==================================================================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP7B_crosstalk_ready_master_155.csv")

df = pd.read_csv(path)
assert len(df) == 155 and df["scan_id"].nunique() == 155

# ==================================================================================================
# 1. Clean variables
# ==================================================================================================
def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

bool_cols = ["medial_meniscal_lesion","lateral_meniscal_lesion",
             "medial_high_grade_femoral_cartilage","lateral_high_grade_femoral_cartilage"]

for c in bool_cols:
    df[c] = to_bool(df[c])
    assert df[c].notna().all()

df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10
sex = df["PatientSex"].astype(str).str.strip().str.lower()
df["Male"] = sex.map({"m":1,"male":1,"f":0,"female":0})

assert df[["Age_per10y","Male"]].notna().all().all()

print("=" * 115); print("STEP 7D — Age/sex-adjusted compartment-specific logistic regression"); print("=" * 115)
print(f"N = {len(df)}")
print(f"Age = {df['Age_per10y'].mean()*10:.1f} ± {df['Age_per10y'].std()*10:.1f} years")
print("Sex:", df["Male"].value_counts().sort_index().to_dict())

# ==================================================================================================
# 2. Model helper
#    HC3 robust SE; exposure + age/10y + male
# ==================================================================================================
def fit_adjusted_model(data, compartment, exposure, outcome):
    d = data[[exposure,outcome,"Age_per10y","Male"]].copy()
    d["Exposure"] = d[exposure].astype(int)
    d["Outcome"] = d[outcome].astype(int)

    X = sm.add_constant(d[["Exposure","Age_per10y","Male"]], has_constant="add")
    model = sm.GLM(d["Outcome"], X, family=sm.families.Binomial()).fit(cov_type="HC3")

    rows = []
    for var, label in [("Exposure","Meniscal lesion"), ("Age_per10y","Age per 10 years"), ("Male","Male vs female")]:
        beta, p = model.params[var], model.pvalues[var]
        lo, hi = model.conf_int().loc[var]
        rows.append({
            "Compartment":compartment, "Predictor":label, "Beta":beta, "OR":np.exp(beta),
            "CI95_low":np.exp(lo), "CI95_high":np.exp(hi), "P":p,
            "N":len(d), "Events":int(d["Outcome"].sum()),
            "Exposure_positive":int(d["Exposure"].sum()), "Converged":bool(model.converged)
        })

    return pd.DataFrame(rows)

# ==================================================================================================
# 3. Fit medial and lateral models
# ==================================================================================================
med = fit_adjusted_model(df, "Medial", "medial_meniscal_lesion", "medial_high_grade_femoral_cartilage")
lat = fit_adjusted_model(df, "Lateral", "lateral_meniscal_lesion", "lateral_high_grade_femoral_cartilage")

res = pd.concat([med, lat], ignore_index=True)

# Optional BH-FDR across the two prespecified meniscal-exposure coefficients
exposure_idx = res["Predictor"].eq("Meniscal lesion")
res["Exposure_P_FDR"] = np.nan
res.loc[exposure_idx, "Exposure_P_FDR"] = multipletests(res.loc[exposure_idx, "P"], method="fdr_bh")[1]

# ==================================================================================================
# 4. Main output
# ==================================================================================================
print("\n" + "=" * 115); print("STEP 7D-A — Adjusted meniscus–cartilage associations"); print("=" * 115)

for compartment in ["Medial","Lateral"]:
    r = res[(res["Compartment"] == compartment) & (res["Predictor"] == "Meniscal lesion")].iloc[0]
    print(f"\n[{compartment}]")
    print(f"Events                    : {int(r['Events'])}")
    print(f"Meniscal-lesion positive  : {int(r['Exposure_positive'])}")
    print(f"Adjusted OR               : {r['OR']:.3f} (95% CI {r['CI95_low']:.3f}–{r['CI95_high']:.3f})")
    print(f"P                          : {r['P']:.4f}")
    print(f"BH-FDR q                   : {r['Exposure_P_FDR']:.4f}")
    print(f"Model converged            : {r['Converged']}")

print("\n" + "=" * 115); print("STEP 7D-B — Full model coefficients"); print("=" * 115)
print(res[["Compartment","Predictor","OR","CI95_low","CI95_high","P"]]
      .round({"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

# ==================================================================================================
# 5. Save
# ==================================================================================================
out = os.path.join(analysis_dir, "STEP7D_compartment_association_age_sex_adjusted.csv")
res.to_csv(out, index=False)

print("\n" + "=" * 115)
print("✅ STEP 7D COMPLETE")
print(f"Saved: {out}")
print("=" * 115)
