from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from scipy.stats import norm

# ==================================================================================================
# STEP 7E — Direct test of compartment-specific heterogeneity
# ==================================================================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP7B_crosstalk_ready_master_155.csv")

df = pd.read_csv(path)
assert len(df) == 155 and df["scan_id"].nunique() == 155

def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

for c in ["medial_meniscal_lesion","lateral_meniscal_lesion",
          "medial_high_grade_femoral_cartilage","lateral_high_grade_femoral_cartilage"]:
    df[c] = to_bool(df[c]); assert df[c].notna().all()

df["Age_per10y"] = pd.to_numeric(df["PatientAge"], errors="coerce") / 10
df["Male"] = df["PatientSex"].astype(str).str.strip().str.lower().map({"m":1,"male":1,"f":0,"female":0})
assert df[["Age_per10y","Male"]].notna().all().all()

# ==================================================================================================
# 1. Convert 155 subjects → 310 compartment observations
# ==================================================================================================
med = df[["scan_id","Age_per10y","Male"]].copy()
med["Compartment"] = "Medial"; med["Medial"] = 1
med["Meniscal"] = df["medial_meniscal_lesion"].astype(int)
med["HighGrade"] = df["medial_high_grade_femoral_cartilage"].astype(int)

lat = df[["scan_id","Age_per10y","Male"]].copy()
lat["Compartment"] = "Lateral"; lat["Medial"] = 0
lat["Meniscal"] = df["lateral_meniscal_lesion"].astype(int)
lat["HighGrade"] = df["lateral_high_grade_femoral_cartilage"].astype(int)

long = pd.concat([med, lat], ignore_index=True)
long["Meniscal_x_Medial"] = long["Meniscal"] * long["Medial"]

assert len(long) == 310
assert long["scan_id"].nunique() == 155
assert long.groupby("scan_id").size().eq(2).all()

print("=" * 115); print("STEP 7E — Direct medial-vs-lateral interaction test"); print("=" * 115)
print(f"Subjects                : {long['scan_id'].nunique()}")
print(f"Compartment observations: {len(long)}")
print(f"High-grade events       : {int(long['HighGrade'].sum())}")
print(pd.crosstab(long["Compartment"], long["HighGrade"]))

# ==================================================================================================
# 2. Age/sex-adjusted binomial GEE
#    Lateral = reference compartment
# ==================================================================================================
X = sm.add_constant(long[["Meniscal","Medial","Meniscal_x_Medial","Age_per10y","Male"]], has_constant="add")

model = GEE(
    endog=long["HighGrade"],
    exog=X,
    groups=long["scan_id"],
    family=Binomial(),
    cov_struct=Exchangeable()
)

fit = model.fit(maxiter=100)

print("\n" + "=" * 115); print("STEP 7E-A — Full interaction model"); print("=" * 115)

coef_rows = []
for var in ["Meniscal","Medial","Meniscal_x_Medial","Age_per10y","Male"]:
    beta, se, p = fit.params[var], fit.bse[var], fit.pvalues[var]
    lo, hi = fit.conf_int().loc[var]
    coef_rows.append({
        "Predictor":var, "Beta":beta, "OR":np.exp(beta),
        "CI95_low":np.exp(lo), "CI95_high":np.exp(hi), "P":p
    })

coef = pd.DataFrame(coef_rows)
print(coef.round({"Beta":3,"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

# ==================================================================================================
# 3. Recover compartment-specific meniscal ORs from the interaction model
#
# Lateral log OR = beta_Meniscal
# Medial  log OR = beta_Meniscal + beta_interaction
# Interaction OR = medial OR / lateral OR
# ==================================================================================================
b = fit.params
V = fit.cov_params()

# Lateral
beta_lat = b["Meniscal"]
var_lat = V.loc["Meniscal","Meniscal"]
se_lat = np.sqrt(var_lat)
p_lat = 2 * (1 - norm.cdf(abs(beta_lat / se_lat)))

# Medial
beta_med = b["Meniscal"] + b["Meniscal_x_Medial"]
var_med = (
    V.loc["Meniscal","Meniscal"] +
    V.loc["Meniscal_x_Medial","Meniscal_x_Medial"] +
    2 * V.loc["Meniscal","Meniscal_x_Medial"]
)
se_med = np.sqrt(var_med)
p_med = 2 * (1 - norm.cdf(abs(beta_med / se_med)))

# Interaction
beta_int = b["Meniscal_x_Medial"]
se_int = fit.bse["Meniscal_x_Medial"]
p_int = fit.pvalues["Meniscal_x_Medial"]

summary = pd.DataFrame([
    {
        "Effect":"Lateral meniscal association",
        "OR":np.exp(beta_lat),
        "CI95_low":np.exp(beta_lat - 1.96*se_lat),
        "CI95_high":np.exp(beta_lat + 1.96*se_lat),
        "P":p_lat
    },
    {
        "Effect":"Medial meniscal association",
        "OR":np.exp(beta_med),
        "CI95_low":np.exp(beta_med - 1.96*se_med),
        "CI95_high":np.exp(beta_med + 1.96*se_med),
        "P":p_med
    },
    {
        "Effect":"Medial/Lateral OR ratio (interaction)",
        "OR":np.exp(beta_int),
        "CI95_low":np.exp(beta_int - 1.96*se_int),
        "CI95_high":np.exp(beta_int + 1.96*se_int),
        "P":p_int
    }
])

print("\n" + "=" * 115); print("STEP 7E-B — Compartment-specific effects from one unified model"); print("=" * 115)
print(summary.round({"OR":3,"CI95_low":3,"CI95_high":3,"P":4}).to_string(index=False))

print("\nInterpretation:")
print("Interaction OR >1 means the meniscus–cartilage association is stronger medially than laterally.")
print("The interaction P value is the formal test of whether the two compartment-specific associations differ.")

# ==================================================================================================
# 4. Working-correlation estimate
# ==================================================================================================
print("\nEstimated within-subject working correlation:")
print(fit.cov_struct.dep_params)

# ==================================================================================================
# 5. Save
# ==================================================================================================
coef_path = os.path.join(analysis_dir, "STEP7E_compartment_interaction_GEE_full_coefficients.csv")
summary_path = os.path.join(analysis_dir, "STEP7E_compartment_interaction_GEE_summary.csv")
long_path = os.path.join(analysis_dir, "STEP7E_compartment_long_310.csv")

coef.to_csv(coef_path, index=False)
summary.to_csv(summary_path, index=False)
long.to_csv(long_path, index=False)

print("\n" + "=" * 115)
print("✅ STEP 7E COMPLETE")
print(f"Full coefficients : {coef_path}")
print(f"Effect summary    : {summary_path}")
print(f"Long dataset      : {long_path}")
print("=" * 115)
