from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, math, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio

# ==================================================================================================
# STEP 7C — Compartment-matched unadjusted association
# ==================================================================================================
analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP7B_crosstalk_ready_master_155.csv")

df = pd.read_csv(path)
assert len(df) == 155 and df["scan_id"].nunique() == 155

# Robust bool conversion
def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

bool_cols = [
    "medial_meniscal_lesion","lateral_meniscal_lesion",
    "medial_high_grade_femoral_cartilage","lateral_high_grade_femoral_cartilage"
]

for c in bool_cols:
    df[c] = to_bool(df[c])
    assert df[c].notna().all()

# ==================================================================================================
# Helper: 2×2 association
#
#                    Outcome+
# Exposure+       a            b
# Exposure-       c            d
# ==================================================================================================
def analyze_2x2(data, exposure, outcome, compartment):
    e, y = data[exposure].astype(bool), data[outcome].astype(bool)

    a = int(( e &  y).sum())
    b = int(( e & ~y).sum())
    c = int((~e &  y).sum())
    d = int((~e & ~y).sum())

    table = np.array([[a, b], [c, d]])

    # Prevalence and prevalence ratio
    prev_exp = a / (a + b)
    prev_unexp = c / (c + d)
    pr = prev_exp / prev_unexp

    # Katz log-CI for PR
    se_log_pr = math.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c+d)))
    pr_low = math.exp(math.log(pr) - 1.96*se_log_pr)
    pr_high = math.exp(math.log(pr) + 1.96*se_log_pr)

    # Fisher exact test
    fisher = fisher_exact(table, alternative="two-sided")

    # Conditional maximum-likelihood OR + exact CI
    cor = odds_ratio(table, kind="conditional")
    ci = cor.confidence_interval(confidence_level=0.95)

    return {
        "Compartment": compartment,
        "Exposure": exposure,
        "Outcome": outcome,
        "a_exp_pos": a, "b_exp_neg": b, "c_unexp_pos": c, "d_unexp_neg": d,
        "Exposure_N": a+b, "Unexposed_N": c+d,
        "Outcome_positive_N": a+c,
        "Prev_exposed": prev_exp,
        "Prev_unexposed": prev_unexp,
        "PR": pr, "PR95_low": pr_low, "PR95_high": pr_high,
        "Conditional_OR": cor.statistic,
        "OR95_low": ci.low, "OR95_high": ci.high,
        "Fisher_P": fisher.pvalue
    }

# ==================================================================================================
# Main analyses
# ==================================================================================================
results = []

results.append(analyze_2x2(
    df,
    "medial_meniscal_lesion",
    "medial_high_grade_femoral_cartilage",
    "Medial"
))

results.append(analyze_2x2(
    df,
    "lateral_meniscal_lesion",
    "lateral_high_grade_femoral_cartilage",
    "Lateral"
))

res = pd.DataFrame(results)

# ==================================================================================================
# Output
# ==================================================================================================
print("=" * 115); print("STEP 7C — Compartment-matched meniscus–cartilage association"); print("=" * 115)

for _, r in res.iterrows():
    print(f"\n[{r['Compartment']} compartment]")
    print("2×2 table:")
    print(pd.DataFrame(
        [[r["a_exp_pos"], r["b_exp_neg"]], [r["c_unexp_pos"], r["d_unexp_neg"]]],
        index=["Meniscal lesion +", "Meniscal lesion −"],
        columns=["High-grade cartilage +", "High-grade cartilage −"]
    ))

    print(f"\nHigh-grade prevalence with meniscal lesion    : {100*r['Prev_exposed']:.1f}%")
    print(f"High-grade prevalence without meniscal lesion : {100*r['Prev_unexposed']:.1f}%")
    print(f"Prevalence ratio                              : {r['PR']:.3f} (95% CI {r['PR95_low']:.3f}–{r['PR95_high']:.3f})")
    print(f"Conditional exact OR                          : {r['Conditional_OR']:.3f} (95% CI {r['OR95_low']:.3f}–{r['OR95_high']:.3f})")
    print(f"Fisher exact P                                : {r['Fisher_P']:.4f}")

# ==================================================================================================
# Save
# ==================================================================================================
out = os.path.join(analysis_dir, "STEP7C_compartment_matched_unadjusted_association.csv")
res.to_csv(out, index=False)

print("\n" + "=" * 115)
print("✅ STEP 7C COMPLETE")
print(f"Saved: {out}")
print("=" * 115)
