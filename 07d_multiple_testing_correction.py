from project_paths import DATA_ROOT, ANALYSIS_DIR
# Extend the Step 7C result file and calculate both BH-FDR and Bonferroni corrections.
import os, pandas as pd
from statsmodels.stats.multitest import multipletests

analysis_dir = str(ANALYSIS_DIR)
path = os.path.join(analysis_dir, "STEP7C_compartment_matched_unadjusted_association.csv")

res = pd.read_csv(path)

# Two prespecified compartment tests
pvals = res["Fisher_P"].values

res["P_BH_FDR"] = multipletests(pvals, method="fdr_bh")[1]
res["P_Bonferroni"] = multipletests(pvals, method="bonferroni")[1]

print("=" * 95)
print("STEP 7C-FDR — Multiple-testing correction")
print("=" * 95)

print(
    res[["Compartment","Fisher_P","P_BH_FDR","P_Bonferroni"]]
    .round(4)
    .to_string(index=False)
)

out = os.path.join(
    analysis_dir,
    "STEP7C_compartment_matched_association_multiple_testing.csv"
)

res.to_csv(out, index=False)

print("\n✅ Saved:", out)
