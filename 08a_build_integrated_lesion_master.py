from project_paths import DATA_ROOT, ANALYSIS_DIR
# Comment translated from the original Chinese annotation.
# Generate a single final dataset:
# Integrated lesion-level master containing 112 lesion annotations from 71 scans
# Contains focal geometry, grade, compartment, age/sex, seven whole-femur morphometry measures, and compartment-matched meniscus status for Step 8D.
import os, re, numpy as np, pandas as pd

# ==================================================================================================
# STEP 8A — Integrated lesion-level master + full consistency/QC audit
# ==================================================================================================
D = str(ANALYSIS_DIR)

geometry_path = os.path.join(D, "STEP4B_femoral_lesion_geometry_112.csv")
cart_path     = os.path.join(D, "STEP7B_femoral_cartilage_annotations_with_compartment_FINAL.csv")
master_path   = os.path.join(D, "STEP2_master_cohort_155_FINAL.csv")
morph_path    = os.path.join(D, "STEP3_femoral_morphometry_master_155.csv")
meniscus_path = os.path.join(D, "STEP7A_meniscal_scan_level_compartments_155_FINAL.csv")

geo = pd.read_csv(geometry_path)
cart = pd.read_csv(cart_path)
master = pd.read_csv(master_path)
morph = pd.read_csv(morph_path)
men = pd.read_csv(meniscus_path)

for d in [geo, cart, master, morph, men]:
    d["scan_id"] = d["scan_id"].astype(str).str.strip()

print("=" * 120)
print("STEP 8A — SOURCE AUDIT")
print("=" * 120)
print(f"Geometry           : {len(geo)} rows / {geo['scan_id'].nunique()} scans")
print(f"Cartilage Step7B   : {len(cart)} rows / {cart['scan_id'].nunique()} scans")
print(f"Master             : {len(master)} rows / {master['scan_id'].nunique()} scans")
print(f"Morphometry        : {len(morph)} rows / {morph['scan_id'].nunique()} scans")
print(f"Meniscus Step7A    : {len(men)} rows / {men['scan_id'].nunique()} scans")

assert len(geo) == 112 and geo["scan_id"].nunique() == 71
assert len(cart) == 112 and cart["scan_id"].nunique() == 71
assert len(master) == 155 and master["scan_id"].nunique() == 155
assert len(morph) == 155 and morph["scan_id"].nunique() == 155
assert len(men) == 155 and men["scan_id"].nunique() == 155

# ==================================================================================================
# 1. Annotation key audit: annotation_id is split-local, so use (split, annotation_id)
# ==================================================================================================
for name, d in [("geo", geo), ("cart", cart)]:
    assert {"split","annotation_id","scan_id"}.issubset(d.columns), f"{name}: missing annotation key columns"
    assert not d.duplicated(["split","annotation_id"]).any(), f"{name}: duplicated (split, annotation_id)"

geo_keys = set(map(tuple, geo[["split","annotation_id"]].to_numpy()))
cart_keys = set(map(tuple, cart[["split","annotation_id"]].to_numpy()))

print("\nAnnotation key audit:")
print(f"Geometry keys only : {len(geo_keys - cart_keys)}")
print(f"Step7B keys only   : {len(cart_keys - geo_keys)}")
print(f"Shared keys        : {len(geo_keys & cart_keys)}")

assert geo_keys == cart_keys, "Step4B and Step7B annotation sets differ"

# ==================================================================================================
# 2. Cross-check Step4B geometry against independently rebuilt Step7B annotation mapping
# ==================================================================================================
cart_check = cart[["split","annotation_id","scan_id","grade","grade_order","high_grade","compartment","medial_score"]].copy()
cart_check = cart_check.rename(columns={
    "scan_id":"scan_id_7b", "grade":"grade_7b", "grade_order":"grade_order_7b",
    "high_grade":"high_grade_7b", "compartment":"compartment_7b", "medial_score":"medial_score_7b"
})

audit = geo.merge(cart_check, on=["split","annotation_id"], how="outer", validate="one_to_one", indicator=True)

assert (audit["_merge"] == "both").all()
assert (audit["scan_id"].astype(str) == audit["scan_id_7b"].astype(str)).all()

# Resolve Step4B grade naming robustly
grade_col = next((c for c in ["grade","lesion_grade","femoral_grade"] if c in geo.columns), None)
assert grade_col is not None, f"Cannot locate grade column in Step4B. Columns:\n{geo.columns.tolist()}"

audit["grade_geo"] = audit[grade_col].astype(str)
audit["grade_7b"] = audit["grade_7b"].astype(str)

grade_match = audit["grade_geo"].eq(audit["grade_7b"])
comp_match = audit["compartment"].astype(str).eq(audit["compartment_7b"].astype(str)) if "compartment" in audit.columns else pd.Series(True, index=audit.index)

print("\nStep4B ↔ Step7B consistency:")
print(f"Scan ID agreement    : {(audit['scan_id'].astype(str) == audit['scan_id_7b'].astype(str)).sum()} / 112")
print(f"Grade agreement      : {grade_match.sum()} / 112")
if "compartment" in audit.columns: print(f"Compartment agreement: {comp_match.sum()} / 112")

assert grade_match.all()
assert comp_match.all()

# ==================================================================================================
# 3. Use Step7B as authoritative grade/compartment columns, Step4B as authoritative geometry
# ==================================================================================================
drop_if_present = ["grade","grade_order","high_grade","compartment","medial_score"]
base = geo.drop(columns=[c for c in drop_if_present if c in geo.columns]).merge(
    cart[["split","annotation_id","scan_id","grade","grade_order","high_grade","compartment","medial_score"]],
    on=["split","annotation_id","scan_id"], how="left", validate="one_to_one"
)

assert len(base) == 112

# ==================================================================================================
# 4. Demographics
# ==================================================================================================
demo_cols = ["scan_id","subject_id","PatientAge","PatientSex","MedialDirection"]
assert set(demo_cols).issubset(master.columns)

demo = master[demo_cols].copy()
demo["PatientAge"] = pd.to_numeric(demo["PatientAge"], errors="coerce")
demo["PatientSex"] = demo["PatientSex"].astype(str).str.strip()
demo["MedialDirection"] = demo["MedialDirection"].astype(str).str.strip().str.upper()

sex_lower = demo["PatientSex"].str.lower()
demo["male"] = sex_lower.map({"m":1,"male":1,"f":0,"female":0})
demo["age_per10y"] = demo["PatientAge"] / 10

assert demo["PatientAge"].notna().all()
assert demo["male"].notna().all()
assert demo["MedialDirection"].isin(["L","R"]).all()
assert demo["subject_id"].nunique() == 155

# ==================================================================================================
# 5. Resolve the locked 7 whole-femur morphometry variables
#    Exact matching is attempted after normalization; script stops rather than guessing if unresolved.
# ==================================================================================================
def norm(s): return re.sub(r"[^a-z0-9]+", "", str(s).lower())

aliases = {
    "global_mean_thickness_mm": [
        "mean_thickness_mm","mean_cartilage_thickness_mm","cartilage_mean_thickness_mm","mean_thickness"
    ],
    "global_median_thickness_mm": [
        "median_thickness_mm","median_cartilage_thickness_mm","cartilage_median_thickness_mm","median_thickness"
    ],
    "global_cartilage_volume_mm3": [
        "cartilage_volume_mm3","femoral_cartilage_volume_mm3","volume_mm3","cartilage_volume"
    ],
    "global_surface_area_mm2": [
        "surface_area_mm2","cartilage_surface_area_mm2","femoral_cartilage_surface_area_mm2","surface_area"
    ],
    "global_p10_thickness_mm": [
        "p10_thickness_mm","thickness_p10_mm","thickness_10th_percentile_mm","p10"
    ],
    "global_p90_thickness_mm": [
        "p90_thickness_mm","thickness_p90_mm","thickness_90th_percentile_mm","p90"
    ],
    "global_thin_lt1mm_pct": [
        "thin_lt1mm_pct","thin_fraction_lt1mm_pct","thin_percent_lt1mm","pct_thickness_lt1mm",
        "percent_thickness_lt1mm","thin_below_1mm_pct"
    ]
}

norm_lookup = {}
for c in morph.columns: norm_lookup.setdefault(norm(c), []).append(c)

resolved = {}
for new_name, candidates in aliases.items():
    hits = []
    for candidate in candidates:
        hits.extend(norm_lookup.get(norm(candidate), []))
    hits = list(dict.fromkeys(hits))
    if len(hits) != 1:
        print(f"\nCould not uniquely resolve: {new_name}")
        print("Candidate aliases:", candidates)
        print("Available morphometry columns:")
        print(morph.columns.tolist())
        raise ValueError(f"Resolve morphometry column manually for {new_name}; hits={hits}")
    resolved[new_name] = hits[0]

print("\nLocked whole-femur morphometry mapping:")
for new, old in resolved.items(): print(f"{new:<34} <- {old}")

morph7 = morph[["scan_id"] + list(resolved.values())].copy().rename(columns={v:k for k,v in resolved.items()})
for c in resolved:
    morph7[c] = pd.to_numeric(morph7[c], errors="coerce")

assert morph7["scan_id"].nunique() == 155
assert morph7[list(resolved)].notna().all().all()

# ==================================================================================================
# 6. Meniscus compartment status for later sensitivity analysis
# ==================================================================================================
men_cols = ["scan_id","meniscal_lesion","medial_meniscal_lesion","lateral_meniscal_lesion"]
assert set(men_cols).issubset(men.columns)

men2 = men[men_cols].copy()

def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})

for c in men_cols[1:]:
    men2[c] = to_bool(men2[c])
    assert men2[c].notna().all()

# ==================================================================================================
# 7. Build integrated lesion-level master
# ==================================================================================================
# Remove demographic columns already present in lesion-level source,
# because STEP2 master is the authoritative source for demographics.
base = base.drop(columns=[c for c in ["subject_id","PatientAge","PatientSex","MedialDirection"] if c in base.columns])

integrated = base.merge(demo, on="scan_id", how="left", validate="many_to_one")
integrated = integrated.merge(morph7, on="scan_id", how="left", validate="many_to_one")
integrated = integrated.merge(men2, on="scan_id", how="left", validate="many_to_one")

assert len(integrated) == 112
assert integrated["scan_id"].nunique() == 71
assert integrated["subject_id"].nunique() == 71

integrated["ipsilateral_meniscal_lesion"] = np.where(
    integrated["compartment"].eq("Medial"),
    integrated["medial_meniscal_lesion"],
    integrated["lateral_meniscal_lesion"]
).astype(bool)

integrated["cross_compartment_meniscal_lesion"] = np.where(
    integrated["compartment"].eq("Medial"),
    integrated["lateral_meniscal_lesion"],
    integrated["medial_meniscal_lesion"]
).astype(bool)

# Explicit analysis coding
grade_order = {"1":1, "2A":2, "2B":3, "3":4}
integrated["grade"] = integrated["grade"].astype(str)
integrated["grade_order_locked"] = integrated["grade"].map(grade_order)
integrated["high_grade_locked"] = integrated["grade"].isin(["2B","3"])

assert integrated["grade_order_locked"].notna().all()

# ==================================================================================================
# 8. Core variable QC
# ==================================================================================================
core = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
missing_core = [c for c in core if c not in integrated.columns]
assert not missing_core, f"Missing geometry columns: {missing_core}"

for c in core:
    integrated[c] = pd.to_numeric(integrated[c], errors="coerce")

assert integrated[core].notna().all().all()
assert np.isfinite(integrated[core].to_numpy()).all()
assert integrated["bbox_anisotropy"].between(0,1).all()
assert (integrated["AP_dominance_ratio"] > 0).all()
assert integrated["center_SI_norm"].between(0,1).all()
assert integrated["center_AP_norm"].between(0,1).all()
assert integrated["medial_score"].between(0,1).all()

# ==================================================================================================
# 9. Reproduce locked cohort distributions
# ==================================================================================================
grade_counts = integrated["grade"].value_counts().reindex(["1","2A","2B","3"], fill_value=0)
comp_counts = integrated["compartment"].value_counts().reindex(["Medial","Lateral"], fill_value=0)
lesions_per_scan = integrated.groupby("scan_id").size().value_counts().sort_index()

print("\n" + "=" * 120)
print("STEP 8A — INTEGRATED MASTER QC")
print("=" * 120)

print(f"Rows / lesions       : {len(integrated)}")
print(f"Unique scans         : {integrated['scan_id'].nunique()}")
print(f"Unique subjects      : {integrated['subject_id'].nunique()}")

print("\nGrade counts:")
print(grade_counts)

print("\nCompartment counts:")
print(comp_counts)

print("\nLesions per scan:")
print(lesions_per_scan)

print("\nCore geometry:")
print(integrated[core].describe().loc[["mean","std","50%","min","max"]].round(4).T)

print("\nGlobal morphometry missing values:")
print(integrated[list(resolved)].isna().sum())

print("\nDemographic missing values:")
print(integrated[["PatientAge","male","MedialDirection"]].isna().sum())

print("\nMeniscus variables:")
print(f"Any meniscal lesion               : {integrated['meniscal_lesion'].sum()} lesion-rows")
print(f"Ipsilateral meniscal lesion       : {integrated['ipsilateral_meniscal_lesion'].sum()} lesion-rows")
print(f"Cross-compartment meniscal lesion : {integrated['cross_compartment_meniscal_lesion'].sum()} lesion-rows")

# Locked expected counts from previous validated steps
assert grade_counts.to_dict() == {"1":17,"2A":53,"2B":33,"3":9}
assert comp_counts.to_dict() == {"Medial":61,"Lateral":51}
assert lesions_per_scan.to_dict() == {1:40,2:23,3:6,4:2}
assert int(integrated["high_grade_locked"].sum()) == 42
assert int(((integrated["compartment"]=="Medial") & integrated["high_grade_locked"]).sum()) == 23
assert int(((integrated["compartment"]=="Lateral") & integrated["high_grade_locked"]).sum()) == 19

if "signed_delta_corrected" in integrated.columns:
    print(f"\nSigned-delta corrected annotations: {int(integrated['signed_delta_corrected'].sum())}")
    assert int(integrated["signed_delta_corrected"].sum()) == 3

# ==================================================================================================
# 10. Missingness / duplication / within-scan morphometry consistency
# ==================================================================================================
required = ["scan_id","subject_id","split","annotation_id","grade","grade_order_locked","compartment",
            "PatientAge","age_per10y","male","MedialDirection"] + core + list(resolved)

print("\nRequired-variable missingness:")
miss = integrated[required].isna().sum()
print(miss[miss > 0] if (miss > 0).any() else "None")

assert integrated[required].notna().all().all()
assert not integrated.duplicated(["split","annotation_id"]).any()

# Whole-femur metrics must be constant within scan
within_scan_fail = {}
for c in resolved:
    nuniq = integrated.groupby("scan_id")[c].nunique(dropna=False)
    within_scan_fail[c] = int((nuniq != 1).sum())

print("\nWithin-scan global morphometry inconsistency:")
print(within_scan_fail)
assert all(v == 0 for v in within_scan_fail.values())

# ==================================================================================================
# 11. Recompute core correlation audit — no new inference, only integrity check
# ==================================================================================================
candidate = ["ln_bbox_volume_mm3","bbox_anisotropy","AP_dominance_ratio","center_SI_norm","center_AP_norm","medial_score"]
rho = integrated[candidate].corr(method="spearman")
upper = rho.where(np.triu(np.ones(rho.shape), k=1).astype(bool))
max_abs_rho = upper.abs().max().max()

print(f"\nMaximum |Spearman rho| among locked six features: {max_abs_rho:.3f}")

# ==================================================================================================
# 12. Save final integrated dataset + QC tables
# ==================================================================================================
out_master = os.path.join(D, "STEP8A_integrated_lesion_level_master_112_FINAL.csv")
out_qc = os.path.join(D, "STEP8A_integration_QC_summary.csv")
out_map = os.path.join(D, "STEP8A_morphometry_column_mapping.csv")
out_corr = os.path.join(D, "STEP8A_core_geometry_spearman_QC.csv")

integrated = integrated.sort_values(["scan_id","split","annotation_id"]).reset_index(drop=True)
integrated.to_csv(out_master, index=False)
rho.to_csv(out_corr)

pd.DataFrame({"analysis_variable":list(resolved.keys()), "source_column":list(resolved.values())}).to_csv(out_map, index=False)

qc = pd.DataFrame([
    ["lesion_rows", len(integrated)],
    ["unique_scans", integrated["scan_id"].nunique()],
    ["unique_subjects", integrated["subject_id"].nunique()],
    ["grade_1", grade_counts["1"]],
    ["grade_2A", grade_counts["2A"]],
    ["grade_2B", grade_counts["2B"]],
    ["grade_3", grade_counts["3"]],
    ["high_grade_2B_or_3", integrated["high_grade_locked"].sum()],
    ["medial_annotations", comp_counts["Medial"]],
    ["lateral_annotations", comp_counts["Lateral"]],
    ["max_abs_core_spearman_rho", max_abs_rho],
    ["required_missing_values", integrated[required].isna().sum().sum()]
], columns=["QC_metric","value"])

qc.to_csv(out_qc, index=False)

print("\n" + "=" * 120)
print("✅ STEP 8A COMPLETE")
print(f"Integrated master : {out_master}")
print(f"QC summary        : {out_qc}")
print(f"Morphometry map   : {out_map}")
print(f"Correlation QC    : {out_corr}")
print("=" * 120)
