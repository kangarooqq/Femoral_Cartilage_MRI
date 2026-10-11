from project_paths import DATA_ROOT, ANALYSIS_DIR
import os, pandas as pd

analysis_dir = str(ANALYSIS_DIR)
master = pd.read_csv(os.path.join(analysis_dir, "STEP2_master_cohort_155_FINAL.csv"))
lesion = pd.read_csv(os.path.join(analysis_dir, "STEP4B_femoral_lesion_geometry_112.csv"))

print("=" * 95); print("STEP 4D — Subject-level independence audit"); print("=" * 95)

print(f"Total scans              : {len(master)}")
print(f"Unique scan_id           : {master['scan_id'].nunique()}")
print(f"Unique subject_id        : {master['subject_id'].nunique()}")
print(f"Duplicated subject scans : {master['subject_id'].duplicated().sum()}")

subject_counts = master.groupby("subject_id").size().sort_values(ascending=False)
print("\nScans per subject:")
print(subject_counts.value_counts().sort_index())

repeated = subject_counts[subject_counts > 1]
print(f"\nSubjects with >1 scan    : {len(repeated)}")

if len(repeated):
    repeated_ids = repeated.index
    print("\nRepeated subjects:")
    print(master.loc[master["subject_id"].isin(repeated_ids),
                     ["subject_id","scan_id","timepoint","split"]].sort_values(["subject_id","timepoint"]).to_string(index=False))

# lesion-positive subset
lesion_subject = lesion.merge(master[["scan_id","subject_id","timepoint"]], on="scan_id", how="left", validate="many_to_one")

print("\n" + "=" * 95); print("Lesion-level clustering structure"); print("=" * 95)
print(f"Lesion annotations          : {len(lesion_subject)}")
print(f"Lesion-positive scans       : {lesion_subject['scan_id'].nunique()}")
print(f"Lesion-positive subjects    : {lesion_subject['subject_id'].nunique()}")
print(f"Max lesions per scan        : {lesion_subject.groupby('scan_id').size().max()}")
print(f"Max lesions per subject     : {lesion_subject.groupby('subject_id').size().max()}")

out = os.path.join(analysis_dir, "STEP4D_subject_independence_audit.csv")
master[["subject_id","scan_id","timepoint","split"]].sort_values(["subject_id","timepoint"]).to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
