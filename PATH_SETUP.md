# Set the location of private SKM-TEA data

These scripts no longer contain the original developer's absolute filesystem paths.
They read shared locations from `project_paths.py`.

## Data layout

Point `SKMTEA_DATA_ROOT` at the folder containing your authorized data, for example:

```text
<DATA_ROOT>/
  all_metadata.csv
  SKM_TEA_v1.0_patient_lesion_summary.xlsx    # Step 01 quality-control input
  json/
    train.json
    val.json
    test.json
  analysis_20260927/
    SKM_TEA_femoral_cartilage_results_100K_20260925.csv
    ... intermediate output CSVs from earlier steps ...
```

On Ubuntu/WSL, before running the scripts:

```bash
export SKMTEA_DATA_ROOT="/path/to/your/authorized/FEBio_test"
python 01_data_source_audit_RAW.py
```

If intermediate outputs belong in a different directory:

```bash
export SKMTEA_ANALYSIS_DIR="/path/to/private/analysis_output"
```

If no environment variables are set, scripts look for inputs in a local
`data/` directory beside the scripts and write into `data/analysis_20260927/`.

Only paths were changed in this package. Analytical correctness and full
pipeline execution with original data have **not** been re-validated.

**Do not upload Stanford SKM-TEA source data or participant-level outputs.**
`.gitignore` helps with Git command-line commits but does not protect files
explicitly uploaded through GitHub's browser interface.
