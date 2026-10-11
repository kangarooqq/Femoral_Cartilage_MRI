# Femoral_Cartilage_MRI

Python code for three-dimensional MRI characterization of femoral cartilage lesion geometry, whole-femur cartilage morphometry, and lesion-associated surface microenvironment.

This repository contains the analysis pipeline used for the study:

**"Three-Dimensional MRI Characterization of Femoral Cartilage Lesion Geometry and Surface Microenvironment"**

The pipeline performs quantitative MRI-based characterization of femoral cartilage using three-dimensional surface reconstruction, geometric analysis, and statistical modeling.

---

## Overview

This project investigates whether focal three-dimensional lesion geometry and surface characteristics provide additional information beyond conventional whole-cartilage morphometric measurements.

The analysis includes:

1. Whole-femur cartilage morphometry
2. Focal femoral cartilage lesion geometry characterization
3. Ordinal lesion-grade association analysis
4. Three-dimensional cartilage surface microenvironment analysis
5. Surface geometric anisotropy and orientation analysis
6. Sensitivity and exploratory structural analyses

The workflow was developed for quantitative analysis of knee MRI data from the Stanford Knee MRI with Multi-Task Evaluation (SKM-TEA) dataset.

---

## Data Availability

The original SKM-TEA imaging data are not included in this repository.

Access to the source dataset requires appropriate authorization according to the dataset provider's terms.

This repository contains only analysis scripts and configuration files.

**Do not upload participant-level MRI data or derived files containing protected information.**

---

## Repository Structure

The main analysis scripts are organized according to the analysis workflow:

```
01_data_source_audit_RAW.py
    Data source verification and quality assessment

02_build_master_cohort_RAW.py
    Construction of the master analysis cohort

03_whole_femur_morphometry_RAW.py
    Whole-femur cartilage surface reconstruction and morphometric analysis

04a_build_focal_bbox_geometry.py
04b_construct_focal_geometry.py
04c_geometry_redundancy_audit.py
04d_subject_independence_audit.py
    Three-dimensional focal lesion geometry processing and quality control

05a_univariable_ordinal_gee.py
05b_age_sex_adjusted_ordinal_gee.py
05c_annotation_quality_sensitivity.py
05d_cumulative_grade_threshold_sensitivity.py
    Ordinal lesion-grade association analyses using generalized estimating equations

06a_build_geometry_clusters.py
06b_subject_level_clustering_stability.py
06c_profile_k2_geometry_clusters.py
06d_cluster_grade_association.py
06e_plot_k2_geometry_subgroups.py
    Exploratory geometric clustering analyses

07a_rebuild_meniscal_compartments.py
07b_rebuild_femoral_compartments.py
07c_compartment_matched_unadjusted_association.py
07d_multiple_testing_correction.py
07e_age_sex_adjusted_compartment_logistic.py
07f_compartment_interaction_gee.py
07g_plot_compartment_meniscus_cartilage_FINAL.py
    Meniscus–cartilage compartment analyses

08a_build_integrated_lesion_master.py
08b_primary_multivariable_ordinal_gee.py
08c_whole_femur_morphometry_sensitivity.py
08d_meniscal_pathology_sensitivity.py
    Final integrated analyses and sensitivity models
```

---

## Environment Setup

The analysis was developed using Python.

Required Python packages should be installed according to the analysis environment used for the study.

Before running the scripts, configure the local data paths.

The repository uses environment variables to locate private data:

```bash
export SKMTEA_DATA_ROOT="/path/to/authorized/SKM-TEA/data"
export SKMTEA_ANALYSIS_DIR="/path/to/analysis/output"
```

The path configuration is handled through:

```
project_paths.py
```

No original imaging data are included.

---

## Analysis Workflow

The complete workflow consists of the following steps:

### 1. Data preparation

* Verify dataset structure
* Construct the master cohort
* Prepare cartilage lesion annotations

### 2. Whole-femur morphometry

Three-dimensional femoral cartilage surfaces are reconstructed from cartilage segmentations.

Computed measures include:

* cartilage thickness
* cartilage volume
* surface area
* thickness distribution metrics

### 3. Focal lesion geometry

Three-dimensional lesion descriptors are generated from lesion bounding-box annotations, including:

* bounding-box volume
* geometric anisotropy
* anatomical location
* spatial orientation features

Associations with ordinal cartilage lesion grade are evaluated using generalized estimating equations accounting for within-examination clustering.

### 4. Surface microenvironment analysis

Lesion-associated surface regions are characterized using:

* Gaussian curvature
* tangential thickness-gradient magnitude
* lesion core, perilesional, and remote cartilage regions

### 5. Exploratory geometric analyses

Additional analyses include:

* surface geometric anisotropy
* principal-axis orientation
* geometric subgroup clustering
* compartment-based structural analyses

---

## Reproducibility Notes

This repository provides the computational workflow used for the reported analyses.

Because the original MRI dataset is not distributed with the repository, complete reproduction requires authorized access to the SKM-TEA dataset.

The authors recommend reviewing:

```
PATH_SETUP.md
```

before running the analysis scripts.

---

## Citation

If you use this code, please cite the associated publication:

```
Kang CW, Wu LX, Gong JB, Dai SY, Yu TB.

Three-Dimensional MRI Characterization of Femoral Cartilage Lesion Geometry
and Surface Microenvironment.

[Journal information to be updated after publication]
```

---

## License

License information will be added after publication and according to the intended reuse policy.

---

## Contact

For questions regarding the analysis pipeline, please contact the corresponding authors of the associated study.
