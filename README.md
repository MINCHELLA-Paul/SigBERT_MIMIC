# SigBERT_MIMIC

**SigBERT_MIMIC** is a research repository for survival analysis from longitudinal clinical narratives in MIMIC-III. It adapts the SigBERT workflow to sequential electronic health record data, with the objective of predicting time-to-event outcomes while preserving the temporal structure of each patient's clinical history.

The repository contains the code required to preprocess longitudinal notes, construct document-level representations, encode patient trajectories with path signatures, fit sparse Cox models, and evaluate survival predictions on patient-level test sets.

## Methodological overview

The main pipeline consists of the following stages:

1. **Longitudinal cohort construction**  
   Clinical notes are ordered chronologically for each patient. Survival duration, censoring status, and temporal variables are defined at the patient level.

2. **Clinical text representation**  
   Clinical narratives are transformed into dense vector representations using a domain-specific language model and document-level embedding methods.

3. **Dimensionality reduction**  
   High-dimensional text embeddings can be compressed using principal component analysis (PCA) before temporal encoding.

4. **Path-signature encoding**  
   Each patient's sequence of embeddings is interpreted as a temporal path. Its truncated signature provides a fixed-dimensional representation of the longitudinal trajectory and its higher-order temporal interactions.

5. **Sparse survival modelling**  
   Signature features are used as covariates in an L1-penalized Cox proportional hazards model. Sparsity limits the number of active coefficients and facilitates interpretation.

6. **Patient-level evaluation**  
   Performance is assessed on patients not used for training, using survival-analysis metrics such as the concordance index and, when applicable, time-dependent AUC and Brier score.

## Repository structure

```text
SigBERT_MIMIC/
├── data/             # Local MIMIC-III data and derived datasets (not tracked)
├── notebooks/        # Exploratory analyses and reproducible experiments
├── src/              # Preprocessing, modelling, and evaluation utilities
├── outputs/          # Local figures, tables, and experimental results
├── requirements.txt  # Python dependencies, when provided
└── README.md
```

The exact structure may evolve as the experimental pipeline is consolidated.

## Data availability and the `/data` directory

The contents of `/data` are intentionally **not included in this repository** and must not be committed to GitHub.

MIMIC-III is a credentialed-access clinical database distributed through PhysioNet. Access requires completion of the applicable training, approval of the PhysioNet credentialing procedure, and acceptance of the MIMIC-III Data Use Agreement. This repository does not grant access to MIMIC-III.

Raw MIMIC-III data and patient-level derived files—including filtered cohorts, longitudinal note tables, embeddings, survival labels, and Parquet or CSV exports—are not redistributed here. Authorized users must obtain the source data directly from PhysioNet and reconstruct the required derived datasets locally using the preprocessing code supplied in the repository.

The `.gitignore` should therefore contain at least:

```gitignore
/data/*
!/data/README.md
!/data/.gitkeep
```

This rule keeps the local data outside Git while allowing a short documentation file and an empty directory marker to be versioned. Large clinical files should not be added through Git LFS as a workaround: file size is not the only constraint, since the data-use agreement and patient-data governance requirements still apply.

A typical local directory may contain files such as:

```text
data/
├── raw/              # Credentialed source data; never committed
├── interim/          # Intermediate preprocessing outputs; never committed
└── processed/        # Analysis-ready cohorts and embeddings; never committed
```

## Expected patient-level information

Depending on the experiment, the processed longitudinal table may include:

| Variable | Description |
|---|---|
| `SUBJECT_ID` | Unique patient identifier |
| `note_time` | Clinical-note timestamp |
| `embeddings` | Vector representation of a clinical note |
| `delta_i` | Event indicator: 1 for an observed event, 0 for censoring |
| `survival_time` | Observed event or censoring duration |
| `time_since_first_note` | Relative time within the patient trajectory |

Column names and survival definitions should always be checked against the preprocessing configuration used for a given experiment.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Minchella-Paul/SigBERT_MIMIC.git
cd SigBERT_MIMIC
```

Creating a dedicated Python environment is recommended:

```bash
conda create -n sigbert-mimic python=3.10
conda activate sigbert-mimic
```

If a dependency file is provided, install it with:

```bash
pip install -r requirements.txt
```

Some experiments may additionally require survival-analysis, deep-learning, transformer, and path-signature libraries. The precise environment should be recorded alongside the corresponding experiment.

## Reproducibility and leakage prevention

Train-test separation must be performed at the **patient level**, never at the note level. All notes belonging to a given patient must remain in the same partition.

For landmark analyses, only information observed before the landmark time may be used to construct predictors. PCA transformations, standardization parameters, embedding compression, hyperparameter selection, and survival models must be fitted on the training data and then applied to the test data. These constraints are necessary to prevent patient overlap and temporal information leakage.

## Intended use

This codebase is intended for methodological research and reproducible experimentation in longitudinal survival analysis. It is not a clinical decision-support system and must not be used to make individual medical decisions without independent validation, appropriate governance, and prospective clinical assessment.

## Citation

If this repository contributes to scientific work, please cite the associated SigBERT or SigBERT_MIMIC publication when a definitive reference becomes available. The original MIMIC-III database and PhysioNet should also be cited according to their official citation instructions.

## Author

Paul Minchella  
Research in statistical learning, longitudinal clinical data, and survival analysis.
