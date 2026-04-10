# DSC180B Sepsis Prediction Model

This repository contains the training pipeline for an ICU early warning model developed for **UCSD DSC180B Data Science Capstone**.

The goal of the project is to predict whether sepsis onset will occur within the next **6 hours** using routinely collected ICU vital signs aggregated at the **ICU-hour level**.

---

## Repository Structure

```
dsc180B-sepsis-model/
├── sql/                            # SQL materialized view definitions
│   ├── 01_mv_icd9_patients.sql
│   ├── 02_mv_first_icu_stay.sql
│   ├── 03_mv_icd9_icu_cohort_data.sql
│   ├── 04_mv_management_view_data.sql
│   ├── mv_measurements.sql
│   └── elixhauser_quan_from_cohort.sql
├── notebooks/                      # Pipeline notebooks (run in order)
│   ├── 01_create_dataset.ipynb
│   ├── 02_train_rf.ipynb
│   ├── 03_train_rnn.ipynb
│   └── 04_subgroup_discovery.R
├── data/                           # Local data files (gitignored)
├── requirements.txt
└── README.md
```

---

## Environment Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the subgroup discovery step (`notebooks/04_subgroup_discovery.R`), R must be installed separately with the `tidyverse` and `poLCA` packages:

```r
install.packages(c("tidyverse", "poLCA"))
```

---

## How to Reproduce the Pipeline

### Step 1 — Create materialized views

Run the SQL scripts in `sql/` against a MIMIC-IV PostgreSQL database in numeric order (`01_` through `04_`), then `mv_measurements.sql` for vital sign extraction, and `elixhauser_quan_from_cohort.sql` for comorbidity flags.

### Step 2 — Build the dataset

Run `notebooks/01_create_dataset.ipynb` to construct the ICU-hour feature dataset.

### Step 3 — Train the model

Run `notebooks/02_train_rf.ipynb` to fit the Random Forest classifier (primary model).

Alternatively, run `notebooks/03_train_rnn.ipynb` for the PyTorch RNN baseline.

### Step 4 (optional) — Subgroup discovery

Run `notebooks/04_subgroup_discovery.R` to perform Latent Class Analysis on comorbidity patterns.

---

## Project Overview

Early detection of sepsis is critical in intensive care because timely treatment can significantly improve patient outcomes.

This project builds a machine learning classifier that:

- predicts whether sepsis onset will occur within the next **1–4 hours**
- uses **hourly aggregated vital sign measurements**
- models the prediction task at the **ICU-hour level**
- prevents information leakage using **ICU stay-level train/test splits**

A **Random Forest classifier** is used because it performs well on clinical tabular data and can capture non-linear relationships between physiological signals.

---

## Data Sources

The project uses ICU data derived from clinical records (MIMIC-IV).

Two main derived tables are used:

### Stay-Level Table

Contains: `subject_id`, `hadm_id`, `stay_id`, sepsis labels, onset timestamps.

### Vitals Table

Contains hourly vital sign measurements indexed by `(subject_id, hadm_id, stay_id, hour)`.

These tables are merged to construct the modeling dataset indexed at the **ICU-hour level**.

---

## Feature Engineering

Vital signs are aggregated within each hour using summary statistics: mean, minimum, maximum, median, standard deviation, and count.

The final modeling dataset combines:

1. hourly ICU time grid
2. stay-level metadata
3. aggregated vital sign features

---

## Label Definition

At time **t₀**, the model predicts whether sepsis onset will occur within the next **6 hours**.

- **Time definition**: t₀ = intime + hour
- **Onset timestamp**: `sofa_time` (preferred) or `suspected_infection_time` (fallback)
- **Binary label**: y = 1 if onset_time ∈ (t₀, t₀ + 6 hours], y = 0 otherwise

### Leakage Prevention

To preserve temporal validity, all rows at or after onset are removed — the model only uses information available prior to sepsis onset.

---

## Model Training

Pipeline: MedianImputer and OneHotEncoder → RandomForestClassifier

### Hyperparameters

- number of trees: `500`
- minimum leaf size: `5`
- class imbalance handling: `class_weight="balanced_subsample"`

---

## Authors

UC San Diego — DSC180B Capstone

- Samuel Mahjouri
- Utkarsh Lohia
- Juntong Ye
- Kate Zhou
