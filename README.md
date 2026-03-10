# DSC180B Sepsis Prediction Model

This repository contains the code and data processing pipeline for an ICU early warning model developed for **UCSD DSC180B Data Science Capstone**.

The goal of the project is to predict whether sepsis onset will occur within the next **6 hours** using routinely collected ICU vital signs aggregated at the **ICU-hour level**.

---

## Repository Structure
dsc180B-sepsis-model
│
├── data_creation/ # scripts for extracting and preprocessing ICU data
├── model/ # model training and evaluation scripts
├── service/ # API service for deploying the trained model
├── README.md
└── .gitignore

---

# Folder Description

### data_creation

This folder contains scripts used to construct the modeling dataset from ICU data.

Main tasks include:

- extracting ICU stays
- aligning vital sign measurements to ICU-hour observations
- aggregating vitals into summary statistics
- generating the final modeling dataset

The output of this stage is the **processed feature table used for model training**.

---

### model

This folder contains the machine learning pipeline.

Main components include:

- loading the processed dataset
- preprocessing and feature cleaning
- model training
- model evaluation
- saving trained model artifacts

The primary model used in this project is a **Random Forest classifier**.

---

### service

This folder contains the **model deployment service**.

It loads the trained model and exposes an API endpoint that:

- accepts ICU patient feature inputs
- returns predicted sepsis risk probabilities

This allows the model to be integrated into a monitoring system.

---

# Project Overview

Early detection of sepsis is critical in intensive care because timely treatment can significantly improve patient outcomes.

This project builds a machine learning classifier that:

- predicts whether sepsis onset will occur within the next **1–4 hours**
- uses **hourly aggregated vital sign measurements**
- models the prediction task at the **ICU-hour level**
- prevents information leakage using **ICU stay-level train/test splits**

A **Random Forest classifier** is used because it performs well on clinical tabular data and can capture non-linear relationships between physiological signals.

---

# Data Sources

The project uses ICU data derived from clinical records.

Two main derived tables are used:

### Stay-Level Table

Contains:

- `subject_id`
- `hadm_id`
- `stay_id`
- sepsis labels
- onset timestamps

### Vitals Table

Contains hourly vital sign measurements indexed by:
(subject_id, hadm_id, stay_id, hour)


These tables are merged to construct the modeling dataset indexed at the **ICU-hour level**.

---

# Feature Engineering

Vital signs are aggregated within each hour using summary statistics:

- mean
- minimum
- maximum
- median
- standard deviation
- count


The final modeling dataset combines:

1. hourly ICU time grid
2. stay-level metadata
3. aggregated vital sign features

---

# Label Definition

At time **t₀**, the model predicts whether sepsis onset will occur within the next **6 hours**.

### Time Definition
t₀ = intime + hour


### Onset Timestamp Selection

Sepsis onset time is determined using:

- `sofa_time` (preferred)
- `suspected_infection_time` (fallback)

### Binary Label
y = 1 if onset_time ∈ (t₀, t₀ + 6 hours]
y = 0 otherwise


A positive label therefore indicates that sepsis onset occurs strictly after the current hour and within the next 4 hours.

### Leakage Prevention

To preserve temporal validity:

- all rows at or after onset are removed
- the model only uses information available prior to sepsis onset

---

# Model Training

The machine learning pipeline is:
MedianImputer and OneHotEcncoder → RandomForestClassifier


### Hyperparameters

- number of trees: `500`
- minimum leaf size: `5`
- class imbalance handling: `class_weight="balanced_subsample"`

The model is trained using only information available up to time **t₀**, ensuring a strictly forward-looking prediction setup.

---

# How to Reproduce the Pipeline

The pipeline can be rerun in three stages.

### Step 1 — Build the dataset

Run the scripts in `data_creation/` to generate the ICU-hour feature dataset.

### Step 2 — Train the model

Run the training pipeline in `model/` to fit the Random Forest classifier.

### Step 3 — Start the prediction service

Run the code in `service/` to launch the inference API.

---

# Reproducibility Notes

To reproduce the results:

1. obtain access to the ICU dataset used for this project
2. run the data creation pipeline
3. train the model
4. launch the prediction service

The repository is organized so each stage of the pipeline can be rerun independently.

---

# Authors

UC San Diego — DSC180B Capstone

- Samuel Mahjouri  
- Utkarsh Lohia  
- Juntong Ye  
- Kate Zhou
