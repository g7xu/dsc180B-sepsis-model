# DSC180B Sepsis Prediction Model

Predicts sepsis onset within the next 6 hours using hourly ICU vital signs. Built for the **UCSD DSC180B Data Science Capstone** using MIMIC-IV clinical data and a Random Forest classifier.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --data data/cohort_feature_matrix_labeled_outcome.csv
```

## Repository Structure

```
dsc180B-sepsis-model/
├── src/                            # Training pipeline (Python modules)
│   ├── train.py                    #   CLI entry point
│   ├── data.py                     #   Data loading & timestamp parsing
│   ├── labels.py                   #   Binary target construction
│   ├── features.py                 #   Feature matrix builder
│   ├── split.py                    #   Stay-level train/test split
│   ├── model.py                    #   sklearn pipeline (imputer + RF)
│   └── evaluate.py                 #   Metrics & threshold sweep
├── sql/                            # SQL materialized view definitions
├── notebooks/                      # Exploratory & alternative notebooks
├── data/                           # Local data files (gitignored)
└── requirements.txt
```

## Contributing

1. Fork the repo and create a feature branch
2. Make your changes
3. Open a pull request to `main`

### Authors

UC San Diego — DSC180B Capstone

- Samuel Mahjouri
- Utkarsh Lohia
- Juntong Ye
- Kate Zhou
