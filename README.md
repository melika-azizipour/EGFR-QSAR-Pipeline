# EGFR QSAR Pipeline

Machine learning pipeline for predicting EGFR inhibitory activity (pIC50) using real ChEMBL data.

## Features

- Automatic ChEMBL data retrieval
- Morgan fingerprints
- Electron-conformational descriptors
- PCA dimensionality reduction
- Genetic Algorithm feature selection
- Ensemble learning (XGBoost, Random Forest, Gradient Boosting)

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
python chembl_ml.py
```

## Output

- Model performance metrics (R², MAE, RMSE)
- Prediction plots
- Saved model (.pkl)
