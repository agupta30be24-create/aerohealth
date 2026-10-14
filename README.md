# AeroHealth

**Explainable Deep Learning for Aircraft Engine Failure Prediction and Remaining Useful Life (RUL) Estimation**

Predicts how many flight cycles a turbofan engine has left before failure from multivariate sensor time series, and explains each prediction with SHAP.

## Dataset
NASA C-MAPSS turbofan engine degradation simulation (FD001 to FD004).
Download: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data

The raw data is not stored in this repo. Place the `.txt` files in `data/raw/`.

## Pipeline
Sensor data, then preprocessing, then sliding windows, then model, then RUL, then failure risk, then SHAP explanation.

Models compared: Random Forest, XGBoost, LSTM, GRU, 1D-CNN.
Metrics: RMSE and the NASA asymmetric scoring function.

## Project structure
```
aerohealth/
├── data/
│   ├── raw/          # C-MAPSS files (not tracked)
│   └── processed/    # generated windows (not tracked)
├── notebooks/        # 01_eda, 02_baselines, 03_deep_models, 04_shap
├── src/              # reusable code
├── app/              # Streamlit dashboard
├── results/          # plots and metric tables
└── requirements.txt
```

## Setup
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
conda create -n aerohealth python=3.13 -y
conda activate aerohealth
pip install -r requirements.txt
mkdir data\raw data\processed
```
Copy the C-MAPSS files into `data/raw/`, then run `notebooks/01_eda.ipynb`. It creates `data/processed/fd001_windows.npz`.

## Preprocessing (FD001)
- Dropped 6 constant sensors (s1, s5, s10, s16, s18, s19), leaving 15 features
- RUL capped at 125
- Train/validation split by engine (80/20), never by row
- MinMax scaling fitted on training engines only
- Sliding windows of 30 cycles
- Test: one window per engine (its last 30 cycles)

## Status
- [x] Environment setup
- [x] EDA and preprocessing (FD001)
- [ ] Baselines: Random Forest, XGBoost
- [ ] Deep models: LSTM, GRU, 1D-CNN
- [ ] SHAP explainability
- [ ] Streamlit dashboard
- [ ] FD002 to FD004

## Authors
- agupta30be24-create