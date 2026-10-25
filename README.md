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
Requires Python 3.13.

**With conda:**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
conda create -n aerohealth python=3.13 -y
conda activate aerohealth
pip install -r requirements.txt
mkdir data\raw data\processed
```

**With venv (e.g. VS Code):**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
python -m venv venv
venv\Scripts\activate
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


## Results (FD001, single seed, preliminary)

| Model | Val RMSE | Test RMSE | Test NASA score | Train time (s) |
|---|---|---|---|---|
| Random Forest | 12.88 | 14.11 | 311.0 | 59 |
| XGBoost | 12.56 | 13.72 | 288.6 | 8 |
| LSTM | 11.92 | 15.35 | 423.4 | 224 |
| GRU | 11.59 | 14.88 | 365.7 | 255 |
| 1D-CNN | 11.98 | 14.56 | 344.9 | 51 |

Test RMSE and NASA score use the true (uncapped) RUL of the 100 test engines. Lower is better.
Test has only 100 points and one seed was used, so differences of about 1 RMSE are within noise. No ranking is claimed yet.

## Status
- [x] Environment setup
- [x] EDA and preprocessing (FD001)
- [x] Baselines: Random Forest, XGBoost
- [x] Deep models: LSTM, GRU, 1D-CNN
- [ ] Multiple seeds for the deep models
- [ ] SHAP explainability
- [ ] Streamlit dashboard
- [ ] FD002 to FD004

## Authors
- agupta30be24-create
- chintansood