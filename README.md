# AeroHealth

**Explainable Deep Learning for Aircraft Engine Failure Prediction and Remaining Useful Life (RUL) Estimation**

Predicts how many flight cycles a turbofan engine has left before failure from multivariate sensor time series, converts that into a calibrated failure risk, and explains each prediction with SHAP.

## Dataset
NASA C-MAPSS turbofan engine degradation simulation. This project trains and evaluates on **FD001** (one flight condition, one fault mode). The dashboard and 3D viewer can also load FD002 to FD004 using the same preprocessing, but those datasets are not evaluated here.
Download: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data

The raw `.txt` files are included in `data/raw/`. If you download them yourself, place them there.
Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 (PDF in `data/raw/`). Sensor names used in the dashboard come from its Table 2.

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 / 100 |
| FD002 | 6 | HPC degradation | 260 / 259 |
| FD003 | 1 | HPC + fan degradation | 100 / 100 |
| FD004 | 6 | HPC + fan degradation | 249 / 248 |

## Pipeline
Sensor data → preprocessing → 30-cycle sliding windows → model → RUL → **failure risk** → **SHAP explanation** → dashboard.

Models compared: Random Forest, XGBoost, LSTM, GRU, 1D-CNN.
Metrics: RMSE and the NASA asymmetric scoring function (late predictions penalised more than early ones).

## Project structure
```
aerohealth/
├── data/
│   ├── raw/                  # C-MAPSS files
│   └── processed/            # generated windows (rebuild with src/aerohealth.py, see below)
├── notebooks/
│   ├── 01_eda.ipynb          # EDA + FD001 preprocessing -> fd001_windows.npz
│   ├── 02_baselines.ipynb    # Random Forest, XGBoost (5 seeds)
│   ├── 03_deep_models.ipynb  # LSTM, GRU, 1D-CNN (5 seeds)
│   ├── 04_shap.ipynb         # SHAP explanations for XGBoost (FD001)
│   └── 05_failure_risk.ipynb # RUL -> calibrated failure probability -> risk level (FD001)
├── src/
│   ├── aerohealth.py         # shared pipeline code: prepare(), features, XGBoost, calibration, risk levels
│   └── export_frontend.py    # exports results to web/data/*.json for the 3D frontend
├── app/dashboard.py          # Streamlit dashboard (benchmark fleet + upload your own data)
├── app/server.py             # 3D frontend server: serves web/ and analyses uploads (stdlib only)
├── examples/                 # example upload files (real NASA test engines)
├── web/                      # Three.js 3D frontend (index.html, main.js, data/*.json, vendor/three)
├── results/                  # metric tables (.csv) and plots (.png)
└── requirements.txt          # pinned versions
```

## Setup
Requires Python 3.13. Package versions are pinned in `requirements.txt` so every machine gets identical results (unpinned XGBoost versions gave different test RMSE: 13.72 vs 14.09).

**With conda (macOS / Linux):**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
conda create -n aerohealth python=3.13 -y
conda activate aerohealth
pip install -r requirements.txt
```

**With venv on Windows (e.g. VS Code):**
```
git clone https://github.com/agupta30be24-create/aerohealth.git
cd aerohealth
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(macOS/Linux venv: `source venv/bin/activate`.)

## How to reproduce
Run the notebooks in order from the `notebooks/` folder: 01 → 05. 01 builds `fd001_windows.npz`, which 02 to 05 load. The apps build the FD002-FD004 windows automatically the first time those datasets are opened (they are not tracked because `fd004_windows.npz` is about 111 MB, over GitHub's 100 MB file limit). To build them by hand:
```
cd src
python -c "import aerohealth as ah; [ah.prepare(fd) for fd in ah.DATASETS]"
```
Dashboard (from the repo root):
```
streamlit run app/dashboard.py
```

## Preprocessing
**FD001** (`01_eda`):
- Dropped 6 constant sensors (s1, s5, s10, s16, s18, s19), leaving 15 features
- RUL capped at 125
- Train/validation split by engine (80/20), never by row
- MinMax scaling fitted on training engines only
- Sliding windows of 30 cycles
- Test: one window per engine (its last 30 cycles)

**Shared code for the apps** (`src/aerohealth.py: prepare()`): the same rules, written once. For FD001 it reproduces `fd001_windows.npz` **exactly** (checked array by array). For the other datasets the apps can load, it also scales sensors within each flight condition and pads test engines shorter than 30 cycles.

## Models
- **Random Forest and XGBoost** use 90 summary features per window (last value, mean, std, min, max and slope of each of the 15 sensors), because trees cannot read sequences.
- **LSTM, GRU and 1D-CNN** read the raw 30 x 15 window. They use Adam (lr 1e-3), batch size 256, gradient clipping and early stopping on validation RMSE.
- **Deep models are trained on RUL / 125** and the output is multiplied back. Without this the LSTM got stuck predicting the average (validation RMSE about 41.7) and learned nothing. Do not remove the scaling.
- No hyperparameter tuning has been done.

## Results

### FD001: all models, 5 seeds (42, 1, 2, 3, 4), mean ± std
| Model | Val RMSE | Test RMSE | Test NASA score |
|---|---|---|---|
| **XGBoost** | 12.42 ± 0.06 | **13.89 ± 0.22** | **286 ± 10** |
| Random Forest | 12.82 ± 0.04 | 14.19 ± 0.07 | 325 ± 10 |
| GRU | 11.85 ± 0.26 | 14.81 ± 0.27 | 362 ± 37 |
| 1D-CNN | 11.93 ± 0.08 | 14.85 ± 0.27 | 362 ± 15 |
| LSTM | 12.01 ± 0.17 | 14.98 ± 0.27 | 381 ± 32 |

Per-seed results: `results/tree_runs.csv`, `results/deep_runs.csv`. Test RMSE and NASA score use the true (uncapped) RUL of the 100 test engines. Validation RMSE is on capped labels. Lower is better.

- On the FD001 test set the **tree models beat the deep models for every seed**. The gap (about 1 RMSE) is about 4x the seed-to-seed spread.
- The deep models have the best validation RMSE but the worst test RMSE. Their validation RMSE is the best epoch picked on the validation set, so it is optimistically biased (the tree models are evaluated once). Validation also covers every window of 20 engines, while test is one final window per engine. Compare models on test, not validation.
- Seeds measure training randomness only. With 100 test engines, a different test sample could change the gap, so this is a result **on this test set**, not a general claim that trees are better.

## Failure risk (`05_failure_risk`)
Predicted RUL is converted into **P(true RUL ≤ H)** with a logistic regression fitted on the **validation** engines (the test set is never used for fitting). This corrects the model's bias near failure: about two times in three it predicts **too late** there.

| Level | Rule |
|---|---|
| HIGH | P(fail within H) ≥ 0.5 |
| MEDIUM | P(fail within 2H) ≥ 0.5 |
| LOW | otherwise |

The only chosen numbers are **H = 30** and the cut-off **0.5**. The RUL thresholds are derived (FD001: HIGH if predicted RUL ≤ 31.6).
- **Why H = 30:** validation recall/precision stay about 0.9 for every H from 15 to 60 (`05`, Extra section), so 30 is feasible but the data does not single it out. It is chosen because it leaves 25 FD001 test engines to evaluate on (only 10 at H = 15), and because H is an operational parameter (warning time maintenance needs). It is exposed at the top of the notebook and in the dashboard.
- **Why 0.5:** "more likely than not". `05`, Extra section, shows the trade-off (for example, 0.3 raises recall from 0.93 to 0.96 but nearly doubles false alarms).

**FD001 test result:** 25 test engines truly fail within 30 cycles. 23 of them are rated HIGH and the other 2 MEDIUM, 23 of the 24 HIGH engines really fail within 30 cycles, and **no failing engine is rated LOW**.

## Explainability (`04_shap`)
SHAP TreeExplainer on XGBoost (exact for tree models, so base value + SHAP values = prediction for every engine).
- **FD001:** the top sensors are **s11 (Ps30)** and **s3 (T30)**, both measured at the **high-pressure compressor (HPC) outlet**, and FD001's only fault mode is HPC degradation. Trends over the window matter most (slope = 43% of importance, latest value only 1%).
- SHAP explains what the model uses, not physical causation. Correlated sensors can share importance.

## Dashboard (`app/dashboard.py`)
```
streamlit run app/dashboard.py        # then open http://localhost:8501
```
Two modes (sidebar → **Data → Source**):

**1. NASA test fleet (benchmark).** Dataset selector (FD001-FD004). Fleet overview (HIGH/MEDIUM/LOW counts and an engine table sorted by risk), engine detail (risk, probability, plain-English SHAP reasons, waterfall plot, sensor history), and a method/limitations tab. H and the probability cut-off can be changed live. True RUL is hidden unless "evaluation mode" is on, because it is unknown in real operation.

**2. Upload your own engine data.** Upload a sensor file and get the same predictions, risk levels and explanations for your engines, plus a CSV download.
- **Format:** one row per engine per cycle, 26 numeric columns `engine, cycle, op1, op2, op3, s1 … s21` (C-MAPSS format). NASA's space-separated `.txt` files work as-is. CSV with or without a header also works (`unit`/`time` are accepted for `engine`/`cycle`). Examples: `examples/sample_engines_1condition.csv` and `examples/sample_engines_6conditions.csv` (real NASA test engines), also selectable in the app.
- **Nothing is fitted on uploaded data.** It is transformed with the regime clustering, sensor list and scalers learned from the training engines (`src/aerohealth.py: Preprocessor`), then passed to the trained model. Tested: uploading NASA's `test_FD00x.txt` gives **exactly** the benchmark windows and predictions on all four datasets.
- **Model choice:** flight conditions are detected from the operating settings, so the app suggests a one- or six-condition model. The fault type cannot be detected, so it suggests the model trained on both fault types (FD003 or FD004). The user can override it.
- **Data checks per engine** (thresholds measured on the training data):

| Check | Rule | Why this threshold |
|---|---|---|
| Unknown flight condition | operating settings > 0.1 (common scale) from every condition seen in training | Training noise is ≤ 0.005 and the two closest real conditions are 0.475 apart, so 0.1 cannot be triggered by noise but catches any unseen condition. Six-condition data through a one-condition model: 85% of rows flagged. Correct-model data: 0% |
| Outside training range | scaled sensor value < −0.1 or > 1.1 | More than 10% of the training range beyond it. On NASA's own test data this flags only the known FD004 s6 quirk |
| Short history | fewer than 30 cycles | Window padded by repeating the first cycle, so it carries less information and the prediction is less reliable |

- Bad files (wrong column count, non-numeric values, duplicate engine/cycle pairs) are rejected with a message saying what is wrong.

## 3D frontend (`web/`, Three.js)
```
python app/server.py                   # then open http://localhost:8765  (--port to change)
```
A 3D view of the same results, styled like an engineering viewer (matte materials, soft shadows, no animation except a slowly turning fan):
- **Fleet chart:** each engine is placed on a polar chart where **distance from the centre = predicted remaining life**. Dashed rings mark the derived HIGH/MEDIUM cut-offs. Colour = risk level. Hover for details, click to select.
- **Engine model:** a turbofan (fan → LPC → HPC → combustor → HPT → LPT) where each part is tinted in proportion to the **sum of the SHAP values of the sensors located there** (red shortens predicted life, blue extends it, steel grey = little effect). "Inspect engine" moves the camera to it, "Exploded view" separates the parts. Click a part for its per-sensor breakdown.
- An on-screen **check** shows that model average + part contributions = the model's prediction (SHAP values add up exactly).
- **Upload engine data:** drop a sensor file anywhere on the page. Suggested model, data warnings and CSV download as in the dashboard.

`app/server.py` (Python standard library only, localhost only) serves the page **and analyses uploads**. Uploads go through the same `build_payload()` as the NASA fleets (and the same Preprocessor and trained model as the dashboard), so uploading NASA's `test_FD001.txt` returns exactly the benchmark results. The NASA fleets alone also work from any static server (`python -m http.server 8765 -d web`), but uploads need `app/server.py`. `python src/export_frontend.py` regenerates `web/data/FD00x.json`.

All numbers are computed in Python; the frontend only displays them. Sensor-to-part mapping follows the station names in Saxena et al. (2008), Table 2. It is approximate: core speed (Nc, NRc) and bleed enthalpy are drawn on the HPC, although the core shaft also drives the HPT. Part totals grow with the number of sensors assigned, so compare individual sensors before concluding that one part dominates.

Prepared views via URL: `?fd=FD004&eval=1&engine=10&comp=hpc`, `?engine=85&exploded=1`, `?example=sample_engines_6conditions.csv`. Three.js 0.187.0 (core, OrbitControls, CSS2DRenderer) is included in `web/vendor/` (MIT licence) so it works offline.

## Notes
- **macOS: do not train XGBoost and PyTorch in the same Python process.** Each bundles its own OpenMP runtime, and the combination segfaults. That is why the tree models (02) and the deep models (03) are in separate notebooks.
- FD004 sensor s6 reads slightly below the training range in 2 test engines (39 of 126,480 values). This is a single-sensor quirk, not a regime error.
- Deep models train on CPU (`03` uses CUDA if available, otherwise CPU).

## Status
- [x] Environment setup (pinned versions)
- [x] EDA and preprocessing (FD001)
- [x] Baselines: Random Forest, XGBoost
- [x] Deep models: LSTM, GRU, 1D-CNN
- [x] Multiple seeds (5 seeds, all models, FD001)
- [x] SHAP explainability
- [x] Failure risk (calibrated)
- [x] Streamlit dashboard
- [x] 3D frontend (Three.js)
- [x] Upload your own engine data (dashboard)

Possible extensions: evaluate on FD002 to FD004 (six flight conditions, fan faults), hyperparameter tuning, a bootstrap over test engines for model comparisons.

## Authors
- agupta30be24-create
- chintansood
